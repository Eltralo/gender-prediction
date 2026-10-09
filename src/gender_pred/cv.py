"""Кросс-валидация, финальное обучение и оценка.

Схема: stratified 5-fold (× повторы) на размеченных клиентах (train + val). Для каждого компонента —
out-of-fold предсказания; признаки с метками и число итераций бустинга выбираются только по обучающей
части фолда. Бленд калибруется по OOF; финальные модели обучаются на всех размеченных клиентах.
Test в обучении и подборе не участвует.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import models, sequence
from .data import Vocab
from .features import LabelFeatures, model_matrix


def make_folds(ids: np.ndarray, y: pd.Series, cfg: dict, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    c = cfg["cv"]
    skf = StratifiedKFold(c["n_splits"], shuffle=True, random_state=c["fold_seed"] + repeat)
    return [(ids[a], ids[b]) for a, b in skf.split(ids, y.reindex(ids).to_numpy())]


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, threshold: float, n: int = 2000, level: float = 0.95, seed: int = 0
) -> dict:
    """Точечные AUC / accuracy и перцентильные bootstrap-интервалы."""
    rng = np.random.default_rng(seed)
    aucs, accs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
        accs.append(accuracy_score(y[i], p[i] >= threshold))
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    return {
        "auc": float(roc_auc_score(y, p)),
        "auc_ci": [float(v) for v in np.percentile(aucs, q)],
        "accuracy": float(accuracy_score(y, p >= threshold)),
        "accuracy_ci": [float(v) for v in np.percentile(accs, q)],
        "n": int(len(y)),
    }


@dataclass
class GenderModel:
    """Ансамбль LightGBM, дообученного энкодера CoLES и TabPFN с калиброванным блендом."""

    cfg: dict
    device: str = "cpu"
    log: bool = True
    oof: dict[str, pd.DataFrame] = field(default_factory=dict)
    fold_metrics: list[dict] = field(default_factory=list)

    @property
    def components(self) -> list[str]:
        return [c for c in self.cfg["blend"]["components"] if c != "tabpfn" or self.cfg["tabpfn"]["enabled"]]

    def _print(self, msg: str) -> None:
        if self.log:
            print(msg, flush=True)

    # ------------------------------------------------------------ один фолд
    def _fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tr: np.ndarray,
        va: np.ndarray,
        seqs: sequence.Sequences,
        encoder_state: dict | None,
        seed: int,
    ) -> dict:
        out: dict = {}
        lf = LabelFeatures(self.cfg["features"], seed=seed, n_inner=self.cfg["cv"]["inner_splits"])
        M_tr = model_matrix(X.loc[tr], lf.fit_transform(X.loc[tr], y))
        M_va = model_matrix(X.loc[va], lf.transform(X.loc[va]))
        if "gbm" in self.components:
            out["gbm"] = models.gbm_staged(M_tr, y[tr].to_numpy(), M_va, self.cfg, seed)
            if self.cfg["report"].get("contribution", True) and "lr_emb" in M_tr:
                # тот же бустинг без признака CoLES — для таблицы вклада компонентов
                out["gbm_agg"] = models.gbm_staged(
                    M_tr.drop(columns="lr_emb"), y[tr].to_numpy(), M_va.drop(columns="lr_emb"), self.cfg, seed
                )
        if "tabpfn" in self.components:
            tp = models.TabPFNModel(self.cfg, self.device, seed).fit(M_tr, y[tr].to_numpy())
            out["tabpfn"] = tp.predict_proba(M_va)
        if "seq" in self.components:
            f = self.cfg["sequence"]["finetune"]
            logits = []
            for s in f["seeds"]:
                clf = sequence.finetune(encoder_state, seqs, tr, y, self.cfg, self.device, seed=s)
                logits.append(sequence.predict_logits(clf, seqs, va, self.device, f["predict_max_len"]))
                del clf
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            out["seq"] = np.mean(logits, 0)
        return out

    # ------------------------------------------------------------ CV
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seqs: sequence.Sequences | None,
        encoder_state: dict | None,
        repeats: int | None = None,
    ) -> pd.DataFrame:
        """OOF-логиты компонентов (колонки ``<component>_r<k>``) и метрики бленда по фолдам (вложенно)."""
        ids = np.sort(y.index.to_numpy())
        repeats = repeats or self.cfg["cv"]["repeats"]
        cols: dict[str, pd.Series] = {}
        t0 = time.time()
        for r in range(repeats):
            folds = make_folds(ids, y, self.cfg, r)
            res = []
            for k, (tr, va) in enumerate(folds):
                res.append(self._fold(X, y, tr, va, seqs, encoder_state, seed=r * 10 + k))
                self._print(f"  repeat {r} fold {k} done [{time.time() - t0:.0f}s]")
            for c in ("gbm", "gbm_agg"):
                if c not in res[0]:
                    continue
                its = res[0][c][0]
                curves = [
                    np.array([roc_auc_score(y[va], p) for p in o[c][1]]) for o, (_, va) in zip(res, folds)
                ]
                rounds = models.nested_rounds(curves, its)
                if c == "gbm":
                    self.fold_rounds_ = getattr(self, "fold_rounds_", []) + rounds
                cols[f"{c}_r{r}"] = pd.concat(
                    [
                        pd.Series(models.to_logit(o[c][1][list(its).index(n)]), index=va)
                        for o, n, (_, va) in zip(res, rounds, folds)
                    ]
                )
            for c in ("tabpfn", "seq"):
                if c in self.components:
                    vals = [models.to_logit(o[c]) if c == "tabpfn" else o[c] for o in res]
                    cols[f"{c}_r{r}"] = pd.concat([pd.Series(v, index=va) for v, (_, va) in zip(vals, folds)])
        oof = pd.DataFrame(cols).reindex(ids)
        self.oof_ = oof
        self.fold_metrics = self._nested_blend_metrics(oof, y, repeats)
        return oof

    def _nested_blend_metrics(
        self, oof: pd.DataFrame, y: pd.Series, repeats: int, components: list[str] | None = None
    ) -> list[dict]:
        """Метрики бленда по фолдам: калибровка и веса для фолда k подбираются по остальным фолдам."""
        components = components or self.components
        ids, yv = oof.index.to_numpy(), y.reindex(oof.index).to_numpy()
        thr = self.cfg["blend"]["threshold"]
        rows = []
        for r in range(repeats):
            Z = oof[[f"{c}_r{r}" for c in components]].to_numpy()
            folds = make_folds(ids, y, self.cfg, r)
            pos = {c: i for i, c in enumerate(ids)}
            for k, (_, va) in enumerate(folds):
                vi = np.array([pos[c] for c in va])
                rest = np.setdiff1d(np.arange(len(ids)), vi)
                bl = models.LogitBlend(self.cfg["blend"]["weight_step"]).fit(Z[rest], yv[rest])
                p = bl.predict_proba(Z[vi])
                row = {
                    "repeat": r,
                    "fold": k,
                    "auc": roc_auc_score(yv[vi], p),
                    "accuracy": accuracy_score(yv[vi], p >= thr),
                }
                for i, c in enumerate(components):
                    row[f"auc_{c}"] = roc_auc_score(yv[vi], Z[vi, i])
                rows.append(row)
        return rows

    def contribution(self, y: pd.Series) -> pd.DataFrame:
        """Накопительный вклад: агрегаты -> + признак CoLES -> + дообученный CoLES -> + TabPFN (итог)."""
        repeats = sum(c.startswith("gbm_r") for c in self.oof_.columns)
        names = {
            "gbm_agg": "Агрегаты (LightGBM)",
            "gbm": "+ признак CoLES",
            "seq": "+ дообученный CoLES",
            "tabpfn": "+ TabPFN",
        }
        steps = [["gbm_agg"]] if any(c.startswith("gbm_agg_r") for c in self.oof_.columns) else []
        comps = []
        for c in self.components:
            comps.append(c)
            steps.append(list(comps))
        rows, prev = [], None
        for st in steps:
            m = pd.DataFrame(self._nested_blend_metrics(self.oof_, y, repeats, st))
            auc, acc = m["auc"].mean(), m["accuracy"].mean()
            rows.append(
                {
                    "step": names[st[-1]],
                    "components": "+".join(st),
                    "auc": auc,
                    "auc_std": m["auc"].std(ddof=1),
                    "accuracy": acc,
                    "accuracy_std": m["accuracy"].std(ddof=1),
                    "delta_auc": None if prev is None else auc - prev,
                }
            )
            prev = auc
        df = pd.DataFrame(rows)
        df.loc[df.index[-1], "step"] = df["step"].iloc[-1] + " = итог"
        return df

    # ------------------------------------------------------------ финальное обучение
    def fit_final(
        self, X: pd.DataFrame, y: pd.Series, seqs: sequence.Sequences | None, encoder_state: dict | None
    ) -> GenderModel:
        """Бленд по OOF (повтор 0) и финальные модели на всех размеченных клиентах."""
        ids = np.sort(y.index.to_numpy())
        Z = self.oof_[[f"{c}_r0" for c in self.components]].to_numpy()
        self.blend_ = models.LogitBlend(self.cfg["blend"]["weight_step"]).fit(Z, y.reindex(ids).to_numpy())
        self.label_features_ = LabelFeatures(
            self.cfg["features"], seed=self.cfg["seed"], n_inner=self.cfg["cv"]["inner_splits"]
        )
        M = model_matrix(X.loc[ids], self.label_features_.fit_transform(X.loc[ids], y))
        self.columns_ = list(M.columns)
        yv = y.reindex(ids).to_numpy()
        if "gbm" in self.components:
            self.rounds_ = int(np.mean(self.fold_rounds_) * self.cfg["gbm"]["final_rounds_scale"])
            self.gbm_ = [models.gbm_fit(M, yv, self.rounds_, self.cfg, s) for s in self.cfg["gbm"]["seeds"]]
        if "tabpfn" in self.components:
            self.tabpfn_ = models.TabPFNModel(self.cfg, self.device, self.cfg["seed"]).fit(M, yv)
        if "seq" in self.components:
            self.seq_ = [
                sequence.finetune(encoder_state, seqs, ids, y, self.cfg, self.device, seed=s).cpu()
                for s in self.cfg["sequence"]["finetune"]["seeds"]
            ]
        return self

    def component_logits(self, X: pd.DataFrame, seqs: sequence.Sequences | None) -> pd.DataFrame:
        M = model_matrix(X, self.label_features_.transform(X))[self.columns_]
        out = {}
        if "gbm" in self.components:
            out["gbm"] = models.to_logit(np.mean([m.predict(M) for m in self.gbm_], 0))
        if "tabpfn" in self.components:
            out["tabpfn"] = models.to_logit(self.tabpfn_.predict_proba(M))
        if "seq" in self.components:
            f = self.cfg["sequence"]["finetune"]
            out["seq"] = np.mean(
                [
                    sequence.predict_logits(
                        m.to(self.device), seqs, X.index, self.device, f["predict_max_len"]
                    )
                    for m in self.seq_
                ],
                0,
            )
        return pd.DataFrame(out, index=X.index)[self.components]

    def predict_proba(self, X: pd.DataFrame, seqs: sequence.Sequences | None) -> np.ndarray:
        return self.blend_.predict_proba(self.component_logits(X, seqs).to_numpy())

    # ------------------------------------------------------------ сохранение
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "components": self.components,
            "columns": self.columns_,
            "blend": self.blend_.to_dict(),
            "rounds": getattr(self, "rounds_", None),
        }
        (path / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        with open(path / "label_features.pkl", "wb") as fh:
            pickle.dump(self.label_features_, fh)
        for i, m in enumerate(getattr(self, "gbm_", [])):
            m.save_model(str(path / f"gbm_{i}.txt"))
        if hasattr(self, "tabpfn_"):
            with open(path / "tabpfn.pkl", "wb") as fh:
                pickle.dump(self.tabpfn_, fh)
        for i, m in enumerate(getattr(self, "seq_", [])):
            torch.save(
                {k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()},
                path / f"seq_{i}.pt",
            )
        self.oof_.to_parquet(path / "oof.parquet")

    @classmethod
    def load(cls, path: str | Path, cfg: dict, device: str, seqs: sequence.Sequences | None) -> GenderModel:
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads((path / "model.json").read_text())
        m = cls(cfg=cfg, device=device)
        m.cfg = dict(cfg, blend=dict(cfg["blend"], components=meta["components"]))
        m.columns_, m.blend_ = meta["columns"], models.LogitBlend.from_dict(meta["blend"])
        with open(path / "label_features.pkl", "rb") as fh:
            m.label_features_ = pickle.load(fh)
        m.gbm_ = [lgb.Booster(model_file=str(p)) for p in sorted(path.glob("gbm_*.txt"))]
        if (path / "tabpfn.pkl").exists():
            with open(path / "tabpfn.pkl", "rb") as fh:
                m.tabpfn_ = pickle.load(fh)
            m.tabpfn_.device = device
        m.seq_ = []
        for p in sorted(path.glob("seq_*.pt")):
            clf = sequence.SeqClassifier(sequence.build_encoder(seqs, cfg))
            clf.load_state_dict({k: v.float() for k, v in torch.load(p, map_location="cpu").items()})
            m.seq_.append(clf.eval())
        m.oof_ = pd.read_parquet(path / "oof.parquet")
        return m


def save_vocab(vocab: Vocab, path: str | Path) -> None:
    Path(path).write_text(json.dumps(vocab.to_dict()))


def load_vocab(path: str | Path) -> Vocab:
    return Vocab(**json.loads(Path(path).read_text()))
