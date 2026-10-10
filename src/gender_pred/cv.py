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


def _summary(name: str, rows: list[dict]) -> dict:
    m = pd.DataFrame(rows)
    return {
        "model": name,
        "auc": m["auc"].mean(),
        "auc_std": m["auc"].std(ddof=1),
        "accuracy": m["accuracy"].mean(),
        "accuracy_std": m["accuracy"].std(ddof=1),
    }


def embedding_cv(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    """Эмбеддинг CoLES сам по себе: логрег (как признак lr_emb) на тех же фолдах CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids = np.sort(y.index.to_numpy())
    E = X.loc[ids, X.columns.str.startswith("emb_")]
    rows = []
    for r in range(cfg["cv"]["repeats"]):
        for tr, va in make_folds(ids, y, cfg, r):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=cfg["features"]["embedding_logreg_C"], max_iter=3000),
            ).fit(E.loc[tr], y[tr])
            p = clf.predict_proba(E.loc[va])[:, 1]
            rows.append(
                {
                    "auc": roc_auc_score(y[va], p),
                    "accuracy": accuracy_score(y[va], p >= cfg["blend"]["threshold"]),
                }
            )
    return _summary(f"Эмбеддинг CoLES ({E.shape[1]}), логрег", rows)


UNLABELED_GROUPS = ("act", "amt", "mcc", "tt", "time", "term", "dyn")


def group_signal(X: pd.DataFrame, y: pd.Series, cfg: dict, seed: int = 0) -> pd.DataFrame:
    """AUC LightGBM на каждой группе признаков без меток отдельно и на всех вместе (5-fold, повтор 0)."""
    ids = np.sort(y.index.to_numpy())
    prefix = X.columns.str.split("_").str[0]
    groups = {g: X.columns[prefix == g] for g in UNLABELED_GROUPS}
    groups["all"] = X.columns[np.isin(prefix, UNLABELED_GROUPS)]
    folds = make_folds(ids, y, cfg, 0)
    rows = []
    for g, cols in groups.items():
        res = [
            models.gbm_staged(X.loc[tr, cols], y[tr].to_numpy(), X.loc[va, cols], cfg, seed)
            for tr, va in folds
        ]
        its = res[0][0]
        curves = [np.array([roc_auc_score(y[va], p) for p in o[1]]) for o, (_, va) in zip(res, folds)]
        rounds = models.nested_rounds(curves, its)
        aucs = [c[list(its).index(n)] for c, n in zip(curves, rounds)]
        rows.append(
            {"group": g, "n_features": len(cols), "auc": np.mean(aucs), "auc_std": np.std(aucs, ddof=1)}
        )
    return pd.DataFrame(rows)


def adversarial_auc(X_a: pd.DataFrame, X_b: pd.DataFrame, cfg: dict, seed: int = 0) -> dict:
    """Adversarial validation: отличает ли LightGBM клиентов двух сплитов по признакам без меток."""
    import lightgbm as lgb

    cols = X_a.columns[np.isin(X_a.columns.str.split("_").str[0], UNLABELED_GROUPS)]
    X = pd.concat([X_a[cols], X_b[cols]])
    s = np.r_[np.zeros(len(X_a)), np.ones(len(X_b))]
    params = dict(cfg["gbm"]["params"], seed=seed, num_threads=cfg["gbm"]["num_threads"])
    aucs, imp = [], np.zeros(len(cols))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, s):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], s[tr]), 500)
        aucs.append(roc_auc_score(s[va], m.predict(X.iloc[va])))
        imp += m.feature_importance("gain")
    top = pd.Series(imp, index=cols).sort_values(ascending=False)
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "n_a": len(X_a),
        "n_b": len(X_b),
        "top_features": list(top.index[:5]),
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

    def model_table(self, y: pd.Series) -> pd.DataFrame:
        """Каждая модель отдельно (калибровка по остальным фолдам) и итоговый бленд: метрики по фолдам CV."""
        repeats = sum(c.startswith("gbm_r") for c in self.oof_.columns)
        names = {
            "gbm_agg": "LightGBM, агрегаты",
            "gbm": "LightGBM, агрегаты + признак CoLES",
            "seq": "CoLES, дообученный на пол",
            "tabpfn": "TabPFN v2, 300 признаков",
        }
        comps = [c for c in names if any(k.startswith(f"{c}_r") for k in self.oof_.columns)]
        steps = [(names[c], [c]) for c in comps] + [("Бленд = итог", self.components)]
        return pd.DataFrame(
            [_summary(name, self._nested_blend_metrics(self.oof_, y, repeats, st)) for name, st in steps]
        )

    def compare_boosters(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """XGBoost и CatBoost на той же матрице (агрегаты + признаки с метками), тех же фолдах и с тем же
        вложенным выбором итераций, что LightGBM; метрики, парное сравнение с LightGBM и бленд с ними."""
        ids = np.sort(y.index.to_numpy())
        repeats = sum(c.startswith("gbm_r") for c in self.oof_.columns)
        staged = {"xgb": models.xgb_staged, "cat": models.cat_staged}
        cols: dict[str, pd.Series] = {}
        for r in range(repeats):
            folds = make_folds(ids, y, self.cfg, r)
            res = {name: [] for name in staged}
            for k, (tr, va) in enumerate(folds):
                lf = LabelFeatures(
                    self.cfg["features"], seed=r * 10 + k, n_inner=self.cfg["cv"]["inner_splits"]
                )
                M_tr = model_matrix(X.loc[tr], lf.fit_transform(X.loc[tr], y))
                M_va = model_matrix(X.loc[va], lf.transform(X.loc[va]))
                for name, fn in staged.items():
                    res[name].append(fn(M_tr, y[tr].to_numpy(), M_va, self.cfg, r * 10 + k))
                self._print(f"  comparison repeat {r} fold {k} done")
            for name, out in res.items():
                its = out[0][0]
                curves = [np.array([roc_auc_score(y[va], p) for p in o[1]]) for o, (_, va) in zip(out, folds)]
                rounds = models.nested_rounds(curves, its)
                cols[f"{name}_r{r}"] = pd.concat(
                    [
                        pd.Series(models.to_logit(o[1][list(its).index(n)]), index=va)
                        for o, n, (_, va) in zip(out, rounds, folds)
                    ]
                )
        oof = pd.concat([self.oof_, pd.DataFrame(cols).reindex(self.oof_.index)], axis=1)
        base = pd.DataFrame(self._nested_blend_metrics(oof, y, repeats, ["gbm"]))
        final = pd.DataFrame(self._nested_blend_metrics(oof, y, repeats, self.components))
        rows = []
        for name in staged:
            m = pd.DataFrame(self._nested_blend_metrics(oof, y, repeats, [name]))
            b = pd.DataFrame(self._nested_blend_metrics(oof, y, repeats, [*self.components, name]))
            d, db = m["auc"] - base["auc"], b["auc"] - final["auc"]
            rows.append(
                {
                    **_summary(name, m.to_dict("records")),
                    "delta_vs_lgb": d.mean(),
                    "delta_vs_lgb_std": d.std(ddof=1),
                    "folds_better_than_lgb": int((d > 0).sum()),
                    "blend_with_auc": b["auc"].mean(),
                    "blend_with_delta": db.mean(),
                    "blend_with_delta_std": db.std(ddof=1),
                    "blend_with_folds_better": int((db > 0).sum()),
                    "n_folds": len(d),
                }
            )
        self.comparison_oof_ = oof[list(cols)]
        return pd.DataFrame(rows)

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


def model_summary(results: str | Path) -> pd.DataFrame:
    """Сводная таблица всех моделей из results/: описание, данные, метрики CV, решение о финале."""
    res = Path(results)
    meta = json.loads((res / "cv_metrics.json").read_text())
    weights = dict(zip(meta["components"], meta["blend_weights"]))
    m = pd.read_csv(res / "cv_models.csv").set_index("model")
    cmp = pd.read_csv(res / "model_comparison.csv").set_index("model")

    def fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    def rejected(name: str) -> str:
        r = cmp.loc[name]
        n, better = int(r["n_folds"]), int(r["folds_better_than_lgb"])
        vs = (
            f"AUC {r['delta_vs_lgb']:+.4f} ± {r['delta_vs_lgb_std']:.4f} к LightGBM, "
            f"лучше в {better}/{n} фолдов"
        )
        if better == 0:
            return f"нет: хуже LightGBM ({vs}); в бленде вес 0"
        return (
            f"нет: вровень с LightGBM ({vs}); прирост в бленде {r['blend_with_delta']:+.5f} ± "
            f"{r['blend_with_delta_std']:.4f} AUC — в пределах шума, "
            "оставили одну модель бустинга ради простоты"
        )

    rows = [
        (
            "LightGBM",
            "градиентный бустинг, основная модель",
            "агрегаты клиента + признаки с метками (OOF), в т. ч. логрег по эмбеддингу CoLES",
            m.loc["LightGBM, агрегаты + признак CoLES"],
            f"да, вес {weights['gbm']:.2f}",
        ),
        (
            "TabPFN v2",
            "предобученный трансформер для таблиц, предсказывает без обучения деревьев",
            "300 признаков, отобранных по gain LightGBM внутри обучения",
            m.loc["TabPFN v2, 300 признаков"],
            f"да, вес {weights['tabpfn']:.2f}",
        ),
        (
            "CoLES, дообученный",
            "GRU-энкодер, предобученный контрастно без меток, + голова на пол",
            "последовательность последних 1 200 транзакций клиента",
            m.loc["CoLES, дообученный на пол"],
            f"да, вес {weights['seq']:.2f}",
        ),
        (
            "XGBoost",
            "градиентный бустинг",
            "та же матрица и те же фолды, что у LightGBM",
            cmp.loc["xgb"],
            rejected("xgb"),
        ),
        (
            "CatBoost",
            "градиентный бустинг (симметричные деревья)",
            "та же матрица и те же фолды, что у LightGBM",
            cmp.loc["cat"],
            rejected("cat"),
        ),
        (
            "Бленд = итог",
            "калиброванная взвешенная сумма логитов трёх моделей",
            "out-of-fold предсказания LightGBM, TabPFN и CoLES",
            m.loc["Бленд = итог"],
            "итог, порог 0.5",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "модель": name,
                "что это": what,
                "на чём учится": on,
                "CV ROC AUC": fmt(r["auc"], r["auc_std"]),
                "CV accuracy": fmt(r["accuracy"], r["accuracy_std"]),
                "в финале": verdict,
            }
            for name, what, on, r, verdict in rows
        ]
    ).set_index("модель")
