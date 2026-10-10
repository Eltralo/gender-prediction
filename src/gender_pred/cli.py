"""Точка входа: ``gender-pred train | evaluate | figures``.

train    — признаки, предобучение CoLES, кросс-валидация, финальные модели -> artifacts/, results/
evaluate — предсказание test одним проходом финальной модели, метрики с 95% bootstrap-CI -> results/
figures  — графики по сохранённым results/ -> reports/figures/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cv, features, sequence
from .data import (
    SEQ_STORE,
    SPLITS,
    Vocab,
    download_data,
    get_device,
    labels,
    load_config,
    load_split,
    prepare_events,
    set_seed,
)


@dataclass
class Context:
    """Всё, что нужно для признаков и моделей: конфиг, словарь, последовательности, энкодер."""

    cfg: dict
    device: str
    vocab: Vocab
    seqs: sequence.Sequences
    encoder: sequence.Encoder

    @property
    def artifacts(self) -> Path:
        return Path(self.cfg["paths"]["artifacts"])

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки одного сплита: агрегаты без меток + эмбеддинги CoLES; события — в SEQ_STORE."""
        ev = prepare_events(df)
        SEQ_STORE.add(ev)
        X = features.client_features(ev, self.vocab, self.cfg["features"])
        E = sequence.embed(
            self.encoder, self.seqs, X.index, self.device, self.cfg["sequence"]["pretrain"]["embed_max_len"]
        )
        return pd.concat([X, E.astype(np.float32)], axis=1)


class ArtifactsUnavailableError(RuntimeError):
    """Артефакты не найдены локально и не могут быть скачаны."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        return None


def _download(url: str, target: Path, headers: dict | None = None) -> None:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as fh:
        shutil.copyfileobj(r, fh)


def _from_url(rel: dict, target: Path) -> str:
    if not rel.get("url"):
        raise LookupError("публичная ссылка не задана (artifacts_release.url)")
    _download(rel["url"], target)
    return "публичная ссылка"


def _from_gh(rel: dict, target: Path) -> str:
    if shutil.which("gh") is None:
        raise LookupError("gh не установлен")
    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        raise LookupError("gh не авторизован")
    cmd = [
        "gh",
        "release",
        "download",
        rel["tag"],
        "--repo",
        rel["repo"],
        "--pattern",
        rel["asset"],
        "--dir",
        str(target.parent),
        "--clobber",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise LookupError(f"gh release download: {res.stderr.strip()}")
    return "gh"


def _from_token(rel: dict, target: Path) -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise LookupError("нет GH_TOKEN / GITHUB_TOKEN")
    api = f"https://api.github.com/repos/{rel['repo']}/releases"
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{api}/tags/{rel['tag']}", headers=auth), timeout=60
        ) as r:
            assets = json.load(r)["assets"]
    except urllib.error.HTTPError as e:
        raise LookupError(f"GitHub API: HTTP {e.code}") from e
    asset = next((a for a in assets if a["name"] == rel["asset"]), None)
    if asset is None:
        raise LookupError(f"в релизе {rel['tag']} нет {rel['asset']}")
    # ассет отдаётся редиректом на хранилище; токен туда не передаётся
    req = urllib.request.Request(
        f"{api}/assets/{asset['id']}", headers={**auth, "Accept": "application/octet-stream"}
    )
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
        raise LookupError("GitHub API не вернул ссылку на файл")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise LookupError(f"GitHub API: HTTP {e.code}") from e
        _download(e.headers["Location"], target)
    return "GitHub API (токен)"


def fetch_artifacts(cfg: dict) -> None:
    """Готовит artifacts/: уже распакованные файлы, ``artifacts/artifacts.zip``, положенный вручную,
    либо скачивание из GitHub Release — по публичной ссылке, через ``gh`` или по токену ``GH_TOKEN``."""
    path = Path(cfg["paths"]["artifacts"])
    if (path / "model" / "model.json").exists():
        return
    rel = cfg["artifacts_release"]
    path.mkdir(parents=True, exist_ok=True)
    archive = path / rel["asset"]
    if not archive.exists():
        errors = []
        for source in (_from_url, _from_gh, _from_token):
            try:
                print(f"artifacts: скачано ({source(rel, archive)})")
                break
            except (LookupError, OSError) as e:
                archive.unlink(missing_ok=True)
                errors.append(f"  - {e}")
        else:
            raise ArtifactsUnavailableError(
                f"Не удалось получить {rel['asset']} из релиза {rel['tag']} репозитория {rel['repo']}:\n"
                + "\n".join(errors)
                + "\n\nВарианты:\n"
                "  1) gh auth login  (аккаунт с доступом к репозиторию) и повторить запуск;\n"
                "  2) export GH_TOKEN=<токен с правом чтения репозитория> и повторить запуск;\n"
                f"  3) скачать {rel['asset']} со страницы https://github.com/{rel['repo']}/releases/tag/{rel['tag']}"
                f" и положить в {archive} — распакуется автоматически;\n"
                "  4) обучить всё заново: retrain: true в configs/default.yaml / RETRAIN = True"
                " в ноутбуке / make train (GPU, ~25 мин)."
            )
    with zipfile.ZipFile(archive) as z:
        z.extractall(path)
    if not (path / "model" / "model.json").exists():
        raise ArtifactsUnavailableError(f"{archive} не содержит model/model.json")


def build_context(cfg: dict, splits: dict[str, pd.DataFrame], log: bool = True) -> Context:
    """Словарь по train, последовательности всех клиентов, энкодер CoLES (обучение или загрузка)."""
    set_seed(cfg["seed"])
    device = get_device(cfg)
    art = Path(cfg["paths"]["artifacts"])
    if not cfg["retrain_seq"] or not cfg["retrain_tabular"]:
        fetch_artifacts(cfg)
    vocab = (
        cv.load_vocab(art / "vocab.json")
        if (art / "vocab.json").exists() and not cfg["retrain_tabular"]
        else Vocab.from_train(splits["train"], cfg["features"]["top_mcc_amount"])
    )
    ev = prepare_events(pd.concat(splits.values(), ignore_index=True))
    seqs = sequence.Sequences(ev, vocab)
    enc_path = art / "encoder.pt"
    if cfg["retrain_seq"]:
        p = cfg["sequence"]["pretrain"]
        # без меток: all — все клиенты (включая test), labeled — только train + val
        clients = (
            seqs.ids
            if p["clients"] == "all"
            else np.concatenate([splits[s]["customer_id"].unique() for s in ("train", "val")])
        )
        encoder = sequence.pretrain(seqs, clients, cfg, device, cfg["seed"], verbose=log)
        art.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), enc_path)
        cv.save_vocab(vocab, art / "vocab.json")
    else:
        encoder = sequence.build_encoder(seqs, cfg)
        encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
    return Context(cfg, device, vocab, seqs, encoder.to(device).eval())


def train(cfg: dict, ctx: Context | None = None, splits: dict | None = None) -> cv.GenderModel:
    """Кросс-валидация на train+val, финальные модели на всех размеченных клиентах, сохранение."""
    splits = splits or {s: load_split(s, cfg["paths"]["data"]) for s in SPLITS}
    ctx = ctx or build_context(cfg, splits)
    lab = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    X, y = ctx.features(lab), labels(lab)
    model = cv.GenderModel(cfg, ctx.device)
    model.cross_validate(X, y, ctx.seqs, ctx.encoder.state_dict())
    model.fit_final(X, y, ctx.seqs, ctx.encoder.state_dict())
    model.save(ctx.artifacts / "model")
    write_cv_results(cfg, model, y)
    write_model_table(cfg, model, X, y)
    write_descriptive_results(cfg, model, splits["train"])
    write_data_results(cfg, X, y, ctx.features(splits["train"]), ctx.features(splits["test"]))
    return model


def _mcc_desc(train: pd.DataFrame) -> dict[int, str]:
    if "mcc_code_desc" not in train:
        return {}
    return train.drop_duplicates("mcc_code").set_index("mcc_code")["mcc_code_desc"].astype(str).to_dict()


TE_STATS = {
    "wmean": "среднее по операциям",
    "mean": "среднее по категориям",
    "max": "максимум",
    "min": "минимум",
    "std": "разброс",
    "lo_sum": "сумма лог-шансов",
    "lo_mean": "средний лог-шанс",
    "wlo_sum": "сумма лог-шансов, взвеш.",
    "conf_lo_sum": "сумма лог-шансов с доверием",
}
NAMED = {
    "lr_tfidf": "Логрег по профилю MCC и типов (OOF)",
    "lr_emb": "Логрег по эмбеддингу CoLES (OOF)",
    "amt_out_logsum": "Сумма расходов (log)",
    "amt_in_logsum": "Сумма поступлений (log)",
    "amt_out_la_mean": "Средний расход (log)",
    "amt_in_cnt_share": "Доля поступлений",
    "time_night_share": "Доля ночных операций",
    "time_night_out_la": "Средний ночной расход (log)",
    "time_midnight_share": "Доля операций без времени",
    "act_n_tr": "Число операций",
}


def feature_label(name: str, mcc_desc: dict[int, str]) -> str:
    """Короткая русская подпись признака для графиков."""
    parts = name.split("_")
    if name in NAMED:
        return NAMED[name]
    if parts[0] == "mcc" and parts[1] in ("cnt", "sum", "la") and parts[-1].isdigit():
        what = {"cnt": "доля операций", "sum": "доля трат", "la": "средний чек"}[parts[1]]
        return f"MCC «{mcc_desc.get(int(parts[-1]), parts[-1])[:32]}»: {what}"
    if parts[0] == "te":
        what = "MCC" if parts[1] == "mcc" else "типа"
        return f"Target encoding {what}: {TE_STATS.get('_'.join(parts[2:]), '')}"
    if name.startswith("time_dow_"):
        return f"День недели {parts[-1]}: доля {'операций' if parts[2] == 'cnt' else 'трат'}"
    if name.startswith("time_hour_") and parts[-1].isdigit():
        return f"Доля операций в {parts[-1]} ч"
    return name


def write_descriptive_results(cfg: dict, model: cv.GenderModel, train: pd.DataFrame) -> None:
    """Таблицы для графиков: перекос MCC по полу (только train) и важности финального LightGBM."""
    res = Path(cfg["paths"]["results"])
    desc = _mcc_desc(train)
    pres = train.groupby(["customer_id", "mcc_code"]).size().unstack(fill_value=0) > 0
    g = labels(train).reindex(pres.index)
    n = pres.sum()
    skew = pd.DataFrame(
        {"mcc": n.index, "n_clients": n.to_numpy(), "male_share": (pres.T @ g / n).to_numpy()}
    )
    skew = skew[skew["n_clients"] >= 200].assign(base=g.mean(), desc=lambda d: d["mcc"].map(desc).fillna(""))
    skew.to_csv(res / "mcc_gender_skew.csv", index=False)
    if getattr(model, "gbm_", None):
        gain = np.mean([m.feature_importance("gain") for m in model.gbm_], 0)
        imp = pd.DataFrame({"feature": model.gbm_[0].feature_name(), "gain": gain})
        imp = imp.assign(gain_share=imp["gain"] / imp["gain"].sum()).sort_values("gain", ascending=False)
        imp["label"] = [feature_label(f, desc) for f in imp["feature"]]
        imp.to_csv(res / "feature_importance.csv", index=False)
    model.contribution_.to_csv(res / "contribution.csv", index=False)


def fit_predict(
    ctx: Context,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Интерфейс шаблона: train и val объединяются в обучающую выборку, test только предсказывается."""
    cfg = ctx.cfg
    if cfg["retrain_tabular"]:
        X, y = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        model = cv.GenderModel(cfg, ctx.device)
        model.cross_validate(X, y, ctx.seqs, ctx.encoder.state_dict())
        model.fit_final(X, y, ctx.seqs, ctx.encoder.state_dict())
        model.save(ctx.artifacts / "model")
    else:
        model = cv.GenderModel.load(ctx.artifacts / "model", cfg, ctx.device, ctx.seqs)
    proba = model.predict_proba(X_test, ctx.seqs)
    return proba, (proba >= cfg["blend"]["threshold"]).astype(int)


def write_cv_results(cfg: dict, model: cv.GenderModel, y: pd.Series) -> dict:
    res = Path(cfg["paths"]["results"])
    model.contribution_ = model.contribution(y)
    res.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(model.fold_metrics)
    folds.to_csv(res / "cv_folds.csv", index=False)
    summary = {
        "components": model.components,
        "blend_weights": model.blend_.weights_.tolist(),
        "auc_mean": folds["auc"].mean(),
        "auc_std": folds["auc"].std(ddof=1),
        "accuracy_mean": folds["accuracy"].mean(),
        "accuracy_std": folds["accuracy"].std(ddof=1),
        "folds_auc_below_0.88": int((folds["auc"] < 0.88).sum()),
        "folds_accuracy_below_0.80": int((folds["accuracy"] < 0.80).sum()),
        "n_folds": len(folds),
        **{f"auc_{c}_mean": folds[f"auc_{c}"].mean() for c in model.components},
    }
    (res / "cv_metrics.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def write_model_table(cfg: dict, model: cv.GenderModel, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Метрики CV по каждой модели и бленду -> results/cv_models.csv."""
    t = model.model_table(y)
    t = pd.concat([pd.DataFrame([cv.embedding_cv(X, y, cfg)]), t], ignore_index=True)
    t.to_csv(Path(cfg["paths"]["results"]) / "cv_models.csv", index=False)
    return t


def write_data_results(
    cfg: dict, X_lab: pd.DataFrame, y: pd.Series, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Сила сигнала групп признаков (метки train+val) и adversarial validation train vs test (без меток)."""
    res = Path(cfg["paths"]["results"])
    cv.group_signal(X_lab, y, cfg).to_csv(res / "group_signal.csv", index=False)
    adv = cv.adversarial_auc(X_train, X_test, cfg)
    (res / "adversarial_validation.json").write_text(json.dumps(adv, indent=1, ensure_ascii=False))


def evaluate(
    cfg: dict, model: cv.GenderModel | None = None, ctx: Context | None = None, splits: dict | None = None
) -> dict:
    """Единственный проход по test: вероятности, метрики, 95% bootstrap-интервалы."""
    splits = splits or {s: load_split(s, cfg["paths"]["data"]) for s in SPLITS}
    ctx = ctx or build_context(cfg, splits)
    model = model or cv.GenderModel.load(ctx.artifacts / "model", cfg, ctx.device, ctx.seqs)
    X_test, y_test = ctx.features(splits["test"]), labels(splits["test"])
    p = model.predict_proba(X_test, ctx.seqs)
    rep = cfg["report"]
    m = cv.bootstrap_ci(
        y_test.reindex(X_test.index).to_numpy(), p, cfg["blend"]["threshold"], rep["bootstrap"], rep["ci"]
    )
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "test_metrics.json").write_text(json.dumps(m, indent=1))
    pd.DataFrame(
        {"proba": p, "gender": y_test.reindex(X_test.index).to_numpy()}, index=X_test.index
    ).to_parquet(res / "test_predictions.parquet")
    return m


def figures(cfg: dict) -> None:
    from . import plots

    plots.make_all(cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gender-pred")
    ap.add_argument("command", choices=["data", "train", "evaluate", "figures"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--retrain", action="store_true", help="полный пересчёт вместо артефактов из Release")
    args = ap.parse_args(argv)
    cfg = load_config(
        args.config,
        **({"retrain": True, "retrain_tabular": True, "retrain_seq": True} if args.retrain else {}),
    )
    if args.command == "data":
        download_data(cfg["paths"]["data"])
    elif args.command == "train":
        download_data(cfg["paths"]["data"])
        train(dict(cfg, retrain=True, retrain_tabular=True, retrain_seq=True))
    elif args.command == "evaluate":
        try:
            print(json.dumps(evaluate(cfg), indent=1))
        except ArtifactsUnavailableError as e:
            sys.exit(f"Ошибка: {e}")
    else:
        figures(cfg)


if __name__ == "__main__":
    main()
