"""Загрузка данных, конфиг, подготовка транзакций, хранилище последовательностей и анализ данных."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

T0 = pd.Timestamp("2000-01-01")
N_DAYS = 457
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- конфиг и окружение
def load_config(path: str | Path = "configs/default.yaml", **overrides: Any) -> dict:
    """Читает YAML-конфиг; ``overrides`` перекрывают ключи верхнего уровня (например ``retrain=True``)."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(overrides)
    for part in ("retrain_tabular", "retrain_seq"):
        if cfg.get(part) is None:
            cfg[part] = cfg["retrain"]
    return cfg


def get_device(cfg: dict) -> str:
    import torch

    if cfg.get("device", "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — глобальный сид для сторонних библиотек
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------- данные
def load_split(split: str, path: str | Path = "data") -> pd.DataFrame:
    """Сплит в формате шаблона задания: term_id NaN -> -1, сортировка по клиенту и времени."""
    df = pd.read_csv(Path(path) / f"{split}.csv")
    df["term_id"] = df["term_id"].fillna(-1)
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    return df.sort_values(["customer_id", "tr_datetime"], kind="stable").reset_index(drop=True)


def download_data(path: str | Path = "data") -> None:
    """Скачивает train/val/test.csv из HF-датасета ``mks-logic/gender_prediction``."""
    import shutil

    from huggingface_hub import hf_hub_download

    Path(path).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = Path(path) / f"{split}.csv"
        if not target.exists():
            src = hf_hub_download("mks-logic/gender_prediction", f"{split}.csv", repo_type="dataset")
            shutil.copy(src, target)


def labels(df: pd.DataFrame) -> pd.Series:
    """Пол клиента (index = customer_id)."""
    return df.groupby("customer_id")["gender"].first().sort_index()


# ---------------------------------------------------------------- обзор данных
def split_overview(
    splits: dict[str, pd.DataFrame], label_splits: tuple[str, ...] = ("train",)
) -> pd.DataFrame:
    """Строки, клиенты, период и баланс классов по сплитам; метки показываются только для ``label_splits``."""
    rows = {}
    for name, df in splits.items():
        lab = labels(df) if name in label_splits else None
        rows[name] = {
            "строк": len(df),
            "клиентов": df["customer_id"].nunique(),
            "начало": df["tr_datetime"].min().date(),
            "конец": df["tr_datetime"].max().date(),
            "дней": (df["tr_datetime"].max() - df["tr_datetime"].min()).days + 1,
            "доля мужчин": round(lab.mean(), 3) if lab is not None else "—",
        }
    return pd.DataFrame(rows).T


def data_quality(df: pd.DataFrame) -> pd.Series:
    """Пропуски, дубликаты, нераспознанные даты, согласованность пола, выбросы сумм."""
    a = df["amount"].abs()
    p99, p999 = a.quantile([0.99, 0.999])
    per_client = df.groupby("customer_id")["gender"].nunique() if "gender" in df else None
    return pd.Series(
        {
            "строк": len(df),
            "полных дубликатов строк": int(df.duplicated().sum()),
            "нераспознанных дат (NaT)": int(df["tr_datetime"].isna().sum()),
            "клиентов с > 1 значением пола": int((per_client > 1).sum()) if per_client is not None else "—",
            "|amount|: 99-й перцентиль": round(p99, 2),
            "|amount|: 99.9-й перцентиль": round(p999, 2),
            "|amount|: максимум": round(a.max(), 2),
            "доля операций с |amount| > 99.9-го перцентиля": round((a > p999).mean(), 4),
        },
        dtype=object,
    )


@dataclass
class Vocab:
    """Словари категорий, фиксируемые по train: одинаковые колонки на любом сплите."""

    mcc: list[int]
    tr_type: list[int]
    mcc_top: list[int]

    @classmethod
    def from_train(cls, train: pd.DataFrame, top_mcc: int = 40) -> Vocab:
        vc = train["mcc_code"].value_counts()
        return cls(
            mcc=sorted(int(x) for x in train["mcc_code"].unique()),
            tr_type=sorted(int(x) for x in train["tr_type"].unique()),
            mcc_top=sorted(int(x) for x in vc.index[:top_mcc]),
        )

    def to_dict(self) -> dict:
        return {"mcc": self.mcc, "tr_type": self.tr_type, "mcc_top": self.mcc_top}


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    """Транзакции в компактном виде: день, час (полночь = время неизвестно), знак и лог-сумма."""
    sec = (df["tr_datetime"] - T0).dt.total_seconds().to_numpy()
    tod = sec % 86400
    amt = df["amount"].to_numpy(dtype=float)
    term = df["term_id"].astype(str).to_numpy()
    ev = pd.DataFrame(
        {
            "cid": df["customer_id"].to_numpy(),
            "mcc": df["mcc_code"].to_numpy(),
            "tt": df["tr_type"].to_numpy(),
            "amt": amt,
            "la": np.log1p(np.abs(amt)),
            "out": amt < 0,
            "term": np.where(np.isin(term, ["-1", "-1.0", "nan"]), "-1", term),
            "day": (sec // 86400).astype(np.int32),
            "t": sec / 86400.0,
            "mid": tod == 0,
            "hour": (tod // 3600).astype(np.int8),
        }
    )
    ev["dow"] = (ev["day"] % 7).astype(np.int8)
    return ev.sort_values(["cid", "t"], kind="stable").reset_index(drop=True)


@dataclass
class SequenceStore:
    """Транзакции клиентов для последовательных моделей.

    ``generate_features`` регистрирует сюда события каждого сплита, а ``fit_predict_proba``
    достаёт последовательности по индексу таблицы признаков (customer_id).
    """

    frames: dict[int, pd.DataFrame] = field(default_factory=dict)

    def add(self, events: pd.DataFrame) -> None:
        for cid, g in events.groupby("cid", sort=False):
            self.frames[int(cid)] = g

    def events(self, ids: Any) -> pd.DataFrame:
        return pd.concat([self.frames[int(c)] for c in ids], ignore_index=True)


SEQ_STORE = SequenceStore()


# ---------------------------------------------------------------- анализ данных (раздел «Данные» ноутбука)


def midnight_check(df: pd.DataFrame, top: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """00:00:00 против соседних секунд; типы операций с наибольшей долей 00:00:00."""
    sec = (df["tr_datetime"] - df["tr_datetime"].dt.normalize()).dt.total_seconds()
    mid = sec == 0
    summary = pd.Series(
        {
            "операций в 00:00:00": int(mid.sum()),
            "в среднем за секунду 00:00:01–00:00:59": round(sec.between(1, 59).sum() / 59, 1),
            "в среднем за секунду суток": round(len(sec) / 86400, 1),
            "доля операций в 00:00:00": round(mid.mean(), 3),
        },
        dtype=object,
    )
    by_type = (
        df.assign(mid=mid)
        .groupby("tr_type")
        .agg(desc=("tr_type_desc", "first"), операций=("mid", "size"), доля_00=("mid", "mean"))
        .sort_values("операций", ascending=False)
    )
    by_type["доля всех 00:00:00"] = (by_type["операций"] * by_type["доля_00"]) / mid.sum()
    return summary, by_type.sort_values("доля всех 00:00:00", ascending=False).head(top).round(3)


def amount_scale(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Масштаб сумм: самая частая |amount| как единица и распределение |amount| в этих единицах."""
    a = df["amount"].abs()
    unit = float(a.mode().iloc[0])
    r = a / unit
    return unit, r.round(2).value_counts().head(12).rename("операций")


def round_share(df: pd.DataFrame, unit: float, tol: float = 0.002) -> pd.Series:
    """Доля операций клиента с целым числом единиц масштаба (index = customer_id)."""
    r = df["amount"].abs() / unit
    return (np.abs(r - np.round(r)) < tol).groupby(df["customer_id"]).mean()


def terminal_overview(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Клиентов на терминал (train) и пересечение терминалов train/test (без меток)."""

    def known(df: pd.DataFrame) -> pd.DataFrame:
        t = df["term_id"].astype(str)
        return df.assign(term=t)[~t.isin(["-1", "-1.0", "nan"])]

    tr, te = known(train), known(test)
    per_term = tr.groupby("term")["customer_id"].nunique()
    seen = te["term"].isin(set(tr["term"]))
    return pd.Series(
        {
            "терминалов в train": len(per_term),
            "клиентов на терминал, медиана": int(per_term.median()),
            "доля терминалов с одним клиентом": round((per_term == 1).mean(), 3),
            "доля операций test на терминалах из train": round(seen.mean(), 3),
            "доля клиентов test с хотя бы одним терминалом из train": round(
                te.loc[seen, "customer_id"].nunique() / test["customer_id"].nunique(), 3
            ),
        },
        dtype=object,
    )
