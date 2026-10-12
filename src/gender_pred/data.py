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

# начало периода: даты в датасете условные и отсчитываются от 2000-01-01
T0 = pd.Timestamp("2000-01-01")
# длина периода в днях (2000-01-01 … 2001-04-01) — размер дневной сетки для динамики
N_DAYS = 457
# имена сплитов задания; разбиение сделано по клиентам, не по времени
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- конфиг и окружение
def load_config(path: str | Path = "configs/default.yaml", **overrides: Any) -> dict:
    """Читает YAML-конфиг; ``overrides`` перекрывают ключи верхнего уровня (например ``retrain=True``)."""
    # весь пайплайн управляется одним YAML: гиперпараметры, сиды, пути, флаги пересчёта
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # аргументы вызова (например retrain=True из ноутбука) важнее значений из файла
    cfg.update(overrides)
    # частичные флаги пересчёта: если не заданы явно, наследуют общий флаг retrain
    for part in ("retrain_tabular", "retrain_seq"):
        if cfg.get(part) is None:
            cfg[part] = cfg["retrain"]
    return cfg


def get_device(cfg: dict) -> str:
    # torch импортируется лениво: модулю данных он нужен только здесь
    import torch

    # device: auto — GPU, если есть, иначе CPU; так ноутбук запускается на любой машине
    if cfg.get("device", "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]


def set_seed(seed: int) -> None:
    import torch

    # фиксируем все генераторы случайных чисел, чтобы результаты воспроизводились
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — глобальный сид для сторонних библиотек
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # хэширование строк в Python тоже детерминируем (порядок в множествах/словарях)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------- данные
def load_split(split: str, path: str | Path = "data") -> pd.DataFrame:
    """Сплит в формате шаблона задания: term_id NaN -> -1, сортировка по клиенту и времени."""
    df = pd.read_csv(Path(path) / f"{split}.csv")
    # пропуск терминала кодируем как -1 («терминал неизвестен»), как в шаблоне задания
    df["term_id"] = df["term_id"].fillna(-1)
    # строки с датой → datetime; нераспознанные стали бы NaT (в этих данных таких нет)
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    # порядок «клиент → время» нужен последовательной модели; stable сохраняет порядок равных
    return df.sort_values(["customer_id", "tr_datetime"], kind="stable").reset_index(drop=True)


def download_data(path: str | Path = "data") -> None:
    """Скачивает train/val/test.csv из HF-датасета ``mks-logic/gender_prediction``."""
    import shutil

    from huggingface_hub import hf_hub_download

    Path(path).mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = Path(path) / f"{split}.csv"
        # уже скачанные файлы не трогаем — повторный запуск ничего не перекачивает
        if not target.exists():
            # hf_hub_download кладёт файл в общий кэш Hugging Face; копируем его в data/
            src = hf_hub_download("mks-logic/gender_prediction", f"{split}.csv", repo_type="dataset")
            shutil.copy(src, target)


def labels(df: pd.DataFrame) -> pd.Series:
    """Пол клиента (index = customer_id)."""
    # пол повторяется в каждой строке клиента; берём первое значение → одна метка на клиента
    return df.groupby("customer_id")["gender"].first().sort_index()


# ---------------------------------------------------------------- обзор данных
def split_overview(
    splits: dict[str, pd.DataFrame], label_splits: tuple[str, ...] = ("train",)
) -> pd.DataFrame:
    """Строки, клиенты, период и баланс классов по сплитам; метки показываются только для ``label_splits``."""
    rows = {}
    for name, df in splits.items():
        # метки читаем только у разрешённых сплитов: метки test в анализе не используются
        lab = labels(df) if name in label_splits else None
        rows[name] = {
            # объём, период наблюдений и баланс классов сплита
            "строк": len(df),
            "клиентов": df["customer_id"].nunique(),
            "начало": df["tr_datetime"].min().date(),
            "конец": df["tr_datetime"].max().date(),
            "дней": (df["tr_datetime"].max() - df["tr_datetime"].min()).days + 1,
            # доля мужчин = среднее бинарной метки (1 — мужчина)
            "доля мужчин": round(lab.mean(), 3) if lab is not None else "—",
        }
    # словарь сплитов → таблица: строка на сплит, колонка на показатель
    return pd.DataFrame(rows).T


@dataclass
class Vocab:
    """Словари категорий, фиксируемые по train: одинаковые колонки на любом сплите."""

    # все MCC и типы операций, встреченные в train; категории, которых нет в train,
    # на других сплитах просто не попадут в признаки — набор колонок от сплита не зависит
    mcc: list[int]
    tr_type: list[int]
    # 40 самых частых MCC: только для них считается средний чек (у редких он шумный и часто пустой)
    mcc_top: list[int]

    @classmethod
    def from_train(cls, train: pd.DataFrame, top_mcc: int = 40) -> Vocab:
        # value_counts → MCC, отсортированные по числу операций (самые частые сверху)
        vc = train["mcc_code"].value_counts()
        return cls(
            # отсортированные списки кодов — порядок колонок признаков одинаков при каждом запуске
            mcc=sorted(int(x) for x in train["mcc_code"].unique()),
            tr_type=sorted(int(x) for x in train["tr_type"].unique()),
            mcc_top=sorted(int(x) for x in vc.index[:top_mcc]),
        )

    def to_dict(self) -> dict:
        # для сохранения словаря в artifacts/vocab.json: при загрузке модели колонки будут те же
        return {"mcc": self.mcc, "tr_type": self.tr_type, "mcc_top": self.mcc_top}


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    """Транзакции в компактном виде: день, час (полночь = время неизвестно), знак и лог-сумма."""
    # время операции в секундах от начала периода
    sec = (df["tr_datetime"] - T0).dt.total_seconds().to_numpy()
    # секунды от начала суток: 0 означает ровно 00:00:00
    tod = sec % 86400
    amt = df["amount"].to_numpy(dtype=float)
    # term_id после fillna(-1) бывает числом; приводим к строке, чтобы -1 и -1.0 совпали
    term = df["term_id"].astype(str).to_numpy()
    ev = pd.DataFrame(
        {
            # клиент, категория продавца (MCC) и тип операции — как есть
            "cid": df["customer_id"].to_numpy(),
            "mcc": df["mcc_code"].to_numpy(),
            "tt": df["tr_type"].to_numpy(),
            "amt": amt,
            # la = log(1 + |amount|): у сумм тяжёлые хвосты (максимум ≈ 30 × 99.9-й перцентиль),
            # в логарифме крупные операции не доминируют, а среднее la — это геометрическое
            # среднее чека; модуль — потому что знак кодирует направление (он хранится в out),
            # +1 — чтобы нулевая сумма давала 0, а не −∞
            "la": np.log1p(np.abs(amt)),
            # out: отрицательная сумма — расход, положительная — поступление
            "out": amt < 0,
            # все варианты пропуска терминала сводим к одной категории "-1" — «терминал неизвестен»
            "term": np.where(np.isin(term, ["-1", "-1.0", "nan"]), "-1", term),
            # номер дня от начала периода (целый) — для окон, интервалов и дневной сетки
            "day": (sec // 86400).astype(np.int32),
            # время в днях с дробной частью — для упорядочивания и затухающих средних
            "t": sec / 86400.0,
            # mid: время ровно 00:00:00 — это «время неизвестно» (так приходят 12% операций,
            # в основном один тип покупок), а не полночь; такие операции не идут в часовой профиль
            "mid": tod == 0,
            # час операции 0…23
            "hour": (tod // 3600).astype(np.int8),
        }
    )
    # день недели как day % 7: даты условные, поэтому настоящий weekday неизвестен,
    # но недельный цикл сохраняется — 7 значений, какие из них выходные, модель найдёт сама
    ev["dow"] = (ev["day"] % 7).astype(np.int8)
    # сортировка «клиент → время»: так операции клиента идут подряд в хронологическом порядке
    return ev.sort_values(["cid", "t"], kind="stable").reset_index(drop=True)


@dataclass
class SequenceStore:
    """Транзакции клиентов для последовательных моделей.

    ``generate_features`` регистрирует сюда события каждого сплита, а ``fit_predict_proba``
    достаёт последовательности по индексу таблицы признаков (customer_id).
    """

    # customer_id → его операции (уже подготовленные prepare_events)
    frames: dict[int, pd.DataFrame] = field(default_factory=dict)

    def add(self, events: pd.DataFrame) -> None:
        # раскладываем операции по клиентам; sort=False — не тратим время на сортировку групп
        for cid, g in events.groupby("cid", sort=False):
            self.frames[int(cid)] = g

    def events(self, ids: Any) -> pd.DataFrame:
        # собираем операции нужных клиентов обратно в одну таблицу в заданном порядке
        return pd.concat([self.frames[int(c)] for c in ids], ignore_index=True)


# общее хранилище: шаблон задания передаёт между функциями только таблицы признаков,
# а последовательности для CoLES достаются отсюда по customer_id
SEQ_STORE = SequenceStore()
