"""Графики по сохранённым results/. Единый стиль; сохранение в reports/figures/NN_name.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .features import GROUP_PREFIXES

FEMALE, MALE = "#D1495B", "#30638E"
NEUTRAL, ACCENT, INK = "#A7ADB5", "#3E8E7E", "#2B2D33"
GROUP_COLORS = {
    "mcc": "#30638E",
    "te": "#5A8DB8",
    "lr": "#3E8E7E",
    "amt": "#D1495B",
    "time": "#E39B5B",
    "dyn": "#8C6BB1",
    "act": "#A7ADB5",
    "tt": "#6C7A89",
    "term": "#C9B27C",
    "emb": "#3E8E7E",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "figure.dpi": 100,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
        }
    )


def table(df: pd.DataFrame, precision: int = 3):
    """Таблица для ноутбука: выравнивание влево (заголовки и ячейки), фиксированная точность чисел."""
    left = [("text-align", "left")]
    return (
        df.style.format(precision=precision, na_rep="—")
        .set_properties(**{"text-align": "left"})
        .set_table_styles([{"selector": "th", "props": left}, {"selector": "td", "props": left}])
    )


def _save(fig: plt.Figure, cfg: dict, name: str) -> Path:
    out = Path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, transparent=False)
    return path


def _results(cfg: dict) -> Path:
    return Path(cfg["paths"]["results"])


def hourly_by_gender(df: pd.DataFrame) -> plt.Figure:
    """Доля операций по часам у мужчин и женщин; операции с временем 00:00:00 исключены (время неизвестно)."""
    t = df["tr_datetime"]
    known = (t - t.dt.normalize()) != pd.Timedelta(0)
    d = df.loc[known].assign(hour=t[known].dt.hour)
    share = d.groupby("gender")["hour"].value_counts(normalize=True).unstack(0).sort_index() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(share.index, share[0], color=FEMALE, lw=2, marker="o", ms=3, label="женщины")
    ax.plot(share.index, share[1], color=MALE, lw=2, marker="o", ms=3, label="мужчины")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("час операции")
    ax.set_ylabel("доля операций, %")
    ax.set_title("Время операций по полу (train, без 00:00:00)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def group_signal(cfg: dict) -> plt.Figure:
    """AUC LightGBM на каждой группе признаков отдельно (5-fold на train+val)."""
    from .features import GROUP_PREFIXES

    d = pd.read_csv(_results(cfg) / "group_signal.csv").sort_values("auc")
    names = [GROUP_PREFIXES.get(g, "Все агрегаты вместе") for g in d["group"]]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    colors = [ACCENT if g == "all" else GROUP_COLORS.get(g, NEUTRAL) for g in d["group"]]
    ax.barh(
        range(len(d)),
        d["auc"],
        xerr=d["auc_std"],
        color=colors,
        height=0.65,
        error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2},
    )
    ax.set_yticks(range(len(d)), [f"{n} ({k})" for n, k in zip(names, d["n_features"])])
    for i, v in enumerate(d["auc"]):
        ax.text(0.502, i, f"{v:.3f}", va="center", fontsize=8.5, color="white" if v > 0.6 else INK)
    ax.set_xlim(0.5, 0.92)
    ax.axvline(0.88, color=FEMALE, lw=1, ls="--")
    ax.text(0.878, -0.45, "требование 0.88", color=FEMALE, fontsize=8, ha="right", va="center")
    ax.set_xlabel("ROC AUC, 5 фолдов на train + val (в скобках — число признаков)")
    ax.set_title("Сила сигнала групп признаков (LightGBM на одной группе)")
    fig.tight_layout()
    return fig


def mcc_gender_skew(cfg: dict, top: int = 10) -> plt.Figure:
    """Diverging bar: MCC с наибольшим перекосом по полу (отклонение доли мужчин от средней, train)."""
    d = pd.read_csv(_results(cfg) / "mcc_gender_skew.csv")
    d = d.assign(delta=d["male_share"] - d["base"]).sort_values("delta")
    d = pd.concat([d.head(top), d.tail(top)])
    # 95% интервал Уилсона для доли мужчин среди клиентов с этим MCC
    z, n, p = 1.96, d["n_clients"], d["male_share"]
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    lo, hi = (centre - half - d["base"]) * 100, (centre + half - d["base"]) * 100
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = np.where(d["delta"] < 0, FEMALE, MALE)
    ax.barh(range(len(d)), d["delta"] * 100, color=colors, height=0.72)
    x = d["delta"].to_numpy() * 100
    ax.errorbar(
        x,
        range(len(d)),
        xerr=[x - lo.to_numpy(), hi.to_numpy() - x],
        fmt="none",
        ecolor=INK,
        lw=0.8,
        capsize=2,
    )
    ax.set_yticks(range(len(d)), [s if len(s) <= 46 else s[:44] + "…" for s in d["desc"]], fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Доля мужчин среди клиентов с этим MCC минус средняя, п.п. (95% CI)")
    ax.set_title("MCC с наибольшим перекосом по полу", pad=22)
    kw = dict(transform=ax.transAxes, fontsize=9, va="bottom")
    ax.text(0.0, 1.01, "← чаще женщины", color=FEMALE, **kw)
    ax.text(1.0, 1.01, "чаще мужчины →", color=MALE, ha="right", **kw)
    fig.tight_layout()
    return fig


def component_contribution(cfg: dict) -> plt.Figure:
    """Вклад компонентов: AUC (и accuracy) на кросс-валидации от агрегатов к итоговому бленду."""
    d = pd.read_csv(_results(cfg) / "contribution.csv")
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(d) + 1.4))
    y = np.arange(len(d))[::-1]
    colors = [NEUTRAL] * (len(d) - 1) + [ACCENT]
    ax.barh(y, d["auc"], color=colors, height=0.6)
    ax.set_xlim(0.875, d["auc"].max() + 0.007)
    for yi, (_, r) in zip(y, d.iterrows()):
        delta = "" if pd.isna(r.get("delta_auc")) else f"  (+{r['delta_auc']:.4f})"
        ax.text(
            r["auc"] + 0.0004,
            yi,
            f"{r['auc']:.4f}{delta}   accuracy {r['accuracy']:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, d["step"])
    ax.axvline(0.88, color=FEMALE, lw=1, ls="--")
    ax.text(0.8803, y.min() - 0.45, "требование задания 0.88", color=FEMALE, fontsize=8.5, va="top")
    ax.set_ylim(y.min() - 0.9, y.max() + 0.5)
    ax.set_xlabel("ROC AUC, кросс-валидация 2×5 фолдов (accuracy при пороге 0.5)")
    ax.set_title("Вклад компонентов")
    fig.tight_layout()
    return fig


def roc_test(cfg: dict) -> plt.Figure:
    """ROC на test с AUC и 95% доверительным интервалом."""
    p = pd.read_parquet(_results(cfg) / "test_predictions.parquet")
    m = json.loads((_results(cfg) / "test_metrics.json").read_text())
    fpr, tpr, _ = roc_curve(p["gender"], p["proba"])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    ax.plot(fpr, tpr, color=MALE, lw=2)
    ax.set_xlabel("Доля ложноположительных")
    ax.set_ylabel("Доля истинноположительных")
    ax.set_title("ROC на test")
    ax.text(
        0.97,
        0.05,
        f"AUC = {m['auc']:.3f}\n95% CI [{m['auc_ci'][0]:.3f}; {m['auc_ci'][1]:.3f}]\n"
        f"accuracy = {m['accuracy']:.3f}\n95% CI [{m['accuracy_ci'][0]:.3f}; {m['accuracy_ci'][1]:.3f}]",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9.5,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def feature_importance(cfg: dict, top: int = 15) -> plt.Figure:
    """Топ признаков финального LightGBM по gain, цвет — группа признаков."""
    d = pd.read_csv(_results(cfg) / "feature_importance.csv").head(top).iloc[::-1]
    groups = d["feature"].str.split("_").str[0]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(range(len(d)), d["gain_share"] * 100, color=[GROUP_COLORS.get(g, NEUTRAL) for g in groups])
    ax.set_yticks(range(len(d)), d["label"], fontsize=8.5)
    ax.set_xlabel("Доля суммарного gain, %")
    ax.set_title("Важность признаков (LightGBM)")
    used = list(dict.fromkeys(groups[::-1]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(g, NEUTRAL)) for g in used]
    ax.legend(
        handles, [GROUP_PREFIXES.get(g, g) for g in used], frameon=False, fontsize=8.5, loc="lower right"
    )
    fig.tight_layout()
    return fig


def pipeline_schema(cfg: dict | None = None) -> plt.Figure:
    """Схема пайплайна (та же, что mermaid-диаграмма в README)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = {
        "data": (0.07, 0.5, "Транзакции\nклиента", NEUTRAL),
        "agg": (0.30, 0.74, "Агрегаты:\nMCC-профиль, суммы,\nвремя, динамика", NEUTRAL),
        "coles": (0.30, 0.24, "CoLES-энкодер\n(GRU, предобучение\nбез меток)", NEUTRAL),
        "gbm": (0.55, 0.86, "LightGBM", MALE),
        "tabpfn": (0.55, 0.58, "TabPFN", MALE),
        "ft": (0.55, 0.24, "CoLES, дообученный\nна пол", MALE),
        "blend": (0.78, 0.5, "Калибровка\nи бленд логитов", ACCENT),
        "pred": (0.95, 0.5, "Вероятность\nи класс", ACCENT),
    }
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.05, 1.02)
    ax.axis("off")
    h = 0.2

    def width(key: str) -> float:
        return 0.09 if key in ("data", "pred") else 0.16

    for key, (x, y, text, color) in boxes.items():
        bw = width(key)
        ax.add_patch(
            FancyBboxPatch(
                (x - bw / 2, y - h / 2),
                bw,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                fc="white",
                ec=color,
                lw=1.6,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8.6, color=INK)

    def arrow(a: str, b: str, label: str = "") -> None:
        (x1, y1), (x2, y2) = boxes[a][:2], boxes[b][:2]
        if abs(x1 - x2) < 1e-9:  # вертикальная стрелка: от верхнего края к нижнему
            p1, p2 = (x1, y1 + h / 2 + 0.012), (x2, y2 - h / 2 - 0.012)
        else:  # от правого края к левому
            p1, p2 = (x1 + width(a) / 2 + 0.012, y1), (x2 - width(b) / 2 - 0.012, y2)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color=INK, lw=0.9))
        if label:
            ax.text(x1 + 0.012, (p1[1] + p2[1]) / 2, label, fontsize=8, color=ACCENT, va="center")

    for a, b in [
        ("data", "agg"),
        ("data", "coles"),
        ("agg", "gbm"),
        ("agg", "tabpfn"),
        ("coles", "ft"),
        ("gbm", "blend"),
        ("tabpfn", "blend"),
        ("ft", "blend"),
        ("blend", "pred"),
    ]:
        arrow(a, b)
    arrow("coles", "agg", "эмбеддинг")
    ax.set_title("Схема решения")
    fig.tight_layout()
    return fig


FIGURES = {
    "00_pipeline.png": pipeline_schema,
    "01_mcc_gender_skew.png": mcc_gender_skew,
    "02_component_contribution.png": component_contribution,
    "03_roc_test.png": roc_test,
    "04_feature_importance.png": feature_importance,
    "06_group_signal.png": group_signal,
}
# графики по сырым транзакциям train (нужен data/train.csv)
DATA_FIGURES = {
    "05_hourly_by_gender.png": hourly_by_gender,
}


def make_all(cfg: dict, show: bool = False) -> list[Path]:
    from .data import load_split

    style()
    train_csv = Path(cfg["paths"]["data"]) / "train.csv"
    train = load_split("train", cfg["paths"]["data"]) if train_csv.exists() else None
    jobs = [(name, lambda fn=fn: fn(cfg)) for name, fn in FIGURES.items()]
    if train is not None:
        jobs += [(name, lambda fn=fn: fn(train)) for name, fn in DATA_FIGURES.items()]
    saved = []
    for name, make in jobs:
        try:
            fig = make()
        except FileNotFoundError:
            continue
        if fig is None:
            continue
        saved.append(_save(fig, cfg, name))
        if show:
            plt.show()
        plt.close(fig)
    return saved
