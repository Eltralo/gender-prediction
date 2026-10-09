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


def _save(fig: plt.Figure, cfg: dict, name: str) -> Path:
    out = Path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, transparent=False)
    return path


def _results(cfg: dict) -> Path:
    return Path(cfg["paths"]["results"])


def mcc_gender_skew(cfg: dict, top: int = 10) -> plt.Figure:
    """Diverging bar: MCC с наибольшим перекосом по полу (отклонение доли мужчин от средней, train)."""
    d = pd.read_csv(_results(cfg) / "mcc_gender_skew.csv")
    d = d.assign(delta=d["male_share"] - d["base"]).sort_values("delta")
    d = pd.concat([d.head(top), d.tail(top)])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = np.where(d["delta"] < 0, FEMALE, MALE)
    ax.barh(range(len(d)), d["delta"] * 100, color=colors, height=0.72)
    ax.set_yticks(range(len(d)), [s if len(s) <= 46 else s[:44] + "…" for s in d["desc"]], fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Доля мужчин среди клиентов с этим MCC минус средняя доля, п.п.")
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


def coles_umap(cfg: dict) -> plt.Figure | None:
    """UMAP эмбеддингов CoLES по полу (если файл проекции есть)."""
    path = _results(cfg) / "coles_umap.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for g, color, name in ((0, FEMALE, "женщины"), (1, MALE, "мужчины")):
        s = d[d["gender"] == g]
        ax.scatter(s["x"], s["y"], s=3, alpha=0.45, color=color, label=name, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Эмбеддинги CoLES (UMAP)")
    ax.legend(frameon=False, markerscale=4)
    fig.tight_layout()
    return fig


FIGURES = {
    "01_mcc_gender_skew.png": mcc_gender_skew,
    "02_component_contribution.png": component_contribution,
    "03_roc_test.png": roc_test,
    "04_feature_importance.png": feature_importance,
    "05_coles_umap.png": coles_umap,
}


def make_all(cfg: dict, show: bool = False) -> list[Path]:
    style()
    saved = []
    for name, fn in FIGURES.items():
        try:
            fig = fn(cfg)
        except FileNotFoundError:
            continue
        if fig is None:
            continue
        saved.append(_save(fig, cfg, name))
        if show:
            plt.show()
        plt.close(fig)
    return saved
