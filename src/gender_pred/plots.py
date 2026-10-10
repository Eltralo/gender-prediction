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


def _by_gender(ax: plt.Axes, values: pd.Series, gender: pd.Series, bins: np.ndarray) -> None:
    for g, color, name in ((0, FEMALE, "женщины"), (1, MALE, "мужчины")):
        v = values[gender.reindex(values.index).to_numpy() == g]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=color, label=name)


def time_volume(df: pd.DataFrame) -> plt.Figure:
    """Операций по дням периода и доля операций по day % 7 (train)."""
    day = (df["tr_datetime"] - df["tr_datetime"].min()).dt.days
    daily = day.value_counts().sort_index()
    dow = (day % 7).value_counts(normalize=True).sort_index() * 100
    low = dow.nsmallest(2).index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.4, 1]})
    a1.plot(daily.index, daily.to_numpy(), color=NEUTRAL, lw=0.8)
    a1.plot(
        daily.index,
        daily.rolling(28, center=True).mean().to_numpy(),
        color=MALE,
        lw=2,
        label="среднее за 28 дней",
    )
    a1.set_xlabel("день от начала периода")
    a1.set_ylabel("операций в день")
    a1.set_title("Объём операций по дням (train)")
    a1.legend(frameon=False, loc="lower right")
    a2.bar(dow.index, dow.to_numpy(), color=[ACCENT if d in low else NEUTRAL for d in dow.index])
    a2.set_ylim(dow.min() * 0.9, dow.max() * 1.05)
    a2.set_xlabel("day % 7")
    a2.set_ylabel("доля операций, %")
    a2.set_title("По дням недели")
    a2.text(
        0.98,
        0.97,
        "провалы —\nвероятно, выходные",
        color=ACCENT,
        transform=a2.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig


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


def activity_by_gender(df: pd.DataFrame) -> plt.Figure:
    """Распределение числа операций и активных дней на клиента по полу (train, лог-шкала)."""
    from .data import client_activity, labels

    act, g = client_activity(df), labels(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col in zip(axes, ["операций", "активных дней"]):
        v = np.log10(act[col])
        _by_gender(ax, v, g, np.linspace(0, v.max(), 40))
        ax.set_xlabel(f"{col} на клиента")
        ticks = [t for t in (1, 10, 100, 1000, 10000) if np.log10(t) <= v.max()]
        ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
        ax.set_yticks([])
        ax.set_title(f"{col.capitalize()} на клиента (train)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def amounts_by_gender(df: pd.DataFrame) -> plt.Figure:
    """log(1 + |amount|) по полу отдельно для расходов и поступлений (train, по операциям)."""
    from .data import labels

    g = labels(df)
    gender = df["customer_id"].map(g)
    la = np.log1p(df["amount"].abs())
    bins = np.linspace(*la.quantile([0.0005, 0.9995]), 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, mask, name in (
        (axes[0], df["amount"] < 0, "Расходы"),
        (axes[1], df["amount"] > 0, "Поступления"),
    ):
        for gv, color, lab in ((0, FEMALE, "женщины"), (1, MALE, "мужчины")):
            ax.hist(
                la[mask & (gender == gv)],
                bins=bins,
                density=True,
                histtype="step",
                lw=1.8,
                color=color,
                label=lab,
            )
        ax.set_title(f"{name} ({mask.mean():.0%} операций, train)")
        ax.set_xlabel("log(1 + |amount|)")
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def mcc_coverage(df: pd.DataFrame) -> plt.Figure:
    """Накопленная доля операций по MCC, отсортированным по частоте."""
    from .data import mcc_coverage as coverage

    c = coverage(df) * 100
    k80, k95 = int((c < 80).sum() + 1), int((c < 95).sum() + 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(np.arange(1, len(c) + 1), c.to_numpy(), color=MALE, lw=2)
    for k, lvl in ((k80, 80), (k95, 95)):
        ax.plot([k, k, 0], [0, lvl, lvl], color=ACCENT, lw=1, ls="--")
        ax.text(k + 3, lvl - 6, f"{k} MCC — {lvl}% операций", color=ACCENT, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, len(c))
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_ylim(0, 101)
    ax.set_xlabel("число самых частых MCC (лог-шкала)")
    ax.set_ylabel("доля операций, %")
    ax.set_title(f"Покрытие операций кодами MCC (всего {len(c)}, train)")
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


def client_card(
    card: dict,
    ref,
    title: str = "",
    contrib: pd.Series | None = None,
    labels: dict[str, str] | None = None,
    top: int = 8,
) -> plt.Figure:
    """Карточка клиента: предпочтения (доля покупок, lift к среднему), часы и дни недели против среднего
    клиента; RFM, уровень трат, разнообразие и регулярность — в подзаголовке. С ``contrib`` — ещё панель
    вклада топ-признаков в логит LightGBM."""
    if contrib is None:
        fig, (a1, a2, a3) = plt.subplots(
            1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1.6, 1.1, 0.8]}
        )
    else:
        fig = plt.figure(figsize=(13, 7.2))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.1, 0.8], height_ratios=[1, 1.15])
        a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
        _contributions(fig.add_subplot(gs[1, :]), contrib, labels or {}, top)
    pref = card["предпочтения"].iloc[::-1]
    a1.barh(range(len(pref)), pref["доля покупок"] * 100, color=MALE, height=0.6)
    a1.set_yticks(range(len(pref)), [c[:40] for c in pref["категория"]], fontsize=8.5)
    for i, (sh, lf) in enumerate(zip(pref["доля покупок"], pref["lift"])):
        txt = f"{sh:.0%}" + ("" if np.isnan(lf) else f"  ×{lf:.1f}")
        a1.text(sh * 100 + 1, i, txt, va="center", fontsize=8.5)
    a1.set_xlim(0, pref["доля покупок"].max() * 100 * 1.45)
    a1.set_xlabel("доля покупок, %   (×lift к среднему клиенту)")
    a1.set_title("Предпочтения (покупки)", fontsize=10.5)
    a2.plot(ref.hours.index, ref.hours * 100, color=NEUTRAL, lw=1.5, label="средний клиент")
    a2.plot(card["часы"].index, card["часы"] * 100, color=ACCENT, lw=2, label="клиент")
    a2.set_xticks(range(0, 24, 6))
    a2.set_xlabel("час")
    a2.set_title("Часы (без 00:00:00), %", fontsize=10.5)
    a2.legend(frameon=False, fontsize=8)
    w = 0.38
    a3.bar(ref.dow.index - w / 2, ref.dow * 100, width=w, color=NEUTRAL)
    a3.bar(card["дни недели"].index + w / 2, card["дни недели"] * 100, width=w, color=ACCENT)
    a3.set_xticks(range(7))
    a3.set_xlabel("day % 7")
    a3.set_title("Дни недели, %", fontsize=10.5)
    lvl, div, reg, r = card["уровень трат"], card["разнообразие"], card["регулярность"], card["RFM"]

    def v(k: str) -> str:
        return (
            f"{r.loc[k, 'значение']:,.0f}".replace(",", " ")
            + f" (p{r.loc[k, 'перцентиль среди train'] * 100:.0f})"
        )

    rfm_line = (
        f"RFM: recency {v('recency, дней')} дн · {v('операций в месяц')} опер./мес · "
        f"{v('активных дней в месяц')} акт. дн./мес · расходы {v('сумма расходов')} · чек {v('средний чек')}"
    )
    rest = (
        f"покупки {lvl['доля покупок в расходах']:.0%} расходов · "
        f"крупных {lvl['доля крупных (> p90 train)']:.0%}"
        f" · категорий {div['категорий']}, "
        f"HHI {div['HHI покупок']:.2f}"
        f" · активных дней {reg['доля активных дней']:.0%}, CV интервалов {reg['вариация интервалов (CV)']}"
    )
    fig.suptitle(f"{title}\n{rfm_line}\n{rest}", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return fig


def rfm_by_gender(r: pd.DataFrame, gender: pd.Series) -> plt.Figure:
    """Распределения RFM по полу (train, лог-шкала для частоты и денег)."""
    cols = list(r.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.0))
    for ax, c in zip(axes, cols):
        v = r[c] if c.startswith("recency") else np.log10(r[c].clip(lower=1e-9))
        lo, hi = v.quantile([0.01, 0.99])
        _by_gender(ax, v.clip(lo, hi), gender, np.linspace(lo, hi, 35))
        ax.set_title(c, fontsize=10)
        ax.set_yticks([])
        if not c.startswith("recency"):
            ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
            ax.set_xticks(ticks, [f"{10**t:,.0f}".replace(",", " ") for t in ticks], fontsize=8)
    axes[0].set_xlim(0, r[cols[0]].quantile(0.95))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("RFM по полу (train)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def mcc_gender_lift(d: pd.DataFrame, top: int = 10) -> plt.Figure:
    """Lift MCC (доля мужчин-пользователей / доля женщин-пользователей) с 95% CI, топ в каждую сторону."""
    d = pd.concat([d.head(top), d.tail(top)])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.log2(d["lift"].to_numpy())
    ax.barh(range(len(d)), x, color=np.where(x < 0, FEMALE, MALE), height=0.72)
    lo, hi = np.log2(d["lo"].to_numpy()), np.log2(d["hi"].to_numpy())
    ax.errorbar(x, range(len(d)), xerr=[x - lo, hi - x], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_yticks(range(len(d)), [s if len(s) <= 46 else s[:44] + "…" for s in d["desc"]], fontsize=8.5)
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    ax.set_xticks(ticks, [f"×{2.0**t:g}" if t >= 0 else f"1/{2.0**-t:g}" for t in ticks])
    ax.set_xlim(min(lo.min(), -1) - 0.2, max(hi.max(), 1) + 0.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("lift: доля мужчин, пользующихся MCC / доля женщин (лог-шкала, 95% CI)")
    ax.set_title("Предпочтения мужчин и женщин (train, MCC с ≥ 200 клиентами)", pad=22)
    kw = dict(transform=ax.transAxes, fontsize=9, va="bottom")
    ax.text(0.0, 1.01, "← чаще женщины", color=FEMALE, **kw)
    ax.text(1.0, 1.01, "чаще мужчины →", color=MALE, ha="right", **kw)
    fig.tight_layout()
    return fig


def shap_summary(d: pd.DataFrame, labels: dict[str, str]) -> plt.Figure:
    """Средний |вклад| признаков в логит LightGBM; цвет — направление (больше значение → мужчина/женщина)."""
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(d) + 1.3))
    ax.barh(range(len(d)), d["mean_abs"], color=np.where(d["direction"] > 0, MALE, FEMALE), height=0.65)
    ax.set_yticks(range(len(d)), [labels.get(f, f)[:56] for f in d.index], fontsize=8.5)
    ax.set_xlabel("средний |вклад| в логит LightGBM (pred_contrib)")
    ax.set_title("Какие признаки определяют пол")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MALE, FEMALE)]
    ax.legend(
        handles,
        ["больше значение → мужчина", "больше значение → женщина"],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    fig.tight_layout()
    return fig


def _contributions(ax: plt.Axes, contrib: pd.Series, labels: dict[str, str], top: int) -> None:
    c = contrib.drop("bias")
    c = c[c.abs().sort_values(ascending=False).index[:top]].iloc[::-1]
    ax.barh(range(len(c)), c.to_numpy(), color=np.where(c > 0, MALE, FEMALE), height=0.65)
    ax.set_yticks(range(len(c)), [labels.get(k, k)[:60] for k in c.index], fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("вклад в логит LightGBM  (← женщина | мужчина →)")
    ax.set_title(f"Вклад топ-{top} признаков в предсказание LightGBM", fontsize=10.5)


def client_contributions(contrib: pd.Series, labels: dict[str, str], title: str, top: int = 8) -> plt.Figure:
    """Вклад топ-признаков в логит LightGBM для одного клиента: > 0 — к «мужчине», < 0 — к «женщине»."""
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.2))
    _contributions(ax, contrib, labels, top)
    ax.set_title(title)
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
    "05_coles_umap.png": coles_umap,
    "11_group_signal.png": group_signal,
}
# графики по сырым транзакциям train (нужен data/train.csv)
DATA_FIGURES = {
    "06_time_volume.png": time_volume,
    "07_hourly_by_gender.png": hourly_by_gender,
    "08_activity_by_gender.png": activity_by_gender,
    "09_amounts_by_gender.png": amounts_by_gender,
    "10_mcc_coverage.png": mcc_coverage,
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
