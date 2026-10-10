"""Клиентские признаки.

Без меток (считаются на любом сплите отдельно, колонки фиксированы словарём из train):
``act_`` активность, ``amt_`` суммы, ``mcc_`` MCC-профиль, ``tt_`` профиль типов операций,
``time_`` время, ``term_`` терминалы, ``dyn_`` концентрация / затухающие средние / автокорреляция / окна.

С метками (только внутри обучения: для обучающих клиентов — out-of-fold по внутренним фолдам):
``te_`` сглаженный target encoding MCC и типов операций, ``lr_tfidf`` логрег по TF-IDF профилей,
``lr_emb`` логрег по эмбеддингам CoLES.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import N_DAYS, T0, Vocab

GROUP_PREFIXES = {
    "act": "Активность",
    "amt": "Суммы",
    "mcc": "MCC-профиль",
    "tt": "Типы операций",
    "time": "Время",
    "term": "Терминалы",
    "dyn": "Динамика",
    "te": "Target encoding",
    "lr": "Линейные модели",
    "raw": "Счётчики",
    "emb": "Эмбеддинги CoLES",
}


# ---------------------------------------------------------------- утилиты
def _share(ev: pd.DataFrame, key: str, cols, weight: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = ev.groupby(["cid", key])
    m = (g.size() if weight is None else g[weight].sum()).unstack(fill_value=0)
    m = m.reindex(columns=cols, fill_value=0).astype(float)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0), m


def _entropy(m: pd.DataFrame) -> np.ndarray:
    p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0).sum(axis=1)


def _hhi(m: pd.DataFrame) -> np.ndarray:
    p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    return (p**2).sum(axis=1)


def _autocorr(M: np.ndarray, mask: np.ndarray, lag: int) -> np.ndarray:
    a, b = M[:, :-lag], M[:, lag:]
    v = mask[:, :-lag] & mask[:, lag:]
    n = v.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(v, a, 0).sum(axis=1) / n
        mb = np.where(v, b, 0).sum(axis=1) / n
        da, db = np.where(v, a - ma[:, None], 0), np.where(v, b - mb[:, None], 0)
        r = (da * db).sum(axis=1) / np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    r[n < 10] = np.nan
    return r


# ---------------------------------------------------------------- группы без меток
def activity(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    g = ev.groupby("cid")
    f = pd.DataFrame(
        {
            "n_tr": g.size(),
            "n_days": g["day"].nunique(),
            "first_day": g["day"].min(),
            "last_day": g["day"].max(),
            "n_mcc": g["mcc"].nunique(),
            "n_tt": g["tt"].nunique(),
        }
    )
    f["span"] = f["last_day"] - f["first_day"] + 1
    f["days_share"] = f["n_days"] / f["span"]
    f["tr_per_day"] = f["n_tr"] / f["n_days"]
    f["log_n_tr"] = np.log1p(f["n_tr"])
    ud = ev[["cid", "day"]].drop_duplicates()
    gap = ud["day"].groupby(ud["cid"]).diff().groupby(ud["cid"])
    f["gap_mean"], f["gap_std"], f["gap_max"], f["gap_med"] = gap.mean(), gap.std(), gap.max(), gap.median()
    mon = ev.assign(m=ev["day"] // 30).groupby(["cid", "m"]).size().unstack(fill_value=0)
    cols = mon.columns.to_numpy()
    lo = (f["first_day"] // 30).reindex(mon.index).to_numpy()[:, None]
    hi = (f["last_day"] // 30).reindex(mon.index).to_numpy()[:, None]
    mask = (cols[None, :] >= lo) & (cols[None, :] <= hi)
    mv = np.where(mask, mon.to_numpy(), np.nan)
    f["month_cv"] = pd.Series(np.nanstd(mv, 1) / (np.nanmean(mv, 1) + 1e-9), index=mon.index)
    f["month_active_share"] = pd.Series(np.nansum(mv > 0, 1) / mask.sum(axis=1), index=mon.index)
    return f.reindex(idx)


def amounts(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for name, m in (("out", ev["out"]), ("in", ~ev["out"])):
        s = ev[m]
        g = s.groupby("cid")
        a = s["amt"].abs().groupby(s["cid"])
        out[f"{name}_cnt"] = g.size()
        out[f"{name}_logsum"] = np.log1p(a.sum())
        for stat in ("mean", "std", "min", "max", "skew"):
            out[f"{name}_la_{stat}"] = getattr(g["la"], stat)()
        q = g["la"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
        for c in q.columns:
            out[f"{name}_la_q{int(round(c * 100))}"] = q[c]
        out[f"{name}_max_share"] = a.max() / a.sum()
        out[f"{name}_amt_nuniq_share"] = s["amt"].groupby(s["cid"]).nunique() / g.size()
    f = pd.DataFrame(out).reindex(idx)
    f[["out_cnt", "in_cnt"]] = f[["out_cnt", "in_cnt"]].fillna(0)
    f["in_cnt_share"] = f["in_cnt"] / (f["in_cnt"] + f["out_cnt"])
    f["log_balance"] = f["in_logsum"].fillna(0) - f["out_logsum"].fillna(0)
    f["la_mean_all"] = ev.groupby("cid")["la"].mean().reindex(idx)
    return f


def mcc_profile(ev: pd.DataFrame, idx: np.ndarray, vocab: Vocab) -> tuple[pd.DataFrame, dict]:
    cnt_share, cnt = _share(ev, "mcc", vocab.mcc)
    out = ev[ev["out"]].assign(a=lambda x: x["amt"].abs())
    sum_share, summ = _share(out, "mcc", vocab.mcc, weight="a")
    la = out[out["mcc"].isin(vocab.mcc_top)].groupby(["cid", "mcc"])["la"].mean().unstack()
    la = la.reindex(columns=vocab.mcc_top)
    f = pd.concat(
        [cnt_share.add_prefix("cnt_"), sum_share.add_prefix("sum_"), la.add_prefix("la_")], axis=1
    ).reindex(idx)
    share_cols = [c for c in f.columns if not c.startswith("la_")]
    f[share_cols] = f[share_cols].fillna(0)
    raw = {"mcc_cnt": cnt.reindex(idx, fill_value=0), "mcc_sum": summ.reindex(idx, fill_value=0)}
    return f, raw


def trtype_profile(ev: pd.DataFrame, idx: np.ndarray, vocab: Vocab) -> tuple[pd.DataFrame, dict]:
    cnt_share, cnt = _share(ev, "tt", vocab.tr_type)
    sum_share, _ = _share(ev.assign(a=ev["amt"].abs()), "tt", vocab.tr_type, weight="a")
    f = pd.concat([cnt_share.add_prefix("cnt_"), sum_share.add_prefix("sum_")], axis=1)
    return f.reindex(idx).fillna(0), {"tt_cnt": cnt.reindex(idx, fill_value=0)}


def time_profile(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    known = ev[~ev["mid"]]
    hour, _ = _share(known, "hour", list(range(24)))
    dow, _ = _share(ev, "dow", list(range(7)))
    out = ev[ev["out"]].assign(a=lambda x: x["amt"].abs())
    dow_sum, _ = _share(out, "dow", list(range(7)), weight="a")
    f = pd.concat(
        [hour.add_prefix("hour_"), dow.add_prefix("dow_cnt_"), dow_sum.add_prefix("dow_sum_")], axis=1
    )
    f = f.reindex(idx)
    f["midnight_share"] = ev.groupby("cid")["mid"].mean()
    ang = known["hour"].to_numpy() * 2 * np.pi / 24
    t = pd.DataFrame(
        {
            "cid": known["cid"].to_numpy(),
            "s": np.sin(ang),
            "c": np.cos(ang),
            "h": known["hour"].to_numpy(),
            "night": known["hour"].to_numpy() < 6,
            "eve": known["hour"].to_numpy() >= 18,
        }
    )
    g = t.groupby("cid")
    f["hour_sin"], f["hour_cos"] = g["s"].mean(), g["c"].mean()
    f["hour_R"] = np.sqrt(f["hour_sin"] ** 2 + f["hour_cos"] ** 2)
    f["hour_std"], f["night_share"], f["eve_share"] = g["h"].std(), g["night"].mean(), g["eve"].mean()
    hc = known.groupby(["cid", "hour"]).size().unstack(fill_value=0)
    f["hour_entropy"] = pd.Series(_entropy(hc), index=hc.index)
    ko = out[~out["mid"]]
    night = ko["hour"] < 6
    f["night_out_la"] = ko[night].groupby("cid")["la"].mean()
    f["day_out_la"] = ko[~night].groupby("cid")["la"].mean()
    return f


def terminals(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["na_share"] = (ev["term"] == "-1").groupby(ev["cid"]).mean().reindex(idx)
    tc = ev[ev["term"] != "-1"].groupby(["cid", "term"]).size()
    g = tc.groupby(level=0)
    tot = g.sum()
    f["n"] = g.size()
    f["tr_per_term"] = tot / g.size()
    f["top1_share"] = g.max() / tot
    f["top3_share"] = g.apply(lambda s: s.nlargest(3).sum()) / tot
    p = tc / g.transform("sum")
    f["entropy"] = (-(p * np.log(p))).groupby(level=0).sum()
    f["repeat_share"] = tc[tc >= 2].groupby(level=0).sum() / tot
    f["n"] = f["n"].fillna(0)
    return f


def dynamics(ev: pd.DataFrame, idx: np.ndarray, raw: dict, cfg: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["mcc_cnt_hhi"] = _hhi(raw["mcc_cnt"])
    f["mcc_sum_hhi"] = _hhi(raw["mcc_sum"])
    f["mcc_cnt_entropy"] = _entropy(raw["mcc_cnt"])
    f["mcc_sum_entropy"] = _entropy(raw["mcc_sum"])
    f["tt_cnt_entropy"] = _entropy(raw["tt_cnt"])
    f["mcc_eff_n"] = 1 / f["mcc_cnt_hhi"].replace(0, np.nan)

    def decayed(s: pd.DataFrame, col: str, name: str) -> None:
        last = s.groupby("cid")["t"].transform("max")
        for tau in cfg["decay_tau_days"]:
            w = np.exp(-(last - s["t"]) / tau)
            f[f"{name}_ewm{tau}"] = (w * s[col]).groupby(s["cid"]).sum() / w.groupby(s["cid"]).sum()

    decayed(ev[ev["out"]], "la", "out_la")
    decayed(ev[~ev["out"]], "la", "in_la")
    decayed(ev.assign(o=ev["out"].astype(float)), "o", "out_frac")

    ci = pd.Index(idx)
    r = ci.get_indexer(ev["cid"])
    o = ev["out"].to_numpy()
    day = ev["day"].to_numpy()
    cnt = np.zeros((len(ci), N_DAYS))
    spend = np.zeros((len(ci), N_DAYS))
    np.add.at(cnt, (r, day), 1)
    np.add.at(spend, (r[o], day[o]), -ev["amt"].to_numpy()[o])
    spend = np.log1p(spend)
    first = ev.groupby("cid")["day"].min().reindex(idx).to_numpy()
    last = ev.groupby("cid")["day"].max().reindex(idx).to_numpy()
    days = np.arange(N_DAYS)
    mask = (days[None, :] >= first[:, None]) & (days[None, :] <= last[:, None])
    for lag in cfg["autocorr_lags"]:
        f[f"ac_cnt_l{lag}"] = _autocorr(cnt, mask, lag)
        f[f"ac_sum_l{lag}"] = _autocorr(spend, mask, lag)
    nw = N_DAYS // 7
    weekly = cnt[:, : nw * 7].reshape(len(ci), nw, 7).sum(2)
    wmask = mask[:, : nw * 7].reshape(len(ci), nw, 7).all(2)
    f["ac_week_l1"] = _autocorr(weekly, wmask, 1)
    f["ac_week_l4"] = _autocorr(weekly, wmask, 4)

    lastd = ev.groupby("cid")["day"].transform("max")
    ntot = ev.groupby("cid").size()
    for n in cfg["windows_days"]:
        w = ev[ev["day"] > lastd - n]
        wo = w[w["out"]]
        g = w.groupby("cid")
        f[f"w{n}_cnt_share"] = g.size() / ntot
        f[f"w{n}_out_la_mean"] = wo.groupby("cid")["la"].mean()
        f[f"w{n}_in_cnt"] = (~w["out"]).groupby(w["cid"]).sum()
        f[f"w{n}_n_mcc"] = g["mcc"].nunique()
        f[f"w{n}_out_logsum"] = np.log1p(wo["amt"].abs().groupby(wo["cid"]).sum())

    mo = ev[ev["out"]].assign(m=lambda x: x["day"] // 30, a=lambda x: x["amt"].abs())
    ms = np.log1p(mo.groupby(["cid", "m"])["a"].sum()).reset_index()

    def slope(g: pd.DataFrame) -> float:
        return np.polyfit(g["m"], g["a"], 1)[0] if len(g) >= 3 else np.nan

    f["out_month_slope"] = ms.groupby("cid")[["m", "a"]].apply(slope)
    return f


def client_features(ev: pd.DataFrame, vocab: Vocab, cfg: dict) -> pd.DataFrame:
    """Все признаки без меток + сырые счётчики (``raw_*``) для признаков с метками.

    Индекс — customer_id; набор колонок не зависит от сплита.
    """
    idx = np.sort(ev["cid"].unique())
    mcc, raw = mcc_profile(ev, idx, vocab)
    tt, raw_tt = trtype_profile(ev, idx, vocab)
    raw.update(raw_tt)
    blocks = {
        "act": activity(ev, idx),
        "amt": amounts(ev, idx),
        "mcc": mcc,
        "tt": tt,
        "time": time_profile(ev, idx).reindex(idx),
        "term": terminals(ev, idx),
        "dyn": dynamics(ev, idx, raw, cfg),
        "raw_mcccnt": raw["mcc_cnt"].set_axis([str(c) for c in vocab.mcc], axis=1),
        "raw_mccsum": raw["mcc_sum"].set_axis([str(c) for c in vocab.mcc], axis=1),
        "raw_ttcnt": raw["tt_cnt"].set_axis([str(c) for c in vocab.tr_type], axis=1),
    }
    X = pd.concat([b.add_prefix(f"{k}_") for k, b in blocks.items()], axis=1)
    X.index = pd.Index(idx, name="customer_id")
    return X.astype(np.float32)


# ---------------------------------------------------------------- признаки с метками
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _te_block(counts: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, smoothing: float) -> dict:
    """Статистики target encoding категорий по клиентам ``fit_rows`` (клиентский уровень)."""
    present = (counts.to_numpy()[fit_rows] > 0).astype(float)
    n = present.sum(0)
    m = present.T @ y[fit_rows]
    return {"n": n, "m": m, "smoothing": smoothing}


def _te_apply(counts: pd.DataFrame, stats: dict, prior: float, rows: np.ndarray, name: str) -> pd.DataFrame:
    a = stats["smoothing"]
    te = (stats["m"] + a * prior) / (stats["n"] + a)
    lo = _logit(te) - _logit(np.array(prior))
    conf = stats["n"] / (stats["n"] + a)
    C = counts.to_numpy()[rows]
    present = C > 0
    k = present.sum(axis=1).clip(min=1)
    te_mat = np.where(present, te, np.nan)
    with warnings.catch_warnings():
        # клиенты с одной категорией: std не определён -> NaN (бустинг обрабатывает пропуски)
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.DataFrame(
            {
                f"te_{name}_wmean": (C * te).sum(axis=1) / C.sum(axis=1).clip(min=1),
                f"te_{name}_mean": np.nanmean(te_mat, 1),
                f"te_{name}_max": np.nanmax(te_mat, 1),
                f"te_{name}_min": np.nanmin(te_mat, 1),
                f"te_{name}_std": np.nanstd(te_mat, 1, ddof=1),
                f"te_{name}_lo_sum": (present * lo).sum(axis=1),
                f"te_{name}_lo_mean": (present * lo).sum(axis=1) / k,
                f"te_{name}_wlo_sum": (np.log1p(C) * lo).sum(axis=1),
                f"te_{name}_conf_lo_sum": (present * lo * conf).sum(axis=1),
            },
            index=counts.index[rows],
        )


def _tfidf(M: np.ndarray, idf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    X = np.log1p(M)
    if idf is None:
        idf = np.log((1 + M.shape[0]) / (1 + (M > 0).sum(0))) + 1
    X = X * idf
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12), idf


class LabelFeatures:
    """Признаки с метками: target encoding MCC/типов, логрег по TF-IDF профилей, логрег по эмбеддингам.

    ``fit_transform(X, y)`` возвращает out-of-fold значения для обучающих клиентов (внутренние фолды),
    ``transform(X)`` — значения для новых клиентов по статистикам, посчитанным на всех обучающих.
    """

    def __init__(self, cfg: dict, seed: int = 0, n_inner: int = 5):
        self.cfg, self.seed, self.n_inner = cfg, seed, n_inner

    @staticmethod
    def _raw(X: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return X.loc[:, X.columns.str.startswith(prefix)]

    def _fit_parts(self, X: pd.DataFrame, y: np.ndarray, rows: np.ndarray) -> dict:
        te_cfg = self.cfg["target_encoding"]["smoothing"]
        parts = {
            "prior": float(y[rows].mean()),
            "te_mcc": _te_block(self._raw(X, "raw_mcccnt_"), y, rows, te_cfg["mcc"]),
            "te_tt": _te_block(self._raw(X, "raw_ttcnt_"), y, rows, te_cfg["tr_type"]),
        }
        mats, idfs = [], []
        for M in self._tfidf_inputs(X):
            A, idf = _tfidf(M[rows])
            mats.append(A)
            idfs.append(idf)
        parts["tfidf"] = (
            LogisticRegression(C=self.cfg["tfidf_logreg_C"], max_iter=2000).fit(np.hstack(mats), y[rows]),
            idfs,
        )
        emb = self._raw(X, "emb_")
        if emb.shape[1]:
            parts["emb"] = make_pipeline(
                StandardScaler(), LogisticRegression(C=self.cfg["embedding_logreg_C"], max_iter=3000)
            ).fit(emb.to_numpy()[rows], y[rows])
        return parts

    def _tfidf_inputs(self, X: pd.DataFrame) -> list[np.ndarray]:
        return [
            self._raw(X, "raw_mcccnt_").to_numpy(float),
            self._raw(X, "raw_ttcnt_").to_numpy(float),
            np.log1p(self._raw(X, "raw_mccsum_").to_numpy(float)),
        ]

    def _apply(self, parts: dict, X: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        out = [
            _te_apply(self._raw(X, "raw_mcccnt_"), parts["te_mcc"], parts["prior"], rows, "mcc"),
            _te_apply(self._raw(X, "raw_ttcnt_"), parts["te_tt"], parts["prior"], rows, "tt"),
        ]
        clf, idfs = parts["tfidf"]
        B = np.hstack([_tfidf(M[rows], idf)[0] for M, idf in zip(self._tfidf_inputs(X), idfs)])
        lin = pd.DataFrame({"lr_tfidf": clf.decision_function(B)}, index=X.index[rows])
        if "emb" in parts:
            lin["lr_emb"] = parts["emb"].decision_function(self._raw(X, "emb_").to_numpy()[rows])
        return pd.concat(out + [lin], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        yv = y.reindex(X.index).to_numpy()
        folds = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed).split(X, yv)
        parts = [self._apply(self._fit_parts(X, yv, a), X, b) for a, b in folds]
        self.parts_ = self._fit_parts(X, yv, np.arange(len(X)))
        return pd.concat(parts).reindex(X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._apply(self.parts_, X, np.arange(len(X)))


def feature_groups(X: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """Группа признаков -> число колонок -> примеры колонок (по префиксу имени)."""
    prefix = X.columns.str.split("_").str[0]
    rows = []
    for p, name in GROUP_PREFIXES.items():
        cols = X.columns[prefix == p]
        if len(cols):
            step = max(len(cols) // n_examples, 1)
            rows.append(
                {
                    "группа": name,
                    "префикс": f"{p}_",
                    "признаков": len(cols),
                    "примеры": ", ".join(cols[::step][:n_examples]),
                }
            )
    return pd.DataFrame(rows)


def model_matrix(X: pd.DataFrame, label_feats: pd.DataFrame) -> pd.DataFrame:
    """Итоговая матрица для бустинга: признаки без меток + признаки с метками (без raw_/emb_)."""
    keep = ~X.columns.str.startswith(("raw_", "emb_"))
    return pd.concat([X.loc[:, keep], label_feats.reindex(X.index)], axis=1)


# ---------------------------------------------------------------- профиль клиента (аналитика, без меток)
FINANCIAL_MCC_WORDS = ("наличност", "Денежные переводы", "Финансовые институты")


def financial_mcc(df: pd.DataFrame) -> set[int]:
    """MCC финансовых операций (снятие наличных, переводы) — это не покупки и не предпочтения."""
    desc = df.drop_duplicates("mcc_code").set_index("mcc_code")["mcc_code_desc"].astype(str)
    return {int(m) for m, d in desc.items() if any(w in d for w in FINANCIAL_MCC_WORDS)}


def spend_shares(df: pd.DataFrame, exclude: set[int] | None = None) -> pd.DataFrame:
    """Доли трат клиента по MCC (клиенты × MCC), только расходы; ``exclude`` — MCC, которые не учитываются."""
    out = df[df["amount"] < 0]
    if exclude:
        out = out[~out["mcc_code"].isin(exclude)]
    m = out.assign(a=out["amount"].abs()).pivot_table(
        index="customer_id", columns="mcc_code", values="a", aggfunc="sum", fill_value=0.0
    )
    return m.div(m.sum(axis=1), axis=0)


def _hours_dow(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    t = df["tr_datetime"]
    known = t != t.dt.normalize()
    hours = t[known].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0)
    dow = ((t - T0).dt.days % 7).value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    return hours, dow


def rfm(df: pd.DataFrame) -> pd.DataFrame:
    """RFM клиента: recency — дней с последней операции до конца периода; frequency — операций и активных
    дней в месяц; monetary — сумма расходов за период и средний чек расхода."""
    day = (df["tr_datetime"] - T0).dt.days
    g = day.groupby(df["customer_id"])
    span_months = ((g.max() - g.min() + 1) / 30.4).clip(lower=1.0)
    out = df.loc[df["amount"] < 0, "amount"].abs().groupby(df["customer_id"])
    return pd.DataFrame(
        {
            "recency, дней": (N_DAYS - 1) - g.max(),
            "операций в месяц": g.size() / span_months,
            "активных дней в месяц": g.nunique() / span_months,
            "сумма расходов": out.sum(),
            "средний чек": out.mean(),
        }
    ).fillna(0.0)


@dataclass
class CardReference:
    """Средний клиент train — точка отсчёта карточки: доли покупок по MCC (без снятия наличных и переводов),
    часы, дни недели, порог крупного расхода, RFM."""

    mcc_mean_share: pd.Series
    mcc_clients: pd.Series
    hours: pd.Series
    dow: pd.Series
    large_threshold: float
    desc: dict
    rfm: pd.DataFrame
    financial: set
    min_clients: int = 200

    @classmethod
    def from_train(cls, train: pd.DataFrame, min_clients: int = 200) -> CardReference:
        S = spend_shares(train, exclude=financial_mcc(train))
        hours, dow = _hours_dow(train)
        desc = train.drop_duplicates("mcc_code").set_index("mcc_code")["mcc_code_desc"].astype(str).to_dict()
        large = float(train.loc[train["amount"] < 0, "amount"].abs().quantile(0.9))
        return cls(
            S.mean(), (S > 0).sum(), hours, dow, large, desc, rfm(train), financial_mcc(train), min_clients
        )


def client_card(df: pd.DataFrame, cid: int, ref: CardReference, top: int = 5) -> dict:
    """Карточка клиента: RFM; предпочтения — доли покупок по MCC (без снятия наличных и переводов) и lift
    к среднему клиенту; привычки; уровень трат; разнообразие покупок. Метки не используются."""
    d = df[df["customer_id"] == cid]
    share = spend_shares(d, exclude=ref.financial).iloc[0]
    share = share[share > 0].sort_values(ascending=False)
    supported = share.index[ref.mcc_clients.reindex(share.index, fill_value=0) >= ref.min_clients]
    pref = pd.DataFrame(
        {
            "категория": [ref.desc.get(m, str(m)) for m in share.index],
            "доля покупок": share.to_numpy(),
            "lift": (share / ref.mcc_mean_share.reindex(share.index)).to_numpy(),
        },
        index=share.index,
    )
    pref.loc[~pref.index.isin(supported), "lift"] = np.nan
    hours, dow = _hours_dow(d)
    days = np.sort((d["tr_datetime"] - T0).dt.days.unique())
    gaps = np.diff(days)
    out = d.loc[d["amount"] < 0, "amount"].abs()
    span_months = max((days[-1] - days[0] + 1) / 30.4, 1.0)
    r = rfm(d).iloc[0]
    pct = {k: float((ref.rfm[k] <= v).mean()) for k, v in r.items()}
    return {
        "id": cid,
        "операций": len(d),
        "RFM": pd.DataFrame({"значение": r.round(1), "перцентиль среди train": pd.Series(pct).round(2)}),
        "предпочтения": pref.head(top),
        "выше среднего": pref.dropna(subset=["lift"])
        .query("`доля покупок` >= 0.02")
        .sort_values("lift", ascending=False)
        .head(3),
        "часы": hours,
        "дни недели": dow,
        "регулярность": {
            "активных дней": len(days),
            "доля активных дней": round(len(days) / (days[-1] - days[0] + 1), 3),
            "интервал между активными днями, медиана": float(np.median(gaps)) if len(gaps) else float("nan"),
            "вариация интервалов (CV)": round(float(gaps.std() / gaps.mean()), 2)
            if len(gaps) > 1
            else float("nan"),
        },
        "уровень трат": {
            "средний чек": round(float(out.mean()), 0),
            "расходы в месяц": round(float(out.sum() / span_months), 0),
            "доля крупных (> p90 train)": round(float((out > ref.large_threshold).mean()), 3),
            "доля покупок в расходах": round(
                float(out[~d.loc[out.index, "mcc_code"].isin(ref.financial)].sum() / max(out.sum(), 1e-9)), 3
            ),
        },
        "разнообразие": {
            "категорий": int((share > 0).sum()),
            "HHI покупок": round(float((share**2).sum()), 3),
        },
    }


def segment_clients(
    train: pd.DataFrame, ref: CardReference, k_range: range = range(4, 9), seed: int = 0
) -> tuple[pd.Series, pd.Series]:
    """KMeans по долям трат на покупки по MCC (без снятия наличных и переводов; Хеллингер: sqrt долей;
    MCC с поддержкой, остальные — «прочие»); k — по силуэту. Возвращает сегменты и силуэт для каждого k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    S = spend_shares(train, exclude=financial_mcc(train))
    keep = S.columns[(S > 0).sum() >= ref.min_clients]
    A = S.reindex(columns=keep, fill_value=0.0)
    A["other"] = 1 - A.sum(axis=1)
    H = np.sqrt(A.clip(lower=0).to_numpy())
    sil, fits = {}, {}
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=seed).fit(H)
        sil[k] = silhouette_score(H, km.labels_, sample_size=3000, random_state=seed)
        fits[k] = km.labels_
    best = max(sil, key=sil.get)
    return pd.Series(fits[best], index=A.index, name="segment"), pd.Series(sil, name="силуэт")


def segment_portraits(train: pd.DataFrame, seg: pd.Series, ref: CardReference, y: pd.Series) -> pd.DataFrame:
    """Портреты сегментов: размер, отличительные категории покупок (lift к среднему), привычки, чек,
    доля мужчин (метки train)."""
    S = spend_shares(train, exclude=financial_mcc(train)).reindex(seg.index, fill_value=0.0)
    supported = S.columns[(S > 0).sum() >= ref.min_clients]
    base = S[supported].mean()
    t = train["tr_datetime"]
    known = t != t.dt.normalize()
    h = t[known].dt.hour
    night = (h < 6).groupby(train.loc[known, "customer_id"]).mean()
    evening = (h >= 18).groupby(train.loc[known, "customer_id"]).mean()
    check = train[train["amount"] < 0].groupby("customer_id")["amount"].apply(lambda a: a.abs().mean())

    def short(m: int) -> str:
        return ref.desc.get(m, str(m)).split(" — ")[0].split(",")[0].strip()

    rows = []
    for k, ids in seg.groupby(seg).groups.items():
        mean = S.loc[ids, supported].mean()
        lift = (mean / base)[mean >= 0.02].sort_values(ascending=False)
        rows.append(
            {
                "сегмент": k,
                "название": " + ".join(list(dict.fromkeys(short(m).lower() for m in lift.index[:3]))[:2]),
                "клиентов": len(ids),
                "доля": round(len(ids) / len(seg), 3),
                "топ-5 категорий покупок (доля, lift)": "; ".join(
                    f"{short(m)[:30]} {mean[m]:.0%} ×{lift[m]:.1f}" for m in lift.index[:5]
                ),
                "ночью": round(float(night.reindex(ids).median()), 3),
                "вечером": round(float(evening.reindex(ids).median()), 3),
                "средний чек, медиана": round(float(check.reindex(ids).median()), 0),
                "доля мужчин": round(float(y.reindex(ids).mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("клиентов", ascending=False).set_index("сегмент")


def top_purchases(df: pd.DataFrame, ids, ref: CardReference, k: int = 5) -> pd.Series:
    """Топ-k категорий покупок (без снятия наличных и переводов) с долей трат — для сравнения клиентов."""
    d = df[df["customer_id"].isin(list(ids))]
    S = spend_shares(d, exclude=financial_mcc(d))

    def fmt(row: pd.Series) -> str:
        row = row[row > 0].sort_values(ascending=False).head(k)
        return "; ".join(f"{ref.desc.get(m, str(m)).split(' — ')[0][:28]} {v:.0%}" for m, v in row.items())

    return S.apply(fmt, axis=1).reindex(list(ids))


def nearest_clients(E: pd.DataFrame, cid: int, k: int = 3) -> pd.Series:
    """Ближайшие клиенты по косинусной близости эмбеддингов (index — клиент, значение — близость)."""
    A = E.to_numpy(dtype=float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    sim = pd.Series(A @ A[E.index.get_loc(cid)], index=E.index).drop(cid)
    return sim.sort_values(ascending=False).head(k)


def neighbor_agreement(E: pd.DataFrame, groups: pd.Series, k: int = 5, seed: int = 0) -> pd.Series:
    """Доля k ближайших соседей (по эмбеддингу) из той же группы клиента — против случайных клиентов;
    плюс косинусная близость соседей и случайных пар."""
    A = E.to_numpy(dtype=float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    sim = A @ A.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    g = groups.reindex(E.index).to_numpy()
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, len(A), size=(len(A), k))
    pairs = rng.integers(0, len(A), size=(20000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pd.Series(
        {
            "та же группа у соседей": float((g[nn] == g[:, None]).mean()),
            "та же группа у случайных": float((g[rand] == g[:, None]).mean()),
            "близость соседей, медиана": float(np.median(np.take_along_axis(sim, nn, axis=1))),
            "близость случайных пар, медиана": float(np.median(sim[pairs[:, 0], pairs[:, 1]])),
        }
    )


def shap_summary(contrib: pd.DataFrame, M: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Глобальная важность по pred_contrib: средний |вклад| и направление (знак корреляции Спирмена
    между значением признака и его вкладом: > 0 — большее значение сдвигает к «мужчине»)."""
    C = contrib.drop(columns="bias")
    imp = C.abs().mean().sort_values(ascending=False).head(top)
    direction = {
        f: np.sign(pd.Series(M[f].to_numpy()).corr(pd.Series(C[f].to_numpy()), method="spearman"))
        for f in imp.index
    }
    return pd.DataFrame({"mean_abs": imp, "direction": pd.Series(direction)})
