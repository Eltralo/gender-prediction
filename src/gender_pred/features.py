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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import N_DAYS, Vocab

# префикс колонки → название группы признаков (для таблиц и подписей графиков)
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
    # группируем операции по паре (клиент, категория): число операций или сумма веса в каждой паре
    g = ev.groupby(["cid", key])
    # unstack → таблица клиент × категория; пары, которых у клиента нет, заполняем нулём
    m = (g.size() if weight is None else g[weight].sum()).unstack(fill_value=0)
    # reindex по фиксированному словарю из train: на любом сплите одни и те же колонки в одном порядке
    # (категорий, которых нет в данных сплита, появятся нулевые колонки; новых категорий не будет)
    m = m.reindex(columns=cols, fill_value=0).astype(float)
    # переводим счётчики в доли: делим каждую строку на её сумму (div по axis=0 — построчно),
    # так клиенты с 10 и с 10 000 операций сравнимы; клиент без операций → нули, а не деление на 0
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0), m


def _entropy(m: pd.DataFrame) -> np.ndarray:
    # доли по строкам: распределение операций / трат клиента по категориям
    p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    # энтропия −Σ p·log p: 0 — всё в одной категории, больше — траты разнообразнее
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0).sum(axis=1)


def _hhi(m: pd.DataFrame) -> np.ndarray:
    p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    # индекс Херфиндаля–Хиршмана Σ p²: 1 — все траты в одной категории, ближе к 0 — по многим
    return (p**2).sum(axis=1)


def _autocorr(M: np.ndarray, mask: np.ndarray, lag: int) -> np.ndarray:
    # автокорреляция ряда с самим собой, сдвинутым на lag (дней или недель): насколько поведение
    # повторяется через lag; M — клиенты × дни, mask — дни внутри истории клиента
    a, b = M[:, :-lag], M[:, lag:]
    # берём только пары дней, где оба дня внутри истории клиента
    v = mask[:, :-lag] & mask[:, lag:]
    n = v.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # средние по валидным дням и отклонения от них
        ma = np.where(v, a, 0).sum(axis=1) / n
        mb = np.where(v, b, 0).sum(axis=1) / n
        da, db = np.where(v, a - ma[:, None], 0), np.where(v, b - mb[:, None], 0)
        # корреляция Пирсона между рядом и его сдвигом, построчно для каждого клиента
        r = (da * db).sum(axis=1) / np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    # меньше 10 пар — оценка ненадёжна, оставляем пропуск (бустинг умеет с ним работать)
    r[n < 10] = np.nan
    return r


# ---------------------------------------------------------------- группы без меток
def activity(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    g = ev.groupby("cid")
    # базовые счётчики клиента: операции, дни с операциями, первый/последний день, разнообразие
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
    # длина истории в днях и доля дней с операциями: регулярность пользования картой
    f["span"] = f["last_day"] - f["first_day"] + 1
    f["days_share"] = f["n_days"] / f["span"]
    # операций в активный день и логарифм числа операций (активность различается на порядки)
    f["tr_per_day"] = f["n_tr"] / f["n_days"]
    f["log_n_tr"] = np.log1p(f["n_tr"])
    # перерывы между днями с операциями: уникальные дни клиента → diff внутри клиента
    ud = ev[["cid", "day"]].drop_duplicates()
    gap = ud["day"].groupby(ud["cid"]).diff().groupby(ud["cid"])
    f["gap_mean"], f["gap_std"], f["gap_max"], f["gap_med"] = gap.mean(), gap.std(), gap.max(), gap.median()
    # число операций по 30-дневным «месяцам» → таблица клиент × месяц
    mon = ev.assign(m=ev["day"] // 30).groupby(["cid", "m"]).size().unstack(fill_value=0)
    cols = mon.columns.to_numpy()
    # учитываем только месяцы внутри истории клиента (до первой и после последней операции — не нули)
    lo = (f["first_day"] // 30).reindex(mon.index).to_numpy()[:, None]
    hi = (f["last_day"] // 30).reindex(mon.index).to_numpy()[:, None]
    mask = (cols[None, :] >= lo) & (cols[None, :] <= hi)
    mv = np.where(mask, mon.to_numpy(), np.nan)
    # коэффициент вариации = std / mean по месяцам: неравномерность активности (0 — ровно каждый месяц)
    f["month_cv"] = pd.Series(np.nanstd(mv, 1) / (np.nanmean(mv, 1) + 1e-9), index=mon.index)
    # доля месяцев истории, в которых были операции
    f["month_active_share"] = pd.Series(np.nansum(mv > 0, 1) / mask.sum(axis=1), index=mon.index)
    # reindex по списку клиентов сплита: строки в фиксированном порядке
    return f.reindex(idx)


def amounts(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    # расходы и поступления описываем отдельно: у них разные распределения
    for name, m in (("out", ev["out"]), ("in", ~ev["out"])):
        s = ev[m]
        g = s.groupby("cid")
        a = s["amt"].abs().groupby(s["cid"])
        # число операций и общая сумма (в логарифме из-за тяжёлых хвостов)
        out[f"{name}_cnt"] = g.size()
        out[f"{name}_logsum"] = np.log1p(a.sum())
        # статистики la = log(1 + |amount|): средний чек (геометрический), разброс, крайние, асимметрия
        for stat in ("mean", "std", "min", "max", "skew"):
            out[f"{name}_la_{stat}"] = getattr(g["la"], stat)()
        # квантили la по клиенту → таблица клиент × квантиль: типичные маленькие и крупные операции
        q = g["la"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
        for c in q.columns:
            out[f"{name}_la_q{int(round(c * 100))}"] = q[c]
        # доля самой крупной операции в общей сумме: «одна большая покупка» или много мелких
        out[f"{name}_max_share"] = a.max() / a.sum()
        # доля уникальных сумм: повторяющиеся одинаковые суммы (подписки, переводы) дают малое значение
        out[f"{name}_amt_nuniq_share"] = s["amt"].groupby(s["cid"]).nunique() / g.size()
    f = pd.DataFrame(out).reindex(idx)
    # у клиента может не быть поступлений или расходов — число таких операций 0, а не пропуск
    f[["out_cnt", "in_cnt"]] = f[["out_cnt", "in_cnt"]].fillna(0)
    # доля поступлений среди всех операций
    f["in_cnt_share"] = f["in_cnt"] / (f["in_cnt"] + f["out_cnt"])
    # баланс в логарифмах: log(поступления) − log(расходы)
    f["log_balance"] = f["in_logsum"].fillna(0) - f["out_logsum"].fillna(0)
    # средняя la по всем операциям вместе
    f["la_mean_all"] = ev.groupby("cid")["la"].mean().reindex(idx)
    return f


def mcc_profile(ev: pd.DataFrame, idx: np.ndarray, vocab: Vocab) -> tuple[pd.DataFrame, dict]:
    # доля операций клиента в каждой MCC (клиент × 184 категории); cnt — сырые счётчики
    cnt_share, cnt = _share(ev, "mcc", vocab.mcc)
    # только расходы, вес — модуль суммы
    out = ev[ev["out"]].assign(a=lambda x: x["amt"].abs())
    # доля расходов клиента в каждой MCC: «на что тратит»; summ — сырые суммы по категориям
    sum_share, summ = _share(out, "mcc", vocab.mcc, weight="a")
    # средний чек (среднее la) в категории — только для 40 самых частых MCC: у редких категорий
    # у большинства клиентов нет операций (пропуск) или одна-две, и среднее было бы шумом
    la = out[out["mcc"].isin(vocab.mcc_top)].groupby(["cid", "mcc"])["la"].mean().unstack()
    la = la.reindex(columns=vocab.mcc_top)
    # склеиваем три блока; reindex по клиентам сплита — фиксированный порядок строк
    f = pd.concat(
        [cnt_share.add_prefix("cnt_"), sum_share.add_prefix("sum_"), la.add_prefix("la_")], axis=1
    ).reindex(idx)
    # доли: нет операций в категории = 0; средний чек при отсутствии операций остаётся пропуском
    share_cols = [c for c in f.columns if not c.startswith("la_")]
    f[share_cols] = f[share_cols].fillna(0)
    # сырые счётчики и суммы: вход для динамики (HHI, энтропия) и признаков с метками (TE, TF-IDF)
    raw = {"mcc_cnt": cnt.reindex(idx, fill_value=0), "mcc_sum": summ.reindex(idx, fill_value=0)}
    return f, raw


def trtype_profile(ev: pd.DataFrame, idx: np.ndarray, vocab: Vocab) -> tuple[pd.DataFrame, dict]:
    # доля операций каждого из 74 типов (покупка в POS, снятие наличных, перевод…)
    cnt_share, cnt = _share(ev, "tt", vocab.tr_type)
    # доля суммы (по модулю, расходы и поступления вместе) по типам операций
    sum_share, _ = _share(ev.assign(a=ev["amt"].abs()), "tt", vocab.tr_type, weight="a")
    f = pd.concat([cnt_share.add_prefix("cnt_"), sum_share.add_prefix("sum_")], axis=1)
    # сырые счётчики типов — вход для target encoding и TF-IDF
    return f.reindex(idx).fillna(0), {"tt_cnt": cnt.reindex(idx, fill_value=0)}


def time_profile(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    # для часового профиля берём только операции с известным временем (без 00:00:00)
    known = ev[~ev["mid"]]
    # доля операций по 24 часам; по 7 дням недели (day % 7) — все операции, день известен всегда
    hour, _ = _share(known, "hour", list(range(24)))
    dow, _ = _share(ev, "dow", list(range(7)))
    # доля расходов по дням недели
    out = ev[ev["out"]].assign(a=lambda x: x["amt"].abs())
    dow_sum, _ = _share(out, "dow", list(range(7)), weight="a")
    f = pd.concat(
        [hour.add_prefix("hour_"), dow.add_prefix("dow_cnt_"), dow_sum.add_prefix("dow_sum_")], axis=1
    )
    f = f.reindex(idx)
    # доля операций без времени — отдельный признак (зависит от того, какими каналами платит клиент)
    f["midnight_share"] = ev.groupby("cid")["mid"].mean()
    # час как угол на круге: 23 ч и 0 ч рядом; среднее sin/cos — «средний час» клиента
    ang = known["hour"].to_numpy() * 2 * np.pi / 24
    t = pd.DataFrame(
        {
            "cid": known["cid"].to_numpy(),
            "s": np.sin(ang),
            "c": np.cos(ang),
            "h": known["hour"].to_numpy(),
            # ночь — 0–6 ч, вечер — после 18 ч
            "night": known["hour"].to_numpy() < 6,
            "eve": known["hour"].to_numpy() >= 18,
        }
    )
    g = t.groupby("cid")
    f["hour_sin"], f["hour_cos"] = g["s"].mean(), g["c"].mean()
    # длина среднего вектора: близко к 1 — операции в одно и то же время, к 0 — в разное
    f["hour_R"] = np.sqrt(f["hour_sin"] ** 2 + f["hour_cos"] ** 2)
    f["hour_std"], f["night_share"], f["eve_share"] = g["h"].std(), g["night"].mean(), g["eve"].mean()
    # энтропия распределения по часам: насколько разнообразно время операций
    hc = known.groupby(["cid", "hour"]).size().unstack(fill_value=0)
    f["hour_entropy"] = pd.Series(_entropy(hc), index=hc.index)
    # средний расход ночью и днём (только операции с известным временем)
    ko = out[~out["mid"]]
    night = ko["hour"] < 6
    f["night_out_la"] = ko[night].groupby("cid")["la"].mean()
    f["day_out_la"] = ko[~night].groupby("cid")["la"].mean()
    return f


def terminals(ev: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    # доля операций с неизвестным терминалом (term = "-1")
    f["na_share"] = (ev["term"] == "-1").groupby(ev["cid"]).mean().reindex(idx)
    # число операций в каждой паре (клиент, терминал), только известные терминалы
    tc = ev[ev["term"] != "-1"].groupby(["cid", "term"]).size()
    g = tc.groupby(level=0)
    tot = g.sum()
    # сколько разных терминалов и сколько операций в среднем на терминал
    f["n"] = g.size()
    f["tr_per_term"] = tot / g.size()
    # доля операций в самом частом терминале и в трёх самых частых
    f["top1_share"] = g.max() / tot
    f["top3_share"] = g.apply(lambda s: s.nlargest(3).sum()) / tot
    # transform("sum") возвращает сумму клиента в каждой строке → доли терминалов внутри клиента
    p = tc / g.transform("sum")
    f["entropy"] = (-(p * np.log(p))).groupby(level=0).sum()
    # доля операций в терминалах, где клиент платил больше одного раза («свои» места)
    f["repeat_share"] = tc[tc >= 2].groupby(level=0).sum() / tot
    # клиент без известных терминалов — 0 терминалов, а не пропуск
    f["n"] = f["n"].fillna(0)
    return f


def dynamics(ev: pd.DataFrame, idx: np.ndarray, raw: dict, cfg: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    # концентрация операций и трат по MCC (HHI: 1 — всё в одной категории) и энтропия
    f["mcc_cnt_hhi"] = _hhi(raw["mcc_cnt"])
    f["mcc_sum_hhi"] = _hhi(raw["mcc_sum"])
    f["mcc_cnt_entropy"] = _entropy(raw["mcc_cnt"])
    f["mcc_sum_entropy"] = _entropy(raw["mcc_sum"])
    f["tt_cnt_entropy"] = _entropy(raw["tt_cnt"])
    # эффективное число категорий = 1 / HHI: «сколько категорий клиент реально использует»
    # (при равных долях в k категориях ровно k)
    f["mcc_eff_n"] = 1 / f["mcc_cnt_hhi"].replace(0, np.nan)

    def decayed(s: pd.DataFrame, col: str, name: str) -> None:
        # затухающее (экспоненциально взвешенное) среднее: операция весит exp(−давность / tau),
        # давность — дней до последней операции клиента; недавние операции важнее старых
        last = s.groupby("cid")["t"].transform("max")
        for tau in cfg["decay_tau_days"]:
            # tau = 7 / 30 / 90 дней — за это время вес операции падает в e раз
            w = np.exp(-(last - s["t"]) / tau)
            f[f"{name}_ewm{tau}"] = (w * s[col]).groupby(s["cid"]).sum() / w.groupby(s["cid"]).sum()

    # средний расход и средняя сумма поступлений «в последнее время», доля расходов среди операций
    decayed(ev[ev["out"]], "la", "out_la")
    decayed(ev[~ev["out"]], "la", "in_la")
    decayed(ev.assign(o=ev["out"].astype(float)), "o", "out_frac")

    # дневная сетка клиенты × 457 дней: число операций и лог-сумма расходов в каждый день
    ci = pd.Index(idx)
    r = ci.get_indexer(ev["cid"])
    o = ev["out"].to_numpy()
    day = ev["day"].to_numpy()
    cnt = np.zeros((len(ci), N_DAYS))
    spend = np.zeros((len(ci), N_DAYS))
    # np.add.at корректно накапливает несколько операций в одной клетке (клиент, день)
    np.add.at(cnt, (r, day), 1)
    np.add.at(spend, (r[o], day[o]), -ev["amt"].to_numpy()[o])
    spend = np.log1p(spend)
    # маска «день внутри истории клиента»: до первой и после последней операции — не нули, а «нет данных»
    first = ev.groupby("cid")["day"].min().reindex(idx).to_numpy()
    last = ev.groupby("cid")["day"].max().reindex(idx).to_numpy()
    days = np.arange(N_DAYS)
    mask = (days[None, :] >= first[:, None]) & (days[None, :] <= last[:, None])
    # автокорреляция по дням с лагом 1, 7, 14, 30: повторяется ли активность день ко дню, неделя к неделе
    for lag in cfg["autocorr_lags"]:
        f[f"ac_cnt_l{lag}"] = _autocorr(cnt, mask, lag)
        f[f"ac_sum_l{lag}"] = _autocorr(spend, mask, lag)
    # то же по неделям: дневная сетка → клиенты × недели (сумма по 7 дням)
    nw = N_DAYS // 7
    weekly = cnt[:, : nw * 7].reshape(len(ci), nw, 7).sum(2)
    # неделя считается, только если все её дни внутри истории клиента
    wmask = mask[:, : nw * 7].reshape(len(ci), nw, 7).all(2)
    f["ac_week_l1"] = _autocorr(weekly, wmask, 1)
    f["ac_week_l4"] = _autocorr(weekly, wmask, 4)

    # окна «последние n дней истории клиента» (n = 30 / 90 / 180): поведение в конце периода
    # transform("max") кладёт последний день клиента в каждую его строку
    lastd = ev.groupby("cid")["day"].transform("max")
    ntot = ev.groupby("cid").size()
    for n in cfg["windows_days"]:
        w = ev[ev["day"] > lastd - n]
        wo = w[w["out"]]
        g = w.groupby("cid")
        # доля операций, пришедшаяся на окно; средний расход, число поступлений, число категорий, сумма
        f[f"w{n}_cnt_share"] = g.size() / ntot
        f[f"w{n}_out_la_mean"] = wo.groupby("cid")["la"].mean()
        f[f"w{n}_in_cnt"] = (~w["out"]).groupby(w["cid"]).sum()
        f[f"w{n}_n_mcc"] = g["mcc"].nunique()
        f[f"w{n}_out_logsum"] = np.log1p(wo["amt"].abs().groupby(wo["cid"]).sum())

    # лог-сумма расходов по 30-дневным месяцам → длинная таблица (клиент, месяц, сумма)
    mo = ev[ev["out"]].assign(m=lambda x: x["day"] // 30, a=lambda x: x["amt"].abs())
    ms = np.log1p(mo.groupby(["cid", "m"])["a"].sum()).reset_index()

    def slope(g: pd.DataFrame) -> float:
        # наклон прямой «лог-сумма от номера месяца»: > 0 — расходы растут, < 0 — падают;
        # меньше 3 месяцев — наклон не оцениваем
        return np.polyfit(g["m"], g["a"], 1)[0] if len(g) >= 3 else np.nan

    f["out_month_slope"] = ms.groupby("cid")[["m", "a"]].apply(slope)
    return f


def client_features(ev: pd.DataFrame, vocab: Vocab, cfg: dict) -> pd.DataFrame:
    """Все признаки без меток + сырые счётчики (``raw_*``) для признаков с метками.

    Индекс — customer_id; набор колонок не зависит от сплита.
    """
    # все клиенты сплита в фиксированном (отсортированном) порядке — индекс итоговой таблицы
    idx = np.sort(ev["cid"].unique())
    # MCC- и tr_type-профили отдают ещё сырые счётчики, нужные динамике и признакам с метками
    mcc, raw = mcc_profile(ev, idx, vocab)
    tt, raw_tt = trtype_profile(ev, idx, vocab)
    raw.update(raw_tt)
    # группы признаков; ключ станет префиксом колонок (act_, amt_, mcc_ …)
    blocks = {
        "act": activity(ev, idx),
        "amt": amounts(ev, idx),
        "mcc": mcc,
        "tt": tt,
        "time": time_profile(ev, idx).reindex(idx),
        "term": terminals(ev, idx),
        "dyn": dynamics(ev, idx, raw, cfg),
        # raw_* — сырые счётчики и суммы по каждой категории; в бустинг напрямую не идут,
        # это вход для признаков с метками (target encoding, TF-IDF), считаемых внутри обучения
        "raw_mcccnt": raw["mcc_cnt"].set_axis([str(c) for c in vocab.mcc], axis=1),
        "raw_mccsum": raw["mcc_sum"].set_axis([str(c) for c in vocab.mcc], axis=1),
        "raw_ttcnt": raw["tt_cnt"].set_axis([str(c) for c in vocab.tr_type], axis=1),
    }
    X = pd.concat([b.add_prefix(f"{k}_") for k, b in blocks.items()], axis=1)
    X.index = pd.Index(idx, name="customer_id")
    # float32 — вдвое меньше памяти, точности для признаков достаточно
    return X.astype(np.float32)


# ---------------------------------------------------------------- признаки с метками
def _logit(p: np.ndarray) -> np.ndarray:
    # логит log(p / (1 − p)); обрезаем p, чтобы не получить ±∞
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _te_block(counts: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, smoothing: float) -> dict:
    """Статистики target encoding категорий по клиентам ``fit_rows`` (клиентский уровень)."""
    # считаем по клиентам, а не по операциям: категория «есть у клиента» (1/0), иначе
    # несколько очень активных клиентов определяли бы долю мужчин в категории
    present = (counts.to_numpy()[fit_rows] > 0).astype(float)
    # n — сколько обучающих клиентов пользуются категорией, m — сколько из них мужчин
    n = present.sum(0)
    m = present.T @ y[fit_rows]
    # статистики только по fit_rows: метки остальных клиентов сюда не попадают
    return {"n": n, "m": m, "smoothing": smoothing}


def _te_apply(counts: pd.DataFrame, stats: dict, prior: float, rows: np.ndarray, name: str) -> pd.DataFrame:
    a = stats["smoothing"]
    # сглаженная доля мужчин в категории: (m + a·prior) / (n + a), a = 20 «виртуальных клиентов»
    # со средней долей; у редкой категории (малое n) оценка тянется к средней, у частой — к своей
    te = (stats["m"] + a * prior) / (stats["n"] + a)
    # та же оценка как отклонение от средней доли в шкале лог-шансов: > 0 — «мужская» категория
    lo = _logit(te) - _logit(np.array(prior))
    # надёжность категории: n / (n + a), от 0 (редкая) до 1 (частая)
    conf = stats["n"] / (stats["n"] + a)
    # счётчики операций клиентов, для которых считаем признаки → матрица клиент × категория
    C = counts.to_numpy()[rows]
    present = C > 0
    k = present.sum(axis=1).clip(min=1)
    # значения te только по категориям, которыми клиент пользуется (остальные — NaN)
    te_mat = np.where(present, te, np.nan)
    with warnings.catch_warnings():
        # клиенты с одной категорией: std не определён -> NaN (бустинг обрабатывает пропуски)
        warnings.simplefilter("ignore", RuntimeWarning)
        # 9 сводок по категориям клиента
        return pd.DataFrame(
            {
                # среднее te с весом по числу операций в категории
                f"te_{name}_wmean": (C * te).sum(axis=1) / C.sum(axis=1).clip(min=1),
                # простое среднее, максимум, минимум, разброс по категориям клиента
                f"te_{name}_mean": np.nanmean(te_mat, 1),
                f"te_{name}_max": np.nanmax(te_mat, 1),
                f"te_{name}_min": np.nanmin(te_mat, 1),
                f"te_{name}_std": np.nanstd(te_mat, 1, ddof=1),
                # сумма и среднее лог-шансов («наивный Байес» по набору категорий клиента)
                f"te_{name}_lo_sum": (present * lo).sum(axis=1),
                f"te_{name}_lo_mean": (present * lo).sum(axis=1) / k,
                # сумма лог-шансов с весом log(1 + операций) и с весом надёжности категории
                f"te_{name}_wlo_sum": (np.log1p(C) * lo).sum(axis=1),
                f"te_{name}_conf_lo_sum": (present * lo * conf).sum(axis=1),
            },
            index=counts.index[rows],
        )


def _tfidf(M: np.ndarray, idf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    # клиент как «текст», категории — «слова», число операций — частота слова;
    # TF = log(1 + частота): 100 операций не в 100 раз важнее одной
    X = np.log1p(M)
    # IDF считается только по обучающим клиентам (при применении передаётся готовый):
    # категории, которые есть почти у всех, получают малый вес, редкие — большой
    if idf is None:
        idf = np.log((1 + M.shape[0]) / (1 + (M > 0).sum(0))) + 1
    X = X * idf
    # L2-нормировка строки: важен состав профиля, а не объём операций
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12), idf


class LabelFeatures:
    """Признаки с метками: target encoding MCC/типов, логрег по TF-IDF профилей, логрег по эмбеддингам.

    ``fit_transform(X, y)`` возвращает out-of-fold значения для обучающих клиентов (внутренние фолды),
    ``transform(X)`` — значения для новых клиентов по статистикам, посчитанным на всех обучающих.
    """

    def __init__(self, cfg: dict, seed: int = 0, n_inner: int = 5):
        # n_inner — число внутренних фолдов для out-of-fold расчёта на обучающих клиентах
        self.cfg, self.seed, self.n_inner = cfg, seed, n_inner

    @staticmethod
    def _raw(X: pd.DataFrame, prefix: str) -> pd.DataFrame:
        # колонки одной группы по префиксу (например все raw_mcccnt_*)
        return X.loc[:, X.columns.str.startswith(prefix)]

    def _fit_parts(self, X: pd.DataFrame, y: np.ndarray, rows: np.ndarray) -> dict:
        # всё, что «учится» на метках, учится только на клиентах rows — их метки и используются
        te_cfg = self.cfg["target_encoding"]["smoothing"]
        parts = {
            # средняя доля мужчин среди обучающих клиентов — к ней сглаживается target encoding
            "prior": float(y[rows].mean()),
            "te_mcc": _te_block(self._raw(X, "raw_mcccnt_"), y, rows, te_cfg["mcc"]),
            "te_tt": _te_block(self._raw(X, "raw_ttcnt_"), y, rows, te_cfg["tr_type"]),
        }
        # TF-IDF по трём «текстам»: операции по MCC, операции по типам, суммы по MCC;
        # IDF каждого запоминаем, чтобы применять к новым клиентам те же веса
        mats, idfs = [], []
        for M in self._tfidf_inputs(X):
            A, idf = _tfidf(M[rows])
            mats.append(A)
            idfs.append(idf)
        # логистическая регрессия по склеенным TF-IDF: на входе профиль клиента, на выходе —
        # его логит «мужчина» (одно число), который становится признаком lr_tfidf для бустинга
        parts["tfidf"] = (
            LogisticRegression(C=self.cfg["tfidf_logreg_C"], max_iter=2000).fit(np.hstack(mats), y[rows]),
            idfs,
        )
        # логрег по эмбеддингу CoLES (1024 числа) → одно число lr_emb; стандартизация + сильная
        # L2-регуляризация (малое C), чтобы 1024 признака на нескольких тысячах клиентов не переобучились
        emb = self._raw(X, "emb_")
        if emb.shape[1]:
            parts["emb"] = make_pipeline(
                StandardScaler(), LogisticRegression(C=self.cfg["embedding_logreg_C"], max_iter=3000)
            ).fit(emb.to_numpy()[rows], y[rows])
        return parts

    def _tfidf_inputs(self, X: pd.DataFrame) -> list[np.ndarray]:
        # счётчики операций по MCC и по типам, лог-суммы трат по MCC
        return [
            self._raw(X, "raw_mcccnt_").to_numpy(float),
            self._raw(X, "raw_ttcnt_").to_numpy(float),
            np.log1p(self._raw(X, "raw_mccsum_").to_numpy(float)),
        ]

    def _apply(self, parts: dict, X: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        # применяем выученные статистики к клиентам rows (их метки не используются)
        out = [
            _te_apply(self._raw(X, "raw_mcccnt_"), parts["te_mcc"], parts["prior"], rows, "mcc"),
            _te_apply(self._raw(X, "raw_ttcnt_"), parts["te_tt"], parts["prior"], rows, "tt"),
        ]
        # TF-IDF с IDF обучающих клиентов → логит логрега (decision_function — логит, не вероятность)
        clf, idfs = parts["tfidf"]
        B = np.hstack([_tfidf(M[rows], idf)[0] for M, idf in zip(self._tfidf_inputs(X), idfs)])
        lin = pd.DataFrame({"lr_tfidf": clf.decision_function(B)}, index=X.index[rows])
        if "emb" in parts:
            lin["lr_emb"] = parts["emb"].decision_function(self._raw(X, "emb_").to_numpy()[rows])
        return pd.concat(out + [lin], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        yv = y.reindex(X.index).to_numpy()
        # out-of-fold: обучающие клиенты делятся на n_inner внутренних фолдов; признаки фолда b
        # считаются статистиками, выученными на остальных фолдах a — клиент не видит свою метку.
        # Иначе target encoding и логреги «запомнили» бы метки, и бустинг учился бы на утечке
        folds = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed).split(X, yv)
        parts = [self._apply(self._fit_parts(X, yv, a), X, b) for a, b in folds]
        # для новых клиентов (валидация, test) — статистики по всем обучающим клиентам
        self.parts_ = self._fit_parts(X, yv, np.arange(len(X)))
        # склеиваем куски фолдов и возвращаем строки в исходном порядке клиентов
        return pd.concat(parts).reindex(X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # новые клиенты: их метки не нужны и не используются
        return self._apply(self.parts_, X, np.arange(len(X)))


# описания признаков с метками для сводной таблицы в ноутбуке
LABEL_FEATURES = {
    "lr_tfidf": "логрег по TF-IDF числа операций по MCC и типам и сумм по MCC",
    "lr_emb": "логрег по эмбеддингу CoLES (1024 → 1 число)",
    "te_mcc_*": "доля мужчин по MCC клиента: 9 сводок (среднее, max, разброс, лог-шансы…)",
    "te_tt_wmean": "доля мужчин по типам операций клиента, среднее по операциям",
}


def label_feature_summary(L: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Признаки с метками (out-of-fold): что это, AUC в одиночку по клиентам без пропуска, покрытие."""
    from sklearn.metrics import roc_auc_score

    yv = y.reindex(L.index)

    def auc(c: str) -> float:
        # AUC признака как скора «мужчина»: только клиенты, у которых признак определён
        m = L[c].notna()
        return roc_auc_score(yv[m], L.loc[m, c])

    rows = []
    for name, what in LABEL_FEATURES.items():
        # «te_mcc_*» — группа из 9 колонок: показываем диапазон AUC; иначе — одна колонка
        cols = [c for c in L.columns if c.startswith(name[:-1])] if name.endswith("*") else [name]
        aucs = [auc(c) for c in cols]
        rows.append(
            {
                "признак": name,
                "что это": what,
                "AUC в одиночку": f"{min(aucs):.3f}–{max(aucs):.3f}" if len(cols) > 1 else f"{aucs[0]:.3f}",
                # доля клиентов, у которых определены все колонки признака
                "покрытие, %": round(100 * L[cols].notna().all(axis=1).mean(), 1),
            }
        )
    return pd.DataFrame(rows).set_index("признак")


def model_matrix(X: pd.DataFrame, label_feats: pd.DataFrame) -> pd.DataFrame:
    """Итоговая матрица для бустинга: признаки без меток + признаки с метками (без raw_/emb_)."""
    # raw_* (сырые счётчики) и emb_* (1024 числа эмбеддинга) в бустинг не идут: их информация
    # уже сжата в признаки с метками (te_*, lr_tfidf, lr_emb)
    keep = ~X.columns.str.startswith(("raw_", "emb_"))
    return pd.concat([X.loc[:, keep], label_feats.reindex(X.index)], axis=1)


# ---------------------------------------------------------------- профиль клиента (аналитика, без меток)
