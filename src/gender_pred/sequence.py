"""Последовательные модели: энкодер CoLES (контрастное предобучение без меток) и его дообучение на пол.

Энкодер: эмбеддинги MCC и типа операции, знаковая лог-сумма, час (полночь — «время неизвестно»),
день недели и log Δt -> GRU. Предобучение — CoLES: срезы одного клиента сближаются, чужие отталкиваются
(contrastive loss с hard negative mining).
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Vocab


class Sequences:
    """Транзакции всех клиентов в плоских массивах + смещения по клиентам."""

    def __init__(self, ev: pd.DataFrame, vocab: Vocab):
        # операции клиента подряд и в хронологическом порядке
        ev = ev.sort_values(["cid", "t"], kind="stable")
        # коды категорий → номера 1…N по словарю train; 0 зарезервирован под паддинг и неизвестные коды
        mcc_map = {c: i + 1 for i, c in enumerate(vocab.mcc)}
        tt_map = {c: i + 1 for i, c in enumerate(vocab.tr_type)}
        self.n_mcc, self.n_tt = len(mcc_map) + 1, len(tt_map) + 1
        self.mcc = ev["mcc"].map(mcc_map).fillna(0).to_numpy(np.int64)
        self.tt = ev["tt"].map(tt_map).fillna(0).to_numpy(np.int64)
        # час 0…23; операции 00:00:00 («время неизвестно») получают отдельный код 24
        self.hour = np.where(ev["mid"], 24, ev["hour"]).astype(np.int64)
        self.dow = ev["dow"].to_numpy(np.int64)
        cid = ev["cid"].to_numpy()
        # new — первая операция каждого клиента (смена cid в плоском массиве)
        new = np.r_[True, cid[1:] != cid[:-1]]
        # знаковая лог-сумма: знак — расход / поступление, модуль — la = log(1 + |amount|)
        sla = np.sign(ev["amt"].to_numpy()) * ev["la"].to_numpy()
        # Δt — дней с предыдущей операции клиента; у первой операции клиента 0
        dt = np.diff(ev["t"].to_numpy(), prepend=0.0)
        dt[new] = 0
        # логарифм интервала: от секунд до месяцев — разница на порядки
        dt = np.log1p(np.clip(dt, 0, None))
        # фиксированные масштабы: log1p суммы ~ до 19, log1p Δt (дни) ~ до 6
        self.num = np.stack([sla / 19.0, dt / 6.0], 1).astype(np.float32)
        # для каждого клиента: id, начало в плоском массиве и длина последовательности
        self.ids = cid[new]
        self.start = np.flatnonzero(new)
        self.len = np.diff(np.r_[self.start, len(cid)])
        # customer_id → номер клиента в этих массивах
        self.pos = {int(c): i for i, c in enumerate(self.ids)}

    def full(self, c: int) -> np.ndarray:
        # индексы всех операций клиента в плоских массивах
        p = self.pos[int(c)]
        return np.arange(self.start[p], self.start[p] + self.len[p])

    def tail(self, c: int, max_len: int) -> np.ndarray:
        # последние max_len операций: для предсказания берём самую свежую историю
        idx = self.full(c)
        return idx[-max_len:]

    def batch(self, rows: list[np.ndarray], device: str) -> dict:
        # последовательности разной длины дополняем нулями до самой длинной в батче
        L = max(len(r) for r in rows)
        idx = np.zeros((len(rows), L), dtype=np.int64)
        mask = np.zeros((len(rows), L), dtype=bool)
        # idx — номера операций в плоских массивах, mask — где настоящие операции, а где паддинг
        for i, r in enumerate(rows):
            idx[i, : len(r)] = r
            mask[i, : len(r)] = True
        m = torch.from_numpy(mask).to(device)

        def t(a: np.ndarray) -> torch.Tensor:
            # берём значения нужных операций → тензор батч × длина
            return torch.from_numpy(a[idx]).to(device)

        # паддинг обнуляем (код 0 у категорий, 0 у чисел); GRU всё равно читает только len шагов
        return {
            "mcc": t(self.mcc) * m,
            "tt": t(self.tt) * m,
            "hour": t(self.hour),
            "dow": t(self.dow),
            "num": t(self.num) * m[..., None],
            "len": torch.tensor([len(r) for r in rows]),
        }


class NoisyEmbedding(nn.Embedding):
    def __init__(self, *args, noise: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # во время обучения к эмбеддингам категорий добавляется небольшой гауссов шум —
        # регуляризация (как в эталонном конфиге CoLES); при предсказании шума нет
        e = super().forward(x)
        return e + torch.randn_like(e) * self.noise if self.training and self.noise > 0 else e


class Encoder(nn.Module):
    """GRU-энкодер транзакций; выход — последнее скрытое состояние."""

    def __init__(
        self,
        n_mcc: int,
        n_tt: int,
        hidden: int = 1024,
        mcc_dim: int = 48,
        tr_type_dim: int = 24,
        use_time: bool = True,
        emb_noise: float = 0.003,
    ):
        super().__init__()
        self.use_time = use_time
        # обучаемые эмбеддинги категорий: MCC → 48 чисел, тип операции → 24; код 0 (паддинг) — нулевой вектор
        self.e_mcc = NoisyEmbedding(n_mcc, mcc_dim, padding_idx=0, noise=emb_noise)
        self.e_tt = NoisyEmbedding(n_tt, tr_type_dim, padding_idx=0, noise=emb_noise)
        # вход шага: эмбеддинги MCC и типа + знаковая лог-сумма
        d_in = mcc_dim + tr_type_dim + 1
        if use_time:
            # время: час (25 значений, включая «неизвестно»), день недели, log Δt
            self.e_hour = nn.Embedding(25, 8)
            self.e_dow = nn.Embedding(7, 4)
            d_in += 8 + 4 + 1
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        # обучаемое начальное скрытое состояние вместо нулевого
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden))
        self.hidden = hidden

    def forward(self, b: dict) -> torch.Tensor:
        # собираем вектор каждой операции: батч × длина × d_in
        parts = [self.e_mcc(b["mcc"]), self.e_tt(b["tt"]), b["num"][..., :1]]
        if self.use_time:
            parts += [self.e_hour(b["hour"]), self.e_dow(b["dow"]), b["num"][..., 1:2]]
        x = torch.cat(parts, -1)
        # pack_padded_sequence: GRU проходит только реальные операции, паддинг пропускается
        packed = nn.utils.rnn.pack_padded_sequence(x, b["len"], batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed, self.h0.expand(1, x.shape[0], -1).contiguous())
        # эмбеддинг клиента — скрытое состояние после последней операции (вектор из hidden = 1024 чисел)
        return h[-1]


def build_encoder(seqs: Sequences, cfg: dict) -> Encoder:
    # размеры словарей — из данных, остальное — из конфига (эталонный конфиг CoLES)
    e = cfg["sequence"]["encoder"]
    return Encoder(
        seqs.n_mcc, seqs.n_tt, e["hidden"], e["mcc_dim"], e["tr_type_dim"], e["use_time"], e["emb_noise"]
    )


def contrastive_loss(z: torch.Tensor, labels: torch.Tensor, margin: float, neg_count: int) -> torch.Tensor:
    """Позитивы — все пары срезов одного клиента, негативы — ``neg_count`` ближайших чужих на якорь."""
    # эмбеддинги на единичной сфере: важна только «форма» поведения, расстояния в [0, 2]
    z = F.normalize(z, dim=1)
    # попарные евклидовы расстояния между всеми срезами в батче
    d = (z[:, None] - z[None]).pow(2).sum(-1).clamp(min=1e-12).sqrt()
    # same — пары срезов одного клиента (labels — номер клиента в батче)
    same = labels[:, None] == labels[None]
    # позитивы: срезы одного клиента притягиваются — штраф d² за каждую пару
    pos = d[torch.triu(same, diagonal=1)].pow(2)
    # hard negatives: для каждого среза — neg_count самых близких срезов других клиентов
    hard = d.masked_fill(same, float("inf")).topk(neg_count, dim=1, largest=False).values
    # негативы отталкиваются, пока не окажутся дальше margin; дальше margin штрафа нет
    neg = F.relu(margin - hard).pow(2).flatten()
    return torch.cat([pos, neg]).sum()


def _dropout(rng: np.random.Generator, idx: np.ndarray, p: float) -> np.ndarray:
    # случайно выбрасываем долю p операций — аугментация: энкодер учится быть устойчивым к пропускам
    if p <= 0 or len(idx) < 2:
        return idx
    keep = rng.random(len(idx)) >= p
    return idx[keep] if keep.any() else idx


def _slice(rng: np.random.Generator, n: int, lo: int, hi: int) -> tuple[int, int]:
    # случайный непрерывный срез истории длиной от lo до hi операций (не длиннее самой истории);
    # возвращает начало и длину среза
    length = int(rng.integers(min(lo, n), min(hi, n) + 1))
    return int(rng.integers(0, n - length + 1)), length


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def pretrain(
    seqs: Sequences, clients: Iterable[int], cfg: dict, device: str, seed: int, verbose: bool = True
) -> Encoder:
    """Контрастное предобучение энкодера (CoLES) на последовательностях ``clients`` без меток."""
    p = cfg["sequence"]["pretrain"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # клиенты со слишком короткой историей не дают осмысленных срезов — пропускаем их
    ids = np.array([c for c in clients if seqs.len[seqs.pos[int(c)]] >= p["min_seq_len"]])
    model = build_encoder(seqs, cfg).to(device)
    # Adam + StepLR: lr умножается на gamma каждые step_size эпох (эталонный конфиг CoLES)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=p["step_size"], gamma=p["gamma"])
    # эпоха: клиенты в случайном порядке, батчи по batch_size клиентов
    t0 = time.time()
    for ep in range(1, p["epochs"] + 1):
        perm = rng.permutation(ids)
        total = 0.0
        for i in range(0, len(perm), p["batch_size"]):
            # батч: для каждого клиента n_slices случайных срезов; метка среза — номер клиента в батче.
            # Метки пола не используются: учимся различать клиентов, а не пол
            rows, lab = [], []
            for j, c in enumerate(perm[i : i + p["batch_size"]]):
                full = _dropout(rng, seqs.full(c), p["trx_dropout"])
                for _ in range(p["n_slices"]):
                    s, n = _slice(rng, len(full), p["slice_min"], p["slice_max"])
                    rows.append(full[s : s + n])
                    lab.append(j)
            # эмбеддинги всех срезов и контрастная потеря: свои срезы сближаются, чужие отталкиваются
            z = model(seqs.batch(rows, device))
            loss = contrastive_loss(z, torch.tensor(lab, device=device), p["margin"], p["neg_count"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # суммарная потеря эпохи — для лога
            total += loss.item()
        sched.step()
        # лог раз в 10 эпох: средняя потеря на батч и прошедшее время
        if ep % 10 == 0 or ep == p["epochs"]:
            _log(
                f"  CoLES epoch {ep}/{p['epochs']}  loss {total / max(1, len(perm) // p['batch_size']):.2f}"
                f"  [{time.time() - t0:.0f}s]",
                verbose,
            )
    return model.eval()


@torch.no_grad()
def embed(
    model: Encoder,
    seqs: Sequences,
    clients: Iterable[int],
    device: str,
    max_len: int = 2000,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Эмбеддинг клиента — состояние энкодера после последних ``max_len`` транзакций."""
    model.eval()
    clients = [int(c) for c in clients]
    # сортируем по длине истории: в батче похожие длины — меньше паддинга, быстрее
    order = sorted(clients, key=lambda c: seqs.len[seqs.pos[c]])
    out = {}
    for i in range(0, len(order), batch_size):
        cs = order[i : i + batch_size]
        z = model(seqs.batch([seqs.tail(c, max_len) for c in cs], device)).float().cpu().numpy()
        out.update(zip(cs, z))
    # возвращаем в исходном порядке клиентов: таблица клиент × emb_0 … emb_1023
    E = pd.DataFrame(np.stack([out[c] for c in clients]), index=pd.Index(clients, name="customer_id"))
    return E.set_axis([f"emb_{i}" for i in range(E.shape[1])], axis=1)


class SeqClassifier(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        # голова поверх эмбеддинга: нормализация + линейный слой → один логит «мужчина»
        self.head = nn.Sequential(nn.BatchNorm1d(encoder.hidden), nn.Linear(encoder.hidden, 1))

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.encoder(b)).squeeze(1)


def _random_slice(rng: np.random.Generator, n: int, lo: int, hi: int, rate: float) -> tuple[int, int]:
    # срез для дообучения: обычно 250–350 операций; у коротких историй — не меньше доли rate от длины
    if n <= 5:
        return 0, n
    # границы длины среза не больше самой истории; начало среза — случайное
    lo = min(lo, int(n * rate))
    hi = min(hi, n)
    length = int(rng.integers(lo, hi + 1)) if hi > lo else hi
    return int(rng.integers(0, n - length + 1)), length


def finetune(
    encoder_state: dict, seqs: Sequences, clients: np.ndarray, y: pd.Series, cfg: dict, device: str, seed: int
) -> SeqClassifier:
    """Дообучение предобученного энкодера с BCE-головой на пол."""
    f = cfg["sequence"]["finetune"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # стартуем с весов предобученного энкодера (копия, исходный энкодер не меняется)
    enc = build_encoder(seqs, cfg)
    enc.load_state_dict(encoder_state)
    model = SeqClassifier(enc).to(device).train()
    # разные lr: энкодер дообучается осторожно (1e-4), новая голова — быстрее (5e-3)
    opt = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": f["lr_encoder"]},
            {"params": model.head.parameters(), "lr": f["lr_head"]},
        ]
    )
    # lr умножается на gamma после каждой эпохи
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=f["gamma"])
    # метки только переданных клиентов: в CV это обучающая часть фолда (out-of-fold для остальных)
    yv = y.reindex(clients).to_numpy(np.float32)
    for _ in range(f["epochs"]):
        perm = rng.permutation(len(clients))
        for i in range(0, len(perm), f["batch_size"]):
            bi = perm[i : i + f["batch_size"]]
            # BatchNorm не работает на батче из одного примера
            if len(bi) < 2:
                continue
            # каждый клиент — случайный срез истории (аугментация, как при предобучении)
            rows = []
            for c in clients[bi]:
                full = _dropout(rng, seqs.full(c), f["trx_dropout"])
                s, n = _random_slice(rng, len(full), f["slice_min"], f["slice_max"], f["slice_rate_for_min"])
                rows.append(full[s : s + n])
            # бинарная кросс-энтропия по логиту «мужчина»
            logit = model(seqs.batch(rows, device))
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(yv[bi]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict_logits(
    model: SeqClassifier,
    seqs: Sequences,
    clients: Iterable[int],
    device: str,
    max_len: int = 1200,
    batch_size: int = 256,
) -> np.ndarray:
    # предсказание по последним max_len операциям (1200 — вся история у 96% клиентов);
    # на выходе логиты, не вероятности: бленд работает в пространстве логитов
    model.eval()
    clients = [int(c) for c in clients]
    # предсказываем батчами по batch_size клиентов, без градиентов (no_grad)
    out = np.zeros(len(clients))
    for i in range(0, len(clients), batch_size):
        cs = clients[i : i + batch_size]
        out[i : i + len(cs)] = model(seqs.batch([seqs.tail(c, max_len) for c in cs], device)).cpu().numpy()
    return out
