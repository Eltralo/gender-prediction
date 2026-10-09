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
        ev = ev.sort_values(["cid", "t"], kind="stable")
        mcc_map = {c: i + 1 for i, c in enumerate(vocab.mcc)}
        tt_map = {c: i + 1 for i, c in enumerate(vocab.tr_type)}
        self.n_mcc, self.n_tt = len(mcc_map) + 1, len(tt_map) + 1
        self.mcc = ev["mcc"].map(mcc_map).fillna(0).to_numpy(np.int64)
        self.tt = ev["tt"].map(tt_map).fillna(0).to_numpy(np.int64)
        self.hour = np.where(ev["mid"], 24, ev["hour"]).astype(np.int64)
        self.dow = ev["dow"].to_numpy(np.int64)
        cid = ev["cid"].to_numpy()
        new = np.r_[True, cid[1:] != cid[:-1]]
        sla = np.sign(ev["amt"].to_numpy()) * ev["la"].to_numpy()
        dt = np.diff(ev["t"].to_numpy(), prepend=0.0)
        dt[new] = 0
        dt = np.log1p(np.clip(dt, 0, None))
        # фиксированные масштабы: log1p суммы ~ до 19, log1p Δt (дни) ~ до 6
        self.num = np.stack([sla / 19.0, dt / 6.0], 1).astype(np.float32)
        self.ids = cid[new]
        self.start = np.flatnonzero(new)
        self.len = np.diff(np.r_[self.start, len(cid)])
        self.pos = {int(c): i for i, c in enumerate(self.ids)}

    def full(self, c: int) -> np.ndarray:
        p = self.pos[int(c)]
        return np.arange(self.start[p], self.start[p] + self.len[p])

    def tail(self, c: int, max_len: int) -> np.ndarray:
        idx = self.full(c)
        return idx[-max_len:]

    def batch(self, rows: list[np.ndarray], device: str) -> dict:
        L = max(len(r) for r in rows)
        idx = np.zeros((len(rows), L), dtype=np.int64)
        mask = np.zeros((len(rows), L), dtype=bool)
        for i, r in enumerate(rows):
            idx[i, : len(r)] = r
            mask[i, : len(r)] = True
        m = torch.from_numpy(mask).to(device)

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a[idx]).to(device)

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
        self.e_mcc = NoisyEmbedding(n_mcc, mcc_dim, padding_idx=0, noise=emb_noise)
        self.e_tt = NoisyEmbedding(n_tt, tr_type_dim, padding_idx=0, noise=emb_noise)
        d_in = mcc_dim + tr_type_dim + 1
        if use_time:
            self.e_hour = nn.Embedding(25, 8)
            self.e_dow = nn.Embedding(7, 4)
            d_in += 8 + 4 + 1
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden))
        self.hidden = hidden

    def forward(self, b: dict) -> torch.Tensor:
        parts = [self.e_mcc(b["mcc"]), self.e_tt(b["tt"]), b["num"][..., :1]]
        if self.use_time:
            parts += [self.e_hour(b["hour"]), self.e_dow(b["dow"]), b["num"][..., 1:2]]
        x = torch.cat(parts, -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, b["len"], batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed, self.h0.expand(1, x.shape[0], -1).contiguous())
        return h[-1]


def build_encoder(seqs: Sequences, cfg: dict) -> Encoder:
    e = cfg["sequence"]["encoder"]
    return Encoder(
        seqs.n_mcc, seqs.n_tt, e["hidden"], e["mcc_dim"], e["tr_type_dim"], e["use_time"], e["emb_noise"]
    )


def contrastive_loss(z: torch.Tensor, labels: torch.Tensor, margin: float, neg_count: int) -> torch.Tensor:
    """Позитивы — все пары срезов одного клиента, негативы — ``neg_count`` ближайших чужих на якорь."""
    z = F.normalize(z, dim=1)
    d = (z[:, None] - z[None]).pow(2).sum(-1).clamp(min=1e-12).sqrt()
    same = labels[:, None] == labels[None]
    pos = d[torch.triu(same, diagonal=1)].pow(2)
    hard = d.masked_fill(same, float("inf")).topk(neg_count, dim=1, largest=False).values
    neg = F.relu(margin - hard).pow(2).flatten()
    return torch.cat([pos, neg]).sum()


def _dropout(rng: np.random.Generator, idx: np.ndarray, p: float) -> np.ndarray:
    if p <= 0 or len(idx) < 2:
        return idx
    keep = rng.random(len(idx)) >= p
    return idx[keep] if keep.any() else idx


def _slice(rng: np.random.Generator, n: int, lo: int, hi: int) -> tuple[int, int]:
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
    ids = np.array([c for c in clients if seqs.len[seqs.pos[int(c)]] >= p["min_seq_len"]])
    model = build_encoder(seqs, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=p["step_size"], gamma=p["gamma"])
    t0 = time.time()
    for ep in range(1, p["epochs"] + 1):
        perm = rng.permutation(ids)
        total = 0.0
        for i in range(0, len(perm), p["batch_size"]):
            rows, lab = [], []
            for j, c in enumerate(perm[i : i + p["batch_size"]]):
                full = _dropout(rng, seqs.full(c), p["trx_dropout"])
                for _ in range(p["n_slices"]):
                    s, n = _slice(rng, len(full), p["slice_min"], p["slice_max"])
                    rows.append(full[s : s + n])
                    lab.append(j)
            z = model(seqs.batch(rows, device))
            loss = contrastive_loss(z, torch.tensor(lab, device=device), p["margin"], p["neg_count"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
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
    order = sorted(clients, key=lambda c: seqs.len[seqs.pos[c]])
    out = {}
    for i in range(0, len(order), batch_size):
        cs = order[i : i + batch_size]
        z = model(seqs.batch([seqs.tail(c, max_len) for c in cs], device)).float().cpu().numpy()
        out.update(zip(cs, z))
    E = pd.DataFrame(np.stack([out[c] for c in clients]), index=pd.Index(clients, name="customer_id"))
    return E.set_axis([f"emb_{i}" for i in range(E.shape[1])], axis=1)


class SeqClassifier(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.BatchNorm1d(encoder.hidden), nn.Linear(encoder.hidden, 1))

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.encoder(b)).squeeze(1)


def _random_slice(rng: np.random.Generator, n: int, lo: int, hi: int, rate: float) -> tuple[int, int]:
    if n <= 5:
        return 0, n
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
    enc = build_encoder(seqs, cfg)
    enc.load_state_dict(encoder_state)
    model = SeqClassifier(enc).to(device).train()
    opt = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": f["lr_encoder"]},
            {"params": model.head.parameters(), "lr": f["lr_head"]},
        ]
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=f["gamma"])
    yv = y.reindex(clients).to_numpy(np.float32)
    for _ in range(f["epochs"]):
        perm = rng.permutation(len(clients))
        for i in range(0, len(perm), f["batch_size"]):
            bi = perm[i : i + f["batch_size"]]
            if len(bi) < 2:
                continue
            rows = []
            for c in clients[bi]:
                full = _dropout(rng, seqs.full(c), f["trx_dropout"])
                s, n = _random_slice(rng, len(full), f["slice_min"], f["slice_max"], f["slice_rate_for_min"])
                rows.append(full[s : s + n])
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
    model.eval()
    clients = [int(c) for c in clients]
    out = np.zeros(len(clients))
    for i in range(0, len(clients), batch_size):
        cs = clients[i : i + batch_size]
        out[i : i + len(cs)] = model(seqs.batch([seqs.tail(c, max_len) for c in cs], device)).cpu().numpy()
    return out
