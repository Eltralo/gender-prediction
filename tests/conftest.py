import copy

import numpy as np
import pandas as pd
import pytest

from gender_pred.data import load_config


def synthetic_split(n_clients: int, start_id: int, seed: int) -> pd.DataFrame:
    """Транзакции с полозависимым распределением MCC и сумм."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(start_id, start_id + n_clients):
        g = int(rng.random() < 0.45)
        n = int(rng.integers(30, 120))
        mcc_p = np.full(12, 1.0)
        mcc_p[[0, 1, 2] if g else [9, 10, 11]] *= 4
        mcc = rng.choice(12, n, p=mcc_p / mcc_p.sum())
        days = np.sort(rng.integers(0, 400, n))
        secs = rng.integers(0, 86400, n) * (rng.random(n) > 0.1)
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": cid,
                    "tr_datetime": pd.Timestamp("2000-01-01")
                    + pd.to_timedelta(days, "D")
                    + pd.to_timedelta(secs, "s"),
                    "mcc_code": mcc,
                    "tr_type": rng.integers(0, 5, n),
                    "amount": -np.exp(rng.normal(8 + 0.3 * g, 1, n)) * np.where(rng.random(n) < 0.85, 1, -1),
                    "term_id": np.where(rng.random(n) < 0.5, -1, rng.integers(0, 50, n)).astype(float),
                    "gender": g,
                    "mcc_code_desc": [f"mcc {m}" for m in mcc],
                }
            )
        )
    return (
        pd.concat(rows, ignore_index=True).sort_values(["customer_id", "tr_datetime"]).reset_index(drop=True)
    )


@pytest.fixture
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": synthetic_split(90, 0, 1),
        "val": synthetic_split(30, 1000, 2),
        "test": synthetic_split(40, 2000, 3),
    }


@pytest.fixture
def tiny_cfg(tmp_path) -> dict:
    cfg = load_config("configs/default.yaml", retrain=True)
    cfg = copy.deepcopy(cfg)
    cfg.update(retrain_tabular=True, retrain_seq=True, device="cpu")
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["cv"].update(n_splits=3, repeats=1, inner_splits=3)
    cfg["gbm"].update(max_rounds=100, eval_step=50, num_threads=2)
    cfg["sequence"]["encoder"].update(hidden=16, mcc_dim=4, tr_type_dim=4)
    cfg["sequence"]["pretrain"].update(epochs=1, batch_size=32, min_seq_len=10)
    cfg["sequence"]["finetune"].update(epochs=1, batch_size=16, slice_min=20, slice_max=40)
    cfg["tabpfn"]["enabled"] = False
    cfg["blend"]["components"] = ["gbm", "seq"]
    cfg["report"]["bootstrap"] = 50
    return cfg
