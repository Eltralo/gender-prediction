"""Табличные модели и бленд: LightGBM, TabPFN, калиброванный бленд в пространстве логитов."""

from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- LightGBM
def gbm_staged(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Обучает LightGBM на ``max_rounds`` итераций, возвращает сетку итераций и предсказания на ней."""
    g = cfg["gbm"]
    params = dict(g["params"], seed=seed, num_threads=g["num_threads"])
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), g["max_rounds"])
    its = np.arange(g["eval_step"], g["max_rounds"] + 1, g["eval_step"])
    return its, np.stack([model.predict(X_va, num_iteration=int(i)) for i in its])


def gbm_fit(X: pd.DataFrame, y: np.ndarray, rounds: int, cfg: dict, seed: int) -> lgb.Booster:
    g = cfg["gbm"]
    params = dict(g["params"], seed=seed, num_threads=g["num_threads"])
    return lgb.train(params, lgb.Dataset(X, y), int(rounds))


def nested_rounds(curves: list[np.ndarray], its: np.ndarray) -> list[int]:
    """Число итераций для фолда k — argmax средней AUC-кривой остальных фолдов (без утечки в фолд k)."""
    C = np.vstack(curves)
    return [int(its[np.delete(C, k, axis=0).mean(0).argmax()]) for k in range(len(curves))]


# ---------------------------------------------------------------- TabPFN
class TabPFNModel:
    """TabPFN v2 на top-k признаках; отбор — по gain LightGBM, обученного на тех же обучающих данных."""

    def __init__(self, cfg: dict, device: str, seed: int):
        self.cfg, self.device, self.seed = cfg, device, seed

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabPFNModel:
        sel = gbm_fit(X, y, 1000, self.cfg, self.seed)
        imp = pd.Series(sel.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        self.columns_ = list(imp.index[: self.cfg["tabpfn"]["top_k"]])
        self.X_fit_, self.y_fit_ = X[self.columns_].to_numpy(np.float32), np.asarray(y)
        self._fit_tabpfn()
        return self

    def _fit_tabpfn(self) -> None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        self.model_ = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=self.device,
            n_estimators=self.cfg["tabpfn"]["n_estimators"],
            random_state=self.seed,
            ignore_pretraining_limits=True,
        )
        self.model_.fit(self.X_fit_, self.y_fit_)

    def __getstate__(self) -> dict:
        # сохраняются колонки и обучающая матрица; TabPFN восстанавливается обучением за секунды
        return {k: v for k, v in self.__dict__.items() if k != "model_"}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            self._fit_tabpfn()
        return self.model_.predict_proba(X[self.columns_].to_numpy(np.float32))[:, 1]


# ---------------------------------------------------------------- бленд
def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _simplex(m: int, step: float) -> list[np.ndarray]:
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    if m == 1:
        return [np.ones(1)]
    return [np.array([1 - sum(w), *w]) for w in itertools.product(grid, repeat=m - 1) if sum(w) <= 1 + 1e-9]


class LogitBlend:
    """Калиброванный бленд: Platt для каждого компонента -> взвешенная сумма логитов -> итоговый Platt.

    Веса — на симплексе с шагом ``step`` по AUC. Работает построчно, поэтому применим к test без
    информации о составе test (в отличие от рангового бленда).
    """

    def __init__(self, step: float = 0.05):
        self.step = step

    def _calibrated(self, Z: np.ndarray) -> np.ndarray:
        return np.column_stack([pl.decision_function(Z[:, [i]]) for i, pl in enumerate(self.platt_)])

    def fit(self, Z: np.ndarray, y: np.ndarray) -> LogitBlend:
        self.platt_ = [LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, [i]], y) for i in range(Z.shape[1])]
        C = self._calibrated(Z)
        self.weights_ = max(_simplex(Z.shape[1], self.step), key=lambda w: roc_auc_score(y, C @ w))
        self.final_ = LogisticRegression(C=1e6, max_iter=1000).fit((C @ self.weights_)[:, None], y)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return self.final_.predict_proba((self._calibrated(Z) @ self.weights_)[:, None])[:, 1]

    def to_dict(self) -> dict:
        return {
            "weights": self.weights_.tolist(),
            "platt": [[float(p.coef_[0, 0]), float(p.intercept_[0])] for p in self.platt_],
            "final": [float(self.final_.coef_[0, 0]), float(self.final_.intercept_[0])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> LogitBlend:
        def lr(coef: float, icpt: float) -> LogisticRegression:
            m = LogisticRegression()
            m.coef_, m.intercept_, m.classes_ = np.array([[coef]]), np.array([icpt]), np.array([0, 1])
            return m

        b = cls()
        b.weights_ = np.array(d["weights"])
        b.platt_ = [lr(*p) for p in d["platt"]]
        b.final_ = lr(*d["final"])
        return b
