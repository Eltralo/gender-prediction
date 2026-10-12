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
    # обучаем сразу максимальное число деревьев (3000), а нужное число выберем потом по кривой AUC
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), g["max_rounds"])
    # сетка 50, 100, …, 3000 деревьев и предсказания валидации на каждом шаге
    # → матрица (шаги сетки × клиенты валидации); по ней строится AUC-кривая фолда
    its = np.arange(g["eval_step"], g["max_rounds"] + 1, g["eval_step"])
    return its, np.stack([model.predict(X_va, num_iteration=int(i)) for i in its])


def xgb_staged(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost для сравнения: сетка итераций и предсказания на ней (как ``gbm_staged``)."""
    import xgboost as xgb

    # параметры — из секции comparison конфига; в итоговую модель XGBoost не входит
    c, step = cfg["comparison"], cfg["gbm"]["eval_step"]
    params = dict(c["xgboost"], seed=seed, nthread=cfg["gbm"]["num_threads"])
    model = xgb.train(params, xgb.DMatrix(X_tr, y_tr), c["max_rounds"]["xgboost"])
    # та же сетка числа деревьев, что у LightGBM, — для одинакового вложенного выбора
    its = np.arange(step, c["max_rounds"]["xgboost"] + 1, step)
    dva = xgb.DMatrix(X_va)
    return its, np.stack([model.predict(dva, iteration_range=(0, int(i))) for i in its])


def cat_staged(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost для сравнения: сетка итераций и предсказания на ней (как ``gbm_staged``)."""
    from catboost import CatBoostClassifier

    c, step = cfg["comparison"], cfg["gbm"]["eval_step"]
    model = CatBoostClassifier(
        iterations=c["max_rounds"]["catboost"],
        thread_count=cfg["gbm"]["num_threads"],
        random_seed=seed,
        verbose=0,
        **c["catboost"],
    ).fit(X_tr, y_tr)
    # staged_predict_proba отдаёт предсказания каждые step деревьев — та же сетка, что у LightGBM
    preds = [q[:, 1] for q in model.staged_predict_proba(X_va, eval_period=step)]
    return np.arange(step, step * len(preds) + 1, step), np.stack(preds)


def gbm_fit(X: pd.DataFrame, y: np.ndarray, rounds: int, cfg: dict, seed: int) -> lgb.Booster:
    # LightGBM с заранее выбранным числом деревьев (финальная модель, отбор признаков для TabPFN)
    g = cfg["gbm"]
    params = dict(g["params"], seed=seed, num_threads=g["num_threads"])
    return lgb.train(params, lgb.Dataset(X, y), int(rounds))


def nested_rounds(curves: list[np.ndarray], its: np.ndarray) -> list[int]:
    """Число итераций для фолда k — argmax средней AUC-кривой остальных фолдов (без утечки в фолд k)."""
    # curves — AUC-кривые фолдов (фолды × шаги сетки)
    C = np.vstack(curves)
    # вложенный выбор: для фолда k усредняем кривые всех фолдов, кроме k, и берём лучший шаг;
    # так оценка фолда k не завышена подбором числа деревьев на нём самом
    return [int(its[np.delete(C, k, axis=0).mean(0).argmax()]) for k in range(len(curves))]


# ---------------------------------------------------------------- TabPFN
class TabPFNModel:
    """TabPFN v2 на top-k признаках; отбор — по gain LightGBM, обученного на тех же обучающих данных."""

    def __init__(self, cfg: dict, device: str, seed: int):
        self.cfg, self.device, self.seed = cfg, device, seed

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabPFNModel:
        # TabPFN рассчитан на сотни признаков, у нас 719 → отбираем 300 по важности (gain) LightGBM.
        # Отбор делается внутри fit, то есть только на обучающих клиентах фолда: валидационный фолд
        # в выборе признаков не участвует, иначе оценка была бы завышена
        sel = gbm_fit(X, y, 1000, self.cfg, self.seed)
        imp = pd.Series(sel.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        self.columns_ = list(imp.index[: self.cfg["tabpfn"]["top_k"]])
        # TabPFN «обучается» запоминанием обучающей выборки: она — контекст для предсказания
        self.X_fit_, self.y_fit_ = X[self.columns_].to_numpy(np.float32), np.asarray(y)
        self._fit_tabpfn()
        return self

    def _fit_tabpfn(self) -> None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        # предобученный трансформер TabPFN v2; n_estimators — число прогонов с разной обработкой
        # признаков, ответы которых усредняются; ignore_pretraining_limits снимает встроенные
        # ограничения TabPFN на размер данных (в том числе запрет больших выборок на CPU)
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
        # после загрузки из artifacts модели нет — восстанавливаем её из сохранённой обучающей матрицы
        if not hasattr(self, "model_"):
            self._fit_tabpfn()
        return self.model_.predict_proba(X[self.columns_].to_numpy(np.float32))[:, 1]


# ---------------------------------------------------------------- бленд
def to_logit(p: np.ndarray) -> np.ndarray:
    # вероятность → логит log(p / (1 − p)); обрезка, чтобы p = 0 или 1 не давали ±∞.
    # Бленд складывает логиты: так модели с разной «уверенностью» комбинируются корректно
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _simplex(m: int, step: float) -> list[np.ndarray]:
    # все наборы весов m моделей с шагом step, неотрицательные и в сумме 1 (сетка на симплексе)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    if m == 1:
        return [np.ones(1)]
    # перебираем веса m − 1 моделей, вес первой — остаток до 1
    return [np.array([1 - sum(w), *w]) for w in itertools.product(grid, repeat=m - 1) if sum(w) <= 1 + 1e-9]


class LogitBlend:
    """Калиброванный бленд: Platt для каждого компонента -> взвешенная сумма логитов -> итоговый Platt.

    Веса — на симплексе с шагом ``step`` по AUC. Работает построчно, поэтому применим к test без
    информации о составе test (в отличие от рангового бленда).
    """

    def __init__(self, step: float = 0.05):
        self.step = step

    def _calibrated(self, Z: np.ndarray) -> np.ndarray:
        # применяем Platt каждой модели: калиброванный логит a·logit + b (по колонке на модель)
        return np.column_stack([pl.decision_function(Z[:, [i]]) for i, pl in enumerate(self.platt_)])

    def fit(self, Z: np.ndarray, y: np.ndarray) -> LogitBlend:
        # Z — out-of-fold логиты моделей (клиенты × модели), y — метки тех же клиентов.
        # Platt-калибровка: логистическая регрессия по одному логиту (C=1e6 — практически без
        # регуляризации); приводит шкалы моделей к одной — логиту настоящей вероятности
        self.platt_ = [LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, [i]], y) for i in range(Z.shape[1])]
        C = self._calibrated(Z)
        # веса: перебор сетки на симплексе (шаг 0.05), выбираем веса с максимальным AUC
        self.weights_ = max(_simplex(Z.shape[1], self.step), key=lambda w: roc_auc_score(y, C @ w))
        # итоговый Platt по взвешенной сумме: вероятность бленда снова откалибрована,
        # поэтому порог 0.5 подходит для accuracy без подбора
        self.final_ = LogisticRegression(C=1e6, max_iter=1000).fit((C @ self.weights_)[:, None], y)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        # калибровка каждой модели → взвешенная сумма логитов → итоговая калибровка → P(мужчина)
        return self.final_.predict_proba((self._calibrated(Z) @ self.weights_)[:, None])[:, 1]

    def to_dict(self) -> dict:
        # бленд — это несколько чисел: веса и пары (a, b) калибровок; сохраняются в model.json
        return {
            "weights": self.weights_.tolist(),
            "platt": [[float(p.coef_[0, 0]), float(p.intercept_[0])] for p in self.platt_],
            "final": [float(self.final_.coef_[0, 0]), float(self.final_.intercept_[0])],
        }

    @classmethod
    def from_dict(cls, d: dict) -> LogitBlend:
        def lr(coef: float, icpt: float) -> LogisticRegression:
            # восстанавливаем «обученную» логрегрессию из сохранённых коэффициентов без обучения
            m = LogisticRegression()
            m.coef_, m.intercept_, m.classes_ = np.array([[coef]]), np.array([icpt]), np.array([0, 1])
            return m

        # собираем бленд из сохранённых весов и коэффициентов калибровок
        b = cls()
        b.weights_ = np.array(d["weights"])
        b.platt_ = [lr(*p) for p in d["platt"]]
        b.final_ = lr(*d["final"])
        return b
