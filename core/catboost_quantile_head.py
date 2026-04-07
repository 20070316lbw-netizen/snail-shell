"""
catboost_quantile_head.py - CatBoost 分位数回归头

使用 CatBoost 的原生 Quantile loss function。
接口与 QuantileHead（LightGBM）完全一致，可直接替换。

依赖：
  pip install catboost>=1.0.0
"""

import sys
import os
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_quantile_head import BaseQuantileHead, FitConfig


class CatBoostQuantileHead(BaseQuantileHead):
    """
    四模型分位数回归头（CatBoost 后端）

    使用 CatBoost 的 RMSE / Quantile:alpha=x 损失函数。
    训练四个独立模型：MSE / q10 / q50 / q90。
    """

    backend_name = "CatBoost"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        subsample: float = 0.8,
        colsample_bylevel: float = 0.8,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        thread_count: int = -1,
    ):
        self.n_estimators        = n_estimators
        self.learning_rate       = learning_rate
        self.depth               = depth
        self.subsample           = subsample
        self.colsample_bylevel   = colsample_bylevel
        self.l2_leaf_reg         = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state        = random_state
        self.thread_count        = thread_count

        self.models    = {"mse": None, "q10": None, "q50": None, "q90": None}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def _make_model(self, loss_function: str):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations            = self.n_estimators,
            learning_rate         = self.learning_rate,
            depth                 = self.depth,
            subsample             = self.subsample,
            colsample_bylevel     = self.colsample_bylevel,
            l2_leaf_reg           = self.l2_leaf_reg,
            loss_function         = loss_function,
            early_stopping_rounds = self.early_stopping_rounds,
            random_seed           = self.random_state,
            thread_count          = self.thread_count,
            verbose               = False,
        )

    # ------------------------------------------------------------------
    def fit(self, config: FitConfig) -> None:
        X_tr, y_tr = config.X_train, config.y_train
        eval_set   = (config.X_val, config.y_val) if config.X_val is not None else None

        print("Training CatBoost MSE predictor...")
        self.models["mse"] = self._make_model("RMSE")
        self.models["mse"].fit(X_tr, y_tr, eval_set=eval_set)

        print("Training CatBoost q10 quantile regressor...")
        self.models["q10"] = self._make_model("Quantile:alpha=0.1")
        self.models["q10"].fit(X_tr, y_tr, eval_set=eval_set)

        print("Training CatBoost q50 quantile regressor (anchor)...")
        self.models["q50"] = self._make_model("Quantile:alpha=0.5")
        self.models["q50"].fit(X_tr, y_tr, eval_set=eval_set)

        print("Training CatBoost q90 quantile regressor...")
        self.models["q90"] = self._make_model("Quantile:alpha=0.9")
        self.models["q90"].fit(X_tr, y_tr, eval_set=eval_set)

        self.is_fitted = True
        print("All CatBoost models trained successfully!")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("CatBoostQuantileHead: 模型未训练，请先调用 fit()")
        return {
            "point" : self.models["mse"].predict(X),
            "q10"   : self.models["q10"].predict(X),
            "q50"   : self.models["q50"].predict(X),
            "q90"   : self.models["q90"].predict(X),
        }
