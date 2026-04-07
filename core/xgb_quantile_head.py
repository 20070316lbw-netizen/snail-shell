"""
xgb_quantile_head.py - XGBoost 分位数回归头

使用 XGBoost >= 2.0 的原生分位数目标函数 reg:quantileerror。
接口与 QuantileHead（LightGBM）完全一致，可直接替换。

依赖：
  pip install xgboost>=2.0.0
"""

import sys
import os
import numpy as np
import xgboost as xgb
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_quantile_head import BaseQuantileHead, FitConfig


class XGBQuantileHead(BaseQuantileHead):
    """
    四模型分位数回归头（XGBoost 后端）

    使用 XGBoost 2.0+ 的 reg:quantileerror 目标函数。
    训练四个独立模型：MSE / q10 / q50 / q90。
    """

    backend_name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators        = n_estimators
        self.learning_rate       = learning_rate
        self.max_depth           = max_depth
        self.subsample           = subsample
        self.colsample_bytree    = colsample_bytree
        self.reg_alpha           = reg_alpha
        self.reg_lambda          = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state        = random_state
        self.n_jobs              = n_jobs

        self.models    = {"mse": None, "q10": None, "q50": None, "q90": None}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def _base_params(self) -> Dict:
        return dict(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            max_depth         = self.max_depth,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample_bytree,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
            random_state      = self.random_state,
            n_jobs            = self.n_jobs,
            verbosity         = 0,
            early_stopping_rounds = self.early_stopping_rounds,
        )

    def _make_model(self, objective: str, quantile_alpha: Optional[float] = None):
        params = self._base_params()
        params["objective"] = objective
        if quantile_alpha is not None:
            params["quantile_alpha"] = quantile_alpha
        return xgb.XGBRegressor(**params)

    # ------------------------------------------------------------------
    def fit(self, config: FitConfig) -> None:
        X_tr, y_tr = config.X_train, config.y_train
        eval_set   = [(config.X_val, config.y_val)] if config.X_val is not None else None
        verbose    = False

        print("Training XGB MSE predictor...")
        self.models["mse"] = self._make_model("reg:squarederror")
        self.models["mse"].fit(X_tr, y_tr, eval_set=eval_set, verbose=verbose)

        print("Training XGB q10 quantile regressor...")
        self.models["q10"] = self._make_model("reg:quantileerror", 0.1)
        self.models["q10"].fit(X_tr, y_tr, eval_set=eval_set, verbose=verbose)

        print("Training XGB q50 quantile regressor (anchor)...")
        self.models["q50"] = self._make_model("reg:quantileerror", 0.5)
        self.models["q50"].fit(X_tr, y_tr, eval_set=eval_set, verbose=verbose)

        print("Training XGB q90 quantile regressor...")
        self.models["q90"] = self._make_model("reg:quantileerror", 0.9)
        self.models["q90"].fit(X_tr, y_tr, eval_set=eval_set, verbose=verbose)

        self.is_fitted = True
        print("All XGBoost models trained successfully!")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("XGBQuantileHead: 模型未训练，请先调用 fit()")
        return {
            "point" : self.models["mse"].predict(X),
            "q10"   : self.models["q10"].predict(X),
            "q50"   : self.models["q50"].predict(X),
            "q90"   : self.models["q90"].predict(X),
        }
