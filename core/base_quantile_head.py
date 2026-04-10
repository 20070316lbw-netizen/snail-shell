"""
base_quantile_head.py - 分位数回归头抽象基类

定义所有后端（LightGBM / XGBoost / CatBoost / 神经网络）必须实现的接口，
使 Snail 机制与底层模型完全解耦。

接口约定：
  fit(config: FitConfig)            训练四个分位数模型
  predict(X) -> Dict                返回 point / q10 / q50 / q90
  predict_anchor_and_radius(X)      返回 (a_t, r_t)，供 SnailMechanism 使用
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# FitConfig：训练配置（所有后端共用）
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """训练配置"""
    X_train   : np.ndarray
    y_train   : np.ndarray
    X_val     : Optional[np.ndarray] = None
    y_val     : Optional[np.ndarray] = None
    callbacks : Optional[list] = None       # 仅 LightGBM 使用


# ---------------------------------------------------------------------------
# BaseQuantileHead：抽象基类
# ---------------------------------------------------------------------------

class BaseQuantileHead(ABC):
    """
    分位数回归头抽象基类

    子类必须实现 fit() 和 predict()。
    predict_anchor_and_radius() 有默认实现，子类可覆盖。
    """

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, config: FitConfig) -> None:
        """
        训练四个分位数模型：MSE / q10 / q50 / q90

        参数:
            config: FitConfig 实例，包含训练集和可选验证集
        """
        ...

    # ------------------------------------------------------------------
    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        推断所有分位数和点预测

        参数:
            X: 特征矩阵，形状 (N, F)

        返回:
            字典，必须包含以下键：
              "point" : MSE 点预测，形状 (N,)
              "q10"   : 10% 分位数，形状 (N,)
              "q50"   : 50% 分位数（锚点），形状 (N,)
              "q90"   : 90% 分位数，形状 (N,)
        """
        ...

    # ------------------------------------------------------------------
    def predict_anchor_and_radius(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测锚点和对称半径（供对称 SnailMechanism 使用）

        a_t = q̂_{50,t}
        r_t = (q̂_{90,t} - q̂_{10,t}) / 2

        返回:
            (anchor, radius)，各形状 (N,)
        """
        preds  = self.predict(X)
        anchor = preds["q50"]
        radius = (preds["q90"] - preds["q10"]) / 2
        return anchor, radius

    # ------------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        """后端名称，用于结果表格标注（子类可覆盖）"""
        return self.__class__.__name__
