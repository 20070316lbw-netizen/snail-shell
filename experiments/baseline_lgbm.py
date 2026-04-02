"""
baseline_lgbm.py - 基线实验（Residual / CP / QR / Q50-only）

实现以下基线方法：
1. Residual: 点预测 ± 1.28σ残差
2. CP: Conformal Prediction (EnbPI)
3. QR: 分位数回归无拉回（β=0）
4. Q50-only: 直接输出a_t（β=∞）
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import lightgbm as lgb
from scipy import stats
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantile_head import QuantileHead
from evaluation.metrics import (
    coverage_error,
    winkler_score,
    interval_width,
    mean_absolute_error,
)


class ResidualBaseline:
    """
    Residual基线方法

    区间构造: 点预测 ± 1.28σ残差
    点预测: MSE LightGBM
    """

    def __init__(
        self,
        quantile: float = 0.9,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
    ):
        """
        初始化Residual基线

        Args:
            quantile: 分位数（默认1.28对应80%置信区间）
            n_estimators: 树的数量
            learning_rate: 学习率
        """
        self.quantile = quantile
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            verbose=-1,
        )
        self.residual_std = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        训练模型并计算残差标准差

        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        # 训练模型
        self.model.fit(X_train, y_train)

        # 计算训练集残差
        y_pred = self.model.predict(X_train)
        residuals = y_train - y_pred
        self.residual_std = np.std(residuals)

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测点预测和区间

        Args:
            X: 特征矩阵

        Returns:
            (点预测, 下界, 上界) 元组
        """
        point_pred = self.model.predict(X)

        # 计算区间
        z_score = stats.norm.ppf((1 + self.quantile) / 2)
        half_width = z_score * self.residual_std

        lower = point_pred - half_width
        upper = point_pred + half_width

        return point_pred, lower, upper


class ConformalPredictionBaseline:
    """
    Conformal Prediction基线（EnbPI风格）

    使用q50作为中心预测，区间基于校准集残差分位数
    """

    def __init__(self, alpha: float = 0.2, calibration_split: int = None):
        """
        初始化CP基线

        Args:
            alpha: 显著性水平（1-alpha=置信水平）
            calibration_split: 校准集大小（可选）
        """
        self.alpha = alpha
        self.calibration_split = calibration_split

        # 使用q50分位数回归
        self.q50_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )

        self.residual_quantile = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> None:
        """
        训练模型并在校准集上计算残差分位数

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_cal: 校准特征
            y_cal: 校准标签
        """
        # 训练q50模型
        self.q50_model.fit(X_train, y_train)

        # 如果有校准集，计算残差分位数
        if X_cal is not None and y_cal is not None:
            y_pred_cal = self.q50_model.predict(X_cal)
            residuals = np.abs(y_cal - y_pred_cal)
            self.residual_quantile = np.quantile(residuals, 1 - self.alpha)
        else:
            # 使用训练集残差（不推荐，但作为备选）
            y_pred_train = self.q50_model.predict(X_train)
            residuals = np.abs(y_train - y_pred_train)
            self.residual_quantile = np.quantile(residuals, 1 - self.alpha)

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测点预测和区间

        Args:
            X: 特征矩阵

        Returns:
            (点预测, 下界, 上界) 元组
        """
        point_pred = self.q50_model.predict(X)

        lower = point_pred - self.residual_quantile
        upper = point_pred + self.residual_quantile

        return point_pred, lower, upper


class QRBaseline:
    """
    分位数回归基线（无拉回，β=0）

    使用q10和q90直接构造区间
    """

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.05):
        """
        初始化QR基线

        Args:
            n_estimators: 树的数量
            learning_rate: 学习率
        """
        self.quantile_head = QuantileHead(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        训练四模型分位数回归

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        self.quantile_head.fit(X_train, y_train, X_val, y_val)

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测点预测和区间

        Args:
            X: 特征矩阵

        Returns:
            (点预测, 下界, 上界) 元组
        """
        predictions = self.quantile_head.predict(X)

        # 使用q50作为点预测
        point_pred = predictions["q50"]
        lower = predictions["q10"]
        upper = predictions["q90"]

        return point_pred, lower, upper


class Q50OnlyBaseline:
    """
    Q50-only基线（β=∞，完全拉回）

    直接输出q50作为点预测，区间由q10和q90给出
    但点预测就是q50，所以是退化的完全拉回
    """

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.05):
        """
        初始化Q50-only基线

        Args:
            n_estimators: 树的数量
            learning_rate: 学习率
        """
        self.quantile_head = QuantileHead(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        训练四模型分位数回归

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        self.quantile_head.fit(X_train, y_train, X_val, y_val)

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测点预测和区间

        点预测直接使用q50
        """
        predictions = self.quantile_head.predict(X)

        # 点预测就是q50
        point_pred = predictions["q50"]
        lower = predictions["q10"]
        upper = predictions["q90"]

        return point_pred, lower, upper


def run_baseline_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_methods: Optional[list] = None,
    X_val_q2_4: Optional[np.ndarray] = None,
    y_val_q2_4: Optional[np.ndarray] = None,
) -> Dict:
    """
    运行所有基线实验

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        X_test: 测试特征
        y_test: 测试标签
        baseline_methods: 基线方法列表

    Returns:
        实验结果字典
    """
    if baseline_methods is None:
        baseline_methods = ["residual", "cp", "qr", "q50_only"]

    results = {}

    # Residual基线
    if "residual" in baseline_methods:
        print("Running Residual baseline...")
        residual_model = ResidualBaseline()
        residual_model.fit(X_train, y_train)
        point_pred, lower, upper = residual_model.predict_interval(X_test)

        result_dict = {
            "point_pred": point_pred,
            "lower": lower,
            "upper": upper,
            "interval": (lower, upper),
        }

        if X_val_q2_4 is not None and y_val_q2_4 is not None:
            v_point_pred, v_lower, v_upper = residual_model.predict_interval(X_val_q2_4)
            result_dict["val_q2_4"] = {
                "point_pred": v_point_pred,
                "lower": v_lower,
                "upper": v_upper,
                "interval": (v_lower, v_upper),
            }

        results["Residual"] = result_dict

    # Conformal Prediction基线
    if "cp" in baseline_methods:
        print("Running Conformal Prediction baseline...")
        cp_model = ConformalPredictionBaseline()
        cp_model.fit(X_train, y_train, X_val, y_val)
        point_pred, lower, upper = cp_model.predict_interval(X_test)

        result_dict = {
            "point_pred": point_pred,
            "lower": lower,
            "upper": upper,
            "interval": (lower, upper),
        }

        if X_val_q2_4 is not None and y_val_q2_4 is not None:
            v_point_pred, v_lower, v_upper = cp_model.predict_interval(X_val_q2_4)
            result_dict["val_q2_4"] = {
                "point_pred": v_point_pred,
                "lower": v_lower,
                "upper": v_upper,
                "interval": (v_lower, v_upper),
            }

        results["CP"] = result_dict

    # 分位数回归基线（β=0）
    if "qr" in baseline_methods:
        print("Running QR baseline (β=0)...")
        qr_model = QRBaseline()
        qr_model.fit(X_train, y_train, X_val, y_val)
        point_pred, lower, upper = qr_model.predict_interval(X_test)

        result_dict = {
            "point_pred": point_pred,
            "lower": lower,
            "upper": upper,
            "interval": (lower, upper),
        }

        if X_val_q2_4 is not None and y_val_q2_4 is not None:
            v_point_pred, v_lower, v_upper = qr_model.predict_interval(X_val_q2_4)
            result_dict["val_q2_4"] = {
                "point_pred": v_point_pred,
                "lower": v_lower,
                "upper": v_upper,
                "interval": (v_lower, v_upper),
            }

        results["QR"] = result_dict

    # Q50-only基线（β=∞）
    if "q50_only" in baseline_methods:
        print("Running Q50-only baseline (β=∞)...")
        q50_model = Q50OnlyBaseline()
        q50_model.fit(X_train, y_train, X_val, y_val)
        point_pred, lower, upper = q50_model.predict_interval(X_test)

        result_dict = {
            "point_pred": point_pred,
            "lower": lower,
            "upper": upper,
            "interval": (lower, upper),
        }

        if X_val_q2_4 is not None and y_val_q2_4 is not None:
            v_point_pred, v_lower, v_upper = q50_model.predict_interval(X_val_q2_4)
            result_dict["val_q2_4"] = {
                "point_pred": v_point_pred,
                "lower": v_lower,
                "upper": v_upper,
                "interval": (v_lower, v_upper),
            }

        results["Q50-only"] = result_dict

    return results


if __name__ == "__main__":
    # 测试基线实验
    print("Testing baseline experiments...")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 2000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5

    # 分割数据集
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 运行基线实验
    results = run_baseline_experiment(X_train, y_train, X_val, y_val, X_test, y_test)

    # 评估结果
    print("\nBaseline Results:")
    print("=" * 80)

    for method_name, result in results.items():
        point_pred = result["point_pred"]
        lower = result["lower"]
        upper = result["upper"]

        # 计算评估指标
        ce = coverage_error(y_test, lower, upper)
        ws = winkler_score(y_test, lower, upper)
        iw = interval_width(lower, upper)
        mae = mean_absolute_error(y_test, point_pred)

        print(f"\n{method_name}:")
        print(f"  Coverage Error: {ce:.4f}")
        print(f"  Winkler Score:  {ws:.4f}")
        print(f"  Interval Width: {iw:.4f}")
        print(f"  MAE:            {mae:.4f}")

    print("\nBaseline experiments completed!")
