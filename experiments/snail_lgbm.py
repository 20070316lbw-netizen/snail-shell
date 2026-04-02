"""
snail_lgbm.py - Snail-0.5/1/2/5 实验

实现蜗牛壳变体实验：
- Snail-0.5: β=0.5（温和拉回）
- Snail-1: β=1.0（标准拉回）
- Snail-2: β=2.0（强拉回）
- Snail-5: β=5.0（近似截断）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.quantile_head import QuantileHead, FitConfig
from core.snail_mechanism import SnailMechanism
from core.experiment_data import ExperimentData
from evaluation.metrics import (
    coverage_error,
    winkler_score,
    interval_width,
    mean_absolute_error,
    actual_coverage,
)


class SnailModel:
    """
    蜗牛壳模型

    结合分位数回归头和软拉回机制
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        """
        初始化蜗牛壳模型

        Args:
            beta: 软拉回参数
            n_estimators: 树的数量
            learning_rate: 学习率
            early_stopping_rounds: 早停轮数
            random_state: 随机种子
        """
        self.beta = beta

        # 分位数回归头
        self.quantile_head = QuantileHead(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )

        # 蜗牛壳机制
        self.snail_mechanism = SnailMechanism()

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        训练蜗牛壳模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        # 训练分位数回归头
        config = FitConfig(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        self.quantile_head.fit(config)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测并应用软拉回

        Args:
            X: 特征矩阵

        Returns:
            包含所有预测的字典
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 获取分位数回归预测
        predictions = self.quantile_head.predict(X)

        # 获取锚点和半径
        anchor, radius = self.quantile_head.predict_anchor_and_radius(X)

        # 应用软拉回（只调用一次）
        point_pred = predictions["point"]
        corrected_pred, corrected_q10, corrected_q90, diagnostics = (
            self.snail_mechanism.apply(point_pred, anchor, radius, self.beta)
        )

        # 合并结果
        result = {
            "original_point": point_pred,
            "corrected_point": corrected_pred,
            "anchor": anchor,
            "radius": radius,
            "q10": corrected_q10,  # ✅ 修正后的区间下界（以 corrected_pred 为中心，半宽 = radius）
            "q90": corrected_q90,  # ✅ 修正后的区间上界
            "raw_q10": predictions["q10"],  # 原始分位数（保留备查）
            "raw_q90": predictions["q90"],
            "alpha": diagnostics["mean_alpha"],
            "diagnostics": diagnostics,
        }

        return result

    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测点预测和区间

        Args:
            X: 特征矩阵

        Returns:
            (修正后的点预测, 下界, 上界) 元组
        """
        predictions = self.predict(X)

        point_pred = predictions["corrected_point"]
        lower = predictions["q10"]
        upper = predictions["q90"]

        return point_pred, lower, upper


def run_snail_experiment(
    data: ExperimentData, beta_values: Optional[List[float]] = None
) -> Dict:
    """
    运行蜗牛壳实验

    Args:
        data: 实验数据
        beta_values: 实验用的beta列表

    Returns:
        实验结果字典
    """
    if beta_values is None:
        beta_values = [0.5, 1.0, 2.0, 5.0]

    results = {}

    for beta in beta_values:
        print(f"Running Snail-β={beta} experiment...")

        # 创建并训练模型
        model = SnailModel(beta=beta)
        model.fit(data.X_train, data.y_train, data.X_val, data.y_val)

        # 预测
        predictions = model.predict(data.X_test)
        point_pred = predictions["corrected_point"]
        lower = predictions["q10"]
        upper = predictions["q90"]

        # 存储结果
        method_name = f"Snail-{beta}"
        result_dict = {
            "point_pred": point_pred,
            "lower": lower,
            "upper": upper,
            "interval": (lower, upper),
            "diagnostics": predictions["diagnostics"],
        }

        if data.X_val_q2_4 is not None and data.y_val_q2_4 is not None:
            v_predictions = model.predict(data.X_val_q2_4)
            result_dict["val_q2_4"] = {
                "point_pred": v_predictions["corrected_point"],
                "lower": v_predictions["q10"],
                "upper": v_predictions["q90"],
                "interval": (v_predictions["q10"], v_predictions["q90"]),
                "diagnostics": v_predictions["diagnostics"],
            }

        results[method_name] = result_dict

    return results


def select_best_beta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    beta_candidates: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """
    在验证集上选择最优β值

    使用复合指标：
    Score = W̄ + 10 * max(0, CE - 0.05)

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        beta_candidates: β候选值

    Returns:
        (最优β值, 所有β的评分结果) 元组
    """
    if beta_candidates is None:
        beta_candidates = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]

    # 先训练分位数回归头（所有β共享）
    qh = QuantileHead()
    config = FitConfig(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    qh.fit(config)

    # 获取锚点和半径
    anchor, radius = qh.predict_anchor_and_radius(X_val)

    # 获取点预测（MSE模型）
    point_pred = qh.models["mse"].predict(X_val)

    # 创建蜗牛壳机制
    snail = SnailMechanism(beta_values=beta_candidates)

    # 选择最优β
    best_beta, beta_scores = snail.select_beta(point_pred, anchor, radius, y_val)

    return best_beta, beta_scores


def compare_snail_variants(data: ExperimentData) -> pd.DataFrame:
    """
    比较不同蜗牛壳变体

    Args:
        data: 实验数据

    Returns:
        比较结果DataFrame
    """
    beta_values = [0.5, 1.0, 2.0, 5.0]

    results = run_snail_experiment(data, beta_values=beta_values)

    # 收集结果
    comparison_data = []

    for method_name, result in results.items():
        point_pred = result["point_pred"]
        lower = result["lower"]
        upper = result["upper"]
        diagnostics = result["diagnostics"]

        # 计算评估指标
        ce = coverage_error(y_test, lower, upper)
        ws = winkler_score(y_test, lower, upper)
        iw = interval_width(lower, upper)
        mae = mean_absolute_error(y_test, point_pred)

        comparison_data.append(
            {
                "Method": method_name,
                "Beta": diagnostics["beta"],
                "MAE": mae,
                "Coverage Error": ce,
                "Winkler Score": ws,
                "Interval Width": iw,
                "Mean Alpha": diagnostics["mean_alpha"],
                "Mean Correction": diagnostics["mean_correction"],
            }
        )

    # 创建DataFrame
    df_comparison = pd.DataFrame(comparison_data)

    return df_comparison


if __name__ == "__main__":
    # 测试蜗牛壳实验
    print("Testing Snail experiments...")

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

    # 运行蜗牛壳实验
    print("\nRunning Snail experiments...")
    data = ExperimentData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    results = run_snail_experiment(data)

    # 评估结果
    print("\nSnail Experiment Results:")
    print("=" * 80)

    for method_name, result in results.items():
        point_pred = result["point_pred"]
        lower = result["lower"]
        upper = result["upper"]
        diagnostics = result["diagnostics"]

        # 计算评估指标
        ce = coverage_error(y_test, lower, upper)
        ac = actual_coverage(y_test, lower, upper)
        ws = winkler_score(y_test, lower, upper)
        iw = interval_width(lower, upper)
        mae = mean_absolute_error(y_test, point_pred)

        print(f"\n{method_name}:")
        print(f"  Beta:            {diagnostics['beta']}")
        print(f"  Mean Alpha:      {diagnostics['mean_alpha']:.4f}")
        print(f"  Actual Coverage: {ac:.4f}")
        print(f"  Coverage Error:  {ce:.4f}")
        print(f"  Winkler Score:   {ws:.4f}")
        print(f"  Interval Width:  {iw:.4f}")
        print(f"  MAE:             {mae:.4f}")

    # 选择最优β
    print("\n" + "=" * 80)
    print("Selecting best beta on validation set...")
    best_beta, beta_scores = select_best_beta(X_train, y_train, X_val, y_val)
    print(f"Best beta: {best_beta}")

    # 显示所有β的评分
    print("\nAll beta scores:")
    for beta, score_info in beta_scores.items():
        metrics = score_info["metrics"]
        print(
            f"  β={beta}: Score={metrics['score']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"Winkler={metrics['winkler_mean']:.4f}"
        )

    print("\nSnail experiments completed!")
