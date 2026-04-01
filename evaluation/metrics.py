"""
metrics.py - 评估指标

实现以下评估指标：
1. Coverage Error (CE)
2. Winkler Score
3. Interval Width (IW)
4. Mean Absolute Error (MAE)
5. RankIC
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import pandas as pd


def coverage_error(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.8,
) -> float:
    """
    计算Coverage Error

    CE = |(1/N) * Σ I[q10 ≤ y ≤ q90] - 0.8|

    Args:
        y_true: 真实值
        lower: 区间下界
        upper: 区间上界
        target_coverage: 目标覆盖率（默认0.8）

    Returns:
        Coverage Error值
    """
    # 检查是否在区间内
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)

    return np.abs(coverage - target_coverage)


def winkler_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2
) -> float:
    """
    计算Winkler Score

    W_t = {
        (q90 - q10) + 10 * (q10 - y)   if y < q10
        (q90 - q10)                     if q10 ≤ y ≤ q90
        (q90 - q10) + 10 * (y - q90)   if y > q90
    }

    Args:
        y_true: 真实值
        lower: 区间下界
        upper: 区间上界
        alpha: 显著性水平（默认0.2，对应80%置信区间）

    Returns:
        平均Winkler Score
    """
    # 区间宽度
    width = upper - lower

    # 惩罚项
    penalty_lower = np.where(y_true < lower, (1 / alpha) * (lower - y_true), 0)
    penalty_upper = np.where(y_true > upper, (1 / alpha) * (y_true - upper), 0)

    # 总分
    winkler = width + penalty_lower + penalty_upper

    return np.mean(winkler)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    计算平均区间宽度

    IW = (1/N) * Σ (q90 - q10)

    Args:
        lower: 区间下界
        upper: 区间上界

    Returns:
        平均区间宽度
    """
    width = upper - lower
    return np.mean(width)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差

    MAE = (1/N) * Σ |ŷ* - y|

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MAE值
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差

    MSE = (1/N) * Σ (ŷ* - y)²

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        MSE值
    """
    return np.mean((y_true - y_pred) ** 2)


def rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算RankIC（Spearman秩相关系数）

    RankIC = Spearman(rank(ŷ*), rank(y))

    Args:
        y_pred: 预测值
        y_true: 真实值

    Returns:
        RankIC值
    """
    return stats.spearmanr(y_pred, y_true).correlation


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """
    计算Pinball Loss

    L_q(y, ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))

    Args:
        y_true: 真实值
        y_pred: 预测值
        alpha: 分位数参数

    Returns:
        平均Pinball Loss
    """
    residual = y_true - y_pred
    loss = np.maximum(alpha * residual, (alpha - 1) * residual)
    return np.mean(loss)


def crossing_rate(q10: np.ndarray, q90: np.ndarray) -> float:
    """
    计算分位数交叉率

    如果分位数交叉（q10 > q90），说明分位数估计质量有限

    Args:
        q10: 10%分位数预测
        q90: 90%分位数预测

    Returns:
        交叉率（0-1之间）
    """
    crossing_mask = q10 > q90
    crossing_count = np.sum(crossing_mask)
    total_count = len(q10)

    return crossing_count / total_count if total_count > 0 else 0.0


def composite_score(
    winkler_mean: float, coverage_error: float, target_ce: float = 0.05
) -> float:
    """
    计算复合评分指标

    Score = W̄ + 10 * max(0, CE - 0.05)

    Args:
        winkler_mean: 平均Winkler Score
        coverage_error: Coverage Error
        target_ce: 目标Coverage Error

    Returns:
        复合评分
    """
    return winkler_mean + 10 * max(0, coverage_error - target_ce)


def calculate_all_metrics(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q10: Optional[np.ndarray] = None,
    q90: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        y_true: 真实值
        point_pred: 点预测值
        lower: 区间下界
        upper: 区间上界
        q10: 10%分位数（可选，用于交叉率计算）
        q90: 90%分位数（可选，用于交叉率计算）

    Returns:
        所有指标的字典
    """
    metrics = {}

    # 区间质量指标
    metrics["coverage_error"] = coverage_error(y_true, lower, upper)
    metrics["winkler_score"] = winkler_score(y_true, lower, upper)
    metrics["interval_width"] = interval_width(lower, upper)

    # 点预测质量指标
    metrics["mae"] = mean_absolute_error(y_true, point_pred)
    metrics["mse"] = mean_squared_error(y_true, point_pred)
    metrics["rank_ic"] = rank_ic(point_pred, y_true)

    # 交叉率（如果提供了分位数）
    if q10 is not None and q90 is not None:
        metrics["crossing_rate"] = crossing_rate(q10, q90)

    # 复合评分
    metrics["composite_score"] = composite_score(
        metrics["winkler_score"], metrics["coverage_error"]
    )

    return metrics


def paired_t_test(
    metric_values_a: np.ndarray, metric_values_b: np.ndarray
) -> Tuple[float, float]:
    """
    配对t检验

    t = d̄ / (s_d / √n)
    d_t = W_t^Snail - W_t^CP

    Args:
        metric_values_a: 方法A的指标值（如Snail）
        metric_values_b: 方法B的指标值（如CP）

    Returns:
        (t统计量, p值) 元组
    """
    differences = metric_values_a - metric_values_b
    t_stat, p_value = stats.ttest_1samp(differences, 0)

    return t_stat, p_value


def evaluate_methods(
    y_true: np.ndarray,
    methods_predictions: Dict[str, Dict],
    include_crossing: bool = True,
) -> pd.DataFrame:
    """
    评估多个方法

    Args:
        y_true: 真实值
        methods_predictions: 方法名到预测结果的映射
            每个方法包含：point_pred, lower, upper, q10, q90
        include_crossing: 是否包含交叉率

    Returns:
        评估结果DataFrame
    """
    results = []

    for method_name, predictions in methods_predictions.items():
        point_pred = predictions["point_pred"]
        lower = predictions["lower"]
        upper = predictions["upper"]
        q10 = predictions.get("q10", None)
        q90 = predictions.get("q90", None)

        # 计算指标
        metrics = calculate_all_metrics(y_true, point_pred, lower, upper, q10, q90)

        # 添加方法名
        result = {"Method": method_name}
        result.update(metrics)

        results.append(result)

    # 创建DataFrame
    df_results = pd.DataFrame(results)

    # 如果不需要交叉率，移除该列
    if not include_crossing and "crossing_rate" in df_results.columns:
        df_results = df_results.drop(columns=["crossing_rate"])

    return df_results


if __name__ == "__main__":
    # 测试评估指标
    print("Testing evaluation metrics...")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randn(n_samples)
    point_pred = y_true + np.random.randn(n_samples) * 0.1

    # 模拟区间预测
    lower = point_pred - 0.5 + np.random.randn(n_samples) * 0.1
    upper = point_pred + 0.5 + np.random.randn(n_samples) * 0.1

    # 确保lower < upper
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

    # 计算所有指标
    metrics = calculate_all_metrics(y_true, point_pred, lower, upper)

    print("\nAll metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # 测试Winkler Score计算
    print(f"\nWinkler Score details:")
    ws = winkler_score(y_true, lower, upper)
    print(f"  Average Winkler Score: {ws:.4f}")

    # 测试Coverage Error
    ce = coverage_error(y_true, lower, upper)
    print(f"  Coverage Error: {ce:.4f}")

    # 计算实际覆盖率
    in_interval = (y_true >= lower) & (y_true <= upper)
    actual_coverage = np.mean(in_interval)
    print(f"  Actual coverage: {actual_coverage:.4f}")

    print("\nMetrics testing completed!")
