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
from typing import Dict, Optional, Tuple
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

    参数:
        y_true: 真实值
        lower: 区间下界
        upper: 区间上界
        target_coverage: 目标覆盖率（默认0.8）

    返回:
        Coverage Error值
    """
    # 检查是否在区间内
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)

    return np.abs(coverage - target_coverage)


def actual_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    计算实际覆盖率

    PICP = (1/N) * Σ I[lower ≤ y_true ≤ upper]

    参数:
        y_true: 真实值
        lower: 区间下界
        upper: 区间上界

    返回:
        实际覆盖率
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(in_interval)


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

    参数:
        y_true: 真实值
        lower: 区间下界
        upper: 区间上界
        alpha: 显著性水平（默认0.2，对应80%置信区间）

    返回:
        平均Winkler Score
    """
    # 区间宽度
    width = upper - lower

    # 惩罚项
    penalty_lower = np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0)
    penalty_upper = np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0)

    # 总分
    winkler = width + penalty_lower + penalty_upper

    return np.mean(winkler)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    计算平均区间宽度

    IW = (1/N) * Σ (q90 - q10)

    参数:
        lower: 区间下界
        upper: 区间上界

    返回:
        平均区间宽度
    """
    width = upper - lower
    return np.mean(width)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差

    MAE = (1/N) * Σ |ŷ* - y|

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        MAE值
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差

    MSE = (1/N) * Σ (ŷ* - y)²

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        MSE值
    """
    return np.mean((y_true - y_pred) ** 2)


def rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算RankIC（Spearman秩相关系数）

    RankIC = Spearman(rank(ŷ*), rank(y))

    参数:
        y_pred: 预测值
        y_true: 真实值

    返回:
        RankIC值
    """
    return stats.spearmanr(y_pred, y_true).correlation


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """
    计算Pinball Loss

    L_q(y, ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))

    参数:
        y_true: 真实值
        y_pred: 预测值
        alpha: 分位数参数

    返回:
        平均Pinball Loss
    """
    residual = y_true - y_pred
    loss = np.maximum(alpha * residual, (alpha - 1) * residual)
    return np.mean(loss)


def crossing_rate(q10: np.ndarray, q90: np.ndarray) -> float:
    """
    计算分位数交叉率

    如果分位数交叉（q10 > q90），说明分位数估计质量有限

    参数:
        q10: 10%分位数预测
        q90: 90%分位数预测

    返回:
        交叉率（0-1之间）
    """
    crossing_mask = q10 > q90
    crossing_count = np.sum(crossing_mask)
    total_count = len(q10)

    return crossing_count / total_count if total_count > 0 else 0.0


def composite_score(
    winkler_mean: float, coverage_error: float, k: float = 2.0
) -> float:
    """
    计算复合评分指标

    Score = W̄ + k × CE

    连续线性惩罚，CE 从 0 开始计，CE 越小得分越低（越好）。
    相比旧版（CE ≤ 0.05 无奖励的阶梯式惩罚），此版本持续奖励
    覆盖率向目标 80% 的靠近，使 AS-GSPQR 的 CE 改善能够体现
    在最终评分上。

    参数:
        winkler_mean: 平均 Winkler Score
        coverage_error: Coverage Error（= |实际覆盖率 - 0.8|）
        k: CE 惩罚系数，默认 2.0（可在验证集上调参）

    返回:
        复合评分（越小越好）
    """
    return winkler_mean + k * coverage_error


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

    参数:
        y_true: 真实值
        point_pred: 点预测值
        lower: 区间下界
        upper: 区间上界
        q10: 10%分位数（可选，用于交叉率计算）
        q90: 90%分位数（可选，用于交叉率计算）

    返回:
        所有指标的字典
    """
    metrics = {}

    # 区间质量指标
    metrics["actual_coverage"] = actual_coverage(y_true, lower, upper)
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

    参数:
        metric_values_a: 方法A的指标值（如Snail）
        metric_values_b: 方法B的指标值（如CP）

    返回:
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

    参数:
        y_true: 真实值
        methods_predictions: 方法名到预测结果的映射
            每个方法包含：point_pred, lower, upper, q10, q90
        include_crossing: 是否包含交叉率

    返回:
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
    print("\nWinkler Score details:")
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


# ===========================================================================
# AS-GSPQR 非对称区间专用指标
# ===========================================================================

def skewness_index(
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    区间偏斜指数

    SI = mean( (r_up - r_down) / (r_up + r_down) )

    范围 [-1, 1]：
      SI > 0 → 区间整体偏向上行（r_up > r_down），对应右偏分布
      SI < 0 → 区间整体偏向下行（r_down > r_up），对应左偏分布
      SI ≈ 0 → 近似对称（等价于 GSPQR 的对称假设）

    参数:
        lower  : 区间下界，形状 (N,)
        upper  : 区间上界，形状 (N,)
        center : 修正中心 ŷ*_t，形状 (N,)
        epsilon: 防零除小量

    返回:
        平均偏斜指数（标量）
    """
    r_down = center - lower
    r_up   = upper - center
    denom  = np.maximum(r_up + r_down, epsilon)
    return float(np.mean((r_up - r_down) / denom))


def asymmetric_width_stats(
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    非对称区间宽度统计

    返回上/下半宽的均值、标准差及偏斜指数，便于分析非对称程度。

    参数:
        lower  : 区间下界，形状 (N,)
        upper  : 区间上界，形状 (N,)
        center : 修正中心，形状 (N,)
        epsilon: 防零除小量

    返回:
        字典，包含：
          mean_r_down, std_r_down  — 下行半宽统计
          mean_r_up,   std_r_up    — 上行半宽统计
          skewness_index           — 偏斜指数
          asymmetry_ratio          — r_up / r_down 的均值（>1 偏上，<1 偏下）
    """
    r_down = center - lower
    r_up   = upper - center
    denom  = np.maximum(r_up + r_down, epsilon)

    return {
        "mean_r_down"     : float(np.mean(r_down)),
        "std_r_down"      : float(np.std(r_down)),
        "mean_r_up"       : float(np.mean(r_up)),
        "std_r_up"        : float(np.std(r_up)),
        "skewness_index"  : float(np.mean((r_up - r_down) / denom)),
        "asymmetry_ratio" : float(np.mean(r_up / np.maximum(r_down, epsilon))),
    }


def calculate_all_metrics_asymmetric(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算所有指标（含非对称专用）

    在 calculate_all_metrics 基础上追加 skewness_index 和非对称宽度统计。

    参数:
        y_true     : 真实值
        point_pred : 点预测（修正后中心）
        lower      : 区间下界
        upper      : 区间上界
        center     : 修正中心（若为 None 则用 point_pred 代替）

    返回:
        指标字典
    """
    if center is None:
        center = point_pred

    metrics = calculate_all_metrics(y_true, point_pred, lower, upper)

    # 追加非对称指标
    asym_stats = asymmetric_width_stats(lower, upper, center)
    metrics.update(asym_stats)

    return metrics
