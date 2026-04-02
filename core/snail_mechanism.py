"""
snail_mechanism.py - 可信圆 + 软拉回机制（β 控制）

核心创新：软拉回机制
用分位数回归输出的不确定性半径，动态约束点预测的偏离程度，
防止模型在高波动、regime 切换时产生极端预测。

公式:
α_t = exp(-β * |ŷ_t - a_t| / r_t)
ŷ_t* = α_t * ŷ_t + (1 - α_t) * a_t
"""

import numpy as np
from typing import Tuple, Dict, Optional


def soft_pullback(
    point_pred: np.ndarray, anchor: np.ndarray, radius: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """
    软拉回机制

    公式:
    α_t = exp(-β * |ŷ_t - a_t| / r_t)
    ŷ_t* = α_t * ŷ_t + (1 - α_t) * a_t

    Args:
        point_pred: 点预测值 (ŷ_t)
        anchor: 锚点值 (a_t)
        radius: 可信圆半径 (r_t)
        beta: 插值控制器（控制拉回强度）

    Returns:
        修正后的预测值 (ŷ_t*)
    """
    # 防止除零
    epsilon = 1e-8
    radius_safe = np.maximum(radius, epsilon)

    # 计算偏离度
    deviation = np.abs(point_pred - anchor) / radius_safe

    # 计算α_t
    alpha = np.exp(-beta * deviation)

    # 软拉回
    corrected_pred = alpha * point_pred + (1 - alpha) * anchor

    return corrected_pred


def calculate_alpha(
    point_pred: np.ndarray, anchor: np.ndarray, radius: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """
    计算α_t值

    Args:
        point_pred: 点预测值
        anchor: 锚点值
        radius: 可信圆半径
        beta: 插值控制器

    Returns:
        α_t值
    """
    epsilon = 1e-8
    radius_safe = np.maximum(radius, epsilon)
    deviation = np.abs(point_pred - anchor) / radius_safe
    alpha = np.exp(-beta * deviation)

    return alpha


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


class SnailMechanism:
    """
    蜗牛壳机制（软拉回机制完整实现）

    包含：
    1. 可信圆计算
    2. 软拉回修正
    3. β参数扫描
    4. 交叉率监控
    """

    def __init__(
        self, beta_values: Optional[list] = None, crossing_threshold: float = 0.1
    ):
        """
        初始化蜗牛壳机制

        Args:
            beta_values: β参数列表，用于扫描
            crossing_threshold: 交叉率阈值警告线
        """
        if beta_values is None:
            # 默认β值：0, 0.5, 1, 2, 5, inf
            self.beta_values = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
        else:
            self.beta_values = beta_values

        self.crossing_threshold = crossing_threshold
        self.results = {}

    def apply(
        self,
        point_pred: np.ndarray,
        anchor: np.ndarray,
        radius: np.ndarray,
        beta: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        应用软拉回机制

        Args:
            point_pred: 点预测值
            anchor: 锚点值
            radius: 可信圆半径
            beta: 插值控制器

        Returns:
            (修正后的预测, 诊断信息) 元组
        """
        # 计算α值
        alpha = calculate_alpha(point_pred, anchor, radius, beta)

        # 软拉回
        corrected_pred = soft_pullback(point_pred, anchor, radius, beta)

        # 计算统计信息
        diagnostics = {
            "beta": beta,
            "mean_alpha": np.mean(alpha),
            "std_alpha": np.std(alpha),
            "min_alpha": np.min(alpha),
            "max_alpha": np.max(alpha),
            "mean_deviation": np.mean(
                np.abs(point_pred - anchor) / np.maximum(radius, 1e-8)
            ),
            "mean_correction": np.mean(np.abs(corrected_pred - point_pred)),
        }

        return corrected_pred, diagnostics

    def scan_beta(
        self, point_pred: np.ndarray, anchor: np.ndarray, radius: np.ndarray
    ) -> Dict[float, Tuple[np.ndarray, Dict]]:
        """
        扫描不同β值的效果

        Args:
            point_pred: 点预测值
            anchor: 锚点值
            radius: 可信圆半径

        Returns:
            β值到（修正预测, 诊断信息）的映射
        """
        results = {}

        for beta in self.beta_values:
            corrected_pred, diagnostics = self.apply(point_pred, anchor, radius, beta)
            results[beta] = (corrected_pred, diagnostics)

        self.results = results
        return results

    def select_beta(
        self,
        point_pred: np.ndarray,
        anchor: np.ndarray,
        radius: np.ndarray,
        y_true: np.ndarray,
        scoring_func: Optional[callable] = None,
    ) -> Tuple[float, Dict]:
        """
        选择最优β值

        默认使用复合指标：
        Score = W̄ + 10 * max(0, CE - 0.05)

        其中：
        - W̄: 平均Winkler Score
        - CE: Coverage Error

        Args:
            point_pred: 点预测值
            anchor: 锚点值
            radius: 可信圆半径
            y_true: 真实值
            scoring_func: 自定义评分函数

        Returns:
            (最优β值, 所有β的评分结果) 元组
        """
        # 如果没有提供评分函数，使用默认的
        if scoring_func is None:

            def default_scoring(y_true, pred, anchor, radius):
                # 计算MAE
                mae = np.mean(np.abs(pred - y_true))

                # 计算Coverage Error
                q10 = anchor - radius
                q90 = anchor + radius
                coverage = np.mean((y_true >= q10) & (y_true <= q90))
                ce = np.abs(coverage - 0.8)  # 目标覆盖率80%

                # 计算Winkler Score
                width = q90 - q10
                penalty = np.where(
                    y_true < q10,
                    10 * (q10 - y_true),
                    np.where(y_true > q90, 10 * (y_true - q90), 0),
                )
                winkler = width + penalty
                winkler_mean = np.mean(winkler)

                # 复合指标
                score = winkler_mean + 10 * max(0, ce - 0.05)
                return {
                    "score": score,
                    "mae": mae,
                    "winkler_mean": winkler_mean,
                    "coverage_error": ce,
                    "coverage": coverage,
                }

            scoring_func = default_scoring

        # 扫描β值
        scan_results = self.scan_beta(point_pred, anchor, radius)

        # 评估每个β
        beta_scores = {}
        for beta, (corrected_pred, diagnostics) in scan_results.items():
            metrics = scoring_func(y_true, corrected_pred, anchor, radius)
            beta_scores[beta] = {
                "corrected_pred": corrected_pred,
                "diagnostics": diagnostics,
                "metrics": metrics,
            }

        # 选择最优β
        best_beta = min(
            beta_scores.keys(), key=lambda b: beta_scores[b]["metrics"]["score"]
        )

        return best_beta, beta_scores

    def check_crossing(self, q10: np.ndarray, q90: np.ndarray) -> Dict:
        """
        检查分位数交叉情况

        Args:
            q10: 10%分位数预测
            q90: 90%分位数预测

        Returns:
            交叉统计信息
        """
        crossing_rate_value = crossing_rate(q10, q90)

        return {
            "crossing_rate": crossing_rate_value,
            "threshold": self.crossing_threshold,
            "is_valid": crossing_rate_value <= self.crossing_threshold,
            "message": f"Crossing rate: {crossing_rate_value:.4f} "
            + (
                f"<= threshold {self.crossing_threshold}"
                if crossing_rate_value <= self.crossing_threshold
                else f"> threshold {self.crossing_threshold}"
            ),
        }


def beta_behavior_table() -> None:
    """
    打印β值行为表（来自README）
    """
    print("β值行为表:")
    print("β\tα_t (偏离=r_t)\tα_t (偏离=2r_t)\t行为\t角色")
    print("-" * 60)

    behaviors = [
        (0, 1.000, 1.000, "完全不拉回", "对照组（退化纯QR）"),
        (0.5, 0.607, 0.368, "温和拉回", "Snail-0.5"),
        (1, 0.368, 0.135, "标准拉回", "Snail-1"),
        (2, 0.135, 0.018, "强拉回", "Snail-2"),
        (5, 0.007, 0.000, "近似截断", "Snail-5"),
        ("∞", 0.000, 0.000, "完全拉回", "对照组（退化纯q50）"),
    ]

    for beta, alpha1, alpha2, behavior, role in behaviors:
        print(f"{beta}\t{alpha1:.3f}\t\t{alpha2:.3f}\t\t{behavior}\t{role}")


if __name__ == "__main__":
    # 测试软拉回机制
    print("Testing Snail Mechanism...")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟预测值
    point_pred = np.random.randn(n_samples) * 0.1  # 点预测
    anchor = np.random.randn(n_samples) * 0.05  # 锚点
    radius = np.abs(np.random.randn(n_samples)) * 0.05 + 0.02  # 半径

    # 创建蜗牛壳机制
    snail = SnailMechanism()

    # 应用软拉回
    corrected, diagnostics = snail.apply(point_pred, anchor, radius, beta=1.0)

    print(f"\nDiagnostics for β=1.0:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.4f}")

    # 扫描β值
    print("\nScanning β values...")
    scan_results = snail.scan_beta(point_pred, anchor, radius)

    for beta in snail.beta_values:
        _, diag = scan_results[beta]
        print(
            f"  β={beta}: mean_alpha={diag['mean_alpha']:.4f}, "
            f"mean_correction={diag['mean_correction']:.4f}"
        )

    # 检查交叉率
    q10 = anchor - radius
    q90 = anchor + radius
    crossing_info = snail.check_crossing(q10, q90)
    print(f"\nCrossing check: {crossing_info['message']}")

    # 打印β行为表
    print("\n" + "=" * 60)
    beta_behavior_table()
