"""
asymmetric_mechanism.py - AS-GSPQR 非对称软拉回机制

核心创新：非对称区间 + 方向感知门控（Direction-Aware Gating）

设计原则：
  1. 半径锚定到中位数 a_t，与修正中心 ŷ*_t 完全解耦（消除循环依赖）
  2. 门控强度根据分歧方向选择对应的不确定性尺度
  3. 区间形状直接继承分位数模型的原始非对称信息

数学定义：
  r_down = a_t - q̂_{10,t}              (下行半径，以中位数为基准)
  r_up   = q̂_{90,t} - a_t             (上行半径，以中位数为基准)
  r_dir  = r_up   if δ_t > 0           (方向感知：看多分歧用上行尺度)
           r_down  if δ_t ≤ 0          (方向感知：看空分歧用下行尺度)
  λ_t = exp(-β · |δ_t| / r_dir)
  ŷ*_t = λ_t · ŷ_t + (1 - λ_t) · a_t
  Ĉ*_t = [ŷ*_t - r_down,  ŷ*_t + r_up]

对比对称版本（GSPQR）：
  - 对称：r_t = (q90 - q10) / 2，区间宽度固定为 2*r_t
  - 非对称：宽度 = r_down + r_up = q90 - q10（总宽不变，但分布非对称）
  注：总区间宽度与原始分位数区间相同，改变的是宽度在中心两侧的分配。
"""

import sys
import os
import numpy as np
from typing import Tuple, Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import crossing_rate


# ---------------------------------------------------------------------------
# 基础函数（无状态，可独立调用）
# ---------------------------------------------------------------------------

def compute_asymmetric_radii(
    anchor: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算非对称半径（锚定到中位数，与修正中心解耦）

    r_down = a_t - q̂_{10,t}   ≥ 0（若分位数无交叉）
    r_up   = q̂_{90,t} - a_t   ≥ 0

    参数:
        anchor : a_t = q̂_{50,t}，形状 (N,)
        q10    : q̂_{10,t}，形状 (N,)
        q90    : q̂_{90,t}，形状 (N,)

    返回:
        (r_down, r_up)，各形状 (N,)
    """
    r_down = anchor - q10
    r_up   = q90 - anchor

    # 数值安全：截断到 0（防止分位数轻微交叉导致负半径）
    r_down = np.maximum(r_down, 0.0)
    r_up   = np.maximum(r_up,   0.0)

    return r_down, r_up


def directional_gate(
    delta: np.ndarray,
    r_down: np.ndarray,
    r_up: np.ndarray,
    beta: float,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    方向感知门控函数

    r_dir_t = r_up   if δ_t > 0    (看多分歧，用上行不确定性衡量)
              r_down  if δ_t ≤ 0   (看空分歧，用下行不确定性衡量)
    λ_t = exp(-β · |δ_t| / r_dir_t)

    参数:
        delta  : δ_t = ŷ_t - a_t，形状 (N,)
        r_down : 下行半径，形状 (N,)
        r_up   : 上行半径，形状 (N,)
        beta   : 拉回强度参数 β ≥ 0
        epsilon: 防零除的数值下界

    返回:
        (lambda_t, r_dir)：门控系数和所选方向半径，各形状 (N,)
    """
    # 选择方向半径
    r_dir = np.where(delta > 0, r_up, r_down)
    r_dir_safe = np.maximum(r_dir, epsilon)

    # SNR = |δ_t| / r_dir
    snr = np.abs(delta) / r_dir_safe

    # 门控系数
    if np.isinf(beta):
        # β=∞：SNR=0（均值==中位数）时不拉回，否则完全拉回
        lambda_t = np.where(snr == 0.0, 1.0, 0.0)
    else:
        lambda_t = np.exp(-beta * snr)

    return lambda_t, r_dir


def asymmetric_soft_pullback(
    point_pred: np.ndarray,
    anchor: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    beta: float = 1.0,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    完整非对称软拉回（单次 β 调用）

    参数:
        point_pred : ŷ_t（MSE 模型输出），形状 (N,)
        anchor     : a_t = q̂_{50,t}，形状 (N,)
        q10        : q̂_{10,t}，形状 (N,)
        q90        : q̂_{90,t}，形状 (N,)
        beta       : 拉回强度 β ≥ 0
        epsilon    : 数值安全小量

    返回:
        (corrected_center, lower, upper, diagnostics)
        - corrected_center : ŷ*_t，形状 (N,)
        - lower            : ŷ*_t - r_down，形状 (N,)
        - upper            : ŷ*_t + r_up，形状 (N,)
        - diagnostics      : 诊断信息字典
    """
    # Step 1：计算非对称半径
    r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)

    # Step 2：分歧
    delta = point_pred - anchor

    # Step 3：方向感知门控
    lambda_t, r_dir = directional_gate(delta, r_down, r_up, beta, epsilon)

    # Step 4：修正中心
    corrected_center = lambda_t * point_pred + (1 - lambda_t) * anchor

    # Step 5：非对称区间（以修正中心为基准，半径锚到中位数）
    lower = corrected_center - r_down
    upper = corrected_center + r_up

    # 诊断信息
    diagnostics = {
        "beta"             : beta,
        "mean_lambda"      : float(np.mean(lambda_t)),
        "std_lambda"       : float(np.std(lambda_t)),
        "mean_delta"       : float(np.mean(np.abs(delta))),
        "mean_r_down"      : float(np.mean(r_down)),
        "mean_r_up"        : float(np.mean(r_up)),
        "mean_width"       : float(np.mean(upper - lower)),
        "mean_correction"  : float(np.mean(np.abs(corrected_center - point_pred))),
        # 偏斜指数：(r_up - r_down) / (r_up + r_down)，范围 [-1, 1]
        # 正值：区间整体偏向上行，负值：偏向下行
        "mean_skew_index"  : float(np.mean(
            (r_up - r_down) / np.maximum(r_up + r_down, epsilon)
        )),
    }

    return corrected_center, lower, upper, diagnostics


# ---------------------------------------------------------------------------
# AsymmetricSnailMechanism：类封装，支持 β 扫描 / 选择
# ---------------------------------------------------------------------------

class AsymmetricSnailMechanism:
    """
    非对称蜗牛壳机制

    接口与 SnailMechanism 保持一致，方便对比实验替换。

    主要方法：
      apply(point_pred, anchor, q10, q90, beta)  → 单次预测
      scan_beta(...)                              → 扫描 β 网格
      select_beta(..., y_true)                    → 在验证集上选最优 β
    """

    def __init__(
        self,
        beta_values: Optional[List[float]] = None,
        crossing_threshold: float = 0.1,
    ):
        if beta_values is None:
            self.beta_values = [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")]
        else:
            self.beta_values = beta_values

        self.crossing_threshold = crossing_threshold
        self._scan_cache: Dict = {}

    # ------------------------------------------------------------------
    def apply(
        self,
        point_pred: np.ndarray,
        anchor: np.ndarray,
        q10: np.ndarray,
        q90: np.ndarray,
        beta: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        单次非对称软拉回

        返回:
            (corrected_center, lower, upper, diagnostics)
        """
        return asymmetric_soft_pullback(point_pred, anchor, q10, q90, beta)

    # ------------------------------------------------------------------
    def scan_beta(
        self,
        point_pred: np.ndarray,
        anchor: np.ndarray,
        q10: np.ndarray,
        q90: np.ndarray,
    ) -> Dict[float, Tuple]:
        """
        扫描所有 β 值

        返回:
            {beta: (corrected_center, lower, upper, diagnostics)}
        """
        results = {}
        for beta in self.beta_values:
            results[beta] = self.apply(point_pred, anchor, q10, q90, beta)
        self._scan_cache = results
        return results

    # ------------------------------------------------------------------
    def select_beta(
        self,
        point_pred: np.ndarray,
        anchor: np.ndarray,
        q10: np.ndarray,
        q90: np.ndarray,
        y_true: np.ndarray,
        scoring_func=None,
    ) -> Tuple[float, Dict]:
        """
        在验证集上选最优 β（最小化复合评分）

        复合评分：Score = W̄ + 10 · max(0, CE - 0.05)

        返回:
            (best_beta, {beta: score_info})
        """
        if scoring_func is None:
            def scoring_func(y_true, center, lower, upper):
                width = upper - lower
                penalty = np.where(
                    y_true < lower, 10 * (lower - y_true),
                    np.where(y_true > upper, 10 * (y_true - upper), 0.0)
                )
                winkler_mean = float(np.mean(width + penalty))
                coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
                ce = abs(coverage - 0.8)
                mae = float(np.mean(np.abs(center - y_true)))
                score = winkler_mean + 2.0 * ce
                return {
                    "score": score,
                    "winkler_mean": winkler_mean,
                    "coverage_error": ce,
                    "coverage": coverage,
                    "mae": mae,
                }

        scan_results = self.scan_beta(point_pred, anchor, q10, q90)
        beta_scores = {}

        for beta, (center, lower, upper, diag) in scan_results.items():
            metrics = scoring_func(y_true, center, lower, upper)
            beta_scores[beta] = {
                "corrected_center": center,
                "lower": lower,
                "upper": upper,
                "diagnostics": diag,
                "metrics": metrics,
            }

        best_beta = min(beta_scores, key=lambda b: beta_scores[b]["metrics"]["score"])
        return best_beta, beta_scores

    # ------------------------------------------------------------------
    def check_crossing(self, q10: np.ndarray, q90: np.ndarray) -> Dict:
        """检查分位数交叉情况（与 SnailMechanism 接口一致）"""
        rate = crossing_rate(q10, q90)
        return {
            "crossing_rate": rate,
            "threshold": self.crossing_threshold,
            "is_valid": rate <= self.crossing_threshold,
        }


# ---------------------------------------------------------------------------
# 简单自测（python core/asymmetric_mechanism.py）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AsymmetricSnailMechanism 自测 ===\n")
    np.random.seed(42)
    N = 1000

    # 模拟数据：q10 < q50 < q90，带一定偏斜
    q10    = np.random.randn(N) * 0.05 - 0.08
    q50    = np.random.randn(N) * 0.03
    q90    = np.random.randn(N) * 0.06 + 0.09
    point  = q50 + np.random.randn(N) * 0.04    # MSE 预测

    mech = AsymmetricSnailMechanism()

    # 单次调用
    center, lower, upper, diag = mech.apply(point, q50, q10, q90, beta=1.0)
    print(f"β=1.0 诊断：")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # β 扫描
    print("\nβ 扫描结果（mean_lambda / mean_width / mean_skew_index）：")
    scan = mech.scan_beta(point, q50, q10, q90)
    for beta, (c, lo, hi, d) in scan.items():
        print(f"  β={beta:5}: λ̄={d['mean_lambda']:.3f}  "
              f"宽度={d['mean_width']:.4f}  "
              f"偏斜={d['mean_skew_index']:+.3f}")

    # 对比对称版本的宽度
    r_sym = (q90 - q10) / 2
    print(f"\n对称版本平均宽度 (2·r_t): {float(np.mean(2*r_sym)):.4f}")
    r_down, r_up = compute_asymmetric_radii(q50, q10, q90)
    print(f"非对称总宽度 (r_down+r_up): {float(np.mean(r_down+r_up)):.4f}  （应与上行相同）")
    print(f"平均 r_down: {float(np.mean(r_down)):.4f}")
    print(f"平均 r_up  : {float(np.mean(r_up)):.4f}")
