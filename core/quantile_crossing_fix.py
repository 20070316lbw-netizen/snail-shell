"""
quantile_crossing_fix.py - 分位数交叉修复

分位数交叉（q10 > q50 或 q50 > q90）会引发一系列下游问题：

  1. radius = (q90 - q10) / 2 < 0
     → SnailMechanism 计算 alpha 时除以负半径，拉回方向反转
  2. Winkler Score 计算错误
     → width = upper - lower < 0，惩罚项符号错误
  3. coverage_error / actual_coverage 统计失真
     → lower > upper 导致没有样本"在区间内"

修复策略
--------
以 q50（中位数 / 锚点）为基准：
  fixed_q10 = min(q10, q50)   # q10 不得高于锚点
  fixed_q90 = max(q90, q50)   # q90 不得低于锚点
  q50 保持不变

数学保证：
  fixed_q10 ≤ q50 ≤ fixed_q90  （恒成立）
  (fixed_q90 - fixed_q10) / 2 ≥ 0  （radius 非负）

设计原则
--------
- q50 不变：锚点是 SnailMechanism 的核心输入，修改它会改变拉回方向
- 最小扰动：只截断越界的一侧，不改变另一侧
- predict() 保持原始输出：便于用 crossing_info() 诊断交叉情况
- 仅在 predict_anchor_and_radius() 中自动应用，对上层透明
"""

import numpy as np
from typing import Dict, Tuple


def fix_quantile_crossing(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    修复分位数交叉，强制 q10 ≤ q50 ≤ q90

    参数:
        q10 : 10% 分位数预测，形状 (N,)
        q50 : 50% 分位数预测（锚点），形状 (N,)
        q90 : 90% 分位数预测，形状 (N,)

    返回:
        (fixed_q10, q50, fixed_q90) 三元组，各形状 (N,)
        其中 fixed_q10 ≤ q50 ≤ fixed_q90 严格成立
    """
    fixed_q10 = np.minimum(q10, q50)
    fixed_q90 = np.maximum(q90, q50)
    return fixed_q10, q50.copy(), fixed_q90


def crossing_info(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Dict[str, object]:
    """
    诊断三类分位数交叉情况

    三类交叉按严重程度递增：
      类型 A：q10 > q50（10% 分位数高于中位数）
      类型 B：q90 < q50（90% 分位数低于中位数）
      类型 C：q10 > q90（最严重，上下界完全倒置）

    参数:
        q10 : 10% 分位数，形状 (N,)
        q50 : 50% 分位数，形状 (N,)
        q90 : 90% 分位数，形状 (N,)

    返回:
        字典，包含：
          n_q10_above_q50    : 类型 A 样本数
          n_q90_below_q50    : 类型 B 样本数
          n_q10_above_q90    : 类型 C 样本数
          rate_q10_above_q50 : 类型 A 比率
          rate_q90_below_q50 : 类型 B 比率
          rate_q10_above_q90 : 类型 C 比率（等价于 crossing_rate）
          any_crossing       : 是否存在任何交叉（bool）
    """
    n = len(q10)
    if n == 0:
        return {
            "n_q10_above_q50":    0,
            "n_q90_below_q50":    0,
            "n_q10_above_q90":    0,
            "rate_q10_above_q50": 0.0,
            "rate_q90_below_q50": 0.0,
            "rate_q10_above_q90": 0.0,
            "any_crossing":       False,
        }

    n_a = int(np.sum(q10 > q50))
    n_b = int(np.sum(q90 < q50))
    n_c = int(np.sum(q10 > q90))

    return {
        "n_q10_above_q50":    n_a,
        "n_q90_below_q50":    n_b,
        "n_q10_above_q90":    n_c,
        "rate_q10_above_q50": n_a / n,
        "rate_q90_below_q50": n_b / n,
        "rate_q10_above_q90": n_c / n,
        "any_crossing":       (n_a + n_b + n_c) > 0,
    }
