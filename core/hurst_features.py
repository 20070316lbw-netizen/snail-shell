"""
hurst_features.py - Hurst 特征工程

将 core/hurst.py 的算法批量应用到 features_cn 格式的 DataFrame，
生成可直接拼入训练集的 hurst_Nm 特征列。

核心函数
--------
compute_hurst_features
    批量为每只股票计算滚动 Hurst，追加到 long-format DataFrame。

hurst_regime_label
    将 H 值离散化为市场状态标签（-1 / 0 / +1）。

数据假设
--------
- 输入 DataFrame 包含列：ticker, date, mom_20d（列名可自定义）
- date 列应为 datetime 类型或可被 pd.to_datetime 解析
- 月频数据，相邻两行 date 差通常为 28~31 天
- max_gap_days=45 判定数据缺口；超过此间隔时两侧视为独立序列段

性能说明
--------
对 5000 只股票 × 83 个月点，window=36 时约需 10~30 秒（单进程）。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from core.hurst import rolling_hurst


# ===========================================================================
# 内部辅助
# ===========================================================================

def _segment_indices(dates: pd.Series, max_gap_days: int) -> List[tuple]:
    """
    将日期序列按数据缺口切分成连续段，返回 [(start, end), ...] 索引列表。

    新段的起始条件：与前一行的日期差 > max_gap_days 天。

    参数:
        dates       : 已按升序排列的日期序列（pd.Series，DatetimeLike）
        max_gap_days: 视为缺口的最大允许间隔天数

    返回:
        [(start0, end0), (start1, end1), ...]
        每个 (start, end) 对应 dates.iloc[start:end]
    """
    n = len(dates)
    if n == 0:
        return []

    dates_dt = pd.to_datetime(dates).reset_index(drop=True)
    diffs = dates_dt.diff().dt.days.fillna(0)

    gap_positions = list(np.where(diffs.values > max_gap_days)[0])
    boundaries = [0] + gap_positions + [n]

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ===========================================================================
# 批量 Hurst 特征
# ===========================================================================

def compute_hurst_features(
    df: pd.DataFrame,
    windows: List[int] = None,
    method: str = "dfa",
    returns_col: str = "mom_20d",
    ticker_col: str = "ticker",
    date_col: str = "date",
    max_gap_days: int = 45,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    计算滚动 Hurst 特征并追加到 DataFrame。

    对每只股票的 returns_col 序列：
      1. 按 date_col 升序排列
      2. 识别日期缺口（> max_gap_days）并分段处理
      3. 在每段内调用 rolling_hurst(window=w)
      4. 将结果写回新列 hurst_{w}m（w ∈ windows）

    输出 DataFrame 保留原始行顺序，仅新增 hurst_Nm 列。
    无法计算的位置（序列过短、全 NaN 等）填 NaN。

    参数:
        df           : long-format DataFrame，含 ticker、date、mom_20d 等列
        windows      : 窗口列表（默认 [24, 36]）
        method       : "dfa"（推荐，月频）或 "rs"
        returns_col  : 收益率列名（默认 "mom_20d"）
        ticker_col   : 股票代码列名（默认 "ticker"）
        date_col     : 日期列名（默认 "date"）
        max_gap_days : 判定数据缺口的天数阈值
        min_periods  : rolling_hurst 的 min_periods 参数
                       （默认各 window 的 window // 2）

    返回:
        新 DataFrame（原数据的 copy），追加 hurst_{w}m 列（float64）
    """
    if windows is None:
        windows = [24, 36]

    result = df.copy()

    for window in windows:
        col_name = f"hurst_{window}m"
        result[col_name] = np.nan
        mp = min_periods if min_periods is not None else (window // 2)

        for ticker, group in df.groupby(ticker_col, sort=False):
            group_sorted = group.sort_values(date_col)
            original_idx = group_sorted.index
            dates = group_sorted[date_col]
            returns = group_sorted[returns_col].values.astype(float)

            H_values = np.full(len(group_sorted), np.nan)

            for seg_start, seg_end in _segment_indices(dates, max_gap_days):
                seg_returns = returns[seg_start:seg_end]
                seg_H = rolling_hurst(
                    seg_returns,
                    window=window,
                    method=method,
                    min_periods=mp,
                )
                H_values[seg_start:seg_end] = seg_H

            result.loc[original_idx, col_name] = H_values

    return result


# ===========================================================================
# H → 状态标签
# ===========================================================================

def hurst_regime_label(
    H: float,
    lower: float = 0.45,
    upper: float = 0.55,
) -> int:
    """
    将 Hurst 指数离散化为市场状态标签。

    区间划分：
      H < lower            →  -1  （均值回归 / 震荡市）
      lower ≤ H ≤ upper    →   0  （随机游走 / 中性）
      H > upper            →  +1  （趋势持续 / 动量市）

    参数:
        H     : Hurst 指数（float）；NaN 返回 0
        lower : 均值回归 / 随机游走分界（默认 0.45）
        upper : 随机游走 / 趋势 分界（默认 0.55）

    返回:
        整数标签 -1 / 0 / +1
    """
    if np.isnan(H):
        return 0
    if H < lower:
        return -1
    elif H > upper:
        return 1
    else:
        return 0


# ===========================================================================
# 便捷向量化版本
# ===========================================================================

def hurst_regime_labels(
    H_series: np.ndarray,
    lower: float = 0.45,
    upper: float = 0.55,
) -> np.ndarray:
    """
    向量化版本的 hurst_regime_label，处理整列 H 值。

    参数:
        H_series : Hurst 指数数组
        lower    : 均值回归上界（默认 0.45）
        upper    : 随机游走上界（默认 0.55）

    返回:
        int8 数组，各元素为 -1 / 0 / +1
    """
    H = np.asarray(H_series, dtype=float)
    labels = np.zeros(len(H), dtype=np.int8)
    labels[H < lower]  = -1
    labels[H > upper]  =  1
    # NaN → 保持 0（默认值）
    return labels


def summarize_hurst_features(df: pd.DataFrame, windows: List[int] = None) -> Dict:
    """
    汇总 hurst_Nm 列的基本统计量，便于快速诊断。

    参数:
        df      : 包含 hurst_Nm 列的 DataFrame
        windows : 要汇总的窗口列表（默认自动检测 df 中的 hurst_* 列）

    返回:
        字典，键为列名，值为统计字典 {"count", "nan_rate", "mean", "std",
        "min", "p25", "p50", "p75", "max"}
    """
    if windows is not None:
        cols = [f"hurst_{w}m" for w in windows]
    else:
        cols = [c for c in df.columns if c.startswith("hurst_")]

    summary = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        n_total = len(s)
        n_valid = s.notna().sum()
        summary[col] = {
            "count":    int(n_valid),
            "nan_rate": float((n_total - n_valid) / n_total) if n_total > 0 else 0.0,
            "mean":     float(s.mean()),
            "std":      float(s.std()),
            "min":      float(s.min()),
            "p25":      float(s.quantile(0.25)),
            "p50":      float(s.quantile(0.50)),
            "p75":      float(s.quantile(0.75)),
            "max":      float(s.max()),
        }
    return summary
