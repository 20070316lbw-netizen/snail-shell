"""
hurst.py - Hurst 指数计算

实现两种算法：
  - R/S 分析（Rescaled Range）：经典方法，理解直觉清晰
  - DFA（Detrended Fluctuation Analysis）：对短序列更准确，金融月频推荐

以及配套工具：
  - rolling_hurst：滚动计算，用于特征工程
  - hurst_to_beta：将 H 映射为 SnailMechanism 的 β 参数
  - adaptive_beta：端到端的推断时自适应 β

背景
----
Hurst 指数 H 刻画时间序列的长程依赖性：
  H > 0.5  趋势持续（动量市场），预测可以偏离锚点更远 → 较小 β
  H = 0.5  随机游走（无记忆），使用中性 β
  H < 0.5  均值回归（震荡市场），预测应强力拉回锚点 → 较大 β

数据说明（本项目）
-----------------
features_cn 为月频截面，mom_20d 字段（20 交易日累计收益率 ≈ 月度收益率）
是 Hurst 计算的推荐输入序列。每只股票最多约 83 个月点，存在数据断层。
推荐参数：window=36，min_periods=24。
"""

import numpy as np
from typing import Optional


# ===========================================================================
# 内部辅助
# ===========================================================================

def _validate_series(series: np.ndarray, min_len: int, name: str = "series") -> np.ndarray:
    """转为 float64，过滤 NaN，检查长度。"""
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < min_len:
        raise ValueError(
            f"hurst: {name} 有效长度 {len(s)} < 最小要求 {min_len}。"
            "请增大序列长度或减小 min_lag/scales。"
        )
    return s


def _log_spaced_ints(lo: int, hi: int, n: int = 20) -> np.ndarray:
    """在 [lo, hi] 之间生成 log 等间距整数，去重后返回。"""
    raw = np.unique(np.round(np.geomspace(lo, hi, n)).astype(int))
    return raw[(raw >= lo) & (raw <= hi)]


# ===========================================================================
# R/S 分析
# ===========================================================================

def hurst_rs(
    series: np.ndarray,
    min_lag: int = 4,
    max_lag: Optional[int] = None,
    n_lags: int = 20,
) -> float:
    """
    R/S 分析法（Rescaled Range）计算 Hurst 指数。

    算法：
      1. 对 log 等间距的子窗口长度 n ∈ [min_lag, max_lag]：
         - 把序列分成若干不重叠的长度为 n 的片段
         - 对每个片段计算 R（极差）/ S（标准差）
         - 所有片段的 R/S 均值 → E[R/S](n)
      2. OLS 拟合 log(E[R/S]) = H·log(n) + const
      3. H = 斜率

    适用场景：长序列（N ≥ 100）。短序列有已知正偏差；此时推荐 hurst_dfa。

    参数:
        series   : 收益率序列（非价格），NaN 会被过滤
        min_lag  : 最小子窗口长度（≥ 4）
        max_lag  : 最大子窗口长度（默认 len(series) // 2）
        n_lags   : 候选窗口数量

    返回:
        H 值（float），通常在 (0, 1) 之间
    """
    s = _validate_series(series, min_len=max(8, min_lag * 2), name="hurst_rs.series")
    N = len(s)
    max_lag = max_lag or (N // 2)
    max_lag = min(max_lag, N // 2)

    lags = _log_spaced_ints(min_lag, max_lag, n_lags)
    if len(lags) < 2:
        raise ValueError(
            f"hurst_rs: 有效 lag 数量不足（{len(lags)}），"
            f"请减小 min_lag 或增大序列长度（当前 N={N}）。"
        )

    rs_values = []
    for n in lags:
        n_segments = N // n
        if n_segments < 1:
            continue
        rs_list = []
        for i in range(n_segments):
            seg = s[i * n : (i + 1) * n]
            mean_seg = np.mean(seg)
            dev = np.cumsum(seg - mean_seg)
            R = np.max(dev) - np.min(dev)
            S = np.std(seg, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((n, np.mean(rs_list)))

    if len(rs_values) < 2:
        raise ValueError(
            "hurst_rs: 有效 R/S 点数不足，无法拟合。请增大序列长度。"
        )

    log_n  = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # OLS: log_rs = H * log_n + const
    H = np.polyfit(log_n, log_rs, 1)[0]
    return float(H)


# ===========================================================================
# DFA
# ===========================================================================

def hurst_dfa(
    series: np.ndarray,
    scales: Optional[list] = None,
    order: int = 1,
    n_scales: int = 20,
) -> float:
    """
    去趋势波动分析（Detrended Fluctuation Analysis）计算 Hurst 指数。

    相比 R/S，DFA 对短序列（N ≥ 30）偏差更小，
    是金融月频数据的推荐算法。

    算法：
      1. 计算集成序列 X_k = Σ(r_i - mean(r))，k=1..N
      2. 对 log 等间距的尺度 s ∈ scales：
         - 将 X 分成长度为 s 的不重叠片段
         - 在每个片段内用 order 阶多项式去趋势
         - F(s) = sqrt( mean(残差²) )
      3. OLS 拟合 log(F(s)) = H·log(s) + const
      4. H = 斜率

    参数:
        series  : 收益率序列，NaN 会被过滤
        scales  : 尺度列表（默认自动生成：[4, ..., N//4]，log 等间距）
        order   : 去趋势多项式阶数（1=线性，标准选项）
        n_scales: 自动生成时的候选尺度数量

    返回:
        H 值（float）
    """
    s = _validate_series(series, min_len=16, name="hurst_dfa.series")
    N = len(s)

    if scales is None:
        min_s = 4
        max_s = N // 4
        if max_s < min_s:
            raise ValueError(
                f"hurst_dfa: 序列长度 {N} 太短，无法生成有效尺度。"
                f"需要 N ≥ {min_s * 4}。"
            )
        scales = _log_spaced_ints(min_s, max_s, n_scales).tolist()

    scales = sorted(set(int(sc) for sc in scales if sc >= 4 and sc <= N // 2))
    if len(scales) < 2:
        raise ValueError(
            f"hurst_dfa: 有效尺度数量不足（{len(scales)}），"
            f"请增大序列长度（当前 N={N}）或调整 scales 参数。"
        )

    # 集成序列
    X = np.cumsum(s - np.mean(s))

    fluctuations = []
    for sc in scales:
        n_seg = N // sc
        if n_seg < 1:
            continue
        f2_list = []
        t = np.arange(sc, dtype=float)
        for i in range(n_seg):
            seg = X[i * sc : (i + 1) * sc]
            # order 阶多项式去趋势
            coeffs = np.polyfit(t, seg, order)
            trend  = np.polyval(coeffs, t)
            f2_list.append(np.mean((seg - trend) ** 2))
        if f2_list:
            F = np.sqrt(np.mean(f2_list))
            if F > 0:
                fluctuations.append((sc, F))

    if len(fluctuations) < 2:
        raise ValueError(
            "hurst_dfa: 有效波动点数不足，无法拟合。请增大序列长度。"
        )

    log_s = np.log([v[0] for v in fluctuations])
    log_F = np.log([v[1] for v in fluctuations])

    H = np.polyfit(log_s, log_F, 1)[0]
    return float(H)


# ===========================================================================
# 滚动 Hurst
# ===========================================================================

def rolling_hurst(
    series: np.ndarray,
    window: int = 36,
    method: str = "dfa",
    min_periods: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    滚动计算 Hurst 指数。

    输出与 series 等长，前缀不满 min_periods 的位置填 NaN。
    若某个窗口内计算失败（序列太短、全 NaN 等），该位置也返回 NaN。

    参数:
        series     : 收益率序列（月频推荐 window=36）
        window     : 滚动窗口长度（单位与 series 一致）
        method     : "dfa"（推荐，月频）或 "rs"（长序列适用）
        min_periods: 开始计算所需的最小有效点数（默认 window // 2）
        **kwargs   : 透传给 hurst_dfa 或 hurst_rs 的额外参数

    返回:
        H 数组，形状 (len(series),)，float64，NaN 表示无法计算
    """
    series = np.asarray(series, dtype=float)
    N = len(series)
    min_periods = min_periods or (window // 2)
    fn = hurst_dfa if method == "dfa" else hurst_rs

    result = np.full(N, np.nan)

    for t in range(window - 1, N):
        seg = series[t - window + 1 : t + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < min_periods:
            continue
        try:
            result[t] = fn(valid, **kwargs)
        except (ValueError, np.linalg.LinAlgError):
            pass  # 保持 NaN

    return result


# ===========================================================================
# H → β 映射
# ===========================================================================

def hurst_to_beta(
    H: float,
    beta_base: float = 1.0,
    sensitivity: float = 2.0,
    beta_min: float = 0.0,
    beta_max: float = 5.0,
) -> float:
    """
    将 Hurst 指数线性映射为 SnailMechanism 的 β 参数。

    公式：
      β = clip( beta_base + (0.5 - H) × sensitivity,  beta_min,  beta_max )

    直觉：
      H = 0.5  →  β = beta_base           （随机游走，中性拉回）
      H → 1.0  →  β 减小（趋势市，少拉回，让预测沿趋势延伸）
      H → 0.0  →  β 增大（均值回归市，多拉回向锚点靠拢）

    参数:
        H           : Hurst 指数，理论范围 (0, 1)
        beta_base   : H=0.5 时对应的中性 β（通常由验证集选出）
        sensitivity : 每单位 H 偏离 0.5 所带来的 β 变化量
                      sensitivity=2 → H 从 0.5 变到 0 时 β 增加 1.0
        beta_min    : β 下界（裁剪）
        beta_max    : β 上界（裁剪）

    返回:
        β 值（float）
    """
    if np.isnan(H):
        return float(beta_base)
    beta = beta_base + (0.5 - float(H)) * sensitivity
    return float(np.clip(beta, beta_min, beta_max))


# ===========================================================================
# 端到端自适应 β
# ===========================================================================

def adaptive_beta(
    returns: np.ndarray,
    window: int = 36,
    method: str = "dfa",
    beta_base: float = 1.0,
    sensitivity: float = 2.0,
    beta_min: float = 0.0,
    beta_max: float = 5.0,
    min_periods: Optional[int] = None,
) -> float:
    """
    给定近期收益率序列，计算当前时刻的自适应 β（标量）。

    = hurst_to_beta( rolling_hurst(returns, window)[-1], ... )

    若无法计算 Hurst（序列过短或全 NaN），返回 beta_base。

    参数:
        returns    : 近期收益率序列（月频推荐，至少 min_periods 个有效点）
        window     : 用于计算 Hurst 的滚动窗口
        method     : "dfa" 或 "rs"
        beta_base  : H=0.5 时的中性 β
        sensitivity: H → β 映射的灵敏度系数
        beta_min   : β 下界
        beta_max   : β 上界
        min_periods: rolling_hurst 的 min_periods 参数

    返回:
        β 值（float）
    """
    H_series = rolling_hurst(
        returns, window=window, method=method, min_periods=min_periods
    )
    # 取最后一个有效值
    valid = H_series[~np.isnan(H_series)]
    H = float(valid[-1]) if len(valid) > 0 else np.nan
    return hurst_to_beta(H, beta_base, sensitivity, beta_min, beta_max)
