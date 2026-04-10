"""
spiral_monitor.py - 二维锚点 + 螺旋拟合 + 外扩速度

将二维锚点 p_t = (a_t, r_t) 投影到极坐标，拟合对数螺线：
log ρ_t = log A + Bθ_t

外扩速度：
v_t = (ρ_t - ρ_{t-1}) / (θ_t - θ_{t-1} + ε)

失效预警（滚动窗口 W=60）：
Alert_t = 1[v_t > μ_{v,t} + 2σ_{v,t}]
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats


def cartesian_to_polar(a: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将笛卡尔坐标 (a, r) 转换为极坐标 (ρ, θ)

    参数:
        a: 锚点值（x坐标）
        r: 半径值（y坐标）

    返回:
        (ρ, θ) 元组
    """
    # 计算极径
    rho = np.sqrt(a**2 + r**2)

    # 计算极角（atan2返回-π到π的值）
    theta = np.arctan2(r, a)

    return rho, theta


def log_spiral_model(theta: np.ndarray, log_A: float, B: float) -> np.ndarray:
    """
    对数螺线模型：log ρ = log A + B * θ

    参数:
        theta: 极角
        log_A: log(A)
        B: 螺线参数

    返回:
        log(ρ)
    """
    return log_A + B * theta


def fit_log_spiral(rho: np.ndarray, theta: np.ndarray) -> Tuple[float, float, float]:
    """
    拟合对数螺线

    参数:
        rho: 极径序列
        theta: 极角序列

    返回:
        (log_A, B, R²) 元组
    """
    # 取对数
    log_rho = np.log(rho + 1e-8)  # 防止log(0)

    # 线性回归拟合 log(ρ) = log_A + B * θ
    # 使用最小二乘法
    A_matrix = np.vstack([np.ones_like(theta), theta]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, log_rho, rcond=None)

    log_A, B = coeffs

    # 计算R²
    y_pred = log_A + B * theta
    ss_res = np.sum((log_rho - y_pred) ** 2)
    ss_tot = np.sum((log_rho - np.mean(log_rho)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return log_A, B, r_squared


def calculate_expansion_velocity(
    rho: np.ndarray, theta: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """
    计算外扩速度

    公式: v_t = (ρ_t - ρ_{t-1}) / (θ_t - θ_{t-1} + ε)

    参数:
        rho: 极径序列
        theta: 极角序列
        epsilon: 防爆参数

    返回:
        外扩速度序列
    """
    if len(rho) < 2:
        return np.array([])

    # 计算差分
    delta_rho = np.diff(rho)
    delta_theta = np.diff(theta)

    # 计算速度（添加epsilon防止除零）
    velocity = delta_rho / (delta_theta + epsilon)

    return velocity


def rolling_alert(
    velocity: np.ndarray, window: int = 60, threshold_sigma: float = 2.0
) -> np.ndarray:
    """
    计算滚动窗口的失效预警

    Alert_t = 1[v_t > μ_{v,t} + 2σ_{v,t}]

    参数:
        velocity: 外扩速度序列
        window: 滚动窗口大小
        threshold_sigma: σ倍数阈值

    返回:
        预警信号序列（1=预警，0=正常）
    """
    n = len(velocity)
    alerts = np.zeros(n)

    if n > window:
        vel_series = pd.Series(velocity)

        # 计算滚动窗口的统计量 (排除当前时刻)
        # pandas 的 std 默认是样本标准差 (ddof=1)，这里指定 ddof=0 匹配 numpy.std 的默认行为
        rolling_window = vel_series.shift(1).rolling(window=window)
        mu = rolling_window.mean().to_numpy()
        sigma = rolling_window.std(ddof=0).to_numpy()

        # 判断是否预警
        # 我们只在 [window, n) 范围内比较，避免 NaN 带来的 RuntimeWarning
        mask = np.zeros(n, dtype=bool)
        mask[window:] = velocity[window:] > (mu[window:] + threshold_sigma * sigma[window:])
        alerts[mask] = 1

    return alerts


class SpiralMonitor:
    """
    螺旋监控器

    监控锚点轨迹在极坐标中的形态，
    计算外扩速度和失效预警
    """

    def __init__(
        self,
        window: int = 60,
        threshold_sigma: float = 2.0,
        r_squared_threshold: float = 0.5,
    ):
        """
        初始化螺旋监控器

        参数:
            window: 滚动窗口大小
            threshold_sigma: σ倍数阈值
            r_squared_threshold: R²阈值（低于此值时警告）
        """
        self.window = window
        self.threshold_sigma = threshold_sigma
        self.r_squared_threshold = r_squared_threshold

        # 存储结果
        self.results = {}

    def analyze(
        self,
        anchor: np.ndarray,
        radius: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        分析锚点轨迹的螺旋形态

        参数:
            anchor: 锚点序列
            radius: 半径序列
            timestamps: 时间戳（可选）

        返回:
            分析结果字典
        """
        # 转换为极坐标
        rho, theta = cartesian_to_polar(anchor, radius)

        # 拟合对数螺线
        log_A, B, r_squared = fit_log_spiral(rho, theta)

        # 计算外扩速度
        velocity = calculate_expansion_velocity(rho, theta)

        # 计算预警信号
        alerts = rolling_alert(velocity, self.window, self.threshold_sigma)

        # 存储结果
        self.results = {
            "rho": rho,
            "theta": theta,
            "log_A": log_A,
            "B": B,
            "r_squared": r_squared,
            "velocity": velocity,
            "alerts": alerts,
            "mean_velocity": np.mean(velocity) if len(velocity) > 0 else 0,
            "std_velocity": np.std(velocity) if len(velocity) > 0 else 0,
            "alert_rate": np.mean(alerts) if len(alerts) > 0 else 0,
        }

        return self.results

    def get_trajectory_stats(self) -> Dict:
        """
        获取轨迹统计信息

        返回:
            轨迹统计字典
        """
        if not self.results:
            return {}

        stats_dict = {
            "spiral_quality": {
                "log_A": self.results["log_A"],
                "B": self.results["B"],
                "r_squared": self.results["r_squared"],
                "is_good_fit": self.results["r_squared"] >= self.r_squared_threshold,
            },
            "expansion": {
                "mean_velocity": self.results["mean_velocity"],
                "std_velocity": self.results["std_velocity"],
                "velocity_range": (
                    np.min(self.results["velocity"])
                    if len(self.results["velocity"]) > 0
                    else 0,
                    np.max(self.results["velocity"])
                    if len(self.results["velocity"]) > 0
                    else 0,
                ),
            },
            "alerts": {
                "alert_rate": self.results["alert_rate"],
                "total_alerts": np.sum(self.results["alerts"]),
                "window": self.window,
                "threshold_sigma": self.threshold_sigma,
            },
        }

        return stats_dict

    def check_spiral_validity(self) -> Dict:
        """
        检查螺旋形态的有效性

        返回:
            有效性检查结果
        """
        if not self.results:
            return {"is_valid": False, "message": "No analysis results available"}

        r_squared = self.results["r_squared"]

        if r_squared < self.r_squared_threshold:
            return {
                "is_valid": False,
                "r_squared": r_squared,
                "threshold": self.r_squared_threshold,
                "message": f"Poor spiral fit: R² = {r_squared:.3f} < {self.r_squared_threshold}",
            }
        else:
            return {
                "is_valid": True,
                "r_squared": r_squared,
                "threshold": self.r_squared_threshold,
                "message": f"Good spiral fit: R² = {r_squared:.3f} >= {self.r_squared_threshold}",
            }

    def predict_next_rho(self, theta_next: np.ndarray) -> np.ndarray:
        """
        基于拟合的螺线预测下一个极径

        参数:
            theta_next: 下一个极角

        返回:
            预测的极径
        """
        if not self.results:
            raise ValueError("Must call analyze() before prediction")

        log_A = self.results["log_A"]
        B = self.results["B"]

        log_rho_pred = log_A + B * theta_next
        rho_pred = np.exp(log_rho_pred)

        return rho_pred


def visualize_spiral(
    anchor: np.ndarray,
    radius: np.ndarray,
    spiral_monitor: Optional[SpiralMonitor] = None,
) -> None:
    """
    可视化螺旋轨迹（简单的文本可视化）

    参数:
        anchor: 锚点序列
        radius: 半径序列
        spiral_monitor: 螺旋监控器（可选）
    """
    if spiral_monitor is None:
        spiral_monitor = SpiralMonitor()
        results = spiral_monitor.analyze(anchor, radius)
    else:
        results = spiral_monitor.results

    print("螺旋轨迹分析:")
    print(f"  螺线参数: log_A = {results['log_A']:.4f}, B = {results['B']:.4f}")
    print(f"  拟合质量: R² = {results['r_squared']:.4f}")
    print(
        f"  外扩速度: 均值 = {results['mean_velocity']:.6f}, "
        f"标准差 = {results['std_velocity']:.6f}"
    )
    print(f"  预警率: {results['alert_rate']:.4f}")


if __name__ == "__main__":
    # 测试螺旋监控器
    print("Testing Spiral Monitor...")

    # 生成模拟数据（模拟一个扩张的螺旋）
    np.random.seed(42)
    n_samples = 200

    # 生成渐进的极角
    theta = np.linspace(0, 4 * np.pi, n_samples)

    # 生成渐进的极径（对数螺线）
    A = 0.1
    B = 0.2
    rho = A * np.exp(B * theta) + np.random.randn(n_samples) * 0.01

    # 转换为笛卡尔坐标
    anchor = rho * np.cos(theta)
    radius = rho * np.sin(theta)

    # 创建螺旋监控器
    monitor = SpiralMonitor(window=30, threshold_sigma=2.0)

    # 分析
    results = monitor.analyze(anchor, radius)

    print("\n分析结果:")
    print(f"  螺线参数: log_A = {results['log_A']:.4f}, B = {results['B']:.4f}")
    print(f"  真实参数: log_A = {np.log(A):.4f}, B = {B:.4f}")
    print(f"  拟合质量: R² = {results['r_squared']:.4f}")
    print(f"  外扩速度: 均值 = {results['mean_velocity']:.6f}")
    print(f"  预警率: {results['alert_rate']:.4f}")

    # 检查螺旋有效性
    validity = monitor.check_spiral_validity()
    print(f"\n螺旋有效性: {validity['message']}")

    # 获取统计信息
    stats = monitor.get_trajectory_stats()
    print("\n轨迹统计:")
    for category, values in stats.items():
        print(f"  {category}: {values}")

    # 预测下一个极径
    theta_next = np.array([4 * np.pi + 0.1])
    rho_pred = monitor.predict_next_rho(theta_next)
    print(f"\n预测下一个极径 (θ={theta_next[0]:.2f}): ρ = {rho_pred[0]:.4f}")
