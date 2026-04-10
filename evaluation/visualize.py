"""
visualize.py - 可视化函数

实现以下可视化：
1. Pareto曲线（Coverage vs Width）
2. 时间动态图（Winkler Score vs Time）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_pareto_curve(
    methods_data: Dict[str, Dict],
    y_true: np.ndarray,
    target_coverage: float = 0.8,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    绘制Pareto曲线

    所有方法在Coverage Error vs Width空间的散点图
    目标是左下角（覆盖误差小且区间窄）

    参数:
        methods_data: 方法名到预测结果的映射
        y_true: 真实值
        target_coverage: 目标覆盖率
        figsize: 图形大小
    """
    # 收集数据
    coverage_errors = []
    avg_widths = []
    method_names = []

    for method_name, predictions in methods_data.items():
        lower = predictions["lower"]
        upper = predictions["upper"]

        # 计算覆盖率并求Coverage Error
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage_rate = np.mean(in_interval)
        ce = np.abs(coverage_rate - target_coverage)

        # 计算平均区间宽度
        avg_width = np.mean(upper - lower)

        coverage_errors.append(ce)
        avg_widths.append(avg_width)
        method_names.append(method_name)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点图
    ax.scatter(avg_widths, coverage_errors, s=100, alpha=0.6)

    # 添加标签
    for i, method_name in enumerate(method_names):
        ax.annotate(
            method_name,
            (avg_widths[i], coverage_errors[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            alpha=0.8,
        )

    # 设置图形属性
    ax.set_xlabel("平均区间宽度", fontsize=12)
    ax.set_ylabel("Coverage Error", fontsize=12)
    ax.set_title("Pareto曲线: Coverage Error vs 区间宽度", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 添加理想区域标记
    ax.text(
        0.05,
        0.05,
        "理想区域\n(窄区间 + 低Coverage Error)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


def plot_time_dynamics(
    y_true: np.ndarray,
    methods_predictions: Dict[str, Dict],
    timestamps: Optional[np.ndarray] = None,
    window: int = 20,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    绘制时间动态图

    Winkler Score vs Time，重点观察regime切换期间的稳定性对比

    参数:
        y_true: 真实值
        methods_predictions: 方法名到预测结果的映射
        timestamps: 时间戳（可选）
        window: 滚动窗口大小
        figsize: 图形大小
    """
    n_samples = len(y_true)

    # 如果没有提供时间戳，使用索引
    if timestamps is None:
        timestamps = np.arange(n_samples)

    # 计算每个方法的滚动Winkler Score
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 第一个子图：原始Winkler Score
    ax1 = axes[0]

    for method_name, predictions in methods_predictions.items():
        lower = predictions["lower"]
        upper = predictions["upper"]

        # 计算每个时间点的Winkler Score
        width = upper - lower
        # 惧罚系数与 evaluation/metrics.winkler_score 保持一致：2/alpha = 2/0.2 = 10（80% CI）
        _alpha = 0.2
        penalty_lower = np.where(y_true < lower, (2 / _alpha) * (lower - y_true), 0)
        penalty_upper = np.where(y_true > upper, (2 / _alpha) * (y_true - upper), 0)
        winkler_scores = width + penalty_lower + penalty_upper

        # 绘制原始分数
        ax1.plot(timestamps, winkler_scores, alpha=0.3, linewidth=0.8)

        # 绘制滚动平均
        if len(winkler_scores) >= window:
            rolling_avg = pd.Series(winkler_scores).rolling(window=window).mean()
            ax1.plot(timestamps, rolling_avg, label=method_name, linewidth=2)

    ax1.set_xlabel("时间", fontsize=12)
    ax1.set_ylabel("Winkler Score", fontsize=12)
    ax1.set_title(
        f"时间动态图: Winkler Score (滚动窗口={window})", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # 第二个子图：区间宽度
    ax2 = axes[1]

    for method_name, predictions in methods_predictions.items():
        lower = predictions["lower"]
        upper = predictions["upper"]

        # 计算区间宽度
        width = upper - lower

        # 绘制原始宽度
        ax2.plot(timestamps, width, alpha=0.3, linewidth=0.8)

        # 绘制滚动平均
        if len(width) >= window:
            rolling_avg = pd.Series(width).rolling(window=window).mean()
            ax2.plot(timestamps, rolling_avg, label=method_name, linewidth=2)

    ax2.set_xlabel("时间", fontsize=12)
    ax2.set_ylabel("区间宽度", fontsize=12)
    ax2.set_title(f"区间宽度动态图 (滚动窗口={window})", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_beta_comparison(
    beta_scores: Dict[float, Dict], figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    绘制β值比较图

    显示不同β值下的评估指标

    参数:
        beta_scores: β值到评分结果的映射
        figsize: 图形大小
    """
    # 提取数据
    betas = []
    scores = []
    maes = []
    winklers = []
    coverages = []

    for beta, score_info in beta_scores.items():
        metrics = score_info["metrics"]

        betas.append(str(beta))
        scores.append(metrics["score"])
        maes.append(metrics["mae"])
        winklers.append(metrics["winkler_mean"])
        coverages.append(metrics["coverage"])

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 复合评分
    axes[0, 0].bar(betas, scores, alpha=0.7)
    axes[0, 0].set_xlabel("β值")
    axes[0, 0].set_ylabel("复合评分")
    axes[0, 0].set_title("复合评分 (越低越好)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # MAE
    axes[0, 1].bar(betas, maes, alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("β值")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].set_title("平均绝对误差 (越低越好)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Winkler Score
    axes[1, 0].bar(betas, winklers, alpha=0.7, color="green")
    axes[1, 0].set_xlabel("β值")
    axes[1, 0].set_ylabel("Winkler Score")
    axes[1, 0].set_title("Winkler Score (越低越好)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 覆盖率
    axes[1, 1].bar(betas, coverages, alpha=0.7, color="red")
    axes[1, 1].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="目标: 80%")
    axes[1, 1].set_xlabel("β值")
    axes[1, 1].set_ylabel("覆盖率")
    axes[1, 1].set_title("实际覆盖率 (目标: 80%)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].legend()

    plt.suptitle("不同β值下的性能比较", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_method_comparison(
    methods_metrics: pd.DataFrame, figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    绘制方法比较图

    参数:
        methods_metrics: 包含方法评估指标的DataFrame
        figsize: 图形大小
    """
    # 设置图形
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # 需要绘制的指标
    metrics_to_plot = [
        "mae",
        "coverage_error",
        "winkler_score",
        "interval_width",
        "rank_ic",
        "composite_score",
    ]

    # 指标标签
    metric_labels = [
        "MAE",
        "Coverage Error",
        "Winkler Score",
        "Interval Width",
        "RankIC",
        "Composite Score",
    ]

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        if metric in methods_metrics.columns:
            methods = methods_metrics["Method"]
            values = methods_metrics[metric]

            # 创建条形图
            bars = axes[i].bar(methods, values, alpha=0.7)

            # 设置标签
            axes[i].set_xlabel("方法")
            axes[i].set_ylabel(label)
            axes[i].set_title(label)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].grid(True, alpha=0.3, axis="y")

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.suptitle("方法性能比较", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_spiral_trajectory(
    anchor: np.ndarray,
    radius: np.ndarray,
    alerts: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    绘制螺旋轨迹

    参数:
        anchor: 锚点序列
        radius: 半径序列
        alerts: 预警信号（可选）
        figsize: 图形大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 第一个子图：笛卡尔坐标
    ax1 = axes[0]

    if alerts is not None:
        # 根据预警信号着色
        colors = ["red" if alert else "blue" for alert in alerts]
        ax1.scatter(anchor, radius, c=colors, alpha=0.6, s=20)

        # 添加图例
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.6, label="正常"),
            Patch(facecolor="red", alpha=0.6, label="预警"),
        ]
        ax1.legend(handles=legend_elements)
    else:
        ax1.plot(anchor, radius, "b-", alpha=0.6, linewidth=1)
        ax1.scatter(anchor, radius, alpha=0.6, s=20)

    ax1.set_xlabel("锚点 (a_t)")
    ax1.set_ylabel("半径 (r_t)")
    ax1.set_title("锚点轨迹 (笛卡尔坐标)")
    ax1.grid(True, alpha=0.3)

    # 第二个子图：极坐标
    ax2 = axes[1]

    # 转换为极坐标
    theta = np.arctan2(radius, anchor)
    rho = np.sqrt(anchor**2 + radius**2)

    if alerts is not None:
        # 根据预警信号着色
        colors = ["red" if alert else "blue" for alert in alerts]
        ax2.scatter(theta, rho, c=colors, alpha=0.6, s=20)
    else:
        ax2.plot(theta, rho, "b-", alpha=0.6, linewidth=1)
        ax2.scatter(theta, rho, alpha=0.6, s=20)

    ax2.set_xlabel("极角 (θ)")
    ax2.set_ylabel("极径 (ρ)")
    ax2.set_title("锚点轨迹 (极坐标)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("螺旋轨迹分析", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    绘制回归诊断图

    参数:
        y_true: 真实值
        y_pred: 预测值
        residuals: 残差（可选）
        figsize: 图形大小
    """
    if residuals is None:
        residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. 预测值 vs 真实值
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        linewidth=2,
        label="理想线",
    )
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("预测值 vs 真实值")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 残差分布
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("残差")
    axes[1].set_ylabel("频数")
    axes[1].set_title("残差分布")
    axes[1].grid(True, alpha=0.3)

    # 3. 残差 vs 预测值
    axes[2].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[2].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[2].set_xlabel("预测值")
    axes[2].set_ylabel("残差")
    axes[2].set_title("残差 vs 预测值")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("回归诊断图", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试可视化函数
    print("Testing visualization functions...")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500

    y_true = np.random.randn(n_samples)

    # 模拟几个方法
    methods_data = {}

    # 方法1：Snail-1
    methods_data["Snail-1"] = {
        "point_pred": y_true + np.random.randn(n_samples) * 0.1,
        "lower": y_true - 0.5 + np.random.randn(n_samples) * 0.1,
        "upper": y_true + 0.5 + np.random.randn(n_samples) * 0.1,
    }

    # 方法2：CP
    methods_data["CP"] = {
        "point_pred": y_true + np.random.randn(n_samples) * 0.15,
        "lower": y_true - 0.6 + np.random.randn(n_samples) * 0.1,
        "upper": y_true + 0.6 + np.random.randn(n_samples) * 0.1,
    }

    # 方法3：QR
    methods_data["QR"] = {
        "point_pred": y_true + np.random.randn(n_samples) * 0.12,
        "lower": y_true - 0.55 + np.random.randn(n_samples) * 0.1,
        "upper": y_true + 0.55 + np.random.randn(n_samples) * 0.1,
    }

    # 确保lower < upper
    for method in methods_data.values():
        method["lower"], method["upper"] = (
            np.minimum(method["lower"], method["upper"]),
            np.maximum(method["lower"], method["upper"]),
        )

    print("Plotting Pareto curve...")
    plot_pareto_curve(methods_data, y_true)

    print("Plotting time dynamics...")
    plot_time_dynamics(y_true, methods_data)

    print("Visualization testing completed!")
