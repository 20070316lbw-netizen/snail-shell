import numpy as np
import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())

from core.data_loader import DataLoader
from core.quantile_head import QuantileHead, FitConfig
from core.snail_mechanism import SnailMechanism
from core.spiral_monitor import SpiralMonitor

def generate_report(date, metrics, alerts, top_gainers):
    report_content = f"""# 🐌 Snail-Shell 每日市场预测报告 ({date})

## 1. 市场概况 & 螺旋监控
| 指标 | 状态/数值 | 备注 |
| :--- | :--- | :--- |
| **对数螺线拟合 R²** | {metrics['r_squared']:.4f} | { "✅ 拟合良好" if metrics['r_squared'] >= 0.5 else "⚠️ 形态模糊" } |
| **平均外扩速度 v_t** | {metrics['mean_velocity']:.6f} | |
| **失效预警 (Alert)** | { "🔴 触发预警" if metrics['alert_rate'] > 0 else "🟢 正常" } | 滚动窗口 W=60 |
| **预警率** | {metrics['alert_rate']:.2%} | 近期触发频次 |

## 2. 软拉回修正 (Snail-1.0)
本报告基于 β=1.0 的软拉回机制，通过不确定性半径动态修正点预测。

### 预测涨幅 Top 10 标的
| 股票代码 | 修正后预测收益率 (%) | 80% 置信区间 [q10, q90] | 备注 |
| :--- | :--- | :--- | :--- |
"""
    rows = [
        f"| {row.ticker} | {row.corrected_pred*100:.2f}% | [{row.q10*100:.2f}%, {row.q90*100:.2f}%] | |\n"
        for row in top_gainers.itertuples(index=False)
    ]

    rows.append(
        "\n## 3. 统计分布总结\n"
        f"- **样本数量**: {metrics['n_samples']}\n"
        f"- **平均预测半径**: {metrics['mean_radius']*100:.2f}%\n"
    )

    report_content += "".join(rows)
    
    report_path = f"reports/daily/report_{date.replace('-', '')}.md"
    os.makedirs("reports/daily", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return report_path

def predict():
    # 1. 加载数据
    with DataLoader() as loader:
        df = loader.get_features()
    
    if df.empty:
        print("数据库中没有找到数据。")
        return

    # 获取最新日期
    latest_date = str(df['date'].max())
    df_latest = df[df['date'].astype(str) == latest_date].copy()
    
    print(f"正在为 {latest_date} 生成预测...")

    # 2. 训练/加载模型 (此处为了示例，每次运行简单训练)
    # 实际生产中应保存并加载 .pkl
    X_train_df = df[df['date'].astype(str) < "2024-01-01"] # 粗略划分训练集
    X_train = X_train_df.iloc[:, 4:16].values.astype(np.float32) # 使用特征列
    y_train = X_train_df['label_next_month'].values.astype(np.float32)
    
    qh = QuantileHead(n_estimators=200) # 快速训练
    qh.fit(FitConfig(X_train=X_train, y_train=y_train))
    
    # 3. 执行预测
    X_latest = df_latest.iloc[:, 4:16].values.astype(np.float32)
    
    # 获取锚点和半径
    anchor, radius = qh.predict_anchor_and_radius(X_latest)
    
    # 此处模拟点预测 (实际应用中应由另一个全回归模型输出，或使用 q50 作为基准)
    # 为了演示，直接用 q50 作为原始点预测
    point_pred = anchor 
    
    # 应用 Snail
    snail = SnailMechanism()
    corrected_pred, q10, q90, diagnostics = snail.apply(point_pred, anchor, radius, beta=1.0)
    
    df_latest['corrected_pred'] = corrected_pred
    df_latest['q10'] = q10
    df_latest['q90'] = q90
    
    # 4. 螺旋监控
    monitor = SpiralMonitor()
    res = monitor.analyze(anchor, radius)
    
    # 汇总指标
    metrics = {
        "r_squared": res.get("r_squared", 0),
        "mean_velocity": res.get("mean_velocity", 0),
        "alert_rate": res.get("alert_rate", 0),
        "n_samples": len(df_latest),
        "mean_radius": np.mean(radius)
    }
    
    # Top 10 标的
    top_gainers = df_latest.sort_values("corrected_pred", ascending=False).head(10)
    
    # 5. 生成报告
    report_path = generate_report(latest_date, metrics, res['alerts'], top_gainers)
    print(f"报告已生成: {report_path}")

if __name__ == "__main__":
    predict()
