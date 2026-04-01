# 🐌 snail-shell

> A股日频收益率区间预测框架，基于 LightGBM 分位数回归与软拉回机制

---

## 项目概述

**snail-shell** 是一个针对A股日频数据的区间预测系统。核心创新是**软拉回机制**：用分位数回归输出的不确定性半径，动态约束点预测的偏离程度，防止模型在高波动、regime 切换时产生极端预测。

锚点轨迹在特征空间中呈现出近似对数螺线的形态（观察结论，非先验假设），外扩速度作为模型失效的几何预警指标，项目因此得名。

**基座模型**：LightGBM（四个独立模型）  
**预测目标**：未来 $h=1$ 日收益率（规划扩展至 $h \in \{1, 5, 10\}$）  
**数据**：A股日频数据

---

## 核心机制

### 第一层：四模型输出

训练四个独立的 LightGBM 模型，共享输入特征：

| 模型 | objective | alpha | 输出 | 用途 |
|---|---|---|---|---|
| 点预测器 | `regression` | — | $\hat{y}_t$ | 软拉回的被拉对象 |
| 分位数 q10 | `quantile` | 0.1 | $\hat{y}_{q_{10},t}$ | 区间下界 |
| 分位数 q50 | `quantile` | 0.5 | $\hat{y}_{q_{50},t}$ | 锚点 $a_t$ |
| 分位数 q90 | `quantile` | 0.9 | $\hat{y}_{q_{90},t}$ | 区间上界 |

损失函数（Pinball Loss）：

$$\mathcal{L}_q(y, \hat{y}) = \max\left(q(y-\hat{y}),\ (q-1)(y-\hat{y})\right)$$

### 第二层：可信圆

$$a_t = \hat{y}_{q_{50},t}, \qquad r_t = \frac{\hat{y}_{q_{90},t} - \hat{y}_{q_{10},t}}{2}$$

- $a_t$：锚点，代表预测的中心趋势
- $r_t$：可信圆半径，代表模型的不确定性

### 第三层：软拉回机制（核心）

$$\alpha_t = \exp\left(-\beta \cdot \frac{|\hat{y}_t - a_t|}{r_t}\right)$$

$$\hat{y}_t^* = \alpha_t \cdot \hat{y}_t + (1 - \alpha_t) \cdot a_t$$

$\beta$ 是插值控制器，控制拉回强度：

| $\beta$ | $\alpha_t$（偏离 $= r_t$） | $\alpha_t$（偏离 $= 2r_t$） | 行为 | 角色 |
|---|---|---|---|---|
| $0$ | 1.000 | 1.000 | 完全不拉回 | 对照组（退化纯 QR） |
| $0.5$ | 0.607 | 0.368 | 温和拉回 | Snail-0.5 |
| $1$ | 0.368 | 0.135 | 标准拉回 | Snail-1 |
| $2$ | 0.135 | 0.018 | 强拉回 | Snail-2 |
| $5$ | 0.007 | 0.000 | 近似截断 | Snail-5 |
| $\infty$ | 0.000 | 0.000 | 完全拉回 | 对照组（退化纯 q50） |

### 第四层：螺旋监控

将二维锚点 $\mathbf{p}_t = (a_t, r_t)$ 投影到极坐标，拟合对数螺线：

$$\log \rho_t = \log A + B\theta_t$$

外扩速度：

$$v_t = \frac{\rho_t - \rho_{t-1}}{\theta_t - \theta_{t-1} + \epsilon}$$

失效预警（滚动窗口 $W=60$）：

$$\text{Alert}_t = \mathbf{1}\left[v_t > \mu_{v,t} + 2\sigma_{v,t}\right]$$

> **定位声明**：螺旋监控目前是**事后归因工具**，$R^2$ 和外扩速度作为观察变量报告，不作为实时交易信号。螺线形态为观察结论而非先验假设，若 $R^2 < 0.5$ 如实记录。

---

## 数据规格

**输入特征** $\mathbf{x}_t \in \mathbb{R}^d$：

- 过去 $T=20$ 天收益率序列
- 成交量变化率
- 动量因子（5日、20日）

**滚动 z-score 标准化**（窗口 $W=60$，前60天使用 expanding window）：

$$\tilde{x}_{t,i} = \frac{x_{t,i} - \mu_{t-W:t,i}}{\sigma_{t-W:t,i} + \epsilon}, \qquad \epsilon = 10^{-8}$$

**数据切分**：

| 集合 | 时间范围 | 用途 |
|---|---|---|
| 训练集 | 2019-01-01 ~ 2021-12-31 | 模型训练 |
| 验证集 Q1 | 2022-01-01 ~ 2022-03-31 | CP calibration split |
| 验证集 Q2~Q4 | 2022-04-01 ~ 2022-12-31 | $\beta$ 选择与评估 |
| 测试集 | 2023-01-01 ~ 2024-12-31 | 最终评估 |

---

## 实验设计

### 核心假设

**声明A（区间质量）**：在A股日频非平稳数据上，蜗牛壳的区间在 Coverage Error 相近的前提下，Winkler Score 低于 Conformal Prediction；目标是在高波动、regime 切换等特定子场景下经验性地优于 CP。

**声明B（点预测质量）**：蜗牛壳的 MAE 和 RankIC 优于纯 QR(q50) 和 MSE baseline，说明软拉回机制对点预测有增益。

### 对照组

| 组别 | 区间来源 | 点预测来源 | $\beta$ |
|---|---|---|---|
| Residual | 点预测 ± 1.28σ 残差 | MSE LightGBM | — |
| CP | EnbPI（calibration: 2022 Q1） | q50 | — |
| QR | Pinball Loss 无拉回 | MSE LightGBM | $0$ |
| Q50-only | — | 直接输出 $a_t$ | $\infty$ |
| Snail-0.5 | Pinball Loss + 软拉回 | MSE LightGBM | $0.5$ |
| Snail-1 | Pinball Loss + 软拉回 | MSE LightGBM | $1$ |
| Snail-2 | Pinball Loss + 软拉回 | MSE LightGBM | $2$ |
| Snail-5 | Pinball Loss + 软拉回 | MSE LightGBM | $5$ |

**CP baseline 规格**：
- Nonconformity score：$|y_t - \hat{y}_{q_{50},t}|$
- Calibration split：2022 Q1
- 区间构造：$\hat{y}_{q_{50}} \pm \text{quantile}_{0.9}(\text{residuals on calibration set})$

### $\beta$ 选择标准（验证集复合指标）

$$\text{Score} = \bar{W} + 10 \cdot \max(0,\ \text{CE} - 0.05)$$

$\beta$ 的选择与模型超参调优解耦：先在训练集确定 LightGBM 超参，再在验证集 Q2~Q4 上选 $\beta$。

---

## 评估指标

评估体系分两块，职责分明：

### 区间质量（区分 Residual / CP / QR）

**Coverage Error**（第一关，硬约束）：

$$\text{CE} = \left|\frac{1}{N}\sum_{t=1}^N \mathbf{1}[\hat{y}_{q_{10},t} \leq y_t \leq \hat{y}_{q_{90},t}] - 0.8\right|$$

**Winkler Score**（第二关，主要胜负标准，越低越好）：

$$W_t = \begin{cases} (q_{90,t}-q_{10,t}) + 10 \cdot (q_{10,t}-y_t) & y_t < q_{10,t} \\ q_{90,t}-q_{10,t} & q_{10,t} \leq y_t \leq q_{90,t} \\ (q_{90,t}-q_{10,t}) + 10 \cdot (y_t-q_{90,t}) & y_t > q_{90,t} \end{cases}$$

**区间宽度**（辅助）：$\text{IW} = \frac{1}{N}\sum_t (q_{90,t} - q_{10,t})$

### 点预测质量（区分 Snail 各变体）

**MAE**：$\text{MAE} = \frac{1}{N}\sum_t |\hat{y}_t^* - y_t|$

**RankIC**：$\text{RankIC}_t = \text{Spearman}(\text{rank}(\hat{y}_t^*),\ \text{rank}(y_t))$

### 统计显著性

对核心指标差异补充配对 t 检验：

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \qquad d_t = W_t^{\text{Snail}} - W_t^{\text{CP}}$$

### 第三关：2022 Q2~Q4 单独报告

所有指标在 2022 Q2~Q4 段单独计算，这是 regime 切换最密集的区间，是最有说服力的场景。

---

## 可视化

**图一：Pareto 曲线** —— 所有方法在 Coverage vs Width 空间的散点图，目标是左下角（覆盖率高且区间窄）。

**图二：时间动态图** —— Winkler Score vs Time，重点观察 2022 段各方法的稳定性对比。

---

## 完整数据流

```
输入特征 x_t
    ↓ 滚动 z-score（W=60，前60天 expanding window）
┌──────────────────────────────────────┐
│  LightGBM MSE + early_stop → ŷ_t    │  点预测
│  LightGBM q10 + early_stop → q̂₁₀  │
│  LightGBM q50 + early_stop → a_t    │  锚点
│  LightGBM q90 + early_stop → q̂₉₀  │
└──────────────────────────────────────┘
    ↓ Quantile Crossing 后处理 → 统计 Crossing Rate
r_t = (q̂₉₀ - q̂₁₀) / 2               可信圆半径
    ↓
α_t = exp(-β · |ŷ_t - a_t| / r_t)
    ↓
ŷ_t* = α_t · ŷ_t + (1-α_t) · a_t    → MAE / RankIC（区分 Snail 变体）
[q̂₁₀, q̂₉₀]                          → Coverage Error / Winkler（对比 CP）
p_t = (a_t, r_t)                      → 螺旋监控（v_t 加 ε 防爆炸）
```

---

## 仓库结构

```
snail-shell/
  core/
    quantile_head.py       # Pinball Loss + 四模型训练
    snail_mechanism.py     # 可信圆 + 软拉回（β 控制）
    spiral_monitor.py      # 二维锚点 + 螺旋拟合 + 外扩速度
  experiments/
    baseline_lgbm.py       # Residual / CP / QR / Q50-only
    snail_lgbm.py          # Snail-0.5/1/2/5
  evaluation/
    metrics.py             # CE / Winkler / IW / MAE / RankIC
    visualize.py           # Pareto 曲线 / 时间动态图
  notes/
    math_derivation.md     # 数学推导记录
  README.md
```

---

## Limitation

1. **覆盖率保证**：蜗牛壳无理论覆盖率保证，CE 的控制依赖 Pinball Loss 的经验校准质量。
2. **$\beta$ 选择**：最优 $\beta$ 在验证集上选取，存在对验证集过拟合的风险；$\beta$ 选择与模型超参调优解耦以缓解此问题。
3. **螺线形态**：锚点轨迹是否呈对数螺线为观察结论，非先验假设，$R^2$ 作为形态质量指标一并报告。
4. **Crossing Rate**：若 > 10%，说明分位数估计质量有限，拉回机制可能掩盖了模型本身的缺陷。
5. **数据范围**：实验仅覆盖 A 股日频数据，结论泛化性待验证。
6. **螺旋监控**：目前是事后归因工具，不作为实时交易信号。

---

## 相关工作

- Romano et al. (2019). *Conformalized Quantile Regression*. NeurIPS.
- Xu & Xie (2021). *Conformal Prediction Interval for Dynamic Time Series*. ICML.
- Gibbs & Candès (2021). *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS.

---

*snail-shell v2.0 · A股日频区间预测框架*