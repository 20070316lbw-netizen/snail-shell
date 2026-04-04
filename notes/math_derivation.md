# 数学推导记录

## 1. Pinball Loss（分位数回归损失函数）

### 定义
对于分位数 $q \in (0,1)$，Pinball Loss定义为：

$$\mathcal{L}_q(y, \hat{y}) = \max\left(q(y-\hat{y}),\ (q-1)(y-\hat{y})\right)$$

### 等价形式
可以重写为：
$$\mathcal{L}_q(y, \hat{y}) = \begin{cases}
q(y-\hat{y}) & \text{if } y \geq \hat{y} \\
(1-q)(\hat{y}-y) & \text{if } y < \hat{y}
\end{cases}$$

### 梯度
对 $\hat{y}$ 求导：
$$\frac{\partial \mathcal{L}_q}{\partial \hat{y}} = \begin{cases}
-q & \text{if } y \geq \hat{y} \\
1-q & \text{if } y < \hat{y}
\end{cases}$$

### 特殊情况
- 当 $q = 0.5$ 时，Pinball Loss等价于MAE（平均绝对误差）
- 当 $q = 0.9$ 时，对低估的惩罚是对高估惩罚的9倍

---

## 2. 软拉回机制

### 核心公式
软拉回机制通过以下公式修正点预测：

$$\hat{y}_t^* = \alpha_t \cdot \hat{y}_t + (1 - \alpha_t) \cdot a_t$$

$$q_{10,t}^* = \hat{y}_t^* - r_t, \qquad q_{90,t}^* = \hat{y}_t^* + r_t$$

### 核心逻辑说明
1. **点预测修正**：$\hat{y}_t^*$ 被拉向锚点 $a_t$。
2. **区间重构**：**关键步骤**。软拉回不仅修正点预测，还以 $\hat{y}_t^*$ 为中心重新构建预测区间 $[q_{10,t}^*, q_{90,t}^*]$，但保持其半径 $r_t$ 不变。这确保了 $\beta$ 的调整能直接影响 Coverage 和 Winkler Score。

### 参数解释
- $\hat{y}_t$：原始点预测（MSE LightGBM输出）
- $a_t$：锚点（q50分位数回归输出）
- $r_t$：可信圆半径
- $\beta$：插值控制器（拉回强度参数）
- $\alpha_t$：混合系数（0-1之间）

### 几何解释
1. **当 $|\hat{y}_t - a_t| \ll r_t$**：
   - $\alpha_t \approx 1$，$\hat{y}_t^* \approx \hat{y}_t$
   - 点预测几乎不受影响

2. **当 $|\hat{y}_t - a_t| \approx r_t$**：
   - $\alpha_t = \exp(-\beta)$
   - 拉回强度由 $\beta$ 控制

3. **当 $|\hat{y}_t - a_t| \gg r_t$**：
   - $\alpha_t \to 0$，$\hat{y}_t^* \to a_t$
   - 点预测被强烈拉向锚点

### β值的行为表

| β | α_t (偏离=r_t) | α_t (偏离=2r_t) | 行为 | 角色 |
|---|----------------|------------------|------|------|
| 0 | 1.000 | 1.000 | 完全不拉回 | 对照组（退化纯QR） |
| 0.5 | 0.607 | 0.368 | 温和拉回 | Snail-0.5 |
| 1 | 0.368 | 0.135 | 标准拉回 | Snail-1 |
| 2 | 0.135 | 0.018 | 强拉回 | Snail-2 |
| 5 | 0.007 | 0.000 | 近似截断 | Snail-5 |
| ∞ | 0.000 | 0.000 | 完全拉回 | 对照组（退化纯q50） |

---

## 3. 可信圆

### 定义
可信圆由锚点和半径定义：

$$a_t = \hat{y}_{q_{50},t}$$

$$r_t = \frac{\hat{y}_{q_{90},t} - \hat{y}_{q_{10},t}}{2}$$

### 置信区间
对应的80%置信区间为：

$$[\hat{y}_{q_{10},t}, \hat{y}_{q_{90},t}] = [a_t - r_t, a_t + r_t]$$

### 几何意义
- 锚点 $a_t$：预测的中心趋势
- 半径 $r_t$：模型的不确定性度量
- 可信圆：以锚点为中心，半径为 $r_t$ 的区间

---

## 4. 螺旋监控

### 极坐标转换
将二维锚点 $\mathbf{p}_t = (a_t, r_t)$ 投影到极坐标：

$$\rho_t = \sqrt{a_t^2 + r_t^2}$$

$$\theta_t = \arctan2(r_t, a_t)$$

### 对数螺线拟合
观察发现锚点轨迹近似对数螺线：

$$\log \rho_t = \log A + B\theta_t$$

等价于：
$$\rho_t = A \cdot e^{B\theta_t}$$

### 外扩速度
外扩速度定义为：

$$v_t = \frac{\rho_t - \rho_{t-1}}{\theta_t - \theta_{t-1} + \epsilon}$$

其中 $\epsilon = 10^{-8}$ 防止除零。

### 失效预警
使用滚动窗口 $W=60$ 计算预警信号：

$$\text{Alert}_t = \mathbf{1}\left[v_t > \mu_{v,t} + 2\sigma_{v,t}\right]$$

其中：
- $\mu_{v,t}$：过去 $W$ 个时间点的速度均值（不含当前点 $t$）
- $\sigma_{v,t}$：过去 $W$ 个时间点的速度标准差（不含当前点 $t$）

---

## 5. 评估指标

### Coverage Error (CE)
$$\text{CE} = \left|\frac{1}{N}\sum_{t=1}^N \mathbf{1}[\hat{y}_{q_{10},t} \leq y_t \leq \hat{y}_{q_{90},t}] - 0.8\right|$$

### Winkler Score
$$W_t = \begin{cases} 
(q_{90,t}-q_{10,t}) + \frac{2}{\alpha} \cdot (q_{10,t}-y_t) & y_t < q_{10,t} \\ 
q_{90,t}-q_{10,t} & q_{10,t} \leq y_t \leq q_{90,t} \\ 
(q_{90,t}-q_{10,t}) + \frac{2}{\alpha} \cdot (y_t-q_{90,t}) & y_t > q_{90,t} 
\end{cases}$$

其中 $\alpha = 0.2$ 对应80%置信区间，惩罚系数 $\frac{2}{\alpha} = 10$。

### 复合评分
$$\text{Score} = \bar{W} + 10 \cdot \max(0,\ \text{CE} - 0.05)$$

### RankIC
$$\text{RankIC} = \text{Spearman}(\text{rank}(\hat{y}_t^*),\ \text{rank}(y_t))$$

---

## 6. 数据预处理

### 滚动 z-score 标准化
$$\tilde{x}_{t,i} = \frac{x_{t,i} - \mu_{t-W:t,i}}{\sigma_{t-W:t,i} + \epsilon}, \qquad \epsilon = 10^{-8}$$

其中：
- $W = 60$：滚动窗口大小
- 前60天使用expanding window（从第一个样本开始到当前时间点）

### 输入特征
$\mathbf{x}_t \in \mathbb{R}^d$：
- 过去 $T=20$ 天收益率序列
- 成交量变化率
- 动量因子（5日、20日）

---

## 7. 实验设计

### 数据切分
| 集合 | 时间范围 | 用途 |
|---|---|---|
| 训练集 | 2019-01-01 ~ 2021-12-31 | 模型训练 |
| 验证集 Q1 | 2022-01-01 ~ 2022-03-31 | CP calibration split |
| 验证集 Q2~Q4 | 2022-04-01 ~ 2022-12-31 | β 选择与评估 |
| 测试集 | 2023-01-01 ~ 2024-12-31 | 最终评估 |

### β选择标准
在验证集 Q2~Q4 上使用复合指标选择β：

$$\beta^* = \arg\min_{\beta} \left\{ \bar{W}_\beta + 10 \cdot \max(0, \text{CE}_\beta - 0.05) \right\}$$

### 对照组
| 组别 | 区间来源 | 点预测来源 | β |
|---|---|---|---|
| Residual | 点预测 ± 1.28σ 残差 | MSE LightGBM | — |
| CP | EnbPI（calibration: 2022 Q1） | q50 | — |
| QR | Pinball Loss 无拉回 | MSE LightGBM | 0 |
| Q50-only | — | 直接输出 $a_t$ | ∞ |
| Snail-0.5 | Pinball Loss + 软拉回 | MSE LightGBM | 0.5 |
| Snail-1 | Pinball Loss + 软拉回 | MSE LightGBM | 1 |
| Snail-2 | Pinball Loss + 软拉回 | MSE LightGBM | 2 |
| Snail-5 | Pinball Loss + 软拉回 | MSE LightGBM | 5 |

---

## 8. 理论性质

### 分位数回归的一致性
对于分位数回归，当样本量趋于无穷时，估计量收敛到真实分位数：

$$\hat{q}_\tau \xrightarrow{p} q_\tau$$

### 软拉回的收缩性质
软拉回机制是一个收缩估计器：

$$\|\hat{y}_t^* - a_t\| \leq \|\hat{y}_t - a_t\|$$

即修正后的预测总是比原始预测更接近锚点。

### 螺旋监控的预警性质
如果外扩速度持续高于历史水平，可能表明：
1. 市场regime发生切换
2. 模型预测能力下降
3. 需要重新训练或调整参数

---

## 9. 未来扩展

### 多期预测
规划扩展至 $h \in \{1, 5, 10\}$ 日收益率预测：
$$\hat{y}_{t,h} = f_h(\mathbf{x}_t)$$

### 动态β调整
考虑根据市场状态动态调整β值：
$$\beta_t = g(\text{volatility}_t, \text{trend}_t, \ldots)$$

### 集成方法
集成多个β值的预测：
$$\hat{y}_t^{**} = \sum_{\beta} w_\beta \cdot \hat{y}_{t,\beta}^*$$

---

*最后更新：2026-04-05*