# GSPQR 数学推导与完整证明

> 本文档对项目中所有数学命题进行分类标注：
> - **[Theorem + Proof]**：严格可证命题，含完整证明
> - **[Empirical Observation]**：仅有实证支持，无法在一般情况下严格证明
> - **[Design Choice]**：工程设计决策，无需证明，但需说明动机
>
> *最后更新：2026-04-06*

---

## 1. Pinball Loss（分位数回归损失）

### 定义

对分位数 $\tau \in (0, 1)$，Pinball Loss 定义为：

$$\mathcal{L}_\tau(y, \hat{y}) = \max\bigl(\tau(y - \hat{y}),\ (\tau - 1)(y - \hat{y})\bigr)$$

### Theorem 1.1：等价分段形式

$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

**Proof.**

分两种情形讨论。

**情形 1：** $y \geq \hat{y}$，即 $y - \hat{y} \geq 0$。

由于 $\tau \in (0,1)$，有 $\tau > \tau - 1$，故当 $y - \hat{y} \geq 0$ 时，$\tau(y-\hat{y}) \geq (\tau-1)(y-\hat{y})$。

因此 $\mathcal{L}_\tau = \tau(y - \hat{y})$。

**情形 2：** $y < \hat{y}$，即 $y - \hat{y} < 0$。

此时 $\tau(y-\hat{y}) < 0$ 且 $(\tau-1)(y-\hat{y}) > 0$（因 $\tau-1 < 0$，$y-\hat{y} < 0$，负负得正）。

因此 $\mathcal{L}_\tau = (\tau-1)(y-\hat{y}) = (1-\tau)(\hat{y}-y)$。

综合两种情形，等价分段形式得证。$\blacksquare$

---

### Theorem 1.2：Pinball Loss 关于 $\hat{y}$ 是凸函数

**Proof.**

由 Theorem 1.1，$\mathcal{L}_\tau$ 是两个关于 $\hat{y}$ 的仿射函数的逐点最大值。有限个凸函数的逐点最大值仍为凸函数（标准结论）。$\blacksquare$

---

### Theorem 1.3：梯度（次梯度）

$$\frac{\partial \mathcal{L}_\tau}{\partial \hat{y}} = \begin{cases} -\tau & \text{if } y > \hat{y} \\ 1 - \tau & \text{if } y < \hat{y} \end{cases}$$

在 $y = \hat{y}$ 处函数不可微，次梯度集合为 $[\tau - 1,\ -\tau]$（包含 0，与该点为局部最小值一致）。

**Proof.**

当 $y > \hat{y}$ 时，$\mathcal{L}_\tau = \tau(y-\hat{y})$，对 $\hat{y}$ 求导得 $-\tau$。

当 $y < \hat{y}$ 时，$\mathcal{L}_\tau = (1-\tau)(\hat{y}-y)$，对 $\hat{y}$ 求导得 $1-\tau$。

$y = \hat{y}$ 处次梯度为左右导数所围闭区间 $[\tau-1,\ -\tau]$。$\blacksquare$

---

### Theorem 1.4：当 $\tau = 0.5$ 时退化为 MAE

**Proof.**

代入 $\tau = 0.5$：$\mathcal{L}_{0.5} = 0.5 \cdot |y - \hat{y}|$。

最小化期望 $\mathbb{E}[\mathcal{L}_{0.5}]$ 等价于最小化 MAE，其最优解 $\hat{y}^* = \text{median}(y)$，与分位数回归 $\tau=0.5$ 的解一致。$\blacksquare$

---

## 2. 代理目标与自适应中心估计

### 问题设定

设在时间步 $t$，我们有两个预测器：
- $\hat{y}_t$：条件均值估计（MSE 模型）
- $a_t = \hat{q}_{0.5,t}$：条件中位数估计（q50 分位数模型）

**动机**：A 股市场的均值-中位数分离现象表明两者携带互补信息——均值捕捉期望收益，中位数对重尾更鲁棒。

---

### Theorem 2.1：代理目标的闭合解

**代理目标**（对固定 $\lambda_t \in [0,1]$，关于中心 $c$ 最小化加权 MSE）：

$$\hat{y}_t^* = \arg\min_{c \in \mathbb{R}} \Bigl\{ \lambda_t (c - \hat{y}_t)^2 + (1 - \lambda_t)(c - a_t)^2 \Bigr\}$$

**结论**：

$$\hat{y}_t^* = \lambda_t \hat{y}_t + (1 - \lambda_t) a_t \tag{1}$$

**Proof.**

设 $F(c) = \lambda_t(c - \hat{y}_t)^2 + (1-\lambda_t)(c-a_t)^2$，为关于 $c$ 的二次函数，开口向上（系数之和为 1 > 0），存在唯一最小值点。

令 $F'(c) = 0$：

$$2\lambda_t(c - \hat{y}_t) + 2(1-\lambda_t)(c - a_t) = 0$$

$$c(\lambda_t + 1 - \lambda_t) = \lambda_t \hat{y}_t + (1-\lambda_t)a_t$$

$$\therefore \hat{y}_t^* = \lambda_t \hat{y}_t + (1-\lambda_t)a_t \qquad \blacksquare$$

---

### 重要说明：代理目标到门控函数的关系

> **[Design Choice]** — 非推导关系，需明确区分。

Theorem 2.1 表明，对**任意** $\lambda_t \in [0,1]$，最优中心均为 $\hat{y}_t$ 与 $a_t$ 的凸组合。这说明凸组合是合理的中心估计族，但**并未规定** $\lambda_t$ 应如何选取。

GSPQR 选择以下门控函数：

$$\lambda_t = G(r_t, \delta_t) = \exp\!\left(-\beta \cdot \frac{|\delta_t|}{r_t}\right), \quad \delta_t = \hat{y}_t - a_t \tag{2}$$

这是**设计选择**，而非从优化问题推导而来。其合理性由 Section 3 的性质（极限行为、单调性、收缩性）保证。

---

## 3. 门控函数的性质

### 符号约定

- $\delta_t = \hat{y}_t - a_t$：均值与中位数之差（分歧）
- $r_t = (\hat{q}_{90,t} - \hat{q}_{10,t}) / 2$：不确定性半径
- $\eta_t = |\delta_t| / r_t$：归一化信噪比（SNR）
- $\beta \geq 0$：拉回强度参数

### Theorem 3.1：高不确定性时退化为均值预测

$$r_t \to \infty \implies \eta_t \to 0 \implies \lambda_t \to 1 \implies \hat{y}_t^* \to \hat{y}_t$$

**Proof.** 固定 $|\delta_t|$ 有限。$r_t \to \infty$ 时 $\eta_t \to 0$，$\lambda_t = \exp(-\beta\eta_t) \to 1$，代入 (1) 得 $\hat{y}_t^* \to \hat{y}_t$。$\blacksquare$

---

### Theorem 3.2：高置信度时退化为中位数预测

$$r_t \to 0^+,\ |\delta_t| > 0 \implies \eta_t \to +\infty \implies \lambda_t \to 0 \implies \hat{y}_t^* \to a_t$$

**Proof.** 固定 $|\delta_t| > 0$。$r_t \to 0^+$ 时 $\eta_t \to +\infty$，对 $\beta > 0$，$\lambda_t = \exp(-\beta\eta_t) \to 0$，代入 (1) 得 $\hat{y}_t^* \to a_t$。$\blacksquare$

---

### Theorem 3.3：$\beta$ 的极限行为

**(a)** $\beta = 0$：$\lambda_t \equiv 1$，$\hat{y}_t^* = \hat{y}_t$（Snail-0，纯 MSE 基准）。

**(b)** $\beta \to +\infty$，$|\delta_t| > 0$：$\lambda_t \to 0$，$\hat{y}_t^* \to a_t$（Snail-$\infty$，纯 Q50 基准）。

**(c)** $\beta \to +\infty$，$\delta_t = 0$：$\eta_t = 0$，$\lambda_t = 1$，但 $\hat{y}_t = a_t$，故 $\hat{y}_t^* = \hat{y}_t = a_t$，结论一致。

**Proof.**

(a) $\lambda_t = \exp(0) = 1$，直接代入。

(b) $\eta_t > 0$ 固定，$-\beta\eta_t \to -\infty$，$\exp(-\beta\eta_t) \to 0$。

(c) $\delta_t = 0 \Rightarrow \hat{y}_t = a_t$，任何 $\lambda_t$ 均给出相同结果。$\blacksquare$

---

### Theorem 3.4：$\lambda_t$ 关于 SNR 单调递减

$$\frac{\partial \lambda_t}{\partial \eta_t} = -\beta \exp(-\beta \eta_t) \leq 0$$

当 $\beta > 0$ 时严格递减：SNR 越高（分歧相对不确定性越大），拉回越强。

**Proof.** 直接对 $\lambda_t = \exp(-\beta\eta_t)$ 求导，利用 $\exp(\cdot) > 0$，$\beta \geq 0$。$\blacksquare$

---

## 4. 软拉回的几何性质

### Theorem 4.1：软拉回的收缩性

$$|\hat{y}_t^* - a_t| \leq |\hat{y}_t - a_t|$$

**Proof.**

$$\hat{y}_t^* - a_t = \lambda_t\hat{y}_t + (1-\lambda_t)a_t - a_t = \lambda_t(\hat{y}_t - a_t)$$

$$|\hat{y}_t^* - a_t| = \lambda_t|\hat{y}_t - a_t| \leq |\hat{y}_t - a_t| \qquad (\lambda_t \in [0,1]) \qquad \blacksquare$$

等号在 $\lambda_t = 1$（即 $\beta = 0$）时取到。

---

### Theorem 4.2：对称区间宽度不变性

$$q_{10,t}^* = \hat{y}_t^* - r_t,\quad q_{90,t}^* = \hat{y}_t^* + r_t \implies |q_{90,t}^* - q_{10,t}^*| = 2r_t$$

对所有 $\beta$、所有样本恒成立。

**Proof.** $q_{90,t}^* - q_{10,t}^* = 2r_t$，直接展开。$\blacksquare$

**推论**：所有 $\beta$ 变体的区间宽度相同，Winkler Score 的改善完全来自中心位置优化，而非区间压缩。

---

### 概念澄清：可信圆的两个层次

> **[澄清：旧笔记存在不一致，此处修正]**

| 概念 | 中心 | 区间 | 时机 |
|---|---|---|---|
| 原始可信圆 | $a_t$（锚点） | $[a_t - r_t,\ a_t + r_t]$ | 软拉回之前 |
| 最终预测区间 | $\hat{y}_t^*$（修正中心） | $[\hat{y}_t^* - r_t,\ \hat{y}_t^* + r_t]$ | 软拉回之后 |

两者半径相同（均为 $r_t$），但中心不同，边界不同。由 Theorem 4.1，修正中心 $\hat{y}_t^*$ 比 $\hat{y}_t$ 更接近 $a_t$，但不一定与 $a_t$ 重合。

---

## 5. 评估指标的性质

### 5.1 Winkler Score

**定义**（$\alpha = 0.2$，对应 80% 置信区间，惩罚系数 $2/\alpha = 10$）：

$$W_t = \begin{cases}
(q_{90,t} - q_{10,t}) + 10(q_{10,t} - y_t) & y_t < q_{10,t} \\[4pt]
q_{90,t} - q_{10,t} & q_{10,t} \leq y_t \leq q_{90,t} \\[4pt]
(q_{90,t} - q_{10,t}) + 10(y_t - q_{90,t}) & y_t > q_{90,t}
\end{cases}$$

### Theorem 5.1：Winkler Score 非负性

$$W_t \geq 0 \quad \forall\, t$$

**Proof.**

情形 1（区间内）：$W_t = 2r_t \geq 0$。

情形 2（低于下界）：$W_t = 2r_t + 10(q_{10,t} - y_t) \geq 2r_t \geq 0$。

情形 3（高于上界）：$W_t = 2r_t + 10(y_t - q_{90,t}) \geq 2r_t \geq 0$。$\blacksquare$

---

### [Empirical Observation 5.2]：Winkler Score 关于 $\beta$ 的凸性

论文 Section 4.4 报告，在 HS300 测试集上，$\bar{W}(\beta)$ 关于 $\beta$ 呈凸形，最小值在 $\beta^* \in [1.0, 2.0]$。

**说明**：此命题在当前数据集上成立，但**无法在一般情形下严格证明**。$\bar{W}(\beta)$ 的行为取决于联合分布 $(y_t, \hat{y}_t, a_t, r_t)$，无强分布假设时理论凸性无法建立。实证支持见表 2 及图 4。

---

### [局限性说明]：GSPQR 无形式化覆盖保证

GSPQR **不提供**形式化的有限样本覆盖保证，这与 Conformal Prediction（EnbPI）的根本区别。实验中观测覆盖率（84.23%）接近 80% 目标，但属于实证结果，而非理论保证。

---

### 5.2 复合评分

$$\text{Score}(\beta) = \bar{W}(\beta) + 10 \cdot \max(0,\ \text{CE}(\beta) - 0.05) \tag{3}$$

$\beta^*$ 通过在验证集（2022 Q2–Q4）上最小化公式 (3) 选取。惩罚项仅在覆盖误差超过 5 个百分点时激活，在校准约束下鼓励最小化区间宽度。

---

## 6. 分位数回归的统计性质

### [Theorem with Caveats 6.1]：分位数回归的一致性

设 $\{(x_t, y_t)\}$ 为 i.i.d. 样本，$y_t | x_t$ 的条件分布在目标分位数处有正密度，且模型类足够丰富，则 $n \to \infty$ 时：

$$\hat{q}_\tau(x) \xrightarrow{p} q_\tau(x)$$

**重要限制**：(1) A 股日频时序数据不满足 i.i.d. 假设；(2) LightGBM（树模型）的一致性需额外的树深/叶子数增长速率条件；(3) 跨截面设置下真实分位数函数跨股票异质，条件未严格验证。此结论仅作直觉参考，不应在本项目语境下作为严格理论基础引用。

---

## 7. 螺旋监控

### [Empirical Observation 7.1]：锚点轨迹的对数螺线形态

将 $(a_t, r_t)$ 投影到极坐标 $(\rho_t, \theta_t)$，其中 $\rho_t = \sqrt{a_t^2 + r_t^2}$，$\theta_t = \arctan2(r_t, a_t)$。

**实证观察**：在 HS300 数据上，$\log\rho_t$ 与 $\theta_t$ 存在近似线性关系，即 $\rho_t \approx Ae^{B\theta_t}$。

这是**观察性结论，非先验假设**（README 原文）。不同市场、不同时期的形态可能显著不同。

---

### Theorem 7.2：严格对数螺线下外扩速度为常数

若轨迹严格满足 $\rho = Ae^{B\theta}$，则相邻步的外扩速度：

$$v_t = \frac{\Delta\rho}{\Delta\theta} \approx \frac{d\rho}{d\theta} = ABe^{B\theta} = B\rho_t$$

为当前半径的 $B$ 倍，是关于 $\rho_t$ 的线性函数。实际数据偏离此值即为轨迹形态变化的预警信号。

**Proof.** 直接对 $\rho = Ae^{B\theta}$ 关于 $\theta$ 求导，代入差分近似。$\blacksquare$

---

### Theorem 7.3：滚动预警的统计意义

预警规则 $\text{Alert}_t = \mathbf{1}[v_t > \mu_{v,t} + 2\sigma_{v,t}]$。

若历史窗口内 $v_t$ 近似服从正态分布，则在零假设（无 regime 变化）下，$\text{Alert}_t = 1$ 的概率约为 $P(Z > 2) \approx 2.28\%$。超过此比例的触发率提示系统性异常。

**说明**：金融时序中正态假设通常不严格成立，上述概率为近似参考值。

---

## 8. 统计显著性（论文 Section 4.5）

### Theorem 8.1：配对 t 检验的正当性

对每个测试样本 $i$，令 $d_i = W_i^{\text{GSPQR}} - W_i^{\text{EnbPI}}$，检验 $H_0: \mathbb{E}[d_i] = 0$。

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} = -68.5,\quad n = 120{,}512,\quad p \ll 0.001$$

**有效性说明**：

1. **样本量**：$n = 120{,}512$ 保证由中心极限定理 $\bar{d}$ 渐近正态，无需 $d_i$ 自身正态。
2. **横截面相关性**：同一日期内不同股票的 $d_i$ 存在相关性，严格应使用聚类标准误。但鉴于 $|t| = 68.5$ 极大，结论的稳健性不受影响——即使对标准误进行保守调整，显著性依然成立。

---

## 附录：命题完整性总表

| 编号 | 命题 | 类型 | 状态 |
|---|---|---|---|
| Thm 1.1 | Pinball Loss 等价分段形式 | 严格证明 | ✅ |
| Thm 1.2 | Pinball Loss 关于 $\hat{y}$ 凸 | 严格证明 | ✅ |
| Thm 1.3 | Pinball Loss 次梯度 | 严格证明 | ✅ |
| Thm 1.4 | $\tau=0.5$ 退化为 MAE | 严格证明 | ✅ |
| Thm 2.1 | 代理目标闭合解 | 严格证明 | ✅ |
| Design | 门控函数 $G$ 的选取 | 设计选择 | ✅ 已澄清 |
| Thm 3.1 | 高不确定性→信任均值 | 严格证明 | ✅ |
| Thm 3.2 | 高置信度→信任中位数 | 严格证明 | ✅ |
| Thm 3.3 | $\beta$ 极限行为 | 严格证明 | ✅ |
| Thm 3.4 | $\lambda_t$ 关于 SNR 单调递减 | 严格证明 | ✅ |
| Thm 4.1 | 软拉回收缩性 | 严格证明 | ✅ |
| Thm 4.2 | 区间宽度不变性 | 严格证明 | ✅ |
| 澄清 | 可信圆两个层次区分 | 概念修正 | ✅ 已修复 |
| Thm 5.1 | Winkler Score 非负性 | 严格证明 | ✅ |
| Obs 5.2 | Winkler 关于 $\beta$ 凸性 | 实证观察 | ⚠️ 已标注 |
| 局限 | GSPQR 无覆盖保证 | 局限性声明 | ⚠️ 已标注 |
| Thm 6.1 | 分位数一致性（附条件） | 条件定理 | ⚠️ 条件已说明 |
| Obs 7.1 | 对数螺线形态 | 实证观察 | ⚠️ 已标注 |
| Thm 7.2 | 严格螺线下外扩速度 | 严格证明 | ✅ |
| Thm 7.3 | 滚动预警统计意义 | 近似分析 | ⚠️ 条件已说明 |
| Thm 8.1 | 配对 t 检验正当性 | 分析 | ✅ 条件已说明 |
