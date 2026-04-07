# 🐌 snail-shell
This repository accompanies the paper:
["GSPQR: A Geometric Soft-Pullback Mechanism for Quantile Interval Prediction in A-Share Markets"](./GSPQR__A_Geometric_Soft_Pullback_Mechanism_for_Quantile_Interval_Prediction_in_A_Share_Markets%20.pdf)


对应版本使用 'git switch symmetric-v1.0' 可以切换到对称版本


> A股日频收益率区间预测框架，基于 LightGBM 分位数回归与软拉回机制

**snail-shell** 是一个针对A股日频数据的区间预测系统。核心创新是**软拉回机制**：用分位数回归输出的不确定性半径，动态约束点预测的偏离程度，防止模型在高波动、regime 切换时产生极端预测。

锚点轨迹在特征空间中呈现出近似对数螺线的形态（观察结论，非先验假设），外扩速度作为模型失效的几何预警指标，项目因此得名。

---

## 目录结构概要

- **`core/`**: 核心算法实现模块，包括特征加载、分位数回归头、软拉回机制和螺旋监控器。
- **`evaluation/`**: 评估模块，包括用于计算区间质量、点预测质量和复合评分等数学指标的 `metrics.py`，以及可视化组件 `visualize.py`。
- **`experiments/`**: 实验脚本模块，负责执行基准方法（如残差法、Conformal Prediction）和 Snail 软拉回机制变体的训练与测试流程。
- **`examples/`**: 存放示例代码（如测试数据加载 `test_data_loading.py` 等）。
- **`notes/`**: 存放相关的文档笔记，例如数学推导记录。
- **`tests/`**: 用于测试代码（当前为空）。

---

## 数学实现与对应代码

本章节按照仓库代码为标准，详细介绍各核心机制的数学推导，并提供对应的 Python 代码实现。

### 1. Pinball Loss (分位数回归)

**数学公式：**
$$\mathcal{L}_q(y, \hat{y}) = \max\left(q(y-\hat{y}),\ (q-1)(y-\hat{y})\right)$$

**代码实现：** (`core/quantile_head.py` & `evaluation/metrics.py`)
```python
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """
    计算Pinball Loss（分位数回归损失函数）

    公式: L_q(y, ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))
    """
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))
```

### 2. 可信圆（锚点与半径）

利用10%、50%和90%分位数的预测值，可以构建可信圆：

**数学公式：**
$$a_t = \hat{y}_{q_{50},t}$$
$$r_t = \frac{\hat{y}_{q_{90},t} - \hat{y}_{q_{10},t}}{2}$$

- $a_t$：锚点，代表预测的中心趋势
- $r_t$：可信圆半径，代表模型的不确定性

**代码实现：** (`core/quantile_head.py`)
```python
    def predict_anchor_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测锚点和可信圆半径

        锚点: a_t = ŷ_{q50,t}
        半径: r_t = (ŷ_{q90,t} - ŷ_{q10,t}) / 2
        """
        predictions = self.predict(X)

        anchor = predictions["q50"]
        radius = (predictions["q90"] - predictions["q10"]) / 2

        return anchor, radius
```

### 3. 软拉回机制

动态调整点预测结果 $\hat{y}_t$，受制于不确定性半径和控制参数 $\beta$。

**数学公式：**
$$\alpha_t = \begin{cases} \mathbf{1}\left[\frac{|\hat{y}_t - a_t|}{r_t} == 0\right], & \text{if } \beta = \infty \\ \exp\left(-\beta \cdot \frac{|\hat{y}_t - a_t|}{r_t}\right), & \text{otherwise} \end{cases}$$
$$q_{10,t}^* = \hat{y}_t^* - r_t$$
$$q_{90,t}^* = \hat{y}_t^* + r_t$$

**代码实现：** (`core/snail_mechanism.py`)
```python
def calculate_alpha(
    point_pred: np.ndarray, anchor: np.ndarray, radius: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    epsilon = 1e-8
    radius_safe = np.maximum(radius, epsilon)
    deviation = np.abs(point_pred - anchor) / radius_safe

    # β=∞ 时应完全拉回（α=0），但 deviation==0（点预测 == 锚点）时不需拉回（α=1）
    if np.isinf(beta):
        alpha = np.where(deviation == 0.0, 1.0, 0.0)
    else:
        alpha = np.exp(-beta * deviation)

    return alpha

def soft_pullback(
    point_pred: np.ndarray, anchor: np.ndarray, radius: np.ndarray, beta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 计算α_t
    alpha = calculate_alpha(point_pred, anchor, radius, beta)

    # 软拉回（点预测）
    corrected_pred = alpha * point_pred + (1 - alpha) * anchor

    # 关键：同步移动区间中心，保持半径不变
    corrected_q10 = corrected_pred - radius
    corrected_q90 = corrected_pred + radius

    return corrected_pred, corrected_q10, corrected_q90
```

### 4. 螺旋监控 (Logarithmic Spiral)

将二维锚点 $\mathbf{p}_t = (a_t, r_t)$ 投影到极坐标，拟合对数螺线。

#### 极坐标转换与螺线方程
**数学公式：**
$$\rho_t = \sqrt{a_t^2 + r_t^2}$$
$$\theta_t = \text{arctan2}(r_t, a_t)$$
$$\log \rho_t = \log A + B\theta_t$$

**代码实现：** (`core/spiral_monitor.py`)
```python
def cartesian_to_polar(a: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(a**2 + r**2)
    theta = np.arctan2(r, a)
    return rho, theta

def log_spiral_model(theta: np.ndarray, log_A: float, B: float) -> np.ndarray:
    return log_A + B * theta
```

#### 外扩速度
**数学公式：**
$$v_t = \frac{\rho_t - \rho_{t-1}}{\theta_t - \theta_{t-1} + \epsilon}$$

**代码实现：** (`core/spiral_monitor.py`)
```python
def calculate_expansion_velocity(
    rho: np.ndarray, theta: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    if len(rho) < 2:
        return np.array([])
    delta_rho = np.diff(rho)
    delta_theta = np.diff(theta)
    velocity = delta_rho / (delta_theta + epsilon)
    return velocity
```

#### 失效预警
滚动窗口计算：
**数学公式：**
$$\text{Alert}_t = \mathbf{1}\left[v_t > \mu_{v,t} + \text{threshold\_sigma} \cdot \sigma_{v,t}\right]$$

**代码实现：** (`core/spiral_monitor.py`)
```python
def rolling_alert(
    velocity: np.ndarray, window: int = 60, threshold_sigma: float = 2.0
) -> np.ndarray:
    n = len(velocity)
    alerts = np.zeros(n)
    if n > window:
        vel_series = pd.Series(velocity)
        rolling_window = vel_series.shift(1).rolling(window=window)
        mu = rolling_window.mean().to_numpy()
        sigma = rolling_window.std(ddof=0).to_numpy()

        mask = np.zeros(n, dtype=bool)
        mask[window:] = velocity[window:] > (mu[window:] + threshold_sigma * sigma[window:])
        alerts[mask] = 1
    return alerts
```

### 5. 评估指标体系

评估分为区间质量指标与点预测质量指标。

#### 5.1 Coverage Error (CE)
**数学公式：**
$$\text{CE} = \left|\frac{1}{N}\sum_{t=1}^N \mathbf{1}[q_{10,t} \leq y_t \leq q_{90,t}] - \text{target\_coverage}\right|$$

**代码实现：** (`evaluation/metrics.py`)
```python
def coverage_error(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, target_coverage: float = 0.8,
) -> float:
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)
    return np.abs(coverage - target_coverage)
```

#### 5.2 Winkler Score
评估区间的综合质量（包括宽度和未覆盖时的惩罚）。
**数学公式：**
$$W_t = \begin{cases} (q_{90,t}-q_{10,t}) + \frac{2}{\alpha} \cdot (q_{10,t}-y_t) & y_t < q_{10,t} \\ q_{90,t}-q_{10,t} & q_{10,t} \leq y_t \leq q_{90,t} \\ (q_{90,t}-q_{10,t}) + \frac{2}{\alpha} \cdot (y_t-q_{90,t}) & y_t > q_{90,t} \end{cases}$$
*(注意：默认 alpha = 0.2, 对应惩罚系数为 10)*

**代码实现：** (`evaluation/metrics.py`)
```python
def winkler_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2
) -> float:
    width = upper - lower
    penalty_lower = np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0)
    penalty_upper = np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0)
    winkler = width + penalty_lower + penalty_upper
    return np.mean(winkler)
```

#### 5.3 复合评分 (Composite Score)
用于 $\beta$ 的选取。
**数学公式：**
$$\text{Score} = \bar{W} + 10 \cdot \max(0,\ \text{CE} - 0.05)$$

**代码实现：** (`evaluation/metrics.py`)
```python
def composite_score(
    winkler_mean: float, coverage_error: float, target_ce: float = 0.05
) -> float:
    return winkler_mean + 10 * max(0, coverage_error - target_ce)
```

#### 5.4 RankIC & MAE (点预测质量)
**数学公式：**
$$\text{MAE} = \frac{1}{N}\sum_t |\hat{y}_t^* - y_t|$$
$$\text{RankIC} = \text{Spearman}(\text{rank}(\hat{y}_t^*),\ \text{rank}(y_t))$$

**代码实现：** (`evaluation/metrics.py`)
```python
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return stats.spearmanr(y_pred, y_true).correlation
```

#### 5.5 交叉率 (Crossing Rate)
检测分位数模型的预测质量。
**数学公式：**
$$\text{Crossing\_Rate} = \frac{\sum \mathbf{1}[q_{10,t} > q_{90,t}]}{N}$$

**代码实现：** (`evaluation/metrics.py`)
```python
def crossing_rate(q10: np.ndarray, q90: np.ndarray) -> float:
    crossing_mask = q10 > q90
    crossing_count = np.sum(crossing_mask)
    total_count = len(q10)
    return crossing_count / total_count if total_count > 0 else 0.0
```
