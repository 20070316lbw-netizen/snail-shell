"""
quantile_head.py - Pinball Loss + 四模型训练（LightGBM 后端）

包含四个独立的LightGBM模型：
1. 点预测器（MSE回归）
2. 分位数q10（alpha=0.1）
3. 分位数q50（alpha=0.5，锚点）
4. 分位数q90（alpha=0.9）

继承自 BaseQuantileHead，接口与其他后端（XGBoost / CatBoost）完全一致。
"""

import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional

from core.base_quantile_head import BaseQuantileHead, FitConfig  # noqa: F401（re-export FitConfig）


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """
    计算Pinball Loss（分位数回归损失函数）

    公式: L_q(y, ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))

    参数:
        y_true: 真实值
        y_pred: 预测值
        alpha: 分位数参数（0-1之间）

    返回:
        Pinball Loss值
    """
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))


class QuantileHead(BaseQuantileHead):
    """
    四模型分位数回归头（LightGBM 后端）

    训练四个独立的LightGBM模型：
    - MSE回归器（点预测）
    - q10分位数回归器
    - q50分位数回归器（锚点）
    - q90分位数回归器
    """

    backend_name = "LightGBM"

    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        """
        初始化分位数回归头

        参数:
            params: LightGBM参数字典
            n_estimators: 树的数量
            learning_rate: 学习率
            early_stopping_rounds: 早停轮数
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        # 默认参数
        self.default_params = {
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        if params:
            self.default_params.update(params)

        # 初始化模型
        self.models = {
            "mse": None,  # 点预测器
            "q10": None,  # 分位数10%
            "q50": None,  # 分位数50%（锚点）
            "q90": None,  # 分位数90%
        }

        self.is_fitted = False

    def _create_model(
        self, objective: str, alpha: Optional[float] = None
    ) -> lgb.LGBMRegressor:
        """
        创建LightGBM模型

        参数:
            objective: 目标函数类型
            alpha: 分位数参数（仅用于quantile目标）

        返回:
            LightGBM回归器
        """
        params = self.default_params.copy()

        if objective == "regression":
            params["objective"] = "regression"
            params["metric"] = "rmse"
        elif objective == "quantile":
            params["objective"] = "quantile"
            params["alpha"] = alpha
            params["metric"] = "quantile"
        else:
            raise ValueError(f"Unknown objective: {objective}")

        return lgb.LGBMRegressor(**params)

    def fit(
        self,
        config: FitConfig,
    ) -> None:
        """
        训练四个模型

        参数:
            config: 训练配置类实例
        """
        X_train, y_train = config.X_train, config.y_train
        X_val, y_val = config.X_val, config.y_val
        callbacks = config.callbacks

        # 准备验证集
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        eval_names = ["validation"] if eval_set else None

        # 如果没有提供回调函数，使用早停
        if callbacks is None and eval_set:
            callbacks = [
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(100),
            ]

        # 训练MSE点预测器
        print("Training MSE predictor...")
        self.models["mse"] = self._create_model("regression")
        self.models["mse"].fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
        )

        # 训练q10分位数回归器
        print("Training q10 quantile regressor...")
        self.models["q10"] = self._create_model("quantile", alpha=0.1)
        self.models["q10"].fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
        )

        # 训练q50分位数回归器（锚点）
        print("Training q50 quantile regressor (anchor)...")
        self.models["q50"] = self._create_model("quantile", alpha=0.5)
        self.models["q50"].fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
        )

        # 训练q90分位数回归器
        print("Training q90 quantile regressor...")
        self.models["q90"] = self._create_model("quantile", alpha=0.9)
        self.models["q90"].fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
        )

        self.is_fitted = True
        print("All models trained successfully!")

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测所有分位数和点预测

        参数:
            X: 特征矩阵

        返回:
            包含所有预测的字典
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")

        predictions = {}

        # 点预测
        predictions["point"] = self.models["mse"].predict(X)

        # 分位数预测
        predictions["q10"] = self.models["q10"].predict(X)
        predictions["q50"] = self.models["q50"].predict(X)
        predictions["q90"] = self.models["q90"].predict(X)

        return predictions

    def predict_anchor_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测锚点和可信圆半径

        锚点: a_t = ŷ_{q50,t}
        半径: r_t = (ŷ_{q90,t} - ŷ_{q10,t}) / 2

        参数:
            X: 特征矩阵

        返回:
            (锚点, 半径) 元组
        """
        predictions = self.predict(X)

        anchor = predictions["q50"]
        radius = (predictions["q90"] - predictions["q10"]) / 2

        return anchor, radius

    def get_model_info(self) -> Dict[str, Dict]:
        """
        获取模型信息

        返回:
            模型信息字典
        """
        if not self.is_fitted:
            return {}

        info = {}
        for name, model in self.models.items():
            if model is not None:
                info[name] = {
                    "n_estimators": model.n_estimators_,
                    "best_iteration": model.best_iteration_,
                    "feature_importances": model.feature_importances_,
                }

        return info


if __name__ == "__main__":
    # 简单测试
    print("Testing QuantileHead...")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5

    # 分割数据集
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # 创建并训练模型
    qh = QuantileHead(n_estimators=100, early_stopping_rounds=20)
    config = FitConfig(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    qh.fit(config)

    # 预测
    predictions = qh.predict(X_val[:5])
    print("\nPredictions for first 5 samples:")
    for key, values in predictions.items():
        print(f"{key}: {values}")

    # 锚点和半径
    anchor, radius = qh.predict_anchor_and_radius(X_val[:5])
    print(f"\nAnchor: {anchor}")
    print(f"Radius: {radius}")

    # 模型信息
    info = qh.get_model_info()
    print(f"\nModel info keys: {list(info.keys())}")
