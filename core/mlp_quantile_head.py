"""
mlp_quantile_head.py - PyTorch MLP 分位数回归头

用单个多输出 MLP（共享特征提取器 + 四个独立输出头）同时预测：
  - point：MSE 损失（点预测）
  - q10 / q50 / q90：Pinball Loss（分位数回归）

设计原则：
  - CPU 友好：3 层 MLP，宽度 [256, 128, 64]，BatchNorm + Dropout
  - 多任务训练：四个头共享底层表示，单次前向传播产生全部输出
  - 早停：监控验证集总损失，patience 轮无改善则停止

接口与 QuantileHead（LightGBM）完全一致，可直接替换。

依赖：
  pip install torch>=2.0.0
"""

import sys
import os
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_quantile_head import BaseQuantileHead, FitConfig


# ---------------------------------------------------------------------------
# Pinball Loss（PyTorch）
# ---------------------------------------------------------------------------

def _pinball_loss(pred, target, alpha: float):
    """逐元素 Pinball Loss，返回标量均值。"""
    import torch
    residual = target - pred
    loss = torch.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
    return loss.mean()


# ---------------------------------------------------------------------------
# MLP 网络结构
# ---------------------------------------------------------------------------

def _build_network(in_features: int, hidden: list, dropout: float):
    """构建共享特征提取器 + 四输出头的 MLP。"""
    import torch.nn as nn

    layers = []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h

    backbone = nn.Sequential(*layers)

    # 四个独立输出头（各输出 1 个标量）
    head_point = nn.Linear(prev, 1)
    head_q10   = nn.Linear(prev, 1)
    head_q50   = nn.Linear(prev, 1)
    head_q90   = nn.Linear(prev, 1)

    return backbone, head_point, head_q10, head_q50, head_q90


# ---------------------------------------------------------------------------
# MLPQuantileHead
# ---------------------------------------------------------------------------

class MLPQuantileHead(BaseQuantileHead):
    """
    四输出 MLP 分位数回归头（PyTorch CPU 后端）

    共享 Backbone + 四个独立输出头，单次前向传播产生全部预测。
    """

    backend_name = "MLP"

    def __init__(
        self,
        hidden_dims     : list = None,
        dropout         : float = 0.1,
        lr              : float = 1e-4,
        batch_size      : int = 1024,
        max_epochs      : int = 200,
        patience        : int = 20,
        loss_weights    : Dict[str, float] = None,
        random_state    : int = 42,
    ):
        """
        参数:
            hidden_dims  : 隐藏层宽度列表，默认 [256, 128, 64]
            dropout      : Dropout 概率
            lr           : Adam 学习率
            batch_size   : 小批量大小
            max_epochs   : 最大训练轮数
            patience     : 早停等待轮数（验证集总损失无改善）
            loss_weights : 各损失权重，默认 {"point":1, "q10":1, "q50":1, "q90":1}
            random_state : 随机种子
        """
        self.hidden_dims  = hidden_dims or [256, 128, 64]
        self.dropout      = dropout
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.loss_weights = loss_weights or {"point": 1.0, "q10": 1.0, "q50": 1.0, "q90": 1.0}
        self.random_state = random_state

        self._backbone    = None
        self._heads       = None   # dict: point / q10 / q50 / q90
        self.is_fitted    = False

    # ------------------------------------------------------------------
    def fit(self, config: FitConfig) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # NaN 填充：z-score 标准化后 0 = 均值，等价于"该特征无信号"
        X_train_clean = np.nan_to_num(config.X_train, nan=0.0)
        X_tr = torch.tensor(X_train_clean, dtype=torch.float32)
        y_tr = torch.tensor(config.y_train, dtype=torch.float32).unsqueeze(1)

        has_val = config.X_val is not None
        if has_val:
            X_val_clean = np.nan_to_num(config.X_val, nan=0.0)
            X_va = torch.tensor(X_val_clean, dtype=torch.float32)
            y_va = torch.tensor(config.y_val, dtype=torch.float32).unsqueeze(1)

        in_features = X_tr.shape[1]
        backbone, h_pt, h_q10, h_q50, h_q90 = _build_network(
            in_features, self.hidden_dims, self.dropout
        )
        self._backbone = backbone
        self._heads    = {"point": h_pt, "q10": h_q10, "q50": h_q50, "q90": h_q90}

        # 汇总所有参数
        all_params = list(backbone.parameters())
        for h in self._heads.values():
            all_params += list(h.parameters())
        optimizer = torch.optim.Adam(all_params, lr=self.lr)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size, shuffle=True,
        )

        best_val_loss = float("inf")
        no_improve    = 0
        best_state    = None

        print(f"Training MLP ({self.hidden_dims}, max_epochs={self.max_epochs}, patience={self.patience})...")

        for epoch in range(1, self.max_epochs + 1):
            backbone.train()
            for h in self._heads.values():
                h.train()

            for xb, yb in loader:
                optimizer.zero_grad()
                feat  = backbone(xb)
                loss  = (
                    self.loss_weights["point"] * nn.MSELoss()(self._heads["point"](feat), yb)
                    + self.loss_weights["q10"] * _pinball_loss(self._heads["q10"](feat), yb, 0.1)
                    + self.loss_weights["q50"] * _pinball_loss(self._heads["q50"](feat), yb, 0.5)
                    + self.loss_weights["q90"] * _pinball_loss(self._heads["q90"](feat), yb, 0.9)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                optimizer.step()

            # 早停：在验证集上评估
            if has_val:
                backbone.eval()
                for h in self._heads.values():
                    h.eval()
                with torch.no_grad():
                    feat_v = backbone(X_va)
                    val_loss = (
                        nn.MSELoss()(self._heads["point"](feat_v), y_va)
                        + _pinball_loss(self._heads["q10"](feat_v), y_va, 0.1)
                        + _pinball_loss(self._heads["q50"](feat_v), y_va, 0.5)
                        + _pinball_loss(self._heads["q90"](feat_v), y_va, 0.9)
                    ).item()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    no_improve    = 0
                    best_state    = {
                        "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
                        **{name: {k: v.clone() for k, v in h.state_dict().items()}
                           for name, h in self._heads.items()}
                    }
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                    break

                if epoch % 20 == 0:
                    print(f"  [Epoch {epoch:3d}] val_loss={val_loss:.6f}")

        # 恢复最优权重
        if best_state is not None:
            backbone.load_state_dict(best_state["backbone"])
            for name, h in self._heads.items():
                h.load_state_dict(best_state[name])

        backbone.eval()
        for h in self._heads.values():
            h.eval()

        self.is_fitted = True
        print("MLP training complete!")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("MLPQuantileHead: 模型未训练，请先调用 fit()")

        import torch
        self._backbone.eval()
        for h in self._heads.values():
            h.eval()

        with torch.no_grad():
            X_t  = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)
            feat = self._backbone(X_t)
            return {
                "point" : self._heads["point"](feat).squeeze(1).numpy(),
                "q10"   : self._heads["q10"](feat).squeeze(1).numpy(),
                "q50"   : self._heads["q50"](feat).squeeze(1).numpy(),
                "q90"   : self._heads["q90"](feat).squeeze(1).numpy(),
            }
