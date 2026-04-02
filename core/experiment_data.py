"""
experiment_data.py - 数据集类

封装所有用于模型训练和评估的特征和标签数据集。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentData:
    """
    实验数据封装类

    包含用于模型训练、验证和测试的所有数据集，
    以减少跨函数调用时参数传递的复杂性。
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val_q2_4: Optional[np.ndarray] = None
    y_val_q2_4: Optional[np.ndarray] = None
