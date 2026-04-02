"""
experiment_data.py - 实验数据容器

包含所有的训练集、验证集、测试集的划分。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentData:
    """
    实验数据类，封装所有的训练、验证、测试集数据。
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_val_q2_4: Optional[np.ndarray] = None
    y_val_q2_4: Optional[np.ndarray] = None
