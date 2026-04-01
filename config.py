"""
config.py - 项目配置文件

包含项目路径、数据库配置等
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据库路径
DATABASE_PATH = PROJECT_ROOT / "QQ_Quant_DB" / "quant_lab.duckdb"

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 创建目录（如果不存在）
for dir_path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
]:
    dir_path.mkdir(exist_ok=True)

# 模型参数默认值
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 软拉回参数
BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")]

# 数据切分日期（来自README）
DATA_SPLIT = {
    "train_start": "2019-01-01",
    "train_end": "2021-12-31",
    "val_q1_start": "2022-01-01",
    "val_q1_end": "2022-03-31",  # CP calibration split
    "val_q2_4_start": "2022-04-01",
    "val_q2_4_end": "2022-12-31",  # β selection
    "test_start": "2023-01-01",
    "test_end": "2024-12-31",
}

# 评估参数
EVALUATION_PARAMS = {
    "target_coverage": 0.8,  # 80%置信区间
    "winkler_alpha": 0.2,  # 1-alpha = 80%
    "crossing_threshold": 0.1,  # 分位数交叉率阈值
    "spiral_window": 60,  # 螺旋监控窗口
    "spiral_alert_sigma": 2.0,  # 预警阈值（σ倍数）
}


def get_database_path():
    """获取数据库路径"""
    return str(DATABASE_PATH)


def get_project_info():
    """获取项目信息"""
    return {
        "name": "snail-shell",
        "version": "2.0",
        "description": "A股日频收益率区间预测框架",
        "database_path": str(DATABASE_PATH),
        "project_root": str(PROJECT_ROOT),
    }
