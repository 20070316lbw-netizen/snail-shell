"""
export_for_kaggle.py - 导出处理好的特征数据供 Kaggle 使用

将本地 DuckDB 中的特征数据按标准切分处理后，存成 .npz 文件。
上传到 Kaggle Dataset 后，实验脚本直接 np.load() 即可，无需 DuckDB。

输出文件（保存到 kaggle_export/ 目录）：
  snail_data.npz  包含以下 key：
    X_train, y_train   训练集   2019-01-01 ~ 2021-12-31
    X_val,   y_val     验证集 Q1 2022-01-01 ~ 2022-03-31
    X_val_q24, y_val_q24  验证集 Q2-4 2022-04-01 ~ 2022-12-31
    X_test,  y_test    测试集   2023-01-01 ~ 2024-12-31

用法：
  python scripts/export_for_kaggle.py
  python scripts/export_for_kaggle.py --out-dir my_export/
"""

import sys
import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply.*", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import DataLoader
from config import DATA_SPLIT

FEATURE_COLS = [
    "mom_20d", "mom_60d", "mom_12m_minus_1m",
    "vol_60d_res", "sp_ratio", "turn_20d",
    "mom_20d_rank", "mom_60d_rank", "mom_12m_minus_1m_rank",
    "vol_60d_res_rank", "sp_ratio_rank", "turn_20d_rank",
]
LABEL_COL = "label_next_month"


def export(out_dir: str = "kaggle_export") -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("📂 连接数据库，加载全量特征...")
    with DataLoader() as loader:
        start, end = loader.get_date_range()
        print(f"   数据库日期范围: {start} ~ {end}")
        df = loader.get_features()

    print(f"   原始行数: {len(df):,}")

    # 按 ticker 滚动 z-score（与 main.py 完全一致）
    def _zscore(group):
        roll = group[FEATURE_COLS].rolling(window=60, min_periods=1)
        group[FEATURE_COLS] = (group[FEATURE_COLS] - roll.mean()) / (roll.std() + 1e-8)
        return group

    print("⚙️  计算滚动 z-score 标准化...")
    df = df.groupby("ticker", group_keys=False).apply(_zscore)

    # 按日期切分
    def _slice(s, e):
        mask = (df["date"].astype(str) >= str(s)) & (df["date"].astype(str) <= str(e))
        sub = df[mask]
        return (
            sub[FEATURE_COLS].values.astype(np.float32),
            sub[LABEL_COL].values.astype(np.float32),
        )

    sp = DATA_SPLIT
    print("✂️  按标准日期切分...")
    X_train,   y_train   = _slice(sp["train_start"],    sp["train_end"])
    X_val,     y_val     = _slice(sp["val_q1_start"],   sp["val_q1_end"])
    X_val_q24, y_val_q24 = _slice(sp["val_q2_4_start"], sp["val_q2_4_end"])
    X_test,    y_test    = _slice(sp["test_start"],     sp["test_end"])

    print(f"   Train   : {len(y_train):6,} 条   {sp['train_start']} ~ {sp['train_end']}")
    print(f"   Val Q1  : {len(y_val):6,} 条   {sp['val_q1_start']} ~ {sp['val_q1_end']}")
    print(f"   Val Q2-4: {len(y_val_q24):6,} 条   {sp['val_q2_4_start']} ~ {sp['val_q2_4_end']}")
    print(f"   Test    : {len(y_test):6,} 条   {sp['test_start']} ~ {sp['test_end']}")

    # 保存
    out_path = os.path.join(out_dir, "snail_data.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train,   y_train=y_train,
        X_val=X_val,       y_val=y_val,
        X_val_q24=X_val_q24, y_val_q24=y_val_q24,
        X_test=X_test,     y_test=y_test,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n✅ 已保存至: {out_path}  ({size_mb:.1f} MB)")
    print("   上传到 Kaggle Dataset 后在 notebook 里用：")
    print("   data = np.load('/kaggle/input/<dataset>/snail_data.npz')")
    print("   X_train, y_train = data['X_train'], data['y_train']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出特征数据供 Kaggle 使用")
    parser.add_argument("--out-dir", default="kaggle_export", help="输出目录（默认 kaggle_export/）")
    args = parser.parse_args()
    export(args.out_dir)
