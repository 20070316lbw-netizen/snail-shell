"""
multimodel_snail.py - 多后端 AS-GSPQR 对比实验

在相同数据切分下，用不同分位数回归后端（LightGBM / XGBoost / CatBoost）
训练四分位头，然后统一套上 AS-GSPQR 非对称软拉回机制进行对比。

每个后端只训练一次，β 扫描在推断阶段完成（无需重复训练）。

运行方式：
  python experiments/multimodel_snail.py
  python experiments/multimodel_snail.py --backends lgbm xgb      # 指定后端
  python experiments/multimodel_snail.py --beta 1.0               # 指定单一 β
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Type

# 抑制已知的无害 warning
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply.*", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_quantile_head import BaseQuantileHead, FitConfig
from core.quantile_head import QuantileHead
from core.asymmetric_mechanism import asymmetric_soft_pullback
from evaluation.metrics import (
    actual_coverage,
    coverage_error,
    winkler_score,
    interval_width,
    mean_absolute_error,
    rank_ic,
    composite_score,
)


# ---------------------------------------------------------------------------
# 后端注册表（按需导入，避免未安装的库报错）
# ---------------------------------------------------------------------------

def _load_backends(names: List[str]) -> Dict[str, BaseQuantileHead]:
    """按名称加载后端类，未安装的库跳过并警告。"""
    registry: Dict[str, Type[BaseQuantileHead]] = {}

    if "lgbm" in names:
        registry["LightGBM"] = QuantileHead

    if "xgb" in names:
        try:
            from core.xgb_quantile_head import XGBQuantileHead
            registry["XGBoost"] = XGBQuantileHead
        except ImportError:
            print("⚠️  XGBoost 未安装，跳过（pip install xgboost>=2.0.0）")

    if "catboost" in names:
        try:
            from core.catboost_quantile_head import CatBoostQuantileHead
            registry["CatBoost"] = CatBoostQuantileHead
        except ImportError:
            print("⚠️  CatBoost 未安装，跳过（pip install catboost）")

    if "mlp" in names:
        try:
            from core.mlp_quantile_head import MLPQuantileHead
            registry["MLP"] = MLPQuantileHead
        except ImportError:
            print("⚠️  PyTorch 未安装，跳过（pip install torch>=2.0.0）")

    return registry


# ---------------------------------------------------------------------------
# 单后端实验：训练 + β 扫描
# ---------------------------------------------------------------------------

def _run_one_backend(
    backend_name: str,
    head_cls: Type[BaseQuantileHead],
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    X_val_q2_4, y_val_q2_4,
    beta_values: List[float],
) -> List[Dict]:
    """
    训练一个后端，扫描所有 β，返回行列表（每行对应一个 β）。
    """
    print(f"\n{'='*60}")
    print(f"  后端: {backend_name}")
    print(f"{'='*60}")

    # 训练（只跑一次）
    head = head_cls()
    config = FitConfig(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    head.fit(config)

    # 预测（只推断一次，β 扫描不需要重新预测）
    preds_test    = head.predict(X_test)
    preds_val_q24 = head.predict(X_val_q2_4) if X_val_q2_4 is not None else None

    rows = []
    for beta in beta_values:
        # 测试集
        center_t, lower_t, upper_t, _ = asymmetric_soft_pullback(
            preds_test["point"], preds_test["q50"],
            preds_test["q10"],   preds_test["q90"],
            beta=beta,
        )
        cov_t = actual_coverage(y_test, lower_t, upper_t)
        ce_t  = coverage_error(y_test, lower_t, upper_t)
        ws_t  = winkler_score(y_test, lower_t, upper_t)
        iw_t  = interval_width(lower_t, upper_t)
        mae_t = mean_absolute_error(y_test, center_t)
        ric_t = rank_ic(center_t, y_test)
        cs_t  = composite_score(ws_t, ce_t)

        row = {
            "backend"         : backend_name,
            "beta"            : beta,
            "method"          : f"AS-{backend_name}-β={beta}",
            # 测试集
            "test_coverage"   : round(cov_t, 4),
            "test_ce"         : round(ce_t,  4),
            "test_winkler"    : round(ws_t,  4),
            "test_width"      : round(iw_t,  4),
            "test_mae"        : round(mae_t, 4),
            "test_rank_ic"    : round(ric_t, 4),
            "test_composite"  : round(cs_t,  4),
        }

        # 验证集 Q2-Q4（若提供）
        if preds_val_q24 is not None:
            center_v, lower_v, upper_v, _ = asymmetric_soft_pullback(
                preds_val_q24["point"], preds_val_q24["q50"],
                preds_val_q24["q10"],   preds_val_q24["q90"],
                beta=beta,
            )
            cov_v = actual_coverage(y_val_q2_4, lower_v, upper_v)
            ce_v  = coverage_error(y_val_q2_4, lower_v, upper_v)
            ws_v  = winkler_score(y_val_q2_4, lower_v, upper_v)
            cs_v  = composite_score(ws_v, ce_v)
            row.update({
                "val_coverage"  : round(cov_v, 4),
                "val_ce"        : round(ce_v,  4),
                "val_winkler"   : round(ws_v,  4),
                "val_composite" : round(cs_v,  4),
            })

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_multimodel(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    X_val_q2_4 = None,
    y_val_q2_4 = None,
    beta_values: List[float] = None,
    backends: List[str] = None,
) -> pd.DataFrame:
    """
    多后端 AS-GSPQR 对比实验主入口。

    参数:
        X_train/y_train : 训练集（早停验证）
        X_val/y_val     : 早停验证集
        X_test/y_test   : 测试集（最终评估）
        X_val_q2_4      : 验证集 Q2-Q4（Regime Shift 区），可选
        y_val_q2_4      : 同上
        beta_values     : β 网格，默认 [0.5, 1.0, 2.0, 5.0]
        backends        : 后端列表，默认 ["lgbm", "xgb", "catboost"]

    返回:
        汇总结果 DataFrame
    """
    if beta_values is None:
        beta_values = [0.5, 1.0, 2.0, 5.0]
    if backends is None:
        backends = ["lgbm", "xgb", "catboost"]

    loaded = _load_backends(backends)
    if not loaded:
        raise RuntimeError("没有可用的后端，请至少安装一个（lgbm / xgb / catboost）")

    all_rows = []
    for name, cls in loaded.items():
        rows = _run_one_backend(
            name, cls,
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            X_val_q2_4, y_val_q2_4,
            beta_values,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df


def _print_results(df: pd.DataFrame) -> None:
    """格式化打印结果表格。"""
    test_cols = ["method", "test_coverage", "test_ce", "test_winkler",
                 "test_width", "test_mae", "test_rank_ic", "test_composite"]
    print("\n" + "=" * 80)
    print("📊 多后端 AS-GSPQR 对比结果（测试集 2023-2024）")
    print("=" * 80)
    print(df[test_cols].sort_values("test_composite").to_string(index=False))

    if "val_composite" in df.columns:
        val_cols = ["method", "val_coverage", "val_ce", "val_winkler", "val_composite"]
        print("\n" + "=" * 80)
        print("📊 验证集 Q2-Q4（Regime Shift 密集区）")
        print("=" * 80)
        print(df[val_cols].sort_values("val_composite").to_string(index=False))

    # 按后端汇总最优 β
    print("\n" + "=" * 80)
    print("🏆 各后端最优 β（按测试集 composite_score）")
    print("=" * 80)
    best = df.loc[df.groupby("backend")["test_composite"].idxmin(),
                  ["backend", "beta", "test_ce", "test_winkler", "test_composite"]]
    print(best.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多后端 AS-GSPQR 对比实验")
    parser.add_argument(
        "--backends", nargs="+",
        default=["lgbm", "xgb", "catboost", "mlp"],
        choices=["lgbm", "xgb", "catboost", "mlp"],
        help="要测试的后端（默认全部）",
    )
    parser.add_argument(
        "--beta", nargs="+", type=float,
        default=[0.5, 1.0, 2.0, 5.0],
        help="β 网格（默认 0.5 1.0 2.0 5.0）",
    )
    args = parser.parse_args()

    # 加载数据（与 main.py 的 _load_data 保持完全一致）
    import sys, os
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

    print("📂 连接数据库，加载特征数据 ...")
    with DataLoader() as loader:
        df = loader.get_features()

    # 按 ticker 做滚动 z-score（与 main.py 一致）
    def _zscore(group):
        roll = group[FEATURE_COLS].rolling(window=60, min_periods=1)
        group[FEATURE_COLS] = (group[FEATURE_COLS] - roll.mean()) / (roll.std() + 1e-8)
        return group
    df = df.groupby("ticker", group_keys=False).apply(_zscore)

    def _slice(s, e):
        mask = (df["date"].astype(str) >= str(s)) & (df["date"].astype(str) <= str(e))
        sub = df[mask]
        return sub[FEATURE_COLS].values.astype(np.float32), sub[LABEL_COL].values.astype(np.float32)

    sp = DATA_SPLIT
    X_train,  y_train  = _slice(sp["train_start"],    sp["train_end"])
    X_val,    y_val    = _slice(sp["val_q1_start"],   sp["val_q1_end"])
    X_val_q24, y_val_q24 = _slice(sp["val_q2_4_start"], sp["val_q2_4_end"])
    X_test,   y_test   = _slice(sp["test_start"],     sp["test_end"])

    print(f"   Train   : {len(y_train):6d} 条   {sp['train_start']} ~ {sp['train_end']}")
    print(f"   Val Q1  : {len(y_val):6d} 条   {sp['val_q1_start']} ~ {sp['val_q1_end']}")
    print(f"   Val Q2-4: {len(y_val_q24):6d} 条   {sp['val_q2_4_start']} ~ {sp['val_q2_4_end']}")
    print(f"   Test    : {len(y_test):6d} 条   {sp['test_start']} ~ {sp['test_end']}")

    df_results = run_multimodel(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_val_q2_4 = X_val_q24,
        y_val_q2_4 = y_val_q24,
        beta_values = args.beta,
        backends    = args.backends,
    )

    _print_results(df_results)

    # 保存
    os.makedirs("results", exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/multimodel_{ts}.csv"
    df_results.to_csv(out, index=False)
    print(f"\n✅ 结果已保存至: {out}")
