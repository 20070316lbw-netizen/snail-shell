"""
main.py - snail-shell 统一实验入口

用法：
  python main.py train   --mode [baseline|snail|all]  [--beta 1.0]
  python main.py eval    --results <results_dir>
  python main.py compare [--output results/comparison.csv]
  python main.py beta-select
  python main.py monitor [--beta 1.0]

示例：
  python main.py compare               # 运行完整对比实验（基线 + 蜗牛壳）
  python main.py train --mode snail --beta 1.0
  python main.py beta-select           # 在验证集上选最优 β
  python main.py monitor --beta 1.0   # 输出螺旋监控结果
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 确保项目根目录在 sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 延迟导入（避免在 help 时触发重依赖）────────────────────────────────────────
def _import_core():
    from config import DATA_SPLIT, BETA_VALUES, OUTPUTS_DIR, RESULTS_DIR, get_project_info
    from core.data_loader import DataLoader
    from core.quantile_head import QuantileHead
    from core.snail_mechanism import SnailMechanism
    from core.spiral_monitor import SpiralMonitor
    from experiments.baseline_lgbm import run_baseline_experiment
    from experiments.snail_lgbm import run_snail_experiment, select_best_beta
    from evaluation.metrics import evaluate_methods, calculate_all_metrics
    return {
        "DATA_SPLIT": DATA_SPLIT,
        "BETA_VALUES": BETA_VALUES,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "get_project_info": get_project_info,
        "DataLoader": DataLoader,
        "QuantileHead": QuantileHead,
        "SnailMechanism": SnailMechanism,
        "SpiralMonitor": SpiralMonitor,
        "run_baseline_experiment": run_baseline_experiment,
        "run_snail_experiment": run_snail_experiment,
        "select_best_beta": select_best_beta,
        "evaluate_methods": evaluate_methods,
        "calculate_all_metrics": calculate_all_metrics,
    }


# ── 数据加载辅助 ───────────────────────────────────────────────────────────────
def load_data(ctx):
    """
    从 DuckDB 加载数据并按 README 规格切分。

    返回:
        X_train, y_train, X_val_q1, y_val_q1,
        X_val, y_val, X_test, y_test
    """
    split = ctx["DATA_SPLIT"]
    DataLoader = ctx["DataLoader"]

    print("📂 连接数据库，加载特征数据 ...")
    with DataLoader() as loader:
        # 获取日期范围并打印提示
        start, end = loader.get_date_range()
        print(f"   数据库日期范围: {start} ~ {end}")

        feature_cols = [
            "mom_20d", "mom_60d", "mom_12m_minus_1m",
            "vol_60d_res", "sp_ratio", "turn_20d",
            "mom_20d_rank", "mom_60d_rank", "mom_12m_minus_1m_rank",
            "vol_60d_res_rank", "sp_ratio_rank", "turn_20d_rank",
        ]
        label_col = "label_next_month"

        df = loader.get_features()

        # Apply Rolling z-score Normalization by ticker
        # W=60, min_periods=1 for expanding window in first 60 days
        def zscore_normalize(group):
            roll = group[feature_cols].rolling(window=60, min_periods=1)
            mu = roll.mean()
            sigma = roll.std()
            group[feature_cols] = (group[feature_cols] - mu) / (sigma + 1e-8)
            return group

        df = df.groupby('ticker', group_keys=False).apply(zscore_normalize)

        def _slice(df, s, e):
            mask = (df["date"] >= s) & (df["date"] <= e)
            sub = df[mask].copy()
            X = sub[feature_cols].values.astype(np.float32)
            y = sub[label_col].values.astype(np.float32)
            return X, y
        if df.empty:
            raise RuntimeError("features_cn 表为空，请先向数据库中写入数据。")

    Xtr, ytr   = _slice(df, split["train_start"],       split["train_end"])
    Xvq1, yvq1 = _slice(df, split["val_q1_start"],      split["val_q1_end"])
    Xv,   yv   = _slice(df, split["val_q2_4_start"],    split["val_q2_4_end"])
    Xte,  yte  = _slice(df, split["test_start"],         split["test_end"])

    print(f"   Train  : {Xtr.shape[0]:>6} 条   {split['train_start']} ~ {split['train_end']}")
    print(f"   Val Q1 : {Xvq1.shape[0]:>6} 条   {split['val_q1_start']} ~ {split['val_q1_end']}")
    print(f"   Val Q2-4: {Xv.shape[0]:>6} 条  {split['val_q2_4_start']} ~ {split['val_q2_4_end']}")
    print(f"   Test   : {Xte.shape[0]:>6} 条   {split['test_start']} ~ {split['test_end']}")

    return Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte


def _mock_data(ctx, n=3000, n_feat=12, seed=42):
    """当数据库不可用时，生成模拟数据用于功能验证。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat).astype(np.float32)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + rng.randn(n) * 0.5).astype(np.float32)

    t, v = int(n * 0.6), int(n * 0.1)
    Xtr, ytr     = X[:t],          y[:t]
    Xvq1, yvq1   = X[t:t+v],       y[t:t+v]
    Xv,   yv     = X[t+v:t+2*v],   y[t+v:t+2*v]
    Xte,  yte    = X[t+2*v:],       y[t+2*v:]
    return Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte


# ── 打印结果表格 ───────────────────────────────────────────────────────────────
def _print_results_table(df: pd.DataFrame):
    cols_order = [
        "Method", "coverage_error", "winkler_score", "interval_width",
        "mae", "rank_ic", "composite_score",
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    df_show = df[cols_order].copy()

    float_cols = [c for c in df_show.columns if c != "Method"]
    for c in float_cols:
        df_show[c] = df_show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    sep = "─" * 110
    print(f"\n{sep}")
    print(df_show.to_string(index=False))
    print(sep)


# ── 子命令实现 ─────────────────────────────────────────────────────────────────

def cmd_compare(args, ctx):
    """运行完整对比实验（基线 + 所有蜗牛壳变体），输出对比表格。"""
    print("\n🐌 snail-shell — 完整对比实验")
    print("=" * 60)

    # 加载数据
    try:
        Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = load_data(ctx)
    except Exception as e:
        print(f"⚠️  数据库加载失败: {e}")
        print("   使用模拟数据继续运行（仅用于功能验证）...")
        Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = _mock_data(ctx)

    all_preds = {}  # 方法名 → {point_pred, lower, upper}

    # ── 基线实验 ────────────────────────────────────────────────
    print("\n📊 运行基线实验 ...")
    baseline_results = ctx["run_baseline_experiment"](
        Xtr, ytr, Xvq1, yvq1, Xte, yte, X_val_q2_4=Xv, y_val_q2_4=yv
    )
    all_preds.update(baseline_results)

    # ── 蜗牛壳变体 ──────────────────────────────────────────────
    betas = args.betas if args.betas else [0.5, 1.0, 2.0, 5.0]
    print(f"\n🐌 运行蜗牛壳变体 β={betas} ...")
    snail_results = ctx["run_snail_experiment"](
        Xtr, ytr, Xvq1, yvq1, Xte, yte, beta_values=betas, X_val_q2_4=Xv, y_val_q2_4=yv
    )
    all_preds.update(snail_results)

    # ── 汇总评估 (Test Set) ────────────────────────────────────────────────
    print("\n📈 测试集评估指标汇总：")
    df_eval = ctx["evaluate_methods"](yte, all_preds)
    _print_results_table(df_eval)

    # ── 汇总评估 (Val Q2-Q4) ─────────────────────────────────────────────
    print("\n📈 验证集 Q2-Q4 评估指标汇总 (Regime Shift密集区)：")
    val_q2_4_preds = {}
    for method, preds in all_preds.items():
        if "val_q2_4" in preds:
            val_q2_4_preds[method] = preds["val_q2_4"]
    if val_q2_4_preds:
        df_eval_val = ctx["evaluate_methods"](yv, val_q2_4_preds)
        _print_results_table(df_eval_val)

    # ── 统计显著性检验 ────────────────────────────────────────────────────
    print("\n🔬 统计显著性检验 (Test Set Winkler Score: Snail-1.0 vs CP)：")
    if "Snail-1.0" in all_preds and "CP" in all_preds:
        # Calculate individual Winkler Scores for Snail-1.0 and CP on Test Set
        from evaluation.metrics import paired_t_test
        import numpy as np

        # we need to calculate winkler score per sample to do t-test
        from config import EVALUATION_PARAMS
        alpha = EVALUATION_PARAMS.get("winkler_alpha", 0.2)

        def _get_winkler_array(y, lower, upper, alpha=alpha):
            width = upper - lower
            penalty_lower = np.where(y < lower, (2 / alpha) * (lower - y), 0)
            penalty_upper = np.where(y > upper, (2 / alpha) * (y - upper), 0)
            return width + penalty_lower + penalty_upper

        ws_snail = _get_winkler_array(yte, all_preds["Snail-1.0"]["lower"], all_preds["Snail-1.0"]["upper"])
        ws_cp = _get_winkler_array(yte, all_preds["CP"]["lower"], all_preds["CP"]["upper"])

        t_stat, p_val = paired_t_test(ws_snail, ws_cp)
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value:     {p_val:.4f}")
        if p_val < 0.05:
            print("   结论: Snail-1.0 与 CP 的差异在统计上是显著的 (p < 0.05)。")
        else:
            print("   结论: Snail-1.0 与 CP 的差异在统计上不显著 (p >= 0.05)。")

    # 保存结果
    out_path = Path(args.output) if args.output else (
        ctx["RESULTS_DIR"] / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(out_path, index=False)
    print(f"\n✅ 结果已保存至: {out_path}")


def cmd_train(args, ctx):
    """单独训练基线或蜗牛壳模型。"""
    print(f"\n🐌 snail-shell — 训练模式 [{args.mode}]")
    print("=" * 60)

    try:
        Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = load_data(ctx)
    except Exception as e:
        print(f"⚠️  数据库加载失败: {e}")
        print("   使用模拟数据继续运行...")
        Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = _mock_data(ctx)

    results = {}

    if args.mode in ("baseline", "all"):
        print("\n📊 运行基线实验 ...")
        baseline_results = ctx["run_baseline_experiment"](
            Xtr, ytr, Xvq1, yvq1, Xte, yte, X_val_q2_4=Xv, y_val_q2_4=yv
        )
        results.update(baseline_results)

    if args.mode in ("snail", "all"):
        betas = [args.beta] if args.beta else [0.5, 1.0, 2.0, 5.0]
        print(f"\n🐌 训练蜗牛壳模型 β={betas} ...")
        snail_results = ctx["run_snail_experiment"](
            Xtr, ytr, Xvq1, yvq1, Xte, yte, beta_values=betas, X_val_q2_4=Xv, y_val_q2_4=yv
        )
        results.update(snail_results)

    print("\n📈 测试集评估：")
    df_eval = ctx["evaluate_methods"](yte, results)
    _print_results_table(df_eval)

    out_path = ctx["RESULTS_DIR"] / f"train_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(out_path, index=False)
    print(f"\n✅ 结果已保存至: {out_path}")


def cmd_beta_select(args, ctx):
    """在验证集 Q2~Q4 上选择最优 β 值。"""
    print("\n🐌 snail-shell — β 选择")
    print("=" * 60)

    try:
        Xtr, ytr, _, _, Xv, yv, _, _ = load_data(ctx)
    except Exception as e:
        print(f"⚠️  数据库加载失败: {e}")
        print("   使用模拟数据继续运行...")
        Xtr, ytr, _, _, Xv, yv, _, _ = _mock_data(ctx)

    beta_candidates = ctx["BETA_VALUES"]
    print(f"   候选 β 值: {beta_candidates}")

    best_beta, beta_scores = ctx["select_best_beta"](Xtr, ytr, Xv, yv, beta_candidates)

    print(f"\n{'─'*60}")
    print(f"  {'β':>8}  {'Score':>10}  {'MAE':>10}  {'Winkler':>10}")
    print(f"{'─'*60}")
    for beta, info in beta_scores.items():
        m = info["metrics"]
        marker = " ◀ 最优" if beta == best_beta else ""
        print(f"  {str(beta):>8}  {m['score']:>10.4f}  {m['mae']:>10.4f}"
              f"  {m['winkler_mean']:>10.4f}{marker}")
    print(f"{'─'*60}")
    print(f"\n✅ 建议使用 β = {best_beta}")


def cmd_monitor(args, ctx):
    """训练模型并输出螺旋监控报告。"""
    print("\n🐌 snail-shell — 螺旋监控")
    print("=" * 60)

    try:
        Xtr, ytr, _, _, Xv, yv, Xte, yte = load_data(ctx)
    except Exception as e:
        print(f"⚠️  数据库加载失败: {e}")
        print("   使用模拟数据继续运行...")
        Xtr, ytr, _, _, Xv, yv, Xte, yte = _mock_data(ctx)

    beta = args.beta if args.beta else 1.0

    # 训练分位数回归头
    print(f"\n   训练 QuantileHead（β={beta}）...")
    qh = ctx["QuantileHead"]()
    qh.fit(Xtr, ytr, Xv, yv)

    # 测试集预测
    anchor, radius = qh.predict_anchor_and_radius(Xte)

    # 螺旋监控
    monitor = ctx["SpiralMonitor"]()
    spiral_result = monitor.analyze(anchor, radius)

    print(f"\n📐 螺旋监控结果：")
    print(f"  R²（对数螺线拟合）: {spiral_result.get('r_squared', 'N/A'):.4f}")
    print(f"  平均外扩速度 v_t  : {spiral_result.get('mean_velocity', 'N/A'):.6f}")
    alert_count = spiral_result.get("alert_count", "N/A")
    total_count = spiral_result.get("total_count", "N/A")
    print(f"  预警触发次数      : {alert_count} / {total_count}")

    if spiral_result.get("r_squared", 0) < 0.5:
        print("\n  ⚠️  R² < 0.5，锚点轨迹螺线形态不明显，监控结论可信度有限。")

    # 保存螺旋数据
    out_path = ctx["OUTPUTS_DIR"] / f"spiral_monitor_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spiral_result.get("time_series", {})).to_csv(out_path, index=False)
    print(f"\n✅ 螺旋时间序列已保存至: {out_path}")


def cmd_eval(args, ctx):
    """读取已有结果 CSV，重新打印评估表格。"""
    path = Path(args.results)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"\n📈 读取结果: {path}")
    _print_results_table(df)


# ── CLI 入口 ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snail-shell",
        description="🐌 A股日频收益率区间预测框架 snail-shell v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="snail-shell 2.0")

    subs = parser.add_subparsers(dest="command", metavar="<command>")
    subs.required = False

    # ── compare ──
    p_cmp = subs.add_parser("compare", help="运行完整对比实验（基线 + 蜗牛壳全变体）")
    p_cmp.add_argument(
        "--betas", type=float, nargs="+", default=None,
        metavar="β",
        help="蜗牛壳 β 值列表，默认 [0.5, 1.0, 2.0, 5.0]",
    )
    p_cmp.add_argument(
        "--output", type=str, default=None,
        help="结果 CSV 保存路径（默认自动生成带时间戳的文件名）",
    )

    # ── train ──
    p_tr = subs.add_parser("train", help="单独训练指定模型并评估")
    p_tr.add_argument(
        "--mode", choices=["baseline", "snail", "all"], default="all",
        help="训练模式：baseline | snail | all（默认 all）",
    )
    p_tr.add_argument(
        "--beta", type=float, default=None,
        help="蜗牛壳 β 值（--mode snail 时生效，不指定则跑全部变体）",
    )

    # ── eval ──
    p_ev = subs.add_parser("eval", help="读取已有结果 CSV 并展示评估表格")
    p_ev.add_argument("--results", required=True, help="结果 CSV 文件路径")

    # ── beta-select ──
    subs.add_parser("beta-select", help="在验证集 Q2~Q4 上遍历 β，输出最优选择")

    # ── monitor ──
    p_mon = subs.add_parser("monitor", help="训练模型并输出螺旋监控分析")
    p_mon.add_argument(
        "--beta", type=float, default=1.0,
        help="用于螺旋监控的蜗牛壳 β 值（默认 1.0）",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 无子命令时打印帮助
    if args.command is None:
        parser.print_help()
        sys.exit(0)


    # 加载核心模块
    print("⏳ 加载依赖模块...")
    try:
        ctx = _import_core()
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("   请确认已安装 requirements.txt 中的依赖：pip install -r requirements.txt")
        sys.exit(1)

    # 打印项目信息
    info = ctx["get_project_info"]()
    print(f"\n{'='*60}")
    print(f"  {info['name']} v{info['version']} — {info['description']}")
    print(f"  数据库: {info['database_path']}")
    print(f"{'='*60}")

    # 派发命令
    dispatch = {
        "compare":     cmd_compare,
        "train":       cmd_train,
        "eval":        cmd_eval,
        "beta-select": cmd_beta_select,
        "monitor":     cmd_monitor,
    }
    dispatch[args.command](args, ctx)


if __name__ == "__main__":
    main()
