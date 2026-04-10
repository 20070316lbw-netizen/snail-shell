"""
main.py - snail-shell 统一实验入口 v2.1

用法：
  python main.py [全局参数] <命令> [命令参数]

全局参数：
  --verbose / -v        输出详细日志（包含 LightGBM 迭代信息）
  --quiet   / -q        仅输出关键结果表格，抑制过程日志
  --log-file <path>     将日志同步写入文件

命令：
  compare     运行完整对比实验（基线 + 蜗牛壳全变体）
  train       单独训练并评估指定模型
  eval        读取已有结果 CSV，重新展示评估表格
  beta-select 在验证集 Q2~Q4 遍历 β，输出最优选择
  monitor     训练模型并输出螺旋监控分析
  visualize   从已有 CSV 结果生成 Pareto / 对比柱状图
  check       数据质量 & 分位数交叉率快速健康检查
  predict     运行每日预测并生成 Markdown 报告

示例：
  python main.py compare                                  # 完整对比（全部基线 + 全部蜗牛壳）
  python main.py compare --baselines cp,qr --plot         # 仅 CP+QR vs 蜗牛壳，完成后出图
  python main.py compare --no-snail                       # 仅跑基线
  python main.py train --mode snail --beta 1.0            # 单独训练 Snail-1
  python main.py train --mode baseline --baselines cp,qr  # 仅跑 CP 和 QR 基线
  python main.py beta-select --betas 0.5 1 2 5            # 自定义候选集
  python main.py monitor --beta 1.0 --plot                # 螺旋监控 + 轨迹图
  python main.py visualize --from-csv results/xxx.csv     # 从 CSV 生成 Pareto 图
  python main.py visualize --from-csv results/xxx.csv --type all --save-plots outputs/
  python main.py check                                    # 数据健康检查（交叉率等）
  python main.py -v compare                               # 详细日志模式
  python main.py -q compare                               # 静默模式（仅结果表格）
  python main.py predict                                  # 运行每日预测
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 项目根目录 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志系统 ───────────────────────────────────────────────────────────────────
_LOG_FMT_SIMPLE  = "%(message)s"
_LOG_FMT_VERBOSE = "[%(levelname)s %(asctime)s] %(message)s"

log = logging.getLogger("snail-shell")


def _setup_logging(verbose: bool = False, quiet: bool = False, log_file: str = None):
    """配置全局日志。"""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    fmt   = _LOG_FMT_VERBOSE if verbose else _LOG_FMT_SIMPLE

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    # 抑制 LightGBM 自身日志（除非 verbose）
    if not verbose:
        logging.getLogger("lightgbm").setLevel(logging.ERROR)


# ── 延迟导入（避免 --help 时触发重依赖） ────────────────────────────────────────
def _import_core() -> dict:
    from config import (
        DATA_SPLIT, BETA_VALUES, OUTPUTS_DIR, RESULTS_DIR,
        EVALUATION_PARAMS, get_project_info,
    )
    from core.data_loader import DataLoader
    from core.quantile_head import QuantileHead, FitConfig
    from core.snail_mechanism import SnailMechanism
    from core.spiral_monitor import SpiralMonitor
    from experiments.baseline_lgbm import run_baseline_experiment
    from experiments.snail_lgbm import run_snail_experiment, select_best_beta, ExperimentConfig
    from experiments.asymmetric_snail import (
        run_asymmetric_experiment, select_best_beta_asymmetric,
        compare_symmetric_vs_asymmetric, AsymmetricExperimentConfig,
    )
    from evaluation.metrics import (
        evaluate_methods, calculate_all_metrics, paired_t_test, crossing_rate,
        skewness_index, asymmetric_width_stats,
    )
    from evaluation.visualize import (
        plot_pareto_curve, plot_time_dynamics,
        plot_method_comparison, plot_spiral_trajectory,
    )
    return {
        "DATA_SPLIT": DATA_SPLIT,
        "BETA_VALUES": BETA_VALUES,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "EVALUATION_PARAMS": EVALUATION_PARAMS,
        "get_project_info": get_project_info,
        "DataLoader": DataLoader,
        "QuantileHead": QuantileHead,
        "FitConfig": FitConfig,
        "SnailMechanism": SnailMechanism,
        "SpiralMonitor": SpiralMonitor,
        "run_baseline_experiment": run_baseline_experiment,
        "run_snail_experiment": run_snail_experiment,
        "ExperimentConfig": ExperimentConfig,
        "select_best_beta": select_best_beta,
        "run_asymmetric_experiment": run_asymmetric_experiment,
        "select_best_beta_asymmetric": select_best_beta_asymmetric,
        "compare_symmetric_vs_asymmetric": compare_symmetric_vs_asymmetric,
        "AsymmetricExperimentConfig": AsymmetricExperimentConfig,
        "evaluate_methods": evaluate_methods,
        "calculate_all_metrics": calculate_all_metrics,
        "paired_t_test": paired_t_test,
        "crossing_rate": crossing_rate,
        "skewness_index": skewness_index,
        "asymmetric_width_stats": asymmetric_width_stats,
        "plot_pareto_curve": plot_pareto_curve,
        "plot_time_dynamics": plot_time_dynamics,
        "plot_method_comparison": plot_method_comparison,
        "plot_spiral_trajectory": plot_spiral_trajectory,
    }


# ── 数据加载 ───────────────────────────────────────────────────────────────────
def _load_data(ctx) -> tuple:
    """从 DuckDB 加载数据并按 README 规格切分。"""
    split = ctx["DATA_SPLIT"]
    DataLoader = ctx["DataLoader"]

    log.info("📂 连接数据库，加载特征数据 ...")
    with DataLoader() as loader:
        start, end = loader.get_date_range()
        log.info(f"   数据库日期范围: {start} ~ {end}")

        feature_cols = [
            "mom_20d", "mom_60d", "mom_12m_minus_1m",
            "vol_60d_res", "sp_ratio", "turn_20d",
            "mom_20d_rank", "mom_60d_rank", "mom_12m_minus_1m_rank",
            "vol_60d_res_rank", "sp_ratio_rank", "turn_20d_rank",
        ]
        label_col = "label_next_month"
        df = loader.get_features()

        if df.empty:
            raise RuntimeError("features_cn 表为空，请先向数据库中写入数据。")

        # 按 ticker 做滚动 z-score（W=60，前60天 expanding window）
        def _zscore(group):
            roll = group[feature_cols].rolling(window=60, min_periods=1)
            group[feature_cols] = (group[feature_cols] - roll.mean()) / (roll.std() + 1e-8)
            return group

        df = df.groupby("ticker", group_keys=False).apply(_zscore)

    def _slice(df, s, e):
        # 统一转为字符串比较，兼容 date 对象与字符串两种格式
        mask = (df["date"].astype(str) >= str(s)) & (df["date"].astype(str) <= str(e))
        sub = df[mask].copy()
        return sub[feature_cols].values.astype(np.float32), sub[label_col].values.astype(np.float32)

    Xtr, ytr   = _slice(df, split["train_start"],    split["train_end"])
    Xvq1, yvq1 = _slice(df, split["val_q1_start"],   split["val_q1_end"])
    Xv,   yv   = _slice(df, split["val_q2_4_start"], split["val_q2_4_end"])
    Xte,  yte  = _slice(df, split["test_start"],      split["test_end"])

    log.info(f"   Train   : {Xtr.shape[0]:>6} 条   {split['train_start']} ~ {split['train_end']}")
    log.info(f"   Val Q1  : {Xvq1.shape[0]:>6} 条   {split['val_q1_start']} ~ {split['val_q1_end']}")
    log.info(f"   Val Q2-4: {Xv.shape[0]:>6} 条   {split['val_q2_4_start']} ~ {split['val_q2_4_end']}")
    log.info(f"   Test    : {Xte.shape[0]:>6} 条   {split['test_start']} ~ {split['test_end']}")

    return Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte


def _mock_data(n: int = 3000, n_feat: int = 12, seed: int = 42) -> tuple:
    """数据库不可用时，生成模拟数据用于功能验证。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat).astype(np.float32)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + rng.randn(n) * 0.5).astype(np.float32)
    t, v = int(n * 0.6), int(n * 0.1)
    return (X[:t], y[:t],
            X[t:t+v], y[t:t+v],
            X[t+v:t+2*v], y[t+v:t+2*v],
            X[t+2*v:], y[t+2*v:])


def _load_or_mock(ctx) -> tuple:
    """尝试从数据库加载，失败则自动回退到模拟数据并打印警告。"""
    try:
        return _load_data(ctx)
    except Exception as e:
        log.warning(f"⚠️  数据库加载失败: {e}")
        log.warning("   自动回退到模拟数据（仅用于功能验证，结果无实际意义）")
        print(f"⚠️  数据库加载失败: {e}\n   使用模拟数据继续运行...")
        return _mock_data()


# ── 公共工具 ───────────────────────────────────────────────────────────────────
def _print_table(df: pd.DataFrame, title: str = ""):
    """统一打印评估结果表格。skewness_index 列若存在则附加在末尾。"""
    cols_order = [
        "Method", "actual_coverage", "coverage_error", "winkler_score", "interval_width",
        "mae", "rank_ic", "composite_score", "skewness_index",
    ]
    cols_order = [c for c in cols_order if c in df.columns]
    df_show = df[cols_order].copy()
    sep = "─" * 120
    if title:
        print(f"\n{title}")
    print(f"\n{sep}")
    print(df_show.to_string(index=False, na_rep="N/A", float_format=lambda x: f"{x:.4f}"))
    print(sep)


def _print_ce_improvement(df_asym: pd.DataFrame):
    """
    打印 CE 改善专项报告：
    - 逐 β 对比对称 vs 非对称的 Coverage Error
    - 高亮 CE 绝对下降量和相对改善百分比
    - 若 CE 已低于 5% 门槛，标注额外 Composite Score 奖励
    """
    if "coverage_error" not in df_asym.columns or "Method" not in df_asym.columns:
        return

    sym_rows  = df_asym[~df_asym["Method"].str.startswith("AS-")].set_index("Method")
    asym_rows = df_asym[ df_asym["Method"].str.startswith("AS-")].set_index("Method")

    if sym_rows.empty or asym_rows.empty:
        return

    print("\n" + "═" * 60)
    print("📊  CE 改善专项报告（Coverage Error Improvement）")
    print("═" * 60)
    print(f"  {'对称方法':<16} {'AS方法':<18} {'对称CE':>8} {'AS-CE':>8} {'↓绝对':>7} {'↓相对%':>8}  {'达标?':>6}")
    print("  " + "─" * 76)

    CE_THRESHOLD = 0.05
    any_printed = False

    for asym_name, asym_row in asym_rows.iterrows():
        # 匹配：AS-Snail-1.0 → Snail-1.0
        beta_str = asym_name[len("AS-"):]          # e.g. "Snail-1.0"
        sym_name = beta_str                          # "Snail-1.0"

        if sym_name not in sym_rows.index:
            # 回退：取对称组里 CE 最接近的行做参照
            if sym_rows.empty:
                continue
            sym_name = sym_rows["coverage_error"].idxmin()

        ce_sym  = sym_rows.loc[sym_name, "coverage_error"]
        ce_asym = asym_row["coverage_error"]

        delta    = ce_sym - ce_asym                 # 正值 = CE 下降（改善）
        rel_pct  = (delta / ce_sym * 100) if ce_sym > 1e-9 else 0.0

        sym_ok   = "✅" if ce_sym  <= CE_THRESHOLD else "❌"
        asym_ok  = "✅" if ce_asym <= CE_THRESHOLD else "❌"
        arrow    = "↓" if delta > 0 else ("↑" if delta < 0 else "=")

        print(f"  {sym_name:<16} {asym_name:<18} "
              f"{ce_sym:>8.4f} {ce_asym:>8.4f} "
              f"{arrow}{abs(delta):>6.4f} {rel_pct:>+7.1f}%  "
              f"{sym_ok}→{asym_ok}")
        any_printed = True

    if any_printed:
        print("  " + "─" * 76)
        print(f"  门槛: CE ≤ {CE_THRESHOLD:.0%}  |  ✅=已达标  ❌=超出门槛")
        print(f"  注: Composite Score = W̄ + k×CE（k=2.0），CE 持续参与评分；")
        print(f"      CE 越小得分越低（越好），AS-GSPQR 的覆盖率改善可直接体现。")
    print("═" * 60)


def _winkler_array(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float = 0.2) -> np.ndarray:
    """按样本计算 Winkler Score 数组（用于配对 t 检验）。"""
    width = upper - lower
    pl = np.where(y < lower, (2 / alpha) * (lower - y), 0)
    pu = np.where(y > upper, (2 / alpha) * (y - upper), 0)
    return width + pl + pu


def _significance_test(all_preds: dict, yte: np.ndarray, ctx: dict,
                        method_a: str = "Snail-1.0", method_b: str = "CP"):
    """对两方法的 Winkler Score 做配对 t 检验，打印结论。"""
    if method_a not in all_preds or method_b not in all_preds:
        log.debug(f"   跳过显著性检验：{method_a} 或 {method_b} 不在结果中")
        return
    alpha = ctx["EVALUATION_PARAMS"].get("winkler_alpha", 0.2)
    ws_a  = _winkler_array(yte, all_preds[method_a]["lower"], all_preds[method_a]["upper"], alpha)
    ws_b  = _winkler_array(yte, all_preds[method_b]["lower"], all_preds[method_b]["upper"], alpha)
    t_stat, p_val = ctx["paired_t_test"](ws_a, ws_b)
    print(f"\n🔬 统计显著性检验（{method_a} vs {method_b} | Winkler Score, Test Set）：")
    print(f"   t-statistic : {t_stat:+.4f}")
    print(f"   p-value     : {p_val:.4f}")
    verdict = "差异显著 (p < 0.05)" if p_val < 0.05 else "差异不显著 (p ≥ 0.05)"
    print(f"   结论        : {verdict}")


def _save_csv(df: pd.DataFrame, path: Path, label: str = "结果"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ {label}已保存至: {path}")


# ── 子命令：compare ────────────────────────────────────────────────────────────
def cmd_compare(args, ctx):
    """完整对比实验（基线 + 蜗牛壳全变体）。"""
    print("\n🐌 snail-shell — 完整对比实验")
    print("=" * 60)

    Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = _load_or_mock(ctx)
    all_preds = {}

    # ── 基线 ──────────────────────────────────────────────────────
    baselines = [b.strip() for b in args.baselines.split(",")] if args.baselines else None
    bl_label  = f"({args.baselines})" if args.baselines else "(全部)"
    print(f"\n📊 运行基线实验 {bl_label} ...")
    baseline_results = ctx["run_baseline_experiment"](
        Xtr, ytr, Xvq1, yvq1, Xte, yte,
        baseline_methods=baselines,
        X_val_q2_4=Xv, y_val_q2_4=yv,
    )
    # 重命名基线以对齐 Snail 理论框架
    if "QR" in baseline_results:
        baseline_results["Snail-0 (MSE-base)"] = baseline_results.pop("QR")
    if "Q50-only" in baseline_results:
        baseline_results["Snail-inf (Q50-base)"] = baseline_results.pop("Q50-only")
    
    all_preds.update(baseline_results)

    # ── 蜗牛壳变体 ────────────────────────────────────────────────
    if not args.no_snail:
        betas = args.betas if args.betas else [0.5, 1.0, 2.0, 5.0]
        print(f"\n🐌 运行对称蜗牛壳变体 β={betas} ...")
        config = ctx["ExperimentConfig"](
            X_train=Xtr, y_train=ytr,
            X_val=Xvq1,  y_val=yvq1,
            X_test=Xte,  y_test=yte,
            beta_values=betas,
            X_val_q2_4=Xv, y_val_q2_4=yv,
        )
        all_preds.update(ctx["run_snail_experiment"](config))

    # ── AS-GSPQR：非对称变体 ──────────────────────────────────────
    if getattr(args, "asym", False) and not args.no_snail:
        betas = args.betas if args.betas else [0.5, 1.0, 2.0, 5.0]
        print(f"\n📐 运行非对称蜗牛壳变体（AS-GSPQR）β={betas} ...")
        asym_config = ctx["AsymmetricExperimentConfig"](
            X_train=Xtr, y_train=ytr,
            X_val=Xvq1,  y_val=yvq1,
            X_test=Xte,  y_test=yte,
            beta_values=betas,
            X_val_q2_4=Xv, y_val_q2_4=yv,
        )
        asym_results, asym_qh = ctx["run_asymmetric_experiment"](asym_config)

        # 统一格式：corrected_center → point_pred（与 evaluate_methods 接口兼容）
        for name, res in asym_results.items():
            all_preds[name] = {
                "point_pred": res["corrected_center"],
                "lower":      res["lower"],
                "upper":      res["upper"],
            }
            if "val_q2_4" in res:
                all_preds[name]["val_q2_4"] = {
                    "point_pred": res["val_q2_4"]["corrected_center"],
                    "lower":      res["val_q2_4"]["lower"],
                    "upper":      res["val_q2_4"]["upper"],
                }

        # 非对称专用表（含偏斜指数）
        if not getattr(args, "quiet", False):
            sym_results_for_compare = {
                k: {"corrected_point": v.get("point_pred", v.get("corrected_point")),
                    "lower": v["lower"], "upper": v["upper"]}
                for k, v in all_preds.items() if not k.startswith("AS-")
            }
            df_asym = ctx["compare_symmetric_vs_asymmetric"](
                yte, sym_results_for_compare, asym_results
            )
            _print_table(df_asym, "📐 对称 vs 非对称对比（含偏斜指数）：")

            # ── CE 改善专项报告 ──────────────────────────────────
            _print_ce_improvement(df_asym)

    # ── 评估：Test Set ───────────────────────────────────────────
    df_eval = ctx["evaluate_methods"](yte, all_preds)
    _print_table(df_eval, "📈 测试集评估指标汇总：")

    # ── 评估：Val Q2-Q4（Regime Shift 密集区）───────────────────
    val_preds = {m: p["val_q2_4"] for m, p in all_preds.items() if "val_q2_4" in p}
    if val_preds:
        df_val = ctx["evaluate_methods"](yv, val_preds)
        _print_table(df_val, "📈 验证集 Q2-Q4（Regime Shift 密集区）：")

    # ── 显著性检验 ────────────────────────────────────────────────
    if not args.quiet:
        _significance_test(all_preds, yte, ctx, "Snail-1.0", "CP")

    # ── 可视化 ────────────────────────────────────────────────────
    if args.plot:
        print("\n🖼️  生成可视化图表 ...")
        ctx["plot_pareto_curve"](all_preds, yte)
        ctx["plot_time_dynamics"](yte, all_preds)

    # ── 保存 ──────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else (
        ctx["RESULTS_DIR"] / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    _save_csv(df_eval, out_path, "测试集评估结果")


# ── 子命令：train ──────────────────────────────────────────────────────────────
def cmd_train(args, ctx):
    """单独训练基线或蜗牛壳模型并评估。"""
    print(f"\n🐌 snail-shell — 训练模式 [{args.mode}]")
    print("=" * 60)

    Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = _load_or_mock(ctx)
    results = {}

    if args.mode in ("baseline", "all"):
        baselines = [b.strip() for b in args.baselines.split(",")] if args.baselines else None
        print(f"\n📊 运行基线实验 ...")
        results.update(ctx["run_baseline_experiment"](
            Xtr, ytr, Xvq1, yvq1, Xte, yte,
            baseline_methods=baselines,
            X_val_q2_4=Xv, y_val_q2_4=yv,
        ))

    if args.mode in ("snail", "all"):
        betas = [args.beta] if args.beta else [0.5, 1.0, 2.0, 5.0]
        print(f"\n🐌 训练对称蜗牛壳模型 β={betas} ...")
        config = ctx["ExperimentConfig"](
            X_train=Xtr, y_train=ytr,
            X_val=Xvq1,  y_val=yvq1,
            X_test=Xte,  y_test=yte,
            beta_values=betas,
            X_val_q2_4=Xv, y_val_q2_4=yv,
        )
        results.update(ctx["run_snail_experiment"](config))

    if args.mode in ("asym", "all"):
        betas = [args.beta] if args.beta else [0.5, 1.0, 2.0, 5.0]
        print(f"\n📐 训练非对称蜗牛壳模型（AS-GSPQR）β={betas} ...")
        asym_config = ctx["AsymmetricExperimentConfig"](
            X_train=Xtr, y_train=ytr,
            X_val=Xvq1,  y_val=yvq1,
            X_test=Xte,  y_test=yte,
            beta_values=betas,
            X_val_q2_4=Xv, y_val_q2_4=yv,
        )
        asym_results, _ = ctx["run_asymmetric_experiment"](asym_config)
        for name, res in asym_results.items():
            results[name] = {
                "point_pred": res["corrected_center"],
                "lower":      res["lower"],
                "upper":      res["upper"],
            }

    df_eval = ctx["evaluate_methods"](yte, results)
    _print_table(df_eval, "📈 测试集评估：")

    out_path = ctx["RESULTS_DIR"] / f"train_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _save_csv(df_eval, out_path, "训练评估结果")


# ── 子命令：eval ───────────────────────────────────────────────────────────────
def cmd_eval(args, ctx):
    """读取已有结果 CSV，重新打印评估表格。"""
    path = Path(args.results)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"\n📈 读取结果: {path}  ({len(df)} 条方法)")
    _print_table(df)


# ── 子命令：beta-select ────────────────────────────────────────────────────────
def cmd_beta_select(args, ctx):
    """在验证集 Q2~Q4 上选择最优 β 值。"""
    print("\n🐌 snail-shell — β 选择")
    print("=" * 60)

    Xtr, ytr, _, _, Xv, yv, _, _ = _load_or_mock(ctx)
    beta_candidates = args.betas if args.betas else ctx["BETA_VALUES"]
    print(f"   候选 β 值: {beta_candidates}")

    best_beta, beta_scores = ctx["select_best_beta"](Xtr, ytr, Xv, yv, beta_candidates)

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {'β':>8}  {'复合Score':>12}  {'MAE':>10}  {'Winkler':>10}  {'Coverage':>10}")
    print(sep)
    for beta, info in beta_scores.items():
        m = info["metrics"]
        marker = "  ◀ 最优" if beta == best_beta else ""
        print(f"  {str(beta):>8}  {m['score']:>12.4f}  {m['mae']:>10.4f}"
              f"  {m['winkler_mean']:>10.4f}  {m['coverage']:>10.4f}{marker}")
    print(sep)
    print(f"\n✅ 建议使用 β = {best_beta}")


# ── 子命令：monitor ────────────────────────────────────────────────────────────
def cmd_monitor(args, ctx):
    """训练模型并输出螺旋监控报告。"""
    print("\n🐌 snail-shell — 螺旋监控")
    print("=" * 60)

    Xtr, ytr, _, _, Xv, yv, Xte, yte = _load_or_mock(ctx)
    beta = args.beta

    print(f"\n   训练 QuantileHead（β={beta}）...")
    qh = ctx["QuantileHead"]()
    cfg = ctx["FitConfig"](X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)
    qh.fit(cfg)

    anchor, radius = qh.predict_anchor_and_radius(Xte)
    monitor = ctx["SpiralMonitor"]()
    res = monitor.analyze(anchor, radius)

    r2         = res.get("r_squared", float("nan"))
    mean_v     = res.get("mean_velocity", float("nan"))
    alerts     = res.get("alerts", np.array([]))
    alert_count= int(np.sum(alerts))
    total_count= len(alerts)
    alert_rate = res.get("alert_rate", 0.0)

    print("\n📐 螺旋监控结果：")
    r2_warn = "  ⚠️  < 0.5，形态不明显，结论可信度有限" if r2 < 0.5 else ""
    print(f"  R²（对数螺线拟合）   : {r2:.4f}{r2_warn}")
    print(f"  螺线参数 (log_A, B)  : ({res.get('log_A', float('nan')):.4f}, "
          f"{res.get('B', float('nan')):.4f})")
    print(f"  平均外扩速度 v_t     : {mean_v:.6f}")
    print(f"  速度标准差           : {res.get('std_velocity', float('nan')):.6f}")
    print(f"  预警触发次数         : {alert_count} / {total_count}  (预警率 {alert_rate:.2%})")

    if args.plot:
        print("\n🖼️  生成螺旋轨迹图 ...")
        ctx["plot_spiral_trajectory"](anchor, radius,
                                      alerts=alerts if alert_count > 0 else None)

    # 保存时间序列
    velocity = res.get("velocity", np.array([]))
    ts = {
        "anchor": anchor,
        "radius": radius,
        "rho"   : res.get("rho", np.full_like(anchor, np.nan)),
        "theta" : res.get("theta", np.full_like(anchor, np.nan)),
        "velocity": np.concatenate([[np.nan], velocity]) if len(velocity) > 0 else np.full_like(anchor, np.nan),
        "alert" : alerts if len(alerts) == len(anchor) else np.zeros_like(anchor),
    }
    out_path = ctx["OUTPUTS_DIR"] / f"spiral_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _save_csv(pd.DataFrame(ts), out_path, "螺旋时间序列")


# ── 子命令：visualize（新增）─────────────────────────────────────────────────
def cmd_visualize(args, ctx):
    """从已有 CSV 结果生成可视化图表。

    支持的图表：
      pareto    Coverage Error vs Interval Width 散点图
      compare   各方法各指标的对比柱状图
      all       以上全部
    """
    print("\n🐌 snail-shell — 可视化")
    print("=" * 60)

    path = Path(args.from_csv)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"   读取结果: {path}  ({len(df)} 条方法，{list(df.columns)})")

    import matplotlib.pyplot as plt

    save_dir = Path(args.save_plots) if args.save_plots else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    vis_type = args.type

    # ── Pareto 图 ────────────────────────────────────────────────
    if vis_type in ("pareto", "all"):
        print("\n   绘制 Pareto 曲线 ...")

        # 颜色映射：Snail-* 用暖色，基线用冷色
        palette = {"Residual": "#5B9BD5", "CP": "#70AD47", "QR": "#FFC000",
                   "Q50-only": "#9E480E"}
        snail_colors = ["#FF6B6B", "#FF8E53", "#FFA500", "#C0392B"]

        fig, ax = plt.subplots(figsize=(10, 7))
        snail_idx = 0
        for _, row in df.iterrows():
            method = row.get("Method", "?")
            iw = row.get("interval_width", np.nan)
            ce = row.get("coverage_error", np.nan)
            color = palette.get(method, snail_colors[snail_idx % len(snail_colors)])
            if method not in palette:
                snail_idx += 1
            ax.scatter(iw, ce, s=160, color=color, zorder=5, alpha=0.85,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(method, (iw, ce), xytext=(6, 5),
                        textcoords="offset points", fontsize=9, color=color)

        ax.set_xlabel("平均区间宽度 (Interval Width)，越窄越好 →", fontsize=12)
        ax.set_ylabel("Coverage Error，越小越好 ↓", fontsize=12)
        ax.set_title("Pareto 曲线：区间质量对比（目标：左下角）",
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.annotate("◀ 理想区域", xy=(ax.get_xlim()[0], ax.get_ylim()[0]),
                    xytext=(0.05, 0.07), textcoords="axes fraction",
                    fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray"))
        plt.tight_layout()

        if save_dir:
            sp = save_dir / f"pareto_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.savefig(sp, dpi=150, bbox_inches="tight")
            print(f"   📊 Pareto 图已保存: {sp}")
        plt.show()

    # ── 对比柱状图 ───────────────────────────────────────────────
    if vis_type in ("compare", "all"):
        print("\n   绘制方法对比柱状图 ...")
        ctx["plot_method_comparison"](df)
        if save_dir:
            print("   💡 提示：柱状图暂不支持自动保存，请在弹出窗口手动另存。")

    if vis_type not in ("pareto", "compare", "all"):
        print(f"❌ 未知图表类型: {vis_type}，支持 pareto | compare | all")
        sys.exit(1)


# ── 子命令：check（新增）─────────────────────────────────────────────────────
def cmd_check(args, ctx):
    """数据质量与模型健康检查：数据分布统计 + 分位数交叉率 + 区间质量预检。"""
    print("\n🐌 snail-shell — 数据 & 模型健康检查")
    print("=" * 60)

    Xtr, ytr, Xvq1, yvq1, Xv, yv, Xte, yte = _load_or_mock(ctx)

    # ── 数据分割统计 ──────────────────────────────────────────────
    print("\n📊 数据分割统计：")
    splits = [
        ("Train",    Xtr,  ytr),
        ("Val Q1",   Xvq1, yvq1),
        ("Val Q2-4", Xv,   yv),
        ("Test",     Xte,  yte),
    ]
    print(f"  {'分割':>10}  {'样本数':>8}  {'特征数':>8}  "
          f"{'y均值':>10}  {'y标准差':>10}  {'y最小':>10}  {'y最大':>10}")
    print("  " + "─" * 72)
    for name, X, y in splits:
        has_nan = np.any(np.isnan(X)) or np.any(np.isnan(y))
        nan_mark = "  ⚠️NaN" if has_nan else ""
        print(f"  {name:>10}  {len(y):>8}  {X.shape[1]:>8}  "
              f"{np.nanmean(y):>10.4f}  {np.nanstd(y):>10.4f}  "
              f"{np.nanmin(y):>10.4f}  {np.nanmax(y):>10.4f}{nan_mark}")

    # ── 分位数交叉率检查 ─────────────────────────────────────────
    print("\n⏳ 训练轻量级分位数模型检查交叉率（n_estimators=100）...")
    n_est = 100  # 快速训练，仅用于健康诊断
    qh = ctx["QuantileHead"](n_estimators=n_est)
    cfg = ctx["FitConfig"](X_train=Xtr, y_train=ytr, X_val=Xvq1, y_val=yvq1)
    qh.fit(cfg)

    cr = ctx["crossing_rate"]
    threshold = ctx["EVALUATION_PARAMS"].get("crossing_threshold", 0.1)

    print(f"\n📐 分位数交叉率（阈值：{threshold:.0%}）：")
    print(f"  {'分割':>10}  {'交叉率':>10}  {'状态':>20}")
    print("  " + "─" * 44)
    for name, X, y in splits:
        preds = qh.predict(X)
        rate  = cr(preds["q10"], preds["q90"])
        ok    = rate <= threshold
        status = "✅ 正常" if ok else f"⚠️  超阈值 (>{threshold:.0%})"
        print(f"  {name:>10}  {rate:>10.2%}  {status}")

    # ── Coverage 预检（不依赖真实区间，仅用残差估计） ──────────────
    print("\n📐 Coverage 预检（q10/q90 对 Test Set）：")
    preds_te = qh.predict(Xte)
    in_interval = (yte >= preds_te["q10"]) & (yte <= preds_te["q90"])
    coverage = np.mean(in_interval)
    target   = ctx["EVALUATION_PARAMS"].get("target_coverage", 0.8)
    ce       = abs(coverage - target)
    print(f"  实际覆盖率   : {coverage:.2%}  （目标 {target:.0%}）")
    print(f"  Coverage Error: {ce:.4f}  {'✅ < 5%' if ce < 0.05 else '⚠️  超过 5%'}")

    print("\n✅ 健康检查完成")


# ── 子命令：predict ──────────────────────────────────────────────────────────
def cmd_predict(args, ctx):
    """运行每日预测并生成 Markdown 报告。"""
    print("\n🐌 snail-shell — 每日预测")
    print("=" * 60)
    try:
        from scripts.daily_predict import predict
        predict()
    except ImportError as e:
        print(f"❌ 导入 daily_predict 失败: {e}")
        sys.exit(1)


# ── CLI 解析 ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snail-shell",
        description="🐌 A股日频收益率区间预测框架 snail-shell v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="snail-shell 2.1")

    # ── 全局日志控制 ──
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--verbose", "-v", action="store_true",
                   help="输出详细调试日志")
    g.add_argument("--quiet", "-q", action="store_true",
                   help="仅输出关键结果，抑制过程日志")
    parser.add_argument("--log-file", metavar="PATH",
                        help="将日志同步写入文件（与屏幕输出并行）")

    subs = parser.add_subparsers(dest="command", metavar="<command>")
    subs.required = False

    # ── compare ──────────────────────────────────────────────────
    p_cmp = subs.add_parser("compare", help="运行完整对比实验（基线 + 蜗牛壳全变体）")
    p_cmp.add_argument("--betas", type=float, nargs="+", default=None, metavar="β",
                       help="蜗牛壳 β 值列表，默认 [0.5, 1.0, 2.0, 5.0]")
    p_cmp.add_argument("--baselines", type=str, default=None, metavar="LIST",
                       help="逗号分隔基线列表，如 cp,qr。默认全部 (residual,cp,qr,q50_only)")
    p_cmp.add_argument("--no-snail", action="store_true",
                       help="跳过蜗牛壳变体，仅运行基线对比")
    p_cmp.add_argument("--asym", action="store_true",
                       help="同时运行 AS-GSPQR 非对称变体并输出对称 vs 非对称对比表")
    p_cmp.add_argument("--plot", action="store_true",
                       help="实验完成后自动显示 Pareto + 时间动态图")
    p_cmp.add_argument("--output", type=str, default=None,
                       help="结果 CSV 保存路径（默认带时间戳自动生成）")

    # ── train ─────────────────────────────────────────────────────
    p_tr = subs.add_parser("train", help="单独训练指定模型并评估")
    p_tr.add_argument("--mode", choices=["baseline", "snail", "asym", "all"], default="all",
                      help="训练模式：baseline | snail | asym | all（默认 all）")
    p_tr.add_argument("--beta", type=float, default=None,
                      help="单个 β 值（--mode snail 时生效；不指定则跑全部变体）")
    p_tr.add_argument("--baselines", type=str, default=None, metavar="LIST",
                      help="基线选择，逗号分隔，如 cp,qr")

    # ── eval ──────────────────────────────────────────────────────
    p_ev = subs.add_parser("eval", help="读取已有结果 CSV 并展示评估表格")
    p_ev.add_argument("--results", required=True, metavar="PATH",
                      help="结果 CSV 文件路径")

    # ── beta-select ───────────────────────────────────────────────
    p_bs = subs.add_parser("beta-select", help="在验证集 Q2~Q4 上遍历 β，输出最优选择")
    p_bs.add_argument("--betas", type=float, nargs="+", default=None, metavar="β",
                      help="候选 β 列表（默认使用 config.BETA_VALUES）")

    # ── monitor ───────────────────────────────────────────────────
    p_mon = subs.add_parser("monitor", help="训练模型并输出螺旋监控分析")
    p_mon.add_argument("--beta", type=float, default=1.0,
                       help="螺旋监控时使用的蜗牛壳 β 值（默认 1.0）")
    p_mon.add_argument("--plot", action="store_true",
                       help="生成螺旋轨迹可视化图（笛卡尔 + 极坐标双子图）")

    # ── visualize（新增）─────────────────────────────────────────
    p_vis = subs.add_parser("visualize", help="从已有 CSV 结果生成可视化图表")
    p_vis.add_argument("--from-csv", required=True, metavar="PATH",
                       help="结果 CSV 文件路径（来自 compare 或 train 命令的输出）")
    p_vis.add_argument("--type", choices=["pareto", "compare", "all"], default="pareto",
                       help="图表类型：pareto | compare | all（默认 pareto）")
    p_vis.add_argument("--save-plots", metavar="DIR", default=None,
                       help="将图表保存到指定目录（不指定则仅弹窗展示）")

    # ── check（新增）─────────────────────────────────────────────
    subs.add_parser("check", help="数据质量 & 分位数交叉率快速健康检查")

    # ── predict ──────────────────────────────────────────────────
    subs.add_parser("predict", help="执行每日预测生成最新日期的推荐报告")

    return parser


# ── 主入口 ─────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # 初始化日志（全局 --verbose/--quiet/--log-file）
    _setup_logging(
        verbose  = getattr(args, "verbose", False),
        quiet    = getattr(args, "quiet",   False),
        log_file = getattr(args, "log_file", None),
    )

    # 加载核心模块
    if not getattr(args, "quiet", False):
        print("⏳ 加载依赖模块...")
    try:
        ctx = _import_core()
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("   请确认已安装 requirements.txt 中的依赖：pip install -r requirements.txt")
        sys.exit(1)

    # 项目 banner
    info = ctx["get_project_info"]()
    print(f"\n{'='*60}")
    print(f"  {info['name']} v{info['version']} — {info['description']}")
    print(f"  数据库: {info['database_path']}")
    print(f"{'='*60}")

    dispatch = {
        "compare":     cmd_compare,
        "train":       cmd_train,
        "eval":        cmd_eval,
        "beta-select": cmd_beta_select,
        "monitor":     cmd_monitor,
        "visualize":   cmd_visualize,
        "check":       cmd_check,
        "predict":     cmd_predict,
    }
    dispatch[args.command](args, ctx)


if __name__ == "__main__":
    main()
