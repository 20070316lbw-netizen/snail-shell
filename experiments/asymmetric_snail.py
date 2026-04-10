"""
asymmetric_snail.py - AS-GSPQR 非对称软拉回实验

实现 AS-GSPQR Phase 1：非对称区间 + 固定 β

与 snail_lgbm.py 的区别：
  - 区间由非对称半径构造：[ŷ*_t - r_down, ŷ*_t + r_up]
  - 门控函数使用方向感知 SNR（r_dir 随 δ_t 方向切换）
  - 区间宽度 = r_down + r_up = q90 - q10（总宽与原始分位数区间相同）
  - 输出包含非对称专用诊断指标（skewness_index 等）

运行方式：
  python experiments/asymmetric_snail.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantile_head import QuantileHead, FitConfig
from core.asymmetric_mechanism import AsymmetricSnailMechanism, asymmetric_soft_pullback
from evaluation.metrics import (
    coverage_error,
    actual_coverage,
    winkler_score,
    interval_width,
    mean_absolute_error,
    rank_ic,
    composite_score,
    calculate_all_metrics_asymmetric,
)


# ---------------------------------------------------------------------------
# AsymmetricSnailModel：端到端模型类
# ---------------------------------------------------------------------------

class AsymmetricSnailModel:
    """
    AS-GSPQR 模型

    = QuantileHead（4 个 LightGBM）+ AsymmetricSnailMechanism

    接口与 SnailModel（snail_lgbm.py）保持一致，可直接替换对比。
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        self.beta = beta
        self.quantile_head = QuantileHead(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.mechanism = AsymmetricSnailMechanism()
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """训练四模型分位数回归头（与 SnailModel.fit 接口完全相同）"""
        config = FitConfig(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
        )
        self.quantile_head.fit(config)
        self.is_fitted = True

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测并应用非对称软拉回

        Returns 字典：
          corrected_center : ŷ*_t
          lower            : ŷ*_t - r_down
          upper            : ŷ*_t + r_up
          original_point   : ŷ_t（MSE 原始预测）
          anchor           : a_t = q̂_{50,t}
          r_down           : 下行半径
          r_up             : 上行半径
          raw_q10/q90      : 原始分位数（备查）
          diagnostics      : 诊断字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")

        preds = self.quantile_head.predict(X)
        point_pred = preds["point"]
        anchor     = preds["q50"]
        q10        = preds["q10"]
        q90        = preds["q90"]

        center, lower, upper, diag = asymmetric_soft_pullback(
            point_pred, anchor, q10, q90, self.beta
        )

        return {
            "corrected_center" : center,
            "lower"            : lower,
            "upper"            : upper,
            "original_point"   : point_pred,
            "anchor"           : anchor,
            "r_down"           : anchor - q10,       # 原始下行半径（锚到中位数）
            "r_up"             : q90 - anchor,        # 原始上行半径
            "raw_q10"          : q10,
            "raw_q90"          : q90,
            "diagnostics"      : diag,
        }

    # ------------------------------------------------------------------
    def predict_interval(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (修正中心, 下界, 上界)，与 SnailModel.predict_interval 接口一致"""
        result = self.predict(X)
        return result["corrected_center"], result["lower"], result["upper"]


# ---------------------------------------------------------------------------
# ExperimentConfig（复用 snail_lgbm.py 的结构）
# ---------------------------------------------------------------------------

@dataclass
class AsymmetricExperimentConfig:
    X_train     : np.ndarray
    y_train     : np.ndarray
    X_val       : np.ndarray        # 训练早停验证集
    y_val       : np.ndarray
    X_test      : np.ndarray
    y_test      : np.ndarray
    beta_values : List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 5.0])
    # 可选：β 选择用验证集（2022 Q2-Q4）
    X_val_q2_4  : Optional[np.ndarray] = None
    y_val_q2_4  : Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# run_asymmetric_experiment：主实验函数
# ---------------------------------------------------------------------------

def run_asymmetric_experiment(config: AsymmetricExperimentConfig) -> Dict:
    """
    运行 AS-GSPQR 实验（β 网格扫描）

    所有 β 共享同一组训练好的分位数模型（Stage 1 只训练一次），
    β 扫描在推断阶段完成，无需重复训练。

    返回:
        {
          "AS-Snail-{β}": {
            "corrected_center": ...,
            "lower": ..., "upper": ...,
            "diagnostics": ...,
            "val_q2_4": ...   # 若提供了 X_val_q2_4
          }
        }
    """
    print("=" * 60)
    print("AS-GSPQR Phase 1：非对称软拉回实验")
    print("=" * 60)

    # ── Stage 1：训练分位数回归头（所有 β 共享）──────────────────────
    print("\n[Stage 1] 训练分位数回归头（MSE / q10 / q50 / q90）...")
    qh = QuantileHead()
    cfg = FitConfig(
        X_train=config.X_train,
        y_train=config.y_train,
        X_val=config.X_val,
        y_val=config.y_val,
    )
    qh.fit(cfg)
    print("[Stage 1] 训练完成。\n")

    # ── Stage 2：β 扫描（测试集推断）────────────────────────────────
    mech = AsymmetricSnailMechanism(beta_values=config.beta_values)

    # 测试集预测
    test_preds = qh.predict(config.X_test)
    test_point  = test_preds["point"]
    test_anchor = test_preds["q50"]
    test_q10    = test_preds["q10"]
    test_q90    = test_preds["q90"]

    scan_test = mech.scan_beta(test_point, test_anchor, test_q10, test_q90)

    # 验证集（Q2-Q4）推断（若提供）
    scan_val = None
    if config.X_val_q2_4 is not None and config.y_val_q2_4 is not None:
        val_preds  = qh.predict(config.X_val_q2_4)
        scan_val   = mech.scan_beta(
            val_preds["point"], val_preds["q50"],
            val_preds["q10"],   val_preds["q90"],
        )

    # 组装结果
    results = {}
    for beta in config.beta_values:
        center, lower, upper, diag = scan_test[beta]
        label = f"AS-Snail-{beta}"

        entry = {
            "corrected_center" : center,
            "lower"            : lower,
            "upper"            : upper,
            "diagnostics"      : diag,
        }

        if scan_val is not None:
            v_center, v_lower, v_upper, v_diag = scan_val[beta]
            entry["val_q2_4"] = {
                "corrected_center" : v_center,
                "lower"            : v_lower,
                "upper"            : v_upper,
                "diagnostics"      : v_diag,
            }

        results[label] = entry
        print(f"  β={beta:5}  |  "
              f"宽度={diag['mean_width']:.4f}  "
              f"λ̄={diag['mean_lambda']:.3f}  "
              f"偏斜指数={diag['mean_skew_index']:+.3f}")

    print()
    return results, qh


# ---------------------------------------------------------------------------
# select_best_beta_asymmetric：在验证集选最优 β
# ---------------------------------------------------------------------------

def select_best_beta_asymmetric(
    qh: QuantileHead,
    X_val: np.ndarray,
    y_val: np.ndarray,
    beta_candidates: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """
    在验证集（Q2-Q4）上选择最优 β

    使用复合评分：Score = W̄ + 10 · max(0, CE - 0.05)

    参数:
        qh             : 已训练的 QuantileHead
        X_val / y_val  : 验证集特征和标签
        beta_candidates: β 候选列表

    返回:
        (best_beta, beta_scores_dict)
    """
    if beta_candidates is None:
        beta_candidates = [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")]

    val_preds  = qh.predict(X_val)
    point      = val_preds["point"]
    anchor     = val_preds["q50"]
    q10        = val_preds["q10"]
    q90        = val_preds["q90"]

    mech = AsymmetricSnailMechanism(beta_values=beta_candidates)
    best_beta, beta_scores = mech.select_beta(point, anchor, q10, q90, y_val)

    return best_beta, beta_scores


# ---------------------------------------------------------------------------
# compare_symmetric_vs_asymmetric：生成对比表
# ---------------------------------------------------------------------------

def compare_symmetric_vs_asymmetric(
    y_test: np.ndarray,
    sym_results: Dict,         # snail_lgbm.run_snail_experiment 的输出
    asym_results: Dict,        # run_asymmetric_experiment 的输出（第一项）
) -> pd.DataFrame:
    """
    生成对称 vs 非对称的完整对比 DataFrame

    参数:
        y_test      : 测试集真实标签
        sym_results : {"Snail-1.0": {"point_pred", "lower", "upper", ...}, ...}
        asym_results: {"AS-Snail-1.0": {"corrected_center", "lower", "upper", ...}, ...}

    返回:
        pd.DataFrame，含 AC / CE / Winkler / Width / MAE / RankIC / skewness_index
    """
    rows = []

    def _eval(name, center, lower, upper):
        row = {"Method": name}
        # 列名与 evaluate_methods 保持一致，确保 _print_table 能正确识别
        row["actual_coverage"] = actual_coverage(y_test, lower, upper)
        row["coverage_error"]  = coverage_error(y_test, lower, upper)
        row["winkler_score"]   = winkler_score(y_test, lower, upper)
        row["interval_width"]  = interval_width(lower, upper)
        row["mae"]             = mean_absolute_error(y_test, center)
        row["rank_ic"]         = rank_ic(center, y_test)
        row["composite_score"] = composite_score(row["winkler_score"], row["coverage_error"])
        # 偏斜指数（对称版本应接近 0，非对称版本反映市场偏斜方向）
        from evaluation.metrics import skewness_index as si
        row["skewness_index"]  = si(lower, upper, center)
        rows.append(row)

    # 对称结果
    for name, res in sym_results.items():
        _eval(name,
              res.get("corrected_point", res.get("point_pred")),
              res["lower"], res["upper"])

    # 非对称结果
    for name, res in asym_results.items():
        _eval(name, res["corrected_center"], res["lower"], res["upper"])

    df = pd.DataFrame(rows)
    # 排序：winkler_score 升序
    df = df.sort_values("winkler_score").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AS-GSPQR 自测（模拟数据）\n")
    np.random.seed(42)
    N, F = 3000, 10
    X = np.random.randn(N, F)
    y = X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(N) * 0.5

    split = [int(N * 0.6), int(N * 0.8)]
    X_train, X_val, X_test = X[:split[0]], X[split[0]:split[1]], X[split[1]:]
    y_train, y_val, y_test = y[:split[0]], y[split[0]:split[1]], y[split[1]:]

    cfg = AsymmetricExperimentConfig(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        beta_values=[0.0, 0.5, 1.0, 2.0, 5.0],
        X_val_q2_4=X_val, y_val_q2_4=y_val,
    )

    results, qh = run_asymmetric_experiment(cfg)

    print("─" * 70)
    print(f"{'Method':<20} {'AC':>6} {'CE':>7} {'Winkler':>9} "
          f"{'Width':>8} {'MAE':>8} {'SkewIdx':>9}")
    print("─" * 70)

    for name, res in results.items():
        center = res["corrected_center"]
        lower  = res["lower"]
        upper  = res["upper"]
        from evaluation.metrics import skewness_index as si
        print(f"{name:<20} "
              f"{actual_coverage(y_test,lower,upper):>6.4f} "
              f"{coverage_error(y_test,lower,upper):>7.4f} "
              f"{winkler_score(y_test,lower,upper):>9.4f} "
              f"{interval_width(lower,upper):>8.4f} "
              f"{mean_absolute_error(y_test,center):>8.4f} "
              f"{si(lower,upper,center):>+9.4f}")

    # β 选择
    print("\n在验证集选择最优 β：")
    best_beta, scores = select_best_beta_asymmetric(qh, X_val, y_val)
    print(f"  最优 β* = {best_beta}")
    for b, info in scores.items():
        m = info["metrics"]
        print(f"  β={b:5}: Score={m['score']:.4f}  "
              f"Winkler={m['winkler_mean']:.4f}  "
              f"CE={m['coverage_error']:.4f}")
