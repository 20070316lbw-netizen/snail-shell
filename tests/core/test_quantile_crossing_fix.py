"""
Tests for core/quantile_crossing_fix.py and its integration with BaseQuantileHead.

Covers:
  fix_quantile_crossing  — 所有交叉场景及数学保证
  crossing_info          — 三类交叉的诊断计数与比率
  BaseQuantileHead 集成  — predict_anchor_and_radius() 自动修复后 radius ≥ 0
"""

import numpy as np
import pytest
from core.quantile_crossing_fix import fix_quantile_crossing, crossing_info
from core.base_quantile_head import BaseQuantileHead, FitConfig


# ---------------------------------------------------------------------------
# 用于集成测试的最小 Mock 子类
# ---------------------------------------------------------------------------

class _MockQuantileHead(BaseQuantileHead):
    """用固定预测值代替真实模型，方便控制交叉场景。"""

    def __init__(self, q10, q50, q90, point=None):
        self._q10   = np.asarray(q10, dtype=float)
        self._q50   = np.asarray(q50, dtype=float)
        self._q90   = np.asarray(q90, dtype=float)
        self._point = np.asarray(point if point is not None else q50, dtype=float)

    def fit(self, config: FitConfig) -> None:
        pass

    def predict(self, X):
        return {
            "point": self._point.copy(),
            "q10":   self._q10.copy(),
            "q50":   self._q50.copy(),
            "q90":   self._q90.copy(),
        }


# ===========================================================================
# fix_quantile_crossing
# ===========================================================================

class TestFixQuantileCrossing:

    # -----------------------------------------------------------------------
    # 无交叉时输出等于输入
    # -----------------------------------------------------------------------

    def test_no_crossing_q10_unchanged(self):
        q10 = np.array([1.0, 2.0, 3.0])
        q50 = np.array([4.0, 5.0, 6.0])
        q90 = np.array([7.0, 8.0, 9.0])
        fq10, fq50, fq90 = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq10, q10)
        np.testing.assert_array_equal(fq90, q90)

    def test_no_crossing_q50_preserved_exactly(self):
        q50 = np.array([4.0, 5.0, 6.0])
        fq10, fq50_out, fq90 = fix_quantile_crossing(
            q50 - 1.0, q50, q50 + 1.0
        )
        np.testing.assert_array_equal(fq50_out, q50)

    # -----------------------------------------------------------------------
    # 类型 A：q10 > q50
    # -----------------------------------------------------------------------

    def test_type_a_q10_clipped_to_q50(self):
        """q10 高于 q50 时，fixed_q10 应等于 q50。"""
        q10 = np.array([6.0])   # q10 > q50
        q50 = np.array([4.0])
        q90 = np.array([8.0])
        fq10, _, _ = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq10, q50)

    def test_type_a_q90_untouched(self):
        """q10 > q50 时，q90 本身没有问题，不应被改动。"""
        q10 = np.array([6.0])
        q50 = np.array([4.0])
        q90 = np.array([8.0])
        _, _, fq90 = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq90, q90)

    # -----------------------------------------------------------------------
    # 类型 B：q90 < q50
    # -----------------------------------------------------------------------

    def test_type_b_q90_clipped_to_q50(self):
        """q90 低于 q50 时，fixed_q90 应等于 q50。"""
        q10 = np.array([1.0])
        q50 = np.array([5.0])
        q90 = np.array([3.0])   # q90 < q50
        _, _, fq90 = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq90, q50)

    def test_type_b_q10_untouched(self):
        """q90 < q50 时，q10 本身没有问题，不应被改动。"""
        q10 = np.array([1.0])
        q50 = np.array([5.0])
        q90 = np.array([3.0])
        fq10, _, _ = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq10, q10)

    # -----------------------------------------------------------------------
    # 类型 C：q10 > q90（最严重，双侧都越过锚点）
    # -----------------------------------------------------------------------

    def test_type_c_both_sides_fixed(self):
        """q10 > q50 且 q90 < q50 时，两端都被截断到 q50。"""
        q10 = np.array([7.0])   # q10 > q50
        q50 = np.array([5.0])
        q90 = np.array([3.0])   # q90 < q50
        fq10, _, fq90 = fix_quantile_crossing(q10, q50, q90)
        # 修复后 q10 和 q90 都等于 q50 → 零宽度区间
        np.testing.assert_array_equal(fq10, q50)
        np.testing.assert_array_equal(fq90, q50)

    def test_type_c_q10_above_q90_directly(self):
        """即便 q10 > q90 但两者都在 q50 同侧，仍能正确修复。"""
        q10 = np.array([6.0])
        q50 = np.array([5.0])
        q90 = np.array([4.0])   # q90 < q50 < q10
        fq10, _, fq90 = fix_quantile_crossing(q10, q50, q90)
        assert fq10 <= fq90

    # -----------------------------------------------------------------------
    # 数学保证：任意输入下 fixed_q10 ≤ q50 ≤ fixed_q90
    # -----------------------------------------------------------------------

    def test_ordering_guarantee_random_input(self):
        """对 1000 个随机样本，修复后的分位数必须满足 q10 ≤ q50 ≤ q90。"""
        rng = np.random.default_rng(0)
        n = 1000
        q10 = rng.standard_normal(n)
        q50 = rng.standard_normal(n)
        q90 = rng.standard_normal(n)
        fq10, fq50, fq90 = fix_quantile_crossing(q10, q50, q90)
        assert np.all(fq10 <= fq50 + 1e-12)
        assert np.all(fq50 <= fq90 + 1e-12)

    def test_radius_nonnegative_random_input(self):
        """修复后 (q90 - q10) / 2 ≥ 0 严格成立。"""
        rng = np.random.default_rng(1)
        n = 1000
        q10 = rng.standard_normal(n)
        q50 = rng.standard_normal(n)
        q90 = rng.standard_normal(n)
        fq10, _, fq90 = fix_quantile_crossing(q10, q50, q90)
        radius = (fq90 - fq10) / 2
        assert np.all(radius >= -1e-12)

    # -----------------------------------------------------------------------
    # q50 始终不变
    # -----------------------------------------------------------------------

    def test_q50_never_modified(self):
        """无论交叉情况多严重，q50 输出必须与输入完全一致。"""
        rng = np.random.default_rng(2)
        n = 500
        q10 = rng.standard_normal(n)
        q50 = rng.standard_normal(n)
        q90 = rng.standard_normal(n)
        _, fq50, _ = fix_quantile_crossing(q10, q50, q90)
        np.testing.assert_array_equal(fq50, q50)

    # -----------------------------------------------------------------------
    # 边界情况
    # -----------------------------------------------------------------------

    def test_equal_quantiles_unchanged(self):
        """q10 == q50 == q90 时输出应与输入相同。"""
        v = np.array([3.0, 3.0, 3.0])
        fq10, fq50, fq90 = fix_quantile_crossing(v, v, v)
        np.testing.assert_array_equal(fq10, v)
        np.testing.assert_array_equal(fq50, v)
        np.testing.assert_array_equal(fq90, v)

    def test_single_element(self):
        q10 = np.array([2.0])
        q50 = np.array([1.0])   # crossing
        q90 = np.array([3.0])
        fq10, fq50, fq90 = fix_quantile_crossing(q10, q50, q90)
        assert fq10 <= fq50 <= fq90

    def test_output_shapes_preserved(self):
        n = 42
        q10 = np.ones(n) * -1
        q50 = np.zeros(n)
        q90 = np.ones(n)
        fq10, fq50, fq90 = fix_quantile_crossing(q10, q50, q90)
        assert fq10.shape == (n,)
        assert fq50.shape == (n,)
        assert fq90.shape == (n,)

    def test_returns_copy_not_view(self):
        """返回值应为独立副本，修改原数组不应影响输出。"""
        q50 = np.array([2.0, 3.0])
        q10 = q50 - 1.0
        q90 = q50 + 1.0
        _, fq50, _ = fix_quantile_crossing(q10, q50, q90)
        q50[0] = 999.0
        assert fq50[0] != 999.0


# ===========================================================================
# crossing_info
# ===========================================================================

class TestCrossingInfo:

    def test_no_crossing_all_zero(self):
        q10 = np.array([1.0, 2.0, 3.0])
        q50 = np.array([4.0, 5.0, 6.0])
        q90 = np.array([7.0, 8.0, 9.0])
        info = crossing_info(q10, q50, q90)
        assert info["n_q10_above_q50"]    == 0
        assert info["n_q90_below_q50"]    == 0
        assert info["n_q10_above_q90"]    == 0
        assert bool(info["any_crossing"]) is False

    def test_type_a_counted_correctly(self):
        # 2 samples: both have q10 > q50
        q10 = np.array([5.0, 5.0, 1.0])
        q50 = np.array([3.0, 3.0, 3.0])
        q90 = np.array([7.0, 7.0, 7.0])
        info = crossing_info(q10, q50, q90)
        assert info["n_q10_above_q50"] == 2
        assert np.isclose(info["rate_q10_above_q50"], 2 / 3)

    def test_type_b_counted_correctly(self):
        # 1 sample has q90 < q50
        q10 = np.array([1.0, 1.0])
        q50 = np.array([5.0, 5.0])
        q90 = np.array([3.0, 7.0])   # first: q90 < q50
        info = crossing_info(q10, q50, q90)
        assert info["n_q90_below_q50"] == 1
        assert np.isclose(info["rate_q90_below_q50"], 0.5)

    def test_type_c_counted_correctly(self):
        # 1 sample has q10 > q90
        q10 = np.array([8.0, 1.0])
        q50 = np.array([5.0, 5.0])
        q90 = np.array([2.0, 9.0])   # first: q10 > q90
        info = crossing_info(q10, q50, q90)
        assert info["n_q10_above_q90"] == 1
        assert np.isclose(info["rate_q10_above_q90"], 0.5)

    def test_any_crossing_true_when_any_type(self):
        q10 = np.array([6.0])
        q50 = np.array([4.0])
        q90 = np.array([8.0])   # only type A
        info = crossing_info(q10, q50, q90)
        assert bool(info["any_crossing"]) is True

    def test_rates_in_0_1(self):
        rng = np.random.default_rng(3)
        q10 = rng.standard_normal(100)
        q50 = rng.standard_normal(100)
        q90 = rng.standard_normal(100)
        info = crossing_info(q10, q50, q90)
        for key in ["rate_q10_above_q50", "rate_q90_below_q50", "rate_q10_above_q90"]:
            assert 0.0 <= info[key] <= 1.0

    def test_empty_arrays_return_zeros(self):
        info = crossing_info(np.array([]), np.array([]), np.array([]))
        assert info["n_q10_above_q50"]    == 0
        assert info["n_q90_below_q50"]    == 0
        assert info["n_q10_above_q90"]    == 0
        assert bool(info["any_crossing"]) is False

    def test_all_keys_present(self):
        info = crossing_info(np.array([1.0]), np.array([2.0]), np.array([3.0]))
        for key in [
            "n_q10_above_q50", "n_q90_below_q50", "n_q10_above_q90",
            "rate_q10_above_q50", "rate_q90_below_q50", "rate_q10_above_q90",
            "any_crossing",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_type_c_implies_type_a_or_b(self):
        """q10 > q90 のとき必ず q10 > q50 か q90 < q50 の少なくとも一方が成立。"""
        # q10=8 > q90=2: both q10>q50 and q90<q50 (with q50=5)
        info = crossing_info(np.array([8.0]), np.array([5.0]), np.array([2.0]))
        assert info["n_q10_above_q50"] + info["n_q90_below_q50"] > 0


# ===========================================================================
# BaseQuantileHead 集成测试
# ===========================================================================

class TestPredictAnchorAndRadiusIntegration:
    """
    verify that predict_anchor_and_radius() uses the crossing fix,
    guaranteeing radius ≥ 0 even when predict() outputs crossing quantiles.
    """

    def test_radius_nonnegative_with_no_crossing(self):
        head = _MockQuantileHead(q10=[1.0, 2.0], q50=[3.0, 4.0], q90=[5.0, 6.0])
        anchor, radius = head.predict_anchor_and_radius(None)
        assert np.all(radius >= 0.0)

    def test_radius_nonnegative_when_q10_above_q50(self):
        """q10 > q50 → radius would be negative without the fix."""
        head = _MockQuantileHead(
            q10=[6.0, 7.0],   # q10 > q50
            q50=[4.0, 4.0],
            q90=[8.0, 8.0],
        )
        anchor, radius = head.predict_anchor_and_radius(None)
        assert np.all(radius >= 0.0)

    def test_radius_nonnegative_when_q90_below_q50(self):
        """q90 < q50 → radius would be negative without the fix."""
        head = _MockQuantileHead(
            q10=[1.0, 1.0],
            q50=[5.0, 5.0],
            q90=[3.0, 3.0],   # q90 < q50
        )
        anchor, radius = head.predict_anchor_and_radius(None)
        assert np.all(radius >= 0.0)

    def test_radius_nonnegative_severe_crossing(self):
        """q10 > q90 の最悪ケース。"""
        head = _MockQuantileHead(
            q10=[9.0],
            q50=[5.0],
            q90=[2.0],
        )
        anchor, radius = head.predict_anchor_and_radius(None)
        assert radius[0] >= 0.0

    def test_anchor_is_always_q50(self):
        """anchor = q50 は交叉修復で変わらないこと。"""
        q50 = np.array([3.0, 7.0, -1.0])
        head = _MockQuantileHead(
            q10=q50 + 2.0,   # crossing
            q50=q50,
            q90=q50 - 1.0,   # crossing
        )
        anchor, _ = head.predict_anchor_and_radius(None)
        np.testing.assert_array_equal(anchor, q50)

    def test_predict_raw_output_unchanged(self):
        """predict() 本身不受影响，仍可返回含交叉的原始分位数。"""
        head = _MockQuantileHead(
            q10=[6.0],
            q50=[4.0],
            q90=[8.0],
        )
        raw = head.predict(None)
        # raw q10 should still be the original (crossing) value
        assert raw["q10"][0] == 6.0
        assert raw["q50"][0] == 4.0

    def test_radius_correct_value_no_crossing(self):
        """无交叉时 radius = (q90 - q10) / 2 的值不变。"""
        head = _MockQuantileHead(
            q10=[2.0, 4.0],
            q50=[5.0, 6.0],
            q90=[8.0, 10.0],
        )
        _, radius = head.predict_anchor_and_radius(None)
        expected = np.array([(8.0 - 2.0) / 2, (10.0 - 4.0) / 2])
        np.testing.assert_allclose(radius, expected)

    def test_radius_with_partial_crossing(self):
        """一部分有交叉、一部分没有，两段都正确处理。"""
        head = _MockQuantileHead(
            q10=[6.0, 1.0],   # first: crossing (q10>q50), second: normal
            q50=[4.0, 3.0],
            q90=[8.0, 7.0],
        )
        _, radius = head.predict_anchor_and_radius(None)
        # first sample: q10 clipped to q50=4 → radius=(8-4)/2=2
        # second sample: no crossing → radius=(7-1)/2=3
        np.testing.assert_allclose(radius, [2.0, 3.0])
