"""
Tests for core/asymmetric_mechanism.py

Covers:
  compute_asymmetric_radii
  directional_gate
  asymmetric_soft_pullback
  AsymmetricSnailMechanism.apply
  AsymmetricSnailMechanism.scan_beta
  AsymmetricSnailMechanism.select_beta
  AsymmetricSnailMechanism.check_crossing
"""

import numpy as np
import pytest
from core.asymmetric_mechanism import (
    compute_asymmetric_radii,
    directional_gate,
    asymmetric_soft_pullback,
    AsymmetricSnailMechanism,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_asymmetric_data(n=100, seed=0):
    """Return (point_pred, anchor/q50, q10, q90, y_true) with q10 < q50 < q90."""
    rng = np.random.default_rng(seed)
    q50 = rng.standard_normal(n) * 0.03
    q10 = q50 - np.abs(rng.standard_normal(n)) * 0.05 - 0.02   # always below q50
    q90 = q50 + np.abs(rng.standard_normal(n)) * 0.06 + 0.02   # always above q50
    point_pred = q50 + rng.standard_normal(n) * 0.04
    y_true = rng.standard_normal(n) * 0.05
    return point_pred, q50, q10, q90, y_true


# ---------------------------------------------------------------------------
# compute_asymmetric_radii
# ---------------------------------------------------------------------------

class TestComputeAsymmetricRadii:
    def test_basic_calculation(self):
        anchor = np.array([5.0])
        q10    = np.array([3.0])
        q90    = np.array([8.0])
        r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)
        assert np.isclose(r_down, 2.0)  # 5 - 3
        assert np.isclose(r_up,   3.0)  # 8 - 5

    def test_radii_nonnegative(self):
        """Negative half-widths (quantile crossing) are clamped to 0."""
        anchor = np.array([5.0, 5.0])
        q10    = np.array([6.0, 4.0])   # first one crosses: q10 > anchor
        q90    = np.array([7.0, 4.5])   # second one crosses: q90 < anchor
        r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)
        assert np.all(r_down >= 0.0)
        assert np.all(r_up   >= 0.0)

    def test_output_shapes(self):
        n = 50
        rng = np.random.default_rng(1)
        anchor = rng.standard_normal(n)
        q10    = anchor - 0.5
        q90    = anchor + 0.5
        r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)
        assert r_down.shape == (n,)
        assert r_up.shape   == (n,)

    def test_symmetric_quantiles_give_equal_radii(self):
        anchor = np.array([0.0, 0.0])
        q10    = anchor - 2.0
        q90    = anchor + 2.0
        r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)
        np.testing.assert_allclose(r_down, r_up)

    def test_total_width_equals_q90_minus_q10(self):
        anchor = np.array([3.0, 7.0])
        q10    = np.array([1.0, 4.0])
        q90    = np.array([6.0, 9.0])
        r_down, r_up = compute_asymmetric_radii(anchor, q10, q90)
        np.testing.assert_allclose(r_down + r_up, q90 - q10)


# ---------------------------------------------------------------------------
# directional_gate
# ---------------------------------------------------------------------------

class TestDirectionalGate:
    def test_beta_zero_gives_lambda_one(self):
        """β=0 → no pullback → λ=1 regardless of deviation."""
        delta = np.array([1.0, -1.0, 0.5])
        r_down = np.array([1.0, 1.0, 1.0])
        r_up   = np.array([1.0, 1.0, 1.0])
        lambda_t, _ = directional_gate(delta, r_down, r_up, beta=0.0)
        np.testing.assert_allclose(lambda_t, np.ones(3))

    def test_beta_inf_zero_delta_gives_lambda_one(self):
        """β=∞ but δ=0 → SNR=0 → λ=1 (no pullback needed)."""
        delta  = np.array([0.0, 0.0])
        r_down = np.array([1.0, 1.0])
        r_up   = np.array([1.0, 1.0])
        lambda_t, _ = directional_gate(delta, r_down, r_up, beta=np.inf)
        np.testing.assert_allclose(lambda_t, np.ones(2))

    def test_beta_inf_nonzero_delta_gives_lambda_zero(self):
        """β=∞ and δ≠0 → λ=0 (full pullback)."""
        delta  = np.array([1.0, -1.0])
        r_down = np.array([1.0, 1.0])
        r_up   = np.array([1.0, 1.0])
        lambda_t, _ = directional_gate(delta, r_down, r_up, beta=np.inf)
        np.testing.assert_allclose(lambda_t, np.zeros(2))

    def test_positive_delta_uses_r_up(self):
        """When δ>0 the gate should be computed using r_up."""
        delta  = np.array([2.0])
        r_down = np.array([100.0])  # very large → would give λ≈1 if used
        r_up   = np.array([1.0])    # small  → should give λ = exp(-2*2/1)
        lambda_t, r_dir = directional_gate(delta, r_down, r_up, beta=2.0)
        expected = np.exp(-2.0 * 2.0 / 1.0)
        np.testing.assert_allclose(lambda_t, [expected], rtol=1e-6)
        np.testing.assert_allclose(r_dir, r_up)

    def test_negative_delta_uses_r_down(self):
        """When δ≤0 the gate should be computed using r_down."""
        delta  = np.array([-2.0])
        r_down = np.array([1.0])
        r_up   = np.array([100.0])
        lambda_t, r_dir = directional_gate(delta, r_down, r_up, beta=2.0)
        expected = np.exp(-2.0 * 2.0 / 1.0)
        np.testing.assert_allclose(lambda_t, [expected], rtol=1e-6)
        np.testing.assert_allclose(r_dir, r_down)

    def test_lambda_in_0_1(self):
        rng = np.random.default_rng(9)
        delta  = rng.standard_normal(100)
        r_down = np.abs(rng.standard_normal(100)) + 0.1
        r_up   = np.abs(rng.standard_normal(100)) + 0.1
        lambda_t, _ = directional_gate(delta, r_down, r_up, beta=1.0)
        assert np.all(lambda_t >= 0.0) and np.all(lambda_t <= 1.0)


# ---------------------------------------------------------------------------
# asymmetric_soft_pullback
# ---------------------------------------------------------------------------

class TestAsymmetricSoftPullback:
    def setup_method(self):
        self.point_pred, self.anchor, self.q10, self.q90, _ = make_asymmetric_data()

    def test_returns_four_values(self):
        out = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        assert len(out) == 4

    def test_output_shapes(self):
        center, lower, upper, _ = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        n = len(self.point_pred)
        assert center.shape == (n,)
        assert lower.shape  == (n,)
        assert upper.shape  == (n,)

    def test_beta_zero_center_equals_point_pred(self):
        center, _, _, _ = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=0.0
        )
        np.testing.assert_allclose(center, self.point_pred)

    def test_beta_inf_center_equals_anchor(self):
        center, _, _, _ = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=np.inf
        )
        np.testing.assert_allclose(center, self.anchor)

    def test_lower_upper_anchored_to_corrected_center(self):
        """lower = center - r_down, upper = center + r_up."""
        center, lower, upper, _ = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        r_down = np.maximum(self.anchor - self.q10, 0.0)
        r_up   = np.maximum(self.q90   - self.anchor, 0.0)
        np.testing.assert_allclose(lower, center - r_down, rtol=1e-6)
        np.testing.assert_allclose(upper, center + r_up,   rtol=1e-6)

    def test_total_width_equals_original_interval(self):
        """Total width (r_down + r_up) equals q90 - q10 (no quantile crossing)."""
        _, lower, upper, _ = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        width_new      = upper - lower
        width_original = self.q90 - self.q10
        np.testing.assert_allclose(width_new, width_original, rtol=1e-6)

    def test_diagnostics_keys(self):
        _, _, _, diag = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        for key in ["beta", "mean_lambda", "std_lambda", "mean_delta",
                    "mean_r_down", "mean_r_up", "mean_width",
                    "mean_correction", "mean_skew_index"]:
            assert key in diag, f"Missing diagnostics key: {key}"

    def test_skew_index_reflects_asymmetry(self):
        """When r_up > r_down on average, skew_index should be positive."""
        anchor = np.zeros(50)
        q10    = anchor - 1.0   # r_down = 1
        q90    = anchor + 3.0   # r_up   = 3
        point  = anchor + 0.1
        _, _, _, diag = asymmetric_soft_pullback(point, anchor, q10, q90, beta=1.0)
        assert diag["mean_skew_index"] > 0.0


# ---------------------------------------------------------------------------
# AsymmetricSnailMechanism
# ---------------------------------------------------------------------------

class TestAsymmetricSnailMechanismInit:
    def test_default_beta_values(self):
        mech = AsymmetricSnailMechanism()
        assert 0.0 in mech.beta_values
        assert float("inf") in mech.beta_values

    def test_custom_beta_values(self):
        mech = AsymmetricSnailMechanism(beta_values=[0.1, 1.0])
        assert mech.beta_values == [0.1, 1.0]

    def test_default_crossing_threshold(self):
        mech = AsymmetricSnailMechanism()
        assert mech.crossing_threshold == 0.1


class TestAsymmetricSnailMechanismApply:
    def setup_method(self):
        self.mech = AsymmetricSnailMechanism()
        self.point_pred, self.anchor, self.q10, self.q90, _ = make_asymmetric_data()

    def test_delegates_to_asymmetric_soft_pullback(self):
        """apply() should return the same result as the free function."""
        result_class = self.mech.apply(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        result_func  = asymmetric_soft_pullback(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        np.testing.assert_allclose(result_class[0], result_func[0])
        np.testing.assert_allclose(result_class[1], result_func[1])
        np.testing.assert_allclose(result_class[2], result_func[2])

    def test_default_beta_is_one(self):
        out_default = self.mech.apply(self.point_pred, self.anchor, self.q10, self.q90)
        out_explicit = self.mech.apply(
            self.point_pred, self.anchor, self.q10, self.q90, beta=1.0
        )
        np.testing.assert_allclose(out_default[0], out_explicit[0])


class TestAsymmetricSnailMechanismScanBeta:
    def setup_method(self):
        self.mech = AsymmetricSnailMechanism()
        self.point_pred, self.anchor, self.q10, self.q90, _ = make_asymmetric_data()

    def test_returns_dict_for_each_beta(self):
        results = self.mech.scan_beta(
            self.point_pred, self.anchor, self.q10, self.q90
        )
        assert set(results.keys()) == set(self.mech.beta_values)

    def test_cached_on_instance(self):
        self.mech.scan_beta(self.point_pred, self.anchor, self.q10, self.q90)
        assert len(self.mech._scan_cache) == len(self.mech.beta_values)

    def test_each_entry_is_four_tuple(self):
        results = self.mech.scan_beta(
            self.point_pred, self.anchor, self.q10, self.q90
        )
        for beta, entry in results.items():
            assert len(entry) == 4


class TestAsymmetricSnailMechanismSelectBeta:
    def setup_method(self):
        self.mech = AsymmetricSnailMechanism()
        self.point_pred, self.anchor, self.q10, self.q90, self.y_true = (
            make_asymmetric_data()
        )

    def test_returns_best_beta_in_beta_values(self):
        best_beta, _ = self.mech.select_beta(
            self.point_pred, self.anchor, self.q10, self.q90, self.y_true
        )
        assert best_beta in self.mech.beta_values

    def test_best_beta_minimises_score(self):
        best_beta, beta_scores = self.mech.select_beta(
            self.point_pred, self.anchor, self.q10, self.q90, self.y_true
        )
        best_score = beta_scores[best_beta]["metrics"]["score"]
        for beta, info in beta_scores.items():
            assert info["metrics"]["score"] >= best_score - 1e-10

    def test_all_betas_evaluated(self):
        _, beta_scores = self.mech.select_beta(
            self.point_pred, self.anchor, self.q10, self.q90, self.y_true
        )
        assert set(beta_scores.keys()) == set(self.mech.beta_values)

    def test_custom_scoring_function_called_for_each_beta(self):
        calls = []

        def scorer(y_true, center, lower, upper):
            calls.append(1)
            return {"score": 0.0, "winkler_mean": 0.0,
                    "coverage_error": 0.0, "coverage": 0.8, "mae": 0.0}

        self.mech.select_beta(
            self.point_pred, self.anchor, self.q10, self.q90, self.y_true,
            scoring_func=scorer
        )
        assert len(calls) == len(self.mech.beta_values)

    def test_score_info_contains_expected_keys(self):
        _, beta_scores = self.mech.select_beta(
            self.point_pred, self.anchor, self.q10, self.q90, self.y_true
        )
        for beta, info in beta_scores.items():
            for key in ["corrected_center", "lower", "upper", "diagnostics", "metrics"]:
                assert key in info, f"β={beta} missing key '{key}'"


class TestAsymmetricSnailMechanismCheckCrossing:
    def setup_method(self):
        self.mech = AsymmetricSnailMechanism(crossing_threshold=0.1)

    def test_no_crossings_is_valid(self):
        q10 = np.array([1.0, 2.0, 3.0])
        q90 = np.array([5.0, 6.0, 7.0])
        info = self.mech.check_crossing(q10, q90)
        assert bool(info["is_valid"]) is True
        assert np.isclose(info["crossing_rate"], 0.0)

    def test_all_crossings_is_invalid(self):
        q10 = np.array([5.0, 6.0])
        q90 = np.array([1.0, 2.0])
        info = self.mech.check_crossing(q10, q90)
        assert bool(info["is_valid"]) is False

    def test_returns_threshold_in_result(self):
        info = self.mech.check_crossing(np.array([1.0]), np.array([2.0]))
        assert info["threshold"] == self.mech.crossing_threshold

    def test_crossing_rate_consistent_with_metrics_module(self):
        """Result should match the crossing_rate function from evaluation/metrics."""
        from evaluation.metrics import crossing_rate as metrics_crossing_rate
        q10 = np.array([5.0, 1.0, 5.0, 1.0])
        q90 = np.array([2.0, 3.0, 2.0, 3.0])
        info = self.mech.check_crossing(q10, q90)
        expected_rate = metrics_crossing_rate(q10, q90)
        assert np.isclose(info["crossing_rate"], expected_rate)
