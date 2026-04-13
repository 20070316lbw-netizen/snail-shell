"""
Tests for the SnailMechanism class in core/snail_mechanism.py

Covers:
  SnailMechanism.__init__
  SnailMechanism.apply
  SnailMechanism.scan_beta
  SnailMechanism.select_beta
  SnailMechanism.check_crossing
"""

import numpy as np
import pytest
from core.snail_mechanism import SnailMechanism


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    point_pred = rng.standard_normal(n) * 0.1
    anchor     = rng.standard_normal(n) * 0.05
    radius     = np.abs(rng.standard_normal(n)) * 0.05 + 0.02
    y_true     = rng.standard_normal(n) * 0.08
    return point_pred, anchor, radius, y_true


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestSnailMechanismInit:
    def test_default_beta_values(self):
        sm = SnailMechanism()
        assert 0.0 in sm.beta_values
        assert np.inf in sm.beta_values

    def test_custom_beta_values(self):
        betas = [0.0, 1.0, 10.0]
        sm = SnailMechanism(beta_values=betas)
        assert sm.beta_values == betas

    def test_default_crossing_threshold(self):
        sm = SnailMechanism()
        assert sm.crossing_threshold == 0.1

    def test_custom_crossing_threshold(self):
        sm = SnailMechanism(crossing_threshold=0.05)
        assert sm.crossing_threshold == 0.05

    def test_results_initially_empty(self):
        sm = SnailMechanism()
        assert sm.results == {}


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

class TestSnailMechanismApply:
    def setup_method(self):
        self.sm = SnailMechanism()
        self.point_pred, self.anchor, self.radius, self.y_true = make_data()

    def test_returns_four_values(self):
        out = self.sm.apply(self.point_pred, self.anchor, self.radius, beta=1.0)
        assert len(out) == 4

    def test_corrected_pred_shape(self):
        corrected_pred, corrected_q10, corrected_q90, diag = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=1.0
        )
        assert corrected_pred.shape == self.point_pred.shape

    def test_q10_and_q90_symmetric_around_corrected_pred(self):
        corrected_pred, corrected_q10, corrected_q90, _ = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=1.0
        )
        # corrected_q10 = corrected_pred - radius
        # corrected_q90 = corrected_pred + radius
        np.testing.assert_allclose(corrected_q10, corrected_pred - self.radius)
        np.testing.assert_allclose(corrected_q90, corrected_pred + self.radius)

    def test_beta_zero_leaves_point_pred_unchanged(self):
        corrected_pred, _, _, _ = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=0.0
        )
        np.testing.assert_allclose(corrected_pred, self.point_pred)

    def test_beta_inf_pulls_to_anchor(self):
        corrected_pred, _, _, _ = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=np.inf
        )
        np.testing.assert_allclose(corrected_pred, self.anchor)

    def test_corrected_pred_between_anchor_and_point_pred(self):
        """For 0 < β < ∞ the correction must lie strictly between anchor and point."""
        corrected_pred, _, _, _ = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=1.0
        )
        # Distance from anchor should be ≤ original distance
        orig_dist = np.abs(self.point_pred - self.anchor)
        corr_dist = np.abs(corrected_pred   - self.anchor)
        assert np.all(corr_dist <= orig_dist + 1e-10)

    def test_diagnostics_keys(self):
        _, _, _, diag = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=1.0
        )
        for key in ["beta", "mean_alpha", "std_alpha", "min_alpha",
                    "max_alpha", "mean_deviation", "mean_correction"]:
            assert key in diag, f"Missing diagnostics key: {key}"

    def test_diagnostics_beta_stored(self):
        _, _, _, diag = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=2.0
        )
        assert diag["beta"] == 2.0

    def test_alpha_in_0_1(self):
        _, _, _, diag = self.sm.apply(
            self.point_pred, self.anchor, self.radius, beta=1.0
        )
        assert 0.0 <= diag["min_alpha"] <= diag["max_alpha"] <= 1.0


# ---------------------------------------------------------------------------
# scan_beta
# ---------------------------------------------------------------------------

class TestSnailMechanismScanBeta:
    def setup_method(self):
        self.sm = SnailMechanism()
        self.point_pred, self.anchor, self.radius, _ = make_data()

    def test_returns_dict_keyed_by_beta(self):
        results = self.sm.scan_beta(self.point_pred, self.anchor, self.radius)
        assert set(results.keys()) == set(self.sm.beta_values)

    def test_each_entry_has_four_elements(self):
        results = self.sm.scan_beta(self.point_pred, self.anchor, self.radius)
        for beta, entry in results.items():
            assert len(entry) == 4, f"Expected 4 elements for β={beta}"

    def test_results_stored_on_instance(self):
        self.sm.scan_beta(self.point_pred, self.anchor, self.radius)
        assert len(self.sm.results) == len(self.sm.beta_values)

    def test_higher_beta_gives_more_correction(self):
        """Mean correction should be monotonically non-decreasing in β."""
        results = self.sm.scan_beta(self.point_pred, self.anchor, self.radius)
        finite_betas = sorted([b for b in self.sm.beta_values if not np.isinf(b)])
        corrections = [results[b][3]["mean_correction"] for b in finite_betas]
        for i in range(len(corrections) - 1):
            assert corrections[i] <= corrections[i + 1] + 1e-10

    def test_custom_beta_values_scanned(self):
        sm = SnailMechanism(beta_values=[0.1, 0.5])
        results = sm.scan_beta(self.point_pred, self.anchor, self.radius)
        assert set(results.keys()) == {0.1, 0.5}


# ---------------------------------------------------------------------------
# select_beta
# ---------------------------------------------------------------------------

class TestSnailMechanismSelectBeta:
    def setup_method(self):
        self.sm = SnailMechanism()
        self.point_pred, self.anchor, self.radius, self.y_true = make_data()

    def test_returns_best_beta_and_scores_dict(self):
        best_beta, beta_scores = self.sm.select_beta(
            self.point_pred, self.anchor, self.radius, self.y_true
        )
        assert best_beta in self.sm.beta_values
        assert isinstance(beta_scores, dict)

    def test_best_beta_has_lowest_score(self):
        best_beta, beta_scores = self.sm.select_beta(
            self.point_pred, self.anchor, self.radius, self.y_true
        )
        best_score = beta_scores[best_beta]["metrics"]["score"]
        for beta, info in beta_scores.items():
            assert info["metrics"]["score"] >= best_score - 1e-10

    def test_all_betas_evaluated(self):
        _, beta_scores = self.sm.select_beta(
            self.point_pred, self.anchor, self.radius, self.y_true
        )
        assert set(beta_scores.keys()) == set(self.sm.beta_values)

    def test_custom_scoring_function_used(self):
        """A custom scoring function that always returns 0 → any beta could win."""
        call_log = []

        def my_scorer(y_true, pred, q10, q90):
            call_log.append(1)
            return {"score": 0.0}

        self.sm.select_beta(
            self.point_pred, self.anchor, self.radius, self.y_true,
            scoring_func=my_scorer
        )
        assert len(call_log) == len(self.sm.beta_values)

    def test_each_score_entry_contains_predictions(self):
        _, beta_scores = self.sm.select_beta(
            self.point_pred, self.anchor, self.radius, self.y_true
        )
        for beta, info in beta_scores.items():
            for key in ["corrected_pred", "corrected_q10", "corrected_q90",
                        "diagnostics", "metrics"]:
                assert key in info, f"β={beta} missing key '{key}'"


# ---------------------------------------------------------------------------
# check_crossing
# ---------------------------------------------------------------------------

class TestSnailMechanismCheckCrossing:
    def setup_method(self):
        self.sm = SnailMechanism(crossing_threshold=0.1)

    def test_no_crossing_is_valid(self):
        q10 = np.array([1.0, 2.0, 3.0])
        q90 = np.array([5.0, 6.0, 7.0])
        info = self.sm.check_crossing(q10, q90)
        assert bool(info["is_valid"]) is True
        assert np.isclose(info["crossing_rate"], 0.0)

    def test_all_crossing_is_invalid(self):
        q10 = np.array([5.0, 6.0, 7.0])
        q90 = np.array([1.0, 2.0, 3.0])
        info = self.sm.check_crossing(q10, q90)
        assert bool(info["is_valid"]) is False
        assert np.isclose(info["crossing_rate"], 1.0)

    def test_threshold_respected(self):
        # 1 out of 10 crossings → rate = 0.1, threshold = 0.1 → valid (<=)
        q10 = np.array([5.0] + [1.0] * 9)
        q90 = np.array([2.0] + [5.0] * 9)
        info = self.sm.check_crossing(q10, q90)
        assert bool(info["is_valid"]) is True

    def test_returns_threshold_in_result(self):
        info = self.sm.check_crossing(np.array([1.0]), np.array([2.0]))
        assert info["threshold"] == self.sm.crossing_threshold

    def test_message_field_present(self):
        info = self.sm.check_crossing(np.array([1.0]), np.array([2.0]))
        assert "message" in info
