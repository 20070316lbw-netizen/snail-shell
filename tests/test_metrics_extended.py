"""
Tests for the untested functions in evaluation/metrics.py

Covers:
  actual_coverage, winkler_score, interval_width, mean_absolute_error,
  mean_squared_error, rank_ic, pinball_loss, crossing_rate,
  composite_score, calculate_all_metrics, paired_t_test,
  evaluate_methods, skewness_index, asymmetric_width_stats,
  calculate_all_metrics_asymmetric
"""

import numpy as np
import pytest
import pandas as pd
from evaluation.metrics import (
    actual_coverage,
    winkler_score,
    interval_width,
    mean_absolute_error,
    mean_squared_error,
    rank_ic,
    pinball_loss,
    crossing_rate,
    composite_score,
    calculate_all_metrics,
    paired_t_test,
    evaluate_methods,
    skewness_index,
    asymmetric_width_stats,
    calculate_all_metrics_asymmetric,
)


# ---------------------------------------------------------------------------
# actual_coverage
# ---------------------------------------------------------------------------

class TestActualCoverage:
    def test_all_inside(self):
        y_true = np.array([3.0, 5.0, 7.0])
        lower  = np.array([1.0, 1.0, 1.0])
        upper  = np.array([9.0, 9.0, 9.0])
        assert np.isclose(actual_coverage(y_true, lower, upper), 1.0)

    def test_none_inside(self):
        y_true = np.array([0.0, 10.0])
        lower  = np.array([2.0, 2.0])
        upper  = np.array([8.0, 8.0])
        assert np.isclose(actual_coverage(y_true, lower, upper), 0.0)

    def test_partial_coverage(self):
        # 3 out of 5 inside -> 0.6
        y_true = np.array([3.0, 5.0, 7.0, 0.0, 10.0])
        lower  = np.array([2.0] * 5)
        upper  = np.array([8.0] * 5)
        assert np.isclose(actual_coverage(y_true, lower, upper), 0.6)

    def test_boundary_values_are_included(self):
        y_true = np.array([2.0, 8.0])
        lower  = np.array([2.0, 2.0])
        upper  = np.array([8.0, 8.0])
        assert np.isclose(actual_coverage(y_true, lower, upper), 1.0)

    def test_returns_float(self):
        y_true = np.array([5.0])
        lower  = np.array([1.0])
        upper  = np.array([9.0])
        result = actual_coverage(y_true, lower, upper)
        assert isinstance(float(result), float)


# ---------------------------------------------------------------------------
# winkler_score
# ---------------------------------------------------------------------------

class TestWinklerScore:
    def test_perfect_coverage_no_penalty(self):
        """When all points are inside, score equals interval width only."""
        y_true = np.array([5.0, 5.0, 5.0])
        lower  = np.array([4.0, 4.0, 4.0])
        upper  = np.array([6.0, 6.0, 6.0])
        # width = 2; no penalty -> winkler = 2.0
        assert np.isclose(winkler_score(y_true, lower, upper), 2.0)

    def test_penalty_below_lower(self):
        """Point below lower bound incurs (2/alpha) * (lower - y_true) penalty."""
        y_true = np.array([3.0])
        lower  = np.array([5.0])
        upper  = np.array([7.0])
        alpha  = 0.2
        width  = 2.0
        penalty = (2 / alpha) * (5.0 - 3.0)  # 10 * 2 = 20
        expected = width + penalty             # 22
        assert np.isclose(winkler_score(y_true, lower, upper, alpha=alpha), expected)

    def test_penalty_above_upper(self):
        """Point above upper bound incurs (2/alpha) * (y_true - upper) penalty."""
        y_true = np.array([9.0])
        lower  = np.array([5.0])
        upper  = np.array([7.0])
        alpha  = 0.2
        width  = 2.0
        penalty = (2 / alpha) * (9.0 - 7.0)  # 10 * 2 = 20
        expected = width + penalty             # 22
        assert np.isclose(winkler_score(y_true, lower, upper, alpha=alpha), expected)

    def test_mean_over_multiple_samples(self):
        """Score is averaged over samples."""
        y_true = np.array([5.0, 9.0])   # first inside, second outside
        lower  = np.array([4.0, 4.0])
        upper  = np.array([6.0, 6.0])
        alpha  = 0.2
        w0 = 2.0                                          # inside: just width
        w1 = 2.0 + (2 / alpha) * (9.0 - 6.0)            # outside: width + penalty
        expected = (w0 + w1) / 2
        assert np.isclose(winkler_score(y_true, lower, upper, alpha=alpha), expected)

    def test_score_increases_with_worse_coverage(self):
        """A model that misses more should have a higher (worse) Winkler score."""
        y_true = np.array([10.0])
        lower  = np.array([4.0])
        upper  = np.array([6.0])
        score_close = winkler_score(y_true, lower, upper)  # y 4 units above upper
        # Move y further away
        score_far   = winkler_score(np.array([20.0]), lower, upper)
        assert score_far > score_close


# ---------------------------------------------------------------------------
# interval_width
# ---------------------------------------------------------------------------

class TestIntervalWidth:
    def test_uniform_width(self):
        lower = np.array([1.0, 1.0, 1.0])
        upper = np.array([3.0, 3.0, 3.0])
        assert np.isclose(interval_width(lower, upper), 2.0)

    def test_variable_widths_averaged(self):
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 4.0])
        # widths: [2, 4] -> mean = 3.0
        assert np.isclose(interval_width(lower, upper), 3.0)

    def test_zero_width(self):
        lower = np.array([5.0, 5.0])
        upper = np.array([5.0, 5.0])
        assert np.isclose(interval_width(lower, upper), 0.0)

    def test_returns_scalar(self):
        result = interval_width(np.array([1.0]), np.array([2.0]))
        assert np.ndim(result) == 0


# ---------------------------------------------------------------------------
# mean_absolute_error
# ---------------------------------------------------------------------------

class TestMeanAbsoluteError:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(mean_absolute_error(y_true, y_pred), 0.0)

    def test_constant_error(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert np.isclose(mean_absolute_error(y_true, y_pred), 1.0)

    def test_mixed_sign_residuals(self):
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([2.0, 2.0])  # residuals: -1, +1
        assert np.isclose(mean_absolute_error(y_true, y_pred), 1.0)

    def test_asymmetric_errors(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 3.0])  # |errors|: 1, 3 -> mean 2
        assert np.isclose(mean_absolute_error(y_true, y_pred), 2.0)


# ---------------------------------------------------------------------------
# mean_squared_error
# ---------------------------------------------------------------------------

class TestMeanSquaredError:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert np.isclose(mean_squared_error(y_true, y_pred), 0.0)

    def test_constant_error(self):
        y_true = np.array([3.0, 3.0, 3.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        # squared errors: 4, 4, 4 -> mean = 4
        assert np.isclose(mean_squared_error(y_true, y_pred), 4.0)

    def test_mse_greater_than_mae_for_large_errors(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 3.0])
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        # MSE penalises large errors more: (1+9)/2=5 > (1+3)/2=2
        assert mse > mae


# ---------------------------------------------------------------------------
# rank_ic
# ---------------------------------------------------------------------------

class TestRankIC:
    def test_perfect_rank_correlation(self):
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        assert np.isclose(rank_ic(y_pred, y_true), 1.0)

    def test_perfect_inverse_rank_correlation(self):
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(rank_ic(y_pred, y_true), -1.0)

    def test_zero_rank_correlation(self):
        # y_pred ranks: 1,2,3,4  |  y_true ranks: 2,4,1,3
        # d = [-1,-2,2,1]  d²sum=10  r = 1-6*10/(4*15) = 0
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        y_true = np.array([20.0, 40.0, 10.0, 30.0])
        result = rank_ic(y_pred, y_true)
        assert np.isclose(result, 0.0)

    def test_result_in_range(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal(100)
        y_true = rng.standard_normal(100)
        result = rank_ic(y_pred, y_true)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# pinball_loss  (metrics.py version)
# ---------------------------------------------------------------------------

class TestPinballLossMetrics:
    def test_zero_loss_for_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        for alpha in [0.1, 0.5, 0.9]:
            assert np.isclose(pinball_loss(y_true, y_pred, alpha=alpha), 0.0)

    def test_default_alpha_is_half_mae(self):
        y_true = np.array([1.0, 3.0, 5.0])
        y_pred = np.array([0.0, 2.0, 7.0])
        mae = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.5), mae / 2.0)

    def test_asymmetry_alpha_0_1_penalises_over_prediction(self):
        """alpha=0.1: over-predicting (y_pred > y_true) is penalised more."""
        y_true = np.array([0.0])
        y_pred_over  = np.array([1.0])   # y_pred > y_true
        y_pred_under = np.array([-1.0])  # y_pred < y_true
        loss_over  = pinball_loss(y_true, y_pred_over,  alpha=0.1)
        loss_under = pinball_loss(y_true, y_pred_under, alpha=0.1)
        assert loss_over > loss_under

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        for alpha in [0.1, 0.5, 0.9]:
            assert pinball_loss(y_true, y_pred, alpha=alpha) >= 0.0


# ---------------------------------------------------------------------------
# crossing_rate
# ---------------------------------------------------------------------------

class TestCrossingRate:
    def test_no_crossings(self):
        q10 = np.array([1.0, 2.0, 3.0])
        q90 = np.array([5.0, 6.0, 7.0])
        assert np.isclose(crossing_rate(q10, q90), 0.0)

    def test_all_crossings(self):
        q10 = np.array([5.0, 6.0, 7.0])
        q90 = np.array([1.0, 2.0, 3.0])
        assert np.isclose(crossing_rate(q10, q90), 1.0)

    def test_partial_crossings(self):
        q10 = np.array([5.0, 1.0, 5.0, 1.0])
        q90 = np.array([2.0, 3.0, 2.0, 3.0])
        # 2 crossings out of 4 -> 0.5
        assert np.isclose(crossing_rate(q10, q90), 0.5)

    def test_equal_values_not_crossing(self):
        """q10 == q90 should NOT count as a crossing (not strictly >)."""
        q10 = np.array([3.0, 3.0])
        q90 = np.array([3.0, 3.0])
        assert np.isclose(crossing_rate(q10, q90), 0.0)

    def test_empty_array_returns_zero(self):
        assert np.isclose(crossing_rate(np.array([]), np.array([])), 0.0)


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_zero_ce(self):
        """When CE=0, composite score equals winkler_mean."""
        assert np.isclose(composite_score(3.0, 0.0, k=2.0), 3.0)

    def test_formula(self):
        assert np.isclose(composite_score(3.0, 0.5, k=2.0), 4.0)

    def test_higher_k_penalises_more(self):
        score_low_k  = composite_score(1.0, 0.2, k=1.0)
        score_high_k = composite_score(1.0, 0.2, k=5.0)
        assert score_high_k > score_low_k

    def test_lower_is_better(self):
        good = composite_score(1.0, 0.01)
        bad  = composite_score(3.0, 0.5)
        assert good < bad


# ---------------------------------------------------------------------------
# calculate_all_metrics
# ---------------------------------------------------------------------------

class TestCalculateAllMetrics:
    def setup_method(self):
        rng = np.random.default_rng(7)
        n = 200
        self.y_true   = rng.standard_normal(n)
        self.point    = self.y_true + rng.standard_normal(n) * 0.1
        self.lower    = self.point - 1.0
        self.upper    = self.point + 1.0
        self.q10      = self.lower
        self.q90      = self.upper

    def test_returns_dict_with_required_keys(self):
        metrics = calculate_all_metrics(
            self.y_true, self.point, self.lower, self.upper
        )
        for key in ["actual_coverage", "coverage_error", "winkler_score",
                    "interval_width", "mae", "mse", "rank_ic", "composite_score"]:
            assert key in metrics, f"Missing key: {key}"

    def test_crossing_rate_included_when_quantiles_provided(self):
        metrics = calculate_all_metrics(
            self.y_true, self.point, self.lower, self.upper,
            q10=self.q10, q90=self.q90
        )
        assert "crossing_rate" in metrics

    def test_crossing_rate_absent_when_quantiles_not_provided(self):
        metrics = calculate_all_metrics(
            self.y_true, self.point, self.lower, self.upper
        )
        assert "crossing_rate" not in metrics

    def test_metric_values_are_reasonable(self):
        metrics = calculate_all_metrics(
            self.y_true, self.point, self.lower, self.upper
        )
        assert 0.0 <= metrics["actual_coverage"] <= 1.0
        assert metrics["coverage_error"] >= 0.0
        assert metrics["winkler_score"] >= 0.0
        assert metrics["interval_width"] >= 0.0
        assert metrics["mae"] >= 0.0
        assert metrics["mse"] >= 0.0
        assert -1.0 <= metrics["rank_ic"] <= 1.0

    def test_perfect_predictions_have_low_mae_and_mse(self):
        metrics = calculate_all_metrics(
            self.y_true, self.y_true, self.lower, self.upper
        )
        assert np.isclose(metrics["mae"], 0.0)
        assert np.isclose(metrics["mse"], 0.0)


# ---------------------------------------------------------------------------
# paired_t_test
# ---------------------------------------------------------------------------

class TestPairedTTest:
    def test_returns_tuple_of_two_floats(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        t, p = paired_t_test(a, b)
        assert isinstance(float(t), float)
        assert isinstance(float(p), float)

    def test_p_value_in_range(self):
        rng = np.random.default_rng(99)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        _, p = paired_t_test(a, b)
        assert 0.0 <= p <= 1.0

    def test_identical_arrays_gives_nan_t(self):
        """d = 0 for all → std of differences is 0 → t is NaN."""
        a = np.array([1.0, 2.0, 3.0])
        t, _ = paired_t_test(a, a)
        assert np.isnan(t)

    def test_large_consistent_difference_gives_small_p(self):
        """If method A is consistently better, p should be small."""
        a = np.ones(100) * 2.0
        b = np.ones(100) * 1.0
        _, p = paired_t_test(a, b)
        assert p < 0.01


# ---------------------------------------------------------------------------
# evaluate_methods
# ---------------------------------------------------------------------------

class TestEvaluateMethods:
    def setup_method(self):
        rng = np.random.default_rng(13)
        n = 100
        self.y_true = rng.standard_normal(n)
        point = self.y_true + rng.standard_normal(n) * 0.2
        self.methods = {
            "ModelA": {
                "point_pred": point,
                "lower": point - 1.0,
                "upper": point + 1.0,
            },
            "ModelB": {
                "point_pred": point + 0.5,
                "lower": point - 0.5,
                "upper": point + 0.5,
            },
        }

    def test_returns_dataframe(self):
        result = evaluate_methods(self.y_true, self.methods)
        assert isinstance(result, pd.DataFrame)

    def test_has_one_row_per_method(self):
        result = evaluate_methods(self.y_true, self.methods)
        assert len(result) == len(self.methods)

    def test_method_column_contains_method_names(self):
        result = evaluate_methods(self.y_true, self.methods)
        assert set(result["Method"]) == set(self.methods.keys())

    def test_crossing_rate_excluded_when_requested(self):
        # Add q10/q90 so it would normally be included
        for name, pred in self.methods.items():
            pred["q10"] = pred["lower"]
            pred["q90"] = pred["upper"]
        result = evaluate_methods(self.y_true, self.methods, include_crossing=False)
        assert "crossing_rate" not in result.columns

    def test_crossing_rate_included_with_quantiles(self):
        for name, pred in self.methods.items():
            pred["q10"] = pred["lower"]
            pred["q90"] = pred["upper"]
        result = evaluate_methods(self.y_true, self.methods, include_crossing=True)
        assert "crossing_rate" in result.columns


# ---------------------------------------------------------------------------
# skewness_index
# ---------------------------------------------------------------------------

class TestSkewnessIndex:
    def test_symmetric_interval_gives_zero(self):
        center = np.array([5.0, 5.0, 5.0])
        lower  = center - 1.0
        upper  = center + 1.0
        # r_down == r_up -> SI = 0
        assert np.isclose(skewness_index(lower, upper, center), 0.0)

    def test_positive_skew_when_upper_half_wider(self):
        center = np.array([0.0])
        lower  = center - 1.0   # r_down = 1
        upper  = center + 3.0   # r_up   = 3
        si = skewness_index(lower, upper, center)
        assert si > 0.0

    def test_negative_skew_when_lower_half_wider(self):
        center = np.array([0.0])
        lower  = center - 3.0   # r_down = 3
        upper  = center + 1.0   # r_up   = 1
        si = skewness_index(lower, upper, center)
        assert si < 0.0

    def test_result_in_range(self):
        rng = np.random.default_rng(5)
        n = 100
        center = rng.standard_normal(n)
        half = np.abs(rng.standard_normal(n)) + 0.1
        lower = center - half
        upper = center + half * rng.uniform(0.5, 2.0, n)
        si = skewness_index(lower, upper, center)
        assert -1.0 <= si <= 1.0


# ---------------------------------------------------------------------------
# asymmetric_width_stats
# ---------------------------------------------------------------------------

class TestAsymmetricWidthStats:
    def test_returns_expected_keys(self):
        center = np.array([0.0, 0.0])
        lower  = center - 1.0
        upper  = center + 2.0
        stats  = asymmetric_width_stats(lower, upper, center)
        for key in ["mean_r_down", "std_r_down", "mean_r_up",
                    "std_r_up", "skewness_index", "asymmetry_ratio"]:
            assert key in stats, f"Missing key: {key}"

    def test_symmetric_gives_zero_skewness_and_unit_ratio(self):
        center = np.array([0.0, 0.0, 0.0])
        lower  = center - 1.0
        upper  = center + 1.0
        stats  = asymmetric_width_stats(lower, upper, center)
        assert np.isclose(stats["skewness_index"], 0.0)
        assert np.isclose(stats["asymmetry_ratio"], 1.0)

    def test_asymmetry_ratio_greater_than_one_when_upside_wider(self):
        center = np.array([0.0])
        lower  = center - 1.0
        upper  = center + 3.0
        stats  = asymmetric_width_stats(lower, upper, center)
        assert stats["asymmetry_ratio"] > 1.0

    def test_mean_r_values_correct(self):
        center = np.array([0.0, 0.0])
        lower  = center - np.array([1.0, 3.0])
        upper  = center + np.array([2.0, 4.0])
        stats  = asymmetric_width_stats(lower, upper, center)
        # mean r_down = (1+3)/2 = 2; mean r_up = (2+4)/2 = 3
        assert np.isclose(stats["mean_r_down"], 2.0)
        assert np.isclose(stats["mean_r_up"],   3.0)


# ---------------------------------------------------------------------------
# calculate_all_metrics_asymmetric
# ---------------------------------------------------------------------------

class TestCalculateAllMetricsAsymmetric:
    def setup_method(self):
        rng = np.random.default_rng(21)
        n = 150
        self.y_true  = rng.standard_normal(n)
        self.center  = self.y_true + rng.standard_normal(n) * 0.1
        self.lower   = self.center - np.abs(rng.standard_normal(n)) * 0.5
        self.upper   = self.center + np.abs(rng.standard_normal(n)) * 0.5

    def test_includes_standard_metrics(self):
        metrics = calculate_all_metrics_asymmetric(
            self.y_true, self.center, self.lower, self.upper
        )
        for key in ["actual_coverage", "coverage_error", "winkler_score",
                    "mae", "mse", "composite_score"]:
            assert key in metrics

    def test_includes_asymmetric_metrics(self):
        metrics = calculate_all_metrics_asymmetric(
            self.y_true, self.center, self.lower, self.upper
        )
        for key in ["skewness_index", "mean_r_down", "mean_r_up",
                    "std_r_down", "std_r_up", "asymmetry_ratio"]:
            assert key in metrics

    def test_center_defaults_to_point_pred(self):
        """When center is omitted, it should use point_pred for asymmetric stats."""
        m_with_center = calculate_all_metrics_asymmetric(
            self.y_true, self.center, self.lower, self.upper, center=self.center
        )
        m_without_center = calculate_all_metrics_asymmetric(
            self.y_true, self.center, self.lower, self.upper, center=None
        )
        # Both should be identical since point_pred == center here
        assert np.isclose(
            m_with_center["skewness_index"],
            m_without_center["skewness_index"],
        )
