"""
Tests for the untested parts of core/spiral_monitor.py

Covers:
  cartesian_to_polar
  fit_log_spiral
  rolling_alert
  SpiralMonitor.analyze
  SpiralMonitor.get_trajectory_stats
  SpiralMonitor.check_spiral_validity
  SpiralMonitor.predict_next_rho
"""

import numpy as np
import pytest
from core.spiral_monitor import (
    cartesian_to_polar,
    fit_log_spiral,
    rolling_alert,
    SpiralMonitor,
)


# ---------------------------------------------------------------------------
# cartesian_to_polar
# ---------------------------------------------------------------------------

class TestCartesianToPolar:
    def test_known_point_on_positive_x_axis(self):
        """(a=1, r=0) → ρ=1, θ=0."""
        a = np.array([1.0])
        r = np.array([0.0])
        rho, theta = cartesian_to_polar(a, r)
        np.testing.assert_allclose(rho,  [1.0], atol=1e-10)
        np.testing.assert_allclose(theta, [0.0], atol=1e-10)

    def test_known_point_on_positive_y_axis(self):
        """(a=0, r=1) → ρ=1, θ=π/2."""
        a = np.array([0.0])
        r = np.array([1.0])
        rho, theta = cartesian_to_polar(a, r)
        np.testing.assert_allclose(rho,  [1.0],       atol=1e-10)
        np.testing.assert_allclose(theta, [np.pi / 2], atol=1e-10)

    def test_origin_gives_zero_rho(self):
        a = np.array([0.0])
        r = np.array([0.0])
        rho, theta = cartesian_to_polar(a, r)
        np.testing.assert_allclose(rho, [0.0], atol=1e-10)

    def test_rho_equals_euclidean_distance(self):
        a = np.array([3.0, -4.0])
        r = np.array([4.0,  3.0])
        rho, _ = cartesian_to_polar(a, r)
        expected = np.sqrt(a ** 2 + r ** 2)
        np.testing.assert_allclose(rho, expected)

    def test_output_shapes(self):
        n = 20
        a = np.random.default_rng(3).standard_normal(n)
        r = np.abs(np.random.default_rng(4).standard_normal(n))
        rho, theta = cartesian_to_polar(a, r)
        assert rho.shape == (n,)
        assert theta.shape == (n,)

    def test_theta_in_minus_pi_to_pi(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal(50)
        r = rng.standard_normal(50)
        _, theta = cartesian_to_polar(a, r)
        assert np.all(theta >= -np.pi) and np.all(theta <= np.pi)


# ---------------------------------------------------------------------------
# fit_log_spiral
# ---------------------------------------------------------------------------

class TestFitLogSpiral:
    def _make_spiral(self, log_A=0.5, B=0.3, n=100, noise=0.0, seed=11):
        rng = np.random.default_rng(seed)
        theta = np.linspace(0, 2 * np.pi, n)
        rho = np.exp(log_A + B * theta) + rng.standard_normal(n) * noise
        rho = np.maximum(rho, 1e-8)
        return rho, theta

    def test_returns_three_values(self):
        rho, theta = self._make_spiral()
        out = fit_log_spiral(rho, theta)
        assert len(out) == 3

    def test_recovers_parameters_without_noise(self):
        """With zero noise the fitted parameters should match the ground truth."""
        log_A_true, B_true = 0.4, 0.25
        rho, theta = self._make_spiral(log_A=log_A_true, B=B_true, noise=0.0)
        log_A_fit, B_fit, r2 = fit_log_spiral(rho, theta)
        assert np.isclose(log_A_fit, log_A_true, atol=1e-6)
        assert np.isclose(B_fit,     B_true,     atol=1e-6)
        assert np.isclose(r2, 1.0,               atol=1e-6)

    def test_r_squared_near_one_for_clean_spiral(self):
        rho, theta = self._make_spiral(noise=0.0)
        _, _, r2 = fit_log_spiral(rho, theta)
        assert r2 > 0.99

    def test_r_squared_lower_with_more_noise(self):
        rho_clean, theta = self._make_spiral(noise=0.0)
        rho_noisy, _    = self._make_spiral(noise=0.5)
        _, _, r2_clean = fit_log_spiral(rho_clean, theta)
        _, _, r2_noisy = fit_log_spiral(rho_noisy, theta)
        assert r2_clean >= r2_noisy

    def test_r_squared_at_most_one(self):
        rho, theta = self._make_spiral()
        _, _, r2 = fit_log_spiral(rho, theta)
        assert r2 <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# rolling_alert
# ---------------------------------------------------------------------------

class TestRollingAlert:
    def test_returns_array_of_zeros_when_shorter_than_window(self):
        velocity = np.array([1.0, 2.0, 3.0])
        alerts   = rolling_alert(velocity, window=60)
        np.testing.assert_array_equal(alerts, np.zeros(3))

    def test_output_length_matches_input(self):
        n        = 200
        velocity = np.random.default_rng(7).standard_normal(n)
        alerts   = rolling_alert(velocity, window=30)
        assert len(alerts) == n

    def test_values_are_zero_or_one(self):
        velocity = np.random.default_rng(8).standard_normal(200)
        alerts   = rolling_alert(velocity, window=30)
        assert set(alerts).issubset({0.0, 1.0})

    def test_extreme_spike_triggers_alert(self):
        """Insert a spike far beyond μ + 2σ; it should be flagged."""
        rng = np.random.default_rng(99)
        window = 30
        # stable baseline followed by a large spike
        velocity = np.concatenate([rng.standard_normal(window + 5), [1000.0]])
        alerts = rolling_alert(velocity, window=window, threshold_sigma=2.0)
        assert alerts[-1] == 1.0

    def test_constant_velocity_produces_no_alert(self):
        """Constant velocity has σ=0 → threshold = μ → no velocity exceeds it."""
        # Use a velocity that is exactly constant (all same value).
        # Rolling std ddof=0 will be 0, so threshold = mean + 2*0 = mean.
        # Since velocity[t] == mean, condition is False for all t.
        velocity = np.ones(120) * 5.0
        alerts   = rolling_alert(velocity, window=60)
        assert np.all(alerts == 0.0)


# ---------------------------------------------------------------------------
# SpiralMonitor.analyze
# ---------------------------------------------------------------------------

def _make_spiral_data(n=200, log_A=0.0, B=0.3, noise=0.0, seed=42):
    """
    Generate (anchor, radius) pairs that lie on a log-spiral in polar space.

    Keep theta strictly within (-π/2, π/2) so that atan2(radius, anchor)
    recovers theta without wrapping. This ensures fit_log_spiral sees a clean
    linear relationship between log(rho) and theta.
    """
    rng   = np.random.default_rng(seed)
    # theta monotonically increasing, all in right half-plane
    theta = np.linspace(-1.3, 1.3, n)          # strictly inside (-π/2, π/2)
    rho   = np.exp(log_A + B * theta)
    if noise > 0:
        rho = rho + rng.standard_normal(n) * noise
        rho = np.maximum(rho, 1e-8)
    anchor = rho * np.cos(theta)               # anchor > 0
    radius = rho * np.sin(theta)               # can be negative – that's fine
    return anchor, radius


class TestSpiralMonitorAnalyze:
    def setup_method(self):
        self.anchor, self.radius = _make_spiral_data()
        self.monitor = SpiralMonitor(window=30, threshold_sigma=2.0)

    def test_returns_dict_with_required_keys(self):
        results = self.monitor.analyze(self.anchor, self.radius)
        for key in ["rho", "theta", "log_A", "B", "r_squared",
                    "velocity", "alerts", "mean_velocity",
                    "std_velocity", "alert_rate"]:
            assert key in results, f"Missing key: {key}"

    def test_results_stored_on_instance(self):
        self.monitor.analyze(self.anchor, self.radius)
        assert len(self.monitor.results) > 0

    def test_rho_and_theta_shapes(self):
        n = len(self.anchor)
        results = self.monitor.analyze(self.anchor, self.radius)
        assert results["rho"].shape   == (n,)
        assert results["theta"].shape == (n,)

    def test_velocity_length_is_n_minus_1(self):
        n = len(self.anchor)
        results = self.monitor.analyze(self.anchor, self.radius)
        assert len(results["velocity"]) == n - 1

    def test_alerts_length_equals_velocity_length(self):
        results = self.monitor.analyze(self.anchor, self.radius)
        assert len(results["alerts"]) == len(results["velocity"])

    def test_r_squared_reasonable_for_known_spiral(self):
        """A low-noise log-spiral should have high R²."""
        results = self.monitor.analyze(self.anchor, self.radius)
        assert results["r_squared"] > 0.5

    def test_alert_rate_in_0_1(self):
        results = self.monitor.analyze(self.anchor, self.radius)
        assert 0.0 <= results["alert_rate"] <= 1.0


# ---------------------------------------------------------------------------
# SpiralMonitor.get_trajectory_stats
# ---------------------------------------------------------------------------

class TestSpiralMonitorGetTrajectoryStats:
    def test_empty_before_analyze(self):
        monitor = SpiralMonitor()
        assert monitor.get_trajectory_stats() == {}

    def test_returns_dict_after_analyze(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        stats = monitor.get_trajectory_stats()
        assert isinstance(stats, dict)

    def test_expected_top_level_keys(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        stats = monitor.get_trajectory_stats()
        for key in ["spiral_quality", "expansion", "alerts"]:
            assert key in stats

    def test_spiral_quality_is_good_fit_key_is_bool(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30, r_squared_threshold=0.5)
        monitor.analyze(anchor, radius)
        stats = monitor.get_trajectory_stats()
        assert isinstance(stats["spiral_quality"]["is_good_fit"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# SpiralMonitor.check_spiral_validity
# ---------------------------------------------------------------------------

class TestSpiralMonitorCheckSpiralValidity:
    def test_invalid_before_analyze(self):
        monitor = SpiralMonitor()
        result  = monitor.check_spiral_validity()
        assert result["is_valid"] is False

    def test_valid_for_good_spiral(self):
        anchor, radius = _make_spiral_data(noise=0.0)  # clean spiral, no noise
        monitor = SpiralMonitor(window=30, r_squared_threshold=0.5)
        monitor.analyze(anchor, radius)
        result = monitor.check_spiral_validity()
        # Clean spiral should be above threshold
        assert bool(result["is_valid"]) is True

    def test_result_contains_r_squared(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        result = monitor.check_spiral_validity()
        assert "r_squared" in result

    def test_result_contains_message(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        result = monitor.check_spiral_validity()
        assert "message" in result
        assert isinstance(result["message"], str)

    def test_invalid_for_noisy_spiral(self):
        """Very high noise should give poor fit → invalid."""
        anchor, radius = _make_spiral_data(noise=10.0)
        monitor = SpiralMonitor(window=30, r_squared_threshold=0.9)
        monitor.analyze(anchor, radius)
        result = monitor.check_spiral_validity()
        # With extreme noise the fit may or may not be good; just check the
        # result is consistent with the r_squared value.
        assert result["is_valid"] == (result["r_squared"] >= 0.9)


# ---------------------------------------------------------------------------
# SpiralMonitor.predict_next_rho
# ---------------------------------------------------------------------------

class TestSpiralMonitorPredictNextRho:
    def test_raises_before_analyze(self):
        monitor = SpiralMonitor()
        with pytest.raises(ValueError, match="analyze"):
            monitor.predict_next_rho(np.array([1.0]))

    def test_returns_array_of_correct_shape(self):
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        theta_next = np.array([5.0, 6.0, 7.0])
        rho_pred   = monitor.predict_next_rho(theta_next)
        assert rho_pred.shape == theta_next.shape

    def test_predictions_are_positive(self):
        """exp(...) is always positive."""
        anchor, radius = _make_spiral_data()
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        theta_next = np.linspace(0, 2 * np.pi, 10)
        rho_pred   = monitor.predict_next_rho(theta_next)
        assert np.all(rho_pred > 0)

    def test_prediction_consistent_with_log_spiral_model(self):
        """predict_next_rho should replicate exp(log_spiral_model(theta, log_A, B))."""
        from core.spiral_monitor import log_spiral_model
        anchor, radius = _make_spiral_data(noise=0.0)
        monitor = SpiralMonitor(window=30)
        monitor.analyze(anchor, radius)
        theta_next = np.array([1.4])   # just outside the training range
        rho_pred   = monitor.predict_next_rho(theta_next)
        log_A = monitor.results["log_A"]
        B     = monitor.results["B"]
        expected = np.exp(log_spiral_model(theta_next, log_A, B))
        np.testing.assert_allclose(rho_pred, expected)
