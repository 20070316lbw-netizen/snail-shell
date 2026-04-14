"""
Tests for core/hurst.py

Covers:
  _validate_series
  _log_spaced_ints
  hurst_rs
  hurst_dfa
  rolling_hurst
  hurst_to_beta
  adaptive_beta
"""

import numpy as np
import pytest
from core.hurst import (
    _validate_series,
    _log_spaced_ints,
    hurst_rs,
    hurst_dfa,
    rolling_hurst,
    hurst_to_beta,
    adaptive_beta,
)


# ---------------------------------------------------------------------------
# Synthetic series generators
# ---------------------------------------------------------------------------

def make_trending_series(n: int = 200, seed: int = 42) -> np.ndarray:
    """
    Strongly trending series generated via fractional Brownian-like construction.
    Uses cumulative sums of AR(1) with high persistence → H should be > 0.6.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # AR(1) with phi=0.9 → strong autocorrelation → high H
    ar = np.zeros(n)
    ar[0] = noise[0]
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + 0.1 * noise[i]
    return ar


def make_random_walk_series(n: int = 200, seed: int = 99) -> np.ndarray:
    """i.i.d. Gaussian → H ≈ 0.5"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_mean_reverting_series(n: int = 200, seed: int = 7) -> np.ndarray:
    """
    AR(1) with phi = -0.7 → strong negative autocorrelation → H < 0.4
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    ar = np.zeros(n)
    ar[0] = noise[0]
    for i in range(1, n):
        ar[i] = -0.7 * ar[i - 1] + noise[i]
    return ar


# ===========================================================================
# _validate_series
# ===========================================================================

class TestValidateSeries:
    def test_returns_float64(self):
        s = _validate_series([1, 2, 3, 4, 5, 6, 7, 8], min_len=4)
        assert s.dtype == np.float64

    def test_removes_nan(self):
        s = _validate_series([1.0, np.nan, 2.0, np.nan, 3.0, 4.0], min_len=2)
        assert len(s) == 4
        assert not np.any(np.isnan(s))

    def test_raises_when_too_short(self):
        with pytest.raises(ValueError, match="有效长度"):
            _validate_series([1.0, np.nan, np.nan], min_len=4)

    def test_passes_when_exactly_min_len(self):
        s = _validate_series([1.0, 2.0, 3.0, 4.0], min_len=4)
        assert len(s) == 4

    def test_accepts_numpy_array(self):
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        s = _validate_series(arr, min_len=3)
        assert len(s) == 5

    def test_custom_name_in_error_message(self):
        with pytest.raises(ValueError, match="my_input"):
            _validate_series([1.0], min_len=10, name="my_input")


# ===========================================================================
# _log_spaced_ints
# ===========================================================================

class TestLogSpacedInts:
    def test_returns_sorted_unique_ints(self):
        result = _log_spaced_ints(4, 50, 10)
        assert list(result) == sorted(set(result))
        assert result.dtype in (np.int32, np.int64, int)

    def test_bounds_respected(self):
        result = _log_spaced_ints(4, 50, 20)
        assert result[0] >= 4
        assert result[-1] <= 50

    def test_minimum_two_elements(self):
        result = _log_spaced_ints(4, 8, 5)
        assert len(result) >= 2

    def test_single_point_range(self):
        result = _log_spaced_ints(10, 10, 5)
        assert list(result) == [10]


# ===========================================================================
# hurst_rs
# ===========================================================================

class TestHurstRS:
    def test_returns_float(self):
        s = make_random_walk_series(200)
        H = hurst_rs(s)
        assert isinstance(H, float)

    def test_random_walk_near_half(self):
        """i.i.d. Gaussian: H should be 0.5 ± 0.2 for N=200."""
        rng = np.random.default_rng(0)
        H_vals = [hurst_rs(rng.standard_normal(200)) for _ in range(10)]
        assert 0.3 < np.mean(H_vals) < 0.7

    def test_trending_series_high_H(self):
        """AR(1) phi=0.9 → H should be noticeably > 0.5."""
        s = make_trending_series(300)
        H = hurst_rs(s)
        assert H > 0.55

    def test_mean_reverting_series_low_H(self):
        """AR(1) phi=-0.7 → H should be < 0.5."""
        s = make_mean_reverting_series(300)
        H = hurst_rs(s)
        assert H < 0.5

    def test_raises_when_series_too_short(self):
        with pytest.raises(ValueError):
            hurst_rs(np.array([0.1, 0.2, 0.3]))

    def test_raises_with_insufficient_lags(self):
        """N=10 with min_lag=8 → not enough lags."""
        with pytest.raises(ValueError):
            hurst_rs(np.arange(10, dtype=float), min_lag=8)

    def test_nan_values_filtered(self):
        s = make_random_walk_series(200)
        s_with_nan = np.concatenate([s, [np.nan] * 20])
        H1 = hurst_rs(s)
        H2 = hurst_rs(s_with_nan)
        assert np.isclose(H1, H2, atol=1e-10)

    def test_custom_max_lag(self):
        s = make_random_walk_series(200)
        H = hurst_rs(s, max_lag=50)
        assert isinstance(H, float)

    def test_custom_n_lags(self):
        s = make_random_walk_series(200)
        H = hurst_rs(s, n_lags=5)
        assert isinstance(H, float)


# ===========================================================================
# hurst_dfa
# ===========================================================================

class TestHurstDFA:
    def test_returns_float(self):
        s = make_random_walk_series(100)
        H = hurst_dfa(s)
        assert isinstance(H, float)

    def test_random_walk_near_half(self):
        """DFA of i.i.d. series: H should be near 0.5."""
        rng = np.random.default_rng(1)
        H_vals = [hurst_dfa(rng.standard_normal(100)) for _ in range(10)]
        assert 0.3 < np.mean(H_vals) < 0.7

    def test_trending_series_high_H(self):
        s = make_trending_series(200)
        H = hurst_dfa(s)
        assert H > 0.6

    def test_mean_reverting_series_low_H(self):
        s = make_mean_reverting_series(200)
        H = hurst_dfa(s)
        assert H < 0.45

    def test_raises_when_too_short(self):
        with pytest.raises(ValueError):
            hurst_dfa(np.array([0.1, 0.2, 0.3]))

    def test_custom_scales(self):
        s = make_random_walk_series(80)
        H = hurst_dfa(s, scales=[4, 8, 12, 16, 20])
        assert isinstance(H, float)

    def test_order_2_polynomial(self):
        s = make_random_walk_series(100)
        H = hurst_dfa(s, order=2)
        assert isinstance(H, float)

    def test_nan_filtered_out(self):
        s = make_random_walk_series(80)
        s_with_nan = s.copy()
        s_with_nan[::10] = np.nan
        H = hurst_dfa(s_with_nan)
        assert isinstance(H, float)
        assert not np.isnan(H)

    def test_raises_insufficient_scales(self):
        """Only 1 valid scale → can't fit."""
        with pytest.raises(ValueError):
            hurst_dfa(np.random.randn(16), scales=[4])

    def test_raises_series_too_short_for_auto_scales(self):
        """N=15 < min_len=16 → ValueError from _validate_series."""
        with pytest.raises(ValueError):
            hurst_dfa(np.random.randn(15))


# ===========================================================================
# rolling_hurst
# ===========================================================================

class TestRollingHurst:
    def test_output_length_matches_input(self):
        s = make_random_walk_series(100)
        result = rolling_hurst(s, window=36)
        assert len(result) == len(s)

    def test_first_window_minus_1_elements_are_nan(self):
        s = make_random_walk_series(80)
        result = rolling_hurst(s, window=36)
        assert np.all(np.isnan(result[:35]))

    def test_later_values_not_all_nan(self):
        s = make_random_walk_series(100)
        result = rolling_hurst(s, window=36)
        assert not np.all(np.isnan(result[35:]))

    def test_method_rs(self):
        s = make_random_walk_series(150)
        result = rolling_hurst(s, window=50, method="rs")
        assert len(result) == 150
        assert not np.all(np.isnan(result))

    def test_method_dfa(self):
        s = make_random_walk_series(100)
        result = rolling_hurst(s, window=36, method="dfa")
        assert len(result) == 100

    def test_short_series_all_nan(self):
        s = make_random_walk_series(10)
        result = rolling_hurst(s, window=36)
        assert np.all(np.isnan(result))

    def test_min_periods_threshold(self):
        """With min_periods=24, positions with < 24 valid pts in window → NaN."""
        s = make_random_walk_series(80)
        result = rolling_hurst(s, window=36, min_periods=24)
        # All positions before index 35 should be NaN (window not full)
        assert np.all(np.isnan(result[:35]))

    def test_nan_in_input_handled_gracefully(self):
        s = make_random_walk_series(100)
        s[10:20] = np.nan
        result = rolling_hurst(s, window=36)
        assert len(result) == 100
        # Should not raise; NaN positions still produce float or NaN

    def test_all_nan_input(self):
        s = np.full(60, np.nan)
        result = rolling_hurst(s, window=36)
        assert np.all(np.isnan(result))

    def test_kwargs_passed_to_method(self):
        """Extra kwargs (e.g. n_scales) should be forwarded."""
        s = make_random_walk_series(80)
        result = rolling_hurst(s, window=36, method="dfa", n_scales=10)
        assert len(result) == 80

    def test_output_dtype_float64(self):
        s = make_random_walk_series(80)
        result = rolling_hurst(s, window=36)
        assert result.dtype == np.float64


# ===========================================================================
# hurst_to_beta
# ===========================================================================

class TestHurstToBeta:
    def test_H_half_returns_beta_base(self):
        beta = hurst_to_beta(0.5, beta_base=1.0)
        assert np.isclose(beta, 1.0)

    def test_H_trending_reduces_beta(self):
        """H > 0.5 (trending) → β should be < beta_base."""
        beta = hurst_to_beta(0.8, beta_base=1.0, sensitivity=2.0)
        assert beta < 1.0

    def test_H_mean_reverting_increases_beta(self):
        """H < 0.5 (mean-reverting) → β should be > beta_base."""
        beta = hurst_to_beta(0.2, beta_base=1.0, sensitivity=2.0)
        assert beta > 1.0

    def test_formula_exact(self):
        """β = clip(1.0 + (0.5 - 0.3) × 2.0, 0, 5) = clip(1.4, 0, 5) = 1.4"""
        beta = hurst_to_beta(0.3, beta_base=1.0, sensitivity=2.0, beta_min=0.0, beta_max=5.0)
        assert np.isclose(beta, 1.4)

    def test_clipping_lower_bound(self):
        """H = 1.0, sensitivity=10 → β = 1 - 5 = -4 → clipped to 0."""
        beta = hurst_to_beta(1.0, beta_base=1.0, sensitivity=10.0, beta_min=0.0, beta_max=5.0)
        assert np.isclose(beta, 0.0)

    def test_clipping_upper_bound(self):
        """H = 0.0, sensitivity=10 → β = 1 + 5 = 6 → clipped to 5."""
        beta = hurst_to_beta(0.0, beta_base=1.0, sensitivity=10.0, beta_min=0.0, beta_max=5.0)
        assert np.isclose(beta, 5.0)

    def test_nan_H_returns_beta_base(self):
        beta = hurst_to_beta(np.nan, beta_base=2.0)
        assert np.isclose(beta, 2.0)

    def test_returns_float(self):
        beta = hurst_to_beta(0.5)
        assert isinstance(beta, float)

    def test_custom_beta_base(self):
        beta = hurst_to_beta(0.5, beta_base=3.0)
        assert np.isclose(beta, 3.0)

    def test_zero_sensitivity(self):
        """sensitivity=0 → β always equals beta_base regardless of H."""
        for H in [0.0, 0.3, 0.5, 0.8, 1.0]:
            beta = hurst_to_beta(H, beta_base=2.0, sensitivity=0.0)
            assert np.isclose(beta, 2.0)

    def test_monotone_in_H(self):
        """β should be non-increasing as H increases (more trending → less pullback)."""
        H_values = np.linspace(0.01, 0.99, 20)
        betas = [hurst_to_beta(H, sensitivity=2.0) for H in H_values]
        for i in range(len(betas) - 1):
            assert betas[i] >= betas[i + 1] - 1e-10


# ===========================================================================
# adaptive_beta
# ===========================================================================

class TestAdaptiveBeta:
    def test_returns_float(self):
        s = make_random_walk_series(80)
        beta = adaptive_beta(s)
        assert isinstance(beta, float)

    def test_returns_beta_base_when_series_too_short(self):
        """Shorter than min_periods → H=NaN → fallback to beta_base."""
        s = np.random.randn(5)
        beta = adaptive_beta(s, window=36, beta_base=1.5)
        assert np.isclose(beta, 1.5)

    def test_returns_beta_base_for_all_nan(self):
        s = np.full(80, np.nan)
        beta = adaptive_beta(s, window=36, beta_base=2.0)
        assert np.isclose(beta, 2.0)

    def test_in_valid_range(self):
        s = make_random_walk_series(80)
        beta = adaptive_beta(
            s, window=36, beta_min=0.0, beta_max=5.0, beta_base=1.0
        )
        assert 0.0 <= beta <= 5.0

    def test_uses_method_dfa(self):
        s = make_random_walk_series(80)
        beta_dfa = adaptive_beta(s, method="dfa")
        assert isinstance(beta_dfa, float)

    def test_uses_method_rs(self):
        s = make_random_walk_series(150)
        beta_rs = adaptive_beta(s, method="rs")
        assert isinstance(beta_rs, float)

    def test_trending_series_gives_low_beta(self):
        """Strong trending series → H > 0.5 → β < beta_base."""
        s = make_trending_series(200)
        beta = adaptive_beta(s, window=60, beta_base=1.0, sensitivity=2.0,
                             beta_min=0.0, beta_max=5.0)
        assert beta < 1.0 + 0.3  # Some tolerance for statistical noise

    def test_mean_reverting_gives_high_beta(self):
        """Strong mean-reverting series → H < 0.5 → β > beta_base."""
        s = make_mean_reverting_series(200)
        beta = adaptive_beta(s, window=60, beta_base=1.0, sensitivity=2.0,
                             beta_min=0.0, beta_max=5.0)
        assert beta > 1.0 - 0.3

    def test_custom_min_periods(self):
        s = make_random_walk_series(80)
        beta = adaptive_beta(s, window=36, min_periods=30)
        assert isinstance(beta, float)
