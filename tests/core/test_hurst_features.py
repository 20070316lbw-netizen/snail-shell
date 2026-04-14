"""
Tests for core/hurst_features.py

Covers:
  _segment_indices
  compute_hurst_features
  hurst_regime_label
  hurst_regime_labels
  summarize_hurst_features
"""

import numpy as np
import pandas as pd
import pytest
from core.hurst_features import (
    _segment_indices,
    compute_hurst_features,
    hurst_regime_label,
    hurst_regime_labels,
    summarize_hurst_features,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_monthly_dates(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Generate n monthly dates (approx. 30-day spacing)."""
    return pd.date_range(start=start, periods=n, freq="MS")


def make_features_df(
    n_tickers: int = 3,
    n_months: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic features_cn-style DataFrame.
    Columns: ticker, date, mom_20d
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tickers):
        ticker = f"T{i:04d}"
        dates = make_monthly_dates(n_months)
        returns = rng.standard_normal(n_months) * 0.05
        for date, ret in zip(dates, returns):
            rows.append({"ticker": ticker, "date": date, "mom_20d": ret})
    df = pd.DataFrame(rows)
    return df


def make_gapped_features_df(seed: int = 0) -> pd.DataFrame:
    """
    Single ticker with a 3-year gap (2018-01 to 2021-01).
    """
    rng = np.random.default_rng(seed)
    dates_before = pd.date_range("2015-01-01", "2017-12-01", freq="MS")
    dates_after  = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
    all_dates = dates_before.append(dates_after)
    n = len(all_dates)
    returns = rng.standard_normal(n) * 0.05
    df = pd.DataFrame({
        "ticker": "GAPPED",
        "date": all_dates,
        "mom_20d": returns,
    })
    return df


# ===========================================================================
# _segment_indices
# ===========================================================================

class TestSegmentIndices:
    def test_no_gap_returns_single_segment(self):
        dates = pd.Series(make_monthly_dates(24))
        segs = _segment_indices(dates, max_gap_days=45)
        assert len(segs) == 1
        assert segs[0] == (0, 24)

    def test_gap_splits_into_two_segments(self):
        dates_before = pd.date_range("2015-01-01", "2017-12-01", freq="MS")
        dates_after  = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
        dates = pd.Series(dates_before.append(dates_after))
        segs = _segment_indices(dates, max_gap_days=45)
        assert len(segs) == 2
        assert segs[0][0] == 0
        assert segs[1][1] == len(dates)

    def test_segments_are_contiguous_and_non_overlapping(self):
        dates_before = pd.date_range("2015-01-01", "2017-12-01", freq="MS")
        dates_after  = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
        dates = pd.Series(dates_before.append(dates_after))
        segs = _segment_indices(dates, max_gap_days=45)
        prev_end = 0
        for start, end in segs:
            assert start == prev_end
            assert end > start
            prev_end = end
        assert prev_end == len(dates)

    def test_empty_series_returns_empty_list(self):
        segs = _segment_indices(pd.Series([], dtype="datetime64[ns]"), max_gap_days=45)
        assert segs == []

    def test_single_element_returns_one_segment(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01"]))
        segs = _segment_indices(dates, max_gap_days=45)
        assert len(segs) == 1
        assert segs[0] == (0, 1)

    def test_multiple_gaps(self):
        """Three separate segments."""
        d1 = pd.date_range("2015-01-01", periods=12, freq="MS")
        d2 = pd.date_range("2018-01-01", periods=12, freq="MS")
        d3 = pd.date_range("2021-01-01", periods=12, freq="MS")
        dates = pd.Series(d1.append(d2).append(d3))
        segs = _segment_indices(dates, max_gap_days=45)
        assert len(segs) == 3

    def test_segment_sizes_correct(self):
        d1 = pd.date_range("2015-01-01", periods=12, freq="MS")
        d2 = pd.date_range("2018-01-01", periods=24, freq="MS")
        dates = pd.Series(d1.append(d2))
        segs = _segment_indices(dates, max_gap_days=45)
        assert segs[0] == (0, 12)
        assert segs[1] == (12, 36)

    def test_large_max_gap_treats_everything_as_one_segment(self):
        d1 = pd.date_range("2015-01-01", periods=12, freq="MS")
        d2 = pd.date_range("2018-01-01", periods=12, freq="MS")
        dates = pd.Series(d1.append(d2))
        segs = _segment_indices(dates, max_gap_days=9999)
        assert len(segs) == 1

    def test_small_max_gap_splits_monthly_series(self):
        """max_gap_days=20 splits every month (30-day spacing > 20)."""
        dates = pd.Series(make_monthly_dates(5))
        segs = _segment_indices(dates, max_gap_days=20)
        # Every consecutive pair has ~30 days → all are gaps
        assert len(segs) == 5


# ===========================================================================
# compute_hurst_features
# ===========================================================================

class TestComputeHurstFeatures:
    def setup_method(self):
        self.df = make_features_df(n_tickers=3, n_months=60)

    def test_output_shape_same_as_input(self):
        result = compute_hurst_features(self.df, windows=[24])
        assert result.shape[0] == self.df.shape[0]

    def test_output_has_new_hurst_columns(self):
        result = compute_hurst_features(self.df, windows=[24, 36])
        assert "hurst_24m" in result.columns
        assert "hurst_36m" in result.columns

    def test_original_columns_preserved(self):
        result = compute_hurst_features(self.df, windows=[24])
        for col in self.df.columns:
            assert col in result.columns

    def test_original_data_unchanged(self):
        result = compute_hurst_features(self.df, windows=[24])
        pd.testing.assert_frame_equal(
            result[self.df.columns], self.df, check_like=False
        )

    def test_hurst_column_is_float64(self):
        result = compute_hurst_features(self.df, windows=[24])
        assert result["hurst_24m"].dtype == np.float64

    def test_early_rows_are_nan(self):
        """First 23 rows per ticker (window=24) should be NaN."""
        result = compute_hurst_features(self.df, windows=[24])
        for ticker in self.df["ticker"].unique():
            ticker_rows = result[result["ticker"] == ticker].sort_values("date")
            assert ticker_rows["hurst_24m"].iloc[:23].isna().all()

    def test_later_rows_have_valid_hurst(self):
        result = compute_hurst_features(self.df, windows=[24])
        for ticker in self.df["ticker"].unique():
            ticker_rows = result[result["ticker"] == ticker].sort_values("date")
            valid = ticker_rows["hurst_24m"].dropna()
            assert len(valid) > 0

    def test_multiple_tickers_independent(self):
        """Each ticker's Hurst should be computed independently."""
        rng = np.random.default_rng(5)
        # Ticker A: trending (AR(1) phi=0.8)
        n = 60
        a = np.zeros(n)
        noise = rng.standard_normal(n)
        a[0] = noise[0]
        for i in range(1, n):
            a[i] = 0.8 * a[i - 1] + 0.2 * noise[i]

        # Ticker B: mean-reverting (AR(1) phi=-0.7)
        b = np.zeros(n)
        noise2 = rng.standard_normal(n)
        b[0] = noise2[0]
        for i in range(1, n):
            b[i] = -0.7 * b[i - 1] + noise2[i]

        dates = make_monthly_dates(n)
        df = pd.DataFrame([
            {"ticker": "A", "date": d, "mom_20d": v} for d, v in zip(dates, a)
        ] + [
            {"ticker": "B", "date": d, "mom_20d": v} for d, v in zip(dates, b)
        ])
        result = compute_hurst_features(df, windows=[36])

        H_A = result[result["ticker"] == "A"]["hurst_36m"].dropna().mean()
        H_B = result[result["ticker"] == "B"]["hurst_36m"].dropna().mean()
        assert H_A > H_B, f"Expected trending H_A={H_A:.3f} > mean-reverting H_B={H_B:.3f}"

    def test_gap_handling_produces_nan_after_gap(self):
        """
        After a data gap, the first window-1 observations in the new segment
        should be NaN (not enough history).
        """
        df = make_gapped_features_df()
        result = compute_hurst_features(df, windows=[24])
        sorted_result = result.sort_values("date")

        # Find position of gap: last date before gap is 2017-12-01
        before_gap = sorted_result[sorted_result["date"] < pd.Timestamp("2020-01-01")]
        after_gap  = sorted_result[sorted_result["date"] >= pd.Timestamp("2021-01-01")]

        # First rows after gap should be NaN (new segment, insufficient window)
        assert after_gap["hurst_24m"].iloc[:23].isna().all()

    def test_gap_handling_segment_before_gap_has_values(self):
        """Rows before the gap (sufficient length) should have valid H."""
        df = make_gapped_features_df()
        result = compute_hurst_features(df, windows=[24])
        before_gap = result[result["date"] < pd.Timestamp("2020-01-01")].sort_values("date")
        valid = before_gap["hurst_24m"].dropna()
        assert len(valid) > 0

    def test_custom_returns_col(self):
        df = self.df.copy()
        df["custom_ret"] = df["mom_20d"]
        result = compute_hurst_features(df, windows=[24], returns_col="custom_ret")
        assert "hurst_24m" in result.columns

    def test_custom_ticker_and_date_cols(self):
        df = self.df.rename(columns={"ticker": "stock_id", "date": "period"})
        result = compute_hurst_features(
            df, windows=[24], ticker_col="stock_id", date_col="period"
        )
        assert "hurst_24m" in result.columns

    def test_default_windows_used_when_none(self):
        """Default windows=[24, 36] should produce both columns."""
        result = compute_hurst_features(self.df)
        assert "hurst_24m" in result.columns
        assert "hurst_36m" in result.columns

    def test_method_rs_accepted(self):
        result = compute_hurst_features(self.df, windows=[36], method="rs")
        assert "hurst_36m" in result.columns

    def test_row_order_preserved(self):
        """Output row order should match input row order."""
        result = compute_hurst_features(self.df, windows=[24])
        pd.testing.assert_index_equal(result.index, self.df.index)

    def test_short_ticker_all_nan(self):
        """A ticker with only 10 rows and window=36 → all NaN."""
        rng = np.random.default_rng(0)
        n = 10
        df_short = pd.DataFrame({
            "ticker": "SHORT",
            "date": make_monthly_dates(n),
            "mom_20d": rng.standard_normal(n),
        })
        result = compute_hurst_features(df_short, windows=[36])
        assert result["hurst_36m"].isna().all()


# ===========================================================================
# hurst_regime_label
# ===========================================================================

class TestHurstRegimeLabel:
    def test_trending_returns_positive_one(self):
        assert hurst_regime_label(0.7) == 1

    def test_mean_reverting_returns_negative_one(self):
        assert hurst_regime_label(0.2) == -1

    def test_random_walk_returns_zero(self):
        assert hurst_regime_label(0.5) == 0

    def test_exact_lower_bound_is_neutral(self):
        assert hurst_regime_label(0.45) == 0

    def test_exact_upper_bound_is_neutral(self):
        assert hurst_regime_label(0.55) == 0

    def test_just_below_lower_is_mean_reverting(self):
        assert hurst_regime_label(0.44999) == -1

    def test_just_above_upper_is_trending(self):
        assert hurst_regime_label(0.55001) == 1

    def test_nan_returns_zero(self):
        assert hurst_regime_label(np.nan) == 0

    def test_custom_bounds(self):
        assert hurst_regime_label(0.4, lower=0.3, upper=0.6) == 0
        assert hurst_regime_label(0.25, lower=0.3, upper=0.6) == -1
        assert hurst_regime_label(0.7, lower=0.3, upper=0.6) == 1

    def test_returns_int(self):
        label = hurst_regime_label(0.6)
        assert isinstance(label, int)

    def test_H_zero_is_mean_reverting(self):
        assert hurst_regime_label(0.0) == -1

    def test_H_one_is_trending(self):
        assert hurst_regime_label(1.0) == 1


# ===========================================================================
# hurst_regime_labels (vectorized)
# ===========================================================================

class TestHurstRegimeLabels:
    def test_basic_classification(self):
        H = np.array([0.2, 0.5, 0.8])
        labels = hurst_regime_labels(H)
        assert list(labels) == [-1, 0, 1]

    def test_nan_maps_to_zero(self):
        H = np.array([np.nan, 0.5, np.nan])
        labels = hurst_regime_labels(H)
        assert list(labels) == [0, 0, 0]

    def test_output_dtype_int8(self):
        H = np.array([0.2, 0.5, 0.8])
        labels = hurst_regime_labels(H)
        assert labels.dtype == np.int8

    def test_output_length_matches_input(self):
        H = np.linspace(0.0, 1.0, 50)
        labels = hurst_regime_labels(H)
        assert len(labels) == 50

    def test_consistent_with_scalar_version(self):
        H_vals = np.linspace(0.0, 1.0, 20)
        vec_labels = hurst_regime_labels(H_vals)
        for i, h in enumerate(H_vals):
            assert int(vec_labels[i]) == hurst_regime_label(float(h))

    def test_custom_bounds_applied(self):
        H = np.array([0.4, 0.5, 0.7])
        labels = hurst_regime_labels(H, lower=0.45, upper=0.65)
        assert list(labels) == [-1, 0, 1]


# ===========================================================================
# summarize_hurst_features
# ===========================================================================

class TestSummarizeHurstFeatures:
    def setup_method(self):
        self.df = make_features_df(n_tickers=2, n_months=60)
        self.result = compute_hurst_features(self.df, windows=[24, 36])

    def test_returns_dict_with_hurst_columns(self):
        summary = summarize_hurst_features(self.result)
        assert "hurst_24m" in summary
        assert "hurst_36m" in summary

    def test_summary_has_required_keys(self):
        summary = summarize_hurst_features(self.result)
        required_keys = {"count", "nan_rate", "mean", "std", "min", "p25",
                         "p50", "p75", "max"}
        for col, stats in summary.items():
            assert required_keys.issubset(stats.keys()), f"{col} missing keys"

    def test_count_matches_non_nan(self):
        summary = summarize_hurst_features(self.result)
        for col, stats in summary.items():
            expected = int(self.result[col].notna().sum())
            assert stats["count"] == expected

    def test_nan_rate_in_0_1(self):
        summary = summarize_hurst_features(self.result)
        for col, stats in summary.items():
            assert 0.0 <= stats["nan_rate"] <= 1.0

    def test_custom_windows_filter(self):
        summary = summarize_hurst_features(self.result, windows=[24])
        assert "hurst_24m" in summary
        assert "hurst_36m" not in summary

    def test_ignores_missing_window(self):
        """Windows not present in df are silently skipped."""
        summary = summarize_hurst_features(self.result, windows=[99])
        assert len(summary) == 0

    def test_empty_df_handles_gracefully(self):
        empty = self.result.iloc[0:0]
        # should not raise
        summary = summarize_hurst_features(empty, windows=[24])
        # count should be 0 or key absent
        for col, stats in summary.items():
            assert stats["count"] == 0

    def test_mean_within_plausible_range(self):
        """Hurst values should roughly be between 0 and 1."""
        summary = summarize_hurst_features(self.result)
        for col, stats in summary.items():
            if stats["count"] > 0:
                assert 0.0 <= stats["mean"] <= 1.0
