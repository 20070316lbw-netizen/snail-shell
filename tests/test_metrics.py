import numpy as np
import warnings
from evaluation.metrics import coverage_error

class TestCoverageError:
    def test_perfect_coverage(self):
        """Test when actual coverage exactly matches target coverage."""
        # 10 samples, 8 inside -> 0.8 coverage
        y_true = np.array([5, 5, 5, 5, 5, 5, 5, 5, 0, 10])
        lower = np.array([2]*10)
        upper = np.array([8]*10)

        # Expected CE = |0.8 - 0.8| = 0.0
        ce = coverage_error(y_true, lower, upper, target_coverage=0.8)
        assert np.isclose(ce, 0.0)

    def test_under_coverage(self):
        """Test when actual coverage is less than target coverage."""
        # 10 samples, 5 inside -> 0.5 coverage
        y_true = np.array([5, 5, 5, 5, 5, 0, 0, 0, 10, 10])
        lower = np.array([2]*10)
        upper = np.array([8]*10)

        # Expected CE = |0.5 - 0.8| = 0.3
        ce = coverage_error(y_true, lower, upper, target_coverage=0.8)
        assert np.isclose(ce, 0.3)

    def test_over_coverage(self):
        """Test when actual coverage is greater than target coverage."""
        # 10 samples, 10 inside -> 1.0 coverage
        y_true = np.array([5]*10)
        lower = np.array([2]*10)
        upper = np.array([8]*10)

        # Expected CE = |1.0 - 0.8| = 0.2
        ce = coverage_error(y_true, lower, upper, target_coverage=0.8)
        assert np.isclose(ce, 0.2)

    def test_different_target_coverage(self):
        """Test with a custom target coverage."""
        # 10 samples, 9 inside -> 0.9 coverage
        y_true = np.array([5]*9 + [10])
        lower = np.array([2]*10)
        upper = np.array([8]*10)

        # Target coverage = 0.9
        # Expected CE = |0.9 - 0.9| = 0.0
        ce = coverage_error(y_true, lower, upper, target_coverage=0.9)
        assert np.isclose(ce, 0.0)

        # Target coverage = 0.95
        # Expected CE = |0.9 - 0.95| = 0.05
        ce2 = coverage_error(y_true, lower, upper, target_coverage=0.95)
        assert np.isclose(ce2, 0.05)

    def test_boundary_inclusion(self):
        """Test that boundaries are inclusive (y >= lower and y <= upper)."""
        # Values exactly on boundary should be counted as inside
        y_true = np.array([2, 8])
        lower = np.array([2, 2])
        upper = np.array([8, 8])

        # 2 samples, 2 inside -> 1.0 coverage
        # Expected CE = |1.0 - 0.8| = 0.2
        ce = coverage_error(y_true, lower, upper, target_coverage=0.8)
        assert np.isclose(ce, 0.2)

        # Target coverage = 1.0
        # Expected CE = |1.0 - 1.0| = 0.0
        ce2 = coverage_error(y_true, lower, upper, target_coverage=1.0)
        assert np.isclose(ce2, 0.0)

    def test_zero_coverage(self):
        """Test when actual coverage is 0."""
        # 10 samples, 0 inside -> 0.0 coverage
        y_true = np.array([0, 10, 0, 10, 0, 10, 0, 10, 0, 10])
        lower = np.array([2]*10)
        upper = np.array([8]*10)

        # Expected CE = |0.0 - 0.8| = 0.8
        ce = coverage_error(y_true, lower, upper, target_coverage=0.8)
        assert np.isclose(ce, 0.8)

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        y_true = np.array([])
        lower = np.array([])
        upper = np.array([])

        # np.mean on an empty array raises RuntimeWarning and returns NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ce = coverage_error(y_true, lower, upper, target_coverage=0.8)

            # Should have generated a warning
            assert len(w) >= 1
            assert issubclass(w[-1].category, RuntimeWarning)

            # Result should be NaN
            assert np.isnan(ce)
