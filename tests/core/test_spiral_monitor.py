import numpy as np
import pytest
from core.spiral_monitor import log_spiral_model

def test_log_spiral_model_basic():
    """Test basic positive parameters."""
    theta = np.array([0.0, np.pi/2, np.pi])
    log_A = 1.0
    B = 0.5

    expected = log_A + B * theta
    result = log_spiral_model(theta, log_A, B)

    np.testing.assert_array_almost_equal(result, expected)

def test_log_spiral_model_zeros():
    """Test with zero parameters."""
    theta = np.array([1.0, 2.0, 3.0])

    # Test zero log_A
    expected_zero_A = 0.5 * theta
    result_zero_A = log_spiral_model(theta, 0.0, 0.5)
    np.testing.assert_array_almost_equal(result_zero_A, expected_zero_A)

    # Test zero B
    expected_zero_B = np.array([1.0, 1.0, 1.0])
    result_zero_B = log_spiral_model(theta, 1.0, 0.0)
    np.testing.assert_array_almost_equal(result_zero_B, expected_zero_B)

    # Test all zeros
    expected_all_zero = np.zeros_like(theta)
    result_all_zero = log_spiral_model(theta, 0.0, 0.0)
    np.testing.assert_array_almost_equal(result_all_zero, expected_all_zero)

def test_log_spiral_model_negative():
    """Test with negative parameters and negative theta."""
    theta = np.array([-np.pi, -np.pi/2, 0.0, np.pi])
    log_A = -2.0
    B = -0.1

    expected = log_A + B * theta
    result = log_spiral_model(theta, log_A, B)

    np.testing.assert_array_almost_equal(result, expected)

def test_log_spiral_model_empty():
    """Test with empty array."""
    theta = np.array([])
    log_A = 1.0
    B = 0.5

    expected = np.array([])
    result = log_spiral_model(theta, log_A, B)

    np.testing.assert_array_equal(result, expected)
