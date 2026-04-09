import numpy as np
import pytest
from core.spiral_monitor import log_spiral_model, calculate_expansion_velocity

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

def test_calculate_expansion_velocity_basic():
    """Test basic positive expansion velocity."""
    rho = np.array([1.0, 2.0, 4.0])
    theta = np.array([0.0, 0.5, 1.5])

    # Expected v_t = (rho_t - rho_{t-1}) / (theta_t - theta_{t-1} + epsilon)
    # v[0] = (2.0 - 1.0) / (0.5 - 0.0) = 1.0 / 0.5 = 2.0
    # v[1] = (4.0 - 2.0) / (1.5 - 0.5) = 2.0 / 1.0 = 2.0
    # The actual calculation in calculate_expansion_velocity returns velocity which has length n-1
    epsilon = 1e-8
    expected = np.array([
        (2.0 - 1.0) / (0.5 - 0.0 + epsilon),
        (4.0 - 2.0) / (1.5 - 0.5 + epsilon)
    ])
    result = calculate_expansion_velocity(rho, theta, epsilon=epsilon)
    np.testing.assert_array_almost_equal(result, expected)

def test_calculate_expansion_velocity_short():
    """Test arrays of length < 2."""
    # Length 0
    rho_empty = np.array([])
    theta_empty = np.array([])
    result_empty = calculate_expansion_velocity(rho_empty, theta_empty)
    np.testing.assert_array_equal(result_empty, np.array([]))

    # Length 1
    rho_single = np.array([1.0])
    theta_single = np.array([0.5])
    result_single = calculate_expansion_velocity(rho_single, theta_single)
    np.testing.assert_array_equal(result_single, np.array([]))

def test_calculate_expansion_velocity_zero_division():
    """Test zero division handled by epsilon."""
    rho = np.array([1.0, 2.0])
    theta = np.array([0.5, 0.5])  # Zero difference in theta

    epsilon = 1e-8
    expected = np.array([(2.0 - 1.0) / epsilon])
    result = calculate_expansion_velocity(rho, theta, epsilon=epsilon)
    np.testing.assert_array_almost_equal(result, expected)
