import numpy as np

from core.snail_mechanism import soft_pullback

def test_soft_pullback_basic():
    """Test basic mathematical correctness of the soft pullback mechanism."""
    point_pred = np.array([1.0, -1.0])
    anchor = np.array([0.5, 0.0])
    radius = np.array([0.5, 1.0])
    beta = 1.0

    # Calculation for index 0:
    # deviation = |1.0 - 0.5| / 0.5 = 1.0
    # alpha = exp(-1.0 * 1.0) = exp(-1) = 0.367879...
    # corrected = alpha * 1.0 + (1 - alpha) * 0.5 = 0.5 + 0.5 * alpha = 0.6839...

    # Calculation for index 1:
    # deviation = |-1.0 - 0.0| / 1.0 = 1.0
    # alpha = exp(-1.0 * 1.0) = exp(-1) = 0.367879...
    # corrected = alpha * (-1.0) + (1 - alpha) * 0.0 = -alpha = -0.367879...

    alpha = np.exp(-1.0)
    expected_0 = alpha * 1.0 + (1 - alpha) * 0.5
    expected_1 = alpha * -1.0 + (1 - alpha) * 0.0

    expected = np.array([expected_0, expected_1])

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, expected)

def test_soft_pullback_beta_zero():
    """Test with beta=0, which should return exactly point_pred (no pullback)."""
    point_pred = np.array([2.0, 3.0, 4.0])
    anchor = np.array([1.0, 1.0, 1.0])
    radius = np.array([0.5, 0.5, 0.5])
    beta = 0.0

    # alpha = exp(0) = 1.0
    # corrected = 1.0 * point_pred + 0.0 * anchor = point_pred

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, point_pred)

def test_soft_pullback_beta_inf():
    """Test with beta=np.inf, which should return exactly anchor (full pullback)."""
    point_pred = np.array([2.0, 3.0, 4.0])
    anchor = np.array([1.0, 1.0, 1.0])
    radius = np.array([0.5, 0.5, 0.5])
    beta = np.inf

    # alpha = exp(-inf) = 0.0
    # corrected = 0.0 * point_pred + 1.0 * anchor = anchor

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, anchor)

def test_soft_pullback_zero_radius():
    """Test with zero radius, should be handled by epsilon."""
    point_pred = np.array([2.0])
    anchor = np.array([1.0])
    radius = np.array([0.0])
    beta = 1.0

    # deviation = |2.0 - 1.0| / 1e-8 = 1e8
    # alpha = exp(-1e8) ≈ 0.0
    # corrected ≈ anchor

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, anchor, atol=1e-7)

def test_soft_pullback_negative_radius():
    """Test with negative radius, should also be handled by epsilon."""
    point_pred = np.array([2.0])
    anchor = np.array([1.0])
    radius = np.array([-0.5])
    beta = 1.0

    # epsilon is max(radius, 1e-8)
    # radius becomes 1e-8
    # deviation = |2.0 - 1.0| / 1e-8 = 1e8
    # alpha = exp(-1e8) ≈ 0.0
    # corrected ≈ anchor

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, anchor, atol=1e-7)

def test_soft_pullback_identical_predictions():
    """Test when point_pred and anchor are identical."""
    point_pred = np.array([1.5, 2.5])
    anchor = np.array([1.5, 2.5])
    radius = np.array([0.5, 0.5])
    beta = 1.0

    # deviation = 0 / 0.5 = 0
    # alpha = exp(0) = 1.0
    # corrected = 1.0 * 1.5 + 0.0 * 1.5 = 1.5

    result = soft_pullback(point_pred, anchor, radius, beta)

    np.testing.assert_allclose(result, point_pred)
    np.testing.assert_allclose(result, anchor)
