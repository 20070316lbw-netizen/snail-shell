import numpy as np
from core.quantile_head import pinball_loss

def test_pinball_loss_basic():
    """Test basic pinball loss calculation for arrays"""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 2.0, 4.0])

    # Residuals: [1.0, 0.0, -1.0]

    # alpha = 0.5
    # For res > 0: max(0.5 * 1.0, -0.5 * 1.0) = 0.5
    # For res == 0: 0
    # For res < 0: max(0.5 * -1.0, -0.5 * -1.0) = 0.5
    # Mean: (0.5 + 0.0 + 0.5) / 3 = 1.0 / 3 = 0.333...
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.5), 1.0 / 3.0)

    # alpha = 0.1
    # For res > 0 (1.0): max(0.1 * 1.0, -0.9 * 1.0) = 0.1
    # For res == 0: 0
    # For res < 0 (-1.0): max(0.1 * -1.0, -0.9 * -1.0) = 0.9
    # Mean: (0.1 + 0.0 + 0.9) / 3 = 1.0 / 3 = 0.333...
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.1), 1.0 / 3.0)

    # alpha = 0.9
    # For res > 0 (1.0): max(0.9 * 1.0, -0.1 * 1.0) = 0.9
    # For res == 0: 0
    # For res < 0 (-1.0): max(0.9 * -1.0, -0.1 * -1.0) = 0.1
    # Mean: (0.9 + 0.0 + 0.1) / 3 = 1.0 / 3 = 0.333...
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.9), 1.0 / 3.0)


def test_pinball_loss_symmetric_alpha_0_5():
    """Test that alpha=0.5 is exactly half the Mean Absolute Error"""
    y_true = np.array([1.5, -2.0, 3.1, 4.2])
    y_pred = np.array([1.0, -1.0, 3.1, 5.0])

    mae = np.mean(np.abs(y_true - y_pred))
    pl = pinball_loss(y_true, y_pred, alpha=0.5)

    assert np.isclose(pl, mae / 2.0)


def test_pinball_loss_zero_residual():
    """Test when predictions exactly match true values"""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.1), 0.0)
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.5), 0.0)
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.9), 0.0)


def test_pinball_loss_alpha_boundaries():
    """Test edge cases with alpha = 0.0 and alpha = 1.0"""
    y_true = np.array([1.0, 0.0, -1.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    # Residuals: [1.0, 0.0, -1.0]

    # alpha = 1.0
    # L = max(1*(y-y^), 0*(y-y^)) = max(y-y^, 0)
    # [max(1,0), max(0,0), max(-1,0)] = [1, 0, 0] -> mean = 1/3
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=1.0), 1.0 / 3.0)

    # alpha = 0.0
    # L = max(0*(y-y^), -1*(y-y^)) = max(0, y^-y)
    # [max(0, -1), max(0, 0), max(0, 1)] = [0, 0, 1] -> mean = 1/3
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.0), 1.0 / 3.0)


def test_pinball_loss_multidimensional_arrays():
    """Test with 2D arrays to ensure np.mean works properly across all dimensions"""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[0.0, 2.0], [4.0, 4.0]])

    # Residuals: [[1.0, 0.0], [-1.0, 0.0]]
    # alpha = 0.5
    # Max values: [[0.5, 0.0], [0.5, 0.0]]
    # Mean: 1.0 / 4 = 0.25
    assert np.isclose(pinball_loss(y_true, y_pred, alpha=0.5), 0.25)
