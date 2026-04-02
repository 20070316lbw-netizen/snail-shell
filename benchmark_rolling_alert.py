import time
import numpy as np
import pandas as pd
from core.spiral_monitor import rolling_alert

# Create dummy data
np.random.seed(42)
n_samples = 100000
velocity = np.random.randn(n_samples)
window = 60
threshold_sigma = 2.0

# Benchmark current implementation
start_time = time.time()
alerts_original = rolling_alert(
    velocity, window=window, threshold_sigma=threshold_sigma
)
original_time = time.time() - start_time

print(f"Original implementation time: {original_time:.4f} seconds")

# Proposed pandas implementation
start_time = time.time()
n = len(velocity)
alerts = np.zeros(n)
if n > window:
    vel_series = pd.Series(velocity)

    # We want to match: window_velocities = velocity[i - window : i]
    # This is the past `window` elements EXCLUDING current index `i`.
    # `rolling(window).mean()` calculates the mean of [i-window+1 : i+1]
    # Therefore, `.shift(1).rolling(window).mean()` calculates mean of [i-window : i]
    # Note: `rolling(window)` will produce NaNs for the first `window-1` elements of the shifted series,
    # which correspond to indices 0 to `window-1`.

    # Pandas std calculates sample std by default (ddof=1), numpy std calculates population std (ddof=0).
    # Since original code uses `np.std`, we must specify `ddof=0` in pandas std.
    mu = vel_series.shift(1).rolling(window=window).mean().to_numpy()
    sigma = vel_series.shift(1).rolling(window=window).std(ddof=0).to_numpy()

    # We only care about indices from `window` to `n-1`
    mask = (np.arange(n) >= window) & (velocity > mu + threshold_sigma * sigma)
    alerts[mask] = 1

pandas_time = time.time() - start_time
print(f"Pandas implementation time: {pandas_time:.4f} seconds")
print(f"Speedup: {original_time / pandas_time:.2f}x")

# Verify correctness
print("Difference:", np.sum(np.abs(alerts_original - alerts)))
