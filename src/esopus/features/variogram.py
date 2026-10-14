"""Empirical (cross-)variograms used to choose the antecedent-discharge lags.

For discharge Q and turbidity C at a gage, the empirical cross-variogram at
lag h is ``gamma(h) = mean[(Q(t+h) - Q(t)) (C(t+h) - C(t))] / 2``. Its first
rise to the sill gives the decorrelation range (~17 h); local minima beyond it
("dips") mark lags at which discharge and turbidity re-correlate. The dips
of the site-averaged cross-variogram over the main gages (9 sites, lags up to
one year) gave the lags used in ``esopus.features.engineering``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelmin, savgol_filter

#: Gages used for the lag analysis.
VARIOGRAM_NODES = (26, 27, 29, 30, 31, 32, 33, 34, 38)


def _lag_steps(n: int, dt_hours: float, n_lags: int, max_lag_hours) -> np.ndarray:
    max_steps = int(max_lag_hours / dt_hours) if max_lag_hours else n // 2
    return np.unique(np.round(np.linspace(1, max_steps, n_lags)).astype(int))


def variogram(series: pd.Series, n_lags: int = 269, max_lag_hours: float = 8760):
    """Empirical semivariogram ``(lag_hours, gamma)`` of a regular hourly series."""
    s = series.dropna()
    v = s.to_numpy(float)
    dt = (s.index[1] - s.index[0]).total_seconds() / 3600
    steps = _lag_steps(len(v), dt, n_lags, max_lag_hours)
    return steps * dt, np.array([np.mean((v[k:] - v[:-k]) ** 2) / 2 for k in steps])


def cross_variogram(x: pd.Series, y: pd.Series, n_lags: int = 269, max_lag_hours: float = 8760):
    """Empirical cross-variogram ``(lag_hours, gamma)`` on hours where both series are observed."""
    df = pd.concat([x, y], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    dt = (df.index[1] - df.index[0]).total_seconds() / 3600
    steps = _lag_steps(len(a), dt, n_lags, max_lag_hours)
    return steps * dt, np.array([np.mean((a[k:] - a[:-k]) * (b[k:] - b[:-k])) / 2 for k in steps])


def find_dips(lag_hours: np.ndarray, gamma: np.ndarray, smooth_window: int = 11, poly_order: int = 3,
              min_prominence: float = 0.03) -> pd.DataFrame:
    """Local minima of the Savitzky-Golay-smoothed variogram after it first reaches 80 % of the sill."""
    g = savgol_filter(gamma, window_length=smooth_window, polyorder=poly_order)
    sill = np.percentile(g, 90)
    risen = np.where(g >= 0.8 * sill)[0]
    start = risen[0] if len(risen) else 5
    (minima,) = argrelmin(g[start:], order=5)
    rows = []
    for i in minima + start:
        depth = (g[max(0, i - 10): i + 10].max() - g[i]) / sill
        if depth >= min_prominence:
            rows.append({"lag_hours": float(lag_hours[i]), "semivariance": float(gamma[i]), "rel_dip_depth": round(depth, 4)})
    return pd.DataFrame(rows)
