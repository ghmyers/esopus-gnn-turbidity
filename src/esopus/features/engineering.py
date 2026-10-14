"""Antecedent-discharge feature engineering.

Turbidity responds to current discharge and to the catchment's discharge
history. Cross-variograms of discharge vs. turbidity (``notebooks/
01_exploratory_data_analysis.ipynb``) show a decorrelation range of ~17 h
and secondary "dips" (renewed correlation) near 197, 2,648, 5,753 and
7,387 h. For every node, discharge-derived features are built at these lags:

========================  ==========================================================
``Q_cfs_range_{mean,max,std}``   rolling statistics over the primary range (17 h)
``Q_cfs_ant_{L}h_{mean,max,std}``  rolling statistics (window = 10 % of L) of Q shifted by L
``Q_cfs_diff_{a}h_to_{b}h``       Q(t-a) - Q(t-b) for consecutive lags
``Q_cfs_diff_now_to_{L}h``        Q(t) - Q(t-L)
``Q_cfs_cumsum_{L}h``             sum of Q over the last L hours
``Q_cfs_ratio_now_to_{L}h``       Q(t) / mean antecedent Q at lag L
========================  ==========================================================

Shifts and windows are counted in rows (hourly records), exactly as in the
original analysis.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

#: Lags (hours) from the cross-variogram analysis; the first is the primary range.
ANTECEDENT_LAGS = (17, 197, 2648, 5753, 7387)


def _window(lag_hours: float, dt_hours: float, fraction: float = 0.10) -> int:
    return max(1, int(round((lag_hours * fraction) / dt_hours)))


def antecedent_features(
    df: pd.DataFrame, lags: Sequence[float] = ANTECEDENT_LAGS, q_col: str = "Q (cfs)",
    fraction: float = 0.10, subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append antecedent-discharge features (see module docstring) to one node's table.

    ``subset`` keeps only the listed new features (all are computed).
    """
    dt = (df.index[1] - df.index[0]).total_seconds() / 3600
    q = df[q_col]
    name = q_col.replace(" ", "_").replace("(", "").replace(")", "")
    lag_list = [float(l) for l in lags]
    new = {}

    primary = max(1, int(round(lag_list[0] / dt)))
    new[f"{name}_range_mean"] = q.rolling(primary).mean()
    new[f"{name}_range_max"] = q.rolling(primary).max()
    new[f"{name}_range_std"] = q.rolling(primary).std()

    for lag in lag_list:
        L, steps = int(round(lag)), int(round(round(lag) / dt))
        shifted = q.shift(steps).rolling(_window(L, dt, fraction))
        new[f"{name}_ant_{L}h_mean"] = shifted.mean()
        new[f"{name}_ant_{L}h_max"] = shifted.max()
        new[f"{name}_ant_{L}h_std"] = shifted.std()

    for a, b in zip(lag_list[:-1], lag_list[1:]):
        a, b = int(round(a)), int(round(b))
        new[f"{name}_diff_{a}h_to_{b}h"] = q.shift(int(round(a / dt))) - q.shift(int(round(b / dt)))
    for lag in lag_list:
        L = int(round(lag))
        new[f"{name}_diff_now_to_{L}h"] = q - q.shift(int(round(L / dt)))
    for lag in lag_list:
        L = int(round(lag))
        new[f"{name}_cumsum_{L}h"] = q.rolling(int(round(L / dt))).sum()
    for lag in lag_list:
        L = int(round(lag))
        ant_mean = q.shift(int(round(L / dt))).rolling(_window(L, dt, fraction)).mean()
        new[f"{name}_ratio_now_to_{L}h"] = q / ant_mean.replace(0, np.nan)

    if subset is not None:
        new = {k: v for k, v in new.items() if k in subset}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def correlated_features_to_drop(
    G, threshold: float = 0.8, stride: int = 6, keep: Sequence[str] = ("Downward Long-Wave Radiation Flux", "Specific Humidity"),
) -> list:
    """Features to drop because |Pearson r| >= ``threshold`` with an earlier column.

    Correlations are computed on every ``stride``-th hour of all nodes'
    stacked feature tables (constant columns ignored). For each correlated
    pair the *later* column is dropped, except columns listed in ``keep``.
    """
    frames = [d["x"].iloc[::stride] for _, d in G.nodes(data=True) if d.get("x") is not None]
    num = pd.concat(frames, axis=0).select_dtypes(include=[np.number])
    num = num.loc[:, num.nunique() > 1]
    corr = num.corr(method="pearson").to_numpy()
    cols = list(num.columns)
    drop = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr[i, j]) >= threshold and cols[j] not in drop:
                drop.append(cols[j])
    return [c for c in drop if c not in keep]
