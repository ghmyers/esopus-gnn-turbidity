"""Goodness-of-fit metrics for turbidity predictions.

Per node, observations and predictions (FNU) are paired, NaN pairs dropped,
negative predictions clipped to 0, and the following computed:

* ``r2``    coefficient of determination (scikit-learn; equals NSE here)
* ``mae``   mean absolute error (FNU)
* ``rmse``  root mean squared error (FNU)
* ``mape``  mean absolute percentage error (fraction; unstable near 0 FNU)
* ``kge``   Kling-Gupta efficiency, 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
            (Gupta et al., 2009); KGE > -0.41 beats the mean-flow benchmark
            (Knoben et al., 2019)
* ``nse``   Nash-Sutcliffe efficiency (Nash & Sutcliffe, 1970)
* ``pbias`` mean relative error x 100 (%), positive = over-prediction

Nodes with fewer than ``min_obs`` valid pairs are omitted. Network summaries
report the median (and mean/min/max) across nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

METRIC_COLS = ["r2", "mae", "rmse", "mape", "kge", "nse", "pbias"]


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta efficiency (1 = perfect)."""
    if len(obs) < 3:
        return float("nan")
    r, _ = pearsonr(obs, sim)
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency (1 = perfect, 0 = as good as the observed mean)."""
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return float("nan") if denom == 0 else float(1.0 - np.sum((obs - sim) ** 2) / denom)


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean relative error in percent (positive = over-prediction)."""
    return float(np.mean((sim - obs) / (np.abs(obs) + 1e-8)) * 100.0)


def compute_node_metrics(obs: np.ndarray, sim: np.ndarray, min_obs: int = 10) -> Dict[str, float]:
    """All metrics for one node's series; ``{}`` if fewer than ``min_obs`` valid pairs."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if valid.sum() < min_obs:
        return {}
    o, s = obs[valid], np.clip(sim[valid], 0.0, None)
    return {
        "r2": float(r2_score(o, s)),
        "mae": float(mean_absolute_error(o, s)),
        "rmse": float(np.sqrt(mean_squared_error(o, s))),
        "mape": float(mean_absolute_percentage_error(o, s)),
        "kge": kge(o, s),
        "nse": nse(o, s),
        "pbias": pbias(o, s),
    }


def compute_all_metrics(
    obs_all: np.ndarray,
    sim_all: np.ndarray,
    node_ids: List[int],
    exclude_from_summary: Optional[List[int]] = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Per-node metrics for (T, N) arrays; index ``node``, plus an ``in_summary`` flag."""
    if obs_all.shape != sim_all.shape or obs_all.shape[1] != len(node_ids):
        raise ValueError(f"Shape mismatch: obs {obs_all.shape}, sim {sim_all.shape}, {len(node_ids)} nodes.")
    exclude = set(exclude_from_summary or [])
    rows = []
    for i, nid in enumerate(node_ids):
        m = compute_node_metrics(obs_all[:, i], sim_all[:, i], min_obs=min_obs)
        if m:
            rows.append({"node": nid, "in_summary": nid not in exclude, **m})
    if not rows:
        return pd.DataFrame(columns=["node", "in_summary", *METRIC_COLS]).set_index("node")
    return pd.DataFrame(rows).set_index("node")


def summarise_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Median, mean, min and max of each metric across nodes flagged ``in_summary``."""
    df = metrics_df[metrics_df["in_summary"].astype(bool)]
    if df.empty:
        return pd.DataFrame(index=["median", "mean", "min", "max"], columns=METRIC_COLS, dtype=float)
    agg = df[[c for c in METRIC_COLS if c in df.columns]]
    return pd.DataFrame({"median": agg.median(), "mean": agg.mean(), "min": agg.min(), "max": agg.max()}).T


def flat_summary(summary_df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    """``{prefix}{stat}_{metric}`` -> value, finite values only (for MLflow)."""
    out = {}
    for stat in summary_df.index:
        for metric in summary_df.columns:
            v = float(summary_df.loc[stat, metric])
            if np.isfinite(v):
                out[f"{prefix}{stat}_{metric}"] = v
    return out
