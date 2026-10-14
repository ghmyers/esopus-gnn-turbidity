"""Relative turbidity-load source tracing.

For a time window, the turbidity load delivered by each tributary node is
approximated by the cumulative product of turbidity and discharge,
``sum_t C(t) * Q(t)``, and expressed as a fraction of the summed load of all
tributaries. This is done twice:

* **measured**  — with observed turbidity;
* **predicted** — with the model's turbidity at the same node.

Comparing the two tells how well the model attributes sediment to its
sources. Only nodes with observed discharge (``DISCHARGE_NODES``) can be
traced, main-stem nodes (``MAIN_ESOPUS_NODES``) are excluded because they
integrate upstream loads, and only hours where *every* discharge node has
measured (not gap-filled) discharge are used (``G.graph["discharge_fill_flag"]``).

Predictions must be stamped with the time of the target they predict
(``esopus.data.window_target_dates``); otherwise predicted concentrations
are multiplied by the wrong hour's discharge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from esopus.constants import DISCHARGE_NODES, MAIN_ESOPUS_NODES, OUTLET_NODE
from esopus.interpretation.events import Window, detect_storm_events, outlet_turbidity


def _dedup(s: pd.Series) -> pd.Series:
    return s[~s.index.duplicated(keep="first")]


def _normalise(d: Dict[int, float]) -> Dict[int, float]:
    total = sum(d.values())
    return {n: round(v / total, 4) for n, v in d.items()} if total > 0 else d


def source_trace(
    G,
    preds: np.ndarray,
    pred_dates: pd.DatetimeIndex,
    node_ids: List[int],
    window: Window,
    discharge_nodes: Sequence[int] = DISCHARGE_NODES,
    main_stem_nodes: Sequence[int] = MAIN_ESOPUS_NODES,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Predicted and measured relative load per tributary over ``window``.

    Parameters
    ----------
    G : graph with observed ``y`` (FNU), ``x["Q (cfs)"]`` and
        ``G.graph["discharge_fill_flag"]`` (T x node DataFrame, 1 = gap-filled).
        Use an *unmasked* graph so measured loads use all observations.
    preds : (T_pred, N) predicted turbidity (FNU), columns in ``node_ids`` order.
    pred_dates : target timestamps of ``preds`` rows.
    window : (start, end) timestamps, inclusive.

    Returns
    -------
    predicted : {node: fraction}, fractions sum to ~1 (rounded to 4 dp)
    measured  : {node: fraction}
    observed_fraction : share of hours in ``window`` with all-measured discharge
    """
    start, end = window
    fill = G.graph.get("discharge_fill_flag")
    observed_times: Optional[pd.DatetimeIndex] = None
    observed_fraction = 1.0
    if fill is not None:
        cols = [n for n in discharge_nodes if n in fill.columns]
        all_measured = (fill[cols].loc[start:end] == 0).all(axis=1)
        observed_fraction = float(all_measured.mean()) if len(all_measured) else 0.0
        if not all_measured.any():
            raise ValueError("No hours with measured discharge at every discharge node in this window.")
        observed_times = all_measured[all_measured].index

    def restrict(idx: pd.Index) -> pd.Index:
        return idx if observed_times is None else idx.intersection(observed_times)

    # The outlet load is computed for reference/diagnostics only; fractions
    # are normalised over tributaries, so it cancels out of the result.
    outlet_load = _dedup(G.nodes[OUTLET_NODE]["y"]) * _dedup(G.nodes[OUTLET_NODE]["x"]["Q (cfs)"])
    outlet_total = float(np.nansum(outlet_load.reindex(restrict(outlet_load.loc[start:end].index))))
    if outlet_total == 0:
        raise ValueError("Outlet load is zero over the measured-discharge hours of this window.")

    pred_series = {nid: pd.Series(preds[:, i], index=pred_dates) for i, nid in enumerate(node_ids)}
    tributaries = [n for n in G.nodes() if n in discharge_nodes and n not in main_stem_nodes]

    predicted, measured = {}, {}
    for node in tributaries:
        q = _dedup(G.nodes[node]["x"]["Q (cfs)"]).loc[start:end]

        if node in pred_series:
            p = pred_series[node].loc[start:end]
            idx = restrict(p.index.intersection(q.index))
            if len(idx):
                predicted[node] = float(np.nansum(p[idx] * q[idx])) / outlet_total

        c = _dedup(G.nodes[node]["y"]).loc[start:end]
        if c.isna().all():
            continue
        idx = restrict(c.index.intersection(q.index))
        if len(idx):
            load = float(np.nansum(c[idx] * q[idx]))
            if load > 0:
                measured[node] = load / outlet_total

    return _normalise(predicted), _normalise(measured), observed_fraction


# ---------------------------------------------------------------------------
# Network-level evaluation (full record and per storm event)
# ---------------------------------------------------------------------------

@dataclass
class SourceTracingResult:
    """Load fractions and error statistics for one model.

    full_*   : fractions over the full prediction record
    events_* : each node's share averaged over all storm events (the
               "turbidity events" bars of Figure 4; a missing observed share
               counts as 0)
    event_pairs : per-event, per-node shares (one row per event x node;
               ``measured`` is NaN when the sensor had no data in that event)
    """
    full_predicted: Dict[int, float]
    full_measured: Dict[int, float]
    events_predicted: Dict[int, float]
    events_measured: Dict[int, float]
    event_pairs: pd.DataFrame       # event, start, end, node, measured, predicted, observed_fraction
    stats: Dict[str, float]

    def ranking(self) -> pd.DataFrame:
        """Tributaries ranked by measured full-record load fraction."""
        nodes = sorted(self.full_measured, key=self.full_measured.get, reverse=True)
        return pd.DataFrame({
            "node": nodes,
            "measured_load_frac": [self.full_measured[n] for n in nodes],
            "predicted_load_frac": [self.full_predicted.get(n, np.nan) for n in nodes],
        })


def _errors(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """MAE/RMSE in percentage points of load share."""
    d = (predicted - measured) * 100
    return {"mae_pct": float(np.mean(np.abs(d))), "rmse_pct": float(np.sqrt(np.mean(d ** 2)))}


def evaluate_source_tracing(
    G,
    preds: np.ndarray,
    pred_dates: pd.DatetimeIndex,
    node_ids: List[int],
    exclude_nodes: Sequence[int] = (),
) -> SourceTracingResult:
    """Source tracing over the full prediction record and every storm event.

    ``exclude_nodes`` are left out of the error statistics (used to report
    ungaged-validation errors with and without the withheld node).
    """
    full_window = (pred_dates[0], pred_dates[-1])
    full_pred, full_meas, _ = source_trace(G, preds, pred_dates, node_ids, full_window)

    events = detect_storm_events(outlet_turbidity(G))
    rows = []
    for k, (s, e) in enumerate(events):
        try:
            p, m, frac = source_trace(G, preds, pred_dates, node_ids, (s, e))
        except ValueError:
            continue  # no hours with measured discharge at every gauge
        rows.extend(
            {"event": k + 1, "start": s, "end": e, "node": n, "measured": m.get(n, np.nan),
             "predicted": p[n], "observed_fraction": frac}
            for n in p
        )
    pairs = pd.DataFrame(rows)

    # Event-mean shares (Figure 4 outline bars): averaged over all traced events,
    # a missing observed share (sensor without data in that event) counts as 0.
    n_events = pairs["event"].nunique()
    ev_meas = (pairs.groupby("node")["measured"].sum(min_count=1) / n_events).dropna().to_dict()
    ev_pred = pairs.groupby("node")["predicted"].mean().to_dict()

    def frac_errors(meas, pred, prefix):
        common = [n for n in meas if n in pred and n not in exclude_nodes]
        m, p = np.array([meas[n] for n in common]), np.array([pred[n] for n in common])
        out = {f"{prefix}_{k}": v for k, v in _errors(m, p).items()}
        rel = (p - m) / np.where(np.abs(m) > 0, np.abs(m), 1) * 100  # per-site relative error of the load share
        # Mean relative error is dominated by tributaries with tiny observed shares
        # (and the plain PBIAS of shares that sum to 1 is always 0), so the median
        # is the reported bias statistic; the mean is kept for reference.
        out[f"{prefix}_median_bias_pct"] = float(np.median(rel))
        out[f"{prefix}_pbias_pct"] = float(np.mean(rel))
        return out

    stats = {**frac_errors(full_meas, full_pred, "full"), **frac_errors(ev_meas, ev_pred, "event_mean")}
    ev = pairs.dropna(subset=["measured"])            # per-event regression: matched pairs only
    ev = ev[~ev["node"].isin(exclude_nodes)]
    if len(ev) >= 3:
        reg = linregress(ev["measured"], ev["predicted"])
        stats.update({f"per_event_{k}": v for k, v in _errors(ev["measured"].to_numpy(), ev["predicted"].to_numpy()).items()})
        stats.update({
            "per_event_slope": float(reg.slope), "per_event_intercept": float(reg.intercept),
            "per_event_r2": float(reg.rvalue ** 2), "per_event_p": float(reg.pvalue),
            "per_event_median_bias_pct": float(np.median((ev["predicted"] - ev["measured"]) / ev["measured"] * 100)),
            "per_event_pbias_pct": float(np.mean((ev["predicted"] - ev["measured"]) / ev["measured"].where(ev["measured"].abs() > 0, 1)) * 100),
            "n_event_pairs": int(len(ev)), "n_events": int(ev["event"].nunique()),
        })
    return SourceTracingResult(full_pred, full_meas, ev_pred, ev_meas, pairs, stats)
