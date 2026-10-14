"""Storm (turbidity) event detection at the watershed outlet.

An event is a run of hours with outlet turbidity above
``STORM_TURBIDITY_THRESHOLD_FNU`` (250 FNU). Each run is padded by
``STORM_EVENT_BUFFER`` (24 h) on both sides — the run ends at the first hour
back below the threshold, so the post-event buffer starts there — clipped to
the record, and overlapping windows are merged.

These windows define the 33 events used for event-scale source tracing and
the storm/baseflow split of the attribution analysis.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from esopus.constants import OUTLET_NODE, STORM_EVENT_BUFFER, STORM_TURBIDITY_THRESHOLD_FNU

Window = Tuple[pd.Timestamp, pd.Timestamp]


def outlet_turbidity(G) -> pd.Series:
    """Observed outlet turbidity (FNU) with duplicate timestamps removed."""
    y = G.nodes[OUTLET_NODE]["y"]
    return y[~y.index.duplicated(keep="first")]


def detect_storm_events(
    y: pd.Series,
    threshold: float = STORM_TURBIDITY_THRESHOLD_FNU,
    buffer: pd.Timedelta = STORM_EVENT_BUFFER,
) -> List[Window]:
    """Merged, buffered windows around exceedances of ``threshold`` in ``y``."""
    above = y > threshold
    runs, in_event, start = [], False, None
    for t, flag in above.items():
        if flag and not in_event:
            start, in_event = t, True
        elif not flag and in_event:
            runs.append((start, t))
            in_event = False
    if in_event:
        runs.append((start, above.index[-1]))

    windows = sorted((max(y.index[0], s - buffer), min(y.index[-1], e + buffer)) for s, e in runs)
    merged: List[Window] = []
    for s, e in windows:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def storm_mask(y: pd.Series, **kwargs) -> pd.Series:
    """Boolean series over ``y.index``: True inside any storm-event window."""
    mask = pd.Series(False, index=y.index)
    for s, e in detect_storm_events(y, **kwargs):
        mask[s:e] = True
    return mask
