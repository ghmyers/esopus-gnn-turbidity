"""Attach hourly time series and static attributes to the network.

For every node:

1. AORC meteorology (8 variables) defines the hourly time index. The
   AORC timestamps are treated as US-Eastern local time and converted to UTC,
   as in the original analysis (see docs/KNOWN_ISSUES.md — AORC is distributed
   in UTC, so this may shift forcings by 4-5 h relative to the gage data).
2. If the node has a USGS gage file, 15-min discharge and turbidity are
   averaged to hourly values, hours with negative or missing discharge (or
   negative turbidity) are dropped, and the result is right-joined onto
   the AORC index. Ungaged nodes get all-NaN discharge and turbidity.
3. Static attributes (``data/node_metadata.csv``: drainage area, channel
   slope, flowline length, bankfull width; ``data/node_catchment_attributes.csv``:
   NHDPlus catchment attributes) are added as constant columns.

The node's ``x`` table holds every feature; ``y`` holds turbidity (FNU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from esopus.paths import REPO_ROOT

CATCHMENT_ATTRIBUTES_CSV = REPO_ROOT / "data" / "node_catchment_attributes.csv"
NODE_METADATA_CSV = REPO_ROOT / "data" / "node_metadata.csv"
METADATA_STATIC = ["DA_sqmi", "SLOPE", "StreamLength_m", "Channel_width"]
TURBIDITY_COL, Q_COL = "turbidity (FNU)", "Q (cfs)"


def aorc_to_utc(aorc: pd.DataFrame) -> pd.DataFrame:
    """Interpret AORC timestamps as America/New_York and convert to UTC (original convention)."""
    df = aorc.copy()
    df.index = pd.to_datetime(df.index).tz_localize("America/New_York", nonexistent="shift_forward",
                                                    ambiguous="NaT").tz_convert("UTC")
    df = df[~df.index.isna()]
    df.index.name = "datetime"
    return df


def hourly_usgs(usgs: pd.DataFrame) -> pd.DataFrame:
    """15-min gage data -> hourly means; drop hours with negative/missing Q or negative turbidity."""
    df = usgs.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    hourly = df.resample("h").mean()
    if Q_COL in hourly:
        hourly = hourly.loc[hourly[Q_COL] >= 0]          # also drops hours without discharge
    if TURBIDITY_COL in hourly:
        hourly = hourly.loc[hourly[TURBIDITY_COL] >= 0]  # also drops hours without turbidity
    return hourly


def node_tables(aorc_utc: pd.DataFrame, usgs: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """Merge one node's gage data onto its AORC index; return ``(x, y)``."""
    if usgs is not None:
        merged = hourly_usgs(usgs).merge(aorc_utc, how="right", left_index=True, right_index=True)
    else:
        merged = aorc_utc.copy()
    for col in (Q_COL, TURBIDITY_COL):
        if col not in merged:
            merged[col] = np.nan
    y = merged.pop(TURBIDITY_COL).rename("y")
    return merged, y


def add_static_attributes(G: nx.DiGraph) -> None:
    """Append static per-node attributes as constant columns of ``x``."""
    meta = pd.read_csv(NODE_METADATA_CSV).set_index("node")[METADATA_STATIC]
    cat = pd.read_csv(CATCHMENT_ATTRIBUTES_CSV, index_col="node")
    cat = cat.drop(columns=[c for c in cat.columns if c in meta.columns])
    for n in G.nodes:
        x = G.nodes[n]["x"]
        static = pd.concat([cat.loc[n], meta.loc[n]])
        G.nodes[n]["x"] = pd.concat([x, pd.DataFrame({c: float(v) for c, v in static.items()}, index=x.index)], axis=1)


def attach_time_series(G: nx.DiGraph, aorc_dir: Path, usgs_dir: Path) -> None:
    """Fill every node's ``x``/``y`` from ``aorc_dir/node_<id>.csv`` and ``usgs_dir/<site_no>.csv``."""
    for n, d in G.nodes(data=True):
        aorc = pd.read_csv(Path(aorc_dir) / f"node_{n}.csv", index_col=0, parse_dates=True)
        site = d.get("site_no")
        f = Path(usgs_dir) / f"{site}.csv" if site else None
        usgs = pd.read_csv(f, index_col=0) if f is not None and f.exists() else None
        x, y = node_tables(aorc_to_utc(aorc), usgs)
        G.nodes[n]["x"], G.nodes[n]["y"] = x, y
