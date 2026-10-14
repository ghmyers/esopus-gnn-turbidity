"""Final per-node clean-up and derived hydraulics.

* :func:`interpolate_short_gaps` — time-weighted linear interpolation of gaps
  of at most 24 h inside the record (features and turbidity).
* :func:`add_specific_stream_power` — SSP = ρg·Q·S / w (W m⁻²) with
  ρg = 9810 N m⁻³, Q in m³ s⁻¹, S = SLOPE (ft/mi) / 5280 and w the bankfull
  width (m; Catskill regional curve w = 17.1·DA^0.46 ft if missing).
* :func:`remove_pressure_outliers` — surface pressure < 87,500 Pa set to NaN.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def interpolate_short_gaps(G: nx.DiGraph, max_gap_hours: int = 24) -> None:
    for n in G.nodes:
        x = G.nodes[n]["x"]
        for col in x.columns:
            if x[col].isna().any():
                x[col] = x[col].interpolate(method="time", limit=max_gap_hours, limit_direction="forward",
                                            limit_area="inside")
        y = G.nodes[n]["y"]
        if y.isna().any():
            G.nodes[n]["y"] = y.interpolate(method="time", limit=max_gap_hours, limit_direction="forward",
                                            limit_area="inside")


def add_specific_stream_power(G: nx.DiGraph) -> None:
    for n in G.nodes:
        x = G.nodes[n]["x"]
        if "Channel_width" not in x or x["Channel_width"].isna().all():
            x["Channel_width"] = 17.1 * float(x["DA_sqmi"].dropna().iloc[0]) ** 0.460 * 0.3048
        x["SSP"] = 9810 * (x["Q (cfs)"] * 0.0283168) * (x["SLOPE"] / 5280) / x["Channel_width"]


def remove_pressure_outliers(G: nx.DiGraph, min_pa: float = 87_500.0) -> int:
    removed = 0
    for n in G.nodes:
        x = G.nodes[n]["x"]
        if "Pressure" in x:
            bad = x["Pressure"] < min_pa
            removed += int(bad.sum())
            x.loc[bad, "Pressure"] = np.nan
    return removed
