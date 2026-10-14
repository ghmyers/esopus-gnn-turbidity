"""Physically motivated feature groups and attribution summaries.

Each of the 14 model inputs is assigned to one group by its name:

* ``SSP``                  specific stream power
* ``Q (cfs)``              current discharge
* ``Meteorological``       V-component of wind
* ``Storm Antecedent``     short-memory discharge terms (17-h and 197-h lags,
                           range statistics); "Antecedent Q (short-term)" in figures
* ``Seasonal+ Antecedent`` long-memory discharge terms (2,648-7,387 h lags);
                           "Antecedent Q (long-term)" in figures

Attributions are summarised as fractions of total mean |IG| per group,
optionally split into turbidity-event and non-event regimes
(``esopus.interpretation.events``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

GROUP_ORDER = ["SSP", "Q (cfs)", "Meteorological", "Storm Antecedent", "Seasonal+ Antecedent"]
GROUP_COLORS = {
    "SSP": "#C0392B",
    "Q (cfs)": "#378ADD",
    "Meteorological": "#5F5E5A",
    "Storm Antecedent": "#8E44AD",
    "Seasonal+ Antecedent": "#D4A017",
}


def assign_feature_group(name: str) -> str:
    """Group of one feature column (first matching rule wins)."""
    if "SSP" in name:
        return "SSP"
    if name == "Q (cfs)":
        return "Q (cfs)"
    if "V-Component" in name:
        return "Meteorological"
    if "range" in name or "_17h" in name or "_197h" in name:
        return "Storm Antecedent"
    return "Seasonal+ Antecedent"


def group_fractions(per_feature: np.ndarray, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """Normalise a per-feature attribution vector (F,) into group fractions."""
    total = float(per_feature.sum())
    if total == 0:
        return None
    totals = {g: 0.0 for g in GROUP_ORDER}
    for value, name in zip(per_feature, feature_names):
        totals[assign_feature_group(name)] += float(value)
    return {g: totals[g] / total for g in GROUP_ORDER}


def regime_group_fractions(
    target_dates: pd.DatetimeIndex,
    attrs_full_all: np.ndarray,
    feature_names: List[str],
    storm: pd.Series,
    is_storm: bool,
) -> Optional[Dict[str, float]]:
    """Group fractions over the IG samples in one regime.

    ``attrs_full_all`` (S, N, F, T_in) are one node's per-window attributions,
    ``target_dates`` their target timestamps and ``storm`` a boolean storm
    mask (nearest-timestamp lookup). Mean |IG| is taken over the selected
    samples and time, summed over source nodes, then grouped.
    """
    flags = storm.reindex(target_dates, method="nearest").to_numpy(dtype=bool)
    idx = np.where(flags == is_storm)[0]
    if len(idx) == 0:
        return None
    per_feature = np.abs(attrs_full_all[idx]).mean(axis=(0, 3)).sum(axis=0)
    return group_fractions(per_feature, feature_names)
