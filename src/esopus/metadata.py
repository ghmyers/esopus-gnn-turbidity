"""Static per-node metadata shipped with the repository (``data/node_metadata.csv``).

Columns: node, name, site_code, basin, has_turbidity, has_discharge,
easting/northing (EPSG:5070, m), DA_sqmi / DA_km2 (drainage area), SLOPE,
StreamLength_m, Channel_width. Drainage area and channel geometry were copied
from the pre-feature graph (``esopus_graph.pkl``) so analyses that sort or
colour nodes by drainage area do not need to load that 6 GB file.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from esopus.paths import REPO_ROOT

NODE_METADATA_CSV = REPO_ROOT / "data" / "node_metadata.csv"


@lru_cache(maxsize=1)
def node_metadata() -> pd.DataFrame:
    """Node metadata indexed by node ID."""
    return pd.read_csv(NODE_METADATA_CSV, keep_default_na=False, na_values=[""]).set_index("node")


def drainage_area(node_id: int, unit: str = "km2") -> float:
    """Drainage area of a node in ``"km2"`` or ``"sqmi"``."""
    return float(node_metadata().loc[int(node_id), "DA_km2" if unit == "km2" else "DA_sqmi"])
