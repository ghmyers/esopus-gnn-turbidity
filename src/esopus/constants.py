"""Study-wide constants for the Upper Esopus Creek (UEC) network.

Node IDs refer to the *final* graph numbering (``esopus_graph.pkl`` and the
feature-engineered graph derived from it); see ``docs/pipeline.md`` for how
the original NHD-junction IDs map onto these.

Anything that was previously copy-pasted between scripts and notebook cells
(node lists, labels, basin membership, storm-event thresholds) lives here so
every analysis uses exactly the same definitions.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Study period and data split
# ---------------------------------------------------------------------------

#: First timestamp of the modelled record (hourly, UTC).
RECORD_START = "2016-10-01"
#: Everything strictly before this date is training; on/after is test.
TEST_SPLIT_DATE = "2023-09-19"

# ---------------------------------------------------------------------------
# Key nodes
# ---------------------------------------------------------------------------

#: Watershed outlet: Esopus Creek at Coldbrook (USGS 01362500).
OUTLET_NODE = 33
#: Shandaken Tunnel inflow (inter-basin transfer from the Schoharie Reservoir).
#: Excluded as a regression donor because its discharge is managed, not natural.
SHANDAKEN_TUNNEL_NODE = 41

#: Nodes with continuous USGS discharge records. Source tracing requires both
#: turbidity and discharge, so load contributions are only computed here.
DISCHARGE_NODES = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 41, 42, 44, 63]

#: Main-stem Esopus nodes. These integrate upstream tributary loads and are
#: excluded when attributing load to individual tributaries.
MAIN_ESOPUS_NODES = (28, 33, 41)

#: Nodes with any turbidity observations (USGS + partner monitoring sites).
TURBIDITY_NODES = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42,
                   43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 61, 62, 63, 64]

#: Display names for gaged nodes, used in figures and tables.
LABEL_MAP = {
    26: "Birch Ck",
    27: "Esopus at Oliverea",
    28: "Esopus Ck at Allaben",
    29: "Woodland Ck at Phoenicia",
    30: "Stony Clove at Chichester",
    31: "Warner Ck near Chichester",
    32: "Beaver Kill",
    33: "Esopus at Cold Brook",
    34: "Little Beaver Kill",
    35: "Stony Clove Ck at Janssen Rd",
    38: "Bushnell Ck at Shandaken",
    40: "Woodland Ck at Wilmont Way",
    42: "Broad St Hollow Brook",
    43: "Panther Kill at Panther Kill Rd",
    44: "Panther Kill at Woodland Valley Rd",
    45: "Stony Clove above HWY 214",
    48: "Stony Clove at Silver Hollow Rd",
    49: "Stony Clove above Warner Ck",
    50: "Warner Ck at Silver Hollow Rd",
    51: "Warner Ck near Phoenicia",
    52: "Stony Clove above Moggre Rd",
    53: "Warner Ck near Chichester",
    54: "Warner Ck near Carl Mountain",
    55: "Stony Clove at Lanesville",
    61: "Stony Clove at Wright Rd",
    62: "Stony Clove near Lanesville",
    63: "Myrtle Brook at State HWY 12",
    64: "Stony Clove at Edgewood",
}

#: Nodes for which Integrated Gradients attributions are computed (all
#: labelled gaged sites).
IG_TARGET_NODES = list(LABEL_MAP)

# ---------------------------------------------------------------------------
# Sub-basin membership (used to colour maps and group adjacency weights)
# ---------------------------------------------------------------------------

BASIN_MAP = {
    # Main Esopus stem
    3: "Main Esopus", 5: "Main Esopus", 6: "Main Esopus", 10: "Main Esopus",
    13: "Main Esopus", 14: "Main Esopus", 17: "Main Esopus", 18: "Main Esopus",
    27: "Main Esopus", 28: "Main Esopus", 33: "Main Esopus", 41: "Main Esopus",
    # Stony Clove
    16: "Stony Clove", 19: "Stony Clove", 30: "Stony Clove", 35: "Stony Clove",
    45: "Stony Clove", 48: "Stony Clove", 49: "Stony Clove", 52: "Stony Clove",
    55: "Stony Clove", 61: "Stony Clove", 62: "Stony Clove", 63: "Stony Clove",
    64: "Stony Clove",
    # Woodland Creek
    7: "Woodland Ck", 9: "Woodland Ck", 12: "Woodland Ck", 29: "Woodland Ck",
    39: "Woodland Ck", 40: "Woodland Ck", 43: "Woodland Ck", 44: "Woodland Ck",
    # Beaver Kill
    23: "Beaver Kill", 24: "Beaver Kill", 25: "Beaver Kill", 32: "Beaver Kill",
    # Birch Creek
    0: "Birch Ck", 1: "Birch Ck", 2: "Birch Ck", 26: "Birch Ck",
    # Little Beaver Kill
    20: "Little Beaver Kill", 22: "Little Beaver Kill", 34: "Little Beaver Kill",
    # Bushnell Creek
    4: "Bushnell Ck", 38: "Bushnell Ck",
    # Warner Creek
    21: "Warner Ck", 31: "Warner Ck", 50: "Warner Ck", 51: "Warner Ck",
    53: "Warner Ck", 54: "Warner Ck", 65: "Warner Ck",
    # Broad Street Hollow
    15: "Broad St Hollow", 42: "Broad St Hollow",
}

BASIN_COLORS = {
    "Main Esopus": "#2C7BB6",
    "Stony Clove": "#D7191C",
    "Woodland Ck": "#1A9641",
    "Beaver Kill": "#FDAE61",
    "Birch Ck": "#ABD9E9",
    "Little Beaver Kill": "#A6D96A",
    "Bushnell Ck": "#762A83",
    "Warner Ck": "#B4532A",
    "Broad St Hollow": "#172B2E",
    "Other": "#888780",
}


def basin_of(node_id: int) -> str:
    """Sub-basin name for a node ("Other" for unmapped nodes)."""
    return BASIN_MAP.get(int(node_id), "Other")


# ---------------------------------------------------------------------------
# Storm-event definition (source tracing and IG regime analysis)
# ---------------------------------------------------------------------------

#: A turbidity event is any period with outlet turbidity above this (FNU).
STORM_TURBIDITY_THRESHOLD_FNU = 250.0
#: Each event window is padded by this much on both sides, then overlapping
#: windows are merged.
STORM_EVENT_BUFFER = pd.Timedelta(hours=24)

# ---------------------------------------------------------------------------
# Ungaged (leave-one-out) validation folds
# ---------------------------------------------------------------------------

#: Short code -> node withheld from the training split. These are the four
#: largest measured turbidity-load contributors (see the baseline load
#: ranking written by ``scripts/15_ungaged_validation.py``).
UNGAGED_FOLDS = {
    "SC2": 30,    # Stony Clove Creek below Ox Clove at Chichester
    "WOOD1": 29,  # Woodland Creek at Phoenicia
    "EC12": 27,   # Esopus Creek at Oliverea
    "BK1": 32,   # Beaver Kill at Mount Tremper
}

#: Kling-Gupta efficiency of a mean-flow benchmark (Knoben et al., 2019).
KGE_MEAN_BENCHMARK = -0.41

# ---------------------------------------------------------------------------
# Short site codes used in the manuscript figures
# ---------------------------------------------------------------------------

#: Monitoring-site codes used in the paper's figures, for every gaged node.
SITE_CODES = {
    26: "BIRC1", 27: "EC12", 28: "EC9", 29: "WOOD1", 30: "SC2", 31: "WARN1",
    32: "BK1", 33: "EC1", 34: "LB1", 35: "SC8", 38: "BUSH1", 40: "WOOD4",
    42: "BS1", 43: "WOOD7", 44: "WOOD6", 45: "SC1", 48: "SC4", 49: "SC5",
    50: "WARN2", 51: "WARN3", 52: "SC6", 53: "WARN4", 54: "WARN5", 55: "SC7",
    61: "SC9", 62: "SC10", 63: "SC11", 64: "SC12",
}

#: Gaged nodes in the attribution-vs-drainage-area regressions (Figure S7).
ATTRIBUTION_NODES = [33, 28, 30, 27, 32, 29, 34, 26, 35, 42, 61, 51, 40, 43, 64]


def site_label(node_id: int, with_code: bool = True) -> str:
    """Display label, e.g. "Stony Clove at Chichester (SC2)"."""
    name = LABEL_MAP.get(int(node_id), f"Node {node_id}")
    code = SITE_CODES.get(int(node_id))
    return f"{name} ({code})" if with_code and code else name
