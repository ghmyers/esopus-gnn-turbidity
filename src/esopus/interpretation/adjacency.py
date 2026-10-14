"""Analysis of the learned (adaptive) adjacency matrix.

GraphWaveNet learns ``A = softmax(relu(E1 @ E2.T))`` (rows sum to 1); entry
``A[i, j]`` is the weight with which node i aggregates information from node j
in the dense graph convolution. This module

* extracts ``A`` from a trained model,
* aggregates it to stream reaches of the Stream Feature Inventory (SFI) for
  the chord diagram of Figure 5, and
* computes turbidity-sensor density per reach (sensors per km of modelled
  channel), the colour scale of Figure 5.

Data files (committed in ``data/``):
``stream_reaches_sfi_2022.csv`` — SFI reaches with their graph nodes;
``node_metadata.csv`` — node channel length, basin, ...
"""

from __future__ import annotations

import ast
from typing import Dict, List

import numpy as np
import pandas as pd

from esopus.constants import TURBIDITY_NODES
from esopus.metadata import node_metadata
from esopus.paths import REPO_ROOT

SFI_CSV = REPO_ROOT / "data" / "stream_reaches_sfi_2022.csv"


def learned_adjacency(model) -> np.ndarray:
    """(N, N) learned adjacency of a trained ``WaveNetModel`` as a NumPy array."""
    import torch

    with torch.no_grad():
        return model.learned_adjacency().cpu().numpy()


# ---------------------------------------------------------------------------
# Stream reaches (Stream Feature Inventory)
# ---------------------------------------------------------------------------

def load_reaches() -> pd.DataFrame:
    """SFI reaches with parsed node lists and a display name."""
    sfi = pd.read_csv(SFI_CSV)
    sfi["nodes"] = sfi["nodes"].apply(ast.literal_eval)
    sfi["display_name"] = sfi["stream_name"].map(_reach_display_name)
    return sfi


def _reach_display_name(stream_name: str) -> str:
    names = {
        "esopus_creek_all": "Esopus Creek",
        "esopus_lost_cove_bushnellsville": "Upper Esopus Creek\nLost Cove to Bushnellsville",
        "esopus_upstream_lost_cove": "Upper Esopus Creek\nUpstream Lost Cove",
        "broadstreet_hollow": "Broad St Hollow",
        "bushnellsville_creek": "Bushnellsville Creek",
        "little_beaver_kill": "Little Beaver Kill",
        "stony_clove_creek": "Stony Clove",
    }
    return names.get(stream_name, stream_name.replace("_", " ").title())


def node_to_reach(sfi: pd.DataFrame) -> Dict[int, int]:
    """Map node ID -> row index of its SFI reach."""
    return {n: r for r, nodes in sfi["nodes"].items() for n in nodes}


def reach_sensor_density(sfi: pd.DataFrame) -> pd.Series:
    """Turbidity sensors per km of modelled channel, per reach.

    Channel length is the sum of ``StreamLength_m`` (length of the NHD
    flowline ending at each node) over the reach's nodes.
    """
    length_m = node_metadata()["StreamLength_m"]
    dens = {}
    for r, nodes in sfi["nodes"].items():
        km = length_m.reindex(nodes).sum() / 1000
        dens[r] = sum(n in TURBIDITY_NODES for n in nodes) / km if km > 0 else np.nan
    return pd.Series(dens, name="sensors_per_km")


def aggregate_to_reaches(A: np.ndarray, node_ids: List[int], sfi: pd.DataFrame) -> np.ndarray:
    """Sum learned weights between nodes of different reaches -> (R, R).

    ``out[a, b]`` = sum of ``A[i, j]`` over nodes i in reach a and j in reach b
    (a != b); within-reach weights are ignored.
    """
    n2r = node_to_reach(sfi)
    out = np.zeros((len(sfi), len(sfi)))
    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            ri, rj = n2r.get(ni), n2r.get(nj)
            if i != j and ri is not None and rj is not None and ri != rj:
                out[ri, rj] += A[i, j]
    return out
