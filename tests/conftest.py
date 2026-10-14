"""Shared fixtures: a tiny synthetic river network with the same structure as the real graph."""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from esopus.config import load_config

FEATURES = ["Q (cfs)", "SSP", "V-Component of Wind"]


def make_graph(n_hours: int = 400, seed: int = 0) -> nx.DiGraph:
    """5 nodes (0,1 -> 2 -> 4 <- 3), hourly UTC index, turbidity at nodes 2 and 4."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-08-01", periods=n_hours, freq="h", tz="UTC")
    G = nx.DiGraph()
    G.add_edges_from([(0, 2), (1, 2), (2, 4), (3, 4)])
    for n in G.nodes:
        q = np.abs(np.cumsum(rng.normal(0, 1, n_hours))) + 5
        x = pd.DataFrame({"Q (cfs)": q, "SSP": q * 0.1, "V-Component of Wind": rng.normal(0, 1, n_hours)}, index=idx)
        y = pd.Series(2 * q + rng.normal(0, 0.5, n_hours) if n in (2, 4) else np.nan, index=idx, name="y")
        G.nodes[n].update(x=x, y=y, pos=(float(n) * 1000, float(n % 2) * 1000))
    G.graph["discharge_fill_flag"] = pd.DataFrame(0, index=idx, columns=sorted(G.nodes))
    return G


@pytest.fixture
def graph() -> nx.DiGraph:
    return make_graph()


@pytest.fixture
def cfg(tmp_path, graph):
    """Full-model config pointed at the synthetic graph, with a short test split."""
    path = tmp_path / "graph.pkl"
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    c = load_config("configs/graphwavenet_full.yaml", overrides={
        "data": {"graph_pkl": str(path), "test_split_date": "2023-08-13"},
        "loader": {"batch_size": 8},
        "training": {"num_epochs": 3, "patience": 3, "eval_interval": 1},
    })
    return c
