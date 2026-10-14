"""Step 01 — Build the river-network graph (topology only).

The published network (57 nodes, 56 directed edges, outlet = node 33) is read
from ``data/node_metadata.csv`` and ``data/network_edges.csv`` and checked
(acyclic, every node drains to the outlet).

Usage
-----
    python scripts/01_build_network.py

Output
------
    data/processed/network.pkl
"""

from __future__ import annotations

import argparse
import pickle

import networkx as nx

from esopus import paths
from esopus.constants import OUTLET_NODE
from esopus.graph.network import load_network


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    G = load_network()
    assert nx.is_directed_acyclic_graph(G), "network must be acyclic"
    unreachable = [n for n in G if n != OUTLET_NODE and not nx.has_path(G, n, OUTLET_NODE)]
    print(f"[network] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"nodes not draining to the outlet: {unreachable or 'none'}")
    out = paths.ensure_dir(paths.PROCESSED_DIR) / "network.pkl"
    with open(out, "wb") as f:
        pickle.dump(G, f)
    print(f"[network] wrote {out}")


if __name__ == "__main__":
    main()
