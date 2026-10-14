"""River-network topology.

Nodes are NHD medium-resolution flowline junctions plus monitoring gages
snapped onto the flowlines; directed edges are flowline segments pointing
downstream. The network used in the paper (57 nodes, 56 edges, outlet = node
33) was built in GIS from the NHD flowlines and gage locations, followed by a
few manual edits (the Coldbrook→node 18 edge was removed, the Shandaken
Tunnel was attached at node 28, three Stony Clove reaches were re-oriented).
The final network is committed as two tables:

``data/node_metadata.csv``  node ID, name, site number, position (EPSG:5070), ...
``data/network_edges.csv``  upstream -> downstream node pairs with NHDPlus attributes

:func:`load_network` builds the graph from these tables.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from esopus.paths import REPO_ROOT

NODE_METADATA_CSV = REPO_ROOT / "data" / "node_metadata.csv"
NETWORK_EDGES_CSV = REPO_ROOT / "data" / "network_edges.csv"


def load_network(nodes_csv: Path = NODE_METADATA_CSV, edges_csv: Path = NETWORK_EDGES_CSV) -> nx.DiGraph:
    """Published network: nodes with ``pos``/``site_no``/``name``; edges with NHDPlus attributes."""
    nodes = pd.read_csv(nodes_csv, dtype={"site_no": str}, keep_default_na=False, na_values=[""])
    edges = pd.read_csv(edges_csv)
    G = nx.DiGraph()
    for r in nodes.itertuples():
        G.add_node(int(r.node), pos=(float(r.easting), float(r.northing)),
                   site_no=r.site_no if isinstance(r.site_no, str) else None, name=r.name)
    for r in edges.to_dict("records"):
        u, v = int(r.pop("upstream_node")), int(r.pop("downstream_node"))
        G.add_edge(u, v, **r)
    return G
