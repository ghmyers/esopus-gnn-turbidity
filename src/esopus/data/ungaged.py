"""Masking used by the leave-one-out "ungaged node" validation.

To test whether the model can predict turbidity at a site it has never seen,
one gaged node's turbidity is withheld from the TRAINING period only:

* the node's raw ``y`` (FNU) is set to NaN for every timestamp before the
  test split date, **before** the scaler is fit, so the withheld data cannot
  influence the turbidity scaling range;
* the node's test-period observations are left untouched so the retrained
  model can be scored against them.

The node's input features (discharge, meteorology, ...) are not masked: an
ungaged site would still have modelled forcings. ``prepare_data`` applies this
mask automatically when ``cfg.ungaged.withheld_node`` is set.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd


def mask_node_train_split(G: nx.Graph, node_id: int, split_date: str) -> Dict[str, int]:
    """Set ``G.nodes[node_id]["y"]`` to NaN before ``split_date`` (in place).

    Parameters
    ----------
    G : graph loaded fresh from disk. Do not reuse it for "observed" values.
    node_id : node whose training-period turbidity is withheld.
    split_date : ISO date; must equal ``cfg.data.test_split_date``.

    Returns
    -------
    dict with observation counts before/after masking (for logging).
    """
    if node_id not in G.nodes:
        raise KeyError(f"Node {node_id} is not in the graph.")
    y = G.nodes[node_id]["y"]
    train_mask = y.index < pd.Timestamp(split_date, tz=y.index.tz)

    n_test_before = int(y[~train_mask].notna().sum())
    report = {
        "node_id": int(node_id),
        "n_train_observed_before": int(y[train_mask].notna().sum()),
        "n_test_observed": n_test_before,
    }

    y_masked = y.copy()
    y_masked[train_mask] = np.nan
    G.nodes[node_id]["y"] = y_masked

    # Hard checks: never train on a masking bug.
    assert int(y_masked[train_mask].notna().sum()) == 0, "training observations remain after masking"
    assert int(y_masked[~train_mask].notna().sum()) == n_test_before, "test observations were altered"

    print(
        f"[ungaged] Node {node_id}: withheld {report['n_train_observed_before']:,} training "
        f"observations; {n_test_before:,} test observations kept for scoring."
    )
    return report
