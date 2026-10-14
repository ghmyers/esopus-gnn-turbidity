"""Step 03 — Feature engineering and random-forest feature selection.

1. Drop features with |r| >= 0.8 to an earlier feature (keeping longwave
   radiation and specific humidity).
2. Engineer antecedent-discharge features at lags 17, 197, 2,648, 5,753 and
   7,387 h (``esopus.features.engineering``).
3. Rank all features with a random forest fit on the training period
   (before 2023-09-19) and report the top 14 (``--skip-ranking`` to skip;
   needs ~24 GB RAM and ~10 min on 16 cores).
4. Rebuild each node's feature table with the selected features (engineered
   on the full 2014-2024 record so long lags are populated), then trim the
   record to the outlet's turbidity record (2016-10-01 to 2024-10-20).

By default step 4 uses the published selection, ``data/selected_features.txt``,
whose order is the column order the archived checkpoints expect. A fresh
ranking selects the same 14 features, but nearly tied importances can swap
adjacent ranks; use ``--use-new-ranking`` to build from the fresh ranking
instead (then re-train the models).

Usage
-----
    python scripts/03_engineer_features.py
    python scripts/03_engineer_features.py --skip-ranking

Inputs
------
    data/processed/esopus_graph.pkl            (scripts/02)

Outputs
-------
    data/processed/esopus_graph_features.pkl   model input graph (configs point here)
    data/processed/rf_features.txt             features selected by the fresh ranking
    data/processed/rf_feature_importance.csv   full importance ranking
    outputs/models/random_forest_feature_selection.joblib
"""

from __future__ import annotations

import argparse
import pickle

import joblib
import pandas as pd

from esopus import paths
from esopus.constants import OUTLET_NODE, TEST_SPLIT_DATE
from esopus.data import load_graph
from esopus.features.engineering import ANTECEDENT_LAGS, antecedent_features, correlated_features_to_drop
from esopus.features.selection import build_training_table, rank_features
from esopus.paths import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(paths.PROCESSED_DIR / "esopus_graph.pkl"))
    ap.add_argument("--output", default=str(paths.PROCESSED_DIR / "esopus_graph_features.pkl"))
    ap.add_argument("--n-features", type=int, default=14)
    ap.add_argument("--skip-ranking", action="store_true", help="do not re-run the random-forest ranking")
    ap.add_argument("--use-new-ranking", action="store_true", help="build the graph from the fresh ranking")
    args = ap.parse_args()
    published = [l.strip() for l in open(REPO_ROOT / "data" / "selected_features.txt") if l.strip()]

    G = load_graph(args.input)
    drop = correlated_features_to_drop(G)
    print(f"[features] dropping {len(drop)} correlated features")
    for n in G.nodes:
        G.nodes[n]["x"] = G.nodes[n]["x"].drop(columns=[c for c in drop if c in G.nodes[n]["x"].columns])

    selected = published
    if not args.skip_ranking:
        table = build_training_table(G)
        print(f"[features] random forest on {len(table):,} node-hours x {table.shape[1] - 2} features ...")
        res = rank_features(table, TEST_SPLIT_DATE)
        del table
        print(f"[features] RF test-period scores (scaled): {res.test_scores}")
        ranked = res.top(args.n_features)
        same = set(ranked) == set(published)
        print(f"[features] fresh top-{args.n_features}: {ranked}\n[features] same set as published: {same}")
        if args.use_new_ranking:
            selected = ranked
        res.importances.rename("importance").to_csv(paths.PROCESSED_DIR / "rf_feature_importance.csv", index_label="feature")
        (paths.PROCESSED_DIR / "rf_features.txt").write_text("\n".join(ranked) + "\n")
        joblib.dump(res.model, paths.ensure_dir(paths.MODELS_DIR) / "random_forest_feature_selection.joblib")
    print(f"[features] selected ({len(selected)}): {selected}")

    # Rebuild node tables with only the selected features, then trim to the outlet record.
    # As in the original analysis, all nodes are stacked and sorted by time before
    # being split again; this only reorders rows that share a timestamp (the
    # duplicated hour at the spring daylight-saving change), but keeps the graph
    # bit-identical to the archived one.
    y_out = G.nodes[OUTLET_NODE]["y"]
    start, end = y_out.first_valid_index(), y_out.last_valid_index()
    stacked = []
    for n in G.nodes:
        x = G.nodes[n]["x"]
        df = antecedent_features(x[[c for c in x.columns if c in selected]], ANTECEDENT_LAGS, subset=selected)
        df["node"] = n
        stacked.append(df)
    stacked = pd.concat(stacked).sort_index()
    for n in G.nodes:
        df = stacked[stacked["node"] == n].loc[start:end]
        G.nodes[n]["x"] = df[selected]
        G.nodes[n]["y"] = G.nodes[n]["y"].loc[start:end]  # targets are not re-sorted
    del stacked
    with open(args.output, "wb") as f:
        pickle.dump(G, f)
    print(f"[features] wrote {args.output}  ({start} -> {end})")


if __name__ == "__main__":
    main()
