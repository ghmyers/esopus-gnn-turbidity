"""Step 02 — Assemble the hourly graph with filled discharge and derived attributes.

1. Load the network (scripts/01).
2. Attach AORC meteorology, USGS discharge/turbidity and static attributes to
   every node (``esopus.graph.assemble``).
3. Fit the discharge regressions on Stony Clove (node 30) and Coldbrook
   (node 33) and save them with the measured discharge (Figures S1-S2), then
   fill discharge gaps by regression and drainage-area scaling, recording the
   fill flags (``esopus.graph.discharge_fill``).
4. Interpolate gaps <= 24 h, compute specific stream power, remove pressure
   outliers (``esopus.graph.attributes``).

Usage
-----
    python scripts/02_build_graph.py

Inputs (Zenodo deposit, see data/README.md)
------
    data/processed/network.pkl, data/raw/aorc/node_<id>.csv, data/raw/usgs/<site_no>.csv,
    data/raw/streamstats/streamstats_attributes.csv

Outputs
-------
    data/processed/esopus_graph.pkl                     pre-feature graph (input of scripts/03)
    outputs/analyses/discharge_regressions/
        measured_discharge.csv.gz                       hourly gaged discharge before filling
        regressions_stony_clove.csv, regressions_coldbrook.csv   OLS fits (slope, intercept, R², n)
"""

from __future__ import annotations

import argparse
import pickle

from esopus import paths
from esopus.data import load_graph
from esopus.graph.assemble import add_static_attributes, attach_time_series
from esopus.graph.attributes import add_specific_stream_power, interpolate_short_gaps, remove_pressure_outliers
from esopus.graph.discharge_fill import COLDBROOK, STONY_CLOVE, fill_discharge, fit_regressions, measured_discharge
from esopus.paths import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streamstats", default=str(paths.RAW_DIR / "streamstats" / "streamstats_attributes.csv"))
    args = ap.parse_args()

    G = load_graph(paths.PROCESSED_DIR / "network.pkl")
    attach_time_series(G, paths.RAW_DIR / "aorc", paths.RAW_DIR / "usgs")
    add_static_attributes(G)

    reg_dir = paths.ensure_dir(paths.ANALYSES_DIR / "discharge_regressions")
    measured_discharge(G).to_csv(reg_dir / "measured_discharge.csv.gz")
    for ref, name in ((STONY_CLOVE, "stony_clove"), (COLDBROOK, "coldbrook")):
        fit_regressions(G, ref).to_csv(reg_dir / f"regressions_{name}.csv", index=False)
    print(f"[graph] discharge regressions written to {reg_dir}")

    flags = fill_discharge(G, args.streamstats)
    print(f"[graph] discharge filled; share of filled hours per node:\n{(flags.mean() * 100).round(1).to_string()}")
    interpolate_short_gaps(G, max_gap_hours=24)
    add_specific_stream_power(G)
    print(f"[graph] pressure outliers removed: {remove_pressure_outliers(G):,}")

    columns = [c.strip() for c in open(REPO_ROOT / "data" / "pre_feature_columns.txt") if c.strip()]
    for n in G.nodes:  # canonical column order (the correlation filter in step 05 depends on it)
        G.nodes[n]["x"] = G.nodes[n]["x"][columns]
    out = paths.PROCESSED_DIR / "esopus_graph.pkl"
    with open(out, "wb") as f:
        pickle.dump(G, f)
    print(f"[graph] wrote {out}")


if __name__ == "__main__":
    main()
