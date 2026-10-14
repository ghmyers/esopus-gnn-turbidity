"""Step 11 — Evaluate trained models and save their full-record predictions.

Usage
-----
    python scripts/11_evaluate.py --config configs/graphwavenet_full.yaml
    python scripts/11_evaluate.py --config configs/graphwavenet_full.yaml configs/graphwavenet_topology_only.yaml configs/ungaged/*.yaml

Inputs
------
    data/processed/esopus_graph_features.pkl
    outputs/models/<run>.pt, outputs/scalers/<run>_scaler.pkl

Outputs (``outputs/predictions/<run>/``)
-------
    per_node_metrics.csv     test-period metrics per gaged node (headline results)
    summary_metrics.csv      median / mean / min / max of those metrics across nodes
    metrics_by_period.csv    per-node metrics for the train, test and full record
    predicted_fnu.csv.gz     hourly predictions, train + test (timestamp = predicted hour)
    observed_fnu.csv.gz      matching observations (true values, never masked)

Every downstream analysis (scripts 12-16, figures) reads these files.
"""

from __future__ import annotations

import argparse

from esopus.config import load_config
from esopus.runtime import evaluate_run, get_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    for c in args.config:
        evaluate_run(load_config(c), device)


if __name__ == "__main__":
    main()
