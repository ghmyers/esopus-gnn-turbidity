"""Step 10 — Train a GraphWaveNet model from a config file.

Usage
-----
    python scripts/10_train.py --config configs/graphwavenet_full.yaml
    python scripts/10_train.py --config configs/graphwavenet_topology_only.yaml
    python scripts/10_train.py --config configs/graphwavenet_full.yaml --debug   # 2-epoch smoke test

(The ungaged-validation folds are trained by scripts/15_ungaged_validation.py.)

Inputs
------
    data/processed/esopus_graph_features.pkl     (from scripts/03, or the Zenodo deposit)

Outputs (``<run>`` = experiment.run_name in the config; ``<run>_debug`` with --debug)
-------
    outputs/models/<run>.pt                 best-epoch weights (lowest test-split masked MSE)
    outputs/models/<run>_resume.pt          per-epoch resume state (removed when training finishes)
    outputs/models/<run>_history.json       per-epoch train / test loss
    outputs/scalers/<run>_scaler.pkl        scaler fitted on the training split
    outputs/mlruns/                         MLflow run (unless --no-mlflow)

If the job is interrupted, re-running the same command resumes from the last
completed epoch.

Re-training reproduces the published checkpoints only up to GPU
non-determinism; use the archived checkpoints to reproduce the paper's
numbers exactly.
"""

from __future__ import annotations

import argparse

from esopus.config import load_config
from esopus.runtime import get_device, train_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (see configs/)")
    ap.add_argument("--debug", action="store_true", help="2 epochs, patience 2, run name suffixed '_debug'")
    ap.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    ap.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto)")
    args = ap.parse_args()

    overrides = {"training": {"num_epochs": 2, "patience": 2, "eval_interval": 1}} if args.debug else None
    cfg = load_config(args.config, overrides=overrides)
    if args.debug:
        cfg.experiment.run_name += "_debug"
    train_run(cfg, get_device(args.device), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
