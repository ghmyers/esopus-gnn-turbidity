"""Step 12 — Integrated Gradients attributions over the test period.

For every gaged target node, attributes the model's turbidity prediction to
the inputs of every node, feature and lag hour, for one window per day over
the test period (stride 24 h, 50 Gauss-Legendre steps). Used for Figure S7
(scripts/20), the ablation comparison (scripts/14) and the attribution-
stability analysis (scripts/16).

Results are cached per node; existing cache files are reused, so the run can
be interrupted and restarted. With the default ``--ig-batch-size 1`` (as in
the paper) all 28 nodes take a few GPU-hours; ``--ig-batch-size 50`` takes
~20 min on a laptop GPU and changes group shares by < 0.001.

Usage
-----
    python scripts/12_integrated_gradients.py --config configs/graphwavenet_full.yaml
    python scripts/12_integrated_gradients.py --config configs/ungaged/SC2.yaml --ig-batch-size 50
    python scripts/12_integrated_gradients.py --config configs/graphwavenet_full.yaml --nodes 33 64 --max-windows 5   # quick check (not cached)

Inputs
------
    data/processed/esopus_graph_features.pkl
    outputs/models/<run>.pt, outputs/scalers/<run>_scaler.pkl

Outputs (``outputs/interpretation/<run>/``)
-------
    background_ig.npz, node_<id>/ig_full_test.npz
    (array layouts are documented in ``esopus.interpretation.integrated_grads``)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from esopus import paths
from esopus.config import load_config
from esopus.constants import IG_TARGET_NODES
from esopus.interpretation.integrated_grads import full_test_ig, load_or_build_background
from esopus.runtime import get_device, load_trained_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", default=IG_TARGET_NODES, help="target nodes (default: all gaged)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="limit the number of test windows (quick check; results go to a temporary directory)")
    ap.add_argument("--ig-batch-size", type=int, default=1,
                    help="integration steps per forward pass (paper: 1; 50 is ~15x faster on a GPU)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    data, model = load_trained_run(cfg, device)
    t_in = cfg.data.num_timesteps_in

    # Scaled inputs: training split (for the baseline) and test split.
    X_train = data.train_dataset.X.numpy()
    X_test = data.test_dataset.X.numpy()

    cache = paths.interpretation_dir(cfg.run_name)
    if args.max_windows:
        cache = Path(tempfile.mkdtemp(prefix="ig_check_"))
        X_test = X_test[:, :, : args.max_windows * 24 + t_in - 1]
        print(f"[ig] quick check: {args.max_windows} windows per node, results in {cache}")
    background = load_or_build_background(paths.interpretation_dir(cfg.run_name), X_train, t_in, cfg.training.seed, device)

    for k, nid in enumerate(args.nodes, 1):
        res = full_test_ig(model, X_test, data.test_dates, data.edge_index, data.edge_weight, background,
                           data.node_ids, nid, cache, t_in, device=device, internal_batch_size=args.ig_batch_size)
        print(f"[ig] ({k}/{len(args.nodes)}) node {nid}: {res['attrs_full_all'].shape[0]} test windows", flush=True)


if __name__ == "__main__":
    main()
