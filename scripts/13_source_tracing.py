"""Step 13 — Relative turbidity-load source tracing.

For each model, compares tributary load shares (sum of turbidity x discharge)
computed from observed vs. predicted turbidity over (i) the full record and
(ii) each storm event separately (outlet turbidity > 250 FNU, +/- 24 h;
33 events). Event shares are also averaged per node across events.

Usage
-----
    python scripts/13_source_tracing.py --config configs/graphwavenet_full.yaml
    python scripts/13_source_tracing.py --config configs/graphwavenet_full.yaml configs/graphwavenet_topology_only.yaml

Inputs
------
    data/processed/esopus_graph_features.pkl      observed turbidity, discharge, discharge_fill_flag
    outputs/predictions/<run>/predicted_fnu.csv.gz (from scripts/11)

Outputs (``outputs/analyses/source_tracing/<run>/``)
-------
    load_fractions.csv   node, measured/predicted share: full record and mean over storm events
    event_pairs.csv      per-event, per-node measured vs. predicted shares
    summary.json         MAE / RMSE (percentage points), %bias, per-event regression slope and r^2
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from esopus import paths
from esopus.config import load_config
from esopus.data import load_graph
from esopus.evaluation import PredictionSet
from esopus.interpretation.source_tracing import evaluate_source_tracing


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    G = load_graph(cfg.graph_path)  # unmasked observations
    preds = PredictionSet.load(paths.predictions_dir(cfg.run_name))
    res = evaluate_source_tracing(G, preds.predicted, preds.dates, preds.node_ids)

    out = paths.ensure_dir(paths.ANALYSES_DIR / "source_tracing" / cfg.run_name)
    nodes = sorted(set(res.full_measured) | set(res.full_predicted), key=lambda n: -res.full_measured.get(n, 0))
    pd.DataFrame({
        "node": nodes,
        "measured_full": [res.full_measured.get(n) for n in nodes],
        "predicted_full": [res.full_predicted.get(n) for n in nodes],
        "measured_events": [res.events_measured.get(n) for n in nodes],
        "predicted_events": [res.events_predicted.get(n) for n in nodes],
    }).to_csv(out / "load_fractions.csv", index=False)
    res.event_pairs.to_csv(out / "event_pairs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(res.stats, indent=2))

    s = res.stats
    print(f"[{cfg.run_name}] full record: MAE {s['full_mae_pct']:.2f} pp, RMSE {s['full_rmse_pct']:.2f} pp | "
          f"event means: MAE {s['event_mean_mae_pct']:.2f} pp | per event ({s.get('n_events', 0)} events, "
          f"{s.get('n_event_pairs', 0)} pairs): slope {s.get('per_event_slope', float('nan')):.3f}, "
          f"r2 {s.get('per_event_r2', float('nan')):.3f}, MAE {s.get('per_event_mae_pct', float('nan')):.2f} pp")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    args = ap.parse_args()
    for c in args.config:
        run(c)


if __name__ == "__main__":
    main()
