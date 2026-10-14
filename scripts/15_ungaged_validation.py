"""Step 15 — Leave-one-out "ungaged node" validation.

For each of the four largest measured turbidity-load contributors
(SC2 = node 30, WOOD1 = 29, EC12 = 27, BK1 = 32):

1. withhold that node's turbidity from the TRAINING split only and retrain
   the full model from scratch (configs/ungaged/<code>.yaml);
2. score the retrained model at the withheld node over the test period, against
   its true observations, and compare with the full (baseline) model;
3. repeat full-record source tracing and report errors with and without the
   withheld node.

Test-period Integrated Gradients for the folds (used by scripts/16) are
computed with scripts/12 on each fold's config.

Usage
-----
    python scripts/15_ungaged_validation.py                 # train (if needed) + analyse all folds
    python scripts/15_ungaged_validation.py --only SC2      # one fold
    python scripts/15_ungaged_validation.py --skip-training # analyse archived checkpoints only

Requires the full model's predictions (scripts/11 on configs/graphwavenet_full.yaml).

Outputs
-------
    outputs/models/gwn_ungaged_<code>.pt, outputs/scalers/..., outputs/predictions/gwn_ungaged_<code>/
    outputs/analyses/ungaged/ungaged_validation_metrics.csv   one row per fold
    outputs/analyses/ungaged/baseline_load_ranking.csv        measured vs. full-model load shares
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from esopus import paths
from esopus.config import load_config
from esopus.constants import UNGAGED_FOLDS
from esopus.data import load_graph
from esopus.evaluation import PredictionSet
from esopus.interpretation.source_tracing import source_trace
from esopus.runtime import evaluate_run, get_device, train_run

METRICS = ["kge", "nse", "mae", "rmse", "r2", "pbias"]


def full_record_trace(G, preds: PredictionSet):
    return source_trace(G, preds.predicted, preds.dates, preds.node_ids, (preds.dates[0], preds.dates[-1]))


def errors_pp(meas: dict, pred: dict, exclude=()) -> tuple[float, float]:
    nodes = [n for n in meas if n in pred and n not in exclude]
    d = np.array([pred[n] - meas[n] for n in nodes]) * 100
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(UNGAGED_FOLDS), default=list(UNGAGED_FOLDS))
    ap.add_argument("--skip-training", action="store_true", help="use existing checkpoints")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    out = paths.ensure_dir(paths.ANALYSES_DIR / "ungaged")

    base_cfg = load_config("configs/graphwavenet_full.yaml")
    G = load_graph(base_cfg.graph_path)  # unmasked observations for measured loads
    base_preds = PredictionSet.load(paths.predictions_dir(base_cfg.run_name))
    base_metrics = base_preds.metrics("test")
    base_pred_frac, meas_frac, _ = full_record_trace(G, base_preds)
    ranking = pd.DataFrame({"node": sorted(meas_frac, key=meas_frac.get, reverse=True)})
    ranking["measured_load_frac"] = ranking["node"].map(meas_frac)
    ranking["full_model_predicted_load_frac"] = ranking["node"].map(base_pred_frac)
    ranking.to_csv(out / "baseline_load_ranking.csv", index_label="rank")
    print("[ungaged] measured load ranking (top 6):\n", ranking.head(6).to_string(index=False))

    rows = []
    for code in args.only:
        cfg = load_config(f"configs/ungaged/{code}.yaml")
        node = cfg.ungaged.withheld_node
        print(f"\n===== fold {code}: node {node} withheld from training =====")
        if not args.skip_training:
            train_run(cfg, device, use_mlflow=not args.no_mlflow)  # resumes / overwrites
        preds = evaluate_run(cfg, device)

        m = preds.metrics("test")
        pred_frac, _, _ = full_record_trace(G, preds)
        mae_all, rmse_all = errors_pp(meas_frac, pred_frac)
        mae_wo, rmse_wo = errors_pp(meas_frac, pred_frac, exclude=[node])
        row = {"code": code, "node_id": node,
               **{f"baseline_{k}": base_metrics.loc[node, k] for k in METRICS},
               **{f"ungaged_{k}": m.loc[node, k] if node in m.index else np.nan for k in METRICS},
               "n_valid_test_obs": int(np.isfinite(preds.subset("test").observed[:, preds.node_ids.index(node)]).sum()),
               "measured_load_frac": meas_frac.get(node), "baseline_predicted_load_frac": base_pred_frac.get(node),
               "ungaged_predicted_load_frac": pred_frac.get(node),
               "source_trace_mae_pp_all": mae_all, "source_trace_rmse_pp_all": rmse_all,
               "source_trace_mae_pp_excl_withheld": mae_wo, "source_trace_rmse_pp_excl_withheld": rmse_wo}
        rows.append(row)
        print(f"[{code}] node {node} test KGE {row['ungaged_kge']:.3f} (full model {row['baseline_kge']:.3f}); "
              f"load share measured {row['measured_load_frac']:.3f}, predicted {row['ungaged_predicted_load_frac']:.3f}")

    table = pd.DataFrame(rows)
    path = out / "ungaged_validation_metrics.csv"
    if path.exists() and set(args.only) != set(UNGAGED_FOLDS):  # keep folds run earlier
        old = pd.read_csv(path)
        table = pd.concat([old[~old["code"].isin(table["code"])], table], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"\n[ungaged] wrote {path}")


if __name__ == "__main__":
    main()
