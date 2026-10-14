"""Step 14 — Topology-only ablation: full model vs. NHD adjacency only.

Compares the full model (configs/graphwavenet_full.yaml, adjacency "both") with
the ablation (configs/graphwavenet_topology_only.yaml, adjacency "fixed"),
which differs only by the absence of the learned adaptive adjacency.

Prerequisites (for both runs): scripts/10 (or archived checkpoints),
scripts/11 and scripts/13; scripts/12 for the attribution comparison
(skipped automatically if the IG caches are missing).

Usage
-----
    python scripts/14_ablation_topology_only.py

Outputs (``outputs/analyses/ablation/``)
-------
    param_counts.json                 trainable parameters of both models
    ablation_metrics_summary.csv      median/min/max KGE, NSE, MAE, RMSE (test) and differences
    per_node_comparison.csv           per-node test metrics of both models
    source_tracing_comparison.csv     source-tracing errors of both models
    ig_da_correlation.json            Spearman correlation of storm-period SSP / long-term
                                      antecedent-Q attribution share with drainage area
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from scipy.stats import spearmanr

from esopus import paths
from esopus.config import load_config
from esopus.data import load_graph
from esopus.interpretation.events import outlet_turbidity, storm_mask
from esopus.interpretation.feature_groups import regime_group_fractions
from esopus.interpretation.integrated_grads import load_full_test_ig
from esopus.metadata import drainage_area
from esopus.models import build_model
from esopus.runtime import get_device

CONFIGS = {"full": "configs/graphwavenet_full.yaml", "ablation": "configs/graphwavenet_topology_only.yaml"}


def ig_da_correlations(cfg, G, feature_names, node_ids, test_dates) -> dict | None:
    """Spearman rho between drainage area and storm-period group shares, or None if no IG cache."""
    ig = load_full_test_ig(paths.interpretation_dir(cfg.run_name), node_ids, test_dates, cfg.data.num_timesteps_in)
    if not ig:
        return None
    storm = storm_mask(outlet_turbidity(G))
    fracs = {n: regime_group_fractions(r["target_dates"], r["attrs_full_all"], feature_names, storm, True)
             for n, r in ig.items()}
    fracs = {n: f for n, f in fracs.items() if f}
    da = [drainage_area(n) for n in fracs]
    return {
        "n_nodes": len(fracs),
        "SSP_vs_DA_spearman": float(spearmanr(da, [f["SSP"] for f in fracs.values()]).correlation),
        "long_term_antecedentQ_vs_DA_spearman": float(
            spearmanr(da, [f["Seasonal+ Antecedent"] for f in fracs.values()]).correlation),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = paths.ensure_dir(paths.ANALYSES_DIR / "ablation")
    cfgs = {k: load_config(v) for k, v in CONFIGS.items()}
    G = load_graph(cfgs["full"].graph_path)
    node_ids = sorted(G.nodes())
    feature_names = list(G.nodes[node_ids[0]]["x"].columns)
    dates = G.nodes[node_ids[0]]["x"].index
    test_dates = dates[dates >= pd.Timestamp(cfgs["full"].data.test_split_date, tz=dates.tz)]

    # ---- parameter counts --------------------------------------------------
    device = get_device("cpu")
    counts = {k: build_model(c, len(feature_names), len(node_ids), device).count_parameters() for k, c in cfgs.items()}
    counts["difference"] = counts["full"] - counts["ablation"]
    (out / "param_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"[ablation] trainable parameters: {counts}")

    # ---- prediction skill --------------------------------------------------
    per_node = {k: pd.read_csv(paths.predictions_dir(c.run_name) / "per_node_metrics.csv", index_col="node")
                for k, c in cfgs.items()}
    pd.concat(per_node, axis=1).to_csv(out / "per_node_comparison.csv")
    rows = []
    for metric in ["kge", "nse", "mae", "rmse"]:
        for stat in ["median", "min", "max"]:
            f, a = (getattr(per_node[k][metric], stat)() for k in ("full", "ablation"))
            rows.append({"metric": metric, "stat": stat, "full_model": f, "ablation_model": a, "difference": f - a})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "ablation_metrics_summary.csv", index=False)
    print(summary[summary.stat == "median"].round(4).to_string(index=False))

    # ---- source tracing ----------------------------------------------------
    st = {}
    for k, c in cfgs.items():
        f = paths.ANALYSES_DIR / "source_tracing" / c.run_name / "summary.json"
        if f.exists():
            st[k] = json.loads(f.read_text())
    if st:
        tab = pd.DataFrame(st)
        tab.to_csv(out / "source_tracing_comparison.csv")
        print(tab.loc[["full_mae_pct", "full_rmse_pct", "per_event_mae_pct", "per_event_rmse_pct"]].round(2).to_string())
    else:
        print("[ablation] no source-tracing results; run scripts/13 for both configs")

    # ---- attribution vs drainage area -------------------------------------
    corr = {k: ig_da_correlations(c, G, feature_names, node_ids, test_dates) for k, c in cfgs.items()}
    if all(corr.values()):
        (out / "ig_da_correlation.json").write_text(json.dumps(corr, indent=2))
        print(f"[ablation] IG vs drainage area (storm periods): {json.dumps(corr, indent=1)}")
    else:
        print("[ablation] IG caches missing for one model; run scripts/12 for both configs")


if __name__ == "__main__":
    main()
