"""Step 16 — Stability of the attribution-vs-drainage-area relationships.

Repeats the supplementary Attribution_vs_DA analysis for every trained model
with a full-test IG cache (scripts/12):

* the full model (baseline),
* the topology-only ablation, and
* the four leave-one-out ungaged-validation models (SC2, WOOD1, EC12, BK1),
  which share the full model's architecture but were retrained from scratch
  with one node's training-period turbidity withheld.

For each model, the share of mean |IG| per feature group is regressed (OLS)
on ln(drainage area, km²): Q, SSP and short-term antecedent Q during
turbidity events, and long-term antecedent Q outside events. Each regression
is repeated for the 15 nodes of the published figure and for all 28 gaged
nodes. The spread of the relationships is reported as:

* within each fit: slope standard error and 95 % confidence interval,
  intercept standard error, R², residual standard error, p, Spearman rho;
* across the five retrainings of the full architecture (baseline + four
  ungaged folds): mean, standard deviation, min and max of slope, R² and rho;
* per node: standard deviation of each group share across the five
  retrainings, and the Spearman rank agreement of node shares between models.

Usage
-----
    python scripts/12_integrated_gradients.py --config configs/ungaged/<code>.yaml --ig-batch-size 50   # per fold, first
    python scripts/16_attribution_stability.py

Outputs
-------
    outputs/analyses/attribution_stability/
        regressions.csv          one row per model x node set x group
        across_models.csv        mean / SD / min / max over the five retrainings of the full architecture
        node_shares.csv          group share per model x regime x node
        node_share_spread.csv    per node and group: mean, SD, CV over the five retrainings
        rank_agreement.csv       Spearman rho of node shares between each pair of models
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from esopus import paths
from esopus.config import load_config
from esopus.constants import ATTRIBUTION_NODES, IG_TARGET_NODES, SITE_CODES
from esopus.data import load_graph
from esopus.interpretation.events import outlet_turbidity, storm_mask
from esopus.interpretation.integrated_grads import load_full_test_ig
from esopus.metadata import drainage_area
from esopus.plotting import attribution as pattr

MODELS = {  # label -> config
    "Full (baseline)": "configs/graphwavenet_full.yaml",
    "Topology-only (ablation)": "configs/graphwavenet_topology_only.yaml",
    "Ungaged SC2": "configs/ungaged/SC2.yaml",
    "Ungaged WOOD1": "configs/ungaged/WOOD1.yaml",
    "Ungaged EC12": "configs/ungaged/EC12.yaml",
    "Ungaged BK1": "configs/ungaged/BK1.yaml",
}
#: Retrainings of the full architecture (used for across-model spread).
SAME_ARCHITECTURE = ["Full (baseline)", "Ungaged SC2", "Ungaged WOOD1", "Ungaged EC12", "Ungaged BK1"]
#: (regime, group) pairs of the supplementary figure.
RELATIONSHIPS = [("storm", "Q (cfs)"), ("storm", "SSP"), ("storm", "Storm Antecedent"),
                 ("baseflow", "Seasonal+ Antecedent")]
NODE_SETS = {"figure (15)": list(ATTRIBUTION_NODES), "all gaged (28)": list(IG_TARGET_NODES)}


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS of y on x with standard errors, 95 % CI, residual SE and Spearman rho."""
    n = len(x)
    fit = stats.linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    t = stats.t.ppf(0.975, n - 2)
    rho = stats.spearmanr(x, y)
    return {
        "n": n, "slope": fit.slope, "slope_se": fit.stderr,
        "slope_ci_low": fit.slope - t * fit.stderr, "slope_ci_high": fit.slope + t * fit.stderr,
        "intercept": fit.intercept, "intercept_se": fit.intercept_stderr,
        "r2": fit.rvalue ** 2, "p": fit.pvalue,
        "residual_se": float(np.sqrt(np.sum(resid ** 2) / (n - 2))),
        "spearman_rho": rho.correlation, "spearman_p": rho.pvalue,
    }


def model_fractions(cfg, G, feature_names, node_ids, storm) -> dict | None:
    dates = G.nodes[node_ids[0]]["x"].index
    test_dates = dates[dates >= pd.Timestamp(cfg.data.test_split_date, tz=dates.tz)]
    ig = load_full_test_ig(paths.interpretation_dir(cfg.run_name), node_ids, test_dates, cfg.data.num_timesteps_in)
    missing = [n for n in IG_TARGET_NODES if n not in ig]
    if missing:
        print(f"[{cfg.run_name}] IG cache missing for {len(missing)} node(s): {missing}")
    return pattr.regime_fractions_by_node(ig, feature_names, storm) if ig else None


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = paths.ensure_dir(paths.ANALYSES_DIR / "attribution_stability")
    cfgs = {k: load_config(v) for k, v in MODELS.items()}
    G = load_graph(cfgs["Full (baseline)"].graph_path)
    node_ids = sorted(G.nodes())
    feature_names = list(G.nodes[node_ids[0]]["x"].columns)
    storm = storm_mask(outlet_turbidity(G))

    fracs = {}
    for label, cfg in cfgs.items():
        f = model_fractions(cfg, G, feature_names, node_ids, storm)
        if f is None:
            print(f"[{label}] no IG cache; run scripts/12 for {cfg.run_name}")
            continue
        fracs[label] = f

    # ---- node shares (long) --------------------------------------------------
    shares = pd.DataFrame([{"model": m, "regime": reg, "node": n, "site": SITE_CODES.get(n, str(n)),
                            "drainage_area_km2": drainage_area(n), "group": g, "share": v}
                           for m, f in fracs.items() for reg, by_node in f.items()
                           for n, groups in by_node.items() for g, v in groups.items()])
    shares.to_csv(out / "node_shares.csv", index=False)

    # ---- regressions --------------------------------------------------------
    rows = []
    for m, f in fracs.items():
        for set_name, nodes in NODE_SETS.items():
            for regime, group in RELATIONSHIPS:
                ns = [n for n in nodes if n in f[regime]]
                x = np.log([drainage_area(n) for n in ns])
                y = np.array([f[regime][n][group] for n in ns])
                rows.append({"model": m, "node_set": set_name, "regime": regime, "group": group, **ols(x, y)})
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "regressions.csv", index=False)

    # ---- spread across retrainings of the full architecture -----------------
    same = reg[reg["model"].isin(SAME_ARCHITECTURE)]
    across = (same.groupby(["node_set", "regime", "group"])[["slope", "r2", "spearman_rho"]]
              .agg(["count", "mean", "std", "min", "max"]))
    across.columns = [f"{a}_{b}" for a, b in across.columns]
    across.reset_index().to_csv(out / "across_models.csv", index=False)

    s = shares[shares["model"].isin(SAME_ARCHITECTURE)]
    spread = s.groupby(["regime", "group", "node", "site"])["share"].agg(["count", "mean", "std"]).reset_index()
    spread["cv"] = spread["std"] / spread["mean"]
    spread.to_csv(out / "node_share_spread.csv", index=False)

    agree = []
    for (regime, group), (a, b) in itertools.product(RELATIONSHIPS, itertools.combinations(fracs, 2)):
        common = [n for n in IG_TARGET_NODES if n in fracs[a][regime] and n in fracs[b][regime]]
        rho = stats.spearmanr([fracs[a][regime][n][group] for n in common],
                              [fracs[b][regime][n][group] for n in common]).correlation
        agree.append({"regime": regime, "group": group, "model_a": a, "model_b": b, "n": len(common), "spearman_rho": rho})
    pd.DataFrame(agree).to_csv(out / "rank_agreement.csv", index=False)


    show = reg[reg["node_set"] == "figure (15)"][["model", "group", "slope", "slope_se", "r2", "residual_se", "p"]]
    print(show.round(4).to_string(index=False))
    print(f"\n[stability] wrote {out}")


if __name__ == "__main__":
    main()
