"""Step 20 — Make the manuscript and supporting-information figures.

Every figure is drawn from files written by the earlier steps; nothing is
re-trained or re-predicted here (the learned adjacency is read from the
checkpoint).

    fig3    KGE map + test-period prediction series       needs scripts/11 (full model)
    fig4    source tracing                                needs scripts/13 (full model)
    fig5    learned-adjacency chord diagram + map         needs outputs/models/gwn_full.pt + sub-watershed polygons
    figS1   discharge regressions vs. Stony Clove (SC2)   needs scripts/02
    figS2   discharge regressions vs. Coldbrook (EC1)     needs scripts/02
    figS4   per-node test metrics                         needs scripts/11
    figS5   test metrics vs. ln(drainage area)            needs scripts/11
    figS7   IG attribution vs. ln(drainage area)          needs scripts/12 (full model)

Figures 1 (study-area map) and 2 (architecture schematic) were drawn in GIS /
graphics software and are not generated here.

Usage
-----
    python scripts/20_make_figures.py                     # every figure
    python scripts/20_make_figures.py --figure fig4 figS7
    python scripts/20_make_figures.py --figure main       # or: supp
    ESOPUS_FONT_SCALE=1.0 python scripts/20_make_figures.py   # smaller fonts

Outputs: figures/main/fig*.png and figures/supplementary/figS*.png
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from esopus import paths
from esopus.config import load_config
from esopus.constants import ATTRIBUTION_NODES
from esopus.data import load_graph
from esopus.evaluation import PredictionSet
from esopus.graph.discharge_fill import COLDBROOK, STONY_CLOVE
from esopus.interpretation.adjacency import learned_adjacency, load_reaches
from esopus.interpretation.events import outlet_turbidity, storm_mask
from esopus.interpretation.integrated_grads import load_full_test_ig
from esopus.models import load_trained_model
from esopus.plotting import adjacency as padj
from esopus.plotting import attribution as pattr
from esopus.plotting import discharge as pdis
from esopus.plotting import performance as pperf
from esopus.plotting import source_tracing as pst
from esopus.plotting.style import apply_style, save_figure
from esopus.runtime import get_device

MAIN = ["fig3", "fig4", "fig5"]
SUPPLEMENTARY = ["figS1", "figS2", "figS4", "figS5", "figS7"]


class Context:
    """Inputs shared by several figures, loaded on first use."""

    def __init__(self):
        self.cfg = load_config("configs/graphwavenet_full.yaml")
        self._G = None

    @property
    def G(self):
        if self._G is None:
            self._G = load_graph(self.cfg.graph_path)
        return self._G

    @property
    def node_ids(self):
        return sorted(self.G.nodes())

    @property
    def metrics(self) -> pd.DataFrame:
        return pd.read_csv(paths.predictions_dir(self.cfg.run_name) / "per_node_metrics.csv", index_col="node")


def fig3(ctx):
    preds = PredictionSet.load(paths.predictions_dir(ctx.cfg.run_name))
    save_figure(pperf.figure_performance(ctx.G, preds, ctx.metrics), "fig3_performance_maps_timeseries")


def fig4(ctx):
    d = paths.ANALYSES_DIR / "source_tracing" / ctx.cfg.run_name
    fig = pst.figure_source_tracing(ctx.G, pd.read_csv(d / "load_fractions.csv"), json.loads((d / "summary.json").read_text()))
    save_figure(fig, "fig4_source_tracing")


def fig5(ctx):
    n_features = len(ctx.G.nodes[ctx.node_ids[0]]["x"].columns)
    model = load_trained_model(ctx.cfg, paths.model_path(ctx.cfg.run_name), n_features, len(ctx.node_ids), get_device("cpu"))
    shp = paths.RAW_DIR / "geospatial" / "subwatersheds"
    if not shp.exists():
        print(f"[fig5] {shp} not found: drawing the map without sub-watershed polygons")
        shp = None
    fig = padj.figure_adjacency_chord(ctx.G, learned_adjacency(model), ctx.node_ids, load_reaches(), shp)
    save_figure(fig, "fig5_adjacency_chord_sensor_density")


def _discharge_regressions(reference: int, name: str, figure_name: str) -> None:
    d = paths.ANALYSES_DIR / "discharge_regressions"
    measured = pd.read_csv(d / "measured_discharge.csv.gz", index_col="datetime")
    regressions = pd.read_csv(d / f"regressions_{name}.csv")
    save_figure(pdis.figure_discharge_regressions(measured, regressions, reference), figure_name, supplementary=True)


def figS1(ctx):
    _discharge_regressions(STONY_CLOVE, "stony_clove", "figS1_discharge_regressions_stony_clove")


def figS2(ctx):
    _discharge_regressions(COLDBROOK, "coldbrook", "figS2_discharge_regressions_coldbrook")


def figS4(ctx):
    save_figure(pperf.figure_node_metric_bars(ctx.metrics), "figS4_node_metrics_barchart", supplementary=True)


def figS5(ctx):
    save_figure(pperf.figure_metrics_vs_da(ctx.metrics), "figS5_metrics_vs_drainage_area", supplementary=True)


def figS7(ctx):
    dates = ctx.G.nodes[ctx.node_ids[0]]["x"].index
    test_dates = dates[dates >= pd.Timestamp(ctx.cfg.data.test_split_date, tz=dates.tz)]
    ig = load_full_test_ig(paths.interpretation_dir(ctx.cfg.run_name), ctx.node_ids, test_dates,
                           ctx.cfg.data.num_timesteps_in)
    feature_names = list(ctx.G.nodes[ctx.node_ids[0]]["x"].columns)
    fracs = pattr.regime_fractions_by_node(ig, feature_names, storm_mask(outlet_turbidity(ctx.G)))
    save_figure(pattr.figure_attribution_vs_da(fracs, ATTRIBUTION_NODES), "figS7_attribution_vs_drainage_area",
                supplementary=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", nargs="+", choices=MAIN + SUPPLEMENTARY + ["main", "supp", "all"], default=["all"])
    args = ap.parse_args()
    apply_style()
    ctx = Context()
    groups = {"main": MAIN, "supp": SUPPLEMENTARY, "all": MAIN + SUPPLEMENTARY}
    for name in dict.fromkeys(n for f in args.figure for n in groups.get(f, [f])):
        print(f"\n=== {name} ===")
        globals()[name](ctx)


if __name__ == "__main__":
    main()
