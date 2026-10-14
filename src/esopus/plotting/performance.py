"""Prediction-skill figures.

* :func:`figure_performance`        — Figure 3: KGE map + test-period series at three sites
* :func:`figure_node_metric_bars`   — Figure S4: per-node test metrics
* :func:`figure_metrics_vs_da`      — Figure S5: KGE/NSE/MAE/RMSE vs. ln(drainage area)
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress

from esopus.constants import KGE_MEAN_BENCHMARK, SITE_CODES, TURBIDITY_NODES, site_label
from esopus.evaluation import PredictionSet
from esopus.metadata import drainage_area
from esopus.plotting.style import (
    fs,
    CB_OBSERVED, CB_PREDICTED, COLOR_NO_METRIC, COLOR_UNGAGED,
    add_scalebar_and_north, draw_river_network, node_positions, panel_label, scatter_nodes,
)

METRIC_BOX = "KGE = {kge:.2f}   NSE = {nse:.2f}\nMAE = {mae:.1f} FNU   RMSE = {rmse:.1f} FNU"


def _timeseries_panel(ax, preds: PredictionSet, metrics: pd.DataFrame, node: int, split: str = "test",
                      legend: bool = False, xlabel: bool = True) -> None:
    sub = preds.subset(split)
    ax.plot(sub.dates, sub.series(node, "observed").values, color=CB_OBSERVED, lw=2.4, label="Observed", zorder=3)
    ax.plot(sub.dates, sub.series(node, "predicted").values, color=CB_PREDICTED, lw=1.92, ls="--",
            label="Predicted", zorder=4)
    if node in metrics.index:
        ax.text(0.99, 0.97, METRIC_BOX.format(**metrics.loc[node]), transform=ax.transAxes, ha="right", va="top",
                fontsize=fs(9), bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.95), zorder=6)
    ax.set_title(site_label(node), fontsize=fs(11))
    ax.set_ylabel("Turbidity (FNU)")
    ax.xaxis.grid(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if xlabel:
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.tick_params(labelbottom=False)
    if legend:
        ax.legend(loc="upper left")


def figure_performance(G, preds: PredictionSet, metrics: pd.DataFrame, series_nodes: Sequence[int] = (30, 29, 27)) -> plt.Figure:
    """Figure 3 — (a) test-period KGE at every gaged node; (b–d) test-period series.

    ``metrics`` are the test-period per-node metrics (``per_node_metrics.csv``).
    """
    # Layout of the submitted figure: large map top-left with its legend and a
    # horizontal KGE colour bar underneath, (d) below the map, (b) and (c)
    # stacked on the right, offset downward.
    fig = plt.figure(figsize=(21, 13.5))
    ax_map = fig.add_axes([0.0, 0.43, 0.54, 0.56])
    ax_b = fig.add_axes([0.585, 0.635, 0.405, 0.255])
    ax_c = fig.add_axes([0.585, 0.235, 0.405, 0.255])
    ax_d = fig.add_axes([0.06, 0.075, 0.45, 0.215])

    pos = node_positions(G)
    draw_river_network(ax_map, G, pos, color="#444444")
    with_kge = [n for n in TURBIDITY_NODES if n in metrics.index and np.isfinite(metrics.loc[n, "kge"])]
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=KGE_MEAN_BENCHMARK, vmax=1.0)
    cmap = plt.get_cmap("RdYlBu")  # colour-blind-safe diverging map (ColorBrewer)
    scatter_nodes(ax_map, pos, [n for n in G.nodes if n not in TURBIDITY_NODES],
                  s=70, c=COLOR_UNGAGED, marker="^", edgecolors="#333333", linewidths=0.5, zorder=4)
    scatter_nodes(ax_map, pos, [n for n in TURBIDITY_NODES if n not in with_kge],
                  s=90, c=COLOR_NO_METRIC, edgecolors="white", linewidths=0.6, zorder=5)
    ax_map.scatter([pos[n][0] for n in with_kge], [pos[n][1] for n in with_kge],
                   c=[cmap(norm(metrics.loc[n, "kge"])) for n in with_kge], s=170, edgecolors="white",
                   linewidths=0.8, zorder=6)
    for n, lab in zip(series_nodes, ("(b)", "(c)", "(d)")):
        ax_map.scatter(*pos[n], s=480, facecolors="none", edgecolors="#1f1f1f", linewidths=1.6, zorder=7)
        ax_map.annotate(lab, xy=pos[n], xytext=(-55, 6), textcoords="offset points", fontsize=fs(10),
                        fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.8), zorder=8)
    add_scalebar_and_north(ax_map)
    ax_map.legend(handles=[
        Line2D([], [], color="#444444", marker=">", label="Flow direction (NHD)"),
        Line2D([], [], ls="", marker="o", ms=10, mfc=cmap(0.9), mec="white", label="Gaged nodes (colored by KGE)"),
        Line2D([], [], ls="", marker="^", ms=8, mfc=COLOR_UNGAGED, mec="#333333", label="Ungaged nodes"),
    ], loc="upper left", bbox_to_anchor=(0.055, 0.425), bbox_transform=fig.transFigure, fontsize=fs(8.5))
    cax = fig.add_axes([0.29, 0.385, 0.235, 0.02])  # horizontal colour bar beside the legend
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("KGE")
    panel_label(ax_map, "(a)", x=0.07, y=0.98)

    _timeseries_panel(ax_b, preds, metrics, series_nodes[0], legend=True, xlabel=False)
    _timeseries_panel(ax_c, preds, metrics, series_nodes[1])
    _timeseries_panel(ax_d, preds, metrics, series_nodes[2])
    for ax, lab in ((ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        panel_label(ax, lab)
    return fig


def figure_metrics_vs_da(metrics: pd.DataFrame) -> plt.Figure:
    """KGE, NSE, MAE, RMSE (test) vs. ln(drainage area, km²) with OLS fits."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    x = np.log(np.array([drainage_area(n) for n in metrics.index]))
    for ax, metric, color, lab in zip(axes.flat, ["kge", "nse", "mae", "rmse"],
                                      ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], "abcd"):
        y = metrics[metric].to_numpy()
        fit = linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.scatter(x, y, color=color, alpha=0.75, s=36)
        ax.plot(xx, fit.intercept + fit.slope * xx, "k--", lw=1.8)
        ax.text(0.98, 0.03, f"y = {fit.slope:.3f}x + {fit.intercept:.3f}\nR² = {fit.rvalue ** 2:.3f}\np = {fit.pvalue:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=fs(7),
                bbox=dict(boxstyle="round", fc="white", ec="#333333", alpha=0.85), zorder=5)
        ax.set_title(metric.upper(), fontsize=fs(11))
        ax.set_xlabel("log(Drainage Area [km²])")
        ax.set_ylabel(metric.upper())
        panel_label(ax, f"({lab})", x=-0.02, y=1.03)
    fig.tight_layout()
    return fig


def figure_node_metric_bars(metrics: pd.DataFrame) -> plt.Figure:
    """Horizontal bars of per-node test KGE, NSE (best first) and MAE, RMSE (lowest first)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 20))
    labels = {n: SITE_CODES.get(n, str(n)) for n in metrics.index}
    for ax, metric, color, asc, lab in zip(axes.flat, ["kge", "nse", "mae", "rmse"],
                                           ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
                                           [False, False, True, True], "abcd"):
        s = metrics[metric].sort_values(ascending=asc)
        ax.barh(range(len(s)), s.values, color=color, alpha=0.8)
        ax.set_yticks(range(len(s)), [labels[n] for n in s.index], fontsize=fs(7))
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(metric.upper())
        ax.yaxis.grid(False)
        panel_label(ax, f"({lab})", x=-0.02, y=1.04)
    fig.suptitle("Per-Node Test Metrics", fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=3)
    return fig
