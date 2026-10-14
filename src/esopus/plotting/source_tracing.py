"""Source-tracing figures.

* :func:`figure_source_tracing` — Figure 4: observed / predicted load maps and
  per-tributary load shares (full record and storm-event mean)
* :func:`figure_load_bars`      — measured vs. predicted shares, optionally
  highlighting one node (ungaged-validation folds)
"""

from __future__ import annotations

import textwrap
from typing import Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from esopus.constants import TURBIDITY_NODES, site_label
from esopus.plotting.style import (
    fs,
    CB_OBSERVED, CB_PREDICTED, COLOR_OBSERVED, COLOR_PREDICTED, add_scalebar_and_north, draw_river_network, node_positions,
    panel_label, scatter_nodes,
)


def _wrapped_label(node: int) -> str:
    return "\n".join(textwrap.wrap(site_label(node), 11))


def _load_map(ax, G, fractions: Dict[int, float], norm, cmap, title: str) -> None:
    pos = node_positions(G)
    draw_river_network(ax, G, pos, color="#444444", width=1.0)
    # Triangles: no turbidity sensor. Hollow circles: turbidity sensor but not a source tributary.
    scatter_nodes(ax, pos, [n for n in G.nodes if n not in TURBIDITY_NODES], s=55, c="white", marker="^",
                  edgecolors="#555555", linewidths=0.8, zorder=4)
    scatter_nodes(ax, pos, [n for n in TURBIDITY_NODES if n not in fractions], s=45, c="white",
                  edgecolors="#555555", linewidths=0.8, zorder=5)
    nodes = sorted(fractions, key=fractions.get)
    sizes = [40 + 360 * fractions[n] / norm.vmax for n in nodes]
    ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes], s=sizes,
               c=[cmap(norm(fractions[n])) for n in nodes], edgecolors="#222222", linewidths=0.6, zorder=6)
    add_scalebar_and_north(ax, loc=(0.78, 0.04))
    ax.set_title(title, fontsize=fs(13))


def figure_source_tracing(G, fractions: pd.DataFrame, stats: Dict[str, float], n_bars: int = 10) -> plt.Figure:
    """Figure 4.

    ``fractions`` is ``load_fractions.csv`` from scripts/13 (columns
    measured_full, predicted_full, measured_events, predicted_events);
    ``stats`` its ``summary.json``.
    """
    fr = fractions.set_index("node")
    vmax = float(np.nanmax(fr[["measured_full", "predicted_full"]].to_numpy()))
    norm, cmap = mcolors.Normalize(0, vmax), plt.get_cmap("Reds")

    fig = plt.figure(figsize=(21, 14.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], height_ratios=[1.15, 1], hspace=0.22, wspace=0.05)
    ax_a, ax_b, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, :2])

    _load_map(ax_a, G, fr["measured_full"].dropna().to_dict(), norm, cmap, "Observed loads")
    _load_map(ax_b, G, fr["predicted_full"].dropna().to_dict(), norm, cmap, "Predicted loads")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label("Relative load contribution", fontsize=fs(12))
    ax_a.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=10, mfc="#C0392B", mec="#222222", label="Gaged node (turbidity\nand streamflow)"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec="#555555", label="Other turbidity sensor"),
        Line2D([], [], ls="", marker="^", ms=9, mfc="white", mec="#555555", label="Ungaged node"),
    ], loc="lower left", bbox_to_anchor=(-0.17, -0.10), fontsize=fs(9.5), borderpad=0.3, labelspacing=0.3,
        handletextpad=0.3)
    panel_label(ax_a, "(a)", x=0.0)
    panel_label(ax_b, "(b)", x=0.0)

    # ---- bars ---------------------------------------------------------------
    nodes = fr["measured_full"].sort_values(ascending=False).index[:n_bars]
    x, w = np.arange(len(nodes)), 0.38
    pct = fr.loc[nodes] * 100
    ax_c.bar(x - w / 2, pct["measured_full"], w, color=CB_OBSERVED, alpha=0.7, label="Observed (full record)", zorder=3)
    ax_c.bar(x + w / 2, pct["predicted_full"], w, color=CB_PREDICTED, alpha=0.7, label="Predicted (full record)", zorder=3)
    ax_c.bar(x - w / 2, pct["measured_events"], w, facecolor="none", edgecolor=CB_OBSERVED, lw=1.6,
             label="Observed (turbidity events)", zorder=4)
    ax_c.bar(x + w / 2, pct["predicted_events"], w, facecolor="none", edgecolor=CB_PREDICTED, lw=1.6,
             label="Predicted (turbidity events)", zorder=4)
    ax_c.set_xticks(x, [_wrapped_label(n) for n in nodes], fontsize=fs(7))
    ax_c.set_xlabel("Node", fontsize=fs(12))
    ax_c.set_ylabel("Relative load contribution (%)", fontsize=fs(12))
    ax_c.xaxis.grid(False)
    ax_c.legend(ncol=2, fontsize=fs(11), loc="upper right")
    ax_c.text(0.5, 1.03,
              f"Full record:  RMSE={stats['full_rmse_pct'] / 100:.3f}  MAE={stats['full_mae_pct'] / 100:.3f}  "
              f"Median bias={stats['full_median_bias_pct']:+.1f}%\n"
              f"Turbidity events:  RMSE={stats['event_mean_rmse_pct'] / 100:.3f}  MAE={stats['event_mean_mae_pct'] / 100:.3f}  "
              f"Median bias={stats['event_mean_median_bias_pct']:+.1f}%",
              transform=ax_c.transAxes, ha="center", va="bottom", fontsize=fs(12), color="#444",
              bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
    panel_label(ax_c, "(c)", x=-0.04, y=1.05)
    return fig


def figure_load_bars(measured: Dict[int, float], predicted: Dict[int, float], highlight: Optional[int] = None,
                     title: str = "", baseline: Optional[Dict[int, float]] = None) -> plt.Figure:
    """Measured vs. predicted full-record load shares (optionally a third, baseline series)."""
    nodes = sorted([n for n in measured if n in predicted], key=measured.get, reverse=True)
    x = np.arange(len(nodes))
    series = [("Observed", measured, COLOR_OBSERVED), ("Predicted", predicted, COLOR_PREDICTED)]
    if baseline is not None:
        series.append(("Full model", baseline, "#378ADD"))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (lab, d, c) in enumerate(series):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, [d.get(n, np.nan) * 100 for n in nodes], w,
                      color=c, alpha=0.85, label=lab, zorder=3)
        if highlight in nodes:
            bars[nodes.index(highlight)].set_edgecolor("black")
            bars[nodes.index(highlight)].set_linewidth(2)
    ax.set_xticks(x, [_wrapped_label(n) for n in nodes], fontsize=fs(8))
    if highlight in nodes:
        ax.get_xticklabels()[nodes.index(highlight)].set_fontweight("bold")
    ax.set_ylabel("Relative load contribution (%)")
    ax.xaxis.grid(False)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig
