"""Shared figure style, colours and map helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import os

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx

from esopus import paths

#: Multiplier applied to every font size (rcParams and explicit sizes in the
#: figure functions). Override with the ESOPUS_FONT_SCALE environment variable.
FONT_SCALE = float(os.environ.get("ESOPUS_FONT_SCALE", 1.8))


def fs(size: float) -> float:
    """Scaled font size."""
    return size * FONT_SCALE


COLOR_OBSERVED = "#2EAF8A"   # teal: every observed series / measured quantity
COLOR_PREDICTED = "#C0392B"  # red: every model prediction
COLOR_UNGAGED = "#888780"
# Colour-blind-safe observed/predicted pair (Okabe-Ito blue / vermilion),
# used in Figures 3, 4 and S1-S2.
CB_OBSERVED = "#0072B2"
CB_PREDICTED = "#D55E00"
COLOR_NO_METRIC = "#CCCCCC"

#: Display names and colours of the IG feature groups (keys from
#: ``esopus.interpretation.feature_groups.GROUP_ORDER``).
GROUP_LABELS = {
    "SSP": "SSP",
    "Q (cfs)": "Q (cfs)",
    "Meteorological": "Meteorological",
    "Storm Antecedent": "Antecedent Q (short-term)",
    "Seasonal+ Antecedent": "Antecedent Q (long-term)",
}
GROUP_PLOT_COLORS = {
    "SSP": "#C0392B",
    "Q (cfs)": "#378ADD",
    "Meteorological": "#A9D6E5",
    "Storm Antecedent": "#8E44AD",
    "Seasonal+ Antecedent": "#D4A017",
}


def apply_style() -> None:
    """Matplotlib rcParams used for every figure."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": fs(9),
        "axes.titlesize": fs(10),
        "axes.titleweight": "bold",
        "axes.labelsize": fs(9),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": "#B4B2A9",
        "legend.fontsize": fs(8),
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#D3D1C7",
        "xtick.labelsize": fs(8),
        "ytick.labelsize": fs(8),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_figure(fig: plt.Figure, name: str, supplementary: bool = False, subdir: Optional[str] = None) -> Path:
    """Save to ``figures/main/<name>.png`` (or ``figures/supplementary/...``) and close."""
    out = paths.SUPP_FIGURES_DIR if supplementary else paths.MAIN_FIGURES_DIR
    if subdir:
        out = out / subdir
    path = paths.ensure_dir(out) / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def panel_label(ax, label: str, x: float = -0.06, y: float = 1.02, size: float = None) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size or fs(13), fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Network maps
# ---------------------------------------------------------------------------

def node_positions(G) -> Dict[int, Tuple[float, float]]:
    """Node -> (easting, northing) in EPSG:5070."""
    return {n: (d["pos"][0], d["pos"][-1]) for n, d in G.nodes(data=True)}


def draw_river_network(ax, G, pos=None, color: str = "#333333", width: float = 0.8, alpha: float = 0.8,
                       arrowsize: int = 8) -> None:
    """River-network edges with flow-direction arrowheads; turns the map frame off."""
    pos = pos or node_positions(G)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=color, width=width, alpha=alpha,
                           arrows=True, arrowsize=arrowsize, arrowstyle="-|>")
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_scalebar_and_north(ax, length_m: float = 5000, loc: Tuple[float, float] = (0.80, 0.06)) -> None:
    """5-km scale bar and north arrow in axes-fraction position ``loc``."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    halo = [pe.withStroke(linewidth=2, foreground="white")]
    ax.plot([bx, bx + length_m], [by, by], color="black", lw=3, solid_capstyle="butt", zorder=20)
    ax.text(bx + length_m / 2, by - 0.025 * (y1 - y0), f"{length_m / 1000:g} km", ha="center", va="top",
            fontsize=fs(9), path_effects=halo, zorder=20)
    nx_, ny_ = bx + length_m * 0.85, by + 0.07 * (y1 - y0)
    ax.annotate("", xy=(nx_, ny_ + 0.05 * (y1 - y0)), xytext=(nx_, ny_),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=20)
    ax.text(nx_, ny_ - 0.005 * (y1 - y0), "N", ha="center", va="top", fontsize=fs(8), fontweight="bold", zorder=20)


def scatter_nodes(ax, pos, nodes: Iterable[int], **kwargs) -> None:
    nodes = [n for n in nodes if n in pos]
    if nodes:
        ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes], **kwargs)
