"""Learned-adjacency figures.

* :func:`figure_adjacency_chord` — Figure 5: (a) chord diagram of the strongest
  learned reach-to-reach connections, (b) sub-watershed map; both coloured by
  turbidity-sensor density
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from esopus.constants import TURBIDITY_NODES
from esopus.interpretation.adjacency import aggregate_to_reaches, reach_sensor_density
from esopus.plotting.style import fs, add_scalebar_and_north, draw_river_network, node_positions, panel_label, scatter_nodes

#: SFI reach -> sub-watershed polygon (``data/raw/geospatial/subwatersheds/<name>.shp``).
REACH_POLYGONS = {
    "esopus_creek_all": "Upper Esopus Creek Downstream",
    "esopus_lost_cove_bushnellsville": "Upper Esopus Creek Lost Cove to Bushnellsville",
    "esopus_upstream_lost_cove": "Upper Esopus Creek Upstream Lost Cove",
    "birch_creek": "Birch Creek",
    "bushnellsville_creek": "Bushnellsville Creek",
    "broadstreet_hollow": "Broad St Hollow",
    "beaver_kill": "Beaver Kill",
    "little_beaver_kill": "Little Beaver Kill",
    "woodland_creek": "Woodland Creek",
    "panther_kill": "Panther Kill",
    "stony_clove_creek": "Stony Clove",
    "warner_creek": "Warner Creek",
}
CHORD_NAMES = {"esopus_creek_all": "Esopus Creek", "little_beaver_kill": "Little Beaver"}
#: Map-label offsets (m, EPSG:5070) for sub-watersheds whose labels would overlap; drawn with a leader line.
LABEL_OFFSETS = {"stony_clove_creek": (1500, 2800), "broadstreet_hollow": (-600, 1500), "warner_creek": (2600, 1600), "beaver_kill": (1800, -1800)}
DENSITY_CMAP = "YlGnBu"


# ---------------------------------------------------------------------------
# Chord-diagram geometry
# ---------------------------------------------------------------------------

def _bezier(t_src: float, t_tgt: float, r: float = 0.96, r_ctrl: float = 0.30, n: int = 150) -> np.ndarray:
    p0 = np.array([r * np.cos(t_src), r * np.sin(t_src)])
    p3 = np.array([r * np.cos(t_tgt), r * np.sin(t_tgt)])
    p1, p2 = p0 * r_ctrl, p3 * r_ctrl
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _gradient_line(ax, pts, c0, c1, lw, alpha=0.8):
    c0, c1 = np.array(mcolors.to_rgba(c0)), np.array(mcolors.to_rgba(c1))
    for i in range(len(pts) - 1):
        f = i / (len(pts) - 2)
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=(1 - f) * c0 + f * c1, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=3)


def draw_chord(ax, R: np.ndarray, names: List[str], colors: List, top_n: int = 15,
               lw_range=(2.5, 9.0), gap_deg: float = 4.0) -> List[float]:
    """Chord diagram of the ``top_n`` largest off-diagonal entries of ``R``.

    Chords run from reach a (row) to reach b (column) with an arrowhead at b;
    width is proportional to the summed learned weight. Arc length reflects each
    reach's total connection weight (minimum 5 %). Returns the plotted weights and
    the reaches shown.
    """
    pairs = sorted(((a, b, R[a, b]) for a in range(len(R)) for b in range(len(R)) if a != b and R[a, b] > 0),
                   key=lambda p: p[2], reverse=True)[:top_n]
    active = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    strength = np.array([R[r].sum() + R[:, r].sum() if r in active else 0.0 for r in range(len(R))])
    frac = np.where(strength > 0, np.maximum(strength / strength.sum(), 0.05), 0.0)
    frac /= frac.sum()
    gap = np.radians(gap_deg)
    seg = frac * (2 * np.pi - len(active) * gap)
    start, end, ang = {}, {}, np.pi / 2
    for r in active:
        start[r], end[r] = ang, ang + seg[r]
        ang = end[r] + gap

    conns = defaultdict(list)
    for ci, (a, b, _) in enumerate(pairs):
        conns[a].append(ci)
        conns[b].append(ci)
    attach = {}
    for r in active:
        cs, width = conns[r], end[r] - start[r]
        for k, ci in enumerate(cs):
            attach[(r, ci)] = (start[r] + end[r]) / 2 if len(cs) == 1 else start[r] + width * (0.15 + 0.70 * k / (len(cs) - 1))

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    for r in active:
        tt = np.linspace(start[r], end[r], 80)
        ax.plot(np.cos(tt), np.sin(tt), color=colors[r], lw=7, solid_capstyle="butt", zorder=4)
        tm = (start[r] + end[r]) / 2
        ax.plot([np.cos(tm), 1.15 * np.cos(tm)], [np.sin(tm), 1.15 * np.sin(tm)], color=colors[r], lw=0.6, zorder=4)
        ax.text(1.18 * np.cos(tm), 1.18 * np.sin(tm), names[r], ha="left" if np.cos(tm) >= 0 else "right",
                va="center", fontsize=fs(12), fontweight="bold", path_effects=halo, zorder=6)

    wmax = pairs[0][2] if pairs else 1.0
    for ci, (a, b, wgt) in enumerate(pairs):
        pts = _bezier(attach[(a, ci)], attach[(b, ci)])
        _gradient_line(ax, pts, colors[a], colors[b], lw_range[0] + (lw_range[1] - lw_range[0]) * wgt / wmax)
        n = len(pts)
        ax.add_patch(FancyArrowPatch(pts[int(0.82 * n)], pts[int(0.88 * n)], arrowstyle="-|>",
                                     mutation_scale=22, color="#333333", lw=0, zorder=6))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return [p[2] for p in pairs], active


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

def _load_polygons(subwatershed_dir: Path):
    import geopandas as gpd
    import pyproj

    os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())  # some conda envs miss this
    polys = {}
    for reach, stem in REACH_POLYGONS.items():
        f = Path(subwatershed_dir) / f"{stem}.shp"
        if f.exists():
            polys[reach] = gpd.read_file(f).to_crs(5070)
    return polys


def figure_adjacency_chord(G, A: np.ndarray, node_ids: List[int], sfi: pd.DataFrame,
                           subwatershed_dir: Optional[Path] = None, top_n: int = 15) -> plt.Figure:
    """Figure 5. Polygons are drawn only if ``subwatershed_dir`` holds the shapefiles."""
    density = reach_sensor_density(sfi)
    norm = mcolors.Normalize(0, float(np.nanpercentile(density, 95)))
    cmap = plt.get_cmap(DENSITY_CMAP)
    colors = [cmap(norm(v)) for v in density]
    names = [CHORD_NAMES.get(s, d.split("\n")[0]) for s, d in zip(sfi["stream_name"], sfi["display_name"])]

    fig = plt.figure(figsize=(14, 20))
    ax_a = fig.add_axes([0.02, 0.52, 0.80, 0.46])
    ax_b = fig.add_axes([0.02, 0.02, 0.80, 0.47])
    cax = fig.add_axes([0.88, 0.25, 0.025, 0.60])

    R = aggregate_to_reaches(A, node_ids, sfi)
    weights, shown = draw_chord(ax_a, R, names, colors, top_n=top_n)
    wq = np.quantile(weights, [0.0, 0.5, 1.0])
    leg = ax_a.legend(handles=[Line2D([], [], color="#888", lw=2.5 + 6.5 * w / max(weights), label=f"{w:.1f}") for w in wq],
                title="Learned connection\nweight", loc="lower right", bbox_to_anchor=(1.08, -0.195), fontsize=fs(10),
                title_fontsize=fs(10), frameon=True, fancybox=True, framealpha=1.0, edgecolor="#555555",
                facecolor="#F4F4F4", borderpad=0.8, handlelength=2.5)
    leg.get_title().set_fontweight("bold")
    leg.get_frame().set_linewidth(1.2)
    panel_label(ax_a, "(a)", x=0.08, y=0.95, size=fs(20))

    # ---- (b) map ------------------------------------------------------------
    pos = node_positions(G)
    if subwatershed_dir is not None:
        polys = _load_polygons(subwatershed_dir)
        # only the sub-watersheds that appear in the chord diagram (panel a)
        shown_names = {sfi.loc[r, "stream_name"] for r in shown}
        polys = {k: v for k, v in polys.items() if k in shown_names}
        order = sorted(polys, key=lambda r: polys[r].area.sum(), reverse=True)  # big first, nested ones on top
        r_index = {s: i for i, s in enumerate(sfi["stream_name"])}
        for reach in order:
            c = colors[r_index[reach]]
            polys[reach].plot(ax=ax_b, facecolor=mcolors.to_rgba(c, 0.30), edgecolor=c, lw=1.2, zorder=1)
            cx, cy = polys[reach].geometry.union_all().representative_point().coords[0]
            dx, dy = LABEL_OFFSETS.get(reach, (0, 0))
            ax_b.annotate(sfi.loc[r_index[reach], "display_name"], xy=(cx, cy), xytext=(cx + dx, cy + dy),
                          ha="center", va="center", fontsize=fs(8), fontweight="bold", zorder=8,
                          path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                          arrowprops=dict(arrowstyle="-", lw=0.8, color="#333") if (dx or dy) else None)
    draw_river_network(ax_b, G, pos, color="#777777", width=2.0, alpha=0.6, arrowsize=10)
    scatter_nodes(ax_b, pos, [n for n in G.nodes if n not in TURBIDITY_NODES], s=75, c="#BBBBBB", marker="^",
                  edgecolors="#333", linewidths=1.0, zorder=5)
    scatter_nodes(ax_b, pos, [n for n in G.nodes if n in TURBIDITY_NODES], s=60, c="#222222",
                  edgecolors="white", linewidths=1.0, zorder=6)
    add_scalebar_and_north(ax_b, loc=(0.82, 0.04))
    ax_b.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=9, mfc="#222222", mec="white", label="Turbidity sensors"),
        Line2D([], [], ls="", marker="^", ms=10, mfc="#BBBBBB", mec="#333", label="Ungaged nodes"),
        Line2D([], [], color="#777777", marker=">", label="Flow direction (NHD)"),
    ], loc="lower left", fontsize=fs(12))
    panel_label(ax_b, "(b)", x=0.08, y=0.95, size=fs(20))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max").set_label(
        "Sensor density (sensors / km)", fontsize=fs(13))
    return fig
