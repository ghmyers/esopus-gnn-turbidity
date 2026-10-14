"""Integrated Gradients figures.

* :func:`regime_fractions_by_node` — feature-group shares of mean |IG| during
  turbidity events and non-event periods for each target node
* :func:`figure_attribution_vs_da`  — Figure S7: group shares vs. ln(drainage area)
  with OLS fits
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from esopus.interpretation.feature_groups import regime_group_fractions
from esopus.metadata import drainage_area
from esopus.plotting.style import GROUP_LABELS, GROUP_PLOT_COLORS, fs


def regime_fractions_by_node(ig: Dict[int, dict], feature_names, storm: pd.Series) -> Dict[str, Dict[int, dict]]:
    """``{"storm": {node: {group: share}}, "baseflow": {...}}`` from full-test IG caches."""
    out = {"storm": {}, "baseflow": {}}
    for regime, flag in (("storm", True), ("baseflow", False)):
        for n, r in ig.items():
            f = regime_group_fractions(r["target_dates"], r["attrs_full_all"], feature_names, storm, flag)
            if f:
                out[regime][n] = f
    return out


def figure_attribution_vs_da(regime_fracs, nodes: Sequence[int]) -> plt.Figure:
    """Group shares vs. ln(drainage area) with OLS fits (supplementary).

    (a) storm periods: Q, SSP and short-term antecedent Q; (b) non-storm
    periods: long-term antecedent Q. Regression statistics are listed under
    each panel, group names in the group colour.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 13))
    panels = [(axes[0], "storm", ["Q (cfs)", "SSP", "Storm Antecedent"], "(a) Turbidity Events", "upper right"),
              (axes[1], "baseflow", ["Seasonal+ Antecedent"], "(b) Non-Events", "upper left")]
    markers = {"Q (cfs)": "o", "SSP": "s", "Storm Antecedent": "^", "Seasonal+ Antecedent": "p"}
    for ax, regime, groups, title, legend_loc in panels:
        fr = regime_fracs[regime]
        ns = [n for n in nodes if n in fr]
        x = np.log([drainage_area(n) for n in ns])
        stats = []
        for g in groups:
            y = np.array([fr[n][g] for n in ns])
            fit = linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.scatter(x, y, marker=markers[g], s=110, color=GROUP_PLOT_COLORS[g], label=GROUP_LABELS[g], zorder=3)
            ax.plot(xx, fit.intercept + fit.slope * xx, "--", lw=2.5, color=GROUP_PLOT_COLORS[g])
            stats.append((g, f"y = {fit.slope:.3f}x + {fit.intercept:.3f}   R² = {fit.rvalue ** 2:.3f}   p = {fit.pvalue:.2e}"))
        ax.set_title(title, fontsize=fs(12))
        ax.set_xlabel("Log(Drainage Area [km²])")
        ax.set_ylabel("Mean |IG| Attribution")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        if regime == "storm":
            ax.set_ylim(0, 0.6)  # as in the original figure: leaves room for the legend above the data
        ax.legend(loc=legend_loc, fontsize=fs(8.5))
        for k, (g, text) in enumerate(stats):  # coloured group name, statistics aligned in a column
            yk = -0.24 - 0.085 * k
            ax.text(0.0, yk, GROUP_LABELS[g], transform=ax.transAxes, va="top", fontsize=fs(8),
                    fontweight="bold", color=GROUP_PLOT_COLORS[g])
            ax.text(0.40, yk, text, transform=ax.transAxes, va="top", fontsize=fs(8))
    fig.subplots_adjust(hspace=0.75, bottom=0.12, top=0.96)
    return fig
