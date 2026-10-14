"""Discharge-filling regressions (Figures S1-S2).

One panel per gaged node: hourly measured discharge against the reference
gage (Stony Clove at Chichester, node 30, for S1; Esopus Creek at Coldbrook,
node 33, for S2) with the OLS fit used to fill that node's gaps. Panels are
ordered by R². Inputs are written by scripts/02_build_graph.py.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esopus.constants import SITE_CODES, site_label
from esopus.plotting.style import CB_OBSERVED, CB_PREDICTED, fs


def figure_discharge_regressions(measured: pd.DataFrame, regressions: pd.DataFrame, reference: int,
                                 min_overlap: int = 30, ncols: int = 4) -> plt.Figure:
    """Grid of node-vs-reference discharge scatter plots with OLS lines.

    ``measured``: hourly discharge before filling (columns = node IDs);
    ``regressions``: :func:`esopus.graph.discharge_fill.fit_regressions` output.
    """
    reg = regressions[regressions["n_overlap"] >= min_overlap].sort_values("r_squared", ascending=False)
    nrows = math.ceil(len(reg) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    ref = measured[str(reference)] if str(reference) in measured else measured[reference]
    ref_name = site_label(reference)
    for ax, (_, r) in zip(axes, reg.iterrows()):
        node = int(r["node"])
        y = measured[str(node)] if str(node) in measured else measured[node]
        ok = ref.notna() & y.notna()
        ax.scatter(ref[ok], y[ok], s=4, alpha=0.3, color=CB_OBSERVED, rasterized=True)
        xl = np.array([ref[ok].min(), ref[ok].max()])
        ax.plot(xl, r["slope"] * xl + r["intercept"], color=CB_PREDICTED, lw=2)
        ax.set_title(f"{SITE_CODES.get(node, f'Node {node}')} (node {node})\n"
                     f"R² = {r['r_squared']:.3f}   n = {int(r['n_overlap']):,}", fontsize=fs(8.5))
        ax.set_xlabel(f"{ref_name} Q (cfs)", fontsize=fs(7))
        ax.set_ylabel("Q (cfs)", fontsize=fs(7))
        ax.tick_params(labelsize=fs(6.5))
    for ax in axes[len(reg):]:
        ax.set_visible(False)
    fig.suptitle(f"Discharge regressions vs. {ref_name}", fontsize=fs(11), fontweight="bold")
    return fig
