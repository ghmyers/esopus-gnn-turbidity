"""Integrated Gradients (IG) attribution of turbidity predictions.

IG (Sundararajan et al., 2017) attributes a prediction F(x) to every input
element by integrating gradients along the straight path from a baseline x'
to the input x::

    IG_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a (x - x')) / dx_i  da

Here F is the model's scaled turbidity prediction at one *target node*, and
the input is one window of every node's features, so the attribution tensor
has shape (N nodes, F features, T_in hours). This shows which upstream nodes,
which features and which hours drove the prediction at the target.

Baseline
--------
The mean feature vector of 200 randomly drawn training hours (per node and
feature), repeated over the T_in window: a "typical, unchanging" state.
A zero baseline would be meaningless because 0.0 is the missing-data sentinel.

Integration
-----------
Captum's ``IntegratedGradients`` with Gauss-Legendre quadrature, 50 steps,
for one window per day over the test period. The published caches evaluate
one integration step per forward pass (``internal_batch_size=1``); batching
the steps is ~15x faster on a GPU and changes group shares by < 0.001.

Cached outputs (``outputs/interpretation/<run_name>/``)
-------------------------------------------------------
``background_ig.npz``                    baseline (1, N*F*T_in)
``node_<id>/ig_full_test.npz``           one window per day over the test period
    attrs_full_all (S, N, F, T_in), attrs_nf_all (S, N, F) = mean |IG| over time,
    attrs_full_mean, attrs_nf_mean, target_dates (S,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from esopus.data.dataset import window_target_dates


class _TargetNodeWrapper(nn.Module):
    """Flat input (B, N*F*T_in) -> scaled prediction at one node (B, 1).

    Captum needs a single input tensor, so the window is flattened and the
    graph tensors are held fixed inside the wrapper.
    """

    def __init__(self, model, edge_index, edge_weight, n_nodes, n_features, t_in, target_idx):
        super().__init__()
        self.model, self.edge_index, self.edge_weight = model, edge_index, edge_weight
        self.shape = (n_nodes, n_features, t_in)
        self.target_idx = target_idx

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(x_flat.shape[0], *self.shape).permute(0, 3, 1, 2)  # (B, T, N, F)
        return self.model(x, self.edge_index, self.edge_weight)[:, self.target_idx, 0].unsqueeze(1)


def build_ig_background(
    X_train_scaled: np.ndarray, t_in: int, n_samples: int = 200, seed: int = 0, device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Baseline window (1, N*F*T_in): mean of ``n_samples`` random training hours."""
    rng = np.random.default_rng(seed)
    t_idx = rng.integers(0, X_train_scaled.shape[2], size=n_samples)
    mean_nf = X_train_scaled[:, :, t_idx].mean(axis=2)                      # (N, F)
    window = np.repeat(mean_nf[:, :, np.newaxis], t_in, axis=2)              # (N, F, T_in)
    return torch.from_numpy(window.reshape(1, -1).astype(np.float32)).to(device)


def compute_ig(
    model: nn.Module,
    x_window: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    background: torch.Tensor,
    node_ids: List[int],
    target_node_id: int,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    internal_batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """IG for one scaled window ``x_window`` (N, F, T_in) w.r.t. one target node.

    ``internal_batch_size`` is the number of integration steps evaluated per
    forward pass. The published caches used 1; larger values are much faster
    on a GPU and agree to floating-point precision.

    Returns ``attrs_full`` (N, F, T_in) and ``attrs_nf`` = mean |IG| over time (N, F).
    """
    model.eval()
    N, F, T = x_window.shape
    wrapper = _TargetNodeWrapper(model, edge_index, edge_weight, N, F, T, node_ids.index(target_node_id)).to(device).eval()
    x = torch.from_numpy(x_window.reshape(1, -1).astype(np.float32)).to(device).requires_grad_(True)
    attrs = IntegratedGradients(wrapper).attribute(
        inputs=x, baselines=background, n_steps=n_steps, target=0,
        method="gausslegendre", internal_batch_size=internal_batch_size,
    )
    attrs_full = attrs.detach().cpu().numpy().reshape(N, F, T)
    return attrs_full, np.abs(attrs_full).mean(axis=2)


# ---------------------------------------------------------------------------
# Batch drivers with on-disk caching
# ---------------------------------------------------------------------------

def daily_test_window_starts(n_test_steps: int, t_in: int, stride: int = 24) -> List[int]:
    """Start positions of one window per ``stride`` hours across the test split."""
    return list(range(0, n_test_steps - t_in + 1, stride))


def full_test_ig(
    model, X_test_scaled: np.ndarray, test_dates: pd.DatetimeIndex, edge_index, edge_weight,
    background: torch.Tensor, node_ids: List[int], target_node_id: int, cache_dir: Path,
    t_in: int, stride: int = 24, n_steps: int = 50, device: torch.device = torch.device("cpu"),
    internal_batch_size: int = 1,
) -> Dict[str, np.ndarray]:
    """IG for one test window per day at ``target_node_id``; cached as ``ig_full_test.npz``."""
    path = Path(cache_dir) / f"node_{target_node_id}" / "ig_full_test.npz"
    starts = daily_test_window_starts(X_test_scaled.shape[2], t_in, stride)
    dates = window_target_dates(test_dates, t_in, starts=starts)
    if path.exists():
        res = dict(np.load(path, allow_pickle=True))
    else:
        af, anf = zip(*(
            compute_ig(model, X_test_scaled[:, :, s:s + t_in], edge_index, edge_weight, background,
                       node_ids, target_node_id, n_steps, device, internal_batch_size)
            for s in starts
        ))
        af, anf = np.stack(af), np.stack(anf)
        res = {"attrs_full_all": af, "attrs_nf_all": anf,
               "attrs_full_mean": np.abs(af).mean(axis=0), "attrs_nf_mean": np.abs(anf).mean(axis=0)}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **res, target_dates=dates.astype("int64"))
    res["target_dates"] = dates  # always recomputed so archived caches get correct timestamps
    return res


def load_or_build_background(cache_dir: Path, X_train_scaled: np.ndarray, t_in: int, seed: int, device) -> torch.Tensor:
    """Load ``background_ig.npz`` from ``cache_dir`` or build and save it."""
    path = Path(cache_dir) / "background_ig.npz"
    if path.exists():
        return torch.from_numpy(np.load(path)["background"]).to(device)
    bg = build_ig_background(X_train_scaled, t_in=t_in, n_samples=200, seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, background=bg.cpu().numpy())
    return bg


def load_full_test_ig(cache_dir: Path, node_ids: Sequence[int], test_dates: pd.DatetimeIndex, t_in: int, stride: int = 24) -> Dict[int, Dict]:
    """Load every cached ``ig_full_test.npz`` under ``cache_dir`` (no model needed)."""
    out = {}
    for nid in node_ids:
        path = Path(cache_dir) / f"node_{nid}" / "ig_full_test.npz"
        if path.exists():
            res = dict(np.load(path, allow_pickle=True))
            res["target_dates"] = window_target_dates(
                test_dates, t_in, starts=daily_test_window_starts(len(test_dates), t_in, stride))
            out[nid] = res
    return out
