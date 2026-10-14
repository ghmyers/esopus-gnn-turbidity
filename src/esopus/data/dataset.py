"""From the graph pickle on disk to model-ready sliding-window datasets.

Input
-----
A NetworkX ``DiGraph`` (edges point downstream) produced by
``scripts/03_engineer_features.py``. Every node carries:

* ``x``   : ``pd.DataFrame`` (T, F) of hourly input features (UTC index)
* ``y``   : ``pd.Series`` (T,) of hourly turbidity in FNU (NaN = not observed)
* ``pos`` : (easting, northing) in EPSG:5070, used only for maps

All nodes share one hourly timestamp index.

Steps performed by :func:`prepare_data`
---------------------------------------
1. Load the graph; optionally withhold one node's training turbidity
   (ungaged validation, see ``esopus.data.ungaged``).
2. Sort node IDs ascending. This order is used for the feature arrays, the
   target arrays *and* the adjacency matrix, which is what keeps node i in
   every tensor referring to the same physical location.
3. Stack features to ``X`` (N, F, T) and targets to ``y`` (N, T).
4. Split chronologically at ``cfg.data.test_split_date``.
5. Fit :class:`~esopus.data.transforms.TurbidityScaler` on the training split
   only (or reuse a supplied, already-fitted scaler) and scale both splits.
6. Wrap each split in a lazy :class:`SlidingWindowDataset`.
7. Build ``edge_index`` / ``edge_weight`` from the river-network adjacency.

Window/timestamp alignment
--------------------------
Window ``i`` uses inputs at positions ``i … i+T_in-1`` and predicts the target
at position ``i+T_in`` (one hour after the last input). Always use
:func:`window_target_dates` to timestamp predictions; an earlier version of
the analysis code stamped them one hour early (see docs/KNOWN_ISSUES.md).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse

from esopus.data.transforms import TurbidityScaler
from esopus.data.ungaged import mask_node_train_split


# ---------------------------------------------------------------------------
# Graph loading and validation
# ---------------------------------------------------------------------------

def load_graph(path: str | Path) -> nx.DiGraph:
    """Unpickle a NetworkX graph produced by the preprocessing scripts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Graph pickle not found: {path}\n"
            "Download it from the Zenodo deposit or rebuild it with "
            "scripts/01-03 (see data/README.md)."
        )
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise ValueError(f"Expected a NetworkX graph in {path}, got {type(G)}.")
    return G


def sorted_node_ids(G: nx.Graph) -> List[int]:
    """Node IDs in ascending order: the canonical row order of every array."""
    return sorted(G.nodes(), key=int)


def validate_graph(G: nx.Graph, node_ids: List[int]) -> pd.DatetimeIndex:
    """Check every node has ``x``/``y`` on a shared index; return that index."""
    first = node_ids[0]
    ref_index = G.nodes[first]["x"].index
    ref_cols = list(G.nodes[first]["x"].columns)
    for nid in node_ids:
        nd = G.nodes[nid]
        if "x" not in nd or "y" not in nd:
            raise ValueError(f"Node {nid} is missing 'x' or 'y'.")
        if not nd["x"].index.equals(ref_index) or not nd["y"].index.equals(ref_index):
            raise ValueError(f"Node {nid} has a different timestamp index than node {first}.")
        if list(nd["x"].columns) != ref_cols:
            raise ValueError(f"Node {nid} has different feature columns than node {first}.")
    return ref_index


def build_raw_arrays(G: nx.Graph, node_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack node tables into ``X`` (N, F, T) and ``y`` (N, T), float32, unscaled."""
    feature_names = list(G.nodes[node_ids[0]]["x"].columns)
    X = np.stack([G.nodes[n]["x"].values for n in node_ids], axis=0).transpose(0, 2, 1)
    y = np.stack([G.nodes[n]["y"].values for n in node_ids], axis=0)
    return X.astype(np.float32), y.astype(np.float32), feature_names


def chronological_split(dates: pd.DatetimeIndex, split_date: str) -> int:
    """Index of the first test timestamp (train = ``dates < split_date``)."""
    split_dt = pd.Timestamp(split_date, tz=dates.tz)
    if not dates[0] < split_dt < dates[-1]:
        raise ValueError(f"split date {split_date} is outside [{dates[0]}, {dates[-1]}].")
    return int((dates < split_dt).sum())


def get_edge_index(G: nx.Graph, node_ids: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """River-network adjacency as PyG ``edge_index`` (2, E) and ``edge_weight`` (E,).

    ``nodelist=node_ids`` makes row/column i of the adjacency matrix the same
    node as row i of ``X`` and ``y``. Edges carry no ``weight`` attribute, so
    every edge weight is 1.
    """
    A = nx.adjacency_matrix(G, nodelist=node_ids)
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(A.toarray().astype(np.float32)))
    return edge_index.to(device), edge_weight.float().to(device)


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """Lazy (input window, target) pairs over a scaled graph time series.

    Parameters
    ----------
    X : (N, F, T) scaled features.     y : (N, T) scaled targets (0.0 = missing).
    T_in : look-back length.           T_out : forecast horizon.

    Window ``i`` returns ``X[:, :, i:i+T_in]`` (N, F, T_in) and
    ``y[:, i+T_in:i+T_in+T_out]`` (N, T_out). There are ``T - T_in - T_out``
    windows (the same count used to train the published models).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, T_in: int, T_out: int) -> None:
        super().__init__()
        if X.ndim != 3 or y.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[2] != y.shape[1]:
            raise ValueError(f"Inconsistent shapes X={X.shape}, y={y.shape}.")
        n_windows = X.shape[2] - T_in - T_out
        if n_windows <= 0:
            raise ValueError(f"Series of length {X.shape[2]} is too short for T_in={T_in}, T_out={T_out}.")
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.T_in, self.T_out, self._len = T_in, T_out, n_windows

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_win = self.X[:, :, i: i + self.T_in]
        y_win = self.y[:, i + self.T_in: i + self.T_in + self.T_out]
        return x_win, y_win


def window_target_dates(
    dates: pd.DatetimeIndex,
    t_in: int,
    starts: Optional[Sequence[int]] = None,
    n_windows: Optional[int] = None,
) -> pd.DatetimeIndex:
    """Timestamp of the target predicted by each window.

    The target of window ``i`` is the timestep right after its last input,
    i.e. position ``i + t_in`` of ``dates``.

    Parameters
    ----------
    dates : the timestamp index of the split the windows were cut from.
    t_in : look-back length.
    starts : window start positions (default ``range(n_windows)``).
    n_windows : number of consecutive windows starting at 0.

    A start whose target falls one step past the end of ``dates`` is
    timestamped by extrapolating the (hourly) index frequency.
    """
    if starts is None:
        if n_windows is None:
            raise ValueError("Pass either `starts` or `n_windows`.")
        starts = np.arange(n_windows)
    pos = np.asarray(starts) + t_in
    step = dates[1] - dates[0]
    out = [dates[p] if p < len(dates) else dates[-1] + step * (p - len(dates) + 1) for p in pos]
    return pd.DatetimeIndex(out)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """Everything downstream code needs about one configured dataset."""
    G: nx.DiGraph
    node_ids: List[int]
    feature_names: List[str]
    scaler: TurbidityScaler
    train_dataset: SlidingWindowDataset
    test_dataset: SlidingWindowDataset
    dates: pd.DatetimeIndex          # full hourly index
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    split_idx: int
    y_raw: np.ndarray                # (N, T) unscaled turbidity (after any ungaged masking)
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    mask_report: Optional[dict] = field(default=None)

    @property
    def T_in(self) -> int:
        return self.train_dataset.T_in

    def target_dates(self, split: str) -> pd.DatetimeIndex:
        """Timestamps of every window target in ``split`` ("train" or "test")."""
        ds, d = (self.train_dataset, self.train_dates) if split == "train" else (self.test_dataset, self.test_dates)
        return window_target_dates(d, ds.T_in, n_windows=len(ds))

    def observed_targets(self, split: str) -> np.ndarray:
        """Unscaled observations aligned with the window targets, shape (T_windows, N)."""
        ds = self.train_dataset if split == "train" else self.test_dataset
        offset = 0 if split == "train" else self.split_idx
        start = offset + ds.T_in
        return self.y_raw[:, start: start + len(ds)].T


def prepare_data(
    cfg,
    device: torch.device,
    scaler: Optional[TurbidityScaler] = None,
    apply_ungaged_mask: bool = True,
    verbose: bool = True,
) -> PreparedData:
    """Run the full data-preparation pipeline described in the module docstring.

    Parameters
    ----------
    cfg : EsopusConfig
    device : where ``edge_index``/``edge_weight`` are placed.
    scaler : an already-fitted scaler (e.g. the archived one for a published
        checkpoint). If ``None`` a new scaler is fit on the training split.
    apply_ungaged_mask : if True (default) and ``cfg.ungaged.withheld_node``
        is set, that node's training-period turbidity is masked (as in
        training). Set False to get the true, unmasked observations.
    """
    G = load_graph(cfg.graph_path)

    mask_report = None
    if apply_ungaged_mask and cfg.ungaged.withheld_node is not None:
        mask_report = mask_node_train_split(G, cfg.ungaged.withheld_node, cfg.data.test_split_date)

    node_ids = sorted_node_ids(G)
    dates = validate_graph(G, node_ids)
    X_raw, y_raw, feature_names = build_raw_arrays(G, node_ids)
    split_idx = chronological_split(dates, cfg.data.test_split_date)

    if scaler is None:
        scaler = TurbidityScaler(eps=cfg.data.eps).fit(
            X_raw[:, :, :split_idx], y_raw[:, :split_idx], node_ids, feature_names
        )

    T_in, T_out = cfg.data.num_timesteps_in, cfg.data.num_timesteps_out
    train_ds = SlidingWindowDataset(
        scaler.transform_X(X_raw[:, :, :split_idx]), scaler.transform_y(y_raw[:, :split_idx]), T_in, T_out
    )
    test_ds = SlidingWindowDataset(
        scaler.transform_X(X_raw[:, :, split_idx:]), scaler.transform_y(y_raw[:, split_idx:]), T_in, T_out
    )
    edge_index, edge_weight = get_edge_index(G, node_ids, device)

    if verbose:
        n_gaged = int(np.any(~np.isnan(y_raw), axis=1).sum())
        print(
            f"[data] {cfg.graph_path.name}: {len(node_ids)} nodes ({n_gaged} with turbidity), "
            f"{len(feature_names)} features, {len(dates):,} hourly steps, {edge_index.shape[1]} edges\n"
            f"[data] train {dates[0]:%Y-%m-%d} -> {dates[split_idx - 1]:%Y-%m-%d} ({len(train_ds):,} windows) | "
            f"test {dates[split_idx]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d} ({len(test_ds):,} windows)"
        )

    return PreparedData(
        G=G, node_ids=node_ids, feature_names=feature_names, scaler=scaler,
        train_dataset=train_ds, test_dataset=test_ds,
        dates=dates, train_dates=dates[:split_idx], test_dates=dates[split_idx:],
        split_idx=split_idx, y_raw=y_raw, edge_index=edge_index, edge_weight=edge_weight,
        mask_report=mask_report,
    )
