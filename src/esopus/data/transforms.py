"""Per-feature min-max scaling with a missing-data sentinel.

``TurbidityScaler`` converts raw graph arrays (features in physical units,
turbidity in FNU) to the [eps, 1 + eps] range the network is trained on, and
converts predictions back to FNU.

Scaling
-------
For every input feature f (statistics pooled over all nodes and training
timesteps)::

    x_scaled = (x - min_f) / (max_f - min_f) + eps

and for turbidity (one global range over all gaged nodes)::

    y_scaled = (y - y_min) / (y_max - y_min) + eps

Missing values (NaN: sensor gaps, ungaged nodes) become the sentinel 0.0.
Because every observed value is shifted up by ``eps``, a scaled value of
exactly 0.0 can only mean "missing", which is what the masked loss
(``esopus.training.losses.masked_mse_loss``) relies on.

Rules
-----
* The scaler is fit on the TRAINING split only and then applied unchanged to
  the test split (no leakage of test-period ranges).
* No log transform is applied. (An earlier version of this class *described*
  a log1p step that was never actually executed; the published models were
  trained with plain min-max scaling, which is what this class implements.)

The class name and module path are kept identical to the ones used for the
paper so archived scaler pickles (``outputs/scalers/*.pkl``) load directly.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np


class TurbidityScaler:
    """Min-max scaler for graph features ``X`` (N, F, T) and targets ``y`` (N, T).

    Parameters
    ----------
    eps : float
        Offset added after scaling so observed zeros differ from the 0.0
        missing-data sentinel.

    Attributes (set by :meth:`fit`)
    ------------------------------
    x_min, x_max : np.ndarray, shape (F,)
        Per-feature training range.
    y_min, y_max : float
        Training range of turbidity (FNU).
    node_ids : list[int]
        Node order the scaler was fit with.
    feature_names : list[str] or None
    """

    def __init__(self, eps: float = 1e-5, **_ignored) -> None:
        # ``**_ignored`` swallows the legacy ``skew_threshold`` argument.
        self.eps = eps
        self.x_min: Optional[np.ndarray] = None
        self.x_max: Optional[np.ndarray] = None
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None
        self.node_ids: Optional[List[int]] = None
        self.feature_names: Optional[List[str]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        node_ids: List[int],
        feature_names: Optional[List[str]] = None,
    ) -> "TurbidityScaler":
        """Learn per-feature and turbidity ranges from the training split.

        Parameters
        ----------
        X_train : (N, F, T_train) raw features, NaN allowed.
        y_train : (N, T_train) raw turbidity in FNU, NaN = not observed.
        node_ids : node order of axis 0.
        feature_names : optional column names of axis 1 (for summaries).
        """
        if X_train.ndim != 3 or y_train.ndim != 2 or X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Expected X (N, F, T) and y (N, T); got {X_train.shape} and {y_train.shape}.")
        self.node_ids = list(node_ids)
        self.feature_names = list(feature_names) if feature_names is not None else None

        self.x_min = np.nanmin(X_train, axis=(0, 2))  # (F,) pooled over nodes and time
        self.x_max = np.nanmax(X_train, axis=(0, 2))
        self.y_min = float(np.nanmin(y_train))
        self.y_max = float(np.nanmax(y_train))
        self._fitted = True

        constant = np.where(self.x_max == self.x_min)[0]
        if len(constant):
            warnings.warn(
                f"Features {constant.tolist()} are constant over the training split; "
                "they will scale to eps everywhere.", UserWarning, stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Scale features (N, F, T) with the training ranges; NaN -> 0.0."""
        self._require_fitted()
        rng = (self.x_max - self.x_min).reshape(1, -1, 1)
        rng = np.where(rng == 0.0, 1.0, rng)  # constant features: avoid divide-by-zero
        X_scaled = (X - self.x_min.reshape(1, -1, 1)) / rng + self.eps
        return np.where(np.isnan(X_scaled), 0.0, X_scaled).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Scale turbidity (any shape) with the training range; NaN -> 0.0."""
        self._require_fitted()
        rng = self.y_max - self.y_min
        rng = rng if rng != 0.0 else 1.0
        y_scaled = (y - self.y_min) / rng + self.eps
        return np.where(np.isnan(y_scaled), 0.0, y_scaled).astype(np.float32)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map scaled turbidity back to FNU; the 0.0 sentinel becomes NaN."""
        self._require_fitted()
        y = np.asarray(y_scaled, dtype=np.float64).copy()
        y = np.where(y == 0.0, np.nan, y)
        return (y - self.eps) * (self.y_max - self.y_min) + self.y_min

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Pickle the fitted scaler."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TurbidityScaler":
        """Load a scaler written by :meth:`save` (including the paper's archived scalers)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected TurbidityScaler, got {type(obj)}")
        return obj

    def summary(self) -> str:
        if not self._fitted:
            return "TurbidityScaler (not fitted)"
        return (
            f"TurbidityScaler: {len(self.node_ids)} nodes, {len(self.x_min)} features, "
            f"turbidity range [{self.y_min:.2f}, {self.y_max:.2f}] FNU, eps={self.eps}"
        )

    __repr__ = summary

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TurbidityScaler has not been fitted. Call .fit() first.")
