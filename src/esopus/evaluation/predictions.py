"""Full-record (train + test) predictions with correct timestamps.

A :class:`PredictionSet` holds hourly predictions and the matching
observations for every node, each row stamped with the timestamp of the
target it predicts (see ``esopus.data.window_target_dates``). It is written
once by ``scripts/11_evaluate.py`` and then reused by source tracing, the
full-record metrics and the figures, so inference only has to run once per
model.

Files (in ``outputs/predictions/<run_name>/``)::

    predicted_fnu.csv.gz   datetime_utc, split, <node_id>...   (model output, FNU, clipped >= 0)
    observed_fnu.csv.gz    datetime_utc, split, <node_id>...   (observations, FNU, NaN = missing)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from esopus.data.dataset import PreparedData
from esopus.data.loaders import make_loader
from esopus.training.metrics import compute_all_metrics, summarise_metrics
from esopus.training.trainer import predict


@dataclass
class PredictionSet:
    """Hourly predictions/observations, shape (T, N), with aligned timestamps."""
    dates: pd.DatetimeIndex
    split: np.ndarray            # "train" / "test" per row
    node_ids: List[int]
    predicted: np.ndarray        # FNU, clipped at 0
    observed: np.ndarray         # FNU, NaN where unobserved

    # ---- slicing helpers -------------------------------------------------
    def subset(self, split: str) -> "PredictionSet":
        """Rows of one split ("train" or "test"); "full" returns self."""
        if split == "full":
            return self
        m = self.split == split
        return PredictionSet(self.dates[m], self.split[m], self.node_ids, self.predicted[m], self.observed[m])

    def series(self, node_id: int, which: str = "predicted") -> pd.Series:
        """One node's predicted or observed series indexed by timestamp."""
        arr = self.predicted if which == "predicted" else self.observed
        return pd.Series(arr[:, self.node_ids.index(node_id)], index=self.dates, name=node_id)

    # ---- metrics ------------------------------------------------------------
    def metrics(self, split: str = "test") -> pd.DataFrame:
        s = self.subset(split)
        return compute_all_metrics(s.observed, s.predicted, s.node_ids)

    def summary(self, split: str = "test") -> pd.DataFrame:
        return summarise_metrics(self.metrics(split))

    # ---- persistence --------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in (("predicted_fnu", self.predicted), ("observed_fnu", self.observed)):
            df = pd.DataFrame(arr, index=self.dates, columns=[str(n) for n in self.node_ids])
            df.index.name = "datetime_utc"
            df.insert(0, "split", self.split)
            df.to_csv(out_dir / f"{name}.csv.gz", float_format="%.7g")

    @classmethod
    def load(cls, out_dir: Path) -> "PredictionSet":
        out_dir = Path(out_dir)
        if not (out_dir / "predicted_fnu.csv.gz").exists():
            raise FileNotFoundError(
                f"No predictions in {out_dir}. Run scripts/11_evaluate.py for this model first."
            )
        pred = pd.read_csv(out_dir / "predicted_fnu.csv.gz", index_col=0, parse_dates=True)
        obs = pd.read_csv(out_dir / "observed_fnu.csv.gz", index_col=0, parse_dates=True)
        node_cols = [c for c in pred.columns if c != "split"]
        return cls(
            dates=pd.DatetimeIndex(pred.index),
            split=pred["split"].to_numpy(),
            node_ids=[int(c) for c in node_cols],
            predicted=pred[node_cols].to_numpy(dtype=float),
            observed=obs[node_cols].to_numpy(dtype=float),
        )


def full_record_predictions(model: torch.nn.Module, data: PreparedData, batch_size: int, device: torch.device) -> PredictionSet:
    """Predict every train and test window and pair each with its observation.

    Observations are taken from the unscaled targets in ``data.y_raw``, so for
    an ungaged-validation model the withheld node's training-period
    observations are NaN; load a separate un-masked ``PreparedData`` when the
    true observations are needed.
    """
    parts = []
    for split, ds in (("train", data.train_dataset), ("test", data.test_dataset)):
        preds_scaled, _ = predict(model, make_loader(ds, batch_size), data.edge_index, data.edge_weight, device)
        preds = np.clip(data.scaler.inverse_transform_y(preds_scaled), 0.0, None)
        parts.append((split, data.target_dates(split), preds, data.observed_targets(split)))

    return PredictionSet(
        dates=parts[0][1].append(parts[1][1]),
        split=np.concatenate([np.full(len(p[1]), p[0]) for p in parts]),
        node_ids=list(data.node_ids),
        predicted=np.concatenate([p[2] for p in parts], axis=0),
        observed=np.concatenate([p[3] for p in parts], axis=0).astype(float),
    )
