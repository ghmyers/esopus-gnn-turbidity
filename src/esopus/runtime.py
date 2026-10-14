"""Small helpers shared by the command-line scripts.

* :func:`get_device`         — CUDA if available, else CPU
* :func:`train_run`          — train the model described by a config (step 10)
* :func:`load_trained_run`   — config -> (data, trained model) using the run's
                               saved scaler and best-epoch weights
* :func:`evaluate_run`       — metrics + full-record predictions for a run (step 11)
* :func:`mlflow_run`         — optional MLflow tracking context
"""

from __future__ import annotations

import contextlib
import json
from typing import Callable, Dict, Iterator, Optional, Tuple

import pandas as pd
import torch

from esopus import paths
from esopus.config import set_seed
from esopus.data import PreparedData, TurbidityScaler, build_loaders, prepare_data
from esopus.evaluation import PredictionSet, full_record_predictions
from esopus.models import WaveNetModel, build_model, load_trained_model
from esopus.training import TrainResult, train


def get_device(prefer: Optional[str] = None) -> torch.device:
    """``prefer`` ("cpu"/"cuda") overrides auto-detection."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_run(
    cfg, device: torch.device, unmasked: bool = False, verbose: bool = True,
) -> Tuple[PreparedData, WaveNetModel]:
    """Prepare data exactly as for training and load the run's best weights.

    The run's own fitted scaler (``outputs/scalers/<run>_scaler.pkl``) is used,
    so predictions are identical to those made when the model was trained.

    ``unmasked=True`` skips the ungaged-validation masking so the returned
    observations include the withheld node's true training-period data
    (the scaler — fit on masked data — is still the run's own).
    """
    scaler_file, model_file = paths.scaler_path(cfg.run_name), paths.model_path(cfg.run_name)
    for f in (scaler_file, model_file):
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. Train the model (scripts/10_train.py) or download the "
                "trained checkpoints from the Zenodo deposit (see data/README.md)."
            )
    data = prepare_data(cfg, device, scaler=TurbidityScaler.load(scaler_file),
                        apply_ungaged_mask=not unmasked, verbose=verbose)
    model = load_trained_model(cfg, model_file, len(data.feature_names), len(data.node_ids), device)
    return data, model


@contextlib.contextmanager
def mlflow_run(cfg, enabled: bool = True) -> Iterator[Optional[Callable[[Dict[str, float], int], None]]]:
    """Start an MLflow run (file store under ``outputs/mlruns``) and yield a
    ``log_fn(metrics, step)``; yields ``None`` when disabled."""
    if not enabled:
        yield None
        return
    import mlflow

    paths.ensure_dir(paths.MLRUNS_DIR)
    mlflow.set_tracking_uri(paths.MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(cfg.experiment.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(cfg.to_flat_dict())
        if cfg.source_file:
            mlflow.log_artifact(cfg.source_file, artifact_path="config")
        yield lambda metrics, step: mlflow.log_metrics(metrics, step=step)


def train_run(cfg, device: torch.device, use_mlflow: bool = True) -> TrainResult:
    """Seed, prepare data, build and train the model; save weights, scaler and history."""
    set_seed(cfg.training.seed)  # before the model is built: reproducible initialisation
    data = prepare_data(cfg, device)
    train_loader, test_loader = build_loaders(cfg, data.train_dataset, data.test_dataset)
    model = build_model(cfg, len(data.feature_names), len(data.node_ids), device)
    print(f"[train] {cfg.run_name}: {model.count_parameters():,} trainable parameters; receptive field "
          f"{model.receptive_field} h of the {cfg.data.num_timesteps_in}-h input window")

    data.scaler.save(paths.scaler_path(cfg.run_name))
    with mlflow_run(cfg, enabled=use_mlflow) as log_fn:
        result = train(
            cfg, model, train_loader, test_loader, data.edge_index, data.edge_weight, device,
            data.scaler, data.node_ids,
            best_model_path=paths.model_path(cfg.run_name),
            resume_state_path=paths.resume_path(cfg.run_name),
            log_fn=log_fn,
        )
        if log_fn:
            log_fn({"best_epoch": result.best_epoch, "best_test_mse": result.best_val_loss}, len(result.train_loss))
    history = {"train_mse": result.train_loss, "test_mse": result.val_loss, "best_epoch": result.best_epoch}
    (paths.MODELS_DIR / f"{cfg.run_name}_history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] saved {paths.model_path(cfg.run_name)}")
    return result


def evaluate_run(cfg, device: torch.device) -> PredictionSet:
    """Score a trained run and write ``outputs/predictions/<run>/`` (see scripts/11)."""
    # Unmasked observations: ungaged folds are scored against the withheld node's true data.
    # (Masking never changes predictions: inputs and the run's scaler are fixed.)
    data, model = load_trained_run(cfg, device, unmasked=True)
    preds = full_record_predictions(model, data, cfg.loader.batch_size, device)

    out = paths.ensure_dir(paths.predictions_dir(cfg.run_name))
    test_metrics = preds.metrics("test")
    test_metrics.to_csv(out / "per_node_metrics.csv")
    summary = preds.summary("test")
    summary.to_csv(out / "summary_metrics.csv")
    pd.concat({p: preds.metrics(p) for p in ("train", "test", "full")}, names=["period"]).to_csv(
        out / "metrics_by_period.csv")
    preds.save(out)

    med = summary.loc["median"]
    print(f"[evaluate] {cfg.run_name}: {len(test_metrics)} gaged nodes, test median KGE {med['kge']:.3f}  "
          f"NSE {med['nse']:.3f}  MAE {med['mae']:.2f}  RMSE {med['rmse']:.2f} FNU -> {out}")
    return preds
