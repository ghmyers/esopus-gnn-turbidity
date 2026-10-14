"""Training loop (resumable), inference, and evaluation.

``train``
    Adam/AdamW + cosine-annealed learning rate, masked MSE loss, gradient
    clipping, and early stopping. Identical in every numerical respect to the
    loop that produced the published checkpoints; it additionally writes a
    resumable state file after every epoch so an interrupted run (e.g. a
    killed HPC job) restarts where it left off.

    As in the published runs, the test split is used for early stopping and
    best-weights selection — there is no separate validation split (see
    docs/KNOWN_ISSUES.md).

``predict``
    Scaled one-step-ahead predictions for every window of a loader.

``evaluate``
    Predictions and observations in FNU plus per-node / network metrics.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from esopus.data.transforms import TurbidityScaler
from esopus.training.losses import masked_mse_loss
from esopus.training.metrics import compute_all_metrics, flat_summary, summarise_metrics

LogFn = Callable[[Dict[str, float], int], None]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over every batch.

    Returns
    -------
    preds_scaled   : (T_windows, N) first forecast step, scaled units.
    targets_scaled : (T_windows, N) matching targets, scaled units (0.0 = missing).
    """
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        y_hat = model(x.to(device, non_blocking=True), edge_index, edge_weight)  # (B, N, T_out)
        preds.append(y_hat[:, :, 0].cpu().numpy())
        targets.append(y[:, :, 0].numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    scaler: TurbidityScaler,
    node_ids: List[int],
    split_name: str = "test",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Predict, convert to FNU (negative predictions clipped to 0) and score.

    Returns ``(preds_fnu, targets_fnu, per_node_metrics, network_summary)``;
    arrays are (T_windows, N) with NaN where the target is unobserved.
    """
    preds_scaled, targets_scaled = predict(model, loader, edge_index, edge_weight, device)
    preds_fnu = np.clip(scaler.inverse_transform_y(preds_scaled), 0.0, None)
    targets_fnu = scaler.inverse_transform_y(targets_scaled)

    metrics_df = compute_all_metrics(targets_fnu, preds_fnu, node_ids)
    summary_df = summarise_metrics(metrics_df)
    if verbose and not summary_df.empty:
        med = summary_df.loc["median"]
        print(
            f"[{split_name}] {len(metrics_df)} gaged nodes | median KGE {med['kge']:.3f}  "
            f"NSE {med['nse']:.3f}  MAE {med['mae']:.2f}  RMSE {med['rmse']:.2f} FNU"
        )
    return preds_fnu, targets_fnu, metrics_df, summary_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    model: nn.Module
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float
    elapsed_min: float


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(
    cfg,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    scaler: TurbidityScaler,
    node_ids: List[int],
    best_model_path: Path,
    resume_state_path: Path,
    log_fn: Optional[LogFn] = None,
) -> TrainResult:
    """Train with early stopping; resume automatically from ``resume_state_path``.

    Each epoch: one pass over ``train_loader`` (masked MSE, gradient clipping
    at ``cfg.training.grad_clip_norm``, optimiser step), a cosine learning-rate
    step, then the masked MSE on ``test_loader``. Whenever that loss improves
    by more than ``min_delta`` the weights are saved to ``best_model_path``.
    Training stops after ``patience`` epochs without improvement.

    After every epoch the full state (model, optimiser, scheduler, RNG,
    early-stopping counters, loss histories) is written to
    ``resume_state_path``; it is deleted once training completes.

    ``log_fn(metrics, step)``, if given, receives per-epoch losses/learning
    rate and, every ``cfg.training.eval_interval`` epochs, network-median
    test metrics (used for MLflow logging by ``scripts/10_train.py``).
    """
    t = cfg.training
    best_model_path, resume_state_path = Path(best_model_path), Path(resume_state_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    opt_cls = torch.optim.AdamW if t.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t.num_epochs, eta_min=t.learning_rate * 0.01
    )

    start_epoch, best_val, best_epoch, epochs_no_improve = 0, float("inf"), 0, 0
    train_hist: List[float] = []
    val_hist: List[float] = []
    best_state: Optional[dict] = None

    if resume_state_path.exists():
        s = torch.load(resume_state_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model_state"])
        optimizer.load_state_dict(s["optimizer_state"])
        scheduler.load_state_dict(s["scheduler_state"])
        _set_rng_state(s["rng_state"])
        start_epoch, best_val, best_epoch = s["epoch"], s["best_val"], s["best_epoch"]
        epochs_no_improve, train_hist, val_hist = s["epochs_no_improve"], s["train_history"], s["val_history"]
        if best_model_path.exists():
            best_state = torch.load(best_model_path, map_location="cpu", weights_only=True)
        print(f"[train] Resuming at epoch {start_epoch + 1}/{t.num_epochs} "
              f"(best test loss {best_val:.5f} at epoch {best_epoch})")

    t0 = time.time()
    for epoch in range(start_epoch, t.num_epochs):
        # ---- training pass ----
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = masked_mse_loss(model(x, edge_index, edge_weight), y, mask_val=t.mask_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.item())
        train_hist.append(float(np.nanmean(batch_losses)))
        scheduler.step()

        # ---- early-stopping loss on the test split ----
        model.eval()
        with torch.no_grad():
            val_losses = [
                masked_mse_loss(model(x.to(device), edge_index, edge_weight), y.to(device), mask_val=t.mask_val).item()
                for x, y in test_loader
            ]
        val_hist.append(float(np.nanmean(val_losses)))
        lr = optimizer.param_groups[0]["lr"]

        improved = val_hist[-1] < best_val - t.min_delta
        print(f"  epoch {epoch + 1:3d}/{t.num_epochs} | train {train_hist[-1]:.5f} | "
              f"test {val_hist[-1]:.5f} | lr {lr:.2e}" + ("  <- best" if improved else ""), flush=True)
        if log_fn:
            log_fn({"train_mse": train_hist[-1], "test_mse": val_hist[-1], "lr": lr}, epoch)

        if (epoch + 1) % t.eval_interval == 0:
            _, _, _, summary = evaluate(model, test_loader, edge_index, edge_weight, device, scaler,
                                        node_ids, split_name=f"epoch {epoch + 1}")
            if log_fn:
                log_fn(flat_summary(summary, prefix="test_"), epoch)

        if improved:
            best_val, best_epoch, epochs_no_improve = val_hist[-1], epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_path)
        else:
            epochs_no_improve += 1

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": _rng_state(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "train_history": train_hist,
            "val_history": val_hist,
        }, resume_state_path)

        if epochs_no_improve >= t.patience:
            print(f"[train] Early stopping after epoch {epoch + 1} ({t.patience} epochs without improvement).")
            break

    elapsed = (time.time() - t0) / 60
    print(f"[train] Done: best test loss {best_val:.5f} at epoch {best_epoch} ({elapsed:.1f} min this session).")
    if best_state is None:
        raise RuntimeError("Training never improved on the initial loss; no best weights were saved.")
    model.load_state_dict(best_state)
    model.to(device)
    resume_state_path.unlink(missing_ok=True)
    return TrainResult(model, train_hist, val_hist, best_epoch, best_val, elapsed)
