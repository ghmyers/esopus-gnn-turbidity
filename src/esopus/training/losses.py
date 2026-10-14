"""Masked loss for sparsely observed graph targets.

Most nodes are ungaged and gaged nodes have sensor gaps, so the scaled target
tensor is mostly the missing-data sentinel 0.0 (see
``esopus.data.transforms``). The loss is therefore computed over observed
entries only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask_val: float = 0.0) -> torch.Tensor:
    """Mean squared error over observed (``targets != mask_val``) entries.

    The mask is rescaled so its mean is 1, which makes the result the MSE per
    *observed* node-timestep. Batches with different numbers of observations
    therefore produce comparable gradient magnitudes.

    Returns 0 (with grad) if the batch contains no observations.
    """
    mask = (targets != mask_val).float()
    n_observed = mask.sum()
    if n_observed == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    mask = mask / (n_observed / mask.numel())
    loss = F.mse_loss(preds, targets, reduction="none") * mask
    return torch.nan_to_num(loss, nan=0.0).mean()
