"""Goodness-of-fit metrics and masked loss."""

import numpy as np
import torch

from esopus.training import compute_node_metrics, kge, masked_mse_loss, nse


def test_perfect_prediction():
    obs = np.linspace(1, 10, 50)
    assert kge(obs, obs) == 1.0 and nse(obs, obs) == 1.0
    m = compute_node_metrics(obs, obs.copy())
    assert m["mae"] == 0 and m["rmse"] == 0


def test_metrics_ignore_nan_and_require_min_obs():
    obs = np.full(20, np.nan)
    obs[:5] = 1.0
    assert compute_node_metrics(obs, np.ones(20)) == {}


def test_masked_mse_ignores_sentinel():
    target = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    pred = torch.tensor([[9.0, 0.5], [0.7, 9.0]])
    # only the two observed entries count: errors 0 and 0.2 -> MSE 0.02
    assert torch.isclose(masked_mse_loss(pred, target), torch.tensor(0.02))
