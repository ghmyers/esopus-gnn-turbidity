"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch

from esopus.models.graphwavenet import WaveNetModel

__all__ = ["WaveNetModel", "build_model", "load_trained_model"]


def build_model(cfg, n_features: int, n_nodes: int, device: torch.device) -> WaveNetModel:
    """Instantiate the GraphWaveNet described by ``cfg.model`` on ``device``."""
    m = cfg.model
    model = WaveNetModel(
        n_nodes=n_nodes,
        in_channels=n_features,
        out_horizon=cfg.data.num_timesteps_out,
        hidden_channels=m.hidden_channels,
        ff_channels=m.ff_channels,
        n_layers=m.n_layers,
        adjacency_mode=m.adjacency_mode,
        embedding_dim=m.embedding_dim,
    )
    return model.to(device)


def load_trained_model(cfg, checkpoint: str | Path, n_features: int, n_nodes: int, device: torch.device) -> WaveNetModel:
    """Build the model for ``cfg`` and load best-epoch weights; returns it in eval mode."""
    model = build_model(cfg, n_features, n_nodes, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
