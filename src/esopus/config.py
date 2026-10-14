"""Experiment configuration: YAML file -> validated, typed config object.

Each top-level YAML section maps onto one dataclass below. Every field the
code reads is declared here with its paper default, so a config file only
needs to list what it changes — but the paper configs in ``configs/`` spell
everything out for transparency.

Example
-------
>>> from esopus.config import load_config
>>> cfg = load_config("configs/graphwavenet_full.yaml")
>>> cfg.model.adjacency_mode
'both'
>>> cfg.graph_path            # absolute path, resolved against the repo root
PosixPath('/.../data/processed/esopus_graph_features.pkl')
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from esopus import paths

VALID_ADJACENCY_MODES = ("fixed", "learned", "both")
VALID_OPTIMIZERS = ("adam", "adamw")


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Identity of a training run.

    ``run_name`` names every artefact the run produces (weights, scaler,
    prediction tables, IG caches), e.g. ``outputs/models/gwn_full.pt``.
    ``mlflow_experiment`` groups runs in the MLflow UI.
    """
    run_name: str = "gwn_full"
    mlflow_experiment: str = "esopus_turbidity"
    notes: str = ""


@dataclass
class DataConfig:
    """Input graph and sliding-window settings.

    graph_pkl         : feature-engineered graph produced by scripts/03 (relative
                        to the repo root; ``data/`` honours ``ESOPUS_DATA_DIR``).
    test_split_date   : first timestamp of the test period (ISO date).
    num_timesteps_in  : look-back window length T_in (hours).
    num_timesteps_out : forecast horizon T_out (hours).
    eps               : offset added after min-max scaling so observed zeros
                        are distinguishable from the 0.0 missing-data sentinel.
    """
    graph_pkl: str = "data/processed/esopus_graph_features.pkl"
    test_split_date: str = "2023-09-19"
    num_timesteps_in: int = 17
    num_timesteps_out: int = 1
    eps: float = 1e-5


@dataclass
class LoaderConfig:
    """DataLoader settings. Windows are never shuffled (temporal order is kept)."""
    batch_size: int = 32
    drop_last_train: bool = True
    num_workers: int = 0


@dataclass
class ModelConfig:
    """GraphWaveNet hyper-parameters (passed to ``tsl``'s GraphWaveNetModel).

    adjacency_mode : "fixed" -> river-network (NHD) adjacency only;
                     "both"  -> NHD adjacency + learned adaptive adjacency.
                     "learned" is accepted but behaves exactly like "both"
                     (see docs/KNOWN_ISSUES.md).
    embedding_dim  : size of the node embeddings E1, E2 that define the
                     adaptive adjacency softmax(relu(E1 E2^T)).
    """
    hidden_channels: int = 16
    ff_channels: int = 32
    n_layers: int = 4
    adjacency_mode: str = "both"
    embedding_dim: int = 10


@dataclass
class TrainingConfig:
    """Optimisation settings.

    The test split doubles as the early-stopping / best-weights selection set,
    exactly as in the published runs (see docs/KNOWN_ISSUES.md).
    """
    num_epochs: int = 75
    learning_rate: float = 3e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    patience: int = 15
    min_delta: float = 0.0
    grad_clip_norm: float = 1.0
    mask_val: float = 0.0
    seed: int = 0
    eval_interval: int = 5


@dataclass
class UngagedConfig:
    """Leave-one-out validation: node whose *training-period* turbidity is
    withheld (set to NaN) before scaling and training. ``None`` = no masking."""
    withheld_node: Optional[int] = None


@dataclass
class EsopusConfig:
    """Complete configuration for one training run."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ungaged: UngagedConfig = field(default_factory=UngagedConfig)
    source_file: Optional[str] = None

    # ---- convenience accessors ------------------------------------------
    @property
    def run_name(self) -> str:
        return self.experiment.run_name

    @property
    def graph_path(self) -> Path:
        return paths.resolve(self.data.graph_pkl)

    # ---- validation -----------------------------------------------------
    def validate(self) -> "EsopusConfig":
        d, m, t = self.data, self.model, self.training
        if d.num_timesteps_in <= 0 or d.num_timesteps_out <= 0:
            raise ValueError("num_timesteps_in and num_timesteps_out must be positive.")
        if d.num_timesteps_out != 1:
            raise ValueError("The analysis code assumes a one-step-ahead model (num_timesteps_out = 1).")
        pd.Timestamp(d.test_split_date)  # raises if unparsable
        if m.adjacency_mode not in VALID_ADJACENCY_MODES:
            raise ValueError(f"adjacency_mode must be one of {VALID_ADJACENCY_MODES}, got {m.adjacency_mode!r}.")
        if t.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}, got {t.optimizer!r}.")
        if t.patience < 1 or t.num_epochs < 1:
            raise ValueError("patience and num_epochs must be >= 1.")
        return self

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten to ``{"section.key": value}`` for MLflow parameter logging."""
        flat = {}
        for section in ("experiment", "data", "loader", "model", "training", "ungaged"):
            for k, v in asdict(getattr(self, section)).items():
                flat[f"{section}.{k}"] = v
        return flat


_SECTIONS = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "loader": LoaderConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "ungaged": UngagedConfig,
}


def _build_section(cls, values: Optional[dict], section: str):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> EsopusConfig:
    """Load and validate a YAML config.

    Parameters
    ----------
    path : str or Path
        YAML file; relative paths are resolved against the repository root.
    overrides : dict, optional
        ``{section: {key: value}}`` applied on top of the file (used e.g. by
        ``--debug`` flags to shorten training).
    """
    path = paths.resolve(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown top-level section(s) in {path}: {sorted(unknown)}")
    for section, vals in (overrides or {}).items():
        raw.setdefault(section, {}).update(vals)
    cfg = EsopusConfig(
        **{name: _build_section(cls, raw.get(name), name) for name, cls in _SECTIONS.items()},
        source_file=str(path),
    )
    return cfg.validate()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) and force deterministic cuDNN.

    Called once at the start of training, *before* the model is built, so
    weight initialisation is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
