"""Canonical file-system locations for data, outputs, and figures.

Every script and module resolves paths through this file, so the repository
can be cloned anywhere and run without editing hard-coded paths.

Layout (defaults, relative to the repository root)::

    data/
      raw/                 USGS, AORC, StreamStats and sub-watershed files (Zenodo deposit)
      processed/           graph pickles produced by scripts 01-03 (training graph also in the deposit)
    outputs/
      models/              trained weights  ({run_name}.pt) + resume states
      scalers/             fitted TurbidityScaler objects ({run_name}_scaler.pkl)
      predictions/<run>/   per-node metrics + hourly predictions (CSV)
      interpretation/<run>/  Integrated Gradients caches (.npz)
      analyses/            discharge regressions, source tracing, ablation, ungaged validation,
                           attribution stability
      mlruns/              MLflow tracking store
    figures/
      main/                manuscript figures
      supplementary/       supporting-information figures

Environment variables override the defaults, which is useful when the
(large) data live on a different disk than the code:

    ESOPUS_DATA_DIR     replaces <repo>/data
    ESOPUS_OUTPUT_DIR   replaces <repo>/outputs
    ESOPUS_FIGURES_DIR  replaces <repo>/figures (contains main/ and supplementary/)

The package must be installed in editable mode (``pip install -e .``) for
``REPO_ROOT`` to point at the cloned repository.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/esopus/paths.py -> parents[0]=esopus, [1]=src, [2]=repository root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("ESOPUS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = Path(os.environ.get("ESOPUS_OUTPUT_DIR", REPO_ROOT / "outputs"))
MODELS_DIR: Path = OUTPUT_DIR / "models"
SCALERS_DIR: Path = OUTPUT_DIR / "scalers"
PREDICTIONS_DIR: Path = OUTPUT_DIR / "predictions"
INTERPRETATION_DIR: Path = OUTPUT_DIR / "interpretation"
ANALYSES_DIR: Path = OUTPUT_DIR / "analyses"
MLRUNS_DIR: Path = OUTPUT_DIR / "mlruns"

FIGURES_DIR: Path = Path(os.environ.get("ESOPUS_FIGURES_DIR", REPO_ROOT / "figures"))
MAIN_FIGURES_DIR: Path = FIGURES_DIR / "main"
SUPP_FIGURES_DIR: Path = FIGURES_DIR / "supplementary"

CONFIGS_DIR: Path = REPO_ROOT / "configs"


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Resolve ``path`` against ``base`` unless it is already absolute.

    Config files store paths relative to the repository root (e.g.
    ``data/processed/esopus_graph_features.pkl``); this turns them into
    absolute paths regardless of the current working directory. A leading
    ``data/`` or ``outputs/`` component is redirected to ``DATA_DIR`` /
    ``OUTPUT_DIR`` so the environment-variable overrides are honoured.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        return DATA_DIR.joinpath(*parts[1:])
    if parts and parts[0] == "outputs":
        return OUTPUT_DIR.joinpath(*parts[1:])
    return base / p


def model_path(run_name: str) -> Path:
    """Best-epoch weights (state_dict) for a training run."""
    return MODELS_DIR / f"{run_name}.pt"


def resume_path(run_name: str) -> Path:
    """Per-epoch resumable training state for a run (deleted on completion)."""
    return MODELS_DIR / f"{run_name}_resume.pt"


def scaler_path(run_name: str) -> Path:
    """Fitted TurbidityScaler for a training run."""
    return SCALERS_DIR / f"{run_name}_scaler.pkl"


def predictions_dir(run_name: str) -> Path:
    """Directory holding a run's metrics and hourly prediction tables."""
    return PREDICTIONS_DIR / run_name


def interpretation_dir(run_name: str) -> Path:
    """Directory holding a run's Integrated Gradients caches."""
    return INTERPRETATION_DIR / run_name


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
