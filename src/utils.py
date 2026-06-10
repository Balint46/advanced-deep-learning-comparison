"""General utility functions for reproducible experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers from the PyTorch initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_device(requested: str | None = None) -> torch.device:
    """Return the requested device, falling back to CUDA/CPU automatically."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    """Create runtime output directories."""
    for path in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable model parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
