"""Experiment orchestration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from .datasets import get_dataloaders
from .evaluate import evaluate
from .models import create_model
from .train import train_model
from .utils import CHECKPOINT_DIR, RESULTS_DIR, count_parameters, ensure_dirs, get_device, set_seed


METRICS_PATH = RESULTS_DIR / "metrics.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_table.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model_name: str
    training_strategy: str
    head_type: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    epochs: int
    frozen_epochs: int = 0
    seed: int = 42
    pretrained: bool = True
    device: str | None = None
    num_workers: int = 2
    val_fraction: float = 0.1

    @property
    def experiment_id(self) -> str:
        lr = f"{self.learning_rate:g}".replace(".", "p")
        wd = f"{self.weight_decay:g}".replace(".", "p")
        return (
            f"{self.dataset}_{self.model_name}_{self.training_strategy}_"
            f"{self.head_type}_lr{lr}_bs{self.batch_size}_wd{wd}_seed{self.seed}"
        )


def _append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def run_experiment(config: ExperimentConfig) -> dict:
    """Run one experiment and append metrics plus final summary rows."""
    ensure_dirs()
    set_seed(config.seed)
    device = get_device(config.device)

    loaders = get_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model = create_model(
        model_name=config.model_name,
        num_classes=loaders.num_classes,
        head_type=config.head_type,
        pretrained=config.pretrained,
    )

    checkpoint_path = CHECKPOINT_DIR / f"{config.experiment_id}.pt"
    training_result = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        training_strategy=config.training_strategy,
        frozen_epochs=config.frozen_epochs,
        checkpoint_path=checkpoint_path,
    )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, loaders.test, criterion, device, desc="Testing")
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    last_epoch = training_result.history[-1]
    generalization_gap = float(last_epoch["train_top1"]) - float(last_epoch["val_top1"])

    shared = {
        "experiment_id": config.experiment_id,
        "dataset": config.dataset,
        "model_name": config.model_name,
        "training_strategy": config.training_strategy,
        "head_type": config.head_type,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "frozen_epochs": config.frozen_epochs,
        "seed": config.seed,
    }
    metric_rows = [{**shared, **row} for row in training_result.history]
    _append_csv(METRICS_PATH, metric_rows)

    summary = {
        "dataset": config.dataset,
        "model_name": config.model_name,
        "training_strategy": config.training_strategy,
        "head_type": config.head_type,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "frozen_epochs": config.frozen_epochs,
        "top1_accuracy": test_metrics["top1"],
        "top5_accuracy": test_metrics["top5"],
        "test_loss": test_metrics["loss"],
        "best_val_loss": training_result.best_val_loss,
        "best_val_epoch": training_result.best_val_epoch,
        "num_parameters": total_params,
        "trainable_parameters": trainable_params,
        "average_time_per_epoch": training_result.average_time_per_epoch,
        "total_training_time": training_result.total_training_time,
        "seed": config.seed,
        "generalization_gap": generalization_gap,
        "experiment_id": config.experiment_id,
    }
    _append_csv(SUMMARY_PATH, [summary])

    return {**summary, "config": asdict(config)}


def configs_from_dicts(items: list[dict]) -> list[ExperimentConfig]:
    return [ExperimentConfig(**item) for item in items]
