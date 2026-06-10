"""Experiment orchestration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
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

    @property
    def checkpoint_name(self) -> str:
        return (
            f"{self.dataset}_{self.model_name}_{self.training_strategy}_"
            f"{self.head_type}_best.pth"
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

    checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name
    config_dict = asdict(config)
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
        checkpoint_metadata={
            "dataset": config.dataset,
            "model_name": config.model_name,
            "num_classes": loaders.num_classes,
            "head_type": config.head_type,
            "training_strategy": config.training_strategy,
            "config": config_dict,
        },
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
        "best_val_top1": training_result.best_val_top1,
        "best_val_top5": training_result.best_val_top5,
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

    return {**summary, "config": config_dict}


def configs_from_dicts(items: list[dict]) -> list[ExperimentConfig]:
    return [ExperimentConfig(**item) for item in items]


def make_hyperparameter_search_configs(search_config: dict) -> list[ExperimentConfig]:
    """Create the constrained CIFAR-10 grid search requested in the assignment."""
    configs: list[ExperimentConfig] = []
    for model_name, learning_rate, batch_size, weight_decay in product(
        search_config["models"],
        search_config["learning_rates"],
        search_config["batch_sizes"],
        search_config["weight_decays"],
    ):
        configs.append(
            ExperimentConfig(
                dataset=search_config.get("dataset", "cifar10"),
                model_name=model_name,
                training_strategy=search_config.get("training_strategy", "freeze_then_unfreeze"),
                head_type=search_config.get("head_type", "linear"),
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                epochs=search_config.get("epochs", 4),
                frozen_epochs=search_config.get("frozen_epochs", 2),
                seed=search_config.get("seed", 42),
                pretrained=search_config.get("pretrained", True),
            )
        )
    return configs


def best_hyperparameters_by_model(summary_path: Path = SUMMARY_PATH) -> dict[str, dict]:
    """Return the best CIFAR-10 row per model using test top-1 accuracy."""
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary table found at {summary_path}. Run the CIFAR-10 search first."
        )

    summary = pd.read_csv(summary_path)
    candidates = summary[
        (summary["dataset"] == "cifar10")
        & (summary["model_name"].isin(["resnet18", "vit_tiny"]))
    ]
    if candidates.empty:
        raise ValueError("No CIFAR-10 ResNet18 or ViT-Tiny rows found in summary_table.csv.")

    best_rows = candidates.sort_values("top1_accuracy", ascending=False).groupby("model_name")
    return {
        model_name: group.iloc[0].to_dict()
        for model_name, group in best_rows
    }


def make_cifar100_configs_from_best(
    summary_path: Path = SUMMARY_PATH,
    epochs: int = 4,
) -> list[ExperimentConfig]:
    """Reuse the best CIFAR-10 hyperparameters for CIFAR-100 experiments."""
    best = best_hyperparameters_by_model(summary_path)
    configs = []
    for row in best.values():
        configs.append(
            ExperimentConfig(
                dataset="cifar100",
                model_name=row["model_name"],
                training_strategy=row["training_strategy"],
                head_type=row["head_type"],
                learning_rate=float(row["learning_rate"]),
                batch_size=int(row["batch_size"]),
                weight_decay=float(row["weight_decay"]),
                epochs=epochs,
                frozen_epochs=int(row["frozen_epochs"]),
                seed=int(row["seed"]),
                pretrained=True,
            )
        )
    return configs
