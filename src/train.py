"""Training loop utilities."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import evaluate
from .metrics import AverageMeter, topk_accuracy
from .models import freeze_backbone, unfreeze_all
from .utils import count_parameters


TRAINING_STRATEGIES = {"full_training", "head_only", "freeze_then_unfreeze"}


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float | int | str]]
    best_val_loss: float
    best_val_top1: float
    best_val_top5: float
    best_val_epoch: int
    total_training_time: float
    average_time_per_epoch: float
    epochs_run: int
    stopped_early: bool


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch and return aggregate metrics."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    start = time.perf_counter()

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        acc1, acc5 = topk_accuracy(logits.detach(), targets, topk=(1, 5))
        batch_size = images.size(0)
        losses.update(float(loss.item()), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        progress.set_postfix(loss=f"{losses.avg:.4f}", top1=f"{top1.avg:.2f}")

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
        "time": time.perf_counter() - start,
    }


def _make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters available for the optimizer.")
    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _apply_strategy_at_start(model: nn.Module, strategy: str) -> None:
    if strategy == "full_training":
        unfreeze_all(model)
    elif strategy in {"head_only", "freeze_then_unfreeze"}:
        freeze_backbone(model)
    else:
        raise ValueError(
            f"Unsupported training_strategy '{strategy}'. "
            f"Choose from {sorted(TRAINING_STRATEGIES)}."
        )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    training_strategy: str,
    frozen_epochs: int = 0,
    patience: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict | None = None,
    fine_tune_lr_multiplier: float = 0.1,
) -> TrainingResult:
    """Train a model with full, head-only, or freeze-then-unfreeze strategy.

    patience > 0 enables early stopping: training halts after `patience` consecutive
    epochs with no improvement in val_top1, then best-epoch weights are restored.
    For freeze_then_unfreeze the patience counter resets at the start of the unfrozen
    phase and is not applied during the frozen warmup.
    """
    training_strategy = training_strategy.lower()
    _apply_strategy_at_start(model, training_strategy)

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, learning_rate, weight_decay)
    model.to(device)

    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    best_val_top1 = float("-inf")
    best_val_top5 = float("-inf")
    best_val_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0
    stopped_early = False
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if training_strategy == "freeze_then_unfreeze" and epoch == frozen_epochs + 1:
            unfreeze_all(model)
            optimizer = _make_optimizer(
                model,
                learning_rate * fine_tune_lr_multiplier,
                weight_decay,
            )
            patience_counter = 0  # fresh counter for the unfrozen phase

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Validation {epoch}",
        )

        improved = val_metrics["top1"] > best_val_top1
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_top1 = val_metrics["top1"]
            best_val_top5 = val_metrics["top5"]
            best_val_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        **(checkpoint_metadata or {}),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_top1": best_val_top1,
                        "best_val_top5": best_val_top5,
                    },
                    checkpoint_path,
                )
        else:
            # Count patience only after the warmup phase for freeze_then_unfreeze
            in_patience_phase = training_strategy != "freeze_then_unfreeze" or epoch > frozen_epochs
            if patience > 0 and in_patience_phase:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped_early = True
                    history.append(
                        {
                            "epoch": epoch,
                            "phase": "train_val",
                            "train_loss": train_metrics["loss"],
                            "train_top1": train_metrics["top1"],
                            "train_top5": train_metrics["top5"],
                            "val_loss": val_metrics["loss"],
                            "val_top1": val_metrics["top1"],
                            "val_top5": val_metrics["top5"],
                            "epoch_time": train_metrics["time"],
                            "trainable_parameters": count_parameters(model, trainable_only=True),
                        }
                    )
                    break

        history.append(
            {
                "epoch": epoch,
                "phase": "train_val",
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "epoch_time": train_metrics["time"],
                "trainable_parameters": count_parameters(model, trainable_only=True),
            }
        )

    # Restore the best-epoch weights before final evaluation
    model.load_state_dict(best_state_dict)

    total_training_time = time.perf_counter() - total_start
    average_time = sum(float(row["epoch_time"]) for row in history) / max(1, len(history))
    return TrainingResult(
        history=history,
        best_val_loss=best_val_loss,
        best_val_top1=best_val_top1,
        best_val_top5=best_val_top5,
        best_val_epoch=best_val_epoch,
        total_training_time=total_training_time,
        average_time_per_epoch=average_time,
        epochs_run=len(history),
        stopped_early=stopped_early,
    )
