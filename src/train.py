"""Training loop utilities."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import AverageMeter, topk_accuracy


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
