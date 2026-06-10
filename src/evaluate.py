"""Evaluation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import AverageMeter, topk_accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> dict[str, float]:
    """Evaluate a model and return loss, top-1, and top-5 accuracy."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for images, targets in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        acc1, acc5 = topk_accuracy(logits, targets, topk=(1, 5))

        batch_size = images.size(0)
        losses.update(float(loss.item()), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}
