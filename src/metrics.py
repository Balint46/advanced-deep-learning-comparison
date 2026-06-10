"""Metric computation utilities."""

from __future__ import annotations

import torch


class AverageMeter:
    """Track a weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[float]:
    """Return top-k accuracies as percentages."""
    max_k = min(max(topk), logits.size(1))
    _, predictions = logits.topk(max_k, dim=1)
    predictions = predictions.t()
    correct = predictions.eq(targets.reshape(1, -1).expand_as(predictions))

    accuracies = []
    batch_size = targets.size(0)
    for k in topk:
        effective_k = min(k, logits.size(1))
        correct_k = correct[:effective_k].reshape(-1).float().sum(0)
        accuracies.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return accuracies
