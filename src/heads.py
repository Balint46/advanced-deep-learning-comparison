"""Configurable classification heads."""

from __future__ import annotations

import torch.nn as nn


def create_head(
    head_type: str,
    in_features: int,
    num_classes: int,
    hidden_dim: int = 512,
    dropout: float = 0.2,
) -> nn.Module:
    """Build one of the assignment's requested classification heads."""
    head_type = head_type.lower()

    if head_type == "linear":
        return nn.Linear(in_features, num_classes)

    if head_type == "one_hidden":
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    if head_type == "two_hidden":
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    raise ValueError(
        f"Unsupported head_type '{head_type}'. "
        "Choose linear, one_hidden, or two_hidden."
    )
