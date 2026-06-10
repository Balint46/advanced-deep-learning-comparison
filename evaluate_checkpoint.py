"""Evaluate a saved checkpoint on the CIFAR test set without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.datasets import get_dataloaders
from src.evaluate import evaluate
from src.models import create_model
from src.utils import CHECKPOINT_DIR, get_device


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CIFAR checkpoint on the test split."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a .pth checkpoint file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size. Defaults to the checkpoint config value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example cuda or cpu.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        if checkpoint_path.parent == Path("."):
            checkpoint_path = CHECKPOINT_DIR / checkpoint_path
        else:
            checkpoint_path = CHECKPOINT_DIR.parent / checkpoint_path

    device = get_device(args.device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})

    dataset = checkpoint["dataset"]
    model_name = checkpoint["model_name"]
    num_classes = int(checkpoint["num_classes"])
    head_type = checkpoint["head_type"]
    batch_size = args.batch_size or int(config.get("batch_size", 128))

    loaders = get_dataloaders(
        dataset_name=dataset,
        batch_size=batch_size,
        val_fraction=float(config.get("val_fraction", 0.1)),
        num_workers=args.num_workers,
        seed=int(config.get("seed", 42)),
    )
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        head_type=head_type,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, loaders.test, criterion, device, desc="Testing checkpoint")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Training strategy: {checkpoint['training_strategy']}")
    print(f"Head: {head_type}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Best validation top-1: {checkpoint['best_val_top1']:.2f}%")
    print(f"Best validation top-5: {checkpoint['best_val_top5']:.2f}%")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Top-1 accuracy: {test_metrics['top1']:.2f}%")
    print(f"Top-5 accuracy: {test_metrics['top5']:.2f}%")


if __name__ == "__main__":
    main()
