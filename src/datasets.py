"""Dataset loading utilities for CIFAR-10 and CIFAR-100."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import DATA_DIR, seed_worker


CIFAR_NORMALIZATION = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "classes": 10,
        "dataset": datasets.CIFAR10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "classes": 100,
        "dataset": datasets.CIFAR100,
    },
}


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int


def _transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    stats = CIFAR_NORMALIZATION[dataset_name]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )
    return train_transform, eval_transform


def _split_train_val(
    dataset_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = int(dataset_size * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    download: bool = True,
) -> DataLoaders:
    """Create train, validation, and test dataloaders for CIFAR datasets."""
    dataset_name = dataset_name.lower()
    if dataset_name not in CIFAR_NORMALIZATION:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Use cifar10 or cifar100.")

    train_transform, eval_transform = _transforms(dataset_name)
    dataset_cls = CIFAR_NORMALIZATION[dataset_name]["dataset"]

    train_full = dataset_cls(DATA_DIR, train=True, transform=train_transform, download=download)
    val_full = dataset_cls(DATA_DIR, train=True, transform=eval_transform, download=download)
    test_set = dataset_cls(DATA_DIR, train=False, transform=eval_transform, download=download)

    train_indices, val_indices = _split_train_val(len(train_full), val_fraction, seed)
    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    return DataLoaders(
        train=DataLoader(train_set, shuffle=True, **loader_args),
        val=DataLoader(val_set, shuffle=False, **loader_args),
        test=DataLoader(test_set, shuffle=False, **loader_args),
        num_classes=int(CIFAR_NORMALIZATION[dataset_name]["classes"]),
    )
