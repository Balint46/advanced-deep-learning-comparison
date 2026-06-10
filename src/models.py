"""Model factory utilities for CNN and transformer classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

from .heads import create_head


MODEL_NAMES = {
    "small_cnn",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "vit_tiny",
    "vit_small",
}


TIMM_MODELS = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
}


class SmallCNNBackbone(nn.Module):
    """Compact CNN backbone that trains quickly on CIFAR-sized images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.num_features = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


class ClassifierModel(nn.Module):
    """Backbone plus configurable classifier head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


@dataclass(frozen=True)
class ModelInfo:
    model: nn.Module
    feature_dim: int


def _create_timm_backbone(model_name: str, pretrained: bool) -> ModelInfo:
    timm_name = TIMM_MODELS[model_name]
    kwargs = {"pretrained": pretrained, "num_classes": 0, "global_pool": "avg"}
    if model_name.startswith("vit_"):
        kwargs.update({"img_size": 32})

    backbone = timm.create_model(timm_name, **kwargs)
    feature_dim = int(backbone.num_features)
    return ModelInfo(model=backbone, feature_dim=feature_dim)


def create_model(
    model_name: str,
    num_classes: int,
    head_type: str = "linear",
    pretrained: bool = True,
) -> nn.Module:
    """Create a model with a replaceable classification head."""
    model_name = model_name.lower()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model '{model_name}'. Options: {sorted(MODEL_NAMES)}")

    if model_name == "small_cnn":
        backbone = SmallCNNBackbone()
        feature_dim = backbone.num_features
    else:
        info = _create_timm_backbone(model_name, pretrained=pretrained)
        backbone = info.model
        feature_dim = info.feature_dim

    head = create_head(head_type, feature_dim, num_classes)
    return ClassifierModel(backbone, head)


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
