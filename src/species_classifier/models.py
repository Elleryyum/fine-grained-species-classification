"""Model factory for controlled architecture and initialization comparisons."""

from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

