"""YAML configuration loading with inheritance and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_parent(config_path: Path, parent_value: str) -> Path:
    parent = Path(parent_value)
    if parent.is_absolute():
        return parent
    if parent.exists():
        return parent.resolve()
    return (config_path.parent / parent).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config, recursively merge its parent, and validate the result."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")

    parent_value = current.pop("inherit", None)
    config = current
    if parent_value:
        parent = _resolve_parent(config_path, str(parent_value))
        config = _deep_merge(load_config(parent), current)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"seed", "dataset", "model", "training", "run"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")

    dataset = config["dataset"]
    for field in (
        "root",
        "train_annotations",
        "test_annotations",
        "output_dir",
        "num_classes",
        "train_per_class",
        "val_per_class",
        "test_per_class",
        "image_size",
    ):
        if field not in dataset:
            raise ValueError(f"dataset.{field} is required")
    for field in (
        "num_classes",
        "train_per_class",
        "val_per_class",
        "test_per_class",
        "image_size",
    ):
        if int(dataset[field]) <= 0:
            raise ValueError(f"dataset.{field} must be positive")

    model = config["model"]
    if model.get("architecture") not in {"resnet18", "resnet50"}:
        raise ValueError("model.architecture must be resnet18 or resnet50")
    if not isinstance(model.get("pretrained"), bool):
        raise ValueError("model.pretrained must be true or false")

    training = config["training"]
    for field in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        if field not in training:
            raise ValueError(f"training.{field} is required")
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("training epochs and batch size must be positive")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

