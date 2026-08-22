"""Shared runtime helpers for command-line entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from .data import ManifestDataset, manifest_paths, read_csv
from .models import build_model
from .transforms import evaluation_transform, training_transform


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def require_manifests(config):
    paths = manifest_paths(config)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "dataset manifests are missing; run species-prepare first:\n" + "\n".join(missing)
        )
    return paths


def build_datasets(config):
    paths = require_manifests(config)
    dataset = config["dataset"]
    image_size = int(dataset["image_size"])
    root = Path(dataset["root"])
    return (
        ManifestDataset(
            root,
            paths["train"],
            training_transform(image_size, bool(config["training"]["augmentation"])),
        ),
        ManifestDataset(root, paths["validation"], evaluation_transform(image_size)),
        ManifestDataset(root, paths["test"], evaluation_transform(image_size)),
        read_csv(paths["classes"]),
        read_csv(paths["test"]),
    )


def load_checkpoint_model(config, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_records = state.get("classes")
    if not class_records:
        raise ValueError("checkpoint does not contain its ordered class metadata")
    model = build_model(
        config["model"]["architecture"],
        len(class_records),
        pretrained=False,
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state, class_records

