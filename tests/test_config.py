from pathlib import Path

import pytest
import yaml

from species_classifier.config import load_config


def _base_config():
    return {
        "seed": 7,
        "dataset": {
            "root": "data/raw",
            "train_annotations": "data/raw/train.json",
            "test_annotations": "data/raw/test.json",
            "output_dir": "data/processed/example",
            "num_classes": 3,
            "train_per_class": 2,
            "val_per_class": 1,
            "test_per_class": 1,
            "image_size": 32,
        },
        "model": {"architecture": "resnet18", "pretrained": False},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "run": {"name": "test", "checkpoint_dir": "checkpoints", "output_dir": "outputs"},
    }


def test_config_inheritance_deep_merges(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    child.write_text(
        yaml.safe_dump(
            {
                "inherit": "base.yaml",
                "model": {"architecture": "resnet50"},
                "run": {"name": "child"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(child)

    assert config["model"] == {"architecture": "resnet50", "pretrained": False}
    assert config["training"]["epochs"] == 2
    assert config["run"]["name"] == "child"


def test_config_rejects_unknown_architecture(tmp_path: Path):
    config = _base_config()
    config["model"]["architecture"] = "unknown"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="architecture"):
        load_config(path)

