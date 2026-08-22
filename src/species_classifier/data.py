"""iNaturalist annotation parsing, deterministic splits, and image datasets."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


CLASS_FIELDS = (
    "label_index",
    "category_id",
    "name",
    "common_name",
    "supercategory",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specific_epithet",
)

MANIFEST_FIELDS = (
    "split",
    "label_index",
    "category_id",
    "image_id",
    "file_name",
    "width",
    "height",
    "name",
    "common_name",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    for key in ("images", "annotations", "categories"):
        if key not in value:
            raise ValueError(f"{path} is missing '{key}'")
    return value


def _images_by_category(annotation: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    image_by_id = {int(image["id"]): image for image in annotation["images"]}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in annotation["annotations"]:
        category_id = int(row["category_id"])
        image_id = int(row["image_id"])
        if image_id not in image_by_id:
            raise ValueError(f"annotation references missing image_id={image_id}")
        grouped[category_id].append(image_by_id[image_id])
    for rows in grouped.values():
        rows.sort(key=lambda item: (str(item["file_name"]), int(item["id"])))
    return grouped


def _write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(
    split: str,
    category: dict[str, Any],
    image: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "label_index": category["label_index"],
        "category_id": category["category_id"],
        "image_id": image["id"],
        "file_name": image["file_name"],
        "width": image.get("width", ""),
        "height": image.get("height", ""),
        "name": category.get("name", ""),
        "common_name": category.get("common_name", ""),
    }


def prepare_manifests(config: dict[str, Any]) -> dict[str, Path]:
    """Create a category list and leakage-free split manifests."""
    dataset = config["dataset"]
    seed = int(config["seed"])
    train_annotation = _load_json(Path(dataset["train_annotations"]))
    test_annotation = _load_json(Path(dataset["test_annotations"]))
    train_pool = _images_by_category(train_annotation)
    test_pool = _images_by_category(test_annotation)

    required_train = int(dataset["train_per_class"]) + int(dataset["val_per_class"])
    required_test = int(dataset["test_per_class"])
    eligible = [
        category
        for category in train_annotation["categories"]
        if len(train_pool[int(category["id"])]) >= required_train
        and len(test_pool[int(category["id"])]) >= required_test
    ]
    num_classes = int(dataset["num_classes"])
    if len(eligible) < num_classes:
        raise ValueError(
            f"only {len(eligible)} categories satisfy the sample requirements; "
            f"requested {num_classes}"
        )

    selected = random.Random(seed).sample(eligible, num_classes)
    selected.sort(key=lambda category: int(category["id"]))
    classes: list[dict[str, Any]] = []
    for label_index, category in enumerate(selected):
        record = {field: category.get(field, "") for field in CLASS_FIELDS}
        record["label_index"] = label_index
        record["category_id"] = int(category["id"])
        classes.append(record)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for category in classes:
        category_id = int(category["category_id"])
        local_train = list(train_pool[category_id])
        random.Random(seed * 1_000_003 + category_id).shuffle(local_train)
        train_count = int(dataset["train_per_class"])
        validation_count = int(dataset["val_per_class"])
        test_count = int(dataset["test_per_class"])
        train_rows.extend(
            _manifest_row("train", category, image)
            for image in local_train[:train_count]
        )
        validation_rows.extend(
            _manifest_row("validation", category, image)
            for image in local_train[train_count : train_count + validation_count]
        )
        test_rows.extend(
            _manifest_row("test", category, image)
            for image in test_pool[category_id][:test_count]
        )

    validate_split_integrity(train_rows, validation_rows, test_rows, num_classes)
    output = Path(dataset["output_dir"])
    paths = {
        "classes": output / "classes.csv",
        "train": output / "train.csv",
        "validation": output / "validation.csv",
        "test": output / "test.csv",
    }
    _write_csv(paths["classes"], CLASS_FIELDS, classes)
    _write_csv(paths["train"], MANIFEST_FIELDS, train_rows)
    _write_csv(paths["validation"], MANIFEST_FIELDS, validation_rows)
    _write_csv(paths["test"], MANIFEST_FIELDS, test_rows)
    return paths


def validate_split_integrity(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_classes: int,
) -> None:
    groups = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    image_sets: dict[str, set[int]] = {}
    expected_labels = set(range(num_classes))
    for name, rows in groups.items():
        if not rows:
            raise ValueError(f"{name} split is empty")
        image_ids = [int(row["image_id"]) for row in rows]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{name} split contains duplicate image IDs")
        labels = {int(row["label_index"]) for row in rows}
        if labels != expected_labels:
            raise ValueError(f"{name} split does not cover the complete label space")
        image_sets[name] = set(image_ids)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = image_sets[left] & image_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} leakage detected for {len(overlap)} images")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def manifest_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = Path(config["dataset"]["output_dir"])
    return {
        "classes": output / "classes.csv",
        "train": output / "train.csv",
        "validation": output / "validation.csv",
        "test": output / "test.csv",
    }


class ManifestDataset(Dataset):
    """Image dataset backed by an explicit CSV manifest."""

    def __init__(self, root: str | Path, manifest: str | Path, transform=None):
        self.root = Path(root)
        self.rows = read_csv(manifest)
        self.transform = transform
        if not self.rows:
            raise ValueError(f"manifest contains no samples: {manifest}")
        labels = [int(row["label_index"]) for row in self.rows]
        counts = Counter(labels)
        expected = set(range(max(labels) + 1))
        if set(counts) != expected:
            raise ValueError("manifest labels must be contiguous from zero")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.root / row["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_index"])

