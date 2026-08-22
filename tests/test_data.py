import csv
import json
from pathlib import Path

from PIL import Image

from species_classifier.data import ManifestDataset, prepare_manifests, read_csv


def _annotation(category_count: int, images_per_category: int, prefix: str, start_id: int):
    categories = []
    images = []
    annotations = []
    image_id = start_id
    for category_id in range(100, 100 + category_count):
        categories.append(
            {
                "id": category_id,
                "name": f"Species {category_id}",
                "common_name": f"Common {category_id}",
            }
        )
        for index in range(images_per_category):
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{prefix}/{category_id}_{index}.jpg",
                    "width": 20,
                    "height": 20,
                }
            )
            annotations.append(
                {"id": image_id, "image_id": image_id, "category_id": category_id}
            )
            image_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def _config(tmp_path: Path):
    return {
        "seed": 42,
        "dataset": {
            "root": str(tmp_path),
            "train_annotations": str(tmp_path / "train.json"),
            "test_annotations": str(tmp_path / "test.json"),
            "output_dir": str(tmp_path / "processed"),
            "num_classes": 2,
            "train_per_class": 2,
            "val_per_class": 1,
            "test_per_class": 1,
            "image_size": 16,
        },
        "model": {"architecture": "resnet18", "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "run": {"name": "test", "checkpoint_dir": "checkpoints", "output_dir": "outputs"},
    }


def test_prepare_manifests_is_balanced_and_leakage_free(tmp_path: Path):
    train = _annotation(3, 4, "train", 1_000)
    test = _annotation(3, 2, "test", 2_000)
    (tmp_path / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (tmp_path / "test.json").write_text(json.dumps(test), encoding="utf-8")

    paths = prepare_manifests(_config(tmp_path))
    rows = {name: read_csv(path) for name, path in paths.items() if name != "classes"}

    assert len(read_csv(paths["classes"])) == 2
    assert len(rows["train"]) == 4
    assert len(rows["validation"]) == 2
    assert len(rows["test"]) == 2
    ids = {name: {row["image_id"] for row in split} for name, split in rows.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_manifest_dataset_loads_rgb_image(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (12, 8), "red").save(image_dir / "sample.jpg")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file_name", "label_index"))
        writer.writeheader()
        writer.writerow({"file_name": "images/sample.jpg", "label_index": 0})

    dataset = ManifestDataset(tmp_path, manifest)
    image, label = dataset[0]

    assert image.mode == "RGB"
    assert image.size == (12, 8)
    assert label == 0

