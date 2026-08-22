"""Return Top-5 predictions for one image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
import torch

from species_classifier.config import load_config
from species_classifier.runtime import load_checkpoint_model, select_device
from species_classifier.transforms import evaluation_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _display_name(record: dict[str, str]) -> str:
    return record.get("common_name") or record.get("name") or record["category_id"]


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    model, _, class_records = load_checkpoint_model(config, args.checkpoint, device)
    transform = evaluation_transform(int(config["dataset"]["image_size"]))
    with Image.open(Path(args.image)) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        probabilities = model(tensor).softmax(dim=1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

    values, indices = probabilities.topk(min(5, len(class_records)))
    print(f"device={device} inference={elapsed_ms:.2f}ms")
    print("Top-5 predictions")
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {_display_name(class_records[index])}: {value * 100:.2f}%")


if __name__ == "__main__":
    main()

