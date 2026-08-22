"""Evaluate a frozen checkpoint on the complete held-out test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from species_classifier.config import load_config
from species_classifier.data import ManifestDataset, read_csv
from species_classifier.evaluation import (
    export_evaluation,
    predict_dataset,
    print_metrics,
)
from species_classifier.reproducibility import set_seed
from species_classifier.runtime import load_checkpoint_model, require_manifests, select_device
from species_classifier.transforms import evaluation_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    paths = require_manifests(config)
    model, state, class_records = load_checkpoint_model(config, args.checkpoint, device)
    test_rows = read_csv(paths["test"])
    test_dataset = ManifestDataset(
        config["dataset"]["root"],
        paths["test"],
        evaluation_transform(int(config["dataset"]["image_size"])),
    )
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["training"].get("num_workers", 2))
    )
    targets, predictions, scores, elapsed = predict_dataset(
        model,
        test_dataset,
        device,
        batch_size,
        workers,
        int(config["seed"]),
    )
    output = args.output_dir or (
        Path(config["run"]["output_dir"]) / str(config["run"]["name"]) / "evaluation"
    )
    metrics = export_evaluation(
        output,
        test_rows,
        class_records,
        targets,
        predictions,
        scores,
        elapsed,
        device,
        batch_size,
    )
    print(f"checkpoint epoch: {int(state.get('epoch', -1)) + 1}")
    print_metrics(metrics)
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()

