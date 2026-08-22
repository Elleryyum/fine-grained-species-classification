"""Train, validate, and finally evaluate a species classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from species_classifier.config import load_config, save_resolved_config
from species_classifier.engine import train_model
from species_classifier.evaluation import (
    export_evaluation,
    plot_history,
    predict_dataset,
    print_metrics,
)
from species_classifier.models import build_model, parameter_count
from species_classifier.reproducibility import set_seed
from species_classifier.runtime import build_datasets, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", help="last checkpoint from an interrupted run")
    parser.add_argument("--epochs", type=int, help="override configured epoch count")
    parser.add_argument("--batch-size", type=int, help="override configured batch size")
    parser.add_argument("--num-workers", type=int, help="override DataLoader workers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    train_dataset, validation_dataset, test_dataset, class_records, test_rows = (
        build_datasets(config)
    )
    expected_classes = int(config["dataset"]["num_classes"])
    if len(class_records) != expected_classes:
        raise ValueError(
            f"classes.csv has {len(class_records)} classes; expected {expected_classes}"
        )

    model = build_model(
        config["model"]["architecture"],
        len(class_records),
        bool(config["model"]["pretrained"]),
    )
    output_dir = Path(config["run"]["output_dir"]) / str(config["run"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    print(
        f"device={device} classes={len(class_records)} "
        f"train={len(train_dataset)} validation={len(validation_dataset)} "
        f"test={len(test_dataset)} parameters={parameter_count(model):,}"
    )
    best_path, history = train_model(
        model,
        train_dataset,
        validation_dataset,
        config,
        class_records,
        device,
        resume=args.resume,
    )
    plot_history(history, output_dir / "training_curves.png")

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    targets, predictions, scores, elapsed = predict_dataset(
        model,
        test_dataset,
        device,
        int(config["training"]["batch_size"]),
        int(config["training"].get("num_workers", 2)),
        int(config["seed"]),
    )
    metrics = export_evaluation(
        output_dir / "test",
        test_rows,
        class_records,
        targets,
        predictions,
        scores,
        elapsed,
        device,
        int(config["training"]["batch_size"]),
    )
    print_metrics(metrics)
    print(f"\nBest checkpoint: {best_path}")
    print(f"Artifacts:       {output_dir}")


if __name__ == "__main__":
    main()

