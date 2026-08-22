"""Model prediction, classification metrics, and evaluation artifacts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .reproducibility import seed_worker, seeded_generator


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=seeded_generator(seed),
        persistent_workers=num_workers > 0,
    )
    model.eval().to(device)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=device.type == "cuda"))
        probabilities = logits.softmax(dim=1).cpu().numpy()
        targets.append(labels.numpy())
        predictions.append(probabilities.argmax(axis=1))
        scores.append(probabilities)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(scores),
        elapsed,
    )


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(scores.shape[1]),
        average="macro",
        zero_division=0,
    )
    top_k = min(5, scores.shape[1])
    top_indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    top5 = np.any(top_indices == targets[:, None], axis=1).mean()
    return {
        "top1_accuracy": float(accuracy_score(targets, predictions)),
        "top5_accuracy": float(top5),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _class_names(class_records: list[dict[str, str]]) -> list[str]:
    names = []
    for record in class_records:
        names.append(record.get("common_name") or record.get("name") or record["category_id"])
    return names


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> None:
    fields = (
        "image_id",
        "file_name",
        "true_label",
        "true_name",
        "predicted_label",
        "predicted_name",
        "confidence",
        "correct",
        "top5_labels",
        "top5_names",
        "top5_scores",
    )
    top_k = min(5, scores.shape[1])
    top_indices = np.argsort(-scores, axis=1)[:, :top_k]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            top = top_indices[index]
            true_label = int(targets[index])
            predicted_label = int(predictions[index])
            writer.writerow(
                {
                    "image_id": row.get("image_id", ""),
                    "file_name": row["file_name"],
                    "true_label": true_label,
                    "true_name": names[true_label],
                    "predicted_label": predicted_label,
                    "predicted_name": names[predicted_label],
                    "confidence": f"{scores[index, predicted_label]:.8f}",
                    "correct": int(true_label == predicted_label),
                    "top5_labels": "|".join(str(int(label)) for label in top),
                    "top5_names": "|".join(names[int(label)] for label in top),
                    "top5_scores": "|".join(f"{scores[index, int(label)]:.8f}" for label in top),
                }
            )


def _plot_confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> np.ndarray:
    matrix = confusion_matrix(targets, predictions, normalize="true")
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Normalized confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return matrix


def _write_confused_pairs(path: Path, matrix: np.ndarray, names: list[str], top: int = 30) -> None:
    counts = matrix.copy()
    np.fill_diagonal(counts, 0)
    ranked = np.dstack(np.unravel_index(np.argsort(counts.ravel())[::-1], counts.shape))[0]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("true_label", "true_name", "predicted_label", "predicted_name", "rate"),
        )
        writer.writeheader()
        written = 0
        for true_label, predicted_label in ranked:
            rate = float(counts[true_label, predicted_label])
            if rate <= 0 or written >= top:
                break
            writer.writerow(
                {
                    "true_label": int(true_label),
                    "true_name": names[int(true_label)],
                    "predicted_label": int(predicted_label),
                    "predicted_name": names[int(predicted_label)],
                    "rate": f"{rate:.6f}",
                }
            )
            written += 1


def export_evaluation(
    output_dir: str | Path,
    manifest_rows: list[dict[str, str]],
    class_records: list[dict[str, str]],
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    elapsed_seconds: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(targets, predictions, scores)
    timing: dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "images": len(targets),
        "inference_seconds": elapsed_seconds,
        "images_per_second": len(targets) / elapsed_seconds,
        "milliseconds_per_image": elapsed_seconds * 1000 / len(targets),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    names = _class_names(class_records)
    _write_predictions(output / "predictions.csv", manifest_rows, targets, predictions, scores, names)
    matrix = _plot_confusion(targets, predictions, output / "confusion_matrix.png")
    _write_confused_pairs(output / "confused_pairs.csv", matrix, names)
    return metrics


def plot_history(history: dict[str, list[float]], path: str | Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_top1"], label="train Top-1")
    axes[1].plot(epochs, history["validation_top1"], label="validation Top-1")
    axes[1].plot(epochs, history["validation_top5"], label="validation Top-5")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def print_metrics(metrics: dict[str, float]) -> None:
    print("\nTest metrics")
    print("------------")
    for name, value in metrics.items():
        print(f"{name:<22} {value:.4f}")

