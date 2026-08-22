"""Training loop, validation, checkpointing, and resume support."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .reproducibility import seed_worker, seeded_generator


def _topk_counts(logits: torch.Tensor, targets: torch.Tensor, values=(1, 5)) -> dict[int, int]:
    maximum = min(max(values), logits.shape[1])
    predictions = logits.topk(maximum, dim=1).indices
    matches = predictions.eq(targets.view(-1, 1))
    return {
        value: int(matches[:, : min(value, maximum)].any(dim=1).sum().item())
        for value in values
    }


def _loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=seeded_generator(seed),
        persistent_workers=workers > 0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top5_correct = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.shape[0]
            counts = _topk_counts(logits, targets)
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            top1_correct += counts[1]
            top5_correct += counts[5]

    return {
        "loss": total_loss / total_examples,
        "top1": top1_correct / total_examples,
        "top5": top5_correct / total_examples,
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: dict[str, Any],
    class_records: list[dict[str, str]],
    device: torch.device,
    resume: str | Path | None = None,
) -> tuple[Path, dict[str, list[float]]]:
    """Train a model and return the validation-selected checkpoint and history."""
    training = config["training"]
    run = config["run"]
    seed = int(config["seed"])
    model.to(device)
    train_loader = _loader(
        train_dataset,
        int(training["batch_size"]),
        int(training.get("num_workers", 2)),
        True,
        seed,
        device,
    )
    validation_loader = _loader(
        validation_dataset,
        int(training["batch_size"]),
        int(training.get("num_workers", 2)),
        False,
        seed,
        device,
    )

    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("supported optimizer: adamw")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if str(training.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("supported scheduler: cosine")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(run["checkpoint_dir"]) / str(run["name"])
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    history = {
        "train_loss": [],
        "train_top1": [],
        "validation_loss": [],
        "validation_top1": [],
        "validation_top5": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }
    start_epoch = 0
    best_validation_top1 = -1.0

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_validation_top1 = float(state["best_validation_top1"])
        history = state["history"]

    epochs = int(training["epochs"])
    save_every = int(run.get("save_every", 5))
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        elapsed = time.perf_counter() - started
        history["train_loss"].append(train_metrics["loss"])
        history["train_top1"].append(train_metrics["top1"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_top1"].append(validation_metrics["top1"])
        history["validation_top5"].append(validation_metrics["top5"])
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["epoch_seconds"].append(elapsed)

        improved = validation_metrics["top1"] > best_validation_top1
        if improved:
            best_validation_top1 = validation_metrics["top1"]
        scheduler.step()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_top1": best_validation_top1,
            "history": history,
            "classes": class_records,
            "config": config,
        }
        if improved:
            _save_checkpoint(best_path, state)
        if (epoch + 1) % save_every == 0:
            _save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:03d}.pth", state)
        _save_checkpoint(last_path, state)
        print(
            f"epoch {epoch + 1:>3}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.4f} | "
            f"validation loss {validation_metrics['loss']:.4f} "
            f"top1 {validation_metrics['top1']:.4f} "
            f"top5 {validation_metrics['top5']:.4f} | {elapsed:.1f}s"
        )

    history_path = Path(run["output_dir"]) / str(run["name"]) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path, history
