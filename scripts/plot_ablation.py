"""Create a compact chart from the archived scratch ablation table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmarks" / "scratch_ablation.csv"
OUTPUT = ROOT / "docs" / "assets" / "scratch_ablation.png"


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = [f"{row['architecture']}\naugmentation {row['augmentation']}" for row in rows]
    top1 = [float(row["top1"]) for row in rows]
    top5 = [float(row["top5"]) for row in rows]
    macro_f1 = [float(row["macro_f1"]) for row in rows]

    positions = np.arange(len(rows))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(positions - width, top1, width, label="Top-1", color="#2f6f8f")
    axis.bar(positions, top5, width, label="Top-5", color="#6aa84f")
    axis.bar(positions + width, macro_f1, width, label="Macro-F1", color="#c55a3d")
    axis.set_ylabel("Score (%)")
    axis.set_title("From-scratch architecture and augmentation ablation")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 65)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()

