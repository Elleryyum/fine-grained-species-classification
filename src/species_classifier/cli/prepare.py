"""Prepare deterministic iNaturalist class and split manifests."""

from __future__ import annotations

import argparse

from species_classifier.config import load_config
from species_classifier.data import prepare_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    paths = prepare_manifests(config)
    print("Prepared deterministic manifests:")
    for name, path in paths.items():
        print(f"  {name:<10} {path}")


if __name__ == "__main__":
    main()

