"""Train the phi0-only model (Stage 1).

Usage:
    python -m dev.dynamo.scripts.train --config dev/dynamo/configs/stage1.yaml
    python -m dev.dynamo.scripts.train --config dev/dynamo/configs/stage1.yaml --lr 5e-4
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import yaml

from dev.dynamo.training.trainer import TrainConfig, Stage1Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phi0-only model (Stage 1)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")

    # Register all TrainConfig fields as optional CLI overrides
    for f in dataclasses.fields(TrainConfig):
        flag = f"--{f.name}"
        if f.type in ("str", "Optional[str]"):
            parser.add_argument(flag, type=str, default=None)
        elif f.type in ("int", "Optional[int]"):
            parser.add_argument(flag, type=int, default=None)
        elif f.type == "float":
            parser.add_argument(flag, type=float, default=None)
        elif f.type == "bool":
            parser.add_argument(flag, type=lambda x: x.lower() == "true", default=None)
        # Skip complex types (List, Tuple) — set those in YAML

    args = parser.parse_args()

    # Load YAML config
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    # Apply CLI overrides
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg_dict[k] = v

    # Handle tuple fields
    if "horizons" in cfg_dict and isinstance(cfg_dict["horizons"], list):
        cfg_dict["horizons"] = tuple(cfg_dict["horizons"])

    # Filter to valid TrainConfig fields
    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    config = TrainConfig(**cfg_dict)
    trainer = Stage1Trainer(config)
    best_path = trainer.train()
    print(f"\nBest checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
