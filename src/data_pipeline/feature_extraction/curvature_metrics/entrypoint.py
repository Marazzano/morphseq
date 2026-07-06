"""curvature_metrics entrypoint — thin filesystem adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_curvature_features
from .contract import validate_curvature_features


def run_curvature_metrics(
    *,
    snip_inventory_csv: Path,
    frame_masks_csv: Path,
    frame_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
) -> None:
    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_masks = pd.read_csv(frame_masks_csv)
    frame_inventory = pd.read_csv(frame_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    df = compute_curvature_features(snip_inventory, frame_masks, frame_inventory)
    validate_curvature_features(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
