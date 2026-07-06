"""mask_geometry entrypoint — the thin filesystem adapter.

Loads inputs (snip_inventory + frame_masks + frame_inventory + the per-well registry),
computes the feature table, validates it with the registry as verifier (check_sources=True),
then writes the per-well shard. No path-minting and no domain logic live here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_mask_geometry_features
from .contract import validate_mask_geometry_features


def run_mask_geometry(
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

    df = compute_mask_geometry_features(snip_inventory, frame_masks, frame_inventory)

    # Registry is the verifier: every feature row must root in a registered animal.
    validate_mask_geometry_features(
        df, physical_embryo_registry_df=registry, check_sources=True
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
