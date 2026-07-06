"""mask_quality_qc entrypoint — the thin filesystem adapter.

Loads the validated snip_inventory (universe + snip->mask handoff) and the per-well canonical
frame_masks shard, resolves config, computes the three flags off canonical masks, validates with
the registry as verifier (check_sources=True), then writes the per-well shard. No path-minting
and no domain logic live here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_mask_quality_qc_flags
from .config import resolve_config
from .contract import validate_mask_quality_qc


def run_mask_quality_qc(
    *,
    snip_inventory_csv: Path,
    frame_masks_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    config_overrides: dict | None = None,
) -> None:
    config = resolve_config(config_overrides)

    snip_inventory = pd.read_csv(snip_inventory_csv)
    frame_masks = pd.read_csv(frame_masks_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    df = compute_mask_quality_qc_flags(snip_inventory, frame_masks, config=config)

    # Registry is the verifier: every QC row must root in a registered animal.
    validate_mask_quality_qc(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
