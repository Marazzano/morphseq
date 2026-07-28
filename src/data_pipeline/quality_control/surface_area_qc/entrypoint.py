"""surface_area_qc entrypoint — the thin filesystem adapter.

Loads inputs (mask_geometry + stage_predictions + the snip_inventory universe + the per-well
registry), resolves config (printing the active band statement), loads/validates the packaged
reference, computes flags, validates with the registry as verifier (check_sources=True), then
writes the per-well shard. No path-minting and no domain logic live here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import compute_surface_area_qc_flags
from .config import band_statement, resolve_config
from .contract import validate_surface_area_qc
from .reference import load_packaged_surface_area_reference


def run_surface_area_qc(
    *,
    mask_geometry_csv: Path,
    stage_predictions_csv: Path,
    snip_inventory_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    config_overrides: dict | None = None,
) -> None:
    config = resolve_config(config_overrides)
    print(band_statement(config))  # self-documenting: the active band, on every run

    mask_geometry = pd.read_csv(mask_geometry_csv)
    stage_predictions = pd.read_csv(stage_predictions_csv)
    snip_inventory = pd.read_csv(snip_inventory_csv)
    registry = pd.read_csv(physical_embryo_registry_csv)

    surface_area_reference = load_packaged_surface_area_reference(config.reference_version)

    df = compute_surface_area_qc_flags(
        mask_geometry, stage_predictions, snip_inventory, surface_area_reference, config=config
    )

    # Registry is the verifier: every QC row must root in a registered animal.
    validate_surface_area_qc(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
