"""Discover physical wells from an external drop-in frame_inventory manifest.

The drop-in twin of ``discover_wells_from_scope_metadata``. Reads the dataset-level
``dropin_frame_inventory.csv`` (the ingress submission surface — atoms only:
``experiment_id`` + ``well_index`` + …), enforces exactly one experiment, derives the global
``well_id`` from the atoms via the identifier grammar, validates, and writes ``discovered_wells.txt``.

This is the SAME ``discovered_wells.txt`` contract the native path emits — so the well-runner fan,
single-well scheduling, and segmentation behave identically for native and drop-in data. Both
producers read a TABLE (never images) at discovery, so ``active_wells`` exists before any per-well work.

Does NOT filter QC or eligibility — ``discovered_wells.txt`` is physical identity only.
``run_wells = discovered_wells ∩ target_wells [∩ eligible_wells]`` is computed later by well_runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.well_discovery.discovered_wells_contract import (
    validate_discovered_wells,
    write_discovered_wells,
)
from data_pipeline.shared.identifiers import build_well_id


def discover_wells_from_handoff(manifest_csv: Path, output_wells: Path) -> None:
    """Read a drop-in frame_inventory manifest and emit ``discovered_wells.txt``."""
    df = pd.read_csv(manifest_csv)
    for col in ("experiment_id", "well_index"):
        if col not in df.columns:
            raise ValueError(
                f"[discover_wells_from_handoff] {manifest_csv} is missing required column {col!r}. "
                "The drop-in manifest must carry the frame-key atoms (experiment_id, well_index, …)."
            )

    experiments = sorted(df["experiment_id"].dropna().astype(str).str.strip().unique())
    if len(experiments) != 1:
        raise ValueError(
            f"[discover_wells_from_handoff] a drop-in submission must contain exactly one "
            f"experiment_id; found {experiments}. Split the manifest by experiment and submit "
            "one experiment at a time."
        )
    experiment_id = experiments[0]

    # Derive the global well_id from the atoms in encounter order (never f-string it inline).
    seen: set[str] = set()
    wells: list[str] = []
    for well_index in df["well_index"].dropna().astype(str):
        well_index = well_index.strip()
        if not well_index:
            continue
        well_id = build_well_id(experiment_id, well_index)
        if well_id in seen:
            continue
        seen.add(well_id)
        wells.append(well_id)

    validate_discovered_wells(wells)
    write_discovered_wells(output_wells, wells)
    print(f"Discovered {len(wells)} wells (drop-in) → {output_wells}")
