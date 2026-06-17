"""Discover physical wells from scope_metadata_mapped.csv.

Single discovery source for Beat 1 (YX1 path). Reads the post-join canonical
metadata CSV, extracts unique global well_ids in encounter order, validates
them, and writes discovered_wells.txt.

scope_metadata_mapped.csv is produced by join_series_mapping_to_scope_metadata,
which is downstream of both YX1 and Keyence mapping — it already contains global
well_ids. This function reads that shared artifact, so it never needs to know
which microscope produced it.

Does NOT filter QC or eligibility — discovered_wells.txt is physical identity only.
run_wells = discovered_wells ∩ target_wells [ ∩ eligible_wells ] is computed later
by well_runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.metadata_ingest.well_discovery.discovered_wells_contract import (
    validate_discovered_wells,
    write_discovered_wells,
)


def discover_wells_from_scope_metadata(mapped_csv: Path, output_wells: Path) -> None:
    """Read scope_metadata_mapped.csv and emit discovered_wells.txt."""
    df = pd.read_csv(mapped_csv)
    if "well_id" not in df.columns:
        raise ValueError(f"{mapped_csv} is missing required well_id column")
    seen: set[str] = set()
    wells: list[str] = []
    for value in df["well_id"].dropna().astype(str):
        well_id = value.strip()
        if not well_id or well_id in seen:
            continue
        seen.add(well_id)
        wells.append(well_id)
    validate_discovered_wells(wells)
    write_discovered_wells(output_wells, wells)
    print(f"Discovered {len(wells)} wells → {output_wells}")
