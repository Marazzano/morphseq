"""Thin orchestrator: load plate metadata → mint identity → validate → write CSV.

Calls:
    load_plate_metadata_pages  (L1 ingest)
    build_well_id              (identity mint)
    validate_plate_metadata    (L2 schema check)

Does NOT:
- parse Excel sheets directly (that is plate_metadata_loader's job);
- enforce biological nullability (that is L3's job);
- write side files (series_number_map.csv was dead — removed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.shared.identifiers import build_well_id
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_loader import load_plate_metadata_pages
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract import (
    REQUIRED_PLATE_METADATA_COLUMNS,
    validate_plate_metadata,
)


def process_plate_layout(
    input_file: Path,
    experiment_id: str,
    output_csv: Path,
) -> pd.DataFrame:
    """Load, normalize, validate, and write plate metadata for one experiment.

    Args:
        input_file: Path to the well-metadata Excel workbook (.xlsx / .xls).
        experiment_id: Experiment identifier (used to mint ``well_id``).
        output_csv: Destination path for the validated ``plate_metadata.csv``.

    Returns:
        Validated DataFrame written to ``output_csv``.

    Raises:
        FileNotFoundError: if ``input_file`` does not exist.
        ValueError: on any L1 ingest error or L2 contract violation.
    """
    pages = load_plate_metadata_pages(input_file)

    _print_page_summary(pages)

    df = pages.table.copy()
    df["experiment_id"] = experiment_id
    df["well_id"] = df["well_index"].map(lambda w: build_well_id(experiment_id, w))

    # Emit the CANONICAL biology columns even when the user omitted them (all-NA). This keeps the
    # L2 column *shape* stable regardless of which fields arrived, so downstream completeness QC (L3)
    # checks isna() on a guaranteed-present column instead of "missing column vs. present-but-NA".
    df = _ensure_canonical_columns(df)

    validate_plate_metadata(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return df


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every REQUIRED_PLATE_METADATA_COLUMN exists; fill absent ones with NA.

    A field the user did not supply is emitted as an all-NA column — the table's column shape never
    depends on which fields the user happened to include. Identity columns (experiment_id, well_id,
    well_index) are already populated upstream; only absent biology fields get the NA column.
    """
    df = df.copy()
    for col in REQUIRED_PLATE_METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _print_page_summary(pages) -> None:
    if pages.accepted:
        print(f"[plate_metadata] accepted pages: {pages.accepted}")
    for page_name, reason in pages.rejected:
        print(f"[plate_metadata] rejected page '{page_name}': {reason}")
