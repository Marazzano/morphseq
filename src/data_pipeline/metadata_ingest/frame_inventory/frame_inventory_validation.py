"""The public frame_inventory gate — sequences the contract rules (L0–L4).

This module owns ``validate_frame_inventory``, the single shared gate used identically by both
producers (native materializer + external drop-in). It is **non-mutating**: on pass it writes the
``.validated`` sentinel; on fail it writes a ``*_frame_inventory.errors.md`` report and raises. It
never rewrites the manifest CSV.

The ordered sequence:

    L0  schema      required columns present                 (io/validators.validate_dataframe_schema)
    L1  uniqueness  identity-anchored image_id key unique     (frame_inventory_image_ids)
    L2  derived ids well_id / image_id recompute from atoms   (assert_derived_ids_consistent)
    L3  grain       scope-aware: per_well = one well; merged = many wells, per-well checks grouped;
                    BF contiguous, channels rectangular, multi-timepoint ⇒ elapsed_time_s
    L4  sources     (only when check_sources=True) paths resolve + images open + dims + µm/px > 0

``check_sources`` is a mode flag (one validator, two modes); ``validation_scope`` selects the L3
grouping (one well vs. the whole experiment). Both default conservatively; every pipeline call site
passes them explicitly so strict behaviour is never inherited by accident.

Naming doctrine: ``grain`` names the row (snip_id, physical_embryo_id, well_id); ``validation_scope``
names the gate. Do not conflate them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.image_materialization.frame_inventory_contract import (
    REQUIRED_FRAME_INVENTORY_COLUMNS,
    UNIQUE_FRAME_INVENTORY_KEY_COLUMNS,
    assert_derived_ids_consistent,
    frame_inventory_image_ids,
)
from data_pipeline.metadata_ingest.frame_inventory.frame_inventory_validation_rules import (
    validate_grain,
    validate_sources,
)
from data_pipeline.shared.identifiers import build_well_id


def _read_frame_inventory_table(path: Path) -> pd.DataFrame:
    # Validate against the microscope-agnostic frame_inventory contract (the live materialized
    # shard's schema), NOT the legacy frame_contract columns. ``z_index`` is required-present but
    # nullable: NA on projection rows, real integer on z_stack rows.
    df = pd.read_csv(path)
    validate_dataframe_schema(
        df,
        list(REQUIRED_FRAME_INVENTORY_COLUMNS),
        "frame_inventory",
        nullable_columns=["z_index"],
    )
    return df


def _validate_unique_keys(df: pd.DataFrame, *, context: str) -> None:
    # Identity-anchored key: recompose the derived image_id from the atoms via the constructors
    # (frame_inventory_image_ids also validates the intermediate well_id), then check uniqueness on
    # that. Routing through the grammar keeps the key from drifting from the identifier code.
    image_ids = frame_inventory_image_ids(df, scope_label=context)
    duplicate_mask = image_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, list(UNIQUE_FRAME_INVENTORY_KEY_COLUMNS)]
        raise ValueError(
            f"Duplicate {context} keys detected (by derived image_id): "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )
    # Cross-check any producer-supplied derived ids against the atom-recomputed values.
    assert_derived_ids_consistent(df, scope_label=context)


def validate_frame_inventory(
    input_csv: Path,
    output_flag: Path,
    *,
    image_root: Path | None = None,
    check_sources: bool = False,
    validation_scope: str = "per_well",
) -> pd.DataFrame:
    """Validate one frame_inventory table (ordered L0–L4) and write its sentinel.

    On PASS: write the ``.validated`` sentinel and return the table.
    On FAIL: write a ``*_frame_inventory.errors.md`` report next to the input, then re-raise.

    ``check_sources=True`` runs the L4 source/image checks (the drop-in entrance and the native
    per-well node use this); the merged node passes ``check_sources=False``. ``validation_scope``
    selects L3 grouping (``"per_well"`` = one well; ``"merged"`` = many wells, checks grouped).
    """
    input_csv = Path(input_csv)
    df: pd.DataFrame | None = None
    try:
        # L0 — schema (also reads the table). Inside the try so even a schema failure reports.
        df = _read_frame_inventory_table(input_csv)
        # L1 + L2 — identity-anchored uniqueness and derived-id consistency.
        _validate_unique_keys(df, context="frame_inventory")
        # L3 — scope-aware grain.
        validate_grain(df, validation_scope=validation_scope, scope_label="frame_inventory")
        # L4 — source/image contract (only in strict mode).
        if check_sources:
            validate_sources(df, image_root=image_root, scope_label="frame_inventory")
    except Exception as exc:  # noqa: BLE001 — report then re-raise; fail loud is the contract.
        _write_errors_report(input_csv, df, validation_scope=validation_scope, error=exc)
        raise

    output_flag = Path(output_flag)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n", encoding="utf-8")
    return df


def _errors_report_path(
    input_csv: Path, df: pd.DataFrame | None, *, validation_scope: str
) -> Path:
    """Robust errors-file name — never depends on a value that might be invalid.

    per_well → ``{well_id}_frame_inventory.errors.md`` when derivable;
    merged   → ``{experiment_id}_frame_inventory.errors.md`` when derivable;
    fallback → ``frame_inventory.errors.md`` (unreadable table / pre-derivation / malformed atoms).
    """
    stem = "frame_inventory"
    if df is None:
        return input_csv.parent / f"{stem}.errors.md"
    try:
        experiment_ids = df["experiment_id"].dropna().astype(str).unique()
        if validation_scope == "per_well":
            well_indices = df["well_index"].dropna().astype(str).unique()
            if len(experiment_ids) == 1 and len(well_indices) == 1:
                stem = f"{build_well_id(experiment_ids[0], well_indices[0])}_frame_inventory"
        elif len(experiment_ids) == 1:
            stem = f"{experiment_ids[0]}_frame_inventory"
    except Exception:  # noqa: BLE001 — any derivation failure → safe fallback name.
        stem = "frame_inventory"
    return input_csv.parent / f"{stem}.errors.md"


def _write_errors_report(
    input_csv: Path, df: pd.DataFrame | None, *, validation_scope: str, error: Exception
) -> None:
    report_path = _errors_report_path(input_csv, df, validation_scope=validation_scope)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# frame_inventory validation failed\n\n"
        f"- **source:** `{input_csv}`\n"
        f"- **validation_scope:** `{validation_scope}`\n\n"
        f"## Error\n\n```\n{error}\n```\n",
        encoding="utf-8",
    )
