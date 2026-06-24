"""The public frame_inventory gate — sequences the contract rules (L0–L4).

This module owns ``validate_frame_inventory``, the single shared gate used identically by both
producers (native materializer + external drop-in). It is **non-mutating**: on pass it writes the
``.validated`` sentinel; on fail it writes an errors report and raises. It never rewrites the
manifest CSV.

Step 1 (green refactor): the implementation moves here VERBATIM from ``frame_inventory.py`` — the
weak schema + identity-anchored-uniqueness + derived-id check. The new ``check_sources`` /
``validation_scope`` keywords are accepted with **conservative defaults** (``check_sources=False``,
``validation_scope="per_well"``) and are not yet acted on, so the move is provably behaviour-neutral.
The strict L0–L4 rules land in Step 2 in ``frame_inventory_validation_rules.py``; this module then
sequences them. Every call site passes the flags explicitly before the strict behaviour is enabled.

Naming doctrine: ``grain`` names the row (snip_id, physical_embryo_id, well_id); ``validation_scope``
names the gate (one well vs. the whole experiment). Do not conflate them.
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


def _read_frame_inventory_table(path: Path) -> pd.DataFrame:
    # Validate against the microscope-agnostic frame_inventory contract (the live materialized
    # shard's schema), NOT the legacy frame_contract columns. ``z_index`` is intentionally absent
    # from the required atoms (NA on projection rows), so it is never null-checked here.
    df = pd.read_csv(path)
    validate_dataframe_schema(df, list(REQUIRED_FRAME_INVENTORY_COLUMNS), "frame_inventory")
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
    """Validate one frame_inventory table and write its sentinel.

    ``image_root`` / ``check_sources`` / ``validation_scope`` are accepted now but only the
    schema + identity-anchored uniqueness + derived-id checks run in this green-refactor pass. The
    strict L3 grain (scope-aware) and L4 source rules land in Step 2.

    TODO(Scope 2): grow this into the strict L0–L4 gate (BF contiguity, channel rectangularity,
    paths exist, images open, real dims == declared, micrometers_per_pixel > 0) — audit finding #2.
    """
    df = _read_frame_inventory_table(Path(input_csv))
    _validate_unique_keys(df, context="frame_inventory")
    output_flag = Path(output_flag)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n", encoding="utf-8")
    return df
