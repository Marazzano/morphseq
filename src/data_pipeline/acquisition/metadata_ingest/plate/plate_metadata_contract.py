"""Plate metadata contract (L2 schema validation).

Plain schema check: does the assembled artifact have the required columns?
Nothing fancy.  Mirrors ``position_well_mapping_contract.py`` in shape.

L2 validates:
- required columns exist (by name);
- ``well_id`` is consistent with ``build_well_id(experiment_id, well_index)``;
- no duplicate ``(experiment_id, well_id)`` rows.

L2 does NOT:
- check null values in biological fields (allowed — blank wells are legitimate);
- enforce biological type constraints (temperature numeric, etc.);
- discover or register entities.

Null / completeness enforcement is L3 (``entity_metadata_completeness_qc``).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.shared.identifiers import build_well_id


REQUIRED_PLATE_METADATA_FIELDS: tuple[str, ...] = (
    "genotype",
    "start_age_hpf",
    "temperature",
    "medium",
)

REQUIRED_PLATE_METADATA_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "well_index",
    *REQUIRED_PLATE_METADATA_FIELDS,
)

_FIELD_FIX_HINT = (
    "Add an 8×12 sheet named '{field}' (rows A–H, cols 1–12) to the well-metadata "
    "workbook, or a long-format sheet/CSV with a '{field}' column."
)


def validate_plate_metadata(
    df: pd.DataFrame,
    *,
    scope_label: str = "plate_metadata",
) -> None:
    """Fail unless ``df`` follows the canonical plate metadata contract.

    Checks required columns exist, well_id consistency, and no duplicate keys.
    Null biological values are explicitly allowed at this layer (L2).

    Raises:
        ValueError: on any contract violation, with a human fix hint.
    """
    for field in REQUIRED_PLATE_METADATA_COLUMNS:
        if field not in df.columns:
            hint = _FIELD_FIX_HINT.format(field=field)
            raise ValueError(
                f"[{scope_label}] missing required field '{field}'. {hint}"
            )

    checked = df.copy()
    checked["experiment_id"] = checked["experiment_id"].astype(str)
    checked["well_index"] = checked["well_index"].astype(str)
    checked["well_id"] = checked["well_id"].astype(str)

    expected_well_id = [
        build_well_id(experiment_id, well_index)
        for experiment_id, well_index in zip(
            checked["experiment_id"], checked["well_index"]
        )
    ]
    bad = checked["well_id"] != expected_well_id
    if bad.any():
        sample = checked.loc[bad, ["experiment_id", "well_index", "well_id"]].head(5).to_dict(
            orient="records"
        )
        raise ValueError(
            f"[{scope_label}] well_id is inconsistent with "
            "build_well_id(experiment_id, well_index). "
            f"First offenders: {sample}"
        )

    dup = df.duplicated(subset=["experiment_id", "well_id"], keep=False)
    if dup.any():
        preview = df.loc[dup, ["experiment_id", "well_id"]].head(10).to_dict(
            orient="records"
        )
        raise ValueError(
            f"[{scope_label}] duplicate (experiment_id, well_id) rows: {preview}"
        )
