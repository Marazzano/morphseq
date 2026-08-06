"""Validate physical (plate-free) position-to-well mapping.

This validator exists to prevent silently generating downstream artifacts under the
wrong well IDs. It checks that `position_well_mapping.csv` can fully map the scope
rows to `well_index` values, and that those `well_index` values are canonical
(A01-style) unless an explicit override is enabled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.position_well_mapping import validate_position_well_mapping
from data_pipeline.acquisition.metadata_ingest.time_helpers import ensure_time_index_column


CANONICAL_WELL_RE = re.compile(r"^[A-H](0[1-9]|1[0-2])$")
OVERRIDE_WELL_RE = re.compile(r"^S\d{2}$")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_physical_well_mapping(
    *,
    scope_metadata_csv: Path,
    mapping_csv: Path,
    allow_unmapped_wells: bool,
) -> dict:
    scope_df = ensure_time_index_column(pd.read_csv(scope_metadata_csv), stage_name="validate_physical_well_mapping.scope")
    mapping_df = pd.read_csv(mapping_csv)
    scope_df["experiment_id"] = scope_df["experiment_id"].astype(str)
    mapping_df["experiment_id"] = mapping_df["experiment_id"].astype(str)
    validate_position_well_mapping(mapping_df, scope_label=str(mapping_csv))

    source_col = "raw_position_label" if "raw_position_label" in scope_df.columns else "position_index"
    if source_col not in scope_df.columns:
        raise ValueError("scope metadata must contain raw_position_label or position_index")

    scope_positions = (
        scope_df.assign(position_index=lambda d: pd.to_numeric(d[source_col], errors="raise").astype(int))
        [["experiment_id", "position_index"]]
        .drop_duplicates()
    )
    covered = scope_positions.merge(
        mapping_df[["experiment_id", "position_index", "well_index"]],
        on=["experiment_id", "position_index"],
        how="left",
        validate="one_to_one",
    )

    unmapped_rows = covered[covered["well_index"].isna()]
    unmapped = unmapped_rows[["experiment_id", "position_index"]].to_dict(orient="records")
    if unmapped:
        # If we can't map any raw wells, Phase 3 should not run unless the user explicitly opted in.
        if not allow_unmapped_wells:
            raise ValueError(
                "Physical well mapping appears incomplete (scope positions not mapped). "
                f"Unmapped preview: {unmapped[:10]}. "
                "Fix mapping inputs (YX1 XY reference / Keyence layout) or set scope_ingest.allow_unmapped_wells=true."
            )

    bad_well_indices: list[str] = []
    for mapped in covered["well_index"].dropna().astype(str):
        mapped = str(mapped)
        if CANONICAL_WELL_RE.match(mapped):
            continue
        if allow_unmapped_wells and OVERRIDE_WELL_RE.match(mapped):
            continue
        bad_well_indices.append(mapped)

    if bad_well_indices:
        preview = sorted(set(bad_well_indices))[:10]
        raise ValueError(
            "Mapped well_index values are not canonical A01-style. "
            f"Bad well_index preview: {preview}. "
            "If you truly want to proceed with non-canonical IDs, set scope_ingest.allow_unmapped_wells=true and use S00-style IDs."
        )

    diagnostics = {
        "n_scope_positions": int(len(scope_positions)),
        "n_mapping_rows": int(len(mapping_df)),
        "allow_unmapped_wells": bool(allow_unmapped_wells),
        "scope_positions_preview": scope_positions.head(12).to_dict(orient="records"),
        "resolved_mapping_preview": covered.head(12).to_dict(orient="records"),
        "unmapped_positions_preview": unmapped[:12],
    }
    return diagnostics


def validate_physical_well_mapping_file(
    *,
    scope_metadata_csv: Path,
    mapping_csv: Path,
    output_flag: Path,
    diagnostics_json: Path | None = None,
    allow_unmapped_wells: bool = False,
) -> dict:
    diagnostics = validate_physical_well_mapping(
        scope_metadata_csv=scope_metadata_csv,
        mapping_csv=mapping_csv,
        allow_unmapped_wells=bool(allow_unmapped_wells),
    )

    output_flag = Path(output_flag)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")

    if diagnostics_json is not None:
        diagnostics_json = Path(diagnostics_json)
        diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    return diagnostics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    p.add_argument("--diagnostics-json", type=Path, required=False, default=None)
    p.add_argument("--allow-unmapped-wells", default="false")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_physical_well_mapping_file(
        scope_metadata_csv=args.scope_metadata_csv,
        mapping_csv=args.mapping_csv,
        output_flag=args.output_flag,
        diagnostics_json=args.diagnostics_json,
        allow_unmapped_wells=_parse_bool(args.allow_unmapped_wells),
    )


if __name__ == "__main__":
    main()
