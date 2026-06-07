"""Build, validate, and merge the frame_inventory product.

The current live pipeline still produces frame_contract.csv as the physical frame table.
This module is a behavior-preserving adapter: it treats that table as the legacy source of the
new frame_inventory product, splits it into per-well shards, validates those shards, and
merges shards back into the experiment-level inventory.

AUDIT (2026-06-07): docs/refactors/streamline-snakemake/target/frame_inventory_well_runner_audit.md
records the open gaps here — validate_frame_inventory is the WEAK schema check (not yet the strict
file-level gate the stitched-handoff contract specifies; finding #2), merge_frame_inventory_shards
duplicates well_runner.concat_well_shards_to_file (finding #4), the well_id filter assumes a global
well_id already in frame_contract.csv (pre-Scope-2 artifacts have a local one; finding #5), and the
unique key still uses time_int rather than the canonical time_index (finding #6).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT, UNIQUE_KEY_FRAME_CONTRACT
from data_pipeline.shared.identifiers import validate_well_id


def _read_frame_inventory_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_inventory")
    return df


def _validate_unique_keys(df: pd.DataFrame, *, context: str) -> None:
    duplicate_mask = df.duplicated(subset=list(UNIQUE_KEY_FRAME_CONTRACT), keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, list(UNIQUE_KEY_FRAME_CONTRACT)]
        raise ValueError(
            f"Duplicate {context} keys detected: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )


def build_frame_inventory_for_well(
    *,
    frame_contract_csv: Path,
    experiment_id: str,
    well_id: str,
    output_csv: Path,
) -> pd.DataFrame:
    """Write one well's frame_inventory shard from the legacy frame_contract table.

    This intentionally preserves the source table's columns and order. The source is validated
    against the frame-contract schema because that is the current physical frame inventory schema.
    """
    global_well_id = validate_well_id(str(well_id))
    frame_df = _read_frame_inventory_table(Path(frame_contract_csv))
    mask = (
        (frame_df["experiment_id"].astype(str) == str(experiment_id))
        & (frame_df["well_id"].astype(str) == global_well_id)
    )
    shard = frame_df.loc[mask].copy()
    if shard.empty:
        # TODO(Scope 2): a pre-Scope-2 frame_contract.csv may carry a LOCAL well_id (e.g. 'A01'),
        # so the global-well_id filter finds nothing — regenerate the source. (Audit finding #5.)
        raise ValueError(
            f"No frame_contract rows for experiment={experiment_id!r}, well_id={global_well_id!r}. "
            "The source may be a pre-Scope-2 artifact whose well_id is still a local label; "
            "regenerate frame_contract.csv with the global well_id grammar."
        )
    _validate_unique_keys(shard, context="frame_inventory")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shard.to_csv(output_csv, index=False)
    return shard


def validate_frame_inventory(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    """Validate one frame_inventory table and write its sentinel.

    TODO(Scope 2): this is the WEAK interim check (schema columns + nulls + unique key only). The
    stitched_handoff_contract.md gate is strict + file-level: paths exist, images open, real dims
    == declared, micrometers_per_pixel > 0, BF contiguous, channels rectangular, derived ids
    recomputed from atoms. Promote this then (audit finding #2).
    """
    df = _read_frame_inventory_table(Path(input_csv))
    _validate_unique_keys(df, context="frame_inventory")
    output_flag = Path(output_flag)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n", encoding="utf-8")
    return df


def merge_frame_inventory_shards(input_csvs: Sequence[Path], output_csv: Path) -> pd.DataFrame:
    """Merge per-well frame_inventory shards into one experiment-level inventory."""
    paths = [Path(p) for p in input_csvs]
    if not paths:
        raise ValueError("merge_frame_inventory_shards: no input shards provided.")

    frames = []
    expected_columns: list[str] | None = None
    for path in paths:
        frame = _read_frame_inventory_table(path)
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"Frame inventory shard {path} has columns {columns}, expected {expected_columns}."
            )
        frames.append(frame)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    _validate_unique_keys(merged, context="frame_inventory")
    # TODO(Scope 2): time_int → canonical time_index (audit finding #6).
    sort_cols = [c for c in ["experiment_id", "well_id", "channel_id", "time_int"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-for-well")
    p_build.add_argument("--frame-contract-csv", type=Path, required=True)
    p_build.add_argument("--experiment", required=True)
    p_build.add_argument("--well-id", required=True)
    p_build.add_argument("--output-csv", type=Path, required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--input-csv", type=Path, required=True)
    p_validate.add_argument("--output-flag", type=Path, required=True)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_merge.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "build-for-well":
        build_frame_inventory_for_well(
            frame_contract_csv=args.frame_contract_csv,
            experiment_id=args.experiment,
            well_id=args.well_id,
            output_csv=args.output_csv,
        )
    elif args.command == "validate":
        validate_frame_inventory(input_csv=args.input_csv, output_flag=args.output_flag)
    elif args.command == "merge":
        merge_frame_inventory_shards(input_csvs=args.inputs, output_csv=args.output_csv)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
