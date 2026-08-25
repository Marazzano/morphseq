"""Merge the frame_inventory product (table ops only).

The materialize_well branch emits the flat per-well shard; ``merge_frame_inventory_shards`` builds
the experiment-level aggregate view over those shards. The **validator moved out** into
``frame_inventory_validation.py`` (the public gate) — this module now owns only product/table ops.
``validate_frame_inventory`` is re-exported here for back-compat with existing importers.

The legacy ``build_frame_inventory_for_well`` adapter (which split the old ``frame_contract.csv``
into per-well shards) was strangled in Step 7 — it was off-DAG and superseded by the live
``materialize_well`` producer, which emits the per-well shard directly.

AUDIT (2026-06-07): docs/data_pipeline/specs/target/frame_inventory_well_runner_audit.md
records the open gaps — finding #4: merge_frame_inventory_shards duplicates
well_runner.concat_well_shards_to_file. (Finding #2 — the weak validator — is being resolved by the
strict-gate growth in frame_inventory_validation.py.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    UNIQUE_FRAME_INVENTORY_KEY_COLUMNS,
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory.frame_inventory_validation import (
    _read_frame_inventory_table,
    validate_frame_inventory,
)

__all__ = [
    "merge_frame_inventory_shards",
    "validate_frame_inventory",
]


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
    # Same identity gate the strict validator and assembler use — frame identity enforced once.
    validate_frame_inventory_identity_contract(merged, scope_label="frame_inventory")
    # Sort on the frame_inventory atoms (projection rows have z_index=NA; z_stack rows sort by plane).
    sort_cols = [c for c in list(UNIQUE_FRAME_INVENTORY_KEY_COLUMNS) if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--input-csv", type=Path, required=True)
    p_validate.add_argument("--output-flag", type=Path, required=True)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_merge.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "validate":
        validate_frame_inventory(input_csv=args.input_csv, output_flag=args.output_flag)
    elif args.command == "merge":
        merge_frame_inventory_shards(input_csvs=args.inputs, output_csv=args.output_csv)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
