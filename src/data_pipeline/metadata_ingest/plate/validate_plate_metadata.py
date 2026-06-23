"""Validate plate_metadata.csv contract and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.metadata_ingest.plate.plate_metadata_contract import validate_plate_metadata


def validate_plate_metadata_csv(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_plate_metadata(df)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_plate_metadata_csv(args.input_csv, args.output_flag)


if __name__ == "__main__":
    main()
