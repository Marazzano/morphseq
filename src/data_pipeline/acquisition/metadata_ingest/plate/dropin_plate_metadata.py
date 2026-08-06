"""Land an external per-well plate table on the canonical plate-metadata seam."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plate_metadata_contract import validate_plate_metadata


def ingest_dropin_plate_metadata(
    *,
    input_csv: Path,
    experiment_id: str,
    output_csv: Path,
    output_flag: Path,
) -> pd.DataFrame:
    """Select one experiment, validate it, and preserve all extension columns."""
    source = pd.read_csv(input_csv)
    if "experiment_id" not in source.columns:
        raise ValueError(
            "drop-in plate metadata must contain an experiment_id column."
        )
    selected = source[
        source["experiment_id"].astype(str).eq(str(experiment_id))
    ].copy()
    if selected.empty:
        available = sorted(source["experiment_id"].dropna().astype(str).unique())
        raise ValueError(
            f"drop-in plate metadata has no rows for experiment {experiment_id!r}; "
            f"available experiments={available[:10]}."
        )
    validate_plate_metadata(
        selected, scope_label=f"dropin_plate_metadata:{experiment_id}"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n", encoding="utf-8")
    return selected


__all__ = ["ingest_dropin_plate_metadata"]
