"""Apply canonical position-to-well mapping to scope metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.time_helpers import add_elapsed_time_columns
from data_pipeline.acquisition.metadata_ingest.time_helpers import add_frame_interval_unit_columns
from data_pipeline.acquisition.metadata_ingest.time_helpers import ensure_time_int_column
from data_pipeline.acquisition.metadata_ingest.position_well_mapping import validate_position_well_mapping
from data_pipeline.shared.identifiers import build_image_id


def apply_position_to_well_mapping(
    scope_metadata_csv: Path,
    mapping_csv: Path,
    output_csv: Path,
    experiment_id: str,
    selected_wells: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Map scope rows to plate wells and canonical IDs for downstream contracts."""
    scope_df = ensure_time_int_column(
        pd.read_csv(scope_metadata_csv),
        stage_name="scope_metadata_position_mapping_input",
    )
    mapping_df = pd.read_csv(mapping_csv)
    scope_df["experiment_id"] = scope_df["experiment_id"].astype(str)
    mapping_df["experiment_id"] = mapping_df["experiment_id"].astype(str)
    validate_position_well_mapping(mapping_df, scope_label=str(mapping_csv))
    mapping_df = mapping_df[mapping_df["experiment_id"] == str(experiment_id)].copy()
    scope_df = scope_df[scope_df["experiment_id"] == str(experiment_id)].copy()
    if scope_df.empty:
        raise ValueError(f"No scope metadata rows for experiment={experiment_id!r}.")
    if mapping_df.empty:
        raise ValueError(f"No position_well_mapping rows for experiment={experiment_id!r}.")

    source_col = "raw_position_label" if "raw_position_label" in scope_df.columns else "position_index"
    if source_col not in scope_df.columns:
        raise ValueError(
            "scope metadata must contain raw_position_label or position_index to apply "
            "position_well_mapping.csv"
        )
    mapped_df = scope_df.copy()
    mapped_df["position_index"] = pd.to_numeric(
        mapped_df[source_col], errors="raise"
    ).astype(int)

    identity_cols = ["experiment_id", "position_index", "well_index", "well_id"]
    mapped_df = mapped_df.drop(columns=["well_index", "well_id"], errors="ignore").merge(
        mapping_df[identity_cols],
        on=["experiment_id", "position_index"],
        how="left",
        validate="many_to_one",
    )

    missing_identity = mapped_df["well_id"].isna()
    if missing_identity.any():
        sample = mapped_df.loc[
            missing_identity, ["experiment_id", "position_index"]
        ].drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(
            "position_well_mapping.csv does not cover all scope metadata positions. "
            f"Missing preview: {sample}"
        )

    mapped_df["channel_id"] = mapped_df.get("channel", "BF").astype(str)

    if "raw_channel_name" in mapped_df.columns:
        mapped_df["channel_name_raw"] = mapped_df["raw_channel_name"].astype(str)
    else:
        mapped_df["channel_name_raw"] = mapped_df["channel_id"].astype(str)

    mapped_df["image_id"] = [
        build_image_id(well_id, channel, int(t))
        for well_id, channel, t in zip(mapped_df["well_id"].astype(str), mapped_df["channel_id"].astype(str), mapped_df["time_int"].astype(int))
    ]

    mapped_df = add_elapsed_time_columns(
        mapped_df,
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    mapped_df = add_frame_interval_unit_columns(mapped_df)

    selected_wells_set = {str(well) for well in (selected_wells or []) if str(well)}
    if selected_wells_set:
        mapped_df = mapped_df[mapped_df["well_id"].astype(str).isin(selected_wells_set)].copy()

    front_cols = [
        "experiment_id",
        "well_id",
        "well_index",
        "time_int",
        "channel_id",
        "image_id",
        "experiment_time_s",
        "frame_interval_s",
        "frame_interval_min",
        "frame_interval_hr",
        "elapsed_time_s",
        "elapsed_time_min",
        "elapsed_time_hr",
    ]
    ordered = [col for col in front_cols if col in mapped_df.columns]
    remainder = [col for col in mapped_df.columns if col not in ordered]
    mapped_df = mapped_df.loc[:, ordered + remainder]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(output_csv, index=False)
    return mapped_df



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--position-well-mapping-csv", type=Path, required=True)
    p.add_argument("--output-scope-metadata-mapped-csv", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_position_to_well_mapping(
        scope_metadata_csv=args.scope_metadata_csv,
        mapping_csv=args.position_well_mapping_csv,
        output_csv=args.output_scope_metadata_mapped_csv,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
