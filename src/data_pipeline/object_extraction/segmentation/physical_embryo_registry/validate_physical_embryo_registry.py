"""Validator for the `physical_embryo_registry` product.

Mirrors the ``frame_masks`` contract/validator split: the contract names the
columns; this validator decides if a registry table is legal. It calls the
string-level ``validate_physical_embryo_id`` row-wise — the same way
``validate_frame_masks`` calls ``parse_track_id`` per row.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.physical_embryo_registry_contract import (
    MERGE_POLICY_VALUES,
    PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS,
)
from data_pipeline.shared.identifiers import (
    build_physical_embryo_id,
    validate_physical_embryo_id,
)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def validate_physical_embryo_registry(physical_embryo_registry: pd.DataFrame) -> None:
    """Fail loud unless the registry table is structurally + identity-consistent.

    Checks:
    - all required columns present;
    - physical_embryo_id present, non-null, UNIQUE (grain = one row per animal) — this
      also ENFORCES the by-construction global-uniqueness invariant when called on the
      merged experiment table (the difference between the registry being a promise and a
      hope);
    - local_embryo_index integer >= 1;
    - physical_embryo_id == build_physical_embryo_id(well_id, local_embryo_index)
      (round-trip; delegates the string check to validate_physical_embryo_id);
    - track_id non-null; maps to exactly one physical_embryo_id within a well_id;
    - no duplicate (well_id, local_embryo_index); no duplicate (well_id, track_id);
    - track_id_source non-empty;
    - merge_policy is one of the three EmbryoMergePolicy values
      ("normal"/"bridged"/"fractured");
    - n_sources is an integer >= 1.

    A FRACTURED animal legally exists at a single time_index (its snip chain is
    single-timepoint) — the registry makes no full-timecourse claim, so no check here
    rejects that layout (EXPERIMENT_GROUP_PLATE_MODEL.md "Known layout consequence").
    """
    _require_columns(physical_embryo_registry, PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS, "physical_embryo_registry")

    df = physical_embryo_registry

    if df["physical_embryo_id"].isna().any():
        raise ValueError("physical_embryo_registry physical_embryo_id must be non-null")
    if df["physical_embryo_id"].duplicated().any():
        dupes = df.loc[df["physical_embryo_id"].duplicated(keep=False), "physical_embryo_id"].head(5).tolist()
        raise ValueError(
            f"physical_embryo_registry physical_embryo_id must be globally unique; examples: {dupes}"
        )

    local_index = pd.to_numeric(df["local_embryo_index"], errors="coerce")
    if local_index.isna().any() or (local_index != local_index.round()).any():
        raise ValueError("physical_embryo_registry local_embryo_index must be an integer")
    if (local_index < 1).any():
        raise ValueError(
            "physical_embryo_registry local_embryo_index must be >= 1 (one-based; _e00 is invalid)"
        )

    if df["track_id"].isna().any() or df["track_id"].astype(str).str.len().eq(0).any():
        raise ValueError("physical_embryo_registry track_id must be non-null (the resolved tracking identity)")
    if df["track_id_source"].astype(str).str.len().eq(0).any():
        raise ValueError("physical_embryo_registry track_id_source must be non-empty")

    # Payload: merge_policy is a known EmbryoMergePolicy value; n_sources is a positive int.
    bad_policy = set(df["merge_policy"].astype(str)) - MERGE_POLICY_VALUES
    if bad_policy:
        raise ValueError(
            "physical_embryo_registry merge_policy must be one of "
            f"{sorted(MERGE_POLICY_VALUES)}; got unexpected value(s): {sorted(bad_policy)}"
        )
    n_sources = pd.to_numeric(df["n_sources"], errors="coerce")
    if n_sources.isna().any() or (n_sources != n_sources.round()).any():
        raise ValueError("physical_embryo_registry n_sources must be an integer")
    if (n_sources < 1).any():
        raise ValueError("physical_embryo_registry n_sources must be >= 1 (a well has at least one acquisition)")

    # Round-trip: physical_embryo_id is the constructor-minted id for (well_id, local_embryo_index),
    # and its embedded well_id agrees with the well_id column.
    for _, row in df.iterrows():
        physical_embryo_id = str(row["physical_embryo_id"])
        well_id = str(row["well_id"])
        idx = int(row["local_embryo_index"])
        validate_physical_embryo_id(physical_embryo_id, well_id=well_id)
        expected = build_physical_embryo_id(well_id, idx)
        if physical_embryo_id != expected:
            raise ValueError(
                f"physical_embryo_registry physical_embryo_id {physical_embryo_id!r} is not the "
                f"constructor-minted {expected!r} for (well_id={well_id!r}, "
                f"local_embryo_index={idx}). Mint via build_physical_embryo_id."
            )

    # No two animals in a well share a local_embryo_index, and no track resolves to two animals.
    if df.duplicated(subset=["well_id", "local_embryo_index"]).any():
        raise ValueError("physical_embryo_registry has duplicate (well_id, local_embryo_index)")
    if df.duplicated(subset=["well_id", "track_id"]).any():
        raise ValueError("physical_embryo_registry has duplicate (well_id, track_id)")

    # A track_id must map to exactly one physical_embryo_id within its well.
    per_track = df.groupby(["well_id", "track_id"])["physical_embryo_id"].nunique()
    ambiguous = per_track[per_track > 1]
    if not ambiguous.empty:
        raise ValueError(
            "physical_embryo_registry: a (well_id, track_id) resolves to more than one "
            f"physical_embryo_id; examples: {ambiguous.index.tolist()[:5]}"
        )
