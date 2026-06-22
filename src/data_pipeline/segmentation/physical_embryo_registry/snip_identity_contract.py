"""The identity-carrying spine validator — owned here, called everywhere.

The registry minting ``physical_embryo_id`` once does not, by itself, prevent
identity from being re-buried downstream. What prevents it is the LOCKED 5-clause
spine law (authored in ``physical_embryo_registry_world.md``):

    1. Every derived row carries its own identity level PLUS all parent identities.
    2. physical_embryo_id is REQUIRED parent identity, not optional provenance.
    3. Syntactic validity of embryo_id/snip_id is INSUFFICIENT if it disagrees with
       physical_embryo_id.
    4. Downstream tables do NOT rediscover physical_embryo_id by parsing IDs ad hoc.
    5. They consume the EXPLICIT column and call the shared spine validator.

This module provides that shared validator. It lives in the registry world (not in
``shared/identifiers``) because this world owns the biological parent→child identity
relationship; ``shared/identifiers`` owns only the string GRAMMAR. Three tiers, no
overlap:

    shared/identifiers/validators.py   → validate_physical_embryo_id(...)  # ONE string is well-formed
    physical_embryo_registry/snip_identity_contract.py
                                       → validate_snip_grain_identity_columns(...)  # IDs in a table AGREE
    <product>/contract.py              → call the spine first, then product-specific columns

Constructor note (verified against ``shared/identifiers/constructors.py``; do NOT
"fix"): ``build_embryo_id(physical_embryo_id, image_id)`` and
``build_snip_id(embryo_id, image_id)`` BOTH take ``image_id``. The parsers invert
them: ``parse_embryo_id`` → ``(physical_embryo_id, channel_id)``; ``parse_snip_id``
→ ``(embryo_id, time_index)``.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.shared.identifiers import (
    build_physical_embryo_id,
    parse_embryo_id,
    parse_image_id,
    parse_physical_embryo_id,
    parse_snip_id,
    validate_physical_embryo_id,
)


# Grain-aware identity spine: own level + all parents. Frame-derived columns
# (image_id/time_index/channel_id) are cross-checked only when present — they are
# not required spine for an embryo-grain (time/channel-aggregated) table.
_SNIP_SPINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
    "embryo_id",
    "snip_id",
)
_EMBRYO_SPINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "physical_embryo_id",
)
_VALID_GRAINS = ("snip", "embryo")


def _require_spine_columns(df: pd.DataFrame, required: tuple[str, ...], scope_label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required identity-spine column(s): {', '.join(missing)}. "
            "Every snip/embryo-grain table must carry its own identity level plus all parent "
            "identities (see the identity-carrying spine law)."
        )
    for col in required:
        if df[col].isna().any():
            raise ValueError(
                f"{scope_label}: identity-spine column {col!r} has null value(s). Parent identity "
                "is required, not optional provenance."
            )


def validate_snip_grain_identity_columns(
    df: pd.DataFrame,
    *,
    grain: str = "snip",
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "snip_grain_table",
) -> None:
    """Fail loud unless a snip/embryo-grain table's identity columns AGREE with each other.

    ``grain`` selects the required spine ("snip" = own level + all parents incl.
    ``embryo_id``/``snip_id``; "embryo" = the coarser time/channel-aggregated grain,
    which carries no ``snip_id``/``embryo_id``). Grain is the CALLER's assertion,
    never guessed.

    One contract, two modes by lifecycle moment (mirrors
    ``validate_yx1_acquisition_inventory(check_sources=...)``). At BUILD time
    (default ``check_sources=False``) this checks declared facts only — that the
    table's own ID columns are internally consistent. At the CONSUME boundary
    (``check_sources=True``) it ADDITIONALLY asserts every ``physical_embryo_id``
    exists in ``physical_embryo_registry_df`` — the moment the "is this animal
    actually registered?" risk appears.

    A row is invalid even if its ``snip_id`` is syntactically valid, if
    ``snip_id``/``embryo_id`` disagree with ``physical_embryo_id``, or if
    ``physical_embryo_id`` is missing. A syntactically valid snip_id is not enough.
    """
    if grain not in _VALID_GRAINS:
        raise ValueError(
            f"{scope_label}: unknown grain {grain!r}; expected one of {_VALID_GRAINS}."
        )

    required = _SNIP_SPINE_COLUMNS if grain == "snip" else _EMBRYO_SPINE_COLUMNS
    _require_spine_columns(df, required, scope_label)

    # physical_embryo_id (parent identity) round-trips against the well_id column.
    for _, row in df.iterrows():
        physical_embryo_id = str(row["physical_embryo_id"])
        well_id = str(row["well_id"])
        experiment_id = str(row["experiment_id"])

        # Delegates the well_id round-trip to the shared string validator: it asserts
        # the embedded well is well-formed AND equals the well_id column.
        validate_physical_embryo_id(physical_embryo_id, well_id=well_id)
        embedded_well_id, local_embryo_index = parse_physical_embryo_id(physical_embryo_id)
        expected = build_physical_embryo_id(embedded_well_id, local_embryo_index)
        if physical_embryo_id != expected:
            raise ValueError(
                f"{scope_label}: physical_embryo_id {physical_embryo_id!r} is not the "
                f"constructor-minted {expected!r} for (well_id, local_embryo_index)."
            )
        if not embedded_well_id.startswith(experiment_id):
            raise ValueError(
                f"{scope_label}: experiment_id {experiment_id!r} disagrees with the "
                f"well_id {embedded_well_id!r} embedded in physical_embryo_id "
                f"{physical_embryo_id!r}."
            )

        if grain == "snip":
            embryo_id = str(row["embryo_id"])
            snip_id = str(row["snip_id"])
            embryo_parent, embryo_channel_id = parse_embryo_id(embryo_id)
            if embryo_parent != physical_embryo_id:
                raise ValueError(
                    f"{scope_label}: embryo_id {embryo_id!r} encodes physical_embryo_id "
                    f"{embryo_parent!r} but the row's physical_embryo_id is "
                    f"{physical_embryo_id!r}. A syntactically valid embryo_id is not enough."
                )
            snip_parent, snip_time_index = parse_snip_id(snip_id)
            if snip_parent != embryo_id:
                raise ValueError(
                    f"{scope_label}: snip_id {snip_id!r} encodes embryo_id {snip_parent!r} "
                    f"but the row's embryo_id is {embryo_id!r}. A syntactically valid snip_id "
                    "is not enough."
                )
            # Where frame-derived columns are present, they must agree with image_id.
            if "image_id" in df.columns and not pd.isna(row.get("image_id")):
                image_id = str(row["image_id"])
                image_well_id, image_channel_id, image_time_index = parse_image_id(image_id)
                if image_well_id != embedded_well_id:
                    raise ValueError(
                        f"{scope_label}: image_id {image_id!r} encodes well_id "
                        f"{image_well_id!r}, disagreeing with physical_embryo_id well "
                        f"{embedded_well_id!r}."
                    )
                if image_channel_id != embryo_channel_id:
                    raise ValueError(
                        f"{scope_label}: image_id {image_id!r} channel {image_channel_id!r} "
                        f"disagrees with embryo_id channel {embryo_channel_id!r}."
                    )
                if "channel_id" in df.columns and str(row["channel_id"]) != image_channel_id:
                    raise ValueError(
                        f"{scope_label}: channel_id column {row['channel_id']!r} disagrees "
                        f"with image_id channel {image_channel_id!r}."
                    )
                if "time_index" in df.columns and int(row["time_index"]) != image_time_index:
                    raise ValueError(
                        f"{scope_label}: time_index column {row['time_index']!r} disagrees "
                        f"with image_id time {image_time_index!r}."
                    )
                if snip_time_index != image_time_index:
                    raise ValueError(
                        f"{scope_label}: snip_id time {snip_time_index!r} disagrees with "
                        f"image_id time {image_time_index!r}."
                    )

    # Uniqueness is the caller's assertion of one-row-per-grain, never guessed.
    unique_key = "snip_id" if grain == "snip" else "physical_embryo_id"
    if df[unique_key].duplicated().any():
        dupes = df.loc[df[unique_key].duplicated(keep=False), unique_key].head(5).tolist()
        raise ValueError(
            f"{scope_label}: {unique_key} must be unique at {grain} grain; examples: {dupes}"
        )

    if check_sources:
        if physical_embryo_registry_df is None:
            raise ValueError(
                f"{scope_label}: check_sources=True requires physical_embryo_registry_df."
            )
        registered = set(physical_embryo_registry_df["physical_embryo_id"].astype(str))
        present = set(df["physical_embryo_id"].astype(str))
        unregistered = sorted(present - registered)
        if unregistered:
            raise ValueError(
                f"{scope_label}: {len(unregistered)} physical_embryo_id(s) are not in the "
                f"physical_embryo_registry; examples: {unregistered[:5]}. Every derived row must "
                "root in a registered animal."
            )
