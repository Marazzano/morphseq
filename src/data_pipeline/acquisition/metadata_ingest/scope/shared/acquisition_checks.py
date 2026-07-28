"""Scope-agnostic acquisition-inventory check primitives.

These are the *thin parameterized skeleton* the acquisition-inventory design calls for
(``target/acquisition_inventory_flow.md`` §"What is SHARED"): pure functions over a DataFrame +
declared column names, with NO knowledge of any specific microscope. Each scope's acquisition
module declares *what* to check (its schema + cell key) and calls these for *how* each check works.

The cell-uniqueness mechanic here is the same one Keyence's collision detector will later build on
(it declares the full raw-unit key ``(well_id, position_index, z_index, channel_index,
time_index_claimed)``); YX1, which cannot collide, declares its tensor cell key and uses these as a
defensive "is this scope really clean?" assertion.

Import direction (enforced by the design): ``scope/shared/*`` imports nothing scope-specific —
stdlib / pandas / the existing shared ``io.validators`` only.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema


def assert_columns_present(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    scope_label: str,
) -> None:
    """Fail loud unless every column in ``columns`` exists AND is non-null in ``df``.

    Thin wrapper over the existing shared ``validate_dataframe_schema`` so the acquisition
    primitives present one consistent entry point and one consistent error voice.
    """
    validate_dataframe_schema(df, list(columns), scope_label)


def assert_positive_column(
    df: pd.DataFrame,
    column: str,
    *,
    scope_label: str,
) -> None:
    """Fail loud unless ``column`` exists and every value is numeric and strictly > 0.

    The calibration/dims guard (``micrometers_per_pixel``, ``image_width_px``,
    ``image_height_px``): a zero or NaN µm/px silently corrupts every downstream physical
    measurement, so it is rejected at the seam, with the words.
    """
    if column not in df.columns:
        raise ValueError(
            f"{scope_label}: required positive column '{column}' is missing "
            f"(present columns: {list(df.columns)})."
        )

    values = pd.to_numeric(df[column], errors="coerce")
    bad_mask = values.isna() | (values <= 0)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        sample = df.loc[bad_mask, column].head(5).tolist()
        raise ValueError(
            f"{scope_label}: column '{column}' must be present and > 0 on every row, but "
            f"{n_bad} row(s) are missing, non-numeric, or <= 0 (e.g. {sample})."
        )


def assert_unique_on_key(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    scope_label: str,
) -> None:
    """Fail loud unless ``df`` has exactly one row per ``key_columns`` tuple (the cell key).

    Uniqueness is asserted on the declared addressing CELL — for YX1 the tensor coordinate
    ``(position_index, z_index, channel_index, time_index)``. A collision here means two raw units
    claim the same cell (the re-acquisition signal the design exists to surface). YX1 should never
    collide, so a failure is a real bug; the message names the offending cells.
    """
    key_columns = list(key_columns)
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: cannot check cell uniqueness — key columns {missing} are missing "
            f"(present columns: {list(df.columns)})."
        )

    duplicated_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicated_mask.any():
        offenders = (
            df.loc[duplicated_mask, key_columns]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        n_colliding_rows = int(duplicated_mask.sum())
        raise ValueError(
            f"{scope_label}: acquisition inventory is not unique on the cell key "
            f"{key_columns} — {n_colliding_rows} row(s) collide on "
            f"{len(offenders)}+ cell(s) (e.g. {offenders}). Exactly one raw unit may occupy "
            "each cell."
        )


def assert_channel_mapping_consistent(
    df: pd.DataFrame,
    *,
    index_column: str = "channel_index",
    raw_column: str = "raw_channel_name",
    normalized_column: str = "channel",
    scope_label: str,
) -> None:
    """Fail loud unless the channel triple is a consistent 1:1:1 mapping within the table.

    The acquisition inventory is the single home of the channel mapping
    (``channel_index`` ↔ ``raw_channel_name`` ↔ ``channel``). This guards against a channel index
    that maps to two raw names, a raw name that maps to two normalized tokens, etc. — exactly the
    kind of drift that today is re-derived (and lost) in three separate places.
    """
    for col in (index_column, raw_column, normalized_column):
        if col not in df.columns:
            raise ValueError(
                f"{scope_label}: cannot check channel-mapping consistency — column '{col}' is "
                f"missing (present columns: {list(df.columns)})."
            )

    triples = df[[index_column, raw_column, normalized_column]].drop_duplicates()

    for left, right in (
        (index_column, raw_column),
        (index_column, normalized_column),
        (raw_column, index_column),
        (raw_column, normalized_column),
    ):
        ambiguous = triples.groupby(left)[right].nunique()
        offending = ambiguous[ambiguous > 1]
        if not offending.empty:
            raise ValueError(
                f"{scope_label}: inconsistent channel mapping — '{left}' value(s) "
                f"{offending.index.tolist()} map to multiple '{right}' values. The channel triple "
                f"({index_column} ↔ {raw_column} ↔ {normalized_column}) must be 1:1."
            )
