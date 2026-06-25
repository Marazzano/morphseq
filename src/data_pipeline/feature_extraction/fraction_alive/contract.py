"""fraction_alive_features contract — continuous viability fraction per snip.

Grain: one row per ``snip_id``. ``fraction_alive`` is in [0, 1]; it is NULLABLE because an empty
embryo mask has no defined viability fraction (documented null, not a QC decision).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import (
    SNIP_FEATURE_TABLE_SPINE_COLUMNS,
    validate_feature_table,
)

FRACTION_ALIVE_PAYLOAD_COLUMNS: tuple[str, ...] = ("fraction_alive",)
_NULLABLE_PAYLOAD_COLUMNS: tuple[str, ...] = ("fraction_alive",)

FRACTION_ALIVE_TABLE_COLUMNS: list[str] = list(SNIP_FEATURE_TABLE_SPINE_COLUMNS + FRACTION_ALIVE_PAYLOAD_COLUMNS)


def validate_fraction_alive_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "fraction_alive_features",
) -> None:
    """Fail loud unless ``df`` is a valid fraction_alive_features table (spine first)."""
    validate_feature_table(
        df,
        required_columns=FRACTION_ALIVE_TABLE_COLUMNS,
        feature_columns=FRACTION_ALIVE_PAYLOAD_COLUMNS,
        nullable_feature_columns=_NULLABLE_PAYLOAD_COLUMNS,
        scope_label=scope_label,
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
    )
    present = df["fraction_alive"].dropna()
    if ((present < 0.0) | (present > 1.0)).any():
        raise ValueError(f"{scope_label}: fraction_alive must be in [0, 1].")
