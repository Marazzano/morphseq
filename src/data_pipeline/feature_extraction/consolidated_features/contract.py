"""consolidated_features contract — the merged per-snip feature table.

Grain: one row per ``snip_id``. The contract guarantees the identity spine + the chosen core
feature columns are present; it allows extra feature columns from any merged product. Feature
nullability is owned by each source contract, so consolidated does not re-assert finiteness — it
guards identity (spine), the join (unique snip_id), and the presence of the core columns.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.shared.feature_table_utils import SNIP_FEATURE_TABLE_ID_COLUMNS
from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    validate_snip_grain_identity_columns,
)

# The minimum core features the consolidated table must carry (mask_geometry MVP). Curvature,
# pose/kinematics, fraction_alive, and stage columns join in when their products are merged.
_CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_um2",
    "perimeter_um",
    "length_um",
    "width_um",
    "centroid_x_um",
    "centroid_y_um",
)

CONSOLIDATED_FEATURES_REQUIRED_COLUMNS: list[str] = list(SNIP_FEATURE_TABLE_ID_COLUMNS + _CORE_FEATURE_COLUMNS)


def validate_consolidated_features(
    df: pd.DataFrame,
    *,
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,
    scope_label: str = "consolidated_features",
) -> None:
    """Fail loud unless ``df`` is a valid consolidated_features table (spine + core columns)."""
    validate_snip_grain_identity_columns(
        df,
        grain="snip_id",
        physical_embryo_registry_df=physical_embryo_registry_df,
        check_sources=check_sources,
        scope_label=scope_label,
    )
    missing = [c for c in CONSOLIDATED_FEATURES_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scope_label}: missing required column(s): {', '.join(missing)}. "
            f"Core consolidated columns are {CONSOLIDATED_FEATURES_REQUIRED_COLUMNS}."
        )
