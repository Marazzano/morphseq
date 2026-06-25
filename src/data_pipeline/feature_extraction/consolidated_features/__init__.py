"""consolidated_features product — the chosen per-snip feature table, merged by snip_id."""

from .compute import consolidate_feature_tables
from .contract import (
    CONSOLIDATED_FEATURES_TABLE_COLUMNS,
    validate_consolidated_features,
)

__all__ = [
    "CONSOLIDATED_FEATURES_TABLE_COLUMNS",
    "validate_consolidated_features",
    "consolidate_feature_tables",
]
