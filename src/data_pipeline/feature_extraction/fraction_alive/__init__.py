"""fraction_alive feature product — continuous viability fraction per snip from VIA masks."""

from .compute import compute_fraction_alive_features
from .contract import (
    FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS,
    validate_fraction_alive_features,
)

__all__ = [
    "FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS",
    "validate_fraction_alive_features",
    "compute_fraction_alive_features",
]
