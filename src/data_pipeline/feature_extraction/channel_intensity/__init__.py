"""Well-grain background null and corrected intensity features.

The other half of the seam: ``object_extraction/channel_intensity`` emits raw poolable evidence per
embryo-time; this package chooses an estimator and derives corrected values. Keeping the estimator
here is what lets the null be re-estimated without re-reading a pixel.

Channel-generic, like its extraction counterpart -- GFP will arrive.
"""

from .correction import SATURATION_CONCERN_FRACTION, correct_row
from .pooling import (
    MINIMUM_POOLED_PIXELS,
    NULL_ESTIMATOR,
    ChannelIntensityNullError,
    PooledNull,
    estimate_null,
    estimate_well_null,
    pool_histograms,
)

__all__ = [
    "ChannelIntensityNullError",
    "PooledNull",
    # The well-grain estimate, plus the stationarity evidence for judging whether pooling over time
    # was a defensible thing to do.
    "estimate_well_null",
    "estimate_null",
    # Exact pooling by summation -- the property the whole design rests on.
    "pool_histograms",
    # Raw + null -> explicitly named additions. Never overwrites an extracted column.
    "correct_row",
    # Versioned so a later estimator ADDS a value rather than silently redefining an old column.
    "NULL_ESTIMATOR",
    "MINIMUM_POOLED_PIXELS",
    "SATURATION_CONCERN_FRACTION",
]
