"""Biological policy for embryo geometry.

``image_geometry`` knows how to place an object on a grid four different ways. It cannot
know which of those four is right, because "right" means head-forward and dorsal-up --
statements about a zebrafish, not about a raster. This package owns that judgment.

The split is the point:

- generic candidate MECHANICS (enumerate rot x flip, place on a grid) -> ``image_geometry``
- embryo POLICY (yolk-left AP, back-up DV, the fallbacks, the evidence) -> here
- consumer-specific RENDERING (snip frame vs 256x576 canonical canvas) -> each caller

Decisions returned here are canvas-independent: no grid shape, no um/px, no anchor. Each
caller composes its own final rotation.

There are FOUR rules in production, not one: two for the yolk-present case and two for the
yolk-missing case, and within each pair they disagree. So BOTH slots are explicit
parameters from closed sets, never hidden defaults. The yolk-present default
(``yolk_back``) is a COMPATIBILITY choice, not an endorsement -- ``yolk_vs_embryo_com``
compares two landmarks on the animal and is the better rule. See ``orientation`` for the
full account.

This package must not import ``data_pipeline`` or ``analyze``; both are clients.
"""

from .orientation import (
    EMPTY_MASK,
    DEGENERATE_MASK,
    LEGACY_UPPER_LEFT_COM,
    LEGACY_VERT_RATIO,
    MISSING_YOLK,
    NO_YOLK_POLICIES,
    PCA_AXIS,
    REGIONPROPS_AXIS,
    YOLK_BACK,
    YOLK_DISABLED,
    YOLK_PRESENT_POLICIES,
    YOLK_VS_EMBRYO_COM,
    YOLK_WARPED_OFF_GRID,
    EmbryoOrientationDecision,
    EmbryoOrientationPolicy,
    compute_back_point,
    decide_embryo_orientation,
    orientation_policy,
)

__all__ = [
    "EmbryoOrientationDecision",
    "EmbryoOrientationPolicy",
    "orientation_policy",
    "decide_embryo_orientation",
    "compute_back_point",
    # Closed vocabularies -- branch on these, never on free text.
    "YOLK_BACK",
    "YOLK_VS_EMBRYO_COM",
    "YOLK_PRESENT_POLICIES",
    "LEGACY_UPPER_LEFT_COM",
    "LEGACY_VERT_RATIO",
    "NO_YOLK_POLICIES",
    "MISSING_YOLK",
    "YOLK_DISABLED",
    "YOLK_WARPED_OFF_GRID",
    "EMPTY_MASK",
    "DEGENERATE_MASK",
    "PCA_AXIS",
    "REGIONPROPS_AXIS",
]
