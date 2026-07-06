"""
DEPRECATED shim. The support-connectivity statistics moved to `support_geometry.py`
(Stage 2 of the phenotype-geometry decision tree). This module re-exports the old
public names so existing imports keep working.

Prefer importing from support_geometry directly:
    from support_geometry import compute_support_geometry, SupportGeometryBundle
"""

from __future__ import annotations

from .support_geometry import (  # noqa: F401
    ConnectednessResult,
    SupportGeometryBundle,
    compute_support_geometry,
    connectedness_pvalue,
    fiedler_value,
    mst_max_edge,
    normalize_shape,
    relative_valley_depth,
    valley_depth,
)
