"""Compatibility shim — the transform primitives now live in the root ``image_geometry`` package.

``GridTransform`` / ``TransformChain`` / ``Interp`` were promoted out of the analysis layer because
they are generic image-grid infrastructure: they import only numpy and cv2 and contain no embryo,
canonical-grid, or optimal-transport vocabulary. Both ``analyze`` and ``data_pipeline`` are clients,
so neither owns them.

This module re-exports them so existing deep imports keep working:

    from analyze.utils.coord.transforms import TransformChain   # still fine

Prefer the canonical path in new code:

    from image_geometry import TransformChain
"""

from __future__ import annotations

from image_geometry.transforms import GridTransform, Interp, TransformChain

__all__ = ["GridTransform", "Interp", "TransformChain"]
