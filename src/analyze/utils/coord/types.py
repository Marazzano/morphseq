"""Public datatypes for `analyze.utils.coord`.

This module is public surface only. No algorithms should live here.

``BoxYX`` / ``CoordConvention`` / ``TransformChain`` are RE-EXPORTED from the root
``image_geometry`` package, which owns the generic image-coordinate primitives. They were promoted
out of here because they know nothing about embryos, canonical grids, or optimal transport, and the
data pipeline needs them too. Importing them from this module still works and always will.

What stays HERE is analysis-specific by construction: ``CoordFrameId`` names analysis frames,
``Frame`` carries a yolk mask, and the Canonical* results describe a particular analysis product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from image_geometry import BoxYX, CoordConvention, TransformChain


CoordFrameId = Literal["work_grid", "canonical_grid", "unknown"]


@dataclass
class Frame:
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    yolk_mask: Optional[np.ndarray] = None
    um_per_px: float = float("nan")
    meta: Optional[dict] = None


@dataclass(frozen=True)
class CanonicalGrid:
    """Descriptor-only canonical reference frame."""

    um_per_px: float
    shape_yx: tuple[int, int]
    anchor_mode: str
    anchor_yx: tuple[float, float]
    coord_convention: CoordConvention = "yx"


@dataclass
class CanonicalMaskResult:
    mask: np.ndarray
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict
    qc: Optional[dict] = None
    content_bbox_yx: Optional[BoxYX] = None
    # Tight bbox of nonzero pixels in canonical canvas coordinates (half-open,
    # no padding).  Set by to_canonical_grid_mask.  None only for legacy results
    # or empty masks.


@dataclass
class CanonicalImageResult:
    image: np.ndarray
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict


@dataclass
class CanonicalFrameResult:
    frame: Frame
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict


@dataclass
class RegisterResult:
    transform: TransformChain
    applied: bool
    meta: dict
    moving_in_fixed: Optional[np.ndarray] = None
