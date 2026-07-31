"""Pixel-coordinate bounding boxes on image grids.

Moved here from ``analyze.utils.coord.types`` when the generic coordinate primitives were promoted
out of the analysis layer: ``BoxYX`` is pure index arithmetic with no knowledge of embryos, canonical
grids, or optimal transport, so both the pipeline and the analysis stack are clients of it rather
than owners. The Canonical*/Frame result containers stayed behind in ``analyze`` because they DO
encode analysis-specific products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


CoordConvention = Literal["yx"]


@dataclass(frozen=True)
class BoxYX:
    """Half-open bounding box [y0, y1) x [x0, x1) in canvas pixel coordinates.

    Contract:
    - Half-open slices: mask[box.to_slices()] selects exactly the content.
    - Canonical pixel coordinates (origin top-left, yx order).
    - Tight by convention; callers apply padding explicitly via .pad().
    - BoxYX is purely index-space. Physical size = h * um_per_px.
    - crop_pad_hw padding is always bottom/right only -- preserves top-left
      anchor alignment.
    """

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.h * self.w

    def to_slices(self) -> tuple[slice, slice]:
        """(slice(y0,y1), slice(x0,x1)) -- use for direct numpy indexing."""
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def contains(self, other: "BoxYX") -> bool:
        """Check if this box fully contains another box."""
        return (
            self.y0 <= other.y0
            and self.y1 >= other.y1
            and self.x0 <= other.x0
            and self.x1 >= other.x1
        )

    def union(self, other: "BoxYX") -> "BoxYX":
        """Smallest box containing both."""
        return BoxYX(
            y0=min(self.y0, other.y0),
            y1=max(self.y1, other.y1),
            x0=min(self.x0, other.x0),
            x1=max(self.x1, other.x1),
        )

    def pad(self, pad_y: int, pad_x: int) -> "BoxYX":
        """Expand by pad_y/pad_x on all sides. Does NOT clamp to canvas."""
        return BoxYX(
            y0=self.y0 - pad_y,
            y1=self.y1 + pad_y,
            x0=self.x0 - pad_x,
            x1=self.x1 + pad_x,
        )

    def clamp(self, h: int, w: int) -> "BoxYX":
        """Clamp to canvas [0,h) x [0,w)."""
        return BoxYX(
            y0=max(0, self.y0),
            y1=min(h, self.y1),
            x0=max(0, self.x0),
            x1=min(w, self.x1),
        )

    def validate(self, h: int, w: int) -> None:
        """Assert 0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w. Raises ValueError."""
        if not (0 <= self.y0 <= self.y1 <= h):
            raise ValueError(
                f"BoxYX y-range invalid: 0 <= {self.y0} <= {self.y1} <= {h} failed"
            )
        if not (0 <= self.x0 <= self.x1 <= w):
            raise ValueError(
                f"BoxYX x-range invalid: 0 <= {self.x0} <= {self.x1} <= {w} failed"
            )

    def intersects(self, other: "BoxYX") -> bool:
        """True when the two half-open boxes overlap in both axes.

        Added during the promotion (no prior consumer needed it): the snip neighbor-exclusion path
        prefilters candidate masks by testing an annulus-expanded target box against each neighbor's
        box, which is this predicate. Half-open semantics mean boxes that merely touch edge-to-edge
        do NOT intersect.
        """
        return (
            self.y0 < other.y1
            and other.y0 < self.y1
            and self.x0 < other.x1
            and other.x0 < self.x1
        )

    @staticmethod
    def from_mask(mask: np.ndarray, *, threshold: float = 0.0) -> Optional["BoxYX"]:
        """Tight bbox of pixels ``> threshold``. Returns None if nothing exceeds it.

        THE THRESHOLD IS SEMANTIC, NOT A TUNING KNOB, and the default deliberately preserves the
        historical ``> 0`` behavior:

        * ``threshold=0.0`` — for a CRISP mask (bool, 0/1, or 0/255) where any nonzero pixel is
          foreground. This is the honest reading of a mask that has not been resampled.
        * ``threshold=0.5`` — for a FLOAT mask that has been through interpolation, where bilinear
          resampling has smeared the boundary into fractional values. Taking ``> 0`` there would
          include every pixel touched by the interpolation kernel's tail, inflating the box by
          roughly the kernel radius on every side.

        Callers pass the value matching their raster's provenance. Unifying the two would silently
        change box extents for one class of caller, so the parameter is explicit and required at the
        call sites that need 0.5.
        """
        ys, xs = np.where(mask > threshold)
        if ys.size == 0:
            return None
        return BoxYX(
            int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        )
