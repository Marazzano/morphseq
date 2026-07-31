"""Transformation primitives for image grids.

`GridTransform` uses OpenCV's affine convention on (x, y) coordinates.
Coordinate *values* in meta are in yx convention unless stated otherwise.

TWO PARALLEL TRUTHS
-------------------
A chain records both, and they are not the same thing:

- **Coordinate truth** — where a location in the source lands in the output. This composes into a
  single matrix, and ``composite_affine()`` derives it.
- **Raster truth** — how the pixel VALUES were produced. This does NOT compose into a matrix, and
  collapsing the chain into one loses it.

The distinction is load-bearing, not pedantic: *correct geometric transform ≠ correct image
resampling*. An affine matrix describes the coordinate mapping perfectly while saying nothing about
the frequency content that must be removed before sampling onto a coarser grid. Downsampling changes
the sampling rate and needs a low-pass prefilter first, or high frequencies fold into false
structure; rotation remaps coordinates and needs interpolation but no area integration. They are
different problems and get different steps.

Measured, not assumed: ``cv2.warpAffine`` **silently ignores ``INTER_AREA``** — passing it produces
output byte-identical to ``INTER_LINEAR``. So a fused affine containing a large downscale cannot
anti-alias at all, no matter which flag it carries. On a checkerboard downscaled 0.414x, a properly
anti-aliased resize collapses it to flat grey (std 1.0-2.1) while ``warpAffine`` leaves std 33.3 of
pure aliasing artifact. That is why ``Resize`` is a separate step kind and not a scale baked into
the affine.

STEP KINDS
----------
``GridTransform.name`` selects the raster operation:

- ``resize``    — anti-aliased scale change (INTER_AREA on downscale). Lands the physical pixel size.
- ``crop_pad``  — pure indexing; no resampling, no interpolation, no new pixel values.
- ``flip_x``    — horizontal mirror.
- anything else — affine warp (rotation / translation / shear), the general case.

INTERPOLATION IS CHOSEN BY THE SEMANTICS OF THE RASTER, not by the call site: ``apply_to_mask``
forces nearest regardless of what a transform's ``interp`` field says, because a categorical raster
must never be blended into fractional values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


Interp = Literal["nearest", "linear"]

# Step-kind tokens. Names are the dispatch key in _apply_step.
RESIZE = "resize"
CROP_PAD = "crop_pad"
FLIP_X = "flip_x"


@dataclass(frozen=True)
class GridTransform:
    name: str
    affine_2x3: np.ndarray
    in_shape_yx: tuple[int, int]
    out_shape_yx: tuple[int, int]
    interp: Interp
    params: dict


def resize_step(
    *,
    in_shape_yx: tuple[int, int],
    out_shape_yx: tuple[int, int],
    anti_alias: bool = True,
) -> GridTransform:
    """An anti-aliased scale change — the step that owns sampling-rate changes.

    ``anti_alias=True`` selects ``INTER_AREA`` when shrinking, which area-averages the source pixels
    covering each output pixel. That prefilter is what keeps high frequencies from folding into
    false structure; without it a ~2.4x downscale of embryo texture adds ~43% spurious
    high-frequency energy, which downstream edge-density metrics read as real detail.

    Upscaling uses linear regardless — there is nothing to prefilter when adding samples.

    The affine is recorded for COORDINATE truth (so ``composite_affine`` can map points through the
    chain); it is deliberately not what executes. See the module docstring.
    """
    sy = out_shape_yx[0] / in_shape_yx[0]
    sx = out_shape_yx[1] / in_shape_yx[1]
    return GridTransform(
        name=RESIZE,
        affine_2x3=np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64),
        in_shape_yx=(int(in_shape_yx[0]), int(in_shape_yx[1])),
        out_shape_yx=(int(out_shape_yx[0]), int(out_shape_yx[1])),
        interp="linear",
        params={
            "anti_alias": bool(anti_alias),
            "scale_y": float(sy),
            "scale_x": float(sx),
            "affine_convention": "opencv_xy",
        },
    )


def crop_pad_step(
    *,
    in_shape_yx: tuple[int, int],
    y0: int,
    x0: int,
    out_shape_yx: tuple[int, int],
) -> GridTransform:
    """Extract a window by INDEXING — no resampling, no interpolation, no invented pixel values.

    The window may extend outside the input; the outside region is zero-filled. Keeping this as a
    distinct kind (rather than a translation in the affine) is what lets the final placement step be
    exactly lossless, and it is why a chain's last step introduces no error of its own.
    """
    return GridTransform(
        name=CROP_PAD,
        affine_2x3=np.array([[1.0, 0.0, float(-x0)], [0.0, 1.0, float(-y0)]], dtype=np.float64),
        in_shape_yx=(int(in_shape_yx[0]), int(in_shape_yx[1])),
        out_shape_yx=(int(out_shape_yx[0]), int(out_shape_yx[1])),
        interp="nearest",
        params={"y0": int(y0), "x0": int(x0), "affine_convention": "opencv_xy"},
    )


def affine_step(
    *,
    affine_2x3: np.ndarray,
    in_shape_yx: tuple[int, int],
    out_shape_yx: tuple[int, int],
    interp: Interp = "linear",
    name: str = "affine",
    params: Optional[dict] = None,
) -> GridTransform:
    """A general affine warp — rotation, translation, shear.

    Keep large scale changes OUT of here and in a ``resize_step``: warpAffine's filter footprint is a
    small local neighborhood, far too narrow when shrinking substantially, and it cannot be widened
    by a flag (INTER_AREA is ignored here).
    """
    merged = {"affine_convention": "opencv_xy"}
    if params:
        merged.update(params)
    return GridTransform(
        name=name,
        affine_2x3=np.asarray(affine_2x3, dtype=np.float64),
        in_shape_yx=(int(in_shape_yx[0]), int(in_shape_yx[1])),
        out_shape_yx=(int(out_shape_yx[0]), int(out_shape_yx[1])),
        interp=interp,
        params=merged,
    )


@dataclass
class TransformChain:
    transforms: list[GridTransform] = field(default_factory=list)

    @staticmethod
    def identity(
        *,
        shape_yx: tuple[int, int],
        interp: Interp,
        name: str = "identity",
    ) -> "TransformChain":
        """Identity convention: affine=[[1,0,0],[0,1,0]], in_shape==out_shape, interp matches downstream apply."""
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return TransformChain(
            transforms=[
                GridTransform(
                    name=name,
                    affine_2x3=affine,
                    in_shape_yx=shape_yx,
                    out_shape_yx=shape_yx,
                    interp=interp,
                    params={"affine_convention": "opencv_xy"},
                )
            ]
        )

    def __len__(self) -> int:
        return len(self.transforms)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """Render a categorical raster. Nearest everywhere, unconditionally."""
        if cv2 is None:
            raise ImportError("cv2 is required to apply transforms.")
        out = mask
        for t in self.transforms:
            out = _apply_step(out, t, is_mask=True)
        return out

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Render an intensity raster. Anti-aliased on downscale, linear on warp."""
        if cv2 is None:
            raise ImportError("cv2 is required to apply transforms.")
        out = image
        for t in self.transforms:
            out = _apply_step(out, t, is_mask=False)
        return out

    def composite_affine(self) -> np.ndarray:
        """COORDINATE truth: one 2x3 mapping a source (x, y) to its output location.

        Use this to map POINTS through the chain. Do NOT use it to render pixels — it cannot express
        the resize step's prefilter, which is precisely the raster truth the chain exists to keep
        separate (see the module docstring).
        """
        composite = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        for t in self.transforms:
            step = np.vstack([np.asarray(t.affine_2x3, dtype=np.float64), [0.0, 0.0, 1.0]])
            full = np.vstack([composite, [0.0, 0.0, 1.0]])
            composite = (step @ full)[:2, :]
        return composite


def _apply_step(arr: np.ndarray, t: GridTransform, *, is_mask: bool) -> np.ndarray:
    """Execute one step. Dispatch is on `name`; interpolation is decided by raster semantics."""
    if cv2 is None:
        raise ImportError("cv2 is required to apply transforms.")
    h_out, w_out = t.out_shape_yx

    if t.name == FLIP_X:
        return cv2.flip(arr, 1)

    if t.name == CROP_PAD:
        # Pure indexing. No interpolation, so a mask stays exactly itself.
        y0 = int(t.params["y0"])
        x0 = int(t.params["x0"])
        out = np.zeros((h_out, w_out), dtype=arr.dtype)
        src_y0, src_x0 = max(y0, 0), max(x0, 0)
        src_y1, src_x1 = min(y0 + h_out, arr.shape[0]), min(x0 + w_out, arr.shape[1])
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return out
        out[src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0] = arr[src_y0:src_y1, src_x0:src_x1]
        return out

    if t.name == RESIZE:
        if is_mask:
            flags = cv2.INTER_NEAREST
        elif t.params.get("anti_alias", True) and (h_out < arr.shape[0] or w_out < arr.shape[1]):
            # THE anti-alias branch. INTER_AREA is meaningful only on cv2.resize (warpAffine
            # ignores it), and only when shrinking.
            flags = cv2.INTER_AREA
        else:
            flags = cv2.INTER_LINEAR
        return cv2.resize(arr.astype(np.float32), (w_out, h_out), interpolation=flags)

    flags = cv2.INTER_NEAREST if (is_mask or t.interp == "nearest") else cv2.INTER_LINEAR
    return cv2.warpAffine(
        arr.astype(np.float32), t.affine_2x3.astype(np.float32), (w_out, h_out), flags=flags
    )


# Backwards-compatible alias: this was the private entrypoint before typed step kinds existed.
_apply_affine = _apply_step
