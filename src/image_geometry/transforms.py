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
``GridTransform.kind`` — a CLOSED set, validated at construction — selects the raster operation:

- ``resize``   — anti-aliased scale change (INTER_AREA on downscale). Lands the physical pixel size.
- ``crop_pad`` — pure indexing; no resampling, no interpolation, no new pixel values.
- ``flip_x``   — horizontal mirror, by indexing.
- ``affine``   — warp (rotation / translation / shear), the general case.

``kind`` is execution; ``name`` is free-form provenance and never controls behavior. An unknown
``kind`` raises rather than falling through to ``affine`` — in geometry code a permissive fallback
silently produces plausible, wrong pixels.

INTERPOLATION IS CHOSEN BY THE SEMANTICS OF THE RASTER, not by the call site: ``apply_to_mask``
forces nearest regardless of a step's ``interp`` field, because a categorical raster must never be
blended into fractional values.

That makes a single ``interp`` field insufficient as provenance — the effective kernel depends on
the step kind, the scale direction, AND whether the passenger is an image or a mask. So every step
records the RESOLVED choice for both passenger types in ``params["image_interp"]`` and
``params["mask_interp"]``. ``interp`` remains the requested image interpolation for affine steps;
read the params for what actually ran.

PADDED PIXELS ARE NOT MEASURED ZEROS
------------------------------------
``crop_pad`` and ``affine`` zero-fill outside the source. For a mask or a BF background that is
harmless, but for quantitative fluorescence a zero-filled pixel is **out of bounds, not observed
zero signal**. Any measurement that could sample outside the source must carry a support mask
alongside the image (1 = backed by source pixels, 0 = synthesized padding) and exclude padding
rather than averaging it in. ``support_mask_for`` builds one.
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

# Step-kind tokens — the CLOSED set of execution semantics.
RESIZE = "resize"
CROP_PAD = "crop_pad"
FLIP_X = "flip_x"
AFFINE = "affine"

StepKind = Literal["resize", "crop_pad", "flip_x", "affine"]
_VALID_KINDS = frozenset({RESIZE, CROP_PAD, FLIP_X, AFFINE})


@dataclass(frozen=True)
class GridTransform:
    """One raster step.

    ``kind`` and ``name`` are deliberately separate:

    * ``kind`` is the CLOSED execution semantics — what this step actually does to pixels. It is
      validated at construction, so an unknown value fails immediately rather than silently
      falling through to a generic affine.
    * ``name`` is free-form human provenance ("rotate_center", "translate_anchor"). It never
      controls behavior.

    Conflating them means a typo changes execution: a step built as ``"reszie"`` would quietly warp
    instead of resampling, producing plausible, aliased output with no error. In geometry code a
    permissive fallback is a defect, not a convenience.
    """

    kind: StepKind
    name: str
    affine_2x3: np.ndarray
    in_shape_yx: tuple[int, int]
    out_shape_yx: tuple[int, int]
    interp: Interp
    params: dict

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"GridTransform: unknown kind={self.kind!r}. Valid kinds: {sorted(_VALID_KINDS)}. "
                "`kind` selects execution semantics and is closed; use `name` for free-form "
                "provenance."
            )


def resize_step(
    *,
    in_shape_yx: tuple[int, int],
    out_shape_yx: tuple[int, int],
    anti_alias: bool = True,
    name: str = "resize",
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
    shrinking = out_shape_yx[0] < in_shape_yx[0] or out_shape_yx[1] < in_shape_yx[1]
    image_interp = "area" if (anti_alias and shrinking) else "linear"
    return GridTransform(
        kind=RESIZE,
        name=name,
        affine_2x3=np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64),
        in_shape_yx=(int(in_shape_yx[0]), int(in_shape_yx[1])),
        out_shape_yx=(int(out_shape_yx[0]), int(out_shape_yx[1])),
        interp="linear",
        params={
            "anti_alias": bool(anti_alias),
            # REALIZED scale, derived from the actual grids. Never a requested factor: integer
            # output dims mean requested != realized, and coordinate truth must follow what the
            # engine did. Deriving from shapes makes the correct value the only reachable one.
            "scale_y": float(sy),
            "scale_x": float(sx),
            # RESOLVED interpolation per passenger type, so provenance is not a wish. A bare
            # `interp` field cannot describe this step: the effective kernel depends on scale
            # direction AND on whether the passenger is an image or a categorical mask.
            "image_interp": image_interp,
            "mask_interp": "nearest",
            "affine_convention": "opencv_xy",
        },
    )


def crop_pad_step(
    *,
    in_shape_yx: tuple[int, int],
    y0: int,
    x0: int,
    out_shape_yx: tuple[int, int],
    name: str = "crop_pad",
) -> GridTransform:
    """Extract a window by INDEXING — no resampling, no interpolation, no invented pixel values.

    The window may extend outside the input; the outside region is zero-filled. Keeping this as a
    distinct kind (rather than a translation in the affine) is what lets the final placement step be
    exactly lossless, and it is why a chain's last step introduces no error of its own.
    """
    return GridTransform(
        kind=CROP_PAD,
        name=name,
        affine_2x3=np.array([[1.0, 0.0, float(-x0)], [0.0, 1.0, float(-y0)]], dtype=np.float64),
        in_shape_yx=(int(in_shape_yx[0]), int(in_shape_yx[1])),
        out_shape_yx=(int(out_shape_yx[0]), int(out_shape_yx[1])),
        interp="nearest",
        params={
            "y0": int(y0), "x0": int(x0),
            # No resampling happens at all, so both passenger types are exact.
            "image_interp": "none", "mask_interp": "none",
            "border_mode": "constant", "border_value": 0.0,
            "affine_convention": "opencv_xy",
        },
    )


def flip_x_step(
    *,
    shape_yx: tuple[int, int],
    interp: Interp = "nearest",
    name: str = "flip_x",
    params: Optional[dict] = None,
) -> GridTransform:
    """Mirror horizontally, by INDEXING — ``cv2.flip(arr, 1)``, no interpolation.

    The recorded affine is the equivalent coordinate mapping ``x -> (w - 1) - x``, kept for
    COORDINATE truth so ``composite_affine`` can map points through the chain. Executing it as a
    warp instead would be byte-identical here (measured), but the indexing path is exact by
    construction rather than by coincidence, and it stays exact if a future step composes a
    non-integer scale in front of it.
    """
    h, w = int(shape_yx[0]), int(shape_yx[1])
    merged = {"affine_convention": "opencv_xy"}
    if params:
        merged.update(params)
    return GridTransform(
        kind=FLIP_X,
        name=name,
        affine_2x3=np.array([[-1.0, 0.0, float(w - 1)], [0.0, 1.0, 0.0]], dtype=np.float64),
        in_shape_yx=(h, w),
        out_shape_yx=(h, w),
        interp=interp,
        params=merged,
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
    merged.setdefault("image_interp", interp)
    merged.setdefault("mask_interp", "nearest")
    return GridTransform(
        kind=AFFINE,
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
                    kind=AFFINE,
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
        """Render a categorical raster. Nearest everywhere, unconditionally.

        CATEGORICAL rasterization — not fractional-coverage or area-preserving resampling. Nearest
        preserves labels and avoids the fractional-mask threshold latches that plague bilinear mask
        resizing, but it does NOT preserve area exactly, and structures smaller than an output pixel
        can vanish entirely on a strong downscale.

        So do not read a transformed mask's pixel count as a precise physical area integral. If a
        future consumer needs that, it wants a separate fractional-coverage path
        (``apply_to_coverage``), not a change to this one.
        """
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


def support_mask_for(chain: TransformChain, in_shape_yx: tuple[int, int]) -> np.ndarray:
    """Which output pixels are backed by real source pixels — 1 — versus synthesized padding — 0.

    Zero-fill is invisible in the rendered image: a padded pixel and a genuinely dark pixel are both
    0. For a binary mask or a BF background that is harmless. For QUANTITATIVE FLUORESCENCE it is
    not — averaging padding into a background annulus silently biases the estimate toward zero, in
    proportion to how close the embryo sits to the frame edge. That makes the bias systematic and
    position-dependent, which is the worst kind.

    Built by pushing an all-ones raster through the same chain with MASK semantics (nearest
    everywhere), so the support is exactly the region the transform drew from — no interpolation
    softening its edge.

    Consumers should treat ``support == 0`` as *no measurement*, distinct from *measured zero*, and
    report the covered fraction rather than silently shrinking the denominator.
    """
    ones = np.ones((int(in_shape_yx[0]), int(in_shape_yx[1])), dtype=np.uint8)
    return (chain.apply_to_mask(ones) > 0).astype(np.uint8)


def _apply_step(arr: np.ndarray, t: GridTransform, *, is_mask: bool) -> np.ndarray:
    """Execute one step. Dispatch is on `kind`; interpolation is decided by raster semantics.

    `name` is provenance and is never read here — a step named "resize" whose kind is `affine`
    warps, and vice versa. The closed dispatch below ends in an explicit raise: a newly added kind
    that nobody wired up must fail loudly, not fall through to the affine branch and emit
    plausible, aliased pixels.
    """
    if cv2 is None:
        raise ImportError("cv2 is required to apply transforms.")
    h_out, w_out = t.out_shape_yx

    if t.kind == FLIP_X:
        return cv2.flip(arr, 1)

    if t.kind == CROP_PAD:
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

    if t.kind == RESIZE:
        if is_mask:
            flags = cv2.INTER_NEAREST
        elif t.params.get("anti_alias", True) and (h_out < arr.shape[0] or w_out < arr.shape[1]):
            # THE anti-alias branch. INTER_AREA is meaningful only on cv2.resize (warpAffine
            # ignores it), and only when shrinking.
            flags = cv2.INTER_AREA
        else:
            flags = cv2.INTER_LINEAR
        return cv2.resize(arr.astype(np.float32), (w_out, h_out), interpolation=flags)

    if t.kind == AFFINE:
        flags = cv2.INTER_NEAREST if (is_mask or t.interp == "nearest") else cv2.INTER_LINEAR
        return cv2.warpAffine(
            arr.astype(np.float32), t.affine_2x3.astype(np.float32), (w_out, h_out), flags=flags
        )

    # Unreachable while `kind` validation and this dispatch agree. If they ever diverge — a kind
    # added to the closed set but not wired here — fail rather than silently warping.
    raise AssertionError(f"Unhandled step kind: {t.kind!r}")


# Backwards-compatible alias: this was the private entrypoint before typed step kinds existed.
_apply_affine = _apply_step
