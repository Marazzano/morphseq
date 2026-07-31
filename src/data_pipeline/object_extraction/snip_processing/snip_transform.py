"""The snip transform seam: one physical crop/rotation recipe, expressed per pixel grid.

THE PROBLEM THIS SOLVES. A snip is defined by biology (where the embryo is, which way it points),
but it is *rendered* onto whatever pixel grid a materialized image product happens to use. Those two
facts were entangled in ``extraction.py`` + ``rotation.py``: geometry was derived from whatever the
image had been rescaled to, so running the same code on two channels of the same embryo produced
correctly-scaled but NON-IDENTICAL crops whenever the products differed in calibration.

That is not hypothetical. There is no contract anywhere forcing two channels' materialized frames to
share dimensions or calibration — ``downsample_factor`` is resolved PER PRODUCT KEY, and
``BF__z_stack`` already ships ``downsample_factor: 4`` while ``BF__projection__focus_stack`` is
native. Two products of the SAME channel already disagree.

THE SEPARATION. Three concerns, previously one function:

    biological geometry    segmentation defines the physical embryo geometry
    pixel-grid conversion  each product supplies its own grid + calibration; this module maps the
                           shared physical recipe onto it
    image rendering        the recipe layer (snip_recipes.py) decides what happens photometrically

TWO LEVELS, and the reason there are two. An affine matrix is expressed IN a coordinate system, so a
single matrix cannot be "the" transform for products on different grids:

    CanonicalSnipTransform   the PHYSICAL recipe, in micrometers — portable across grids
    ResolvedSnipTransform    that recipe spoken in ONE product's pixel dialect — directly applicable

Derive the canonical form once from the segmentation mask, then resolve it per product. RFP is not a
special fork; it is another product asking this engine to speak its coordinate dialect.

INTERPOLATION POLICY. Images linear, masks nearest — enforced by separate entrypoints
(``apply_transform_to_image`` / ``apply_transform_to_mask``) rather than a flag, because the legacy
path bilinearly resized binary masks into fractional values and then repaired them with scattered
``> 0.5`` thresholds. The same PHYSICAL transform does not require identical interpolation; it
requires identical physical coordinates, rotation, crop center, target extent, and output grid.

"DERIVED ONCE" MEANS ONE IMPLEMENTATION, NOT ONE INVOCATION. Separate Snakemake product jobs cannot
share an in-memory object. Derivation is deterministic from identical mask inputs, so each product
job re-derives the same canonical transform rather than consuming a persisted geometry artifact. The
DRY property that matters is one implementation; ``assert_transforms_equivalent`` makes the
determinism directly testable.

Import direction: this module may import numpy/cv2/skimage and the shared mask helpers. It MUST NOT
import the entrypoint, orchestration, tasks, or Snakemake rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from image_geometry import TransformChain, affine_step, resize_step

from data_pipeline.object_extraction.snip_processing.rotation import get_embryo_rotation_angle

# Physical field-of-view agreement tolerance, as a fraction. Two products describing the same
# acquisition should cover the same physical extent: width_px * um_per_px must agree. 2% absorbs
# integer-rounding of downsampled dimensions (expected_downsampled_dims rounds independently per
# product) without admitting a genuinely different canvas.
FIELD_OF_VIEW_RELATIVE_TOLERANCE = 0.02

# Aspect ratio agreement, same reasoning. A nearest-neighbor resize can make two unrelated canvases
# look aligned while quietly stretching one, so shape compatibility is checked explicitly rather
# than inferred from the resize succeeding.
ASPECT_RATIO_RELATIVE_TOLERANCE = 0.02

IMAGE_INTERPOLATION = "linear"
MASK_INTERPOLATION = "nearest"

# Border policy. Recorded on the resolved transform, not left implicit: the CROP OPERATION goes away
# under continuous centering, but its SEMANTICS must not. Legacy zero-filled outside the source, and
# that is load-bearing downstream — augment_snip blends the embryo against a synthetic noise
# background, so what sits outside the mask is a real input to the rendered snip, not padding
# nobody looks at. A replay that reproduced the interior but diverged at the edges would be wrong.
BORDER_MODE = "constant"
BORDER_VALUE = 0.0

# Centering mode. TRANSITIONAL — exists so the resample-kernel migration and the centering repair
# can land as SEPARATE, independently attributable commits rather than one fused pixel change.
#
# THE MODE SELECTS TWO THINGS TOGETHER, deliberately: the centering REFERENCE (which mask the center
# is measured on) and the QUANTIZER (whether that center is truncated to whole pixels). Those are
# exactly the pair the centering commit changes, so they must move together or neither commit
# isolates what it claims to.
#
#   "legacy_latched"  REFERENCE: the rescaled, ROTATED mask, measured on the bounds-expanded canvas
#                     cv2 produces — i.e. what crop_to_embryo_bounds sees (extraction.py:103-116).
#                     QUANTIZER: int() truncation of that center.
#                     Wrong on both counts, and known to translate the whole canvas by a pixel when a
#                     sub-grey-level input change tips a boundary pixel (~10% of real snips).
#   "continuous"      REFERENCE: the SOURCE-grid mask, measured BEFORE rotation and carried in
#                     physical units. QUANTIZER: none — the center is folded into the affine so
#                     placement is exact by construction.
#
# The reference change is the substantive one. Measuring pre-rotation in source coordinates is what
# makes ONE canonical recipe expressible on ANY product grid: a post-rotation center lives on that
# product's rotated canvas and cannot be shared, which would break the BF/RFP sibling
# registerability this seam exists to provide. mean-of-occupied-index-range is NOT
# rotation-equivariant, so the two references genuinely differ for any non-axis-aligned embryo —
# this is a real placement change, not a rounding detail.
#
# `legacy_latched` is retained only until the centering commit lands, and is deleted with it. It is
# NOT a supported configuration — no config surface reaches it, and a test pins that.
CENTERING_LATCHED = "legacy_latched"
CENTERING_CONTINUOUS = "continuous"
DEFAULT_CENTERING = CENTERING_LATCHED

# The mode vocabulary is CLOSED and only one member is a destination. Named `legacy_latched` rather
# than `latched` so the word "legacy" appears at every call site and in every provenance row — a
# flag that reads as a neutral option is how dual behavior becomes permanent and quietly splits a
# dataset in half.
SUPPORTED_CENTERING_MODES = frozenset({CENTERING_LATCHED, CENTERING_CONTINUOUS})

# Modes a production config may select. EMPTY during migration: neither mode is configurable, both
# are chosen in code, and the comparison is run by explicit call. After activation this becomes
# {CENTERING_CONTINUOUS} and the legacy branch is DELETED in a dedicated cleanup commit — see
# TODO(remove-legacy-latched-centering). Leaving the branch reachable would be a trapdoor through
# which a future run could silently recreate the old geometry.
CONFIGURABLE_CENTERING_MODES: frozenset[str] = frozenset()


class SnipTransformError(ValueError):
    """A transform could not be derived, or could not be honestly expressed on a product grid."""


@dataclass(frozen=True)
class CanonicalSnipTransform:
    """The PHYSICAL snip recipe — what to crop and how to rotate, in units that survive a grid change.

    Every field is either physical (micrometers), angular, or a property of the OUTPUT grid. Nothing
    here is expressed in the pixels of the source product, which is what makes it portable.

    ``crop_center_um_xy`` is deliberately physical rather than the legacy integer pixel centroid: the
    legacy center was computed AFTER rescaling and rotation, so it only meant something on that one
    intermediate grid and could not be carried to another product.

    ``source_shape_hw`` / ``source_um_per_px`` describe the grid the transform was DERIVED from (the
    segmentation mask's grid). They are retained for provenance and to let ``transform_for_product``
    check physical field-of-view agreement — not because the recipe depends on that grid.
    """

    source_shape_hw: tuple[int, int]
    source_um_per_px: float
    target_um_per_px: float
    rotation_angle_rad: float
    crop_center_um_xy: tuple[float, float]
    snip_frame_shape_hw: tuple[int, int]

    # TRANSITIONAL, and excluded from equality on purpose (compare=False). CENTERING_LATCHED must
    # measure its center on the rescaled+ROTATED mask to reproduce legacy, which means the mask has
    # to survive as far as transform_for_product. It is NOT part of the physical recipe — two
    # products of the same embryo-time must still compare equal under assert_transforms_equivalent,
    # and the continuous path never reads it. Deleted with the legacy branch; see
    # TODO(remove-legacy-latched-centering).
    source_mask: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def source_field_of_view_um_wh(self) -> tuple[float, float]:
        """Physical extent of the grid this transform was derived from, as (width_um, height_um)."""
        height_px, width_px = self.source_shape_hw
        return (width_px * self.source_um_per_px, height_px * self.source_um_per_px)


@dataclass(frozen=True)
class ResolvedSnipTransform:
    """A canonical recipe expressed on ONE product's pixel grid — directly applicable.

    ``rotated_canvas_shape_hw`` is the bounds-expanded canvas cv2 produces for this rotation; the
    crop window is expressed on THAT canvas, not on the source frame, because rotation happens first.

    The crop window is carried in BOTH physical and product-pixel coordinates on purpose: product
    pixels alone cannot represent one universal recipe across products with different calibration,
    and physical alone is not auditable against the file actually written to disk.
    """

    canonical: CanonicalSnipTransform
    product_shape_hw: tuple[int, int]
    product_um_per_px: float
    scale_factor: float
    rescaled_shape_hw: tuple[int, int]
    rotation_matrix_2x3: tuple[tuple[float, float, float], ...]
    canvas_center_xy: tuple[float, float]
    crop_x0_px: int
    crop_y0_px: int
    crop_x1_px: int
    crop_y1_px: int
    crop_x0_um: float
    crop_y0_um: float
    crop_x1_um: float
    crop_y1_um: float
    border_mode: str
    border_value: float
    centering: str

    @property
    def output_shape_hw(self) -> tuple[int, int]:
        return self.canonical.snip_frame_shape_hw

    def to_chain(self) -> TransformChain:
        """The two-step raster chain this transform executes: anti-aliased resize, then affine.

        Returned rather than hidden so the RASTER truth is inspectable — which prefilter ran, onto
        which grid, with what border policy. ``chain.composite_affine()`` gives the COORDINATE truth
        (where a source point lands) and deliberately cannot express the prefilter.

        There is no crop step: the affine writes directly onto the snip canvas, which is what
        removes the legacy rasterize -> threshold -> truncate -> crop loop entirely.
        """
        return TransformChain([
            resize_step(in_shape_yx=self.product_shape_hw, out_shape_yx=self.rescaled_shape_hw),
            affine_step(
                affine_2x3=np.asarray(self.rotation_matrix_2x3, dtype=np.float64),
                in_shape_yx=self.rescaled_shape_hw,
                out_shape_yx=self.output_shape_hw,
                name="rotate_center",
                params={
                    "border_mode": self.border_mode,
                    "border_value": self.border_value,
                    "canvas_center_xy": self.canvas_center_xy,
                },
            ),
        ])


def derive_snip_transform(
    mask: np.ndarray,
    *,
    source_um_per_px: float,
    target_um_per_px: float,
    snip_frame_shape_hw: tuple[int, int],
    yolk_mask: np.ndarray | None = None,
) -> CanonicalSnipTransform:
    """Derive the physical snip recipe from a segmentation mask alone.

    NO IMAGE PIXELS ARE READ. The rotation angle comes from the mask's PCA orientation and the crop
    center from the mask's extent, which is what makes this transform shareable across every image
    product of the same embryo-time — and why an RFP snip needs no BF snip artifact.

    Args:
        mask: binary embryo mask on the source (segmentation) grid.
        source_um_per_px: calibration of that grid.
        target_um_per_px: calibration the snip is rendered at.
        snip_frame_shape_hw: the snip frame (height, width) in pixels.
        yolk_mask: optional yolk mask; when absent the angle falls back to the embryo
            mass-distribution heuristic, matching legacy behavior.

    Raises:
        SnipTransformError: on a non-positive calibration or an empty/degenerate mask.
    """
    if source_um_per_px <= 0 or target_um_per_px <= 0:
        raise SnipTransformError(
            f"derive_snip_transform: calibrations must be positive; got "
            f"source_um_per_px={source_um_per_px!r}, target_um_per_px={target_um_per_px!r}."
        )

    mask_binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if mask_binary.ndim != 2:
        raise SnipTransformError(
            f"derive_snip_transform: mask must be 2-D; got shape {mask_binary.shape}."
        )
    if not mask_binary.any():
        raise SnipTransformError(
            "derive_snip_transform: mask is empty, so there is no embryo to center on. The caller "
            "must treat this as an invalid snip rather than rendering an arbitrary crop."
        )

    if yolk_mask is None:
        yolk_binary = np.zeros_like(mask_binary)
    else:
        yolk_binary = (np.asarray(yolk_mask) > 0.5).astype(np.uint8)

    # The angle is a property of the MASK, so it is channel-independent by construction.
    rotation_angle_rad = float(get_embryo_rotation_angle(mask_binary, yolk_binary))

    # Crop center in PHYSICAL units, measured on the source grid before any rotation. Using the mean
    # of the occupied index range (not a true centroid) preserves the legacy centering behavior.
    y_indices = np.where(np.max(mask_binary, axis=1) > 0)[0]
    x_indices = np.where(np.max(mask_binary, axis=0) > 0)[0]
    center_y_px = float(np.mean(y_indices))
    center_x_px = float(np.mean(x_indices))

    return CanonicalSnipTransform(
        source_shape_hw=(int(mask_binary.shape[0]), int(mask_binary.shape[1])),
        source_um_per_px=float(source_um_per_px),
        target_um_per_px=float(target_um_per_px),
        rotation_angle_rad=rotation_angle_rad,
        crop_center_um_xy=(center_x_px * source_um_per_px, center_y_px * source_um_per_px),
        snip_frame_shape_hw=(int(snip_frame_shape_hw[0]), int(snip_frame_shape_hw[1])),
        source_mask=mask_binary,
    )


def _legacy_bounds_expanded_rotation(
    *, rescaled_shape_hw: tuple[int, int], angle_deg: float
) -> np.ndarray:
    """The rotation legacy applied: about the rescaled canvas center, onto expanded bounds.

    Mirrors ``rotation.rotate_image`` exactly, including its ``int()`` bound truncation. TRANSITIONAL
    — exists only to let CENTERING_LATCHED reproduce legacy placement; deleted with that branch.
    """
    height, width = rescaled_shape_hw
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return rotation_mat


def _legacy_rotated_bbox_center(
    *,
    canonical: CanonicalSnipTransform,
    rescaled_shape_hw: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    """Legacy's crop center: int(mean(occupied indices)) on the rescaled, ROTATED mask.

    Reproduces ``extraction.crop_to_embryo_bounds`` lines 103-116 on a mask that has been through
    ``extract_embryo_crop``'s bilinear resize and ``apply_rotation_to_snip``'s bounds-expanded
    rotation. The bilinear-resize-then-threshold is legacy's actual behavior (it resized the mask as
    float and compared ``> 0.5``), and it is reproduced rather than corrected because the whole point
    of this mode is to hold placement fixed while the kernel and render path change.

    TRANSITIONAL — deleted with CENTERING_LATCHED; see TODO(remove-legacy-latched-centering).
    """
    if canonical.source_mask is None:
        raise SnipTransformError(
            "transform_for_product: centering=legacy_latched needs the source mask to reproduce "
            "the legacy post-rotation centering reference, but this CanonicalSnipTransform carries "
            "none. Build it with derive_snip_transform (which retains the mask) rather than "
            "constructing it directly, or use centering=continuous."
        )

    rescaled_h, rescaled_w = rescaled_shape_hw
    # Legacy resized the mask as float with bilinear interpolation, then thresholded at 0.5.
    mask_rescaled = cv2.resize(
        canonical.source_mask.astype(np.float64),
        (rescaled_w, rescaled_h),
        interpolation=cv2.INTER_LINEAR,
    )
    rotation_mat = _legacy_bounds_expanded_rotation(
        rescaled_shape_hw=rescaled_shape_hw, angle_deg=angle_deg
    )
    # Recompute the expanded bounds the same way rotate_image does.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(rescaled_h * abs_sin + rescaled_w * abs_cos)
    bound_h = int(rescaled_h * abs_cos + rescaled_w * abs_sin)
    mask_rotated = cv2.warpAffine(mask_rescaled, rotation_mat, (bound_w, bound_h))

    y_indices = np.where(np.max(mask_rotated, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(mask_rotated, axis=0) > 0.5)[0]
    if y_indices.size == 0 or x_indices.size == 0:
        # Legacy returned an all-zero snip here. Centering on the canvas center reproduces that
        # outcome for a mask that has vanished, without a special-cased render path.
        return ((bound_w - 1) / 2, (bound_h - 1) / 2)

    return (float(int(np.mean(x_indices))), float(int(np.mean(y_indices))))


def transform_for_product(
    canonical: CanonicalSnipTransform,
    *,
    product_shape_hw: tuple[int, int],
    product_um_per_px: float,
    centering: str = DEFAULT_CENTERING,
) -> ResolvedSnipTransform:
    """Express a canonical recipe on one product's pixel grid, VALIDATING that this is honest.

    This is not a reshape. A nearest-neighbor resize can make two unrelated canvases look aligned
    while quietly stretching one, so physical field-of-view agreement is checked explicitly: the
    product must cover the same physical extent as the grid the transform was derived from.

    Raises:
        SnipTransformError: on non-positive calibration, or when the product's physical field of
            view or aspect ratio disagrees with the derivation grid beyond tolerance. Failing loud
            here is the point — silently proceeding produces a plausible, misregistered snip.
    """
    if product_um_per_px <= 0:
        raise SnipTransformError(
            f"transform_for_product: product_um_per_px must be positive; got {product_um_per_px!r}."
        )

    product_h, product_w = int(product_shape_hw[0]), int(product_shape_hw[1])
    if product_h <= 0 or product_w <= 0:
        raise SnipTransformError(
            f"transform_for_product: product shape must be positive; got {product_shape_hw!r}."
        )

    product_fov_w = product_w * product_um_per_px
    product_fov_h = product_h * product_um_per_px
    source_fov_w, source_fov_h = canonical.source_field_of_view_um_wh

    for axis, product_fov, source_fov in (
        ("width", product_fov_w, source_fov_w),
        ("height", product_fov_h, source_fov_h),
    ):
        if abs(product_fov - source_fov) > FIELD_OF_VIEW_RELATIVE_TOLERANCE * source_fov:
            raise SnipTransformError(
                f"transform_for_product: physical field of view disagrees on {axis} — "
                f"derivation grid {source_fov:.1f}um vs product {product_fov:.1f}um "
                f"(tolerance {FIELD_OF_VIEW_RELATIVE_TOLERANCE:.0%}). These frames do not cover the "
                "same region, so the segmentation mask does not describe the product's pixels. "
                "Check that both products come from the same acquisition and that neither carries "
                "an unexpected orientation or target_micrometers_per_pixel in its write policy."
            )

    source_aspect = canonical.source_shape_hw[1] / canonical.source_shape_hw[0]
    product_aspect = product_w / product_h
    if abs(product_aspect - source_aspect) > ASPECT_RATIO_RELATIVE_TOLERANCE * source_aspect:
        raise SnipTransformError(
            f"transform_for_product: aspect ratio disagrees — derivation grid {source_aspect:.4f} "
            f"vs product {product_aspect:.4f}. A resize would stretch one axis relative to the "
            "other; suspect a per-product `orientation` (np.rot90 transposes H/W)."
        )

    # Rescale to the target physical pixel size. The REALIZED ratio is what the resize engine
    # actually applies (integer output dims), and it is what every coordinate mapping below must
    # use — mapping with the REQUESTED factor reintroduces a systematic sub-pixel bias.
    requested_scale = product_um_per_px / canonical.target_um_per_px
    rescaled_h = max(1, int(round(product_h * requested_scale)))
    rescaled_w = max(1, int(round(product_w * requested_scale)))
    realized_scale_y = rescaled_h / product_h
    realized_scale_x = rescaled_w / product_w

    # Map the crop center onto the rescaled grid under the PIXEL-CENTER convention:
    #     x_out = sf * (x_src + 0.5) - 0.5
    # The naive sf*x is wrong by (sf-1)/2 — about -0.29 px at production scale — and that error is
    # a uniform shift of every snip, so aggregate image metrics stay clean while every embryo moves
    # relative to the historical embedding space. See image_geometry's
    # test_resize_coordinate_convention.py, which pins this across both engines.
    center_x_src = canonical.crop_center_um_xy[0] / canonical.source_um_per_px
    center_y_src = canonical.crop_center_um_xy[1] / canonical.source_um_per_px
    center_x_rescaled = realized_scale_x * (center_x_src + 0.5) - 0.5
    center_y_rescaled = realized_scale_y * (center_y_src + 0.5) - 0.5

    out_h, out_w = canonical.snip_frame_shape_hw
    angle_deg = float(np.rad2deg(canonical.rotation_angle_rad))

    # TODO(remove-legacy-latched-centering): delete this branch, CENTERING_LATCHED, and the
    # `centering` parameter once the stratified acceptance run is accepted and snips are
    # regenerated. Then `continuous` is not a mode, it is simply how placement works.
    if centering == CENTERING_LATCHED:
        # TRANSITIONAL: reproduce the legacy centering REFERENCE as well as its quantizer, so the
        # kernel + render-path migration can be judged without a placement change mixed in.
        #
        # Legacy measured the center on the mask AFTER rescale AND rotation, on the bounds-expanded
        # canvas (extraction.py:103-116). Because mean-of-occupied-index-range is not
        # rotation-equivariant, measuring it on the SOURCE mask instead — as the continuous path
        # correctly does — moves the embryo by up to tens of pixels for a non-axis-aligned subject.
        # Reproducing legacy therefore requires actually rotating the mask here, not just
        # truncating a source-derived center.
        center_x_rescaled, center_y_rescaled = _legacy_rotated_bbox_center(
            canonical=canonical,
            rescaled_shape_hw=(rescaled_h, rescaled_w),
            angle_deg=angle_deg,
        )
    elif centering != CENTERING_CONTINUOUS:
        raise SnipTransformError(
            f"transform_for_product: unknown centering={centering!r}; "
            f"expected {CENTERING_LATCHED!r} or {CENTERING_CONTINUOUS!r}."
        )

    # THE CENTERING REPAIR. Rotate about the embryo center itself and solve the translation that
    # places it at the canvas center: b = t - R·c. The embryo lands centered BY CONSTRUCTION, in
    # continuous coordinates, so the crop step never consults a thresholded mask and the
    # int(np.mean(...)) latch has nothing to latch onto.
    #
    # Canvas center is (n-1)/2 under the pixel-center convention: for n=4 the center of pixels
    # 0..3 is 1.5, not 2.0. This is where odd/even bugs live, so it is stated rather than implied.
    if centering == CENTERING_LATCHED:
        # The latched center was measured on the bounds-expanded ROTATED canvas, so the rotation
        # must be composed the way legacy composed it — rotate about the rescaled canvas center
        # onto the expanded canvas, then translate that measured point to the snip center — rather
        # than rotating about the center itself.
        rotation_mat = _legacy_bounds_expanded_rotation(
            rescaled_shape_hw=(rescaled_h, rescaled_w), angle_deg=angle_deg
        )
        # Legacy's crop used int(output/2) offsets from the measured center, and wrote into a
        # zero canvas — an integer translation, so it is exact to express as one here.
        rotation_mat[0, 2] += int(out_w / 2) - center_x_rescaled
        rotation_mat[1, 2] += int(out_h / 2) - center_y_rescaled
    else:
        rotation_mat = cv2.getRotationMatrix2D((center_x_rescaled, center_y_rescaled), angle_deg, 1.0)
        rotation_mat[0, 2] += (out_w - 1) / 2 - center_x_rescaled
        rotation_mat[1, 2] += (out_h - 1) / 2 - center_y_rescaled

    # The affine writes straight onto the snip canvas, so there is no bounds-expanded intermediate
    # and no separate crop translation. The window IS the canvas.
    crop_x0 = 0
    crop_y0 = 0

    return ResolvedSnipTransform(
        canonical=canonical,
        product_shape_hw=(product_h, product_w),
        product_um_per_px=float(product_um_per_px),
        scale_factor=float(realized_scale_x),
        rescaled_shape_hw=(rescaled_h, rescaled_w),
        rotation_matrix_2x3=tuple(tuple(float(v) for v in row) for row in rotation_mat),
        canvas_center_xy=((out_w - 1) / 2, (out_h - 1) / 2),
        crop_x0_px=crop_x0,
        crop_y0_px=crop_y0,
        crop_x1_px=crop_x0 + out_w,
        crop_y1_px=crop_y0 + out_h,
        crop_x0_um=crop_x0 * canonical.target_um_per_px,
        crop_y0_um=crop_y0 * canonical.target_um_per_px,
        crop_x1_um=(crop_x0 + out_w) * canonical.target_um_per_px,
        crop_y1_um=(crop_y0 + out_h) * canonical.target_um_per_px,
        border_mode=BORDER_MODE,
        border_value=BORDER_VALUE,
        centering=centering,
    )


def apply_transform_to_image(
    image: np.ndarray,
    resolved: ResolvedSnipTransform,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Render an intensity image through a resolved transform. LINEAR interpolation.

    ``dtype`` is OUTPUT-ALLOCATION POLICY, not a geometric parameter: it sets the dtype of the
    destination buffer. Pass the source dtype to preserve quantitative values (the ``no_change``
    case); pass ``np.uint8`` to reproduce the legacy 8-bit rendering path. It does NOT rescale or
    reinterpret values — a source value too large for the requested dtype will wrap, which is the
    caller's decision to make explicitly rather than this function's to paper over.
    """
    return _apply(image, resolved, is_mask=False, dtype=dtype)


def apply_transform_to_mask(
    mask: np.ndarray,
    resolved: ResolvedSnipTransform,
) -> np.ndarray:
    """Render a binary mask through a resolved transform. NEAREST interpolation, uint8 {0,1} out.

    Nearest-neighbor is not a tuning choice — it is what keeps a mask categorical. The legacy path
    resized masks bilinearly into fractional values and then repaired them with scattered ``> 0.5``
    thresholds; this returns a mask that never stopped being binary.
    """
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    return _apply(binary, resolved, is_mask=True, dtype=np.uint8)


def _apply(
    array: np.ndarray,
    resolved: ResolvedSnipTransform,
    *,
    is_mask: bool,
    dtype: np.dtype | type | None,
) -> np.ndarray:
    """Execute the transform through the shared image_geometry chain. The single rendering path.

    Two steps, not three: anti-aliased resize, then an affine that writes straight onto the snip
    canvas. There is no crop, because the translation that centers the embryo is folded into the
    affine — which is precisely what removes the legacy
    rasterize -> threshold -> truncate -> crop loop.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise SnipTransformError(f"_apply: expected a 2-D array; got shape {array.shape}.")
    if tuple(array.shape) != resolved.product_shape_hw:
        raise SnipTransformError(
            f"_apply: array shape {tuple(array.shape)} does not match the resolved transform's "
            f"product grid {resolved.product_shape_hw}. Resolve the transform against the array "
            "you are actually rendering."
        )

    chain = resolved.to_chain()
    rendered = chain.apply_to_mask(array) if is_mask else chain.apply_to_image(array)
    out_dtype = np.dtype(dtype) if dtype is not None else array.dtype
    return rendered.astype(out_dtype)


def assert_transforms_equivalent(
    first: CanonicalSnipTransform,
    second: CanonicalSnipTransform,
    *,
    label: str = "snip transform",
) -> None:
    """Fail loud unless two canonical transforms describe the same physical recipe.

    Exists because "derived once" is implemented as "deterministically re-derived per product job"
    (see the module docstring). This is what makes that claim testable: two products of the same
    embryo-time must derive identical recipes, or their snips are not pixel-registerable.
    """
    if first != second:
        raise SnipTransformError(
            f"{label}: two derivations disagree, so the products they render would not be "
            f"pixel-registerable.\n  first:  {first}\n  second: {second}"
        )
