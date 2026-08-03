"""Tests for the snip transform seam.

The invariants that matter here are cross-product ones: a transform derived from a mask must be
(a) deterministic, (b) portable onto another product's grid, and (c) loud when that port would be
dishonest. Interpolation policy is tested as behavior (masks stay binary), not as a stored string.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.object_extraction.snip_processing.snip_transform import (
    CENTERING_CONTINUOUS,
    CENTERING_LATCHED,
    CanonicalSnipTransform,
    SnipGridSpec,
    SnipTransformError,
    apply_transform_to_image,
    apply_transform_to_mask,
    assert_transforms_equivalent,
    derive_snip_transform,
    transform_for_product,
)


SOURCE_SHAPE = (64, 64)
SOURCE_UM_PER_PX = 2.17
TARGET_UM_PER_PX = 2.17
SNIP_SHAPE = (32, 32)


def _elongated_mask(shape=SOURCE_SHAPE, *, cy=32, cx=32, half_h=12, half_w=5) -> np.ndarray:
    """A clearly non-square blob, so PCA orientation is well defined rather than degenerate."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = 1
    return mask


def _canonical(mask=None, **overrides):
    kwargs = dict(
        source_um_per_px=SOURCE_UM_PER_PX,
        target_um_per_px=TARGET_UM_PER_PX,
        snip_frame_shape_hw=SNIP_SHAPE,
    )
    kwargs.update(overrides)
    return derive_snip_transform(_elongated_mask() if mask is None else mask, **kwargs)


class TestDerive:
    def test_derives_from_mask_without_reading_any_image(self):
        # The whole sibling-DAG argument rests on this: no image argument exists to pass.
        canonical = _canonical()
        assert isinstance(canonical, CanonicalSnipTransform)
        assert canonical.grid.geometry_source_shape_yx == SOURCE_SHAPE

    def test_crop_center_is_physical_not_pixels(self):
        # A center in um is portable across grids; the legacy integer pixel centroid was not.
        canonical = _canonical(_elongated_mask(cy=32, cx=32))
        center_x_um, center_y_um = canonical.crop_center_um_xy
        assert center_x_um == pytest.approx(31.5 * SOURCE_UM_PER_PX, abs=SOURCE_UM_PER_PX)
        assert center_y_um == pytest.approx(31.5 * SOURCE_UM_PER_PX, abs=SOURCE_UM_PER_PX)

    def test_center_tracks_the_embryo(self):
        left = _canonical(_elongated_mask(cx=16))
        right = _canonical(_elongated_mask(cx=48))
        assert left.crop_center_um_xy[0] < right.crop_center_um_xy[0]

    def test_derivation_is_deterministic(self):
        # "Derived once" is implemented as "re-derived per product job", so determinism IS the
        # contract that keeps sibling products registerable.
        assert_transforms_equivalent(_canonical(), _canonical())

    def test_empty_mask_raises(self):
        with pytest.raises(SnipTransformError, match="mask is empty"):
            _canonical(np.zeros(SOURCE_SHAPE, dtype=np.uint8))

    def test_nonpositive_calibration_raises(self):
        with pytest.raises(SnipTransformError, match="must be positive"):
            _canonical(source_um_per_px=0.0)

    def test_field_of_view_is_physical_extent(self):
        canonical = _canonical()
        w_um, h_um = canonical.geometry_source_field_of_view_um_wh
        assert w_um == pytest.approx(64 * SOURCE_UM_PER_PX)
        assert h_um == pytest.approx(64 * SOURCE_UM_PER_PX)


class TestOneRecipeSpeaksEveryProductGrid:
    """THE REASON THE TWO-LEVEL SPLIT EXISTS, tested against a genuinely different grid.

    Every other test in this file resolves against a product that SHARES the mask's calibration --
    BF and RFP both ship 3.2 um/px today, so comparing them proves the recipe is reused but proves
    nothing about portability. The gap between "two products" and "two different GRIDS" is exactly
    where a real bug lived: transform_for_product converted the physical crop center to pixels using
    the GEOMETRY SOURCE's calibration rather than the product's own, which is invisible whenever the
    two agree and puts the center off-canvas by the calibration ratio the moment they do not. A
    4x-downsampled z_stack rendered completely empty, silently.
    """

    @staticmethod
    def _big_mask():
        # A realistic acquisition grid, so the 4x downsample below is a plausible z_stack product
        # rather than a toy that could hide a scale error in the rounding.
        mask = np.zeros((1200, 1600), dtype=np.uint8)
        mask[500:800, 700:850] = 1
        return mask

    def _canonical_for(self, mask):
        return derive_snip_transform(
            mask, source_um_per_px=3.2, target_um_per_px=7.8, snip_frame_shape_hw=(576, 256)
        )

    def test_same_physical_crop_from_grids_4x_apart(self):
        # The same physical scene rasterized at two calibrations: full resolution, and every 4th
        # pixel. One canonical recipe must land both on the same physical region.
        mask = self._big_mask()
        canonical = self._canonical_for(mask)
        rendered = {}
        for name, (raster, shape, um) in {
            "native": (mask, (1200, 1600), 3.2),
            "ds4": (mask[::4, ::4].copy(), (300, 400), 12.8),
        }.items():
            resolved = transform_for_product(
                canonical, product_shape_hw=shape, product_um_per_px=um,
                centering=CENTERING_CONTINUOUS,
            )
            rendered[name] = apply_transform_to_mask(raster, resolved)
            assert rendered[name].sum() > 0, f"{name} rendered an empty snip"

        a, b = rendered["native"] > 0, rendered["ds4"] > 0
        ya, xa = np.where(a)
        yb, xb = np.where(b)
        # One pixel of tolerance: the ds4 source has 4x coarser pixels, so its mask boundary
        # genuinely quantizes differently. Anything larger is a placement error, not rounding.
        assert abs(ya.mean() - yb.mean()) <= 1.0
        assert abs(xa.mean() - xb.mean()) <= 1.0
        iou = np.logical_and(a, b).sum() / np.logical_or(a, b).sum()
        assert iou > 0.95, f"cross-calibration IoU {iou:.4f}; the two products disagree physically"

    @pytest.mark.parametrize("centering", [CENTERING_LATCHED, CENTERING_CONTINUOUS])
    @pytest.mark.parametrize("downsample", [1, 2, 4])
    def test_every_centering_mode_is_portable_across_calibrations(self, centering, downsample):
        # NO CENTERING MODE MAY KEEP ITS OWN COORDINATE DIALECT. The transitional legacy branch
        # reads a pre-resolved center instead of converting crop_center_um_xy, which is a SECOND
        # path through the seam -- and a second path is exactly where the product-calibration bug
        # hid. Both modes must land the same physical crop on every grid.
        mask = self._big_mask()
        canonical = self._canonical_for(mask)
        raster = mask[::downsample, ::downsample].copy()
        shape = (1200 // downsample, 1600 // downsample)
        resolved = transform_for_product(
            canonical, product_shape_hw=shape, product_um_per_px=3.2 * downsample,
            centering=centering,
        )
        snip = apply_transform_to_mask(raster, resolved)

        # THE SILENT-FAILURE GATE. The bug this class was written for produced a fully empty snip
        # with no error anywhere. A standard product must never render nothing.
        assert snip.sum() > 0, (
            f"{centering} at {downsample}x rendered an EMPTY snip -- the transform placed the "
            "embryo entirely off-canvas"
        )

        reference = apply_transform_to_mask(
            mask,
            transform_for_product(
                canonical, product_shape_hw=(1200, 1600), product_um_per_px=3.2,
                centering=centering,
            ),
        )
        a, b = reference > 0, snip > 0
        ya, xa = np.where(a)
        yb, xb = np.where(b)
        # Rasterization tolerance, not byte equality: a 4x-coarser source has already lost spatial
        # precision, so demanding identical masks would be demanding the wrong thing.
        assert abs(ya.mean() - yb.mean()) <= 1.0
        assert abs(xa.mean() - xb.mean()) <= 1.0
        assert np.logical_and(a, b).sum() / np.logical_or(a, b).sum() > 0.95

    def test_the_latched_center_is_not_a_geometry_source_pixel_escape_hatch(self):
        # WHY THE TRANSITIONAL FIELD IS TOLERABLE. `legacy_center_on_target_rescaled_rotated_grid_xy` is expressed on the
        # RESCALED grid, which is product-invariant (a function of physical FOV and target
        # calibration only), NOT in geometry-source pixels. That is what makes the legacy mode
        # portable despite bypassing the um conversion.
        #
        # It cannot simply be folded into crop_center_um_xy: legacy measures its center on the
        # ROTATED canvas while the continuous path measures on the source, and those are different
        # quantities -- for a rotated embryo they differ by hundreds of micrometers, an axis swap
        # rather than a quantizer step. The field goes away with the legacy branch, not before.
        canonical = self._canonical_for(self._big_mask())
        rescaled_w = transform_for_product(
            canonical, product_shape_hw=(1200, 1600), product_um_per_px=3.2,
            centering=CENTERING_LATCHED,
        ).rescaled_shape_hw[1]
        latched_x = canonical.legacy_center_on_target_rescaled_rotated_grid_xy[0]
        assert 0 <= latched_x <= rescaled_w, (
            "the latched center must live on the shared rescaled grid; a value outside it would be "
            "expressed in some product's own pixels and would not survive a calibration change"
        )
        # A geometry-source-pixel value would sit near 774 (1600-wide grid), not ~317 (656-wide).
        assert latched_x < rescaled_w, "latched center looks like a geometry-source pixel value"

    def test_anisotropic_pixels_are_rejected_not_silently_squared(self):
        # THE ABSTRACTION MUST NOT ADVERTISE WHAT IT CANNOT DELIVER. SnipGridSpec stores
        # calibrations per axis, so the canonical layer says "pixels may be anisotropic" -- while
        # the resolver used to take a scalar and treat them as square. That is the same species as
        # the product-calibration bug: an assumption invisible until a product violates it.
        #
        # Rejecting is correct rather than lazy. Compiling a physical rotation onto a rectangular
        # pixel grid needs M = P_dst @ R_physical @ P_src^-1 with separate per-axis scales; feeding
        # the angle to a pixel-space rotation distorts it, and every scalar downstream (FOV check,
        # resize shape, realized scale, rotation center) assumes one scale today.
        canonical = self._canonical_for(self._big_mask())
        with pytest.raises(SnipTransformError, match="anisotropic product pixels"):
            transform_for_product(
                canonical, product_shape_hw=(1200, 1600), product_um_per_px=(3.2, 4.1)
            )

    def test_an_isotropic_pair_is_accepted_and_matches_the_scalar_form(self):
        # The seam takes a (y, x) pair because the type does; an isotropic pair must be exactly
        # equivalent to the scalar it collapses to, or the honest signature would change behavior.
        canonical = self._canonical_for(self._big_mask())
        as_pair = transform_for_product(
            canonical, product_shape_hw=(1200, 1600), product_um_per_px=(3.2, 3.2),
            centering=CENTERING_CONTINUOUS,
        )
        as_scalar = transform_for_product(
            canonical, product_shape_hw=(1200, 1600), product_um_per_px=3.2,
            centering=CENTERING_CONTINUOUS,
        )
        assert as_pair.rotation_matrix_2x3 == as_scalar.rotation_matrix_2x3
        assert as_pair.rescaled_shape_hw == as_scalar.rescaled_shape_hw

    def test_the_matrices_genuinely_differ(self):
        # The complement, and the reason a single affine cannot be "the" transform: a matrix is
        # expressed IN a coordinate system. If these came out equal, the test above would be
        # passing for the trivial reason that nothing was actually re-expressed.
        canonical = self._canonical_for(self._big_mask())
        native = transform_for_product(
            canonical, product_shape_hw=(1200, 1600), product_um_per_px=3.2,
            centering=CENTERING_CONTINUOUS,
        )
        ds4 = transform_for_product(
            canonical, product_shape_hw=(300, 400), product_um_per_px=12.8,
            centering=CENTERING_CONTINUOUS,
        )
        assert native.scale_factor != pytest.approx(ds4.scale_factor)
        # ... while both resample onto the SAME shared grid, which is what makes them registerable.
        assert native.rescaled_shape_hw == ds4.rescaled_shape_hw

    def test_the_center_uses_the_products_own_calibration(self):
        # The bug, pinned directly. The physical center converted into the product's pixels must
        # scale with THAT product's calibration; using the derivation grid's would put a
        # 4x-coarser product's center 4x too far out.
        canonical = self._canonical_for(self._big_mask())
        center_um_x = canonical.crop_center_um_xy[0]
        ds4 = transform_for_product(
            canonical, product_shape_hw=(300, 400), product_um_per_px=12.8,
            centering=CENTERING_CONTINUOUS,
        )
        expected_src_px = center_um_x / 12.8
        realized = ds4.rescaled_shape_hw[1] / 400
        expected_rescaled = realized * (expected_src_px + 0.5) - 0.5
        # The center must sit INSIDE the shared grid; under the bug it was ~4x beyond it.
        assert 0 <= expected_rescaled <= ds4.rescaled_shape_hw[1]

        # Read it back out of the affine rather than a field: the continuous branch solves
        # b = t - R*c, so with no rotation the translation is (canvas_center - embryo_center) and
        # the embryo center is recoverable. That is the number the bug corrupted.
        out_w = ds4.output_shape_hw[1]
        recovered = (out_w - 1) / 2 - ds4.rotation_matrix_2x3[0][2]
        assert recovered == pytest.approx(expected_rescaled, abs=1e-6)


class TestTransformForProduct:
    def test_same_grid_resolves(self):
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
        )
        assert resolved.product_shape_hw == SOURCE_SHAPE
        assert resolved.output_shape_hw == SNIP_SHAPE

    def test_downsampled_product_resolves_to_same_physical_recipe(self):
        # The BF__z_stack downsample_factor: 4 case — half the pixels, twice the um/px, SAME
        # physical field of view. This must resolve, not raise.
        canonical = _canonical()
        resolved = transform_for_product(
            canonical, product_shape_hw=(32, 32), product_um_per_px=SOURCE_UM_PER_PX * 2,
        )
        assert resolved.canonical == canonical
        assert resolved.product_um_per_px == pytest.approx(SOURCE_UM_PER_PX * 2)

    def test_different_field_of_view_fails_loud(self):
        # Same pixel count, different calibration -> a genuinely different region. Silently
        # proceeding here is what produces a plausible, misregistered snip.
        with pytest.raises(SnipTransformError, match="physical field of view disagrees"):
            transform_for_product(
                _canonical(), product_shape_hw=SOURCE_SHAPE,
                product_um_per_px=SOURCE_UM_PER_PX * 4,
            )

    def test_transposed_product_fails_loud(self):
        # A per-product `orientation` can np.rot90 one product and not another, transposing H/W.
        # Isolate the ASPECT check from the FOV check: a plain transpose leaves total physical
        # extent per axis unequal, so to reach the aspect branch the product must keep each axis's
        # physical extent within tolerance while swapping the ratio. A 64x32 grid viewed as 32x64
        # at the same um/px trips FOV first (correctly), so scale the calibration to hold the
        # width extent and let the ratio be what disagrees.
        canonical = _canonical(_elongated_mask((64, 32), cx=16), source_um_per_px=SOURCE_UM_PER_PX)
        # derivation FOV: w = 32*2.17 = 69.4um, h = 64*2.17 = 138.9um
        # product 64x64 at 1.085 um/px: w = 69.4um, h = 69.4um -> width matches, height does not.
        with pytest.raises(SnipTransformError, match="field of view disagrees on height"):
            transform_for_product(
                canonical, product_shape_hw=(64, 64), product_um_per_px=SOURCE_UM_PER_PX / 2,
            )

    def test_aspect_ratio_mismatch_within_matching_fov_fails_loud(self):
        # The aspect guard must be independently reachable, or it is dead code dressed as a check.
        # It is: with the FOV tolerance at 2%, integer pixel counts admit up to 3.7% aspect drift
        # while BOTH axes' physical extents still pass. This case is that witness.
        #
        #   derivation: 64h x 32w @ 2.0 um/px  -> FOV 64um x 128um, aspect (w/h) = 0.5000
        #   product:    27h x 14w @ 4.65 um/px -> FOV 65.1um x 125.6um  (both within 2%)
        #                                         aspect = 0.5185       (3.7% off -> raises)
        #
        # Physically: one axis was resampled independently of the other, so a resize would stretch
        # the embryo. The FOV check alone cannot see this.
        canonical = CanonicalSnipTransform(
            grid=SnipGridSpec(
                geometry_source_shape_yx=(64, 32),
                geometry_source_um_per_px_yx=(2.0, 2.0),
                default_output_um_per_px_yx=(2.0, 2.0),
                default_output_shape_yx=SNIP_SHAPE,
            ),
            rotation_angle_rad=0.0,
            crop_center_um_xy=(32.0, 64.0),
        )
        with pytest.raises(SnipTransformError, match="aspect ratio disagrees"):
            transform_for_product(
                canonical, product_shape_hw=(27, 14), product_um_per_px=4.65,
            )

    def test_nonpositive_product_calibration_raises(self):
        with pytest.raises(SnipTransformError, match="must be positive"):
            transform_for_product(
                _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=0.0,
            )

    def test_crop_window_carried_in_both_coordinate_systems(self):
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
        )
        # Product pixels alone cannot express one recipe across calibrations; physical alone is not
        # auditable against the written file. Both are carried, and they must agree.
        assert resolved.crop_x1_px - resolved.crop_x0_px == SNIP_SHAPE[1]
        assert resolved.crop_y1_px - resolved.crop_y0_px == SNIP_SHAPE[0]
        assert resolved.crop_x0_um == pytest.approx(resolved.crop_x0_px * TARGET_UM_PER_PX)
        assert resolved.crop_y1_um == pytest.approx(resolved.crop_y1_px * TARGET_UM_PER_PX)


class TestApply:
    def _resolved(self, **overrides):
        kwargs = dict(product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX)
        kwargs.update(overrides)
        return transform_for_product(_canonical(), **kwargs)

    def test_image_output_has_snip_frame_shape(self):
        image = np.full(SOURCE_SHAPE, 100, dtype=np.uint8)
        out = apply_transform_to_image(image, self._resolved())
        assert out.shape == SNIP_SHAPE

    def test_mask_stays_binary(self):
        # THE interpolation regression. The legacy path bilinearly resized binary masks into
        # fractional values, then repaired them with scattered `> 0.5` thresholds.
        out = apply_transform_to_mask(_elongated_mask(), self._resolved())
        assert set(np.unique(out)).issubset({0, 1}), f"mask went fractional: {np.unique(out)}"
        assert out.dtype == np.uint8

    def test_image_dtype_is_output_allocation_policy(self):
        image = np.full(SOURCE_SHAPE, 300, dtype=np.uint16)
        preserved = apply_transform_to_image(image, self._resolved(), dtype=np.uint16)
        assert preserved.dtype == np.uint16
        assert preserved.max() > 255, "uint16 values above 255 must survive the transform"

    def test_uint16_truncates_when_uint8_requested(self):
        # H1 made explicit: requesting uint8 for >255 data wraps. That is the caller's decision,
        # stated here so the behavior is pinned rather than discovered later in production.
        image = np.full(SOURCE_SHAPE, 300, dtype=np.uint16)
        out = apply_transform_to_image(image, self._resolved(), dtype=np.uint8)
        assert out.dtype == np.uint8
        assert out.max() <= 255

    def test_mismatched_array_shape_fails_loud(self):
        wrong = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(SnipTransformError, match="does not match the resolved transform"):
            apply_transform_to_image(wrong, self._resolved())

    def test_image_and_mask_land_on_the_same_grid(self):
        # Sibling registerability, reduced to one assertion: the same resolved transform renders
        # both channels' pixels and the mask onto an identical output grid.
        resolved = self._resolved()
        image_out = apply_transform_to_image(np.full(SOURCE_SHAPE, 100, np.uint8), resolved)
        mask_out = apply_transform_to_mask(_elongated_mask(), resolved)
        assert image_out.shape == mask_out.shape == SNIP_SHAPE

    def test_two_products_of_the_same_embryo_share_one_canonical_recipe(self):
        # The sibling-DAG property end to end: BF and RFP resolve from ONE canonical transform, so
        # their snips are registerable even though each reads its own calibration.
        canonical = _canonical()
        bf = transform_for_product(
            canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX)
        rfp = transform_for_product(
            canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX)
        assert bf.canonical == rfp.canonical
        assert (bf.crop_x0_px, bf.crop_y0_px) == (rfp.crop_x0_px, rfp.crop_y0_px)
        assert bf.rescaled_shape_hw == rfp.rescaled_shape_hw
        # The affine is what places the embryo, so identical matrices IS registerability.
        assert np.allclose(bf.rotation_matrix_2x3, rfp.rotation_matrix_2x3)


class TestCenteringModesAreSeparable:
    """The two pixel-changing commits must be independently attributable.

    The kernel migration lands with ``latched`` centering (legacy placement, new resample kernel);
    the centering repair then flips to ``continuous``. Each commit varies exactly one thing, so when
    a snip moves we can say which change moved it. These tests pin that the modes are genuinely
    distinct and that the default has not silently advanced.
    """

    def test_default_is_still_latched(self):
        # Guards the commit boundary: flipping the default IS the centering commit, and must not
        # happen by accident inside the kernel commit.
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
        )
        assert resolved.centering == CENTERING_LATCHED

    def test_latched_placement_is_integral(self):
        # The legacy quantizer, reproduced: int() truncation on the rescaled grid.
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        # With an integral center and a half-integer canvas center, the translation carries the .5
        # from the canvas side only.
        tx = resolved.rotation_matrix_2x3[0][2]
        assert abs(tx - round(tx)) in (pytest.approx(0.0, abs=1e-9), pytest.approx(0.5, abs=1e-9))

    def test_modes_differ_when_the_center_is_fractional(self):
        # An even-width mask has a half-integer bbox center, which latched truncation destroys.
        mask = _elongated_mask(cx=32, half_w=5)  # cols 27..36 -> center 31.5
        latched = transform_for_product(
            _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        continuous = transform_for_product(
            _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_CONTINUOUS,
        )
        assert latched.rotation_matrix_2x3 != continuous.rotation_matrix_2x3

    def test_latched_holds_the_centering_REFERENCE_not_only_the_quantizer(self):
        """`legacy_latched` must reproduce WHICH MASK the center is measured on, not just int().

        Legacy measured the center on the rescaled, ROTATED mask; the continuous path measures it on
        the source mask BEFORE rotation. mean-of-occupied-index-range is not rotation-equivariant, so
        for a rotated embryo those are different points — tens of pixels apart on real data.

        A diagonal mask makes the difference observable: PCA gives it a non-zero angle, so if latched
        were still deriving its center from the source mask (merely truncated), its placement would
        track the continuous one. It must not.
        """
        mask = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
        for i in range(-6, 7):  # a thick diagonal bar -> well-defined ~45 degree orientation
            np.fill_diagonal(mask[max(0, 10 + i):, max(0, 10 - i):], 1)
        canonical = _canonical(mask)
        assert abs(np.rad2deg(canonical.rotation_angle_rad)) > 5, "fixture must actually be rotated"

        latched = transform_for_product(
            canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_LATCHED,
        )
        continuous = transform_for_product(
            canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
            centering=CENTERING_CONTINUOUS,
        )
        # The two references disagree, so the composed translations must differ by more than the
        # sub-pixel amount an int() truncation alone could ever produce.
        dx = abs(latched.rotation_matrix_2x3[0][2] - continuous.rotation_matrix_2x3[0][2])
        dy = abs(latched.rotation_matrix_2x3[1][2] - continuous.rotation_matrix_2x3[1][2])
        assert max(dx, dy) > 1.0, (
            f"latched and continuous translations differ by only ({dx:.3f}, {dy:.3f}) — latched is "
            "not measuring its center on the ROTATED mask, so it is reproducing legacy's quantizer "
            "without legacy's reference and the kernel commit is not isolated"
        )

    def test_latched_without_a_resolved_center_fails_loud(self):
        # The legacy centering reference is resolved at DERIVATION time and rides along as a scalar
        # pair. A hand-built transform that lacks it must not silently fall back to the source-grid
        # center -- that would render legacy-mode pixels using continuous-mode placement, which is
        # exactly the confound the two modes exist to keep apart.
        import dataclasses

        bare = dataclasses.replace(_canonical(), legacy_center_on_target_rescaled_rotated_grid_xy=None)
        with pytest.raises(SnipTransformError, match="resolved legacy centering reference"):
            transform_for_product(
                bare, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_LATCHED,
            )

    def test_canonical_transform_carries_no_raster(self):
        # THE GATE REQUIREMENT: snip_geometry persists this object and render jobs deserialize it,
        # so it must contain no array. `source_mask` used to live here as field(compare=False) --
        # an input inside the output type, annotated "ignore when comparing", which is the tell it
        # did not belong. Its one consumer (the legacy centering reference) is now resolved during
        # derivation into a scalar pair.
        import dataclasses

        for f in dataclasses.fields(CanonicalSnipTransform):
            value = getattr(_canonical(), f.name)
            assert not isinstance(value, np.ndarray), (
                f"CanonicalSnipTransform.{f.name} holds an ndarray; the canonical transform must "
                "serialize without carrying a raster or it cannot cross the geometry gate"
            )

    def test_two_derivations_of_one_embryo_time_compare_equal(self):
        # Sibling registerability: the recipe is a pure function of the mask, so two derivations
        # must be indistinguishable. Under the gate this is a determinism check rather than the
        # thing guaranteeing registration -- but a failure here means the gate is storing a value
        # that depends on something other than its inputs.
        assert_transforms_equivalent(_canonical(), _canonical())

    def test_legacy_mode_is_named_so_call_sites_cannot_miss_it(self):
        # A flag reading as a neutral option is how dual behavior becomes permanent and quietly
        # splits a dataset. The word "legacy" must appear wherever the mode is written down.
        assert "legacy" in CENTERING_LATCHED

    def test_no_centering_mode_is_config_selectable_during_migration(self):
        # The trapdoor guard. Both modes are chosen in code for the comparison; NEITHER is reachable
        # from a run config. When this set becomes {continuous}, the legacy branch must already be
        # deleted — see TODO(remove-legacy-latched-centering).
        from data_pipeline.object_extraction.snip_processing.snip_transform import (
            CONFIGURABLE_CENTERING_MODES,
            SUPPORTED_CENTERING_MODES,
        )

        assert CONFIGURABLE_CENTERING_MODES == frozenset(), (
            "a centering mode became config-selectable; if this is the activation commit, delete "
            "the legacy branch in the same change rather than leaving both reachable"
        )
        assert CENTERING_LATCHED not in CONFIGURABLE_CENTERING_MODES
        assert SUPPORTED_CENTERING_MODES == {CENTERING_LATCHED, CENTERING_CONTINUOUS}

    def test_unknown_centering_fails_loud(self):
        with pytest.raises(SnipTransformError, match="unknown centering"):
            transform_for_product(
                _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering="sensible-sounding-nonsense",
            )

    def test_latched_mode_reproduces_the_discrete_jump(self):
        """The defect itself, pinned: latched placement QUANTIZES while continuous does not.

        The sweep must move the bbox CENTER, not merely the width — a symmetric mask grown about a
        fixed cx keeps its center at 31.5 for every width, so a width sweep varies nothing and would
        pass or fail for reasons unrelated to the latch.

        Widening by one column at a time (odd/even extent) walks the center in half-pixel steps.
        Latched truncates those to whole pixels, so its placement takes FEWER distinct values than
        continuous — that collapse IS the defect.
        """
        latched, continuous = [], []
        for x1 in range(33, 41):  # cols 27..x1 -> center walks 30.0, 30.5, 31.0, ...
            mask = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
            mask[20:44, 27:x1] = 1
            canonical = _canonical(mask)
            for mode, sink in ((CENTERING_LATCHED, latched), (CENTERING_CONTINUOUS, continuous)):
                resolved = transform_for_product(
                    canonical, product_shape_hw=SOURCE_SHAPE,
                    product_um_per_px=SOURCE_UM_PER_PX, centering=mode,
                )
                sink.append(round(resolved.rotation_matrix_2x3[0][2], 9))

        assert len(set(latched)) < len(set(continuous)), (
            f"latched placement took {len(set(latched))} distinct values and continuous took "
            f"{len(set(continuous))} — latched must collapse sub-pixel centers onto fewer "
            "positions, or it is no longer modeling the legacy int() quantizer"
        )


class TestCenteringIsContinuous:
    """The latch repair. Legacy did int(np.mean(...)) over a thresholded, twice-resampled mask, so a
    sub-grey-level input change could translate the whole canvas by a pixel (~10% of real snips)."""

    def _center_of_mass(self, arr):
        total = arr.sum()
        ys, xs = np.indices(arr.shape)
        return ((ys * arr).sum() / total, (xs * arr).sum() / total)

    def test_subpixel_mask_perturbation_does_not_jump_the_canvas(self):
        # THE regression for the latch. Nudge one boundary pixel — under legacy this could tip the
        # integer centroid and shift everything by a whole pixel. The response must be smooth.
        base = _elongated_mask(cy=32, cx=32)
        nudged = base.copy()
        nudged[32 - 12, 32 - 5] = 0  # flip a single boundary pixel

        shifts = []
        for mask in (base, nudged):
            resolved = transform_for_product(
                _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
            )
            shifts.append((resolved.rotation_matrix_2x3[0][2], resolved.rotation_matrix_2x3[1][2]))

        dx = abs(shifts[0][0] - shifts[1][0])
        dy = abs(shifts[0][1] - shifts[1][1])
        assert dx < 1.0 and dy < 1.0, (
            f"one boundary pixel moved the placement by ({dx:.3f}, {dy:.3f}) px — "
            "a whole-pixel jump means a discrete latch survived"
        )

    def test_source_center_maps_to_the_requested_canvas_center(self):
        """THE invariant. Not 'the translation is fractional' — that is not the property.

        A correct continuous computation can legitimately produce an INTEGER translation when the
        fractional halves cancel: bbox center 27.5, canvas center 15.5, translation exactly -12.0.
        Asserting non-integrality would fail on correct arithmetic. What must hold is that the
        continuous source center lands on the requested continuous output center.
        """
        for cx in (28, 31, 33, 37):
            mask = _elongated_mask(cx=cx)
            canonical = _canonical(mask)
            resolved = transform_for_product(
                canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
            )
            # Source bbox center -> rescaled grid, under the pixel-center convention.
            sx = resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]
            sy = resolved.rescaled_shape_hw[0] / resolved.product_shape_hw[0]
            cx_src = canonical.crop_center_um_xy[0] / canonical.grid.geometry_source_um_per_px_yx[1]
            cy_src = canonical.crop_center_um_xy[1] / canonical.grid.geometry_source_um_per_px_yx[0]
            point = np.array([sx * (cx_src + 0.5) - 0.5, sy * (cy_src + 0.5) - 0.5, 1.0])

            mapped = np.asarray(resolved.rotation_matrix_2x3) @ point
            assert mapped[0] == pytest.approx(resolved.canvas_center_xy[0], abs=1e-6)
            assert mapped[1] == pytest.approx(resolved.canvas_center_xy[1], abs=1e-6)

    def test_non_canceling_fractional_case_is_not_truncated(self):
        """A case where the halves do NOT cancel, so truncation would be visible.

        An ODD-width mask gives an integer bbox center; against a half-integer canvas center the
        difference carries a .5 that a surviving int() would destroy.
        """
        mask = _elongated_mask(cx=32, half_w=5)
        odd = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
        odd[20:44, 30:35] = 1  # cols 30..34 -> bbox center exactly 32.0
        for m in (mask, odd):
            canonical = _canonical(m)
            resolved = transform_for_product(
                canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
            )
            sx = resolved.rescaled_shape_hw[1] / resolved.product_shape_hw[1]
            cx_src = canonical.crop_center_um_xy[0] / canonical.grid.geometry_source_um_per_px_yx[1]
            mapped_x = (
                np.asarray(resolved.rotation_matrix_2x3)
                @ np.array([sx * (cx_src + 0.5) - 0.5, 0.0, 1.0])
            )[0]
            assert mapped_x == pytest.approx(resolved.canvas_center_xy[0], abs=1e-6)

    def test_placement_responds_continuously_to_subpixel_input(self):
        """Sweep across an integer boundary; the response must be smooth, not stepped.

        The legacy int(np.mean(...)) path jumps a whole pixel somewhere in this sweep. A continuous
        path produces monotone, small increments.
        """
        translations = []
        for half_w in (5, 6, 7, 8):  # walks the bbox center by half-pixel steps
            canonical = _canonical(_elongated_mask(cx=32, half_w=half_w))
            resolved = transform_for_product(
                canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
            )
            translations.append(resolved.rotation_matrix_2x3[0][2])
        steps = np.diff(translations)
        assert np.all(np.abs(steps) <= 1.0 + 1e-9), (
            f"placement jumped by {steps} — a step >1px is the signature of a discrete latch"
        )

    def test_embryo_lands_at_the_canvas_center(self):
        # The point of folding translation into the affine: centering by construction.
        mask = _elongated_mask(cy=32, cx=32)
        resolved = transform_for_product(
            _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        rendered = apply_transform_to_mask(mask, resolved)
        cy, cx = self._center_of_mass(rendered.astype(float))
        expect_y, expect_x = (SNIP_SHAPE[0] - 1) / 2, (SNIP_SHAPE[1] - 1) / 2
        assert abs(cy - expect_y) < 1.5 and abs(cx - expect_x) < 1.5, (
            f"embryo landed at ({cy:.2f}, {cx:.2f}), expected ~({expect_y}, {expect_x})"
        )

    @pytest.mark.parametrize("snip_shape", [(32, 32), (33, 33), (32, 33), (576, 256)])
    def test_odd_and_even_output_dimensions(self, snip_shape):
        # (n-1)/2 vs n/2 is where odd/even bugs live, so both parities are exercised.
        mask = _elongated_mask(cy=32, cx=32)
        canonical = derive_snip_transform(
            mask, source_um_per_px=SOURCE_UM_PER_PX, target_um_per_px=TARGET_UM_PER_PX,
            snip_frame_shape_hw=snip_shape,
        )
        resolved = transform_for_product(
            canonical, product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        assert resolved.canvas_center_xy == ((snip_shape[1] - 1) / 2, (snip_shape[0] - 1) / 2)
        rendered = apply_transform_to_mask(mask, resolved)
        assert rendered.shape == snip_shape
        if rendered.any():
            cy, cx = self._center_of_mass(rendered.astype(float))
            assert abs(cy - (snip_shape[0] - 1) / 2) < 2.0
            assert abs(cx - (snip_shape[1] - 1) / 2) < 2.0

    def test_embryo_at_source_boundary_is_zero_padded_not_an_error(self):
        # Clipped by the frame edge: half_h=12 from cy=13 reaches row 1, half_w=5 from cx=6 reaches
        # column 1. Small offsets with the default half-extents would slice negatively and produce
        # an empty mask, which is a different case (covered by test_empty_mask_raises).
        mask = _elongated_mask(cy=13, cx=6)
        resolved = transform_for_product(
            _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        rendered = apply_transform_to_mask(mask, resolved)
        assert rendered.shape == SNIP_SHAPE
        assert set(np.unique(rendered)).issubset({0, 1})

    def test_image_and_mask_stay_aligned(self):
        # One transform renders both, so the mask must land where the bright pixels do.
        mask = _elongated_mask(cy=32, cx=32)
        image = (mask * 200).astype(np.uint8)
        resolved = transform_for_product(
            _canonical(mask), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        img_out = apply_transform_to_image(image, resolved)
        mask_out = apply_transform_to_mask(mask, resolved)
        img_c = self._center_of_mass(img_out.astype(float))
        mask_c = self._center_of_mass(mask_out.astype(float))
        assert abs(img_c[0] - mask_c[0]) < 1.0 and abs(img_c[1] - mask_c[1]) < 1.0

    def test_border_semantics_are_recorded(self):
        # The crop OPERATION disappears under continuous centering; its SEMANTICS must not.
        # augment_snip blends against a synthetic noise background, so what sits outside the mask
        # is a real input to the rendered snip.
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        assert resolved.border_mode == "constant"
        assert resolved.border_value == 0.0

    def test_chain_exposes_raster_truth(self):
        resolved = transform_for_product(
            _canonical(), product_shape_hw=SOURCE_SHAPE, product_um_per_px=SOURCE_UM_PER_PX,
                centering=CENTERING_CONTINUOUS,
        )
        chain = resolved.to_chain()
        assert [t.name for t in chain.transforms] == ["resize", "rotate_center"]
        assert chain.transforms[0].params["anti_alias"] is True


class TestEquivalence:
    def test_equal_transforms_pass(self):
        assert_transforms_equivalent(_canonical(), _canonical())

    def test_differing_transforms_fail_loud(self):
        with pytest.raises(SnipTransformError, match="two derivations disagree"):
            assert_transforms_equivalent(
                _canonical(_elongated_mask(cx=16)), _canonical(_elongated_mask(cx=48)),
            )
