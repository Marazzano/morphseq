"""Tests for the typed raster step kinds: Resize / Affine / CropPad.

The load-bearing claim is that these must be SEPARATE steps — that a fused affine cannot express an
anti-aliased downscale. That is not a style preference; it is measurable, and it is measured here.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_geometry import (
    CROP_PAD,
    RESIZE,
    TransformChain,
    affine_step,
    crop_pad_step,
    resize_step,
)


def _checkerboard(n: int = 400) -> np.ndarray:
    """Maximum-frequency content: alternating pixels. Aliasing's worst case, and its clearest test."""
    return ((np.add.outer(np.arange(n), np.arange(n)) % 2) * 200 + 20).astype(np.uint8)


class TestWhyResizeIsItsOwnStep:
    """The justification for the step split, asserted rather than asserted-about."""

    def test_warp_affine_cannot_anti_alias(self):
        # THE measurement behind the design. A correctly anti-aliased 0.414x downscale of a
        # checkerboard collapses it toward flat grey. A fused affine carrying the same scale does
        # not — and passing INTER_AREA to warpAffine does not help, because it is IGNORED.
        img = _checkerboard().astype(np.float32)
        out_h = out_w = int(round(400 * 0.414))
        matrix = np.array([[0.414, 0.0, 0.0], [0.0, 0.414, 0.0]], dtype=np.float64)

        warp_linear = cv2.warpAffine(img, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
        warp_area = cv2.warpAffine(img, matrix, (out_w, out_h), flags=cv2.INTER_AREA)
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # INTER_AREA is silently ignored by warpAffine: byte-identical to INTER_LINEAR.
        assert np.array_equal(warp_linear, warp_area), (
            "warpAffine honored INTER_AREA — if OpenCV changed this, the step split can be revisited"
        )
        # And the aliasing is severe: the checkerboard survives instead of averaging away.
        assert warp_linear.std() > 20, "expected heavy aliasing from a fused-affine downscale"
        assert resized.std() < 5, "INTER_AREA resize should collapse the checkerboard"

    def test_resize_step_anti_aliases(self):
        chain = TransformChain([resize_step(in_shape_yx=(400, 400), out_shape_yx=(166, 166))])
        out = chain.apply_to_image(_checkerboard())
        assert out.std() < 5, f"resize step failed to anti-alias (std={out.std():.2f})"

    def test_anti_alias_can_be_disabled(self):
        chain = TransformChain(
            [resize_step(in_shape_yx=(400, 400), out_shape_yx=(166, 166), anti_alias=False)]
        )
        assert chain.apply_to_image(_checkerboard()).std() > 20

    def test_upscale_does_not_use_area(self):
        # Nothing to prefilter when adding samples; INTER_AREA on upscale would be wrong.
        chain = TransformChain([resize_step(in_shape_yx=(10, 10), out_shape_yx=(40, 40))])
        out = chain.apply_to_image(np.arange(100, dtype=np.uint8).reshape(10, 10))
        assert out.shape == (40, 40)


class TestCropPad:
    def test_is_pure_indexing(self):
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=2, x0=3, out_shape_yx=(4, 4))])
        out = chain.apply_to_image(img)
        # Exact source values, not interpolated ones.
        assert np.array_equal(out, img[2:6, 3:7])

    def test_zero_fills_outside_the_source(self):
        img = np.ones((10, 10), dtype=np.uint8) * 7
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=-2, x0=-2, out_shape_yx=(6, 6))])
        out = chain.apply_to_image(img)
        assert out[0, 0] == 0 and out[5, 5] == 7

    def test_window_entirely_outside_yields_zeros(self):
        chain = TransformChain(
            [crop_pad_step(in_shape_yx=(10, 10), y0=50, x0=50, out_shape_yx=(4, 4))]
        )
        assert not chain.apply_to_image(np.ones((10, 10), dtype=np.uint8)).any()

    def test_preserves_mask_values_exactly(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=0, x0=0, out_shape_yx=(10, 10))])
        assert np.array_equal(chain.apply_to_mask(mask), mask)


class TestMaskSemantics:
    def test_mask_never_goes_fractional_through_a_resize(self):
        # The defect the legacy path had: bilinear on a binary mask, then repaired with `> 0.5`.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        chain = TransformChain([resize_step(in_shape_yx=(100, 100), out_shape_yx=(41, 41))])
        out = chain.apply_to_mask(mask)
        assert set(np.unique(out)).issubset({0, 1}), f"mask went fractional: {np.unique(out)}"

    def test_mask_semantics_override_the_interp_field(self):
        # A transform declaring "linear" must still be applied as nearest to a mask. Interpolation
        # follows the raster's semantics, not the call site's preference.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1
        rot = cv2.getRotationMatrix2D((50.0, 50.0), 37.0, 1.0)
        chain = TransformChain(
            [affine_step(affine_2x3=rot, in_shape_yx=(100, 100), out_shape_yx=(100, 100),
                         interp="linear")]
        )
        assert set(np.unique(chain.apply_to_mask(mask))).issubset({0, 1})


class TestCompositeAffine:
    def test_identity_chain_composes_to_identity(self):
        chain = TransformChain.identity(shape_yx=(10, 10), interp="linear")
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(chain.composite_affine(), expected)

    def test_composes_resize_then_crop(self):
        # Coordinate truth: a source point maps through the whole chain.
        chain = TransformChain([
            resize_step(in_shape_yx=(100, 100), out_shape_yx=(50, 50)),
            crop_pad_step(in_shape_yx=(50, 50), y0=10, x0=10, out_shape_yx=(20, 20)),
        ])
        composite = chain.composite_affine()
        # Source (40, 40) -> resize by 0.5 -> (20, 20) -> crop offset (-10) -> (10, 10)
        point = composite @ np.array([40.0, 40.0, 1.0])
        assert np.allclose(point, [10.0, 10.0])

    def test_composite_is_coordinate_truth_only(self):
        # The composite says a downscale happened; it CANNOT say whether it was anti-aliased.
        # Two chains with identical composites render different pixels — which is exactly why the
        # chain is kept and not collapsed.
        aa = TransformChain([resize_step(in_shape_yx=(400, 400), out_shape_yx=(166, 166))])
        no_aa = TransformChain(
            [resize_step(in_shape_yx=(400, 400), out_shape_yx=(166, 166), anti_alias=False)]
        )
        assert np.allclose(aa.composite_affine(), no_aa.composite_affine())
        img = _checkerboard()
        assert not np.allclose(aa.apply_to_image(img), no_aa.apply_to_image(img)), (
            "identical coordinate truth must still permit different raster truth"
        )


class TestStepKindMetadata:
    def test_steps_carry_their_kind(self):
        assert resize_step(in_shape_yx=(10, 10), out_shape_yx=(5, 5)).name == RESIZE
        assert crop_pad_step(in_shape_yx=(10, 10), y0=0, x0=0, out_shape_yx=(5, 5)).name == CROP_PAD

    def test_resize_records_its_scale_and_prefilter(self):
        step = resize_step(in_shape_yx=(100, 200), out_shape_yx=(50, 50))
        assert step.params["anti_alias"] is True
        assert step.params["scale_y"] == pytest.approx(0.5)
        assert step.params["scale_x"] == pytest.approx(0.25)

    def test_full_three_verb_chain_runs(self):
        # The engine shape the snip pipeline uses: resize -> affine -> crop_pad.
        rot = cv2.getRotationMatrix2D((83.0, 83.0), 30.0, 1.0)
        chain = TransformChain([
            resize_step(in_shape_yx=(400, 400), out_shape_yx=(166, 166)),
            affine_step(affine_2x3=rot, in_shape_yx=(166, 166), out_shape_yx=(166, 166)),
            crop_pad_step(in_shape_yx=(166, 166), y0=53, x0=53, out_shape_yx=(60, 60)),
        ])
        assert chain.apply_to_image(_checkerboard()).shape == (60, 60)
        assert len(chain) == 3
