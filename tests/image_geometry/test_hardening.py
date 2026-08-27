"""Hardening invariants: closed execution kinds, honest interpolation provenance, support masks.

Each of these closes a way the engine could produce plausible, wrong output without erroring.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_geometry import (
    AFFINE,
    CROP_PAD,
    FLIP_X,
    RESIZE,
    GridTransform,
    TransformChain,
    affine_step,
    crop_pad_step,
    resize_step,
    support_mask_for,
)


class TestKindIsClosedAndSeparateFromName:
    """`kind` executes; `name` describes. Conflating them lets a typo change behavior."""

    def test_unknown_kind_raises_at_construction(self):
        with pytest.raises(ValueError, match="unknown kind"):
            GridTransform(
                kind="reszie",  # the typo that used to silently become a generic affine warp
                name="resize",
                affine_2x3=np.eye(2, 3),
                in_shape_yx=(10, 10), out_shape_yx=(5, 5),
                interp="linear", params={},
            )

    def test_name_is_free_form_and_does_not_execute(self):
        # A step named "resize" but of kind affine must WARP, not resample. Under name-dispatch
        # this silently resampled instead.
        step = affine_step(
            affine_2x3=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
            in_shape_yx=(8, 8), out_shape_yx=(8, 8), name="resize",
        )
        assert step.kind == AFFINE
        img = np.zeros((8, 8), dtype=np.uint8)
        img[4, 2] = 100
        out = TransformChain([step]).apply_to_image(img)
        assert out[4, 4] == pytest.approx(100, abs=1)  # translated, not rescaled

    def test_constructors_set_the_right_kinds(self):
        assert resize_step(in_shape_yx=(10, 10), out_shape_yx=(5, 5)).kind == RESIZE
        assert crop_pad_step(in_shape_yx=(10, 10), y0=0, x0=0, out_shape_yx=(5, 5)).kind == CROP_PAD
        assert affine_step(
            affine_2x3=np.eye(2, 3), in_shape_yx=(10, 10), out_shape_yx=(10, 10)
        ).kind == AFFINE

    def test_identity_chain_is_a_valid_kind(self):
        chain = TransformChain.identity(shape_yx=(10, 10), interp="linear")
        assert chain.transforms[0].kind == AFFINE


class TestInterpolationProvenanceIsResolved:
    """A single `interp` field cannot describe a step whose kernel depends on passenger type."""

    def test_downscale_records_area_for_images_nearest_for_masks(self):
        step = resize_step(in_shape_yx=(100, 100), out_shape_yx=(40, 40))
        assert step.params["image_interp"] == "area"
        assert step.params["mask_interp"] == "nearest"

    def test_upscale_records_linear_not_area(self):
        # Nothing to prefilter when adding samples.
        step = resize_step(in_shape_yx=(40, 40), out_shape_yx=(100, 100))
        assert step.params["image_interp"] == "linear"

    def test_disabled_anti_alias_is_recorded_honestly(self):
        step = resize_step(in_shape_yx=(100, 100), out_shape_yx=(40, 40), anti_alias=False)
        assert step.params["image_interp"] == "linear"
        assert step.params["anti_alias"] is False

    def test_crop_pad_records_no_resampling(self):
        step = crop_pad_step(in_shape_yx=(10, 10), y0=1, x0=1, out_shape_yx=(4, 4))
        assert step.params["image_interp"] == "none"
        assert step.params["mask_interp"] == "none"

    def test_affine_records_requested_image_interp_but_mask_is_always_nearest(self):
        # The gap this closes: provenance used to say interp="linear" while apply_to_mask ran
        # nearest — correct behavior, incomplete metadata.
        step = affine_step(
            affine_2x3=np.eye(2, 3), in_shape_yx=(10, 10), out_shape_yx=(10, 10), interp="linear",
        )
        assert step.params["image_interp"] == "linear"
        assert step.params["mask_interp"] == "nearest"

    def test_resize_scale_is_realized_not_requested(self):
        # 64 * 0.4142 rounds to 27, so the realized ratio is 27/64 = 0.4219. Deriving from shapes
        # makes the correct value the only reachable one.
        step = resize_step(in_shape_yx=(64, 64), out_shape_yx=(27, 27))
        assert step.params["scale_x"] == pytest.approx(27 / 64)
        assert step.params["scale_x"] != pytest.approx(0.4142, abs=1e-3)


class TestSupportMask:
    """Padded pixels are OUT OF BOUNDS, not measured zeros."""

    def test_full_coverage_is_all_ones(self):
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=0, x0=0, out_shape_yx=(10, 10))])
        assert support_mask_for(chain, (10, 10)).all()

    def test_padding_is_marked_unsupported(self):
        # Window half outside the source: exactly half the output has no backing pixels.
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=-5, x0=0, out_shape_yx=(10, 10))])
        support = support_mask_for(chain, (10, 10))
        assert not support[:5, :].any(), "padded rows must be unsupported"
        assert support[5:, :].all(), "real rows must be supported"

    def test_rotation_corners_are_unsupported(self):
        # Rotating a square into a square leaves triangular corners with no source behind them.
        rot = cv2.getRotationMatrix2D((32.0, 32.0), 30.0, 1.0)
        chain = TransformChain([
            affine_step(affine_2x3=rot, in_shape_yx=(64, 64), out_shape_yx=(64, 64))
        ])
        support = support_mask_for(chain, (64, 64))
        assert support[0, 0] == 0, "a corner should fall outside the rotated source"
        assert support[32, 32] == 1, "the center must remain supported"

    def test_distinguishes_padding_from_genuinely_dark_pixels(self):
        # THE point. A dark source pixel and a padded pixel are both 0 in the image; only the
        # support mask tells them apart, and averaging padding into a background annulus would
        # bias it toward zero in proportion to how close the embryo sits to the frame edge.
        source = np.zeros((10, 10), dtype=np.uint8)  # genuinely dark everywhere
        chain = TransformChain([crop_pad_step(in_shape_yx=(10, 10), y0=-5, x0=0, out_shape_yx=(10, 10))])
        rendered = chain.apply_to_image(source)
        support = support_mask_for(chain, (10, 10))
        assert not rendered.any(), "image alone cannot distinguish the two cases"
        assert support.any() and not support.all(), "support mask does distinguish them"


class TestFlipConvention:
    """Image flips are a classic off-by-one; execution and coordinate truth must agree."""

    def test_flip_uses_W_minus_1_and_matches_composite(self):
        w = 8
        arr = np.arange(w, dtype=np.float32).reshape(1, w)
        step = GridTransform(
            kind=FLIP_X, name="flip_x",
            affine_2x3=np.array([[-1.0, 0.0, float(w - 1)], [0.0, 1.0, 0.0]]),
            in_shape_yx=(1, w), out_shape_yx=(1, w), interp="nearest",
            params={"affine_convention": "opencv_xy"},
        )
        chain = TransformChain([step])
        executed = chain.apply_to_image(arr)[0]
        composite = chain.composite_affine()

        for x in range(w):
            # Coordinate truth: x -> (W-1) - x, NOT W - x.
            assert (composite @ np.array([float(x), 0.0, 1.0]))[0] == pytest.approx(w - 1 - x)
            # Raster truth agrees with it.
            assert executed[x] == pytest.approx(arr[0][w - 1 - x])
