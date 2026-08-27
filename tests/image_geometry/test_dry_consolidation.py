"""Pins for the primitives that absorbed duplicated raster/geometry code.

Each test here exists because a copy was REMOVED somewhere else and the behavior it encoded now has
exactly one home. The point is not coverage for its own sake — it is that the consolidation was a
pure refactor, so these assert the pre-existing behavior rather than a new contract.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_geometry import (
    BoxYX,
    bounds_expanded_rotation_matrix,
    expanded_rotation_bounds_wh,
    resize_interpolation_flags,
)


def _legacy_rotate(mat, angle):
    """The block that used to be inlined in four places, verbatim."""
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


class TestBoundsExpandedRotation:
    @pytest.mark.parametrize("shape", [(64, 64), (37, 91), (128, 40), (300, 200)])
    @pytest.mark.parametrize("angle", [0.0, 0.5, 30.0, 45.0, -45.0, 89.9, 90.0, 180.0, -123.7])
    def test_matches_the_legacy_inline_block_exactly(self, shape, angle):
        """Byte-for-byte, not approximately: snip placement is pinned to these dimensions."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=shape).astype(np.uint8)

        mat, bounds_wh = bounds_expanded_rotation_matrix(shape_hw=shape, angle_deg=angle)
        got = cv2.warpAffine(img, mat, bounds_wh)

        assert np.array_equal(got, _legacy_rotate(img, angle))

    def test_bounds_truncate_rather_than_ceil(self):
        """The ``int()`` truncation is contract, not an oversight — see the module docstring.

        A 1x1 canvas rotated 45 degrees has an exact bound of sqrt(2) ~ 1.41; truncation gives 1.
        Pinning this stops a well-meaning ``ceil`` from silently moving every snip.
        """
        assert expanded_rotation_bounds_wh(
            shape_hw=(1, 1), abs_cos=np.sqrt(0.5), abs_sin=np.sqrt(0.5)
        ) == (1, 1)

    def test_bounds_are_width_height_for_opencv_dsize(self):
        """A tall canvas rotated 90 degrees becomes wide. Guards the (W,H)/(H,W) swap."""
        _mat, (bound_w, bound_h) = bounds_expanded_rotation_matrix(
            shape_hw=(100, 20), angle_deg=90.0
        )
        assert (bound_w, bound_h) == (100, 20)

    def test_zero_angle_is_the_identity_canvas(self):
        _mat, bounds_wh = bounds_expanded_rotation_matrix(shape_hw=(37, 91), angle_deg=0.0)
        assert bounds_wh == (91, 37)  # (width, height)


class TestBoxFromMaskThreshold:
    def test_default_threshold_is_the_historical_greater_than_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:6, 4:8] = 1
        assert BoxYX.from_mask(mask) == BoxYX(3, 6, 4, 8)

    def test_half_open_maxima(self):
        """``+1`` on the max — ``mask[box.to_slices()]`` must select exactly the content."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 1:4] = 1
        box = BoxYX.from_mask(mask)
        assert mask[box.to_slices()].sum() == mask.sum()

    def test_empty_mask_is_none(self):
        assert BoxYX.from_mask(np.zeros((8, 8), dtype=np.uint8)) is None

    def test_threshold_is_semantic_for_interpolated_float_masks(self):
        """``> 0`` and ``> 0.5`` genuinely disagree, which is why they were NOT unified.

        A float mask carrying interpolation smear has faint fractional values outside the true
        object. Reading it at ``> 0`` inflates the box; ``0.5`` recovers the crisp extent.
        """
        mask = np.zeros((10, 10), dtype=np.float64)
        mask[4:6, 4:6] = 1.0
        mask[3, 4] = 0.2  # smear from a bilinear resample
        mask[6, 5] = 0.3

        assert BoxYX.from_mask(mask) == BoxYX(3, 7, 4, 6)
        assert BoxYX.from_mask(mask, threshold=0.5) == BoxYX(4, 6, 4, 6)

    def test_threshold_is_exclusive(self):
        """``> threshold``, so a pixel exactly AT the threshold is background."""
        mask = np.full((4, 4), 0.5)
        assert BoxYX.from_mask(mask, threshold=0.5) is None


class TestResizeInterpolationPolicy:
    def test_masks_are_nearest_regardless_of_direction(self):
        for out in [(50, 50), (200, 200), (300, 60)]:
            assert (
                resize_interpolation_flags(
                    in_shape_yx=(100, 100), out_shape_yx=out, is_mask=True
                )
                == cv2.INTER_NEAREST
            )

    def test_images_area_on_shrink_linear_on_growth(self):
        assert (
            resize_interpolation_flags(
                in_shape_yx=(100, 100), out_shape_yx=(50, 50), is_mask=False
            )
            == cv2.INTER_AREA
        )
        assert (
            resize_interpolation_flags(
                in_shape_yx=(100, 100), out_shape_yx=(200, 200), is_mask=False
            )
            == cv2.INTER_LINEAR
        )

    def test_anisotropic_resize_uses_per_axis_not_total_area(self):
        """THE resolved divergence: 100x100 -> 300x60 grows in area but decimates x.

        Area-based said LINEAR here; per-axis says AREA, because aliasing is per-axis and x is
        being decimated 2x. Measured, the two kernels differ by up to ~240 grey levels on such a
        resize, so this is a real pixel choice.
        """
        assert (
            resize_interpolation_flags(
                in_shape_yx=(100, 100), out_shape_yx=(300, 60), is_mask=False
            )
            == cv2.INTER_AREA
        )

    def test_anti_alias_false_disables_the_prefilter(self):
        assert (
            resize_interpolation_flags(
                in_shape_yx=(100, 100), out_shape_yx=(50, 50), is_mask=False, anti_alias=False
            )
            == cv2.INTER_LINEAR
        )


class TestMaskGeometryDelegation:
    def test_bounding_box_matches_boxyx_and_keeps_the_empty_sentinel(self):
        from data_pipeline.object_extraction.segmentation.masks.mask_geometry import (
            EMPTY_BOUNDING_BOX_XYXY_PX,
            mask_bounding_box_xyxy_px,
        )

        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:7, 3:9] = 1
        box = BoxYX.from_mask(mask)
        # xyxy ordering, not yx — the reason this wrapper still exists.
        assert mask_bounding_box_xyxy_px(mask) == (box.x0, box.y0, box.x1, box.y1)

        # The sentinel is a metrics-row contract: (0,0,0,0), never None.
        assert mask_bounding_box_xyxy_px(np.zeros((5, 5), np.uint8)) == EMPTY_BOUNDING_BOX_XYXY_PX
