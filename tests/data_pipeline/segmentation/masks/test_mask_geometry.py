import numpy as np
import pytest

from data_pipeline.segmentation.masks import (
    mask_area_px,
    mask_bounding_box_xyxy_px,
    mask_centroid_xy_px,
    mask_geometry,
)


def test_empty_mask_geometry() -> None:
    mask = np.zeros((3, 4), dtype=bool)

    assert mask_area_px(mask) == 0
    assert mask_bounding_box_xyxy_px(mask) == (0, 0, 0, 0)
    assert mask_centroid_xy_px(mask) == (0.0, 0.0)
    assert mask_geometry(mask) == {
        "area_px": 0,
        "bbox_x_min_px": 0,
        "bbox_y_min_px": 0,
        "bbox_x_max_px": 0,
        "bbox_y_max_px": 0,
        "centroid_x_px": 0.0,
        "centroid_y_px": 0.0,
    }


def test_single_pixel_geometry_uses_half_open_bbox_and_pixel_center_centroid() -> None:
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 2] = 1

    assert mask_area_px(mask) == 1
    assert mask_bounding_box_xyxy_px(mask) == (2, 1, 3, 2)
    assert mask_centroid_xy_px(mask) == (2.5, 1.5)


def test_multi_component_geometry() -> None:
    mask = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    assert mask_geometry(mask) == {
        "area_px": 4,
        "bbox_x_min_px": 1,
        "bbox_y_min_px": 0,
        "bbox_x_max_px": 4,
        "bbox_y_max_px": 3,
        "centroid_x_px": 2.5,
        "centroid_y_px": 1.5,
    }


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2, 1), dtype=bool),
        np.array([[0.0, 1.0]], dtype=float),
        np.array([[0, 2]], dtype=np.uint8),
    ],
)
def test_geometry_rejects_invalid_masks_like_rle(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        mask_geometry(mask)
