import numpy as np
import pytest

from data_pipeline.segmentation.masks import decode_binary_mask_rle, encode_binary_mask_rle


def test_encode_empty_mask() -> None:
    mask = np.zeros((2, 3), dtype=bool)

    assert encode_binary_mask_rle(mask) == {"shape": [2, 3], "counts": [6]}


def test_encode_single_pixel_mask() -> None:
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 1] = 1

    assert encode_binary_mask_rle(mask) == {"shape": [2, 3], "counts": [1, 1, 4]}


def test_encode_multi_component_mask() -> None:
    mask = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )

    assert encode_binary_mask_rle(mask) == {"shape": [2, 4], "counts": [0, 2, 3, 1, 1, 1]}


def test_encode_decode_round_trip() -> None:
    mask = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )

    decoded = decode_binary_mask_rle(encode_binary_mask_rle(mask))

    np.testing.assert_array_equal(decoded, mask.astype(bool))
    assert decoded.dtype == np.bool_


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2, 1), dtype=bool),
        np.array([[0.0, 1.0]], dtype=float),
        np.array([[0, 2]], dtype=np.uint8),
    ],
)
def test_encode_rejects_invalid_masks(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_binary_mask_rle(mask)


@pytest.mark.parametrize(
    "rle",
    [
        {"shape": [2], "counts": [2]},
        {"shape": [2, 2], "counts": [5]},
        {"shape": [2, 2], "counts": [3]},
        {"shape": [2, 2], "counts": [1, -1, 4]},
        {"shape": [2, 2], "counts": []},
    ],
)
def test_decode_rejects_invalid_rle(rle: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        decode_binary_mask_rle(rle)
