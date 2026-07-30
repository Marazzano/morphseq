"""Tests for materialized_image_write_policy.py."""

import numpy as np
import pytest
from PIL import Image

from data_pipeline.acquisition.image_materialization.materialized_image_write_policy import (
    ImageWritePolicy,
    MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS,
    MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS,
    convert_pixel_dtype,
    downsample_image,
    expected_downsampled_dims,
    prepare_image_for_write,
    resolve_image_write_policy,
    suffix_for_policy,
    write_image,
)


def test_resolve_defaults_for_z_stack():
    # z_stack targets a FIXED physical resolution, so its factor is derived from the source
    # calibration rather than being a constant. jpg stays the default while raw data is on disk.
    policy = resolve_image_write_policy({}, "BF__z_stack", native_micrometers_per_pixel=3.25)
    assert policy.file_format == "jpg"
    assert policy.orientation == "none"
    assert policy.downsample_method == "area_resize"
    assert policy.pixel_dtype == "uint8"
    assert policy.jpeg_quality == 85
    assert policy.downsample_factor == pytest.approx(2.0)  # 6.5 / 3.25
    assert suffix_for_policy(policy) == "jpg"


def test_z_stack_lands_on_one_physical_resolution_across_scopes():
    """The point of the target: different natives must converge on the same µm/px."""
    keyence = resolve_image_write_policy({}, "BF__z_stack", native_micrometers_per_pixel=3.7744)
    yx1 = resolve_image_write_policy({}, "BF__z_stack", native_micrometers_per_pixel=3.2308)
    assert 3.7744 * keyence.downsample_factor == pytest.approx(6.5)
    assert 3.2308 * yx1.downsample_factor == pytest.approx(6.5)
    # ...and the factors are genuinely fractional, which the int contract used to forbid.
    assert not float(keyence.downsample_factor).is_integer()


def test_target_without_native_calibration_fails_loud():
    with pytest.raises(ValueError, match="native_micrometers_per_pixel"):
        resolve_image_write_policy({}, "BF__z_stack")


def test_explicit_downsample_factor_override_beats_product_default_target():
    # A caller pinning a factor should not also have to null out a default target it never set.
    cfg = {"image_materialization": {"write_policies": {"BF__z_stack": {"downsample_factor": 4}}}}
    policy = resolve_image_write_policy(cfg, "BF__z_stack")
    assert policy.downsample_factor == 4


def test_override_naming_both_target_and_factor_fails_loud():
    cfg = {
        "image_materialization": {
            "write_policies": {
                "BF__z_stack": {"downsample_factor": 4, "target_micrometers_per_pixel": 6.5}
            }
        }
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_image_write_policy(cfg, "BF__z_stack", native_micrometers_per_pixel=3.25)


def test_never_upsamples_when_source_is_coarser_than_target():
    policy = resolve_image_write_policy({}, "BF__z_stack", native_micrometers_per_pixel=9.0)
    assert policy.downsample_factor == 1.0
    assert policy.downsample_method == "none"


def test_png_override_keeps_downsampling_active():
    """Regression: file_format used to drive downsample_method, so flipping a product to PNG
    silently resolved method='none' and turned downsampling OFF with no error."""
    cfg = {
        "image_materialization": {
            "write_policies": {"BF__z_stack": {"file_format": "png", "jpeg_quality": None}}
        }
    }
    policy = resolve_image_write_policy(cfg, "BF__z_stack", native_micrometers_per_pixel=3.7744)
    assert policy.file_format == "png"
    assert policy.downsample_method == "area_resize"
    assert policy.downsample_factor == pytest.approx(6.5 / 3.7744)


def test_flip_polarity_defaults_true_and_is_overridable():
    # Canonical display polarity is inverted for every product (both scopes); the flag is an
    # explicit, per-product write-policy field, not a hidden constant.
    assert resolve_image_write_policy({}, "BF__projection__focus_stack").flip_polarity is True
    assert resolve_image_write_policy(
        {}, "BF__z_stack", native_micrometers_per_pixel=3.25
    ).flip_polarity is True
    overridden = resolve_image_write_policy(
        {"image_materialization": {"write_policies": {"BF__z_stack": {"flip_polarity": False}}}},
        "BF__z_stack",
        native_micrometers_per_pixel=3.25,
    )
    assert overridden.flip_polarity is False


def test_resolve_override_fills_downsample_method_default():
    cfg = {
        "image_materialization": {
            "write_policies": {
                "BF__z_stack": {
                    "file_format": "jpg",
                    "orientation": "vertical",
                    "jpeg_quality": 90,
                    "downsample_factor": 4,
                    "pixel_dtype": "uint8",
                }
            }
        }
    }
    policy = resolve_image_write_policy(cfg, "BF__z_stack")
    assert policy.orientation == "vertical"
    assert policy.downsample_method == "area_resize"
    assert policy.jpeg_quality == 90


def test_unknown_override_key_fails_loud():
    cfg = {
        "image_materialization": {
            "write_policies": {
                "BF__z_stack": {"compression_mode": "lossy"},
            }
        }
    }
    with pytest.raises(ValueError, match="Unknown image write policy key"):
        resolve_image_write_policy(cfg, "BF__z_stack")


@pytest.mark.parametrize(
    "policy,match",
    [
        (ImageWritePolicy("jpg", "none", 1, "none", "uint16", 85), "jpg.*uint8"),
        (ImageWritePolicy("jpg", "none", 1, "none", "uint8", None), "jpg.*jpeg_quality"),
        (ImageWritePolicy("png", "none", 1, "none", "uint8", 85), "png.*jpeg_quality=None"),
        (ImageWritePolicy("tif", "none", 1, "none", "uint8", 85), "tif.*jpeg_quality=None"),
    ],
)
def test_format_route_validation(policy, match):
    with pytest.raises(ValueError, match=match):
        write_image(np.zeros((4, 4), dtype=np.uint16), "/tmp/not_written.png", policy)


def test_expected_downsampled_dims_block_mean_and_divisibility():
    assert expected_downsampled_dims(512, 256, 4, "block_mean") == (128, 64)
    with pytest.raises(ValueError, match="divisible"):
        expected_downsampled_dims(513, 256, 4, "block_mean")


def test_expected_downsampled_dims_area_resize_allows_nondivisible():
    assert expected_downsampled_dims(2189, 2189, 4, "area_resize") == (547, 547)


def test_downsample_image_block_mean_shape_and_values():
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    policy = ImageWritePolicy("png", "none", 2, "block_mean", "uint16")
    out = downsample_image(image, policy)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[2, 4], [10, 12]], dtype=np.uint16))


def test_downsample_image_area_resize_shape_nondivisible():
    image = np.arange(9 * 9, dtype=np.uint16).reshape(9, 9)
    policy = ImageWritePolicy("jpg", "none", 4, "area_resize", "uint8", 85)
    out = downsample_image(image, policy)
    assert out.shape == (2, 2)
    assert out.dtype == np.uint16


def test_fixed_scale_uint16_to_uint8_no_per_plane_stretch():
    plane_low = np.array([[0, 32768]], dtype=np.uint16)
    plane_high = np.array([[32768, 65535]], dtype=np.uint16)

    low_u8 = convert_pixel_dtype(plane_low, "uint8")
    high_u8 = convert_pixel_dtype(plane_high, "uint8")

    assert low_u8[0, 1] == high_u8[0, 0]
    assert low_u8[0, 1] < 255
    assert high_u8[0, 0] > 0


def test_prepare_downsamples_before_dtype_conversion():
    image = np.array([[0, 65535], [65535, 65535]], dtype=np.uint16)
    policy = ImageWritePolicy("jpg", "none", 2, "block_mean", "uint8", 85)
    out = prepare_image_for_write(image, policy)
    assert out.shape == (1, 1)
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 191


def test_prepare_image_for_write_orientation_none_keeps_native_layout():
    image = np.arange(6, dtype=np.uint16).reshape(3, 2)
    policy = ImageWritePolicy("png", "none", 1, "none", "uint16")
    out = prepare_image_for_write(image, policy)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, image)


def test_prepare_image_for_write_horizontal_leaves_horizontal_image():
    image = np.arange(6, dtype=np.uint16).reshape(2, 3)
    policy = ImageWritePolicy("png", "horizontal", 1, "none", "uint16")
    out = prepare_image_for_write(image, policy)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, image)


def test_prepare_image_for_write_horizontal_rotates_vertical_image():
    image = np.arange(6, dtype=np.uint16).reshape(3, 2)
    policy = ImageWritePolicy("png", "horizontal", 1, "none", "uint16")
    out = prepare_image_for_write(image, policy)
    expected = np.rot90(image)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, expected)


def test_prepare_image_for_write_vertical_rotates_horizontal_image():
    image = np.arange(6, dtype=np.uint16).reshape(2, 3)
    policy = ImageWritePolicy("png", "vertical", 1, "none", "uint16")
    out = prepare_image_for_write(image, policy)
    expected = np.rot90(image)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, expected)


def test_materialized_image_write_policy_columns_drop_source_dims_and_include_orientation():
    assert MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS == (
        "orientation",
        "image_file_format",
        "pixel_dtype",
        "downsample_factor",
        "downsample_method",
        "jpeg_quality",
        "flip_polarity",
    )
    assert MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS == ("jpeg_quality",)


@pytest.mark.parametrize(
    "policy,ext,expected_dtype",
    [
        (ImageWritePolicy("png", "none", 1, "none", "uint8"), "png", np.uint8),
        (ImageWritePolicy("jpg", "none", 4, "area_resize", "uint8", 85), "jpg", np.uint8),
        (ImageWritePolicy("tif", "none", 1, "none", "uint16"), "tif", np.uint16),
    ],
)
def test_write_image_round_trip_per_format(tmp_path, policy, ext, expected_dtype):
    image = np.linspace(0, 65535, num=64, dtype=np.uint16).reshape(8, 8)
    out_path = tmp_path / f"frame.{ext}"

    write_image(image, out_path, policy)

    assert out_path.exists()
    with Image.open(out_path) as im:
        arr = np.asarray(im)
    expected_w, expected_h = expected_downsampled_dims(
        image.shape[1], image.shape[0], policy.downsample_factor, policy.downsample_method
    )
    assert arr.shape == (expected_h, expected_w)
    assert arr.dtype == expected_dtype


# ---------------------------------------------------------------------------
# Polarity default reversal (2026-07-29): _BASE_DEFAULT flipped True -> False so the UNSAFE case
# must be opted into. These pin the reversal: brightfield must still invert, everything else must
# not. Inverting brightfield silently breaks snip_processing (it assumes dark background); inverting
# fluorescence silently corrupts intensity.
# ---------------------------------------------------------------------------

_YX1_NATIVE_UM_PER_PX = 3.2308


def test_base_default_does_not_flip_polarity():
    """An unlisted product must fail SAFE (no inversion), not inherit brightfield behavior."""
    policy = resolve_image_write_policy(None, "GFP__projection__mean")
    assert policy.flip_polarity is False


@pytest.mark.parametrize(
    "product_key,native_um_per_px",
    [
        ("BF__z_stack", _YX1_NATIVE_UM_PER_PX),
        ("BF__projection__focus_stack", None),
    ],
)
def test_brightfield_products_still_invert_after_default_reversal(product_key, native_um_per_px):
    """REGRESSION GUARD for the _BASE_DEFAULT flip: every BF product pins flip_polarity=True.

    If this fails, brightfield is being written non-inverted and snip_processing breaks downstream.
    """
    policy = resolve_image_write_policy(None, product_key, native_um_per_px)
    assert policy.flip_polarity is True, (
        f"{product_key} must pin flip_polarity=True explicitly; _BASE_DEFAULT is False."
    )


def test_rfp_max_is_quantitative_native_uint16_no_flip():
    """The RFP max product must preserve intensity: native scale, uint16, lossless, no inversion."""
    policy = resolve_image_write_policy(None, "RFP__projection__max", _YX1_NATIVE_UM_PER_PX)

    assert policy.flip_polarity is False
    assert policy.pixel_dtype == "uint16"
    assert policy.file_format == "png"
    assert policy.jpeg_quality is None
    # No target um/px: the factor must stay 1 regardless of the scope's native calibration, or a
    # bright punctum would be averaged against its dark surroundings.
    assert policy.downsample_factor == 1
    assert policy.downsample_method == "none"


def test_max_projection_written_png_is_exactly_the_input_max(tmp_path):
    """THE load-bearing pixel test: a uint16 max projection must survive the write path EXACTLY.

    One assertion catches every silent-corruption mode this path can introduce: uint8 narrowing,
    polarity inversion, clipping, spatial resize, and per-image normalization. Values are chosen
    above 255 so an accidental uint8 round-trip cannot pass.
    """
    rng = np.random.default_rng(20260729)
    stack_zyx = rng.integers(300, 65535, size=(9, 32, 24), dtype=np.uint16)
    # Distinct per-plane maxima so an off-by-one in the Z axis is detectable, and a known extreme
    # that must arrive unclipped.
    for z in range(stack_zyx.shape[0]):
        stack_zyx[z, z, 0] = 1000 + z
    stack_zyx[4, 10, 10] = 65535

    expected = stack_zyx.max(axis=0)
    assert expected.dtype == np.uint16
    assert expected.max() == 65535

    policy = resolve_image_write_policy(None, "RFP__projection__max", _YX1_NATIVE_UM_PER_PX)
    out_path = tmp_path / "20250912_A01_RFP_t0000.png"
    write_image(expected, out_path, policy)

    with Image.open(out_path) as im:
        written = np.asarray(im)

    assert written.dtype == np.uint16, "uint8 narrowing would destroy quantitative intensity"
    assert written.shape == expected.shape, "no spatial resize is permitted at native scale"
    np.testing.assert_array_equal(written, expected)
