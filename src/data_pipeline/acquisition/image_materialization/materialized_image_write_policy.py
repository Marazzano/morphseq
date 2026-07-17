"""Write policy for materialized image products.

This module owns how a materialized image product is encoded on disk: file format, downsample
transform, output dtype, and JPEG quality. It is deliberately small. Product selection lives in
``materialization_plan.py``; paths live in ``materialized_image_paths.py``; readers consume the
recorded ``image_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

FileFormat = Literal["png", "jpg", "tif"]
Orientation = Literal["none", "horizontal", "vertical"]
DownsampleMethod = Literal["none", "block_mean", "area_resize"]
PixelDType = Literal["uint8", "uint16"]

_VALID_POLICY_KEYS = frozenset({
    "file_format",
    "orientation",
    "downsample_factor",
    "downsample_method",
    "target_micrometers_per_pixel",
    "pixel_dtype",
    "jpeg_quality",
    "flip_polarity",
})

# Contract columns emitted at the writer/materializer boundary. These are frame_inventory columns,
# but the vocabulary is owned here because it describes the bytes produced by this writer layer.
MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS: tuple[str, ...] = (
    "orientation",
    "image_file_format",
    "pixel_dtype",
    "downsample_factor",
    "downsample_method",
    "jpeg_quality",
    "flip_polarity",
)

MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS: tuple[str, ...] = (
    "jpeg_quality",
)

_BASE_DEFAULT = {
    "file_format": "png",
    "orientation": "none",
    "downsample_factor": 1,
    "downsample_method": "none",
    # When set, the downsample factor is COMPUTED at resolve time as target/native µm/px rather than
    # being a fixed integer. This is what keeps a product at ONE physical resolution across scopes
    # with different native calibrations (Keyence 3.774 µm/px vs YX1 3.231 µm/px); a blind integer
    # factor silently lands them at different µm/px. None = use downsample_factor as given.
    "target_micrometers_per_pixel": None,
    "pixel_dtype": "uint8",
    "jpeg_quality": None,
    # Canonical materialized polarity: invert to bright-embryo/dark-background for EVERY product
    # (projection AND z_stack), both microscopes — downstream snip_processing assumes dark bg.
    # Explicit + per-product overridable here rather than a hidden constant. See
    # image_building/shared/display_polarity.py.
    "flip_polarity": True,
}

# Physical resolution every z-slice product is materialized at, regardless of scope. Matches the
# snip_processing target so z-snips need no upsampling. Set deliberately — the previous blind
# downsample_factor=4 landed Keyence at 15.09 µm/px and YX1 at 12.92 µm/px.
DEFAULT_TARGET_MICROMETERS_PER_PIXEL: float = 6.5

_PRODUCT_DEFAULTS = {
    "BF__z_stack": {
        # jpg remains the default while heavy raw data is still on disk. Flip file_format to "png"
        # (and jpeg_quality to None) for a lossless archive; downsampling is unaffected by format.
        "file_format": "jpg",
        "target_micrometers_per_pixel": DEFAULT_TARGET_MICROMETERS_PER_PIXEL,
        "downsample_method": "area_resize",
        "pixel_dtype": "uint8",
        "jpeg_quality": 85,
    },
    "BF__projection__focus_stack": {
        # The FF stays at NATIVE resolution: it is the detection/segmentation source and the snip
        # source, both of which downsample from it. Do not add a target here without re-tuning them.
        "file_format": "png",
        "downsample_factor": 1,
        "downsample_method": "none",
        "pixel_dtype": "uint8",
        "jpeg_quality": None,
    },
}


@dataclass(frozen=True)
class ImageWritePolicy:
    file_format: FileFormat
    orientation: Orientation
    # May be fractional: a fixed physical target (see target_micrometers_per_pixel) rarely divides
    # a native calibration evenly. ``area_resize`` handles arbitrary scales; ``block_mean`` does not.
    downsample_factor: float
    downsample_method: DownsampleMethod
    pixel_dtype: PixelDType
    jpeg_quality: int | None = None
    # Whether to invert display polarity (bright-embryo/dark-background) for this product. The
    # materializer applies it via image_building/shared/display_polarity.apply_display_polarity.
    flip_polarity: bool = True
    # Resolve-time input only (see _BASE_DEFAULT). Not a frame_inventory column: the recorded
    # downsample_factor + image_micrometers_per_pixel already describe the result exactly.
    target_micrometers_per_pixel: float | None = None


def resolve_image_write_policy(
    config: dict | None,
    product_key: str,
    native_micrometers_per_pixel: float | None = None,
) -> ImageWritePolicy:
    """Resolve the write policy for one canonical image product key.

    ``config["image_materialization"]["write_policies"][product_key]`` overrides locked defaults.
    Unknown override keys fail loud, matching the pipeline's config-boundary style.

    ``native_micrometers_per_pixel`` is REQUIRED when the resolved policy declares
    ``target_micrometers_per_pixel``; the concrete (possibly fractional) downsample factor is
    computed here so the rest of the writer only ever sees a plain factor.
    """
    merged = dict(_BASE_DEFAULT)
    merged.update(_PRODUCT_DEFAULTS.get(str(product_key), {}))

    cfg = config or {}
    write_policies = (cfg.get("image_materialization") or {}).get("write_policies") or {}
    if not isinstance(write_policies, dict):
        raise ValueError(
            "config['image_materialization']['write_policies'] must be a mapping keyed by "
            "canonical product_key."
        )
    overrides = write_policies.get(str(product_key), {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(
            f"config['image_materialization']['write_policies'][{product_key!r}] must be a mapping."
        )
    unknown = sorted(set(overrides) - _VALID_POLICY_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown image write policy key(s) for product {product_key!r}: {unknown}. "
            f"Allowed keys: {sorted(_VALID_POLICY_KEYS)}."
        )
    merged.update(overrides)

    # Resolve target µm/px -> a concrete factor BEFORE the method default (which keys off factor).
    target = merged.pop("target_micrometers_per_pixel", None)
    if "downsample_factor" in overrides and "target_micrometers_per_pixel" in overrides:
        raise ValueError(
            f"Image write policy override for product {product_key!r} sets both "
            f"target_micrometers_per_pixel={overrides['target_micrometers_per_pixel']!r} and "
            f"downsample_factor={overrides['downsample_factor']!r}. They are mutually exclusive: "
            "the target computes the factor. Drop one."
        )
    if "downsample_factor" in overrides:
        # An explicit factor override beats a product-DEFAULT target — the more specific setting
        # wins, and a caller pinning a factor should not have to also null out a default they never
        # asked for.
        target = None
    if target is not None:
        if native_micrometers_per_pixel is None:
            raise ValueError(
                f"Image write policy for product {product_key!r} declares "
                f"target_micrometers_per_pixel={target!r}, but resolve_image_write_policy was "
                "called without native_micrometers_per_pixel. The caller must supply the source "
                "calibration so the factor can be computed."
            )
        native = float(native_micrometers_per_pixel)
        if not native > 0:
            raise ValueError(
                f"native_micrometers_per_pixel must be > 0; got {native_micrometers_per_pixel!r}."
            )
        target = float(target)
        if not target > 0:
            raise ValueError(
                f"target_micrometers_per_pixel must be > 0; got {target!r}."
            )
        # Never upsample: a source coarser than the target keeps its native resolution rather than
        # inventing detail. The honest result is recorded via image_micrometers_per_pixel, which the
        # materializer derives from the ACTUAL written dims.
        merged["downsample_factor"] = max(1.0, target / native)

    if "downsample_method" not in overrides:
        merged["downsample_method"] = _default_downsample_method(
            str(merged["file_format"]), float(merged["downsample_factor"])
        )

    policy = _validate_policy(ImageWritePolicy(**merged))
    return replace(policy, target_micrometers_per_pixel=target)


def suffix_for_policy(policy: ImageWritePolicy) -> str:
    """Return the path-extension token expected by materialized_image_paths.py."""
    return policy.file_format


def expected_downsampled_dims(
    width_px: int,
    height_px: int,
    factor: float,
    method: str,
) -> tuple[int, int]:
    """Return expected ``(width, height)`` after the configured downsample transform.

    ``factor`` may be fractional for ``area_resize``. ``block_mean`` is an integer-only reduction.
    """
    factor = float(factor)
    method = str(method)
    if factor < 1:
        raise ValueError(f"downsample_factor must be >= 1; got {factor}.")
    if method == "none" or factor == 1:
        return int(width_px), int(height_px)
    if method == "area_resize":
        return max(1, int(round(width_px / factor))), max(1, int(round(height_px / factor)))
    if method != "block_mean":
        raise ValueError(
            f"Unknown downsample_method {method!r}; expected 'none', 'block_mean', "
            "or 'area_resize'."
        )
    if not float(factor).is_integer():
        raise ValueError(
            f"block_mean downsample requires an integer factor; got {factor}. Use "
            "'area_resize' for fractional scales (e.g. a fixed µm/px target)."
        )
    factor = int(factor)
    width_px = int(width_px)
    height_px = int(height_px)
    if width_px % factor != 0 or height_px % factor != 0:
        raise ValueError(
            f"block_mean downsample requires native dimensions divisible by factor={factor}; "
            f"got width={width_px}, height={height_px}."
        )
    return width_px // factor, height_px // factor


def downsample_image(image: np.ndarray, policy: ImageWritePolicy) -> np.ndarray:
    """Apply the policy downsample transform to a 2D image."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"downsample_image expects a 2D image; got shape {arr.shape}.")
    factor = float(policy.downsample_factor)
    if policy.downsample_method == "none" or factor == 1:
        return arr
    out_w, out_h = expected_downsampled_dims(
        arr.shape[1], arr.shape[0], factor, policy.downsample_method
    )
    if policy.downsample_method == "area_resize":
        from data_pipeline.object_extraction.segmentation.masks.mask_resize import resize_image_to_shape

        return resize_image_to_shape(arr, (out_h, out_w))
    # block_mean: integer-only; expected_downsampled_dims has already enforced that above.
    factor = int(factor)
    h, w = arr.shape
    reduced = arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    if np.issubdtype(arr.dtype, np.integer):
        return np.rint(reduced).clip(np.iinfo(arr.dtype).min, np.iinfo(arr.dtype).max).astype(arr.dtype)
    return reduced.astype(arr.dtype, copy=False)


def orient_image_for_write(image: np.ndarray, policy: ImageWritePolicy) -> np.ndarray:
    """Rotate a 2D image only when needed to satisfy the write-policy orientation."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"orient_image_for_write expects a 2D image; got shape {arr.shape}.")
    if policy.orientation == "none":
        return arr
    height_px, width_px = arr.shape
    if policy.orientation == "horizontal":
        return arr if width_px >= height_px else np.rot90(arr)
    if policy.orientation == "vertical":
        return arr if height_px >= width_px else np.rot90(arr)
    raise ValueError(
        f"Unknown orientation {policy.orientation!r}; expected 'none', 'horizontal', or 'vertical'."
    )


def convert_pixel_dtype(image: np.ndarray, pixel_dtype: PixelDType) -> np.ndarray:
    """Convert to the encoder/read-back dtype using a fixed range, no per-image normalization."""
    arr = np.asarray(image)
    if pixel_dtype == "uint8":
        if arr.dtype == np.uint8:
            return arr
        if arr.dtype == np.uint16:
            return np.rint(arr.astype(np.float64) * (255.0 / 65535.0)).clip(0, 255).astype(np.uint8)
        return np.rint(arr.astype(np.float64)).clip(0, 255).astype(np.uint8)
    if pixel_dtype == "uint16":
        if arr.dtype == np.uint16:
            return arr
        if arr.dtype == np.uint8:
            return (arr.astype(np.uint16) * 257).astype(np.uint16)
        return np.rint(arr.astype(np.float64)).clip(0, 65535).astype(np.uint16)
    raise ValueError(f"Unknown pixel_dtype {pixel_dtype!r}; expected 'uint8' or 'uint16'.")


def prepare_image_for_write(image: np.ndarray, policy: ImageWritePolicy) -> np.ndarray:
    """Apply the locked write order: orient, downsample, then fixed dtype conversion."""
    oriented = orient_image_for_write(image, policy)
    downsampled = downsample_image(oriented, policy)
    return convert_pixel_dtype(downsampled, policy.pixel_dtype)


def write_image(image: np.ndarray, path: Path, policy: ImageWritePolicy) -> None:
    """Prepare and write one materialized 2D image according to ``policy``."""
    policy = _validate_policy(policy)
    prepared = prepare_image_for_write(image, policy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(prepared)
    if policy.file_format == "jpg":
        pil.save(path, format="JPEG", quality=int(policy.jpeg_quality))
        return
    if policy.file_format == "png":
        pil.save(path, format="PNG")
        return
    if policy.file_format == "tif":
        pil.save(path, format="TIFF")
        return
    raise ValueError(f"Unknown file_format {policy.file_format!r}.")


def _default_downsample_method(file_format: str, factor: float) -> str:
    """Default resampler for a factor. DELIBERATELY format-agnostic.

    This used to key off ``file_format`` and return "none" for anything non-jpg, which meant a
    config that flipped a product to PNG without also naming downsample_method silently resolved to
    method="none" and turned downsampling OFF with no error. Encoding and resampling are orthogonal:
    how many pixels to keep is not a function of how they are compressed.
    """
    if float(factor) == 1.0:
        return "none"
    return "area_resize"


def _validate_policy(policy: ImageWritePolicy) -> ImageWritePolicy:
    fmt = _normalize_format(policy.file_format)
    orientation = _normalize_orientation(policy.orientation)
    method = str(policy.downsample_method)
    dtype = str(policy.pixel_dtype)
    if method not in ("none", "block_mean", "area_resize"):
        raise ValueError(
            f"downsample_method must be 'none', 'block_mean', or 'area_resize'; got {method!r}."
        )
    if dtype not in ("uint8", "uint16"):
        raise ValueError(f"pixel_dtype must be 'uint8' or 'uint16'; got {dtype!r}.")
    factor = float(policy.downsample_factor)
    if factor < 1:
        raise ValueError(f"downsample_factor must be >= 1; got {factor}.")
    if fmt == "jpg":
        if dtype != "uint8":
            raise ValueError("jpg image write policy requires pixel_dtype='uint8'.")
        if policy.jpeg_quality is None:
            raise ValueError("jpg image write policy requires jpeg_quality.")
        quality = int(policy.jpeg_quality)
        if quality < 1 or quality > 100:
            raise ValueError(f"jpeg_quality must be in [1, 100]; got {quality}.")
    else:
        if policy.jpeg_quality is not None:
            raise ValueError(f"{fmt} image write policy requires jpeg_quality=None.")
        quality = None
    return replace(
        policy,
        file_format=fmt,
        orientation=orientation,
        downsample_factor=factor,
        downsample_method=method,  # type: ignore[arg-type]
        pixel_dtype=dtype,  # type: ignore[arg-type]
        jpeg_quality=quality,
    )


def _normalize_format(file_format: str) -> FileFormat:
    fmt = str(file_format).lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt == "tiff":
        fmt = "tif"
    if fmt not in ("png", "jpg", "tif"):
        raise ValueError(
            f"file_format must be one of 'png', 'jpg', 'tif'; got {file_format!r}."
        )
    return fmt  # type: ignore[return-value]


def _normalize_orientation(orientation: str) -> Orientation:
    normalized = str(orientation).lower()
    if normalized not in ("none", "horizontal", "vertical"):
        raise ValueError(
            "orientation must be one of 'none', 'horizontal', 'vertical'; "
            f"got {orientation!r}."
        )
    return normalized  # type: ignore[return-value]
