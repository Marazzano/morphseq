"""Write policy for materialized image products.

This module owns how a materialized image product is encoded on disk: file format, downsample
transform, output dtype, and JPEG quality. It is deliberately small. Product selection lives in
``materialization_plan.py``; paths live in ``materialized_image_paths.py``; readers consume the
recorded ``source_image_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

FileFormat = Literal["png", "jpg", "tif"]
DownsampleMethod = Literal["none", "block_mean", "area_resize"]
PixelDType = Literal["uint8", "uint16"]

_VALID_POLICY_KEYS = frozenset({
    "file_format",
    "downsample_factor",
    "downsample_method",
    "pixel_dtype",
    "jpeg_quality",
})

# Contract columns emitted at the writer/materializer boundary. These are frame_inventory columns,
# but the vocabulary is owned here because it describes the bytes produced by this writer layer.
MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS: tuple[str, ...] = (
    "source_image_width_px",
    "source_image_height_px",
    "image_file_format",
    "pixel_dtype",
    "downsample_factor",
    "downsample_method",
    "jpeg_quality",
)

MATERIALIZED_IMAGE_WRITE_POLICY_NULLABLE_COLUMNS: tuple[str, ...] = (
    "jpeg_quality",
)

_BASE_DEFAULT = {
    "file_format": "png",
    "downsample_factor": 1,
    "downsample_method": "none",
    "pixel_dtype": "uint8",
    "jpeg_quality": None,
}

_PRODUCT_DEFAULTS = {
    "BF__z_stack": {
        "file_format": "jpg",
        "downsample_factor": 4,
        "downsample_method": "area_resize",
        "pixel_dtype": "uint8",
        "jpeg_quality": 85,
    },
    "BF__projection__focus_stack": {
        "file_format": "png",
        "downsample_factor": 1,
        "downsample_method": "none",
        "pixel_dtype": "uint8",
        "jpeg_quality": None,
    },
}

_DEFAULT_METHOD_BY_FORMAT_FACTOR = {
    ("png", 1): "none",
    ("jpg", 4): "area_resize",
    ("tif", 1): "none",
}


@dataclass(frozen=True)
class ImageWritePolicy:
    file_format: FileFormat
    downsample_factor: int
    downsample_method: DownsampleMethod
    pixel_dtype: PixelDType
    jpeg_quality: int | None = None


def resolve_image_write_policy(config: dict | None, product_key: str) -> ImageWritePolicy:
    """Resolve the write policy for one canonical image product key.

    ``config["image_materialization"]["write_policies"][product_key]`` overrides locked defaults.
    Unknown override keys fail loud, matching the pipeline's config-boundary style.
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

    if "downsample_method" not in overrides:
        merged["downsample_method"] = _default_downsample_method(
            str(merged["file_format"]), int(merged["downsample_factor"])
        )

    return _validate_policy(ImageWritePolicy(**merged))


def suffix_for_policy(policy: ImageWritePolicy) -> str:
    """Return the path-extension token expected by materialized_image_paths.py."""
    return policy.file_format


def expected_downsampled_dims(
    width_px: int,
    height_px: int,
    factor: int,
    method: str,
) -> tuple[int, int]:
    """Return expected ``(width, height)`` after the configured downsample transform."""
    factor = int(factor)
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
    factor = int(policy.downsample_factor)
    if policy.downsample_method == "none" or factor == 1:
        return arr
    out_w, out_h = expected_downsampled_dims(
        arr.shape[1], arr.shape[0], factor, policy.downsample_method
    )
    if policy.downsample_method == "area_resize":
        from data_pipeline.segmentation.masks.mask_resize import resize_image_to_shape

        return resize_image_to_shape(arr, (out_h, out_w))
    h, w = arr.shape
    reduced = arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    if np.issubdtype(arr.dtype, np.integer):
        return np.rint(reduced).clip(np.iinfo(arr.dtype).min, np.iinfo(arr.dtype).max).astype(arr.dtype)
    return reduced.astype(arr.dtype, copy=False)


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
    """Apply the locked write order: downsample, then fixed dtype conversion."""
    return convert_pixel_dtype(downsample_image(image, policy), policy.pixel_dtype)


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


def _default_downsample_method(file_format: str, factor: int) -> str:
    if factor == 1:
        return "none"
    if file_format == "jpg":
        return _DEFAULT_METHOD_BY_FORMAT_FACTOR.get((file_format, factor), "block_mean")
    return "none"


def _validate_policy(policy: ImageWritePolicy) -> ImageWritePolicy:
    fmt = _normalize_format(policy.file_format)
    method = str(policy.downsample_method)
    dtype = str(policy.pixel_dtype)
    if method not in ("none", "block_mean", "area_resize"):
        raise ValueError(
            f"downsample_method must be 'none', 'block_mean', or 'area_resize'; got {method!r}."
        )
    if dtype not in ("uint8", "uint16"):
        raise ValueError(f"pixel_dtype must be 'uint8' or 'uint16'; got {dtype!r}.")
    factor = int(policy.downsample_factor)
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
