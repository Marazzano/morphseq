"""Tests for legacy_embeddings.transforms — snip PNG → model input tensor."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from data_pipeline.feature_extraction.legacy_embeddings.transforms import (
    snip_to_model_input_tensor,
)


def _make_png(width: int, height: int, mode: str = "RGB") -> io.BytesIO:
    """Create a small synthetic PNG in memory."""
    rng = np.random.default_rng(0)
    if mode == "L":
        arr = rng.integers(0, 256, (height, width), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _write_png(tmp_path, width: int, height: int, mode: str = "RGB"):
    buf = _make_png(width, height, mode=mode)
    p = tmp_path / f"snip_{width}x{height}_{mode}.png"
    p.write_bytes(buf.read())
    return p


class TestOutputShape:
    def test_default_channels_is_1_grayscale(self, tmp_path):
        # The legacy VAE input_dim=(1, 288, 128) — default must be grayscale
        p = _write_png(tmp_path, 64, 32)
        t = snip_to_model_input_tensor(p, model_input_shape=(32, 64))
        assert t.shape == (1, 32, 64), f"Expected [1, 32, 64] (grayscale default), got {tuple(t.shape)}"

    def test_returns_unbatched_chw(self, tmp_path):
        p = _write_png(tmp_path, 64, 32)
        t = snip_to_model_input_tensor(p, model_input_shape=(32, 64))
        assert t.ndim == 3, f"Expected 3 dims (C, H, W), got {t.ndim}"

    def test_model_input_shape_is_height_width(self, tmp_path):
        # (288, 128) → height=288, width=128 → tensor [1, 288, 128], NOT [1, 128, 288]
        p = _write_png(tmp_path, 50, 80)
        t = snip_to_model_input_tensor(p, model_input_shape=(288, 128))
        assert t.shape == (1, 288, 128), (
            f"model_input_shape=(288,128) should give [1,288,128], got {tuple(t.shape)}"
        )

    def test_channels_3_gives_rgb_shape(self, tmp_path):
        p = _write_png(tmp_path, 32, 32)
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16), model_input_channels=3)
        assert t.shape == (3, 16, 16), f"Expected [3, 16, 16] for channels=3, got {tuple(t.shape)}"

    def test_invalid_channels_raises(self, tmp_path):
        p = _write_png(tmp_path, 32, 32)
        with pytest.raises(ValueError, match="model_input_channels"):
            snip_to_model_input_tensor(p, model_input_shape=(16, 16), model_input_channels=2)


class TestOutputDtype:
    def test_dtype_is_float32(self, tmp_path):
        p = _write_png(tmp_path, 32, 32)
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16))
        assert t.dtype == torch.float32, f"Expected float32, got {t.dtype}"


class TestOutputValues:
    def test_values_in_zero_one(self, tmp_path):
        p = _write_png(tmp_path, 32, 32)
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16))
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0


class TestChannelConversion:
    def test_rgb_png_to_grayscale(self, tmp_path):
        p = _write_png(tmp_path, 32, 32, mode="RGB")
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16), model_input_channels=1)
        assert t.shape[0] == 1, f"Expected 1 channel (grayscale), got {t.shape[0]}"

    def test_grayscale_png_to_grayscale(self, tmp_path):
        p = _write_png(tmp_path, 32, 32, mode="L")
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16), model_input_channels=1)
        assert t.shape[0] == 1

    def test_grayscale_png_to_rgb(self, tmp_path):
        p = _write_png(tmp_path, 32, 32, mode="L")
        t = snip_to_model_input_tensor(p, model_input_shape=(16, 16), model_input_channels=3)
        assert t.shape[0] == 3
