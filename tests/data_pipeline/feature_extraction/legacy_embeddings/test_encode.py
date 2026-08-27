"""Tests for legacy_embeddings.encode — pure encode loop.

Uses a synthetic mock encoder satisfying EncoderProtocol — no real model weights needed.
The mock returns configurable mu / logvar tensors so both present/absent logvar paths
are exercised.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas.api.types as ptypes
import pytest
import torch
from PIL import Image

from data_pipeline.feature_extraction.legacy_embeddings.snip_source import SnipInput
from data_pipeline.feature_extraction.legacy_embeddings.encode import encode_snips


# ── Fixtures ─────────────────────────────────────────────────────────────────

LATENT_DIM = 8


class _FakeEncoder:
    """Minimal EncoderProtocol implementation for tests — no torch.nn.Module needed."""

    def __init__(self, latent_dim: int, emit_logvar: bool = True, nuisance_indices: list[int] | None = None):
        self.latent_dim = latent_dim
        self.emit_logvar = emit_logvar
        self.nuisance_indices = nuisance_indices

    def encode_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        B = x.shape[0]
        mu = torch.arange(B * self.latent_dim, dtype=torch.float32).reshape(B, self.latent_dim)
        logvar = -torch.ones(B, self.latent_dim) if self.emit_logvar else None
        return {"mu": mu, "logvar": logvar}


def _make_snip_inputs(n: int, tmp_path: Path) -> list[SnipInput]:
    """Create n small RGB PNG fixtures and return SnipInput list."""
    inputs = []
    for i in range(n):
        arr = np.full((16, 8, 3), i * 10, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        p = tmp_path / f"snip_{i:03d}.png"
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        p.write_bytes(buf.getvalue())
        inputs.append(SnipInput(snip_id=f"snip_{i:03d}", image_path=p))
    return inputs


MODEL_INPUT_SHAPE = (16, 8)  # (H, W) matching the 16x8 fixture PNGs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRowOrderAndCount:
    def test_empty_input_has_typed_required_schema(self):
        enc = _FakeEncoder(LATENT_DIM)
        df = encode_snips([], encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)

        assert df.empty
        assert list(df.columns) == ["snip_id"] + [f"z_mu_{j:02d}" for j in range(LATENT_DIM)]
        assert all(ptypes.is_numeric_dtype(df[col]) for col in df.columns[1:])

    def test_empty_input_requires_encoder_latent_dim(self):
        class _NoDimensionEncoder:
            def encode_batch(self, x):
                raise AssertionError("empty input must not call encode_batch")

        with pytest.raises(ValueError, match="latent_dim"):
            encode_snips([], encoder=_NoDimensionEncoder(), model_input_shape=MODEL_INPUT_SHAPE)

    def test_output_row_count_matches_input(self, tmp_path):
        snip_inputs = _make_snip_inputs(5, tmp_path)
        enc = _FakeEncoder(LATENT_DIM)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert len(df) == 5

    def test_output_row_order_matches_input_order(self, tmp_path):
        snip_inputs = _make_snip_inputs(5, tmp_path)
        enc = _FakeEncoder(LATENT_DIM)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert list(df["snip_id"]) == [si.snip_id for si in snip_inputs]

    def test_batch_size_smaller_than_input_still_produces_all_rows(self, tmp_path):
        snip_inputs = _make_snip_inputs(7, tmp_path)
        enc = _FakeEncoder(LATENT_DIM)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE, batch_size=3)
        assert len(df) == 7

    def test_model_input_channels_passed_through(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM)
        df1 = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE, model_input_channels=1)
        df3 = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE, model_input_channels=3)
        assert len(df1) == len(df3) == 2


class TestZMuColumns:
    def test_emits_z_mu_columns_count_equals_latent_dim(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        z_mu_cols = [c for c in df.columns if c.startswith("z_mu_")]
        assert len(z_mu_cols) == LATENT_DIM

    def test_z_mu_column_naming_is_zero_padded(self, tmp_path):
        snip_inputs = _make_snip_inputs(1, tmp_path)
        enc = _FakeEncoder(4)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert "z_mu_00" in df.columns
        assert "z_mu_03" in df.columns


class TestEncoderOutputConventions:
    def test_encode_batch_with_logvar(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, emit_logvar=True)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert len([c for c in df.columns if c.startswith("z_mu_")]) == LATENT_DIM

    def test_encode_batch_without_logvar(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, emit_logvar=False)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert len([c for c in df.columns if c.startswith("z_mu_")]) == LATENT_DIM


class TestZSigmaColumns:
    def test_emits_z_sigma_when_logvar_present(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, emit_logvar=True)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        z_sigma_cols = [c for c in df.columns if c.startswith("z_sigma_")]
        assert len(z_sigma_cols) == LATENT_DIM

    def test_no_z_sigma_when_logvar_absent(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, emit_logvar=False)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        z_sigma_cols = [c for c in df.columns if c.startswith("z_sigma_")]
        assert z_sigma_cols == [], f"Expected no z_sigma columns, got {z_sigma_cols}"


class TestBiologicalNuisanceSplit:
    """Disentangled encoders (non-empty nuisance_indices) split columns into z_mu_b_*/z_mu_n_*."""

    def test_flat_naming_when_nuisance_indices_none(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, nuisance_indices=None)
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert "z_mu_00" in df.columns
        assert not any(c.startswith("z_mu_b_") or c.startswith("z_mu_n_") for c in df.columns)

    def test_flat_naming_when_nuisance_indices_empty(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, nuisance_indices=[])
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert "z_mu_00" in df.columns
        assert not any(c.startswith("z_mu_b_") or c.startswith("z_mu_n_") for c in df.columns)

    def test_splits_into_biological_and_nuisance_by_raw_index(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, nuisance_indices=[0, 1, 2])
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert [c for c in df.columns if c.startswith("z_mu_n_")] == ["z_mu_n_00", "z_mu_n_01", "z_mu_n_02"]
        assert [c for c in df.columns if c.startswith("z_mu_b_")] == [
            f"z_mu_b_{j:02d}" for j in range(3, LATENT_DIM)
        ]
        assert not any(c.startswith("z_mu_00") for c in df.columns)

    def test_z_sigma_split_mirrors_z_mu_split(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        enc = _FakeEncoder(LATENT_DIM, emit_logvar=True, nuisance_indices=[0, 1, 2])
        df = encode_snips(snip_inputs, encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert [c for c in df.columns if c.startswith("z_sigma_n_")] == [
            "z_sigma_n_00", "z_sigma_n_01", "z_sigma_n_02",
        ]
        assert [c for c in df.columns if c.startswith("z_sigma_b_")] == [
            f"z_sigma_b_{j:02d}" for j in range(3, LATENT_DIM)
        ]

    def test_empty_input_respects_nuisance_split(self, tmp_path):
        enc = _FakeEncoder(LATENT_DIM, nuisance_indices=[0, 1, 2])
        df = encode_snips([], encoder=enc, model_input_shape=MODEL_INPUT_SHAPE)
        assert df.empty
        assert [c for c in df.columns if c != "snip_id" and c.startswith("z_mu_n_")] == [
            "z_mu_n_00", "z_mu_n_01", "z_mu_n_02",
        ]
        assert [c for c in df.columns if c.startswith("z_mu_b_")] == [
            f"z_mu_b_{j:02d}" for j in range(3, LATENT_DIM)
        ]

    def test_encoder_without_nuisance_indices_attribute_falls_back_to_flat(self, tmp_path):
        """Encoders that don't declare nuisance_indices at all (getattr default) still work."""
        class _EncoderNoNuisanceAttr:
            latent_dim = LATENT_DIM

            def encode_batch(self, x):
                B = x.shape[0]
                return {"mu": torch.zeros(B, LATENT_DIM), "logvar": None}

        snip_inputs = _make_snip_inputs(2, tmp_path)
        df = encode_snips(snip_inputs, encoder=_EncoderNoNuisanceAttr(), model_input_shape=MODEL_INPUT_SHAPE)
        assert "z_mu_00" in df.columns
