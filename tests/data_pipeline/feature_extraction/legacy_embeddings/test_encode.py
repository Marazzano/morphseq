"""Tests for legacy_embeddings.encode — pure encode loop.

Uses a synthetic mock encoder — no real model weights needed. The mock returns a
ModelOutput-like object with configurable embedding / log_covariance attributes so
both encoder output conventions are exercised.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data_pipeline.feature_extraction.legacy_embeddings.snip_source import SnipInput
from data_pipeline.feature_extraction.legacy_embeddings.encode import encode_snips


# ── Fixtures ─────────────────────────────────────────────────────────────────

LATENT_DIM = 8


class _EncoderOutput:
    """Minimal encoder output stub, matching both naming conventions."""

    def __init__(self, mu: torch.Tensor, log_var: torch.Tensor | None, convention: str):
        if convention == "embedding":
            self.embedding = mu
            self.log_covariance = log_var
        elif convention == "mu":
            self.mu = mu
            self.log_var = log_var
        else:
            raise ValueError(f"Unknown convention: {convention}")


class _MockEncoder(torch.nn.Module):
    def __init__(self, latent_dim: int, convention: str = "embedding", emit_logvar: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.convention = convention
        self.emit_logvar = emit_logvar

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        mu = torch.arange(B * self.latent_dim, dtype=torch.float32).reshape(B, self.latent_dim)
        log_var = -torch.ones(B, self.latent_dim) if self.emit_logvar else None
        return _EncoderOutput(mu, log_var, self.convention)


class _MockLitModel:
    def __init__(self, latent_dim: int, convention: str = "embedding", emit_logvar: bool = True):
        self.encoder = _MockEncoder(latent_dim, convention, emit_logvar)

    def eval(self):
        return self

    def to(self, device):
        return self


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
    def test_output_row_count_matches_input(self, tmp_path):
        snip_inputs = _make_snip_inputs(5, tmp_path)
        model = _MockLitModel(LATENT_DIM)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        assert len(df) == 5

    def test_output_row_order_matches_input_order(self, tmp_path):
        snip_inputs = _make_snip_inputs(5, tmp_path)
        model = _MockLitModel(LATENT_DIM)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        assert list(df["snip_id"]) == [si.snip_id for si in snip_inputs]

    def test_batch_size_smaller_than_input_still_produces_all_rows(self, tmp_path):
        snip_inputs = _make_snip_inputs(7, tmp_path)
        model = _MockLitModel(LATENT_DIM)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE, batch_size=3)
        assert len(df) == 7

    def test_model_input_channels_passed_through(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM)
        # channels=1 (default, grayscale) and channels=3 (RGB) should both produce output
        df1 = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE, model_input_channels=1)
        df3 = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE, model_input_channels=3)
        assert len(df1) == len(df3) == 2


class TestZMuColumns:
    def test_emits_z_mu_columns_count_equals_latent_dim(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        z_mu_cols = [c for c in df.columns if c.startswith("z_mu_")]
        assert len(z_mu_cols) == LATENT_DIM

    def test_z_mu_column_naming_is_zero_padded(self, tmp_path):
        snip_inputs = _make_snip_inputs(1, tmp_path)
        model = _MockLitModel(4)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        assert "z_mu_00" in df.columns
        assert "z_mu_03" in df.columns


class TestEncoderOutputConventions:
    def test_handles_embedding_log_covariance_convention(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM, convention="embedding")
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        assert len([c for c in df.columns if c.startswith("z_mu_")]) == LATENT_DIM

    def test_handles_mu_log_var_convention(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM, convention="mu")
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        assert len([c for c in df.columns if c.startswith("z_mu_")]) == LATENT_DIM


class TestZSigmaColumns:
    def test_emits_z_sigma_when_logvar_present(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM, emit_logvar=True)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        z_sigma_cols = [c for c in df.columns if c.startswith("z_sigma_")]
        assert len(z_sigma_cols) == LATENT_DIM

    def test_no_z_sigma_when_logvar_absent(self, tmp_path):
        snip_inputs = _make_snip_inputs(2, tmp_path)
        model = _MockLitModel(LATENT_DIM, emit_logvar=False)
        df = encode_snips(model, snip_inputs, MODEL_INPUT_SHAPE)
        z_sigma_cols = [c for c in df.columns if c.startswith("z_sigma_")]
        assert z_sigma_cols == [], f"Expected no z_sigma columns, got {z_sigma_cols}"
