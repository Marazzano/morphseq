"""Tests for legacy_vae_inference_loader — the legacy VAE adapter boundary.

No real model weights are loaded. Tests use a fake torch.nn.Module to verify that
LegacyVaeEncoder correctly normalizes encoder output conventions and exposes metadata.
"""

from __future__ import annotations

import types
import pytest
import torch
import torch.nn as nn

from data_pipeline.feature_extraction.legacy_embeddings.legacy_vae_inference_loader import (
    LegacyVaeEncoder,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeEncoderOutput:
    def __init__(self, mu, logvar, convention):
        if convention == "embedding":
            self.embedding = mu
            self.log_covariance = logvar
        elif convention == "mu":
            self.mu = mu
            self.log_var = logvar
        else:
            raise ValueError(f"Unknown convention: {convention}")


class _FakeModel(nn.Module):
    """Fake encoder module — called directly (no .encoder attribute), returns _FakeEncoderOutput."""

    def __init__(self, latent_dim: int, convention: str = "embedding", emit_logvar: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.convention = convention
        self.emit_logvar = emit_logvar

    def forward(self, x):
        B = x.shape[0]
        mu = torch.zeros(B, self.latent_dim)
        logvar = torch.ones(B, self.latent_dim) * -1.0 if self.emit_logvar else None
        return _FakeEncoderOutput(mu, logvar, self.convention)


def _make_encoder(latent_dim=8, convention="embedding", emit_logvar=True, cfg_overrides=None):
    model = _FakeModel(latent_dim, convention, emit_logvar)
    cfg = {"name": "SeqVAEConfig", "latent_dim": latent_dim}
    if cfg_overrides:
        cfg.update(cfg_overrides)
    return LegacyVaeEncoder(model, cfg, device="cpu")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMetadataFromConfig:
    def test_model_name_from_config(self):
        enc = _make_encoder(cfg_overrides={"name": "SeqVAEConfig"})
        assert enc.model_name == "SeqVAEConfig"

    def test_latent_dim_from_config(self):
        enc = _make_encoder(latent_dim=64)
        assert enc.latent_dim == 64

    def test_nuisance_indices_none_when_absent(self):
        enc = _make_encoder()
        assert enc.nuisance_indices is None

    def test_nuisance_indices_from_config(self):
        enc = _make_encoder(cfg_overrides={"latent_dim": 8, "nuisance_indices": [0, 1]})
        assert enc.nuisance_indices == [0, 1]


class TestEncodeBatch:
    def test_returns_mu_key(self):
        enc = _make_encoder(latent_dim=8)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert "mu" in out

    def test_returns_logvar_key(self):
        enc = _make_encoder(latent_dim=8, emit_logvar=True)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert "logvar" in out

    def test_mu_shape(self):
        enc = _make_encoder(latent_dim=8)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["mu"].shape == (4, 8)

    def test_logvar_shape_when_present(self):
        enc = _make_encoder(latent_dim=8, emit_logvar=True)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["logvar"].shape == (4, 8)

    def test_logvar_is_none_when_absent(self):
        enc = _make_encoder(latent_dim=8, emit_logvar=False)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["logvar"] is None

    def test_output_is_cpu_detached(self):
        enc = _make_encoder(latent_dim=8)
        x = torch.zeros(4, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["mu"].device.type == "cpu"
        assert not out["mu"].requires_grad

    def test_handles_embedding_log_covariance_convention(self):
        enc = _make_encoder(latent_dim=8, convention="embedding")
        x = torch.zeros(3, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["mu"].shape == (3, 8)

    def test_handles_mu_log_var_convention(self):
        enc = _make_encoder(latent_dim=8, convention="mu")
        x = torch.zeros(3, 1, 16, 8)
        out = enc.encode_batch(x)
        assert out["mu"].shape == (3, 8)

    def test_raises_on_unknown_encoder_output(self):
        class _BadOutput:
            pass
        class _BadModel(nn.Module):
            def forward(self, x):
                return _BadOutput()
        enc = LegacyVaeEncoder(_BadModel(), {"name": "SeqVAEConfig", "latent_dim": 8}, device="cpu")
        with pytest.raises(ValueError, match="neither '.embedding' nor '.mu'"):
            enc.encode_batch(torch.zeros(2, 1, 16, 8))
