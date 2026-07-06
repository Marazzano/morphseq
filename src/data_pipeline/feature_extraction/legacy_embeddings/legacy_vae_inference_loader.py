"""Legacy VAE inference loader — no legacy.vae.* imports anywhere in this file.

The production model (``20241107_ds_sweep01_optimum``) saved its custom conv encoder as
``encoder.pkl`` (cloudpickle, referencing ``src.vae.*`` — an old module path that no longer
exists) and as ``encoder_reformatted.pt`` (a plain state dict added later for exactly this
situation). This loader uses ``encoder_reformatted.pt`` only.

The conv architecture is reconstructed directly from PyTorch primitives — inferred from the
state dict key shapes (verified against input_dim=[1,288,128] in model_config.json):
  5 × (Conv2d kernel=4 stride=2 padding=1, BatchNorm2d, ReLU)
  channels: 1 → 16 → 32 → 64 → 128 → 256
  spatial after 5 layers: 9 × 4 (for 288 × 128 input) → flatten → 9216
  embedding0: Linear(9216, latent_dim)   → mu head (via embedding = Linear(latent_dim, latent_dim))
  log_var:    Linear(9216, latent_dim)   → logvar head

Importing this module from Python 3.10 is safe (no legacy deps). The actual load must run
under Python 3.9 (the legacy weights were saved there).

Public API::

    encoder = load_legacy_vae_encoder(model_dir, device="cpu")
    out = encoder.encode_batch(batch_tensor)   # {"mu": ..., "logvar": ...}

``LegacyVaeEncoder`` exposes metadata from ``model_config.json`` (the saved artifact is truth)::

    encoder.model_name       # "SeqVAEConfig"
    encoder.latent_dim       # 100
    encoder.nuisance_indices # list or None
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

PathLike = Union[str, Path]


# ── Standalone encoder architecture (no legacy.vae.* deps) ───────────────────

class _EncoderOutput:
    """Minimal output container matching the pythae/legacy convention."""
    def __init__(self, embedding: torch.Tensor, log_covariance: torch.Tensor | None):
        self.embedding = embedding
        self.log_covariance = log_covariance


class _LegacyConvEncoder(nn.Module):
    """Standalone reconstruction of the custom conv encoder from encoder_reformatted.pt.

    Architecture derived from state dict key shapes (verified against model_config.json
    input_dim=[1, 288, 128], latent_dim=100):
      - 5 conv blocks: Conv2d(4×4, stride=2, padding=1) → BatchNorm2d → ReLU
        channels 1→16→32→64→128→256; spatial 288×128 → 9×4 → flatten=9216
      - embedding0: Linear(9216, latent_dim)
      - embedding:  Linear(latent_dim, latent_dim)  [second mu refinement layer]
      - log_var:    Linear(9216, latent_dim)
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        channels = [1, 16, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ]
        self.conv_layers = nn.Sequential(*blocks)
        flat_dim = 9 * 4 * 256  # = 9216
        self.embedding0 = nn.Linear(flat_dim, latent_dim)
        self.embedding = nn.Linear(latent_dim, latent_dim, bias=False)
        self.log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> _EncoderOutput:
        h = self.conv_layers(x)
        h_flat = h.flatten(start_dim=1)
        mu = self.embedding(self.embedding0(h_flat))
        logvar = self.log_var(h_flat)
        return _EncoderOutput(embedding=mu, log_covariance=logvar)


# ── Public wrapper ─────────────────────────────────────────────────────────────

class LegacyVaeEncoder:
    """Thin inference-only wrapper around the loaded legacy conv encoder.

    Only ``legacy_vae_inference_loader.py`` constructs this. Callers receive it from
    ``load_legacy_vae_encoder()`` and call ``encode_batch()`` — nothing else leaks out.
    """

    def __init__(self, model: nn.Module, cfg_dict: dict, *, device: str = "cpu") -> None:
        self.model_name: str | None = cfg_dict.get("name")
        self.latent_dim: int | None = cfg_dict.get("latent_dim")
        nuisance = cfg_dict.get("nuisance_indices")
        if hasattr(nuisance, "tolist"):
            nuisance = nuisance.tolist()
        self.nuisance_indices: list | None = nuisance
        self._model = model
        self._device = device

    def encode_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Forward the encoder on a batched input tensor.

        Args:
            x: ``[B, C, H, W]`` float32 tensor, already on the correct device.

        Returns:
            dict with keys:
                ``"mu"``     — ``[B, latent_dim]`` mean embedding (always present).
                ``"logvar"`` — ``[B, latent_dim]`` log-variance, or ``None`` if absent.
        """
        encoder_output = self._model(x)

        mu = getattr(encoder_output, "embedding", None)
        logvar = getattr(encoder_output, "log_covariance", None)

        if mu is None:
            mu = getattr(encoder_output, "mu", None)
            logvar = getattr(encoder_output, "log_var", None)

        if mu is None:
            raise ValueError(
                "Encoder output has neither '.embedding' nor '.mu'. "
                f"Available: {[k for k in dir(encoder_output) if not k.startswith('_')]}"
            )

        return {
            "mu": mu.detach().cpu(),
            "logvar": logvar.detach().cpu() if logvar is not None else None,
        }


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_legacy_vae_encoder(model_dir: PathLike, *, device: str = "cpu") -> LegacyVaeEncoder:
    """Load the legacy VAE encoder from a saved model directory.

    Reads ``model_config.json`` to get metadata, then loads weights from
    ``encoder_reformatted.pt`` (a plain state dict — no legacy pickle deps).

    Does NOT import ``legacy.vae.*`` anywhere. Safe to call from Python 3.9 or 3.10
    as long as the weights file exists.

    Args:
        model_dir: Path to the saved model directory (contains ``model_config.json``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        A ``LegacyVaeEncoder`` ready for inference.

    Raises:
        FileNotFoundError: if ``model_config.json`` or ``encoder_reformatted.pt`` is missing.
        ValueError: if the config names an unexpected model type or the state dict
                    doesn't match the reconstructed architecture.
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {model_dir}")

    with open(config_path) as f:
        cfg_dict = json.load(f)

    model_name = cfg_dict.get("name")
    if model_name != "SeqVAEConfig":
        raise ValueError(
            f"Expected a SeqVAEConfig checkpoint, got {model_name!r}. "
            f"Only SeqVAEConfig models are supported by this loader."
        )

    weights_path = model_dir / "encoder_reformatted.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"encoder_reformatted.pt not found in {model_dir}. "
            f"This file is required (encoder.pkl is unloadable outside the original training env)."
        )

    latent_dim = cfg_dict.get("latent_dim")
    if latent_dim is None:
        raise ValueError(f"model_config.json missing 'latent_dim' field in {model_dir}")

    encoder = _LegacyConvEncoder(latent_dim=latent_dim)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # state_dict uses conv_layers.* indexing that includes ReLU (no-param layers),
    # so there are gaps in the indices (0,1,[2=ReLU],3,4,...). The Sequential we
    # constructed has the same indices because ReLU is at positions 2,5,8,11,14.
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.to(device)

    return LegacyVaeEncoder(encoder, cfg_dict, device=device)
