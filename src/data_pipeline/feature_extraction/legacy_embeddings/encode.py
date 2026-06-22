"""Pure encode loop — loaded encoder + SnipInputs → latents DataFrame.

No I/O, no model loading. The caller is responsible for loading the encoder and
providing SnipInput lists. This function owns only the encode math.

The encoder is typed as ``EncoderProtocol`` — any object with an ``encode_batch``
method that returns ``{"mu": tensor, "logvar": tensor_or_None}`` satisfies it.
``LegacyVaeEncoder`` from ``legacy_vae_inference_loader`` satisfies the protocol;
so do test fakes. ``encode.py`` does not import the adapter.

Column naming: ``z_mu_00``, ``z_mu_01``, ... (zero-padded to 2 digits).
``z_sigma_*`` columns are added when the encoder emits ``"logvar"``.

**Important:** ``z_sigma_*`` values store ``logvar`` directly — **not** standard
deviation. The column name preserves the legacy naming convention; the values are
log-variance. Downstream consumers should be aware of this convention.

Row order in the output DataFrame matches the order of ``snip_inputs`` (shuffle=False).
"""

from __future__ import annotations

from typing import List, Protocol

import pandas as pd
import torch

from data_pipeline.feature_extraction.legacy_embeddings.snip_source import SnipInput
from data_pipeline.feature_extraction.legacy_embeddings.transforms import snip_to_model_input_tensor


class EncoderProtocol(Protocol):
    """Structural interface for any inference encoder used by ``encode_snips``."""

    def encode_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Run inference on a batch.

        Args:
            x: ``[B, C, H, W]`` float32 tensor on the encoder's device.

        Returns:
            dict with keys ``"mu"`` (required) and ``"logvar"`` (optional, may be None).
        """
        ...


def encode_snips(
    snip_inputs: List[SnipInput],
    *,
    encoder: EncoderProtocol,
    model_input_shape: tuple[int, int],
    model_input_channels: int = 1,
    batch_size: int = 64,
    device: str = "cpu",
) -> pd.DataFrame:
    """Encode snip_inputs with the loaded encoder; return a latents DataFrame.

    Args:
        snip_inputs: Ordered list of SnipInputs to encode (order preserved in output).
        encoder: Any object satisfying ``EncoderProtocol`` (keyword-only).
        model_input_shape: ``(height, width)`` — passed to ``snip_to_model_input_tensor``.
        model_input_channels: Channel count the model expects. ``1`` = grayscale
            (default, matching the legacy VAE ``input_dim=(1, 288, 128)``). ``3`` = RGB.
        batch_size: Images per forward pass.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        DataFrame with columns ``snip_id``, ``z_mu_00``, ``z_mu_01``, ..., and
        optionally ``z_sigma_00``, ``z_sigma_01``, ... (log-variance, not std dev).
        Row order matches ``snip_inputs`` order.
    """
    rows: list[dict] = []

    with torch.no_grad():
        for batch_start in range(0, len(snip_inputs), batch_size):
            batch_inputs = snip_inputs[batch_start : batch_start + batch_size]

            tensors = [
                snip_to_model_input_tensor(si.image_path, model_input_shape, model_input_channels)
                for si in batch_inputs
            ]
            x = torch.stack(tensors, dim=0).to(device)

            out = encoder.encode_batch(x)
            mu_np = out["mu"].numpy()
            logvar = out.get("logvar")
            logvar_np = logvar.numpy() if logvar is not None else None

            latent_dim = mu_np.shape[1]

            for i, si in enumerate(batch_inputs):
                row: dict = {"snip_id": si.snip_id}
                for j in range(latent_dim):
                    row[f"z_mu_{j:02d}"] = float(mu_np[i, j])
                if logvar_np is not None:
                    for j in range(latent_dim):
                        row[f"z_sigma_{j:02d}"] = float(logvar_np[i, j])
                rows.append(row)

    return pd.DataFrame(rows)
