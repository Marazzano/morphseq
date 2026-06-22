"""Pure encode loop — loaded encoder + SnipInputs → latents DataFrame.

No I/O, no model loading. The caller (``entrypoint.py``) is responsible for loading
the model and providing SnipInput lists. This function owns only the encode math.

Encoder output conventions handled:
- VAE / MetricVAE: ``encoder_output.embedding`` (mu), ``encoder_output.log_covariance``
- SeqVAE variants:  ``encoder_output.mu``,             ``encoder_output.log_var``

Column naming: ``z_mu_00``, ``z_mu_01``, ... (zero-padded to 2 digits).
``z_sigma_*`` columns are added when ``log_covariance`` / ``log_var`` is present.

**Important:** ``z_sigma_*`` values are taken directly from ``log_covariance`` /
``log_var`` and are **not** transformed to standard deviation. The column name
preserves the legacy naming convention from ``extract_embeddings_legacy``; the values
are log-variance, not sigma. Downstream consumers should be aware of this convention.

Row order in the output DataFrame matches the order of ``snip_inputs`` (shuffle=False).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch

from data_pipeline.feature_extraction.legacy_embeddings.snip_source import SnipInput
from data_pipeline.feature_extraction.legacy_embeddings.transforms import snip_to_model_input_tensor


def _extract_mu_logvar(encoder_output) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (mu, log_var_or_none) from an encoder output, handling both conventions."""
    # VAE / MetricVAE convention
    mu = getattr(encoder_output, "embedding", None)
    log_var = getattr(encoder_output, "log_covariance", None)

    if mu is None:
        # SeqVAE / alternate convention
        mu = getattr(encoder_output, "mu", None)
        log_var = getattr(encoder_output, "log_var", None)

    if mu is None:
        raise ValueError(
            "Encoder output has neither '.embedding' nor '.mu' attribute. "
            f"Available attributes: {[k for k in dir(encoder_output) if not k.startswith('_')]}"
        )

    return mu, log_var


def encode_snips(
    lit_model,
    snip_inputs: List[SnipInput],
    model_input_shape: tuple[int, int],
    model_input_channels: int = 1,
    batch_size: int = 64,
    device: str = "cpu",
) -> pd.DataFrame:
    """Encode snip_inputs with the loaded model; return a latents DataFrame.

    Args:
        lit_model: Loaded legacy AutoModel, already ``.eval()`` and on ``device``.
        snip_inputs: Ordered list of SnipInputs to encode (order preserved in output).
        model_input_shape: ``(height, width)`` — passed to ``snip_to_model_input_tensor``.
        model_input_channels: Channel count the model expects. ``1`` = grayscale
            (default, matching the legacy VAE ``input_dim=(1, 288, 128)``). ``3`` = RGB.
        batch_size: Images per forward pass.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        DataFrame with columns ``snip_id``, ``z_mu_00``, ``z_mu_01``, ..., and
        optionally ``z_sigma_00``, ``z_sigma_01``, ... (when the encoder emits
        log-variance; values are log-variance, not standard deviation).
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
            x = torch.stack(tensors, dim=0).to(device)  # [B, 3, H, W]

            encoder_output = lit_model.encoder(x)
            mu, log_var = _extract_mu_logvar(encoder_output)

            mu_np = mu.cpu().numpy()
            log_var_np = log_var.cpu().numpy() if log_var is not None else None

            latent_dim = mu_np.shape[1]

            for i, si in enumerate(batch_inputs):
                row: dict = {"snip_id": si.snip_id}
                for j in range(latent_dim):
                    row[f"z_mu_{j:02d}"] = float(mu_np[i, j])
                if log_var_np is not None:
                    for j in range(latent_dim):
                        row[f"z_sigma_{j:02d}"] = float(log_var_np[i, j])
                rows.append(row)

    return pd.DataFrame(rows)
