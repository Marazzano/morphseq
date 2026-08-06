"""Pure encode loop — loaded encoder + SnipInputs → latents DataFrame.

No I/O, no model loading. The caller is responsible for loading the encoder and
providing SnipInput lists. This function owns only the encode math.

The encoder is typed as ``EncoderProtocol`` — any object with an ``encode_batch``
method that returns ``{"mu": tensor, "logvar": tensor_or_None}`` satisfies it.
``LegacyVaeEncoder`` from ``legacy_vae_inference_loader`` satisfies the protocol;
so do test fakes. ``encode.py`` does not import the adapter.

Column naming: ``z_mu_00``, ``z_mu_01``, ... (zero-padded to 2 digits) when the encoder has
no ``nuisance_indices``. When the encoder does expose non-empty ``nuisance_indices`` (a
disentangled model — e.g. SeqVAE), columns are instead split by raw latent index into
``z_mu_b_NN`` (biological — indices not in ``nuisance_indices``) and ``z_mu_n_NN`` (nuisance —
indices in ``nuisance_indices``), matching the legacy ``assess_vae_results.py`` convention.
``z_sigma_*``/``z_sigma_b_*``/``z_sigma_n_*`` columns are added the same way when the encoder
emits ``"logvar"``.

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

    latent_dim: int
    nuisance_indices: list[int] | None

    def encode_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Run inference on a batch.

        Args:
            x: ``[B, C, H, W]`` float32 tensor on the encoder's device.

        Returns:
            dict with keys ``"mu"`` (required) and ``"logvar"`` (optional, may be None).
        """
        ...


def _latent_column_names(latent_dim: int, nuisance_indices: list[int] | None, prefix: str) -> list[str]:
    """Column names for one latent family (``z_mu`` or ``z_sigma``), by raw index.

    Flat ``{prefix}_NN`` when there's no disentanglement; ``{prefix}_b_NN``/``{prefix}_n_NN``
    (biological/nuisance) when ``nuisance_indices`` is a non-empty list.
    """
    if not nuisance_indices:
        return [f"{prefix}_{j:02d}" for j in range(latent_dim)]
    nuisance_set = set(nuisance_indices)
    return [
        f"{prefix}_n_{j:02d}" if j in nuisance_set else f"{prefix}_b_{j:02d}"
        for j in range(latent_dim)
    ]


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
    if not snip_inputs:
        latent_dim = getattr(encoder, "latent_dim", None)
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError(
                "Cannot construct an empty latent-embeddings shard because the loaded "
                "encoder does not expose a positive integer latent_dim."
            )
        nuisance_indices = getattr(encoder, "nuisance_indices", None)
        mu_cols = _latent_column_names(latent_dim, nuisance_indices, "z_mu")
        empty = {"snip_id": pd.Series(dtype="string")}
        empty.update({col: pd.Series(dtype="float32") for col in mu_cols})
        return pd.DataFrame(empty)

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
            nuisance_indices = getattr(encoder, "nuisance_indices", None)
            mu_cols = _latent_column_names(latent_dim, nuisance_indices, "z_mu")
            sigma_cols = _latent_column_names(latent_dim, nuisance_indices, "z_sigma")

            for i, si in enumerate(batch_inputs):
                row: dict = {"snip_id": si.snip_id}
                for j in range(latent_dim):
                    row[mu_cols[j]] = float(mu_np[i, j])
                if logvar_np is not None:
                    for j in range(latent_dim):
                        row[sigma_cols[j]] = float(logvar_np[i, j])
                rows.append(row)

    return pd.DataFrame(rows)
