"""Latent embeddings table contract — schema and validator.

The latents parquet produced by the encode step carries:
- ``snip_id``: stable per-snip identity (the join key); required, non-null, unique per shard.
- ``z_mu_*``: latent mean columns (at least one required); numeric, non-null.
- ``z_sigma_*``: latent log-variance columns (optional; included when the model emits them).
  Values are taken directly from ``log_covariance`` / ``log_var`` and are **not** transformed
  to standard deviation — the column name preserves legacy naming convention.

- ``embedding_model_name``: provenance — which trained model produced these vectors. Required,
  non-null. Latents without a model name are unlabeled vials; the encode entrypoint stamps it from
  the resolved ``model_name`` before validating.

No biological metadata is expected here; identity is carried entirely by ``snip_id``.
"""

from __future__ import annotations

import pandas as pd

EMBEDDING_MODEL_NAME_COL = "embedding_model_name"
REQUIRED_COLUMNS: list[str] = ["snip_id", EMBEDDING_MODEL_NAME_COL]


def validate_latent_embeddings(df: pd.DataFrame, *, source: str = "") -> None:
    """Validate a latents DataFrame against the latent embeddings contract.

    Args:
        df: The latents DataFrame to validate.
        source: Optional label (e.g. a file path) included in error messages.

    Raises:
        ValueError: on any contract violation, with a message naming the exact problem.
    """
    loc = f" (source: {source})" if source else ""

    # ── snip_id ───────────────────────────────────────────────────────────────
    if "snip_id" not in df.columns:
        raise ValueError(
            f"Latents table{loc} is missing required column 'snip_id'. "
            f"Got columns: {sorted(df.columns.tolist())}."
        )
    if df["snip_id"].isna().any():
        raise ValueError(f"Latents table{loc} has null values in 'snip_id'.")
    if df["snip_id"].duplicated().any():
        dups = df["snip_id"][df["snip_id"].duplicated()].tolist()
        raise ValueError(
            f"Latents table{loc} has duplicate snip_id values: {dups[:5]}"
            + (" ..." if len(dups) > 5 else "")
        )

    # ── embedding_model_name (provenance) ─────────────────────────────────────
    if EMBEDDING_MODEL_NAME_COL not in df.columns:
        raise ValueError(
            f"Latents table{loc} is missing required column '{EMBEDDING_MODEL_NAME_COL}'. "
            f"The encode entrypoint stamps it from the resolved model_name. "
            f"Got columns: {sorted(df.columns.tolist())}."
        )
    if df[EMBEDDING_MODEL_NAME_COL].isna().any() or (
        df[EMBEDDING_MODEL_NAME_COL].astype(str).str.len() == 0
    ).any():
        raise ValueError(
            f"Latents table{loc} has null/empty '{EMBEDDING_MODEL_NAME_COL}' — every latent row "
            f"must name the model that produced it."
        )

    # ── z_mu_* columns ────────────────────────────────────────────────────────
    z_mu_cols = [c for c in df.columns if c.startswith("z_mu_")]
    if not z_mu_cols:
        raise ValueError(
            f"Latents table{loc} has no 'z_mu_*' columns. "
            f"At least one z_mu_* column is required."
        )
    for col in z_mu_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Latents table{loc}: column '{col}' is not numeric "
                f"(dtype={df[col].dtype})."
            )
        if df[col].isna().any():
            raise ValueError(
                f"Latents table{loc}: column '{col}' has null values."
            )

    # ── z_sigma_* columns (optional) ─────────────────────────────────────────
    z_sigma_cols = [c for c in df.columns if c.startswith("z_sigma_")]
    for col in z_sigma_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Latents table{loc}: column '{col}' is not numeric "
                f"(dtype={df[col].dtype})."
            )
        if df[col].isna().any():
            raise ValueError(
                f"Latents table{loc}: column '{col}' has null values."
            )
