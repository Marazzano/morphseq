"""Tests for legacy_embeddings.entrypoint — the 3.9 batch encode body.

Mocks the model loader (no real weights needed) and writes synthetic per-well snip_inventory
CSVs + PNG fixtures, then asserts run_legacy_embeddings loads the encoder ONCE and writes one
valid latents parquet per well (provenance-stamped, contract-validated).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from data_pipeline.feature_extraction.legacy_embeddings.contract import (
    EMBEDDING_MODEL_NAME_COL,
    validate_latent_embeddings,
)
from data_pipeline.feature_extraction.legacy_embeddings.entrypoint import run_legacy_embeddings

LATENT_DIM = 6
MODEL_INPUT_SHAPE = (16, 8)  # (H, W) matching the fixtures
MODEL_NAME = "test_legacy_vae"

_LOADER = "data_pipeline.feature_extraction.legacy_embeddings.entrypoint.load_legacy_vae_encoder"
_RESOLVE = "data_pipeline.feature_extraction.legacy_embeddings.entrypoint.resolve_legacy_model_dir"


class _FakeEncoder:
    """EncoderProtocol stand-in; counts how many times it was constructed via the loader."""

    def encode_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        b = x.shape[0]
        mu = torch.arange(b * LATENT_DIM, dtype=torch.float32).reshape(b, LATENT_DIM)
        return {"mu": mu, "logvar": None}


def _make_well(tmp_path: Path, well_id: str, n_snips: int) -> Path:
    """Write n PNG fixtures + a snip_inventory CSV (paths relative to the CSV dir). Return CSV path."""
    well_dir = tmp_path / well_id
    well_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_snips):
        arr = np.full((16, 8, 3), (i + 1) * 8, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        png = well_dir / f"{well_id}_s{i:02d}.png"
        png.write_bytes(buf.getvalue())
        rows.append({
            "snip_id": f"{well_id}_s{i:02d}",
            "processed_snip_path": png.name,  # relative to the CSV's own directory
            "is_valid_snip": True,
        })
    csv = well_dir / f"{well_id}_snip_inventory.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_run_legacy_embeddings_writes_one_validated_parquet_per_well(tmp_path):
    csv_a = _make_well(tmp_path, "20250912_B01", n_snips=3)
    csv_b = _make_well(tmp_path, "20250912_C01", n_snips=2)
    out_a = tmp_path / "out" / "20250912_B01_latents.parquet"
    out_b = tmp_path / "out" / "20250912_C01_latents.parquet"

    with patch(_RESOLVE, return_value=tmp_path / "fake_model_dir") as mock_resolve, \
         patch(_LOADER, return_value=_FakeEncoder()) as mock_loader:
        run_legacy_embeddings(
            snip_inventory_csvs=[csv_a, csv_b],
            output_parquets=[out_a, out_b],
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
            model_input_channels=1,
            batch_size=2,
            device="cpu",
        )

    # Model loaded exactly once for the whole batch (the entire point of RUN_BATCH).
    assert mock_loader.call_count == 1
    assert mock_resolve.call_count == 1

    for out, csv in ((out_a, csv_a), (out_b, csv_b)):
        assert out.exists(), f"missing shard {out}"
        df = pd.read_parquet(out)
        validate_latent_embeddings(df, source=str(out))  # passes the full contract
        expected_ids = list(pd.read_csv(csv)["snip_id"])
        assert list(df["snip_id"]) == expected_ids  # row order preserved
        assert (df[EMBEDDING_MODEL_NAME_COL] == MODEL_NAME).all()  # provenance stamped


def test_run_legacy_embeddings_mismatched_input_output_counts_raises(tmp_path):
    csv_a = _make_well(tmp_path, "20250912_B01", n_snips=1)
    with pytest.raises(ValueError, match="counts must match"):
        run_legacy_embeddings(
            snip_inventory_csvs=[csv_a],
            output_parquets=[tmp_path / "a.parquet", tmp_path / "b.parquet"],
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
        )
