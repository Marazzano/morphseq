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
from data_pipeline.feature_extraction.legacy_embeddings.entrypoint import (
    _derive_output_parquets,
    run_legacy_embeddings,
)

LATENT_DIM = 6
MODEL_INPUT_SHAPE = (16, 8)  # (H, W) matching the fixtures
MODEL_NAME = "test_legacy_vae"

_LOADER = "data_pipeline.feature_extraction.legacy_embeddings.entrypoint.load_legacy_vae_encoder"
_RESOLVE = "data_pipeline.feature_extraction.legacy_embeddings.entrypoint.resolve_legacy_model_dir"


class _FakeEncoder:
    """EncoderProtocol stand-in; counts how many times it was constructed via the loader."""

    latent_dim = LATENT_DIM

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
            "processed_snip_path": f"{well_id}/{png.name}",  # relative to output_root
            "is_valid_snip": True,
        })
    csv = well_dir / f"{well_id}_snip_inventory.csv"
    pd.DataFrame(
        rows,
        columns=["snip_id", "processed_snip_path", "is_valid_snip"],
    ).to_csv(csv, index=False)
    return csv


def test_run_legacy_embeddings_writes_one_validated_parquet_per_well(tmp_path):
    csv_a = _make_well(tmp_path, "20250912_B01", n_snips=3)
    csv_b = _make_well(tmp_path, "20250912_C01", n_snips=2)
    out_a = tmp_path / "out" / "20250912_B01_latents.parquet"
    out_b = tmp_path / "out" / "20250912_C01_latents.parquet"
    completion_flag = tmp_path / "out" / "batch_complete.validated"
    completion_flag.parent.mkdir(parents=True)
    completion_flag.write_text("stale\n")
    out_b.parent.mkdir(parents=True, exist_ok=True)
    out_b.with_name(out_b.name + ".validated").write_text("stale\n")

    with patch(_RESOLVE, return_value=tmp_path / "fake_model_dir") as mock_resolve, \
         patch(_LOADER, return_value=_FakeEncoder()) as mock_loader:
        run_legacy_embeddings(
            snip_inventory_csvs=[csv_a, csv_b],
            output_parquets=[out_a, out_b],
            output_root=tmp_path,
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
            model_input_channels=1,
            batch_size=2,
            device="cpu",
            completion_flag=completion_flag,
        )

    # Model loaded exactly once for the whole batch (the entire point of RUN_BATCH).
    assert mock_loader.call_count == 1
    assert mock_resolve.call_count == 1
    assert completion_flag.read_text() == "ok\n"

    for out, csv in ((out_a, csv_a), (out_b, csv_b)):
        assert out.exists(), f"missing shard {out}"
        assert out.with_name(out.name + ".validated").read_text() == "ok\n"
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
            output_root=tmp_path,
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
        )


def test_derive_output_parquets_uses_canonical_per_well_paths(tmp_path):
    inventories = [
        tmp_path / "snips" / "per_well" / "20250912_A01" / "inventory.csv",
        tmp_path / "snips" / "per_well" / "20250912_B02" / "inventory.csv",
    ]
    outputs = _derive_output_parquets(
        snip_inventory_csvs=inventories,
        output_root=tmp_path / "output",
        experiment_id="20250912",
    )
    assert [path.name for path in outputs] == [
        "20250912_A01_latents.parquet",
        "20250912_B02_latents.parquet",
    ]
    assert [path.parent.name for path in outputs] == [
        "20250912_A01",
        "20250912_B02",
    ]


def test_completion_flag_is_not_written_when_a_later_well_fails(tmp_path):
    csv_a = _make_well(tmp_path, "20250912_B01", n_snips=1)
    missing_csv = tmp_path / "missing_snip_inventory.csv"
    out_a = tmp_path / "out" / "20250912_B01_latents.parquet"
    out_b = tmp_path / "out" / "20250912_C01_latents.parquet"
    completion_flag = tmp_path / "out" / "batch_complete.validated"
    completion_flag.parent.mkdir(parents=True)
    completion_flag.write_text("stale\n")
    out_b.with_name(out_b.name + ".validated").write_text("stale\n")

    with patch(_RESOLVE, return_value=tmp_path / "fake_model_dir"), \
         patch(_LOADER, return_value=_FakeEncoder()), \
         pytest.raises(FileNotFoundError):
        run_legacy_embeddings(
            snip_inventory_csvs=[csv_a, missing_csv],
            output_parquets=[out_a, out_b],
            output_root=tmp_path,
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
            completion_flag=completion_flag,
        )

    assert out_a.exists()
    assert out_a.with_name(out_a.name + ".validated").exists()
    assert not completion_flag.exists()
    assert not out_b.with_name(out_b.name + ".validated").exists()


def test_run_legacy_embeddings_writes_and_merges_empty_well(tmp_path):
    csv_nonempty = _make_well(tmp_path, "20250912_B01", n_snips=2)
    csv_empty = _make_well(tmp_path, "20250912_F06", n_snips=0)
    out_nonempty = tmp_path / "out" / "20250912_B01_latents.parquet"
    out_empty = tmp_path / "out" / "20250912_F06_latents.parquet"

    with patch(_RESOLVE, return_value=tmp_path / "fake_model_dir"), \
         patch(_LOADER, return_value=_FakeEncoder()):
        run_legacy_embeddings(
            snip_inventory_csvs=[csv_nonempty, csv_empty],
            output_parquets=[out_nonempty, out_empty],
            output_root=tmp_path,
            models_root=tmp_path / "models",
            model_name=MODEL_NAME,
            model_input_shape=MODEL_INPUT_SHAPE,
            device="cpu",
        )

    empty = pd.read_parquet(out_empty)
    assert empty.empty
    validate_latent_embeddings(empty, source=str(out_empty))

    merged = pd.concat(
        [pd.read_parquet(out_nonempty), empty],
        ignore_index=True,
    )
    validate_latent_embeddings(merged, source="merged latent embeddings")
    assert len(merged) == 2
