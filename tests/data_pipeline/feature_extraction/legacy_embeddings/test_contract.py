"""Tests for legacy_embeddings.contract — latent table schema validator."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_extraction.legacy_embeddings.contract import (
    EMBEDDING_MODEL_NAME_COL,
    validate_latent_embeddings,
)


def _make_df(n: int = 3, latent_dim: int = 4, include_sigma: bool = False) -> pd.DataFrame:
    snip_ids = [f"snip_{i:03d}" for i in range(n)]
    data = {"snip_id": snip_ids, EMBEDDING_MODEL_NAME_COL: ["test_model"] * n}
    for j in range(latent_dim):
        data[f"z_mu_{j:02d}"] = np.random.randn(n).astype(np.float32)
    if include_sigma:
        for j in range(latent_dim):
            data[f"z_sigma_{j:02d}"] = np.random.randn(n).astype(np.float32)
    return pd.DataFrame(data)


class TestValidLatentTables:
    def test_valid_mu_only_table_passes(self):
        validate_latent_embeddings(_make_df())

    def test_valid_mu_and_sigma_table_passes(self):
        validate_latent_embeddings(_make_df(include_sigma=True))

    def test_source_label_included_in_error(self):
        df = _make_df()
        df = df.drop(columns=["snip_id"])
        with pytest.raises(ValueError, match="my_well.parquet"):
            validate_latent_embeddings(df, source="my_well.parquet")


class TestSnipIdViolations:
    def test_missing_snip_id_raises(self):
        df = _make_df().drop(columns=["snip_id"])
        with pytest.raises(ValueError, match="snip_id"):
            validate_latent_embeddings(df)

    def test_null_snip_id_raises(self):
        df = _make_df()
        df.loc[0, "snip_id"] = None
        with pytest.raises(ValueError, match="null"):
            validate_latent_embeddings(df)

    def test_duplicate_snip_id_raises(self):
        df = _make_df()
        df.loc[1, "snip_id"] = df.loc[0, "snip_id"]
        with pytest.raises(ValueError, match="duplicate"):
            validate_latent_embeddings(df)


class TestProvenanceViolations:
    def test_missing_model_name_raises(self):
        df = _make_df().drop(columns=[EMBEDDING_MODEL_NAME_COL])
        with pytest.raises(ValueError, match=EMBEDDING_MODEL_NAME_COL):
            validate_latent_embeddings(df)

    def test_empty_model_name_raises(self):
        df = _make_df()
        df.loc[0, EMBEDDING_MODEL_NAME_COL] = ""
        with pytest.raises(ValueError, match=EMBEDDING_MODEL_NAME_COL):
            validate_latent_embeddings(df)


class TestZMuViolations:
    def test_no_z_mu_columns_raises(self):
        df = pd.DataFrame({"snip_id": ["a", "b"], EMBEDDING_MODEL_NAME_COL: ["m", "m"]})
        with pytest.raises(ValueError, match="z_mu_"):
            validate_latent_embeddings(df)

    def test_z_mu_nan_raises(self):
        df = _make_df()
        df.loc[0, "z_mu_00"] = float("nan")
        with pytest.raises(ValueError, match="null"):
            validate_latent_embeddings(df)

    def test_z_mu_nonnumeric_raises(self):
        df = _make_df()
        df["z_mu_00"] = df["z_mu_00"].astype(str)
        with pytest.raises(ValueError, match="not numeric"):
            validate_latent_embeddings(df)


class TestZSigmaViolations:
    def test_z_sigma_nan_raises_when_present(self):
        df = _make_df(include_sigma=True)
        df.loc[0, "z_sigma_00"] = float("nan")
        with pytest.raises(ValueError, match="null"):
            validate_latent_embeddings(df)

    def test_z_sigma_nonnumeric_raises_when_present(self):
        df = _make_df(include_sigma=True)
        df["z_sigma_00"] = df["z_sigma_00"].astype(str)
        with pytest.raises(ValueError, match="not numeric"):
            validate_latent_embeddings(df)

    def test_no_z_sigma_columns_is_fine(self):
        df = _make_df(include_sigma=False)
        validate_latent_embeddings(df)
