"""TASK_A — DistributionCatalog: from_dataframe / coordinate index / lookup."""

import pandas as pd
import pytest

from morphseq_investigation.engine.catalog import DistributionCatalog
from morphseq_investigation.engine.objects import UNASSIGNED_LABEL


def _synthetic_df(n_per_cell=3):
    """time_bin x genotype x experiment grid, 2 features, an extra label col."""
    rows = []
    i = 0
    for time_bin in (14, 30):
        for genotype in ("wildtype", "b9d2"):
            for experiment in ("expA", "expB"):
                for _ in range(n_per_cell):
                    rows.append(
                        {
                            "embryo_id": f"e{i}",
                            "time_bin": time_bin,
                            "genotype": genotype,
                            "experiment": experiment,
                            "PC1": float(i),
                            "PC2": float(i) * 2.0,
                            "phenotype": "affected" if i % 2 == 0 else "unaffected",
                        }
                    )
                    i += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# from_dataframe
# --------------------------------------------------------------------------- #
def test_from_dataframe_splits_one_distribution_per_coordinate_combo():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    # 2 time_bin x 2 genotype = 4 distributions.
    assert len(catalog.distributions) == 4
    seen_coords = {
        (d.coordinates["time_bin"], d.coordinates["genotype"]) for d in catalog.distributions
    }
    assert seen_coords == {
        (14, "wildtype"),
        (14, "b9d2"),
        (30, "wildtype"),
        (30, "b9d2"),
    }


def test_from_dataframe_attaches_labels_best_effort():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    for distribution in catalog.distributions:
        assert "phenotype" in distribution.labels
        col = distribution.label_column("phenotype")
        assert set(col.values.values()) <= {"affected", "unaffected", UNASSIGNED_LABEL}


def test_from_dataframe_missing_label_value_becomes_unassigned():
    df = _synthetic_df()
    df.loc[df.index[0], "phenotype"] = None
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    missing_row = df.iloc[0]
    dist_id = catalog.resolve_id(
        time_bin=missing_row["time_bin"], genotype=missing_row["genotype"]
    )
    distribution = next(d for d in catalog.distributions if d.distribution_id == dist_id)
    col = distribution.label_column("phenotype")
    assert col.values[missing_row["embryo_id"]] == UNASSIGNED_LABEL


def test_from_dataframe_no_split_columns_yields_one_distribution():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df, sample_id_column="embryo_id", feature_columns=("PC1", "PC2")
    )
    assert len(catalog.distributions) == 1
    assert catalog.distributions[0].coordinates == {}
    assert catalog.coordinate_names == ()


# --------------------------------------------------------------------------- #
# to_index_dataframe / find_ids / resolve_id
# --------------------------------------------------------------------------- #
def test_to_index_dataframe_shape():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    index_df = catalog.to_index_dataframe()
    assert len(index_df) == 4
    assert set(index_df.columns) == {"distribution_id", "time_bin", "genotype"}
    assert index_df["distribution_id"].nunique() == 4


def test_find_ids_hit_and_none():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    hits = catalog.find_ids(time_bin=14)
    assert len(hits) == 2
    none = catalog.find_ids(time_bin=999)
    assert none == ()


def test_resolve_id_exactly_one():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    dist_id = catalog.resolve_id(time_bin=14, genotype="wildtype")
    assert dist_id in {d.distribution_id for d in catalog.distributions}


def test_resolve_id_ambiguous_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError):
        catalog.resolve_id(time_bin=14)


def test_resolve_id_none_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError):
        catalog.resolve_id(time_bin=999)


def test_find_ids_label_kwarg_raises_helpful_message():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError, match="label group, not a coordinate"):
        catalog.find_ids(phenotype="affected")


def test_resolve_id_label_kwarg_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError, match="label group, not a coordinate"):
        catalog.resolve_id(phenotype="affected")
