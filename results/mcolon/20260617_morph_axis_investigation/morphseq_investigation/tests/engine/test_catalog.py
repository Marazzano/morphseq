"""TASK_A — DistributionCatalog: from_dataframe / pool_by / compare / id-helpers."""

import logging

import numpy as np
import pandas as pd
import pytest

from morphseq_investigation.engine.catalog import (
    DistributionCatalog,
    DistributionComparison,
    DistributionComparisons,
)
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


# --------------------------------------------------------------------------- #
# pool_by
# --------------------------------------------------------------------------- #
def test_pool_by_collapses_coordinate_and_unions_samples():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    assert len(catalog.distributions) == 8  # 2x2x2

    pooled = catalog.pool_by("experiment")
    assert pooled.coordinate_names == ("time_bin", "genotype")
    assert len(pooled.distributions) == 4  # 2x2

    # samples union: each pooled distribution has 2x the per-cell rows.
    for distribution in pooled.distributions:
        assert len(distribution.sample_ids) == 6  # 2 experiments x 3 rows
        assert "experiment" not in distribution.coordinates


def test_pool_by_not_a_coordinate_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError):
        catalog.pool_by("experiment")


def test_pool_by_duplicate_sample_id_raises():
    df = _synthetic_df()
    # Force a duplicate embryo_id across experiments within the same time/genotype cell.
    df.loc[df["experiment"] == "expB", "embryo_id"] = df.loc[
        df["experiment"] == "expA", "embryo_id"
    ].to_numpy()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        catalog.pool_by("experiment")


def test_pool_by_labels_ride_along_per_sample():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    pooled = catalog.pool_by("experiment")
    for distribution in pooled.distributions:
        col = distribution.label_column("phenotype")
        assert set(col.values) == set(distribution.sample_ids)


def test_pool_by_records_softened_provenance_note_on_distribution():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    pooled = catalog.pool_by("experiment")
    # The note lives ON the pooled Distribution so it travels with the object
    # outside its catalog (into plotting / cross-population compare).
    for distribution in pooled.distributions:
        assert distribution.pooled_coordinates == ("experiment",)


def test_pool_by_note_accumulates_across_repeated_pooling():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    pooled = catalog.pool_by("experiment").pool_by("genotype")
    for distribution in pooled.distributions:
        assert set(distribution.pooled_coordinates) == {"experiment", "genotype"}


def test_compare_across_pooled_away_coordinate_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    pooled = catalog.pool_by("experiment")
    with pytest.raises(ValueError, match="collapsed by pool_by"):
        pooled.compare(across="experiment")


def test_compare_match_on_pooled_away_coordinate_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    pooled = catalog.pool_by("experiment")
    with pytest.raises(ValueError, match="pooled-away coordinate"):
        pooled.compare(across="genotype", match_on=("time_bin", "experiment"))


# --------------------------------------------------------------------------- #
# compare()
# --------------------------------------------------------------------------- #
def test_compare_default_match_on_infers_all_other_coordinates(caplog):
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with caplog.at_level(logging.INFO):
        comparisons = catalog.compare(across="genotype")
    assert comparisons.match_on == ("time_bin",)
    assert comparisons.across == "genotype"
    assert set(comparisons.values) == {"wildtype", "b9d2"}
    assert len(comparisons.comparisons) == 2  # one per time_bin
    assert any("matching on" in message for message in caplog.messages)


def test_compare_single_split_coordinate_gives_one_global_comparison():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("genotype",),
    )
    comparisons = catalog.compare(across="genotype")
    assert comparisons.match_on == ()
    assert len(comparisons.comparisons) == 1
    assert comparisons.comparisons[0].coordinates == {}


def test_compare_preserves_requested_values_order():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    comparisons = catalog.compare(across="genotype", values=("b9d2", "wildtype"))
    assert comparisons.values == ("b9d2", "wildtype")
    for comparison in comparisons.comparisons:
        assert tuple(comparison.members) == ("b9d2", "wildtype")


def test_compare_missing_member_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError, match="missing member"):
        catalog.compare(across="genotype", values=("wildtype", "nonexistent"))


def test_compare_duplicate_member_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    # match_on defaults to (time_bin, experiment); across=genotype -> unique per group, OK.
    # Force duplicates by omitting experiment from match_on explicitly.
    with pytest.raises(ValueError, match="Cannot omit coordinate"):
        catalog.compare(across="genotype", match_on=("time_bin",))


def test_compare_across_is_label_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError, match="label group, not a coordinate"):
        catalog.compare(across="phenotype")


def test_compare_omitted_but_constant_coordinate_is_allowed():
    df = _synthetic_df()
    # Make experiment constant per (time_bin, genotype) cell by only keeping expA.
    df = df[df["experiment"] == "expA"].copy()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype", "experiment"),
    )
    # experiment is constant ("expA") everywhere, so omitting it from match_on
    # is a valid assertion, not pooling.
    comparisons = catalog.compare(across="genotype", match_on=("time_bin",))
    assert comparisons.match_on == ("time_bin",)
    assert len(comparisons.comparisons) == 2


def test_compare_not_a_coordinate_raises():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    with pytest.raises(ValueError):
        catalog.compare(across="nonexistent_coordinate")


def _comparison_member(name):
    from morphseq_investigation.engine.objects import Distribution
    from morphseq_investigation.engine.identifiers import make_distribution_id

    coords = {"time_bin": 14, "genotype": name}
    return Distribution(
        distribution_id=make_distribution_id(coords),
        sample_ids=(f"{name}_s0",),
        feature_names=("PC1",),
        feature_values=np.zeros((1, 1)),
        coordinates=coords,
    )


def test_distribution_comparisons_invariant_enforced():
    d_wt = _comparison_member("wildtype")
    d_b9 = _comparison_member("b9d2")
    with pytest.raises(ValueError):
        DistributionComparisons(
            comparisons=(
                DistributionComparison(
                    coordinates={"time_bin": 14}, members={"wildtype": d_wt, "b9d2": d_b9}
                ),
            ),
            across="genotype",
            values=("wildtype",),  # mismatched vs actual members -> should raise
            match_on=("time_bin",),
        )


# --------------------------------------------------------------------------- #
# label_groups (PATH A convenience)
# --------------------------------------------------------------------------- #
def test_label_groups_one_per_distribution():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        label_columns=("phenotype",),
        split_columns=("time_bin", "genotype"),
    )
    groups = catalog.label_groups("phenotype", display_name="Phenotype")
    assert len(groups) == 4
    for group in groups:
        assert group.label_name == "phenotype"
        assert group.display_name == "Phenotype"


def test_label_groups_skips_distributions_missing_the_label():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    groups = catalog.label_groups("phenotype")
    assert groups == ()


# --------------------------------------------------------------------------- #
# with_labels / map_distributions / detect_peaks conveniences
# --------------------------------------------------------------------------- #
def test_with_labels_attaches_after_construction():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    label_df = df[["embryo_id", "phenotype"]].copy()
    labeled = catalog.with_labels(label_df, ["phenotype"])
    for distribution in labeled.distributions:
        assert "phenotype" in distribution.labels


def test_map_distributions_generic_escape_hatch():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    mapped = catalog.map_distributions(
        lambda d: d.with_label("constant", {sid: "x" for sid in d.sample_ids})
    )
    for distribution in mapped.distributions:
        assert "constant" in distribution.labels


def test_detect_peaks_is_thin_wrapper_over_map_distributions():
    df = _synthetic_df()
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=("PC1", "PC2"),
        split_columns=("time_bin", "genotype"),
    )
    # TASK_B implemented Distribution.detect_peaks; the catalog convenience
    # delegates to map_distributions and writes the output label onto every
    # distribution, returning a new catalog (frozen — new object).
    result = catalog.detect_peaks(features=("PC1", "PC2"), output_label="resolved_peak")
    assert isinstance(result, DistributionCatalog)
    assert result is not catalog
    for distribution in result.distributions:
        assert "resolved_peak" in distribution.labels
