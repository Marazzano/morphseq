"""TASK_0 — typed-column Distribution + derived views."""

import dataclasses

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    UNASSIGNED_LABEL,
    Distribution,
    DistributionLabelGroup,
    LabelColumn,
    LabelProvenance,
    Grid,
    DensityGrid,
    SampleSet,
    SampleSetGeometry,
    derive_label_groups,
    resolve_label_group,
    LabelGroup,
)
from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet


def _distribution(n=4, feats=("PC1", "PC2"), coords=None):
    coordinates = coords if coords is not None else {"scope_id": "b9d2", "time_bin": 30}
    return Distribution(
        distribution_id=make_distribution_id(coordinates),
        sample_ids=tuple(f"s{i}" for i in range(n)),
        feature_names=feats,
        feature_values=np.arange(n * len(feats), dtype=float).reshape(n, len(feats)),
        coordinates=coordinates,
    )


# --------------------------------------------------------------------------- #
# Distribution shape + hard invariants
# --------------------------------------------------------------------------- #
def test_distribution_minimal_construction():
    d = _distribution()
    assert d.feature_names == ("PC1", "PC2")
    assert d.feature_values.shape == (4, 2)
    assert d.coordinate("time_bin") == 30
    assert d.labels == {}


def test_distribution_has_no_role_or_coordinate_frame():
    # role is a comparison RELATIONSHIP, never a Distribution field (spec).
    fields = {f.name for f in dataclasses.fields(Distribution)}
    assert "coordinate_frame" not in fields
    assert "role" not in fields
    assert "coordinates" in fields and "labels" in fields


def test_distribution_frozen():
    d = _distribution()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.distribution_id = "other"  # type: ignore[misc]


def test_distribution_feature_values_readonly():
    d = _distribution()
    with pytest.raises(ValueError):
        d.feature_values[0, 0] = 999.0


def test_feature_values_column_invariant_enforced():
    with pytest.raises(ValueError):
        Distribution(
            distribution_id="x",
            sample_ids=("s0",),
            feature_names=("PC1", "PC2", "PC3"),
            feature_values=np.zeros((1, 2)),
        )


def test_sample_ids_row_invariant_enforced():
    with pytest.raises(ValueError):
        Distribution(
            distribution_id="x",
            sample_ids=("s0", "s1"),
            feature_names=("PC1",),
            feature_values=np.zeros((1, 1)),
        )


def test_sample_ids_must_be_unique():
    with pytest.raises(ValueError):
        Distribution(
            distribution_id="x",
            sample_ids=("s0", "s0"),
            feature_names=("PC1",),
            feature_values=np.zeros((2, 1)),
        )


def test_feature_values_column_by_name():
    d = _distribution()
    np.testing.assert_array_equal(d.feature_column("PC2"), d.feature_values[:, 1])
    with pytest.raises(KeyError):
        d.feature_column("nope")


# --------------------------------------------------------------------------- #
# with_label — attaches, normalizes coverage, returns NEW object
# --------------------------------------------------------------------------- #
def test_with_label_returns_new_object_original_unchanged():
    d = _distribution()
    d2 = d.with_label("genotype", {"s0": "wildtype", "s1": "b9d2"})
    assert "genotype" not in d.labels  # original untouched
    assert "genotype" in d2.labels
    assert d2 is not d


def test_with_label_missing_samples_become_unassigned():
    d = _distribution()
    d2 = d.with_label("genotype", {"s0": "wildtype"})
    col = d2.label_column("genotype")
    assert col.values["s0"] == "wildtype"
    assert col.values["s1"] == UNASSIGNED_LABEL
    assert col.values["s3"] == UNASSIGNED_LABEL


def test_with_label_stray_sample_raises():
    d = _distribution()
    with pytest.raises(ValueError):
        d.with_label("genotype", {"not_a_sample": "wildtype"})


# --------------------------------------------------------------------------- #
# sample_sets — DERIVED view (one per category, excludes unassigned, not stored)
# --------------------------------------------------------------------------- #
def test_sample_sets_derive_one_per_category():
    d = _distribution().with_label(
        "genotype", {"s0": "wildtype", "s1": "wildtype", "s2": "b9d2"}
    )  # s3 unassigned
    sets = d.sample_sets("genotype")
    assert [s.sample_set_name for s in sets] == ["wildtype", "b9d2"]
    assert sets[0].sample_ids == ("s0", "s1")
    assert sets[1].sample_ids == ("s2",)


def test_sample_sets_exclude_unassigned_and_cover_assigned():
    d = _distribution().with_label("genotype", {"s0": "wildtype", "s2": "b9d2"})
    sets = d.sample_sets("genotype")
    covered = {sid for s in sets for sid in s.sample_ids}
    assert covered == {"s0", "s2"}  # s1, s3 unassigned -> no set


def test_sample_sets_not_stored_on_object():
    d = _distribution().with_label("genotype", {"s0": "wildtype"})
    a = d.sample_sets("genotype")
    b = d.sample_sets("genotype")
    assert "sample_sets" not in {f.name for f in dataclasses.fields(Distribution)}
    assert [s.sample_ids for s in a] == [s.sample_ids for s in b]


def test_sample_sets_read_back_geometry_from_provenance():
    geom = SampleSetGeometry(
        grid_id="g",
        feature_names=("PC1",),
        center=np.array([0.0]),
        radius=1.0,
        r80=0.8,
        cv_radius_from_center=0.1,
    )
    prov = LabelProvenance(method="discover_modes", geometry={"peak_0": geom})
    d = _distribution().with_label(
        "resolved_peak", {"s0": "peak_0", "s1": "peak_0"}, provenance=prov
    )
    sets = d.sample_sets("resolved_peak")
    assert sets[0].geometry is geom


def test_missing_label_column_raises():
    d = _distribution()
    with pytest.raises(KeyError):
        d.sample_sets("nope")
    with pytest.raises(KeyError):
        d.coordinate("nope")


# --------------------------------------------------------------------------- #
# label_group + DistributionLabelGroup.coordinate(FacetKey)
# --------------------------------------------------------------------------- #
def test_label_group_binds_and_defaults_display_name():
    d = _distribution().with_label("genotype", {"s0": "wildtype"})
    lg = d.label_group("genotype")
    assert isinstance(lg, DistributionLabelGroup)
    assert lg.display_name == "genotype"
    assert lg.label_name == "genotype"


def test_label_group_absent_label_raises():
    d = _distribution()
    with pytest.raises(KeyError):
        d.label_group("genotype")


def test_distribution_label_group_coordinate_facet_resolution():
    d = _distribution().with_label("genotype", {"s0": "wildtype"})
    lg = d.label_group("genotype", display_name="Genotype")
    assert lg.coordinate(LabelGroupFacet()) == "Genotype"
    assert lg.coordinate(CoordinateFacet("time_bin")) == 30


def test_distribution_label_group_sample_sets_delegates():
    d = _distribution().with_label("genotype", {"s0": "wildtype", "s1": "b9d2"})
    lg = d.label_group("genotype")
    assert [s.sample_set_name for s in lg.sample_sets()] == ["wildtype", "b9d2"]


# --------------------------------------------------------------------------- #
# discover_modes — documented stub (body lands in TASK_B)
# --------------------------------------------------------------------------- #
def test_discover_modes_is_stub():
    d = _distribution()
    with pytest.raises(NotImplementedError):
        d.discover_modes(features=("PC1",), output_label="resolved_peak", spec={})


# --------------------------------------------------------------------------- #
# LabelColumn.categories excludes unassigned
# --------------------------------------------------------------------------- #
def test_label_column_categories_first_appearance_excludes_unassigned():
    col = LabelColumn(
        name="g",
        values={"s0": "b", "s1": "a", "s2": "b", "s3": UNASSIGNED_LABEL},
    )
    assert col.categories() == ("b", "a")


# --------------------------------------------------------------------------- #
# Grid / DensityGrid / SampleSet slots (kept shapes)
# --------------------------------------------------------------------------- #
def test_grid_axis_count_matches_features():
    with pytest.raises(ValueError):
        Grid(
            grid_id="g",
            feature_names=("PC1", "PC2"),
            axis_values=(np.linspace(0, 1, 5),),
            construction_method="pooled_min_max",
        )


def test_grid_and_density_shapes():
    grid = Grid(
        grid_id="g",
        feature_names=("PC1", "PC2"),
        axis_values=(np.linspace(0, 1, 5), np.linspace(0, 1, 7)),
        construction_method="pooled_min_max",
    )
    dg = DensityGrid(grid_id="g", feature_names=("PC1", "PC2"), density=np.zeros((5, 7)))
    assert dg.density.shape == tuple(len(a) for a in grid.axis_values)


def test_sample_set_optional_slots_default_none():
    s = SampleSet(
        sample_set_id="d__WT",
        sample_set_name="WT",
        distribution_id="d",
        sample_ids=("s0", "s1"),
    )
    assert s.geometry is None and s.hdr is None and s.feature_profile is None


def test_geometry_has_no_run_relative_scalars():
    fields = {f.name for f in dataclasses.fields(SampleSetGeometry)}
    for banned in ("support_fraction", "prominence_rank", "height_relative_to_max", "is_dominant"):
        assert banned not in fields
    assert {"center", "radius", "r80", "cv_radius_from_center", "grid_id", "feature_names"} <= fields


# --------------------------------------------------------------------------- #
# label_groups view (kept helpers)
# --------------------------------------------------------------------------- #
def test_derive_label_groups_view_and_resolution():
    lg = LabelGroup(
        label_group_name="peak",
        distribution_id="d",
        sample_set_ids=("d__peak_0", "d__peak_1"),
        sample_id_to_sample_set_id={},
    )
    view = derive_label_groups([lg])
    assert view["peak"] == ("d__peak_0", "d__peak_1")
    assert resolve_label_group(view, "peak") == ("d__peak_0", "d__peak_1")


def test_derive_label_groups_duplicate_name_raises():
    lg = LabelGroup(
        label_group_name="peak", distribution_id="d",
        sample_set_ids=(), sample_id_to_sample_set_id={},
    )
    with pytest.raises(ValueError):
        derive_label_groups([lg, lg])


def test_resolve_label_group_ambiguous_and_missing():
    view = {"peak_bwA": ("a",), "peak_bwB": ("b",)}
    with pytest.raises(ValueError):
        resolve_label_group(view, "peak")
    with pytest.raises(KeyError):
        resolve_label_group({"genotype": ("g",)}, "peak")
