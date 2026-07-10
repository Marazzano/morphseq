"""TASK_0 — object shape tests (§1–§4)."""

import dataclasses

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    Distribution,
    Grid,
    DensityGrid,
    SampleSet,
    SampleSetGeometry,
    HDR,
    FeatureProfile,
    LabelGroup,
    LabelGroupArtifacts,
    derive_label_groups,
    resolve_label_group,
)


def _distribution(n=4, feats=("PC1", "PC2")):
    return Distribution(
        distribution_id="b9d2_30hpf_reference",
        scope_id="b9d2",
        time_bin=30,
        role="reference",
        sample_ids=tuple(f"s{i}" for i in range(n)),
        feature_names=feats,
        feature_values=np.arange(n * len(feats), dtype=float).reshape(n, len(feats)),
    )


def test_distribution_minimal_construction():
    d = _distribution()
    assert d.feature_names == ("PC1", "PC2")
    assert d.feature_values.shape == (4, 2)


def test_distribution_has_no_coordinate_frame():
    fields = {f.name for f in dataclasses.fields(Distribution)}
    assert "coordinate_frame" not in fields


def test_distribution_frozen():
    d = _distribution()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.scope_id = "other"  # type: ignore[misc]


def test_distribution_feature_values_readonly():
    d = _distribution()
    with pytest.raises(ValueError):
        d.feature_values[0, 0] = 999.0


def test_feature_values_column_invariant_enforced():
    # 3 feature_names but only 2 columns -> raises (#3).
    with pytest.raises(ValueError):
        Distribution(
            distribution_id="x",
            scope_id="b9d2",
            time_bin=30,
            role="reference",
            sample_ids=("s0",),
            feature_names=("PC1", "PC2", "PC3"),
            feature_values=np.zeros((1, 2)),
        )


def test_sample_ids_row_invariant_enforced():
    with pytest.raises(ValueError):
        Distribution(
            distribution_id="x",
            scope_id="b9d2",
            time_bin=30,
            role="reference",
            sample_ids=("s0", "s1"),
            feature_names=("PC1",),
            feature_values=np.zeros((1, 1)),
        )


def test_grid_axis_count_matches_features():
    with pytest.raises(ValueError):
        Grid(
            grid_id="g",
            feature_names=("PC1", "PC2"),
            axis_values=(np.linspace(0, 1, 5),),  # only 1 axis for 2 features
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
        sample_set_id="b9d2_30hpf_reference__genoWT",
        sample_set_name="WT",
        distribution_id="b9d2_30hpf_reference",
        sample_ids=("s0", "s1"),
    )
    assert s.geometry is None and s.hdr is None and s.feature_profile is None


def test_geometry_has_no_run_relative_scalars():
    # #4 — support_fraction / prominence_rank etc must NOT be geometry fields.
    fields = {f.name for f in dataclasses.fields(SampleSetGeometry)}
    for banned in ("support_fraction", "prominence_rank", "height_relative_to_max", "is_dominant"):
        assert banned not in fields
    assert {"center", "radius", "r80", "cv_radius_from_center", "grid_id", "feature_names"} <= fields


def test_labelgroup_defaults_are_factories_not_shared():
    a = LabelGroup(
        label_group_name="g",
        distribution_id="d",
        sample_set_ids=(),
        sample_id_to_sample_set_id={},
    )
    b = LabelGroup(
        label_group_name="g",
        distribution_id="d",
        sample_set_ids=(),
        sample_id_to_sample_set_id={},
    )
    assert a.provenance is not b.provenance  # no shared mutable default


def test_derive_label_groups_view():
    lg = LabelGroup(
        label_group_name="peak",
        distribution_id="d",
        sample_set_ids=("d__peak_0", "d__peak_1"),
        sample_id_to_sample_set_id={},
    )
    view = derive_label_groups([lg])
    assert view["peak"] == ("d__peak_0", "d__peak_1")


def test_derive_label_groups_duplicate_name_raises():
    lg = LabelGroup(
        label_group_name="peak", distribution_id="d",
        sample_set_ids=(), sample_id_to_sample_set_id={},
    )
    with pytest.raises(ValueError):
        derive_label_groups([lg, lg])


def test_resolve_label_group_exact_and_alias():
    view = {"peak_bwA": ("a",), "genotype": ("g",)}
    assert resolve_label_group(view, "genotype") == ("g",)
    assert resolve_label_group(view, "peak") == ("a",)  # unique prefix alias


def test_resolve_label_group_ambiguous_raises():
    view = {"peak_bwA": ("a",), "peak_bwB": ("b",)}
    with pytest.raises(ValueError):
        resolve_label_group(view, "peak")


def test_resolve_label_group_no_match_raises():
    with pytest.raises(KeyError):
        resolve_label_group({"genotype": ("g",)}, "peak")
