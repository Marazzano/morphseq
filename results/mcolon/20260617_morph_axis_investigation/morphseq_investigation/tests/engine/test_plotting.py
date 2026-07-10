"""TASK_D/TASK_C — plotting-as-IR tests (ontology §1b "KDE strips fall out for
free"; TASK_C PATH A / typed FacetKey, see
``docs/tasks_catalog/TASK_C_plotting.md``).

These tests only assert IR shape + the faceting engine's own ``render()``
accepts the emitted ``FigureData`` without error (emit-only smoke, per the
brief — no golden-image assertions). No bespoke drawing is exercised here;
that lives entirely in the real faceting engine renderers.
"""

import numpy as np
import pytest

from analyze.viz.plotting.faceting_engine import (
    FacetSpec,
    FigureData,
    SubplotData,
    TraceData,
    render,
)

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.grid import build_grid, evaluate_density
from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.objects import HDR, Distribution, SampleSet
from morphseq_investigation.engine.plotting import (
    UNASSIGNED_LABEL,
    DistributionGrid,
    IncomparableDistributionsError,
    build_1d_density_grid,
    hdr_band_trace,
    overlay_strip_subplot,
    plot_1d_density_grid,
    sample_set_strip_with_hdr,
    strip_grid_figure,
    strip_trace,
)


def _one_d_grid(seed=0, n=40, loc=0.0, resolution=30, method="pooled_min_max", params=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=loc, scale=1.5, size=n).reshape(-1, 1)
    sample_ids = tuple(f"s{i}" for i in range(n))
    params = dict(params or {})
    params.setdefault("resolution", resolution)
    grid = build_grid(("PC1",), values, sample_ids, method, params)
    return grid, values, sample_ids


def _density_on(grid, values, bandwidth=0.5):
    return evaluate_density(grid, values, bandwidth)


# --------------------------------------------------------------------------- #
# strip_trace
# --------------------------------------------------------------------------- #
def test_strip_trace_maps_axis_values_and_density_onto_trace_xy():
    grid, values, _ = _one_d_grid()
    dg = _density_on(grid, values)

    trace = strip_trace(grid, dg, label="reference")

    assert isinstance(trace, TraceData)
    np.testing.assert_array_equal(trace.x, grid.axis_values[0])
    np.testing.assert_array_equal(trace.y, dg.density.reshape(-1))
    assert trace.label == "reference"


def test_strip_trace_rejects_nd_grid():
    values, sample_ids = (
        np.stack(
            [
                np.random.default_rng(1).normal(size=20),
                np.random.default_rng(2).normal(size=20),
            ],
            axis=1,
        ),
        tuple(f"s{i}" for i in range(20)),
    )
    grid = build_grid(("PC1", "PC2"), values, sample_ids, "pooled_min_max", {"resolution": 10})
    dg = evaluate_density(grid, values, 0.5)
    with pytest.raises(ValueError):
        strip_trace(grid, dg, label="x")


def test_strip_trace_rejects_mismatched_grid_id():
    grid_a, values_a, _ = _one_d_grid(seed=0)
    grid_b, values_b, _ = _one_d_grid(seed=99, loc=10.0)  # different bounds -> different grid_id
    dg_b = _density_on(grid_b, values_b)

    assert grid_a.grid_id != grid_b.grid_id
    with pytest.raises(ValueError):
        strip_trace(grid_a, dg_b, label="x")


# --------------------------------------------------------------------------- #
# overlay on shared grid_id
# --------------------------------------------------------------------------- #
def test_overlay_shared_grid_id_produces_one_subplot_with_n_traces():
    grid, _, _ = _one_d_grid()
    rng = np.random.default_rng(0)
    values_a = rng.normal(loc=-1.0, scale=1.0, size=30).reshape(-1, 1)
    values_b = rng.normal(loc=1.0, scale=1.0, size=25).reshape(-1, 1)
    values_c = rng.normal(loc=0.0, scale=2.0, size=10).reshape(-1, 1)

    dg_a = evaluate_density(grid, values_a, 0.5)
    dg_b = evaluate_density(grid, values_b, 0.5)
    dg_c = evaluate_density(grid, values_c, 0.5)

    subplot = overlay_strip_subplot(
        grid,
        [dg_a, dg_b, dg_c],
        labels=["CE", "HTA", UNASSIGNED_LABEL],
    )

    assert isinstance(subplot, SubplotData)
    assert subplot.heatmap is None
    assert len(subplot.traces) == 3
    for trace, dg in zip(subplot.traces, [dg_a, dg_b, dg_c]):
        np.testing.assert_array_equal(trace.x, grid.axis_values[0])
        np.testing.assert_array_equal(trace.y, dg.density.reshape(-1))


def test_overlay_mismatched_grid_raises():
    grid_a, _, _ = _one_d_grid(seed=0)
    grid_b, values_b, _ = _one_d_grid(seed=123, loc=50.0)
    dg_same = evaluate_density(grid_a, np.zeros((5, 1)), 0.5)
    dg_other = evaluate_density(grid_b, values_b, 0.5)

    assert grid_a.grid_id != grid_b.grid_id
    with pytest.raises(ValueError):
        overlay_strip_subplot(grid_a, [dg_same, dg_other], labels=["a", "b"])


# --------------------------------------------------------------------------- #
# N x M FigureData grid
# --------------------------------------------------------------------------- #
def test_strip_grid_figure_subplot_count_matches_rows_and_smoke_renders():
    n_features = 3
    rows = []
    for i in range(n_features):
        grid, _, _ = _one_d_grid(seed=i, loc=float(i))
        rng = np.random.default_rng(i)
        values_ref = rng.normal(loc=0.0, size=20).reshape(-1, 1)
        values_tgt = rng.normal(loc=1.0, size=20).reshape(-1, 1)
        dg_ref = evaluate_density(grid, values_ref, 0.5)
        dg_tgt = evaluate_density(grid, values_tgt, 0.5)
        rows.append(
            {
                "grid": grid,
                "density_grids": [dg_ref, dg_tgt],
                "labels": ["reference", "target"],
                "title": f"feature_{i}",
            }
        )

    fig_data = strip_grid_figure(rows, title="N-feature x M-comparison strips")

    assert isinstance(fig_data, FigureData)
    assert len(fig_data.subplots) == n_features  # N rows
    for subplot in fig_data.subplots:
        assert len(subplot.traces) == 2  # M comparisons per row

    # Emit-only smoke: render() must accept this FigureData without error.
    # Strips get a wider aspect ratio via FacetSpec, passed to render() here
    # (not baked into the emitter / FigureData).
    facet = FacetSpec(wrap=1, sharex=False, sharey=False)
    result = render(fig_data, backend="matplotlib", facet=facet)
    assert result is not None


# --------------------------------------------------------------------------- #
# unassigned reserved category
# --------------------------------------------------------------------------- #
def test_unassigned_present_as_reserved_muted_trace():
    grid, _, _ = _one_d_grid()
    rng = np.random.default_rng(7)
    values_group = rng.normal(loc=0.0, size=20).reshape(-1, 1)
    values_unassigned = rng.normal(loc=3.0, size=5).reshape(-1, 1)
    dg_group = evaluate_density(grid, values_group, 0.5)
    dg_unassigned = evaluate_density(grid, values_unassigned, 0.5)

    subplot = overlay_strip_subplot(
        grid,
        [dg_group, dg_unassigned],
        labels=["peak_0", UNASSIGNED_LABEL],
    )

    labels = [t.label for t in subplot.traces]
    assert UNASSIGNED_LABEL in labels
    unassigned_trace = subplot.traces[labels.index(UNASSIGNED_LABEL)]
    # Reserved muted style: distinct from a normal fully-opaque line.
    assert unassigned_trace.style.alpha < 1.0
    assert unassigned_trace.style.color == "#B0B0B0"


def test_unassigned_never_dropped_from_overlay():
    grid, _, _ = _one_d_grid()
    rng = np.random.default_rng(3)
    values = [rng.normal(size=10).reshape(-1, 1) for _ in range(3)]
    density_grids = [evaluate_density(grid, v, 0.5) for v in values]
    labels = ["CE", UNASSIGNED_LABEL, "HTA"]

    subplot = overlay_strip_subplot(grid, density_grids, labels=labels)

    assert len(subplot.traces) == 3
    assert [t.label for t in subplot.traces] == labels


# --------------------------------------------------------------------------- #
# HDR band overlay
# --------------------------------------------------------------------------- #
def test_hdr_band_trace_shades_masked_region():
    grid, values, sample_ids = _one_d_grid(n=40)
    dg = _density_on(grid, values)
    resolution = len(grid.axis_values[0])
    mask = np.zeros(resolution, dtype=bool)
    mask[5:15] = True
    hdr = HDR(grid_id=grid.grid_id, feature_names=grid.feature_names, level=0.8, mask=mask)

    band = hdr_band_trace(grid, dg, hdr, label="peak_0")

    assert band.render_as == "band"
    assert band.band_lower is not None and band.band_upper is not None
    np.testing.assert_array_equal(band.band_lower, np.zeros(resolution))
    density_flat = dg.density.reshape(-1)
    expected_upper = np.where(mask, density_flat, 0.0)
    np.testing.assert_array_equal(band.band_upper, expected_upper)


def test_hdr_band_trace_rejects_mismatched_grid_id():
    grid, values, _ = _one_d_grid()
    dg = _density_on(grid, values)
    bad_hdr = HDR(
        grid_id="grid_not_this_one",
        feature_names=grid.feature_names,
        level=0.8,
        mask=np.ones(len(grid.axis_values[0]), dtype=bool),
    )
    with pytest.raises(ValueError):
        hdr_band_trace(grid, dg, bad_hdr, label="peak_0")


def test_sample_set_strip_with_hdr_returns_curve_and_band_when_hdr_present():
    grid, values, sample_ids = _one_d_grid(n=30)
    dg = _density_on(grid, values)
    resolution = len(grid.axis_values[0])
    mask = np.zeros(resolution, dtype=bool)
    mask[10:20] = True
    hdr = HDR(grid_id=grid.grid_id, feature_names=grid.feature_names, level=0.8, mask=mask)

    sample_set = SampleSet(
        sample_set_id="dist__peak_0",
        sample_set_name="peak_0",
        distribution_id="dist",
        sample_ids=sample_ids,
        hdr=hdr,
    )

    traces = sample_set_strip_with_hdr(grid, sample_set, dg)
    assert len(traces) == 2
    assert traces[0].render_as == "line"
    assert traces[1].render_as == "band"


def test_sample_set_strip_without_hdr_returns_only_curve():
    grid, values, sample_ids = _one_d_grid(n=15)
    dg = _density_on(grid, values)
    sample_set = SampleSet(
        sample_set_id="dist__wildtype",
        sample_set_name="wildtype",
        distribution_id="dist",
        sample_ids=sample_ids,
        hdr=None,
    )
    traces = sample_set_strip_with_hdr(grid, sample_set, dg)
    assert len(traces) == 1
    assert traces[0].render_as == "line"


# --------------------------------------------------------------------------- #
# Emit-only smoke: render() accepts a single-subplot FigureData too
# --------------------------------------------------------------------------- #
def test_single_overlay_subplot_smoke_renders():
    grid, _, _ = _one_d_grid()
    rng = np.random.default_rng(0)
    values_a = rng.normal(loc=-1.0, size=20).reshape(-1, 1)
    values_b = rng.normal(loc=1.0, size=20).reshape(-1, 1)
    dg_a = evaluate_density(grid, values_a, 0.5)
    dg_b = evaluate_density(grid, values_b, 0.5)

    subplot = overlay_strip_subplot(grid, [dg_a, dg_b], labels=["reference", "target"])
    fig_data = FigureData(title="single strip", subplots=[subplot])

    result = render(fig_data, backend="matplotlib")
    assert result is not None


# =========================================================================== #
# TASK_C fixtures — hand-built Distributions/DistributionLabelGroups (PATH A).
# No catalog or real peak detection needed (spec §"Develop against fixtures").
# =========================================================================== #
def _make_distribution(coordinates, feature="total_length_um", n_per_group=None, seed=0):
    """A Distribution with one feature and a "genotype" label column attached.

    ``n_per_group`` maps category -> sample count (e.g. {"wildtype": 20,
    "b9d2": 15}); samples not named end up UNASSIGNED via with_label's default.
    """
    n_per_group = dict(n_per_group or {"wildtype": 20, "b9d2": 15})
    rng = np.random.default_rng(seed)
    sample_ids = []
    assignments = {}
    values = []
    loc_by_group = {"wildtype": 0.0, "b9d2": 3.0, "unlabeled_extra": 50.0}
    for group, n in n_per_group.items():
        for i in range(n):
            sid = f"{coordinates.get('time_bin', 'x')}_{group}_{i}"
            sample_ids.append(sid)
            assignments[sid] = group
            loc = loc_by_group.get(group, 0.0)
            values.append(rng.normal(loc=loc, scale=1.0))

    dist = Distribution(
        distribution_id=make_distribution_id(coordinates),
        sample_ids=tuple(sample_ids),
        feature_names=(feature,),
        feature_values=np.asarray(values, dtype=float).reshape(-1, 1),
        coordinates=coordinates,
    )
    return dist.with_label("genotype", assignments)


# --------------------------------------------------------------------------- #
# Typed FacetKey — a bad coordinate name is a construction-time error, not late
# --------------------------------------------------------------------------- #
def test_facet_key_bad_coordinate_name_is_construction_error_not_late_crash():
    dist = _make_distribution({"time_bin": 30})
    group = dist.label_group("genotype", display_name="Genotype")
    # Resolving an unknown coordinate raises immediately at facet-resolution
    # time (inside build_1d_density_grid's bucketing pass), with a precise
    # KeyError naming the missing coordinate — not a downstream/late failure.
    with pytest.raises(KeyError):
        build_1d_density_grid(
            [group], "total_length_um",
            facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("no_such_coordinate"),
        )


def test_coordinate_facet_and_label_group_facet_are_distinct_typed_keys():
    assert CoordinateFacet("time_bin") != CoordinateFacet("scope_id")
    assert CoordinateFacet("time_bin") == CoordinateFacet("time_bin")
    assert LabelGroupFacet() == LabelGroupFacet()


# --------------------------------------------------------------------------- #
# PATH A — build_1d_density_grid(DistributionLabelGroup*)
# --------------------------------------------------------------------------- #
def test_build_1d_density_grid_cells_and_curves_match_label_group_sample_sets():
    dist_30 = _make_distribution({"time_bin": 30}, seed=0)
    dist_48 = _make_distribution({"time_bin": 48}, seed=1)
    groups = [
        dist_30.label_group("genotype", display_name="Genotype"),
        dist_48.label_group("genotype", display_name="Genotype"),
    ]

    grid = build_1d_density_grid(
        groups, "total_length_um",
        facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
    )

    assert isinstance(grid, DistributionGrid)
    assert grid.feature_name == "total_length_um"
    # One cell per (label_group display_name, time_bin) -> 2 cells (30, 48).
    cells = {c.cell for c in grid.curves}
    assert cells == {("Genotype", 30), ("Genotype", 48)}
    # Curves = the label group's SampleSets (wildtype + b9d2, unassigned dropped).
    names_30 = sorted(c.sample_set_name for c in grid.curves if c.cell == ("Genotype", 30))
    assert names_30 == ["b9d2", "wildtype"]


def test_build_1d_density_grid_cell_bounds_from_selected_curves_only():
    # A Distribution carries an extreme-valued sample NOT named by the
    # "genotype" label column (so it is UNASSIGNED under that label group,
    # per with_label's contract) -- dropping it must not stretch the cell's
    # shared grid (spec §PATH A: "cell grid = union of the CURVES SELECTED").
    rng = np.random.default_rng(2)
    sample_ids = [f"wt_{i}" for i in range(20)] + [f"b9d2_{i}" for i in range(15)]
    values = list(rng.normal(loc=0.0, scale=1.0, size=20)) + list(
        rng.normal(loc=3.0, scale=1.0, size=15)
    )
    assignments = {sid: ("wildtype" if sid.startswith("wt") else "b9d2") for sid in sample_ids}

    # dist_without_outlier: exactly these samples, nothing more.
    dist_without_outlier = Distribution(
        distribution_id=make_distribution_id({"time_bin": 30, "variant": "clean"}),
        sample_ids=tuple(sample_ids),
        feature_names=("total_length_um",),
        feature_values=np.asarray(values, dtype=float).reshape(-1, 1),
        coordinates={"time_bin": 30, "variant": "clean"},
    ).with_label("genotype", assignments)

    # dist_with_outlier: same samples + one extreme-valued sample that the
    # "genotype" label column does NOT name (with_label marks it UNASSIGNED).
    outlier_id = "outlier_0"
    sample_ids_with = tuple(sample_ids) + (outlier_id,)
    values_with = np.asarray(values + [500.0], dtype=float).reshape(-1, 1)
    dist_with_outlier = Distribution(
        distribution_id=make_distribution_id({"time_bin": 30, "variant": "with_outlier"}),
        sample_ids=sample_ids_with,
        feature_names=("total_length_um",),
        feature_values=values_with,
        coordinates={"time_bin": 30, "variant": "with_outlier"},
    ).with_label("genotype", assignments)  # outlier_id absent -> UNASSIGNED

    group_with_outlier = dist_with_outlier.label_group("genotype", display_name="Genotype")
    group_without_outlier = dist_without_outlier.label_group("genotype", display_name="Genotype")

    grid_with = build_1d_density_grid([group_with_outlier], "total_length_um")
    grid_without = build_1d_density_grid([group_without_outlier], "total_length_um")

    bounds_with = grid_with.curves[0].grid.axis_values[0]
    bounds_without = grid_without.curves[0].grid.axis_values[0]
    # The outlier is UNASSIGNED under "genotype" -> excluded from
    # Distribution.sample_sets("genotype") -> never enters the pooled bounds.
    # Both grids' selected curves are wildtype/b9d2 only -> identical bounds,
    # proving the outlier's extreme value never stretched the shared grid.
    np.testing.assert_allclose(bounds_with.min(), bounds_without.min())
    np.testing.assert_allclose(bounds_with.max(), bounds_without.max())


def test_build_1d_density_grid_raises_on_mixed_distribution_cell():
    # Facet ONLY on time_bin (NOT LabelGroupFacet), and both distributions
    # share time_bin=30 despite being different distributions (different
    # scope_id) -- they land in the SAME cell, which is the forbidden
    # "overlay two distributions" shape (spec §"One-distribution-per-cell
    # invariant").
    dist_a = _make_distribution({"time_bin": 30, "scope_id": "expA"}, seed=0)
    dist_b = _make_distribution({"time_bin": 30, "scope_id": "expB"}, seed=1)
    groups = [
        dist_a.label_group("genotype", display_name="Genotype"),
        dist_b.label_group("genotype", display_name="Genotype"),
    ]

    with pytest.raises(IncomparableDistributionsError):
        build_1d_density_grid(
            groups, "total_length_um",
            facet_row=CoordinateFacet("time_bin"), facet_col=CoordinateFacet("time_bin"),
        )


def test_build_1d_density_grid_label_group_facet_as_axis_never_raises_mixed_cell():
    # When LabelGroupFacet IS an axis, every cell is single-label-group (hence
    # single-distribution) by construction -- the guard is skipped, not
    # bypassed unsafely, because it cannot fire.
    dist_a = _make_distribution({"time_bin": 30}, seed=0)
    groups = [dist_a.label_group("genotype", display_name="Genotype")]
    grid = build_1d_density_grid(
        groups, "total_length_um",
        facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
    )
    assert len(grid.curves) > 0


# --------------------------------------------------------------------------- #
# plot_1d_density_grid renders the PATH A grid; reference role -> dashed/gray
# --------------------------------------------------------------------------- #
def test_plot_1d_density_grid_renders_path_a_grid():
    dist_30 = _make_distribution({"time_bin": 30}, seed=0)
    group = dist_30.label_group("genotype", display_name="Genotype")
    grid = build_1d_density_grid([group], "total_length_um")

    result = plot_1d_density_grid(grid, reference_role=None)
    assert result is not None
