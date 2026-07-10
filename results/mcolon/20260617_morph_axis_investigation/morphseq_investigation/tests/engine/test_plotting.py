"""TASK_D — plotting-as-IR tests (ontology §1b "KDE strips fall out for free").

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

from morphseq_investigation.engine.grid import build_grid, evaluate_density
from morphseq_investigation.engine.objects import HDR, SampleSet
from morphseq_investigation.engine.plotting import (
    UNASSIGNED_LABEL,
    hdr_band_trace,
    overlay_strip_subplot,
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
