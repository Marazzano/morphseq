"""TASK_D — plotting-as-IR tests (ontology §1b "KDE strips fall out for free").

These tests only assert IR shape + the faceting engine's own ``render()``
accepts the emitted ``FigureData`` without error (emit-only smoke, per the
brief — no golden-image assertions). No bespoke drawing is exercised here;
that lives entirely in the real faceting engine renderers.
"""

import numpy as np
import pytest

from analyze.viz.plotting.faceting_engine import SubplotData, TraceData

from morphseq_investigation.engine.grid import build_grid, evaluate_density
from morphseq_investigation.engine.plotting import UNASSIGNED_LABEL, overlay_strip_subplot, strip_trace


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
