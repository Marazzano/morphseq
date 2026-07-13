import dataclasses

import numpy as np
import pytest

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.objects import (
    DensityEstimate, DensityEstimateSpec, DensityGrid, Distribution, Grid,
)
from morphseq_investigation.engine.plotting import (
    DistributionCurve,
    DistributionGrid,
    build_1d_density_grid,
    label_group_scatter_subplot,
    plot_1d_density_grid,
    strip_trace,
)


def _distribution(time=30, *, with_density=True):
    rng = np.random.default_rng(time)
    distribution = Distribution(
        distribution_id=f"d{time}",
        sample_ids=tuple(f"s{time}_{i}" for i in range(18)),
        feature_names=("x",),
        feature_values=rng.normal(size=(18, 1)),
        coordinates={"time_bin": time},
    ).with_label("provided", {f"s{time}_{i}": "a" if i < 10 else "b" for i in range(18)})
    if with_density:
        axis = np.linspace(distribution.feature_values.min(), distribution.feature_values.max(), 12)
        grid = Grid(f"g{time}", ("x",), (axis,), "fixed_bounds")
        field = DensityGrid(grid.grid_id, ("x",), np.exp(-axis ** 2))
        density = DensityEstimate(
            distribution.distribution_id, ("x",), DensityEstimateSpec(), grid, field
        )
        distribution = dataclasses.replace(distribution, densities=(density,))
    return distribution


def test_strip_trace_reads_existing_density_exactly():
    distribution = _distribution()
    estimate = distribution.shared_density
    trace = strip_trace(estimate.grid, estimate.density_grid, label="all")
    np.testing.assert_array_equal(trace.x, estimate.grid.axis_values[0])
    np.testing.assert_array_equal(trace.y, estimate.density_grid.density)


def test_build_density_grid_consumes_effective_density_without_kde(monkeypatch):
    distribution = _distribution()
    group = distribution.label_groups["provided"]
    monkeypatch.setattr(
        "morphseq_investigation.engine.grid.evaluate_density",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("KDE called")),
    )
    grid = build_1d_density_grid(
        [(distribution, group)], "x",
        facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
    )
    assert len(grid.curves) == 2
    assert {curve.cell for curve in grid.curves} == {("provided", 30)}
    assert all(curve.density is distribution.shared_density.density_grid for curve in grid.curves)


def test_density_builder_raises_clearly_without_effective_density():
    distribution = _distribution(with_density=False)
    with pytest.raises(ValueError, match="no effective density"):
        build_1d_density_grid(
            [(distribution, distribution.label_groups["provided"])], "x"
        )


def test_provided_label_density_free_scatter_works_without_density():
    distribution = _distribution(with_density=False)
    subplot = label_group_scatter_subplot(
        distribution, distribution.label_groups["provided"], "x"
    )
    assert len(subplot.traces) == 2
    assert all(trace.render_as == "scatter" for trace in subplot.traces)


def test_plot_styles_nonrobust_modal_curve_differently(monkeypatch):
    distribution = _distribution()
    estimate = distribution.shared_density
    robust = DistributionCurve(("r", 30), "peak_0", "target", estimate.grid,
                               estimate.density_grid, 8, is_robust=True)
    nonrobust = dataclasses.replace(robust, sample_set_name="peak_1", is_robust=False)
    grid = DistributionGrid("x", LabelGroupFacet(), CoordinateFacet("time_bin"), (robust, nonrobust))
    captured = {}
    monkeypatch.setattr(
        "morphseq_investigation.engine.plotting.render",
        lambda figure, **kwargs: captured.setdefault("figure", figure),
    )
    plot_1d_density_grid(grid, reference_role=None)
    traces = captured["figure"].subplots[0].traces
    assert len(traces) == 2
    assert traces[0].style.linestyle == "-"
    assert traces[1].style.linestyle == ":"
    assert traces[1].style.alpha < traces[0].style.alpha
