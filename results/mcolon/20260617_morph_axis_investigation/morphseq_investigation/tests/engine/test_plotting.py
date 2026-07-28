import dataclasses

import numpy as np
import pandas as pd
import pytest

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.compare import NullTestResult, compare_distributions
from morphseq_investigation.engine.objects import (
    DensityEstimate, DensityEstimateSpec, DensityGrid, Distribution, Grid,
)
from morphseq_investigation.engine.plotting import (
    DistributionCurve,
    DistributionGrid,
    build_1d_density_grid,
    descriptive_comparison_2d_figure,
    descriptive_comparison_subplot,
    label_group_scatter_subplot,
    plot_1d_density_grid,
    plot_distr_metric_over_time,
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


def _null_result(state, observed_overlap):
    if state == "not_tested":
        return NullTestResult()
    if state == "invalid":
        return NullTestResult(
            state=state, observed_overlap=observed_overlap, n_draws=1
        )
    return NullTestResult(
        state=state,
        observed_overlap=observed_overlap,
        null_overlaps=np.array([observed_overlap]),
        valid_draws=1,
        p_value=0.01 if state == "significant" else 0.5,
        n_draws=1,
    )


def test_strip_trace_reads_existing_density_exactly():
    distribution = _distribution()
    estimate = distribution.shared_density
    trace = strip_trace(estimate.grid, estimate.density_grid, label="all")
    np.testing.assert_array_equal(trace.x, estimate.grid.axis_values[0])
    np.testing.assert_array_equal(trace.y, estimate.density_grid.density)


@pytest.mark.parametrize(
    "state, linestyle",
    [
        ("not_tested", ":"),
        ("invalid", "-."),
        ("nonsignificant", "--"),
        ("significant", "-"),
    ],
)
def test_descriptive_comparison_1d_has_explicit_null_state_style(state, linestyle):
    comparison = compare_distributions(
        reference=_distribution(30), targets=(_distribution(31),), grid_size=12
    ).comparisons[0]
    comparison = dataclasses.replace(
        comparison, null_test=_null_result(state, comparison.density_overlap)
    )

    subplot = descriptive_comparison_subplot(comparison)

    assert subplot.traces[1].style.linestyle == linestyle
    np.testing.assert_array_equal(
        subplot.traces[1].y, comparison.target_density.density_grid.density
    )


@pytest.mark.parametrize(
    "state, annotation, significant_cells",
    [
        ("not_tested", "not tested", 0),
        ("invalid", "invalid", 0),
        ("nonsignificant", "n.s.", 0),
        ("significant", "*", 1),
    ],
)
def test_descriptive_comparison_2d_consumes_prepared_fields_only(
    state, annotation, significant_cells, monkeypatch
):
    rng = np.random.default_rng(7)
    reference = Distribution(
        "reference", tuple(f"r{i}" for i in range(20)), ("x", "y"),
        rng.normal(size=(20, 2)), {},
    )
    target = Distribution(
        "target", tuple(f"t{i}" for i in range(20)), ("x", "y"),
        rng.normal(loc=0.4, size=(20, 2)), {},
    )
    comparison = compare_distributions(
        reference=reference, targets=(target,), features=("x", "y"), grid_size=9
    ).comparisons[0]
    comparison = dataclasses.replace(
        comparison, null_test=_null_result(state, comparison.density_overlap)
    )
    monkeypatch.setattr(
        "morphseq_investigation.engine.grid.evaluate_density",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("KDE called by plotting")),
    )

    figure = descriptive_comparison_2d_figure(comparison)

    assert len(figure.subplots) == 2
    np.testing.assert_array_equal(
        figure.subplots[0].heatmap.values,
        comparison.reference_density.density_grid.density,
    )
    np.testing.assert_array_equal(
        figure.subplots[1].heatmap.values,
        comparison.target_density.density_grid.density,
    )
    assert annotation in figure.subplots[1].heatmap.annotations
    assert int(figure.subplots[1].heatmap.sig_mask.sum()) == significant_cells


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


def test_metric_over_time_retains_robust_and_nonrobust_rows_with_distinct_styles():
    table = pd.DataFrame(
        {
            "time_bin": [24, 30, 36, 24, 30],
            "genotype": ["b9d2", "b9d2", "b9d2", "wildtype", "wildtype"],
            "peak_count": [1, 2, 2, 1, 1],
            "is_robust": [True, False, True, True, False],
        }
    )
    figure = plot_distr_metric_over_time(
        table, value="peak_count", time="time_bin", group_by="genotype"
    )
    lines = figure.axes[0].lines
    assert sum(len(line.get_xdata()) for line in lines) == len(table)
    solid = [line for line in lines if line.get_linestyle() == "-"]
    dashed = [line for line in lines if line.get_linestyle() == "--"]
    assert solid and dashed
    assert all(line.get_markerfacecolor() != "none" for line in solid)
    assert all(line.get_markerfacecolor() == "none" for line in dashed)


def test_metric_over_time_supports_generic_relative_metric_and_multi_column_groups():
    table = pd.DataFrame(
        {
            "hpf": [24, 30, 24, 30],
            "target": ["mut", "mut", "mut", "mut"],
            "reference": ["wt", "wt", "ctrl", "ctrl"],
            "relative_radius": [1.1, .9, 1.3, 1.0],
            "reliable": [True, True, False, True],
        }
    )
    figure = plot_distr_metric_over_time(
        table, value="relative_radius", time="hpf",
        group_by=("target", "reference"), style="reliable",
    )
    assert sum(len(line.get_xdata()) for line in figure.axes[0].lines) == len(table)


@pytest.mark.parametrize(
    "kwargs, missing",
    [
        ({"value": "missing", "time": "time", "group_by": "group"}, "missing"),
        ({"value": "value", "time": "missing", "group_by": "group"}, "missing"),
        ({"value": "value", "time": "time", "group_by": "missing"}, "missing"),
        ({"value": "value", "time": "time", "group_by": "group", "style": "missing"}, "missing"),
    ],
)
def test_metric_over_time_validates_declared_columns(kwargs, missing):
    table = pd.DataFrame({"value": [1], "time": [2], "group": ["a"], "is_robust": [True]})
    with pytest.raises(ValueError, match="missing required columns"):
        plot_distr_metric_over_time(table, **kwargs)
