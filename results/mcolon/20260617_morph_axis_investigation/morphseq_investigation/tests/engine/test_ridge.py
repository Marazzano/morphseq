import dataclasses

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from morphseq_investigation.engine.facets import CoordinateFacet, LabelGroupFacet
from morphseq_investigation.engine.objects import (
    DensityEstimate, DensityEstimateSpec, DensityGrid, Distribution, Grid,
)
from morphseq_investigation.engine.plotting import (
    DistributionCurve, DistributionGrid, build_1d_density_grid,
    marginalize_density_estimate,
)
from morphseq_investigation.engine.facets import LabelGroupFacet, CoordinateFacet
from morphseq_investigation.engine.ridge import plot_1d_ridgeline


def _grid():
    values = np.random.default_rng(1).normal(size=(20, 1))
    distribution = Distribution("d", tuple(f"s{i}" for i in range(20)), ("x",), values)
    axis = np.linspace(values.min(), values.max(), 12)
    raster = Grid("g", ("x",), (axis,), "fixed_bounds")
    estimate = DensityEstimate(
        "d", ("x",), DensityEstimateSpec(), raster,
        DensityGrid("g", ("x",), np.exp(-axis ** 2)),
    )
    robust = DistributionCurve(("view", 30), "peak_0", "target", estimate.grid,
                               estimate.density_grid, 10, is_robust=True)
    nonrobust = dataclasses.replace(robust, sample_set_name="peak_1", is_robust=False)
    return DistributionGrid("x", LabelGroupFacet(), CoordinateFacet("time_bin"), (robust, nonrobust))


@pytest.mark.parametrize("variant", ["overlaid", "stacked", "mirror"])
def test_ridge_variants_read_existing_density(variant):
    assert plot_1d_ridgeline(_grid(), variant=variant, reference_role=None) is not None


def test_ridge_keeps_nonrobust_curve_visible_and_dotted():
    figure = plot_1d_ridgeline(_grid(), reference_role=None)
    lines = [line for axis in figure.axes for line in axis.get_lines()]
    assert any(line.get_linestyle() in (":", "dotted") for line in lines)


def test_unknown_ridge_variant_raises():
    with pytest.raises(ValueError):
        plot_1d_ridgeline(_grid(), variant="unknown")


def _two_dimensional_distribution():
    distribution = Distribution(
        "d2", ("s0", "s1", "s2", "s3"), ("x", "y"),
        np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float),
        coordinates={"time_bin": 30},
    ).with_label("peaks", {"s0": "peak_0", "s1": "peak_0", "s2": "peak_1", "s3": "peak_1"})
    x = np.linspace(-2, 2, 41)
    y = np.linspace(-3, 3, 51)
    field = np.exp(-0.5 * (x[:, None] ** 2 + (y[None, :] / 1.5) ** 2))
    raster = Grid("g2", ("x", "y"), (x, y), "fixed_bounds")
    estimate = DensityEstimate(
        "d2", ("x", "y"), DensityEstimateSpec(), raster,
        DensityGrid("g2", ("x", "y"), field),
    )
    group = dataclasses.replace(distribution.get_label_group("peaks"), density=estimate)
    distribution = dataclasses.replace(distribution, label_groups={"peaks": group})
    return distribution, estimate


def test_analytical_marginal_has_selected_shape_and_unit_integral():
    _, estimate = _two_dimensional_distribution()
    marginal = marginalize_density_estimate(estimate, "x")
    assert marginal.feature_names == ("x",)
    assert marginal.density_grid.density.shape == estimate.grid.axis_values[0].shape
    assert np.trapz(marginal.density_grid.density, marginal.grid.axis_values[0]) == pytest.approx(1.0)


def test_retained_two_dimensional_density_drives_one_dimensional_ridge_without_kde(monkeypatch):
    distribution, estimate = _two_dimensional_distribution()
    monkeypatch.setattr(
        "morphseq_investigation.engine.grid.evaluate_density",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("KDE called")),
    )
    density_grid = build_1d_density_grid(
        [(distribution, distribution.get_label_group("peaks"))], "x",
        facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
    )
    assert len(density_grid.curves) == 2
    assert all(curve.density.density.shape == (41,) for curve in density_grid.curves)
    assert plot_1d_ridgeline(density_grid, reference_role=None) is not None
