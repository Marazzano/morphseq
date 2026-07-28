"""Pure density calculation for the distribution engine."""

from __future__ import annotations

import numpy as np

from ..core.bandwidth_tuning import bandwidth_geometry_scales
from .grid import build_grid, evaluate_density
from .objects import DensityEstimate, DensityEstimateSpec, Distribution


DEFAULT_DENSITY_SPEC = DensityEstimateSpec()


def _bandwidth(values: np.ndarray, spec: DensityEstimateSpec) -> float:
    if spec.method != "isotropic_gaussian_kde":
        raise NotImplementedError(f"density method {spec.method!r} is not supported")
    if spec.bandwidth_rule not in {
        "median_kNN_distance",
        "longest_non_outlier_MST_edge",
    }:
        raise NotImplementedError(
            f"bandwidth rule {spec.bandwidth_rule!r} is not supported by the N-D engine"
        )
    scales = bandwidth_geometry_scales(values, include_connectivity_radius=False)
    scale = float(scales.get(spec.bandwidth_rule, np.nan))
    bandwidth = scale * spec.bandwidth_multiplier
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            f"bandwidth rule {spec.bandwidth_rule!r} produced no positive finite bandwidth"
        )
    return bandwidth


def calculate_density(
    distribution: Distribution, *, spec: DensityEstimateSpec | None = None
) -> DensityEstimate:
    """Calculate and return an estimate without mutating any registry or labels."""
    resolved_spec = DEFAULT_DENSITY_SPEC if spec is None else spec
    grid = build_grid(
        distribution.feature_names,
        distribution.feature_values,
        distribution.sample_ids,
        resolved_spec.grid_method,
        resolved_spec.grid_params,
    )
    field = evaluate_density(
        grid,
        distribution.feature_values,
        _bandwidth(distribution.feature_values, resolved_spec),
    )
    return DensityEstimate(
        distribution_id=distribution.distribution_id,
        feature_names=distribution.feature_names,
        spec=resolved_spec,
        grid=grid,
        density_grid=field,
    )
