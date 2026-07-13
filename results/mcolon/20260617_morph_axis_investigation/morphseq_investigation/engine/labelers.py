"""Peak-label orchestration over an already-calculated density estimate.

All analytical work belongs to the core resolver.  This module only validates
the engine boundary, invokes that public resolver, and applies the one adapter.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.density_composition import CanonicalGrid, DensityGrid as CoreDensityGrid
from ..core.distribution_records import (
    DistributionAnalysisContext,
    DistributionRecord,
    PeakResolutionConfig,
    compute_resolved_peaks,
)
from ..core.resolved_peak_analysis import DEFAULT_ANALYSIS_SPEC
from .objects import DensityEstimate, Distribution, LabelGroup
from .peak_adapter import label_group_from_resolved_peaks


def _core_density(density: DensityEstimate) -> tuple[CanonicalGrid, CoreDensityGrid]:
    """Expose an engine density through the core's public raster value object."""
    if len(density.grid.axis_values) != 2:
        raise ValueError("peak resolution currently requires a two-dimensional density")
    x_axis, y_axis = density.grid.axis_values
    grid = CanonicalGrid.from_axis_values(x_axis, y_axis)
    # CanonicalGrid uses xy raster orientation. Engine DensityGrid stores axes
    # in feature order, so its 2-D field is transposed into (y, x) raster order.
    field = CoreDensityGrid(
        xx=grid.xx,
        yy=grid.yy,
        density=np.asarray(density.density_grid.density, dtype=float).T,
        grid=grid,
    )
    return grid, field


def detect_peaks(
    distribution: Distribution,
    *,
    output_label: str,
    density: DensityEstimate,
    resolution_config: PeakResolutionConfig,
) -> LabelGroup:
    """Resolve supplied density and return its typed label group.

    Bootstrap draws calculate their own draw-local densities as part of the
    authoritative voting algorithm.  The full-data density is consumed exactly
    as supplied and is never registered or recalculated here.
    """
    if output_label in distribution.label_groups:
        raise ValueError(f"label group {output_label!r} already exists")
    if density.distribution_id != distribution.distribution_id:
        raise ValueError("density belongs to another distribution")
    if density.feature_names != distribution.feature_names:
        raise ValueError("density ordered features do not match distribution")

    canonical_grid, core_density = _core_density(density)
    analysis_spec = replace(
        DEFAULT_ANALYSIS_SPEC,
        bandwidth_rule=density.spec.bandwidth_rule,
        bandwidth_multiplier=density.spec.bandwidth_multiplier,
    )
    record = DistributionRecord(
        distribution_id=distribution.distribution_id,
        points=distribution.feature_values,
        analysis_context=DistributionAnalysisContext(
            grid=canonical_grid,
            spec=analysis_spec,
            density_grid=core_density,
        ),
    )
    resolved_record = compute_resolved_peaks(record, resolution_config)
    resolved = resolved_record.resolved_peaks
    if resolved is None:  # defensive: the public resolver contract promises one
        raise RuntimeError("core peak resolver returned no resolved distribution")
    return label_group_from_resolved_peaks(
        distribution, resolved, name=output_label, density=density
    )


__all__ = ["detect_peaks", "label_group_from_resolved_peaks"]
