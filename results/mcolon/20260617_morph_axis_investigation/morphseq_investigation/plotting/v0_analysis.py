"""Analysis inputs for the legacy V0 distribution QA figure.

This module owns numerical analysis.  Plot renderers consume its results and
do not infer peak counts or metric values while drawing a figure.
"""

from __future__ import annotations

import numpy as np

from morphseq_investigation.core.peak_counting import PeakCountDetail, peak_count_detail
from morphseq_investigation.core.support_geometry import (
    fiedler_value,
    hdr_concentration_auc,
    mst_max_edge,
    normalize_shape,
    valley_depth,
)


def compute_v0_metric_summary(
    points: np.ndarray,
    *,
    kde=None,
    distribution_id: str | None = None,
) -> dict[str, float | None]:
    """Compute the small V0 metric set on shape-normalized sampled points."""

    pts = normalize_shape(np.asarray(points, dtype=float))
    valley = float(valley_depth(pts, kde=kde))
    if distribution_id is not None and distribution_id.startswith("one_peak_"):
        valley = None
    return {
        "hdr_concentration_auc": float(hdr_concentration_auc(pts, relative=True, kde=kde)),
        "valley_depth": valley,
        "mst_max_edge": float(mst_max_edge(pts)),
        "fiedler": float(fiedler_value(pts)),
    }


def compute_v0_peak_count_summary(
    *,
    truth_density: np.ndarray | None = None,
    observed_density: np.ndarray | None = None,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
) -> dict[str, PeakCountDetail | None]:
    """Compute truth and observed peak-count probes for one V0 distribution."""

    def detail(density: np.ndarray | None) -> PeakCountDetail | None:
        if density is None:
            return None
        return peak_count_detail(
            density,
            min_component_mass_frac=min_component_mass_frac,
            sweep_steps=sweep_steps,
        )

    return {"truth": detail(truth_density), "observed": detail(observed_density)}
