"""Labelers write LABEL GROUPS onto a Distribution (spec §"Construction and
labeling are SEPARATE steps"; TASK_B §"THE ABSTRACTION").

``detect_peaks`` is the SIBLING of ``with_label`` (TASK_0): both attach a new
label column and return a NEW Distribution. Detecting peaks does NOT create a
new distribution and does NOT create a special result object — the old
``(LabelGroup, [SampleSet])`` peer-skeleton return shape is RETIRED. A detected
``peak_0`` is a :class:`~.objects.SampleSet` with its ``geometry``/``hdr``
slots filled; a provided ``wildtype`` (via ``with_label``) is the SAME type
with those slots ``None``. ONE type all the way down.

Where the "more information" a detector produces lives, by grain:
  - per-category (this peak's center / radius / r80 / HDR) -> eager, in
    ``LabelProvenance.geometry`` as a :class:`~.objects.CategoryShape` bundle
    (``{category -> CategoryShape(geometry, hdr, feature_profile)}``).
    ``Distribution.sample_sets`` (TASK_0) UNPACKS each bundle into the derived
    SampleSet's TYPED slots, so ``dist.sample_sets("resolved_peak")`` returns
    geometry- AND hdr-bearing sets directly — no wrapper, no separate unpack call.
  - per-run robustness (bootstrap vote frequencies, count stability,
    ``is_reliable``, the density field, basin labels) -> ``LabelGroupArtifacts``,
    carried in ``LabelProvenance.spec["artifacts"]`` alongside the resolved
    per-sample-set metrics and vote evidence (``LabelColumn``/``LabelProvenance``
    are FROZEN by TASK_0 with no dedicated artifacts slot of their own).

The live 2-D peak machinery (``core.distribution_records.compute_resolved_peaks``,
HDR carving, the grid bridge) is REUSED unchanged — only the OUTPUT SHAPE moved
from a free ``(LabelGroup, [SampleSet])`` tuple onto a Distribution label column.
Peak ids are LOCAL to the distribution that produced them (matching != discovery,
spec §"Distribution matching ≠ peak matching") — nothing here claims that two
distributions' ``peak_0`` correspond.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .grid import build_grid, evaluate_density
from .objects import (
    CategoryShape,
    Distribution,
    DistributionLabelGroup,
    Grid,
    HDR,
    LabelGroupArtifacts,
    LabelProvenance,
    SampleSetGeometry,
)


# A source value counts as "missing" (NA) when it is None or a float NaN.
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))  # float NaN
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# provided-column labeler — Distribution.with_label already covers this (TASK_0).
# --------------------------------------------------------------------------- #
def label_column_from_series(
    distribution: Distribution,
    name: str,
    values: Sequence[Any],
    *,
    missing_name: str = "unknown",
) -> Distribution:
    """Thin convenience: attach a label column from a sequence aligned 1:1 with
    ``distribution.sample_ids``, mapping NA (None / float NaN) source values to
    a single real ``missing_name`` category.

    This is NOT a distinct labeler — it is a small helper around
    ``Distribution.with_label`` (TASK_0) for the common "raw column, possibly
    with NAs" case. The old ``label_genotype(method="column")`` peer labeler and
    its ``(LabelGroup, [SampleSet])`` return are RETIRED; a caller wanting a
    provided/column label group calls ``with_label`` directly (or this helper
    for the NA-bucketing convenience).
    """
    values = list(values)
    if len(values) != len(distribution.sample_ids):
        raise ValueError(
            "values must align 1:1 with distribution.sample_ids: "
            f"{len(values)} values vs {len(distribution.sample_ids)} samples"
        )
    assignments = {
        sid: (missing_name if _is_missing(raw) else str(raw))
        for sid, raw in zip(distribution.sample_ids, values)
    }
    provenance = LabelProvenance(method="column", features=())
    return distribution.with_label(name, assignments, provenance=provenance)


# --------------------------------------------------------------------------- #
# detect_peaks — the unsupervised labeler, folding in the live machinery.
# --------------------------------------------------------------------------- #
def _grid_to_canonical(grid: Grid):
    """Bridge a TASK_A 2-D feature-unit :class:`Grid` to a live ``CanonicalGrid``.

    The live ``compute_resolved_peaks`` machinery (``core.distribution_records``)
    is 2-D-only and speaks ``CanonicalGrid`` (x/y bounds + a single ``grid_size``).
    ``build_grid`` produces equal-length axes from one ``resolution``, so a
    square ``CanonicalGrid`` reproduces the SAME evaluation cell coordinates —
    axes stay in feature units on both sides (no basis change). We assert the
    two axes have equal length (``CanonicalGrid`` cannot express a non-square grid).
    """
    from ..core.density_composition import CanonicalGrid

    if len(grid.feature_names) != 2:
        raise ValueError(
            "detect_peaks currently supports 2-D grids only (the live detector "
            f"is 2-D); got feature_names={grid.feature_names!r}"
        )
    x_axis, y_axis = grid.axis_values
    if len(x_axis) != len(y_axis):
        raise ValueError(
            "detect_peaks requires a square grid (equal axis lengths) to map to "
            f"CanonicalGrid; got {len(x_axis)} vs {len(y_axis)}"
        )
    return CanonicalGrid(
        x_min=float(x_axis[0]),
        x_max=float(x_axis[-1]),
        y_min=float(y_axis[0]),
        y_max=float(y_axis[-1]),
        grid_size=int(len(x_axis)),
    )


def _hdr_mask_for_basin(
    density: np.ndarray,
    basin_labels: np.ndarray,
    basin_value: int,
    level_fraction: float,
) -> np.ndarray:
    """HDR mask for one peak: its basin cells whose density is in the top
    ``level_fraction`` of that basin's mass (a highest-density region).

    Returned mask is on the SAME raster as ``density`` (the engine grid), so it
    is raster-comparable to any other set's HDR sharing the grid_id.
    """
    basin_mask = np.asarray(basin_labels) == int(basin_value)
    dens = np.where(np.isfinite(density), density, 0.0)
    if not basin_mask.any():
        return np.zeros(dens.shape, dtype=bool)
    basin_vals = dens[basin_mask]
    order = np.argsort(basin_vals)[::-1]
    sorted_vals = basin_vals[order]
    total = float(sorted_vals.sum())
    if total <= 0:
        return np.zeros(dens.shape, dtype=bool)
    cum = np.cumsum(sorted_vals) / total
    keep = np.searchsorted(cum, level_fraction) + 1
    threshold = float(sorted_vals[min(keep, len(sorted_vals)) - 1])
    return basin_mask & (dens >= threshold)


def detect_peaks(
    distribution: Distribution,
    *,
    features: Sequence[str],
    output_label: str = "resolved_peak",
    spec: Mapping[str, Any] | None = None,
    resolution: int = 61,
    grid_method: str = "pooled_min_max",
    grid_params: Mapping[str, Any] | None = None,
    resolution_config: Any | None = None,
    bandwidth: float | None = None,
    hdr_level: float = 0.80,
) -> DistributionLabelGroup:
    """Fit peaks on ``distribution``'s OWN points for ``features`` and write a
    label column named ``output_label``. This is the BODY of
    ``Distribution.detect_peaks`` (TASK_0 stub); called from there so the
    public surface stays ``distribution.detect_peaks(...)``.

    The grid is built from THIS distribution's own points ONLY (no pooled
    target/reference grid — the 1-D plot re-grids per cell downstream, TASK_C).
    Peak ids are LOCAL: two independently-run distributions may both produce a
    ``peak_0`` and nothing here claims they correspond.
    """
    from ..core.distribution_records import (
        DistributionAnalysisContext,
        DistributionRecord,
        PeakResolutionConfig,
        compute_resolved_peaks,
    )
    from ..core.resolved_peak_analysis import DEFAULT_ANALYSIS_SPEC

    features = tuple(features)
    analysis_spec = spec if spec is not None else DEFAULT_ANALYSIS_SPEC
    resolution_config = resolution_config or PeakResolutionConfig()

    # --- own-points feature matrix (row j <-> distribution.sample_ids[j]) ---
    points = np.column_stack([distribution.feature_column(f) for f in features])

    # --- grid built from THIS distribution's own points (preserved invariant) ---
    grid = build_grid(
        feature_names=features,
        pooled_values=points,
        fit_sample_ids=distribution.sample_ids,
        method=grid_method,
        params={**dict(grid_params or {}), "resolution": resolution},
    )
    canonical_grid = _grid_to_canonical(grid)

    # --- run the LIVE machinery (vote -> seed -> carve) ---
    record = DistributionRecord(
        distribution_id=distribution.distribution_id,
        points=points,
        analysis_context=DistributionAnalysisContext(grid=canonical_grid, spec=analysis_spec),
    )
    record = compute_resolved_peaks(record, resolution_config)
    resolved = record.resolved_peaks  # ResolvedPeakDistribution (never None here)

    # --- artifact density field (carries grid_id; distinct from TASK_C plot marginals) ---
    default_bw = float(np.median([np.abs(a[1] - a[0]) for a in grid.axis_values]))
    bandwidth = float(bandwidth) if bandwidth is not None else default_bw
    engine_density_grid = evaluate_density(grid, points, bandwidth)
    engine_density = np.asarray(engine_density_grid.density, dtype=float)

    basin_labels = resolved.empirical_basin_labels
    if basin_labels is None:
        basin_labels = np.zeros(engine_density.shape, dtype=int)
    basin_labels = np.asarray(basin_labels, dtype=int)

    artifacts = LabelGroupArtifacts(
        grid_id=grid.grid_id,
        grid=grid,
        density_grid=engine_density_grid,
        basin_labels=basin_labels,
        detection_result=resolved.detection_result,
    )

    # --- carve accepted, robust modes -> one label category each ---
    # sample_peak_ids is positional (0-based peak id per point, -1 = unassigned);
    # joins to REAL sample_ids via row order (points row j <-> sample_ids[j]).
    sample_peak_ids = (
        np.asarray(resolved.sample_peak_ids, dtype=int)
        if resolved.sample_peak_ids is not None
        else np.full(len(points), -1, dtype=int)
    )

    assignments: dict[str, str] = {}
    geometry_by_category: dict[str, CategoryShape] = {}
    per_category_metrics: dict[str, dict[str, float]] = {}

    peaks = tuple(resolved.peaks)
    heights = []
    for peak in peaks:
        detail = peak.detector_detail
        heights.append(float(detail.peak_height) if detail is not None else float("nan"))
    max_height = float(np.nanmax(heights)) if heights and np.any(np.isfinite(heights)) else float("nan")
    supports = [float(p.geometry.total_support_fraction) for p in peaks]
    # prominence_rank: 1 = most-supported (descending support), ties broken by order.
    rank_order = sorted(range(len(peaks)), key=lambda i: -supports[i]) if peaks else []
    prominence_rank = {i: r + 1 for r, i in enumerate(rank_order)}
    dominant_idx = rank_order[0] if rank_order else None

    for local_idx, peak in enumerate(peaks):
        geom = peak.geometry
        peak_id = int(geom.peak_id)  # 0-based positional id used in sample_peak_ids
        category = f"peak_{peak_id}"

        member_mask = sample_peak_ids == peak_id
        member_indices = np.nonzero(member_mask)[0]
        for i in member_indices:
            assignments[distribution.sample_ids[i]] = category

        # INTRINSIC geometry only (feature units, carries grid_id + feature_names):
        #   center_coordinate -> center; radius -> radius;
        #   within_peak_r80_density -> r80; cv_radius_from_center -> cv_radius_from_center.
        geometry = SampleSetGeometry(
            grid_id=grid.grid_id,
            feature_names=features,
            center=np.asarray(geom.center_coordinate, dtype=float),
            radius=float(geom.radius),
            r80=float(geom.within_peak_r80_density),
            cv_radius_from_center=float(geom.cv_radius_from_center),
        )

        # Per-peak HDR on the shared engine grid (raster-comparable via grid_id).
        # peak_id is 0-based; the live basin raster is 0-based too (-1 unassigned,
        # 0..K-1 basins) — match on peak_id directly.
        hdr = HDR(
            grid_id=grid.grid_id,
            feature_names=features,
            level=hdr_level,
            mask=_hdr_mask_for_basin(engine_density, basin_labels, peak_id, hdr_level),
        )

        # EAGER per-category shape (geometry + HDR) — Distribution.sample_sets
        # unpacks this CategoryShape into the SampleSet's typed geometry/hdr slots.
        geometry_by_category[category] = CategoryShape(geometry=geometry, hdr=hdr)

        detail = peak.detector_detail
        height = float(detail.peak_height) if detail is not None else float("nan")
        per_category_metrics[category] = {
            "support_fraction": float(geom.total_support_fraction),
            "prominence_rank": float(prominence_rank[local_idx]),
            "height_relative_to_max": (
                float(height / max_height) if np.isfinite(max_height) and max_height > 0 else float("nan")
            ),
            "is_dominant": 1.0 if local_idx == dominant_idx else 0.0,
        }

    # vote / count_stability / is_reliable -> run-level evidence (spec dict).
    evidence = resolved.resolution_evidence
    run_evidence: dict[str, Any] = {
        "is_reliable": bool(resolved.is_reliable),
        "resolved_peak_count": len(peaks),  # DERIVED = len(sample_set_ids); recorded for audit
        "per_sample_set_metrics": per_category_metrics,
        "artifacts": artifacts,
    }
    if evidence is not None:
        stability = evidence.count_stability
        run_evidence["vote"] = {
            "peak_count_frequencies": dict(stability.vote.peak_count_frequencies),
            "n_draws_requested": int(stability.vote.n_draws_requested),
            "n_draws_valid": int(stability.vote.n_draws_valid),
            "mode_peak_count": stability.mode_peak_count,
            "mode_frequency": float(stability.mode_frequency),
            "count_is_stable": bool(stability.count_is_stable),
            "resolution_succeeded": bool(evidence.resolution_succeeded),
            "basin_validation": tuple(bool(b) for b in evidence.basin_validation),
        }

    provenance = LabelProvenance(
        method="detect_peaks",
        features=features,
        spec={
            "bandwidth_rule": analysis_spec.bandwidth_rule,
            "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
            "peak_detector_method": analysis_spec.peak_detector_method,
            **run_evidence,
        },
        # EAGER per-category CategoryShape (geometry + HDR) — Distribution.
        # sample_sets (objects.py, TASK_0) unpacks it into the SampleSet's typed
        # geometry/hdr slots, so callers read a proper SampleSetGeometry / HDR
        # off sample_sets("resolved_peak") directly (no wrapper, no unpack helper).
        geometry=geometry_by_category,
    )

    new_distribution = distribution.with_label(
        output_label, assignments, provenance=provenance
    )
    return new_distribution.label_group(output_label, display_name=output_label)
