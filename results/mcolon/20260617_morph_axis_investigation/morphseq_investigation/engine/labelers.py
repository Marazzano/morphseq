"""Labelers — the peer contract (ontology "Labelers are peers", §2, §3).

A labeler is ``label(distribution, method, params) -> (LabelGroup, [SampleSet])``.
``genotype`` (provided / column) and ``peak_finding`` (unsupervised, folding in the
live ``core.distribution_records.compute_resolved_peaks`` machinery) are PEERS:
both run the SAME skeleton (:func:`_finalize`), differing only in which optional
SampleSet/LabelGroup slots fill. There is NO ``if genotype ... else if peak ...``
anywhere — the two share one return path and one ``validate_label_group`` call.

  genotype  : geometry=None, provenance.labeler.feature_names=(), artifacts=None,
              usually unassigned=(). Three distinct beasts kept apart (§2):
                * ``unlabeled`` = a REAL SampleSet (literal category, is_missing_value=False)
                * NA source value -> a REAL SampleSet, is_missing_value=True
                * labeler abstention -> LabelGroup.unassigned_sample_ids (no SampleSet)
              Unknown genotype = a REAL SampleSet named "unknown" with its own HDR.

  peak_finding : one SampleSet per accepted, robust mode (geometry filled from
              PeakGeometry), + unassigned residual. The bootstrap VOTE decides the
              count BEFORE carving — two clusters voting to one mode -> ONE SampleSet;
              rejected candidates never become phantom SampleSets (the collapse story
              stays in LabelGroup.provenance). Both target & reference peak runs
              evaluate on ONE grid built from POOLED features (§1b) -> same grid_id
              -> raster-comparable. The truth/empirical branch is DROPPED (real data
              is always empirical; source_type / grid_peak_ids excluded, #9).

Field-by-field peak destination (ontology "Peak-finding as a labeler" table):
  accepted ResolvedPeak (each)          -> one SampleSet
  PeakGeometry (INTRINSIC only)         -> SampleSet.geometry
     center_coordinate                  -> center
     radius                             -> radius
     within_peak_r80_density            -> r80
     cv_radius_from_center              -> cv_radius_from_center
     total_support_fraction  (RUN-RELATIVE) -> LabelGroup.per_sample_set_metrics
  sample_peak_ids (positional)          -> real ids -> SampleSet.sample_ids
                                            + LabelGroup.sample_id_to_sample_set_id
  PeakCandidateDetail / basin validation -> SampleSet.provenance.evidence
  vote / count_stability / is_reliable  -> LabelGroup.provenance
  density_grid / basin_labels / detection_result -> LabelGroup.artifacts
  resolved_peak_count                   -> derived = len(sample_set_ids)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .grid import build_grid, evaluate_density
from .identifiers import make_sample_set_id
from .invariants import validate_label_group
from .objects import (
    DensityGrid,
    Distribution,
    DistributionLabelGroup,
    Grid,
    HDR,
    LabelGroup,
    LabelGroupArtifacts,
    LabelProvenance,
    SampleSet,
    SampleSetGeometry,
)

# A source value counts as "missing" (NA) when it is None or a float NaN. Everything
# else — including the literal string "unlabeled" and "unknown" — is a real category
# and becomes a real SampleSet (§2: unlabeled != NA != abstention).
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))  # float NaN
    except (TypeError, ValueError):
        return False


def _feature_profile_for(distribution: Distribution, member_indices: Sequence[int]):
    """Per-feature mean/std over the members, in FEATURE units (self-describing).

    Any feature in the Distribution's menu can be profiled on any set (§2).
    Returns ``None`` for an empty set (no meaningful stats).
    """
    from .objects import FeatureProfile

    if len(member_indices) == 0:
        return None
    values = distribution.feature_values[np.asarray(member_indices, dtype=int), :]
    return FeatureProfile(
        feature_names=distribution.feature_names,
        mean=values.mean(axis=0),
        std=values.std(axis=0),
    )


# --------------------------------------------------------------------------- #
# The ONE shared skeleton both labelers return through.
# --------------------------------------------------------------------------- #
def _finalize(
    distribution: Distribution,
    label_group_name: str,
    sample_sets: Sequence[SampleSet],
    assignment: Mapping[str, str],
    *,
    unassigned_sample_ids: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    artifacts: LabelGroupArtifacts | None = None,
    per_sample_set_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[LabelGroup, list[SampleSet]]:
    """Assemble the LabelGroup, validate centrally, and return the peer tuple.

    Both labelers funnel here — one return path, one ``validate_label_group``
    call (the proof the ontology holds: no per-labeler drift check).
    """
    sample_sets = list(sample_sets)
    label_group = LabelGroup(
        label_group_name=label_group_name,
        distribution_id=distribution.distribution_id,
        sample_set_ids=tuple(s.sample_set_id for s in sample_sets),
        sample_id_to_sample_set_id=dict(assignment),
        unassigned_sample_ids=tuple(unassigned_sample_ids),
        provenance=dict(provenance or {}),
        artifacts=artifacts,
        per_sample_set_metrics=dict(per_sample_set_metrics or {}),
    )
    validate_label_group(distribution, label_group, sample_sets)
    return label_group, sample_sets


# --------------------------------------------------------------------------- #
# genotype (provided / column) labeler
# --------------------------------------------------------------------------- #
def label_genotype(
    distribution: Distribution,
    method: str = "column",
    params: Mapping[str, Any] | None = None,
) -> tuple[LabelGroup, list[SampleSet]]:
    """Provided/column labeler — one SampleSet per category value.

    The column is handed in via ``params`` (the Distribution NEVER parses ids, §1):

      params["labels"]  : a sequence aligned 1:1 with ``distribution.sample_ids``
                          (the resolved label per sample). REQUIRED.
      params["column"]  : the human name of the source column (recorded in
                          provenance; documentation only). Optional.

    NA source values (None / NaN) collapse into a SINGLE real SampleSet whose name
    is ``params.get("missing_name", "unknown")`` with ``evidence.is_missing_value=
    True``. Every other distinct value — INCLUDING the literal strings "unlabeled"
    and "unknown" — is its own real SampleSet (``is_missing_value=False``). A
    genotype labeler does not abstain, so ``unassigned_sample_ids`` is typically
    ``()`` (a caller may pre-place samples in ``params["unassigned"]`` — those are
    the only abstentions).
    """
    if method != "column":
        raise ValueError(f"genotype labeler only implements method='column'; got {method!r}")
    params = dict(params or {})
    if "labels" not in params:
        raise ValueError(
            "genotype labeler requires params['labels'] aligned to distribution.sample_ids"
        )
    labels = list(params["labels"])
    if len(labels) != len(distribution.sample_ids):
        raise ValueError(
            "params['labels'] must align 1:1 with distribution.sample_ids: "
            f"{len(labels)} labels vs {len(distribution.sample_ids)} samples"
        )
    missing_name = str(params.get("missing_name", "unknown"))
    forced_unassigned = set(params.get("unassigned", ()))

    # Group sample indices by their resolved category (NA -> the missing bucket),
    # preserving first-seen category order for stable SampleSet ids.
    category_order: list[str] = []
    category_indices: dict[str, list[int]] = {}
    category_is_missing: dict[str, bool] = {}
    assignment: dict[str, str] = {}
    unassigned: list[str] = []

    for idx, sample_id in enumerate(distribution.sample_ids):
        if sample_id in forced_unassigned:
            unassigned.append(sample_id)
            continue
        raw = labels[idx]
        if _is_missing(raw):
            category = missing_name
            is_missing = True
        else:
            category = str(raw)
            is_missing = False
        if category not in category_indices:
            category_order.append(category)
            category_indices[category] = []
            category_is_missing[category] = is_missing
        # A literal "unknown" string and NA both map to the same bucket name only
        # if the caller uses the default missing_name; if a real "unknown" category
        # coexists with NA, the is_missing flag reflects whichever was seen first —
        # callers wanting them distinct should pass a non-colliding missing_name.
        category_indices[category].append(idx)
        assignment[sample_id] = ""  # filled below once ids are minted

    sample_sets: list[SampleSet] = []
    for category in category_order:
        member_indices = category_indices[category]
        member_ids = tuple(distribution.sample_ids[i] for i in member_indices)
        set_id = make_sample_set_id(distribution.distribution_id, category)
        for sid in member_ids:
            assignment[sid] = set_id
        sample_sets.append(
            SampleSet(
                sample_set_id=set_id,
                sample_set_name=category,
                distribution_id=distribution.distribution_id,
                sample_ids=member_ids,
                feature_profile=_feature_profile_for(distribution, member_indices),
                hdr=None,  # a caller may attach an HDR later; not this labeler's job
                geometry=None,  # provided labels carry no measured center (§2)
                provenance={
                    "labeler": {
                        "method": "column",
                        "params": {"column": params.get("column")},
                        "feature_names": (),  # genotype uses no features to FORM groups
                    },
                    "evidence": {
                        "source_value": category,
                        "is_missing_value": category_is_missing[category],
                    },
                },
            )
        )

    return _finalize(
        distribution,
        label_group_name=str(params.get("label_group_name", "genotype")),
        sample_sets=sample_sets,
        assignment=assignment,
        unassigned_sample_ids=tuple(unassigned),
        provenance={
            "labeler": {
                "method": "column",
                "params": {"column": params.get("column")},
                "feature_names": (),
            }
        },
        artifacts=None,
    )


# --------------------------------------------------------------------------- #
# peak_finding (unsupervised) labeler — folds in the live machinery.
# --------------------------------------------------------------------------- #
def _grid_to_canonical(grid: Grid):
    """Bridge a TASK_A 2-D feature-unit :class:`Grid` to a live ``CanonicalGrid``.

    The live ``compute_resolved_peaks`` machinery (``core.distribution_records``)
    is 2-D-only and speaks ``CanonicalGrid`` (x/y bounds + a single ``grid_size``).
    TASK_A's ``build_grid`` produces equal-length axes from one ``resolution``, so a
    square ``CanonicalGrid`` reproduces the SAME evaluation cell coordinates — the
    axes stay in feature units on both sides (no basis change). We assert the two
    axes have equal length (CanonicalGrid cannot express a non-square grid).
    """
    from ..core.density_composition import CanonicalGrid

    if len(grid.feature_names) != 2:
        raise ValueError(
            "peak_finding labeler currently supports 2-D grids only "
            f"(the live detector is 2-D); got feature_names={grid.feature_names!r}"
        )
    x_axis, y_axis = grid.axis_values
    if len(x_axis) != len(y_axis):
        raise ValueError(
            "peak_finding requires a square grid (equal axis lengths) to map to "
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

    Returned mask is on the SAME raster as ``density`` (the engine grid), so it is
    raster-comparable to any other set's HDR sharing the grid_id.
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


def label_peak_finding(
    distribution: Distribution,
    method: str = "peak_finding",
    params: Mapping[str, Any] | None = None,
) -> tuple[LabelGroup, list[SampleSet]]:
    """Unsupervised peak labeler — re-expresses ``compute_resolved_peaks``.

    The RETURNED primitive is ``(LabelGroup, [SampleSet])``; the live
    ``ResolvedPeakDistribution`` survives only as the internal builder.

    ``params`` (§1b — the pooled grid is the caller's; we accept it):
      params["grid"]           : a TASK_A :class:`Grid` (2-D, feature-unit axes)
                                 built from POOLED target+reference features, so
                                 target & reference peak runs share one grid_id
                                 (raster-comparable). REQUIRED.
      params["analysis_spec"]  : a ``core.resolved_peak_analysis.ResolvedPeakAnalysisSpec``.
                                 Defaults to that module's DEFAULT_ANALYSIS_SPEC.
      params["resolution_config"] : a ``core.distribution_records.PeakResolutionConfig``
                                 (bootstrap vote knobs). Optional (defaults apply).
      params["bandwidth"]      : scalar KDE bandwidth for the ARTIFACT density_grid
                                 (the TASK_A engine DensityGrid tagged with grid_id).
                                 Defaults to the grid's median axis spacing.
      params["hdr_level"]      : HDR fraction per peak (default 0.80).

    The bootstrap VOTE (inside ``compute_resolved_peaks``) decides the mode count
    BEFORE carving: two clusters voting to one mode -> ONE SampleSet. Rejected
    candidates never become SampleSets; that collapse story lives in
    ``LabelGroup.provenance`` (vote frequencies / count_stability / is_reliable).
    """
    from ..core.distribution_records import (
        DistributionAnalysisContext,
        DistributionRecord,
        PeakResolutionConfig,
        compute_resolved_peaks,
    )
    from ..core.resolved_peak_analysis import DEFAULT_ANALYSIS_SPEC

    if method != "peak_finding":
        raise ValueError(f"peak labeler only implements method='peak_finding'; got {method!r}")
    params = dict(params or {})
    grid = params.get("grid")
    if not isinstance(grid, Grid):
        raise ValueError("peak_finding requires params['grid'] = a TASK_A engine Grid")
    if tuple(grid.feature_names) != tuple(distribution.feature_names):
        raise ValueError(
            "params['grid'].feature_names must match distribution.feature_names "
            f"({grid.feature_names!r} != {distribution.feature_names!r})"
        )

    analysis_spec = params.get("analysis_spec", DEFAULT_ANALYSIS_SPEC)
    resolution_config = params.get("resolution_config") or PeakResolutionConfig()
    hdr_level = float(params.get("hdr_level", 0.80))

    canonical_grid = _grid_to_canonical(grid)
    points = np.asarray(distribution.feature_values, dtype=float)

    # --- run the LIVE machinery (vote -> seed -> carve) as the internal builder ---
    record = DistributionRecord(
        distribution_id=distribution.distribution_id,
        points=points,
        analysis_context=DistributionAnalysisContext(grid=canonical_grid, spec=analysis_spec),
    )
    record = compute_resolved_peaks(record, resolution_config)
    resolved = record.resolved_peaks  # ResolvedPeakDistribution | None (never None here)

    # --- build the TASK_A engine DensityGrid for artifacts (carries grid_id) ---
    default_bw = float(np.median([np.abs(a[1] - a[0]) for a in grid.axis_values]))
    bandwidth = float(params.get("bandwidth", default_bw))
    engine_density_grid: DensityGrid = evaluate_density(grid, points, bandwidth)
    engine_density = np.asarray(engine_density_grid.density, dtype=float)

    # The live basin raster is on the CanonicalGrid (xy meshgrid, shape
    # (grid_size, grid_size)); TASK_A's density is on the same square shape (ij),
    # numerically the same cell coordinates. Keep the live basin_labels as the
    # artifact raster (used for HDR carving), tagged to the engine grid_id.
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

    # --- carve accepted, robust modes -> one SampleSet each ---------------------
    # sample_peak_ids is positional (0-based peak id per point, -1 = unassigned);
    # join to REAL sample_ids via row order (Distribution.feature_values row j
    # <-> sample_ids[j]).
    sample_peak_ids = (
        np.asarray(resolved.sample_peak_ids, dtype=int)
        if resolved.sample_peak_ids is not None
        else np.full(len(points), -1, dtype=int)
    )

    sample_sets: list[SampleSet] = []
    assignment: dict[str, str] = {}
    per_sample_set_metrics: dict[str, dict[str, float]] = {}

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
        name = f"peak_{peak_id}"
        set_id = make_sample_set_id(distribution.distribution_id, name)

        member_mask = sample_peak_ids == peak_id
        member_indices = np.nonzero(member_mask)[0]
        member_ids = tuple(distribution.sample_ids[i] for i in member_indices)
        for sid in member_ids:
            assignment[sid] = set_id

        # INTRINSIC geometry only (feature units, carries grid_id + feature_names):
        #   center_coordinate -> center; radius -> radius;
        #   within_peak_r80_density -> r80; cv_radius_from_center -> cv_radius_from_center.
        # RUN-RELATIVE total_support_fraction is NOT here (it -> per_sample_set_metrics).
        geometry = SampleSetGeometry(
            grid_id=grid.grid_id,
            feature_names=distribution.feature_names,
            center=np.asarray(geom.center_coordinate, dtype=float),
            radius=float(geom.radius),
            r80=float(geom.within_peak_r80_density),
            cv_radius_from_center=float(geom.cv_radius_from_center),
        )

        # Per-peak HDR on the shared engine grid (raster-comparable via grid_id).
        # peak_id is 0-based; the live basin raster is 0-based too (-1 unassigned,
        # 0..K-1 basins) — see _carve_final_peaks. Match on peak_id directly.
        hdr = HDR(
            grid_id=grid.grid_id,
            feature_names=distribution.feature_names,
            level=hdr_level,
            mask=_hdr_mask_for_basin(engine_density, basin_labels, peak_id, hdr_level),
        )

        detail = peak.detector_detail
        sample_sets.append(
            SampleSet(
                sample_set_id=set_id,
                sample_set_name=name,
                distribution_id=distribution.distribution_id,
                sample_ids=member_ids,
                feature_profile=_feature_profile_for(distribution, member_indices),
                hdr=hdr,
                geometry=geometry,
                provenance={
                    "labeler": {
                        "method": "peak_finding",
                        "params": {
                            "bandwidth_rule": analysis_spec.bandwidth_rule,
                            "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
                            "peak_detector_method": analysis_spec.peak_detector_method,
                        },
                        # the features the labeler USED to FORM the groups (§2)
                        "feature_names": tuple(grid.feature_names),
                    },
                    "evidence": {
                        # PeakCandidateDetail (per-candidate) + basin validation.
                        "detector_detail": _detail_to_dict(detail),
                        "peak_provenance": dict(peak.provenance),
                    },
                },
            )
        )

        # RUN-RELATIVE (sibling-relative) metrics live on the LabelGroup (§2, #4).
        height = float(detail.peak_height) if detail is not None else float("nan")
        per_sample_set_metrics[set_id] = {
            "support_fraction": float(geom.total_support_fraction),
            "prominence_rank": float(prominence_rank[local_idx]),
            "height_relative_to_max": (
                float(height / max_height) if np.isfinite(max_height) and max_height > 0 else float("nan")
            ),
            "is_dominant": 1.0 if local_idx == dominant_idx else 0.0,
        }

    # unassigned = real Distribution samples the resolve left in no accepted mode.
    assigned_ids = set(assignment)
    unassigned = tuple(sid for sid in distribution.sample_ids if sid not in assigned_ids)

    # vote / count_stability / is_reliable -> LabelGroup.provenance.
    evidence = resolved.resolution_evidence
    provenance: dict[str, Any] = {
        "labeler": {
            "method": "peak_finding",
            "params": {
                "bandwidth_rule": analysis_spec.bandwidth_rule,
                "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
                "peak_detector_method": analysis_spec.peak_detector_method,
            },
            "feature_names": tuple(grid.feature_names),
        },
        "is_reliable": bool(resolved.is_reliable),
        "resolved_peak_count": (  # DERIVED = len(sample_set_ids); recorded for audit
            None if resolved.resolved_peak_count is None else int(len(sample_sets))
        ),
    }
    if evidence is not None:
        stability = evidence.count_stability
        provenance["vote"] = {
            "peak_count_frequencies": dict(stability.vote.peak_count_frequencies),
            "n_draws_requested": int(stability.vote.n_draws_requested),
            "n_draws_valid": int(stability.vote.n_draws_valid),
            "mode_peak_count": stability.mode_peak_count,
            "mode_frequency": float(stability.mode_frequency),
            "count_is_stable": bool(stability.count_is_stable),
            "resolution_succeeded": bool(evidence.resolution_succeeded),
            "basin_validation": tuple(bool(b) for b in evidence.basin_validation),
        }

    return _finalize(
        distribution,
        label_group_name=str(params.get("label_group_name", "peak")),
        sample_sets=sample_sets,
        assignment=assignment,
        unassigned_sample_ids=unassigned,
        provenance=provenance,
        artifacts=artifacts,
        per_sample_set_metrics=per_sample_set_metrics,
    )


def _detail_to_dict(detail) -> dict[str, Any] | None:
    """Flatten a ``PeakCandidateDetail`` to a plain dict for provenance.evidence."""
    if detail is None:
        return None
    return {
        "candidate_peak_id": int(detail.candidate_peak_id),
        "peak_x": None if detail.peak_x is None else float(detail.peak_x),
        "peak_y": None if detail.peak_y is None else float(detail.peak_y),
        "peak_height": float(detail.peak_height),
        "prominence_ratio": None if detail.prominence_ratio is None else float(detail.prominence_ratio),
        "basin_sample_count": detail.basin_sample_count,
        "basin_sample_fraction": (
            None if detail.basin_sample_fraction is None else float(detail.basin_sample_fraction)
        ),
        "accepted": bool(detail.accepted),
        "reject_reason": detail.reject_reason,
    }


# --------------------------------------------------------------------------- #
# detect_peaks (TASK_B, commit 1) — writes the resolved_peak label column onto
# a Distribution. Calls into the SAME live machinery as label_peak_finding
# above; geometry/HDR eager-attachment + retirement of the old tuple-return
# labelers lands in the next checkpoint.
# --------------------------------------------------------------------------- #
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
) -> DistributionLabelGroup:
    """Fit peaks on ``distribution``'s OWN points for ``features`` and write a
    label column named ``output_label``. This is the BODY of
    ``Distribution.detect_peaks`` (TASK_0 stub); called from there so the
    public surface stays ``distribution.detect_peaks(...)``.

    The grid is built from THIS distribution's own points ONLY (no pooled
    target/reference grid). Peak ids are LOCAL: two independently-run
    distributions may both produce a ``peak_0`` and nothing here claims they
    correspond.
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

    points = np.column_stack([distribution.feature_column(f) for f in features])

    grid = build_grid(
        feature_names=features,
        pooled_values=points,
        fit_sample_ids=distribution.sample_ids,
        method=grid_method,
        params={**dict(grid_params or {}), "resolution": resolution},
    )
    canonical_grid = _grid_to_canonical(grid)

    record = DistributionRecord(
        distribution_id=distribution.distribution_id,
        points=points,
        analysis_context=DistributionAnalysisContext(grid=canonical_grid, spec=analysis_spec),
    )
    record = compute_resolved_peaks(record, resolution_config)
    resolved = record.resolved_peaks

    sample_peak_ids = (
        np.asarray(resolved.sample_peak_ids, dtype=int)
        if resolved.sample_peak_ids is not None
        else np.full(len(points), -1, dtype=int)
    )

    assignments: dict[str, str] = {}
    for peak in resolved.peaks:
        peak_id = int(peak.geometry.peak_id)
        category = f"peak_{peak_id}"
        member_indices = np.nonzero(sample_peak_ids == peak_id)[0]
        for i in member_indices:
            assignments[distribution.sample_ids[i]] = category

    provenance = LabelProvenance(
        method="detect_peaks",
        features=features,
        spec={
            "bandwidth_rule": analysis_spec.bandwidth_rule,
            "bandwidth_multiplier": analysis_spec.bandwidth_multiplier,
            "peak_detector_method": analysis_spec.peak_detector_method,
            "is_reliable": bool(resolved.is_reliable),
            "resolved_peak_count": len(resolved.peaks),
        },
    )

    new_distribution = distribution.with_label(
        output_label, assignments, provenance=provenance
    )
    return new_distribution.label_group(output_label, display_name=output_label)
