"""The single conversion boundary from the core peak resolver to the engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from .objects import (
    DensityEstimate,
    Distribution,
    LabelGroup,
    LabelingProvenance,
    SampleSetGeometry,
    UNASSIGNED_LABEL,
)


def label_group_from_resolved_peaks(
    distribution: Distribution,
    resolved_peak_distribution: Any,
    *,
    name: str,
    density: DensityEstimate,
) -> LabelGroup:
    """Transfer one authoritative resolver result into the catalog ontology.

    This function deliberately performs no density fitting, candidate detection,
    voting, basin reconstruction, or peak-count interpretation.  The resolver's
    sample assignments, geometry, vote interpretation, and provenance are copied
    into their typed engine fields.
    """
    if density.distribution_id != distribution.distribution_id:
        raise ValueError("generating density belongs to another distribution")
    if density.feature_names != distribution.feature_names:
        raise ValueError("generating density ordered features do not match distribution")
    if resolved_peak_distribution.distribution_id != distribution.distribution_id:
        raise ValueError("resolved peaks belong to another distribution")

    evidence = resolved_peak_distribution.resolution_evidence
    if evidence is None:
        raise ValueError("resolved peaks must retain peak-count voting evidence")
    summary = evidence.peak_resolution_summary

    raw_ids = resolved_peak_distribution.sample_peak_ids
    if raw_ids is None:
        if distribution.sample_ids:
            raise ValueError("resolved peaks do not contain sample assignments")
        sample_peak_ids = np.empty(0, dtype=int)
    else:
        sample_peak_ids = np.asarray(raw_ids, dtype=int)
    if sample_peak_ids.ndim != 1 or len(sample_peak_ids) != len(distribution.sample_ids):
        raise ValueError("resolved sample assignments must align 1:1 with distribution samples")

    peaks_by_id = {
        int(peak.geometry.peak_id): peak
        for peak in resolved_peak_distribution.peaks
    }
    if len(peaks_by_id) != len(resolved_peak_distribution.peaks):
        raise ValueError("resolved peak ids must be unique within a label group")
    assigned_ids = {int(value) for value in sample_peak_ids if int(value) >= 0}
    if not assigned_ids.issubset(peaks_by_id):
        raise ValueError("sample assignments reference an unknown resolved peak")

    assignments = {
        sample_id: (
            UNASSIGNED_LABEL if int(peak_id) < 0 else f"peak_{int(peak_id)}"
        )
        for sample_id, peak_id in zip(distribution.sample_ids, sample_peak_ids)
    }
    geometries = {}
    for peak_id, peak in peaks_by_id.items():
        geometry = peak.geometry
        geometries[f"peak_{peak_id}"] = SampleSetGeometry(
            feature_names=density.feature_names,
            center=np.asarray(geometry.center_coordinate, dtype=float),
            radius=float(geometry.radius),
            support_fraction=float(geometry.total_support_fraction),
            r80_radial_concentration=float(geometry.within_peak_r80_density),
            cv_radius_from_center=float(geometry.cv_radius_from_center),
        )

    return LabelGroup(
        name=name,
        distribution_id=distribution.distribution_id,
        assignments=assignments,
        density=density,
        sample_set_geometries=geometries,
        peak_resolution_summary=summary,
        labeling_provenance=LabelingProvenance(
            method="resolved_peaks",
            detail={
                "source_type": resolved_peak_distribution.source_type,
                "resolver_provenance": resolved_peak_distribution.provenance,
            },
        ),
    )
