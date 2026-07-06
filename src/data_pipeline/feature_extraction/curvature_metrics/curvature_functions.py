"""Shape/curvature summary functions computed from an extracted head→tail centerline spline.

Ported (behavior-preserving) from the legacy body-axis pipeline
(``segmentation_sandbox/scripts/body_axis_analysis/curvature_metrics.py``), plus the
length-normalized baseline deviation that legacy computed one layer up in
``src/build/build04_perform_embryo_qc.py`` (``baseline_deviation_um / total_length_um``). That
normalized value is the headline curvature feature every downstream analysis keys on: it measures
how far the midline bows from a straight head-to-tail line, as a fraction of body length, so it is
comparable across embryos of different size and stage.

These functions take spline coordinates (in pixels) + a micron scale and return scalars. They own
no I/O and no mask logic — the centerline is produced by ``geodesic_centerline`` and oriented by
``head_tail_orientation`` before it reaches here.
"""

from __future__ import annotations

import numpy as np

_KEYPOINT_FRACTIONS: dict[str, float] = {"q1": 0.25, "mid": 0.5, "q3": 0.75}


def _perpendicular_distance_to_chord(
    px: np.ndarray, py: np.ndarray, head_xy: np.ndarray, tail_xy: np.ndarray
) -> np.ndarray:
    """Perpendicular distance from each point (px, py) to the head→tail chord.

    Chord as line ``ax + by + c = 0`` through head/tail; if head==tail (degenerate) fall back to the
    plain distance to the head point.
    """
    a = tail_xy[1] - head_xy[1]
    b = -(tail_xy[0] - head_xy[0])
    c = (tail_xy[0] - head_xy[0]) * head_xy[1] - (tail_xy[1] - head_xy[1]) * head_xy[0]
    denom = np.hypot(a, b)
    if denom == 0:
        return np.hypot(px - head_xy[0], py - head_xy[1])
    return np.abs(a * px + b * py + c) / denom


def compute_curvature_summary(
    spline_xy: np.ndarray, pixel_size_um: float, mean_curvature_per_um: float
) -> dict[str, float]:
    """Return the full shape-summary metric dict for one oriented head→tail spline.

    ``spline_xy`` is the (200, 2) B-spline centerline in pixels, oriented head-first.
    ``mean_curvature_per_um`` is the spline's mean analytic curvature (from ``geodesic_centerline``),
    passed through so all summary values live in one dict.
    """
    x, y = spline_xy[:, 0], spline_xy[:, 1]
    head_xy, tail_xy = spline_xy[0], spline_xy[-1]

    deviation_px = _perpendicular_distance_to_chord(x, y, head_xy, tail_xy)
    deviation_um = deviation_px * pixel_size_um
    baseline_deviation_um = float(np.mean(deviation_um))

    arc_length_um = float(np.sum(np.hypot(np.diff(x), np.diff(y))) * pixel_size_um)
    chord_length_um = float(np.hypot(tail_xy[0] - head_xy[0], tail_xy[1] - head_xy[1]) * pixel_size_um)
    arc_length_ratio = arc_length_um / chord_length_um if chord_length_um > 0 else 1.0

    # Headline feature: bow as a fraction of body length (scale-free across size/stage).
    baseline_deviation_normalized = (
        baseline_deviation_um / arc_length_um if arc_length_um > 0 else np.nan
    )

    summary = {
        "total_length_um": arc_length_um,
        "mean_curvature_per_um": mean_curvature_per_um,
        "baseline_deviation_um": baseline_deviation_um,
        "baseline_deviation_normalized": baseline_deviation_normalized,
        "max_baseline_deviation_um": float(np.max(deviation_um)),
        "baseline_deviation_std_um": float(np.std(deviation_um)),
        "arc_length_ratio": arc_length_ratio,
        "chord_length_um": chord_length_um,
    }
    summary.update(_keypoint_deviations(spline_xy, deviation_um))
    return summary


def _keypoint_deviations(spline_xy: np.ndarray, deviation_um: np.ndarray) -> dict[str, float]:
    """Baseline deviation sampled at the quarter / mid / three-quarter points along the spline —
    *where* the body bends, not just how much."""
    n = len(spline_xy)
    result: dict[str, float] = {}
    for name, fraction in _KEYPOINT_FRACTIONS.items():
        idx = int(np.clip(round(fraction * (n - 1)), 0, n - 1))
        result[f"keypoint_deviation_{name}_um"] = float(deviation_um[idx])
    return result
