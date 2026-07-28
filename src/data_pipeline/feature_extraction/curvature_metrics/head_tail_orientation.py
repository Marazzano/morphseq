"""Orient an extracted centerline head→tail by body-width tapering.

Ported (behavior-preserving) from the legacy body-axis pipeline
(``segmentation_sandbox/scripts/body_axis_analysis/spline_utils.py`` — only the head/tail functions
the geodesic path actually used). The geodesic endpoints are unordered (farthest-apart pair), so
before any head-anchored metric we fix the direction: a zebrafish embryo is widest at the head and
tapers to the tail, so we sample width along the spline and put the wide end first.

Orientation matters for the keypoint-deviation metrics (which are indexed as fractions along the
spline from the head); the length / mean-curvature / baseline-deviation summaries are
orientation-invariant.
"""

from __future__ import annotations

import numpy as np

_WIDTH_SAMPLE_COUNT: int = 20
_WIDTH_WINDOW_PX: int = 10


def orient_centerline_head_to_tail(centerline_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``centerline_xy`` reversed if needed so the head (wider end) is first."""
    if len(centerline_xy) == 0:
        return centerline_xy
    return centerline_xy[::-1] if _head_is_at_end(centerline_xy, mask) else centerline_xy


def _head_is_at_end(centerline_xy: np.ndarray, mask: np.ndarray) -> bool:
    """True if the *last* spline point is the head (body width increases start→end)."""
    n_samples = min(_WIDTH_SAMPLE_COUNT, len(centerline_xy))
    sample_indices = np.linspace(0, len(centerline_xy) - 1, n_samples, dtype=int)
    samples = centerline_xy[sample_indices]

    widths = np.array([
        _perpendicular_width(mask, samples, i) for i in range(len(samples))
    ])
    # Sign of the linear trend: positive slope => width grows toward the end => head is at the end.
    slope = np.polyfit(np.arange(len(widths)), widths, 1)[0]
    return slope > 0


def _perpendicular_width(mask: np.ndarray, samples: np.ndarray, i: int) -> float:
    """Body width at sample ``i``: local radius at the ends, perpendicular scan in the interior."""
    if i == 0 or i == len(samples) - 1:
        return _local_width(mask, samples[i], _WIDTH_WINDOW_PX)

    tangent = samples[i + 1] - samples[i - 1]
    perpendicular = np.array([-tangent[1], tangent[0]], dtype=float)
    perpendicular /= np.linalg.norm(perpendicular) + 1e-10
    return _width_along_direction(mask, samples[i], perpendicular, _WIDTH_WINDOW_PX)


def _local_width(mask: np.ndarray, point: np.ndarray, radius_px: int) -> float:
    """Width as the diameter of the equivalent-area disc of mask within ``radius_px`` of ``point``."""
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= radius_px**2
    area = int(np.sum(mask & disc))
    return 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0


def _width_along_direction(
    mask: np.ndarray, point: np.ndarray, direction: np.ndarray, max_dist_px: int
) -> float:
    """Width as the run of mask pixels stepping ``±direction`` from ``point`` until it leaves the mask."""
    def run_length(sign: int) -> int:
        reached = 0
        for step in range(1, max_dist_px):
            p = point + sign * step * direction
            x, y = int(p[0]), int(p[1])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                reached = step
            else:
                break
        return reached

    return run_length(+1) + run_length(-1)
