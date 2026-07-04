"""Geodesic centerline extraction + B-spline curvature — the validated legacy method.

Ported (behavior-preserving) from the legacy body-axis pipeline
(``segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py``, the ``fast=True`` graph
path that ``src/build/utils/curvature_utils.py`` shipped in production). Names and structure are
cleaned for the data_pipeline; the math is unchanged.

Why geodesic (not a bare skeleton PCA-sort): a zebrafish embryo can curl until its head sits beside
its tail. Ordering skeleton pixels by a PCA axis scrambles such a shape, and a raw skeleton branches
at fins — both inflate curvature. Instead we treat the skeleton as an 8-connected graph, find the two
pixels that are farthest apart *along the skeleton* (geodesic distance) as head/tail, trace the
shortest path between them, then fit a cubic B-spline and read curvature analytically off the spline.

Pipeline: preprocessed mask → skeletonize → 8-connected graph → largest connected component →
geodesic endpoints → Dijkstra path → cubic B-spline (200 pts) → analytic curvature κ(s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage import morphology

# Spline is always evaluated at this many points, so downstream stats compare like-for-like.
SPLINE_SAMPLE_COUNT: int = 200
# Above this skeleton size, sample candidate endpoints instead of running Dijkstra from every node.
ENDPOINT_SAMPLE_THRESHOLD: int = 100
# Cubic B-spline needs at least this many control points.
MIN_POINTS_FOR_SPLINE: int = 4
# 8-connected neighbour offsets (a skeleton is a 1px-wide 8-connected curve).
_EIGHT_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass(frozen=True)
class GeodesicCenterline:
    """Result of centerline extraction for one mask.

    ``smoothed_xy`` / ``arc_length_um`` / ``curvature_per_um`` are all length
    ``SPLINE_SAMPLE_COUNT`` when a spline could be fit; when the skeleton was too short for a spline
    they are empty and only ``raw_xy`` is populated (documented low-information case).
    """

    raw_xy: np.ndarray            # (N, 2) skeleton path in pixels, head→tail order (pre-smoothing)
    smoothed_xy: np.ndarray       # (200, 2) B-spline centerline in pixels; empty if unsplinable
    arc_length_um: np.ndarray     # (200,) cumulative arc length along the spline, in microns
    curvature_per_um: np.ndarray  # (200,) analytic curvature at each spline point, in 1/micron


def _build_skeleton_graph(skeleton_points_xy: np.ndarray) -> csr_matrix:
    """Return the 8-connected adjacency matrix over skeleton pixels (edge weight = pixel distance)."""
    point_to_index = {tuple(pt): idx for idx, pt in enumerate(skeleton_points_xy)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []

    for idx, (x, y) in enumerate(skeleton_points_xy):
        for dx, dy in _EIGHT_NEIGHBOUR_OFFSETS:
            neighbour_idx = point_to_index.get((x + dx, y + dy))
            if neighbour_idx is None or neighbour_idx <= idx:
                continue  # add each undirected edge once
            rows.append(idx)
            cols.append(neighbour_idx)
            weights.append(float(np.hypot(dx, dy)))

    if not rows:
        raise ValueError("skeleton graph has no edges (disconnected single pixels)")

    n = len(skeleton_points_xy)
    symmetric_data = weights + weights
    return csr_matrix((symmetric_data, (rows + cols, cols + rows)), shape=(n, n))


def _restrict_to_largest_component(
    skeleton_points_xy: np.ndarray, adjacency: csr_matrix
) -> tuple[np.ndarray, csr_matrix]:
    """Keep only the largest connected component (drops fin twigs and stray specks)."""
    n_components, labels = connected_components(adjacency, directed=False)
    if n_components <= 1:
        return skeleton_points_xy, adjacency

    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    keep = labels == largest_label
    keep_indices = np.where(keep)[0]

    old_to_new = np.full(len(skeleton_points_xy), -1, dtype=int)
    old_to_new[keep_indices] = np.arange(len(keep_indices))

    src, dst = adjacency.nonzero()
    edge_kept = keep[src] & keep[dst]
    kept_adjacency = csr_matrix(
        (adjacency.data[edge_kept], (old_to_new[src[edge_kept]], old_to_new[dst[edge_kept]])),
        shape=(len(keep_indices), len(keep_indices)),
    )
    return skeleton_points_xy[keep_indices], kept_adjacency


def _find_geodesic_endpoints(adjacency: csr_matrix, n_points: int, random_seed: int) -> tuple[int, int]:
    """Return the (start, end) node indices with the largest geodesic separation.

    For large skeletons we sample candidate start nodes (deterministically, via ``random_seed``)
    rather than run Dijkstra from every node — the farthest-apart pair is almost always found from a
    modest sample and it keeps the cost bounded.
    """
    if n_points > ENDPOINT_SAMPLE_THRESHOLD:
        rng = np.random.RandomState(random_seed)
        candidate_starts = rng.choice(n_points, size=ENDPOINT_SAMPLE_THRESHOLD, replace=False)
    else:
        candidate_starts = np.arange(n_points)

    best_pair = (0, min(1, n_points - 1))
    best_distance = 0.0
    for start in candidate_starts:
        distances = dijkstra(adjacency, indices=start, directed=False)
        farthest = int(np.argmax(distances))
        if np.isfinite(distances[farthest]) and distances[farthest] > best_distance:
            best_distance = float(distances[farthest])
            best_pair = (int(start), farthest)
    return best_pair


def _trace_geodesic_path(adjacency: csr_matrix, start_idx: int, end_idx: int, n_points: int) -> list[int]:
    """Return node indices of the shortest skeleton path start→end, ordered head→tail."""
    _, predecessors = dijkstra(adjacency, indices=start_idx, directed=False, return_predecessors=True)
    path = []
    current = end_idx
    while current != -9999 and current != start_idx:
        path.append(current)
        current = int(predecessors[current])
        if len(path) > n_points:  # guard against a malformed predecessor chain
            break
    path.append(start_idx)
    return path[::-1]


def _extract_skeleton_path_xy(mask: np.ndarray, random_seed: int) -> np.ndarray:
    """Return the ordered (x, y) skeleton centerline path in pixels (pre-smoothing)."""
    skeleton = morphology.skeletonize(np.ascontiguousarray(mask.astype(np.uint8)))
    y_skel, x_skel = np.where(skeleton)
    if len(y_skel) < 2:
        raise ValueError("skeleton has too few points")

    points_xy = np.column_stack([x_skel, y_skel])
    adjacency = _build_skeleton_graph(points_xy)
    points_xy, adjacency = _restrict_to_largest_component(points_xy, adjacency)

    n_points = len(points_xy)
    start_idx, end_idx = _find_geodesic_endpoints(adjacency, n_points, random_seed)
    path_indices = _trace_geodesic_path(adjacency, start_idx, end_idx, n_points)
    return points_xy[path_indices]


def _fit_bspline(centerline_xy: np.ndarray, smoothing: float):
    """Fit a cubic B-spline to the raw centerline; return its scipy tck, or None if too short.

    Smoothing scales with point count (``s = smoothing * N``) so a longer spine gets proportionally
    more slack, matching the legacy adaptive smoothing.
    """
    if len(centerline_xy) < MIN_POINTS_FOR_SPLINE:
        return None
    tck, _ = splprep(
        [centerline_xy[:, 0], centerline_xy[:, 1]],
        s=smoothing * len(centerline_xy),
        k=3,
    )
    return tck


def _sample_spline_curvature(tck, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the spline and its analytic curvature at ``SPLINE_SAMPLE_COUNT`` points.

    Curvature: κ = |x'y'' − y'x''| / (x'² + y'²)^{3/2}, read off the B-spline's analytic first and
    second derivatives (no noisy finite differences). Arc length and curvature are converted to
    microns / per-micron.
    """
    u = np.linspace(0.0, 1.0, SPLINE_SAMPLE_COUNT)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    curvature_per_px = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    arc_length_px = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

    smoothed_xy = np.column_stack([x, y])
    return smoothed_xy, arc_length_px * pixel_size_um, curvature_per_px / pixel_size_um


def extract_geodesic_centerline(
    mask: np.ndarray,
    *,
    pixel_size_um: float,
    smoothing: float = 5.0,
    random_seed: int = 42,
) -> GeodesicCenterline:
    """Extract the head→tail centerline of an embryo mask and its analytic B-spline curvature.

    The mask should already be boundary-smoothed (see ``mask_preprocessing.smooth_mask_boundary``).
    If the skeleton is too short to fit a cubic spline, the smoothed/curvature arrays come back empty
    (a documented low-information result) while ``raw_xy`` still carries the short skeleton path.
    """
    raw_xy = _extract_skeleton_path_xy(mask, random_seed)
    tck = _fit_bspline(raw_xy, smoothing)
    if tck is None:
        empty = np.empty(0)
        return GeodesicCenterline(raw_xy=raw_xy, smoothed_xy=np.empty((0, 2)),
                                  arc_length_um=empty, curvature_per_um=empty)

    smoothed_xy, arc_length_um, curvature_per_um = _sample_spline_curvature(tck, pixel_size_um)
    return GeodesicCenterline(
        raw_xy=raw_xy,
        smoothed_xy=smoothed_xy,
        arc_length_um=arc_length_um,
        curvature_per_um=curvature_per_um,
    )
