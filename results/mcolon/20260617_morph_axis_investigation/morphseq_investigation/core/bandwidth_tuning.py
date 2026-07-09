"""Geometry-derived KDE bandwidth calibration utilities for modal V0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist, squareform

from .density_composition import DensityComponentSpec, DensityGrid, DensitySpec
from .peak_counting import PeakCountDetail, peak_count_detail

DEFAULT_KNN_K = 10
DEFAULT_CONNECTIVITY_MASS = 0.90
DEFAULT_MULTIPLIERS = (0.75, 1.00, 1.25, 1.50)
DEFAULT_RULE_NAMES = (
    "median_kNN_distance",
    "q90_kNN_distance",
    "median_MST_edge_length",
    "q90_MST_edge_length",
    "longest_non_outlier_MST_edge",
    "connectivity_90_radius",
    "global_R50",
    "global_R80",
)

RULE_NAME_ALIASES = {
    "graph_connectivity_radius": "connectivity_90_radius",
    "estimated_R50": "global_R50",
    "estimated_R80": "global_R80",
}


@dataclass(frozen=True)
class BridgeRegionMeasurement:
    bridge_component_id: str
    connected_components: tuple[str, str]
    peak_density_pair: float
    bridge_region_density_ratio: float
    bridge_region_mass_fraction: float
    saddle_density_ratio: float
    bridge_region_label: str

    @property
    def bridge_pair_valley_density_ratio(self) -> float:
        return float(self.saddle_density_ratio)

    @property
    def bridge_pair_valley_depth(self) -> float:
        return float(1.0 - self.saddle_density_ratio)


@dataclass(frozen=True)
class DensityGeometryMeasurement:
    density_id: str
    total_mass: float
    peak_detail: PeakCountDetail
    peak_heights: tuple[float, ...]
    peak_height_min: float
    peak_height_median: float
    peak_height_max: float
    global_saddle_density: float | None
    global_valley_density_ratio: float | None
    global_valley_depth: float | None
    pairwise_saddle_density: float | None
    pairwise_valley_density_ratio: float | None
    pairwise_valley_depth: float | None
    bridge_regions: tuple[BridgeRegionMeasurement, ...]

    @property
    def peak_count(self) -> int:
        return int(self.peak_detail.n_modes)

    @property
    def peak_density(self) -> float:
        return float(self.peak_detail.peak_density)

    @property
    def saddle_density(self) -> float | None:
        return self.global_saddle_density

    @property
    def valley_density_ratio(self) -> float | None:
        return self.global_valley_density_ratio

    @property
    def valley_depth(self) -> float | None:
        return self.global_valley_depth

    @property
    def bridge_pair_valley_density_ratio(self) -> float | None:
        return self.pairwise_valley_density_ratio

    @property
    def bridge_pair_valley_depth(self) -> float | None:
        return self.pairwise_valley_depth


@dataclass(frozen=True)
class BandwidthCandidate:
    bandwidth_rule: str
    geometry_scale: float
    bandwidth_multiplier: float
    bandwidth: float


def _component_map(components: Sequence[DensityComponentSpec]) -> dict[str, DensityComponentSpec]:
    return {component.component_id: component for component in components}


def _grid_xy(grid: DensityGrid | None) -> tuple[np.ndarray, np.ndarray]:
    if grid is None:
        raise ValueError("density grid is required")
    return grid.xx, grid.yy


def _normalize_density_grid(density: np.ndarray, grid: DensityGrid) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=float), 0.0, None)
    if grid.grid is None:
        raise ValueError("DensityGrid.grid is required")
    mass = float(np.sum(density) * grid.grid.cell_area)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("density grid has non-positive mass")
    return density / mass


def _bridge_label_from_ratios(
    bridge_region_density_ratio: float,
    bridge_region_mass_fraction: float,
) -> str:
    if bridge_region_density_ratio <= 0.01 and bridge_region_mass_fraction <= 0.01:
        return "no_bridge"
    if 0.01 < bridge_region_density_ratio <= 0.10 and 0.01 < bridge_region_mass_fraction <= 0.05:
        return "low_bridge"
    if 0.10 < bridge_region_density_ratio <= 0.30 and 0.05 < bridge_region_mass_fraction <= 0.15:
        return "moderate_bridge"
    if 0.30 < bridge_region_density_ratio <= 0.70 and 0.15 < bridge_region_mass_fraction <= 0.30:
        return "high_bridge"
    if bridge_region_density_ratio > 0.70:
        return "fully_merged"
    return "unclassified"


def _mode_r80_radius(component: DensityComponentSpec) -> float:
    local = max(float(component.local_width), 1e-3)
    stretch = max(float(component.anisotropy_ratio), 1.0) ** 0.5
    if component.density_profile == "flat_core":
        return float(0.95 * local * stretch)
    return float(1.794 * local * stretch)


def _bridge_region_mask(
    component: DensityComponentSpec,
    grid: DensityGrid,
    by_id: Mapping[str, DensityComponentSpec],
    *,
    width_multiplier: float,
) -> np.ndarray:
    if len(component.connected_components) != 2:
        raise ValueError(f"bridge component {component.component_id!r} must connect two modes")
    left = by_id[component.connected_components[0]]
    right = by_id[component.connected_components[1]]
    start_center = left.anchor
    end_center = right.anchor
    axis = end_center - start_center
    center_distance = float(np.hypot(axis[0], axis[1]))
    if center_distance <= 1e-12:
        raise ValueError(f"bridge component {component.component_id!r} has coincident endpoints")
    unit = axis / center_distance
    left_r80 = _mode_r80_radius(left)
    right_r80 = _mode_r80_radius(right)
    bridge_length = center_distance - left_r80 - right_r80
    if bridge_length <= 0:
        raise ValueError(
            f"bridge component {component.component_id!r} has no inter-boundary span"
        )
    start = start_center + unit * left_r80
    end = end_center - unit * right_r80
    xx, yy = _grid_xy(grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    rel = pts - start[None, :]
    axis_vec = end - start
    axis_norm2 = float(np.dot(axis_vec, axis_vec))
    t = np.clip((rel @ axis_vec) / axis_norm2, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * axis_vec[None, :]
    dist2 = np.sum((pts - proj) ** 2, axis=1)
    bridge_radius = max(float(component.local_width), 1e-3) * float(width_multiplier)
    return (dist2 <= bridge_radius ** 2).reshape(xx.shape)


def _pairwise_interval_mask(
    left: DensityComponentSpec,
    right: DensityComponentSpec,
    grid: DensityGrid,
    *,
    width_multiplier: float,
) -> np.ndarray:
    start_center = left.anchor
    end_center = right.anchor
    axis = end_center - start_center
    center_distance = float(np.hypot(axis[0], axis[1]))
    if center_distance <= 1e-12:
        raise ValueError("pairwise interval has coincident endpoints")
    unit = axis / center_distance
    left_r80 = _mode_r80_radius(left)
    right_r80 = _mode_r80_radius(right)
    bridge_length = center_distance - left_r80 - right_r80
    if bridge_length > 0:
        start = start_center + unit * left_r80
        end = end_center - unit * right_r80
    else:
        start = start_center
        end = end_center
    xx, yy = _grid_xy(grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    rel = pts - start[None, :]
    axis_vec = end - start
    axis_norm2 = float(np.dot(axis_vec, axis_vec))
    if axis_norm2 <= 1e-12:
        axis_vec = end_center - start_center
        axis_norm2 = float(np.dot(axis_vec, axis_vec))
    t = np.clip((rel @ axis_vec) / axis_norm2, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * axis_vec[None, :]
    dist2 = np.sum((pts - proj) ** 2, axis=1)
    corridor_width = max(min(float(left.local_width), float(right.local_width)), 1e-3) * float(width_multiplier)
    return (dist2 <= corridor_width ** 2).reshape(xx.shape)


def _local_peak_density(
    density: np.ndarray,
    grid: DensityGrid,
    center: Sequence[float],
    *,
    search_radius: float,
) -> float:
    xx, yy = _grid_xy(grid)
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    mask = (dx * dx + dy * dy) <= float(search_radius) ** 2
    values = np.asarray(density, dtype=float)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray(density, dtype=float).ravel()
        values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else 0.0


def measure_density_geometry(
    density_grid: DensityGrid,
    spec: DensitySpec,
    *,
    bridge_width_multiplier: float = 0.8,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 200,
) -> DensityGeometryMeasurement:
    """Measure peak and bridge geometry on a density grid.

    The density is normalized on-grid before measuring so the result is stable
    under absolute scale changes.
    """

    if density_grid.grid is None:
        raise ValueError("DensityGrid.grid is required")

    by_id = _component_map(spec.components)
    density = _normalize_density_grid(density_grid.density, density_grid)
    total_mass = float(np.sum(density) * density_grid.grid.cell_area)
    peak_detail = peak_count_detail(
        density,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
    )

    mode_components = [component for component in spec.components if component.component_type == "mode"]
    peak_heights = tuple(
        _local_peak_density(
            density,
            density_grid,
            component.anchor,
            search_radius=max(float(component.local_width), 1e-3) * max(float(component.anisotropy_ratio), 1.0) * 2.0,
        )
        for component in mode_components
    )
    if peak_heights:
        peak_arr = np.asarray(peak_heights, dtype=float)
        peak_height_min = float(np.min(peak_arr))
        peak_height_median = float(np.median(peak_arr))
        peak_height_max = float(np.max(peak_arr))
    else:
        peak_height_min = peak_height_median = peak_height_max = float(peak_detail.peak_density)

    if peak_detail.n_modes >= 2 and peak_detail.split_level is not None and peak_detail.peak_density > 0:
        global_saddle_density = float(peak_detail.split_level)
        global_valley_density_ratio = float(global_saddle_density / peak_detail.peak_density)
        global_valley_depth = float(1.0 - global_valley_density_ratio)
    else:
        global_saddle_density = None
        global_valley_density_ratio = None
        global_valley_depth = None

    if len(mode_components) == 2:
        left = mode_components[0]
        right = mode_components[1]
        left_peak = _local_peak_density(
            density,
            density_grid,
            left.anchor,
            search_radius=max(float(left.local_width), 1e-3) * max(float(left.anisotropy_ratio), 1.0) * 2.0,
        )
        right_peak = _local_peak_density(
            density,
            density_grid,
            right.anchor,
            search_radius=max(float(right.local_width), 1e-3) * max(float(right.anisotropy_ratio), 1.0) * 2.0,
        )
        pairwise_peak_density = float(min(left_peak, right_peak))
        pairwise_mask = _pairwise_interval_mask(
            left,
            right,
            density_grid,
            width_multiplier=bridge_width_multiplier,
        )
        pairwise_values = density[pairwise_mask]
        pairwise_values = pairwise_values[np.isfinite(pairwise_values)]
        if pairwise_values.size and pairwise_peak_density > 0:
            pairwise_saddle_density = float(np.min(pairwise_values))
            pairwise_valley_density_ratio = float(pairwise_saddle_density / pairwise_peak_density)
            pairwise_valley_depth = float(1.0 - pairwise_valley_density_ratio)
        else:
            pairwise_saddle_density = None
            pairwise_valley_density_ratio = None
            pairwise_valley_depth = None
    else:
        pairwise_saddle_density = None
        pairwise_valley_density_ratio = None
        pairwise_valley_depth = None

    bridge_regions: list[BridgeRegionMeasurement] = []
    for component in spec.components:
        if component.component_type != "bridge":
            continue
        if len(component.connected_components) != 2:
            continue
        left = by_id[component.connected_components[0]]
        right = by_id[component.connected_components[1]]
        left_peak = _local_peak_density(
            density,
            density_grid,
            left.anchor,
            search_radius=max(float(left.local_width), 1e-3) * max(float(left.anisotropy_ratio), 1.0) * 2.0,
        )
        right_peak = _local_peak_density(
            density,
            density_grid,
            right.anchor,
            search_radius=max(float(right.local_width), 1e-3) * max(float(right.anisotropy_ratio), 1.0) * 2.0,
        )
        peak_density_pair = float(min(left_peak, right_peak))
        if peak_density_pair <= 0:
            raise ValueError(
                f"bridge component {component.component_id!r} has non-positive peak reference"
            )
        mask = _bridge_region_mask(
            component,
            density_grid,
            by_id,
            width_multiplier=bridge_width_multiplier,
        )
        values = density[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"bridge component {component.component_id!r} has an empty bridge mask")
        bridge_mass = float(np.sum(values) * density_grid.grid.cell_area)
        bridge_mean = float(np.mean(values))
        bridge_min = float(np.min(values))
        bridge_region_density_ratio = float(bridge_mean / peak_density_pair)
        saddle_density_ratio = float(bridge_min / peak_density_pair)
        bridge_regions.append(
            BridgeRegionMeasurement(
                bridge_component_id=component.component_id,
                connected_components=tuple(component.connected_components[:2]),
                peak_density_pair=peak_density_pair,
                bridge_region_density_ratio=bridge_region_density_ratio,
                bridge_region_mass_fraction=bridge_mass,
                saddle_density_ratio=saddle_density_ratio,
                bridge_region_label=_bridge_label_from_ratios(
                    bridge_region_density_ratio,
                    bridge_mass,
                ),
            )
        )

    if not bridge_regions and len(mode_components) == 2:
        left = mode_components[0]
        right = mode_components[1]
        left_peak = _local_peak_density(
            density,
            density_grid,
            left.anchor,
            search_radius=max(float(left.local_width), 1e-3) * max(float(left.anisotropy_ratio), 1.0) * 2.0,
        )
        right_peak = _local_peak_density(
            density,
            density_grid,
            right.anchor,
            search_radius=max(float(right.local_width), 1e-3) * max(float(right.anisotropy_ratio), 1.0) * 2.0,
        )
        peak_density_pair = float(min(left_peak, right_peak))
        mask = _pairwise_interval_mask(
            left,
            right,
            density_grid,
            width_multiplier=bridge_width_multiplier,
        )
        values = density[mask]
        values = values[np.isfinite(values)]
        if values.size and peak_density_pair > 0:
            bridge_mass = float(np.sum(values) * density_grid.grid.cell_area)
            bridge_mean = float(np.mean(values))
            bridge_min = float(np.min(values))
            bridge_regions.append(
                BridgeRegionMeasurement(
                    bridge_component_id="pairwise_mode_interval",
                    connected_components=(left.component_id, right.component_id),
                    peak_density_pair=peak_density_pair,
                    bridge_region_density_ratio=float(bridge_mean / peak_density_pair),
                    bridge_region_mass_fraction=bridge_mass,
                    saddle_density_ratio=float(bridge_min / peak_density_pair),
                    bridge_region_label=_bridge_label_from_ratios(
                        float(bridge_mean / peak_density_pair),
                        bridge_mass,
                    ),
                )
            )

    return DensityGeometryMeasurement(
        density_id=spec.density_id,
        total_mass=total_mass,
        peak_detail=peak_detail,
        peak_heights=peak_heights,
        peak_height_min=peak_height_min,
        peak_height_median=peak_height_median,
        peak_height_max=peak_height_max,
        global_saddle_density=global_saddle_density,
        global_valley_density_ratio=global_valley_density_ratio,
        global_valley_depth=global_valley_depth,
        pairwise_saddle_density=pairwise_saddle_density,
        pairwise_valley_density_ratio=pairwise_valley_density_ratio,
        pairwise_valley_depth=pairwise_valley_depth,
        bridge_regions=tuple(bridge_regions),
    )


def bandwidth_geometry_scales(
    points: np.ndarray,
    *,
    knn_k: int = DEFAULT_KNN_K,
    connectivity_mass: float = DEFAULT_CONNECTIVITY_MASS,
    include_connectivity_radius: bool = True,
) -> dict[str, float]:
    """Point-cloud geometry scales used to derive isotropic KDE bandwidths.

    `include_connectivity_radius=False` skips `_graph_connectivity_radius_from_distance_matrix`
    (an O(n^2) union-find sweep over all pairwise edges) when the caller's
    selected bandwidth rule can never read `connectivity_90_radius`/
    `graph_connectivity_radius` -- those two keys are set to NaN instead.
    Default `True` preserves full behavior for diagnostic callers (e.g.
    `propose_bandwidth_candidates`) that want every scale.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    n = len(pts)
    if n < 2:
        return {}

    dmat = squareform(pdist(pts))
    sorted_d = np.sort(dmat, axis=1)
    k_eff = max(1, min(int(knn_k), n - 1))
    kth = sorted_d[:, k_eff]
    positive = kth[kth > 1e-12]
    knn_median = float(np.median(positive)) if positive.size else float("nan")
    knn_q90 = float(np.quantile(positive, 0.90)) if positive.size else float("nan")

    mst = minimum_spanning_tree(dmat)
    edges = np.asarray(mst.data, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = edges[edges > 1e-12]
    if edges.size:
        q25, q75 = np.quantile(edges, [0.25, 0.75])
        iqr = float(q75 - q25)
        upper_fence = float(q75 + 1.5 * iqr)
        non_outlier = edges[edges <= upper_fence]
        longest_non_outlier = float(np.max(non_outlier)) if non_outlier.size else float(np.max(edges))
        mst_median = float(np.median(edges))
        mst_q90 = float(np.quantile(edges, 0.90))
    else:
        longest_non_outlier = float("nan")
        mst_median = float("nan")
        mst_q90 = float("nan")

    center = np.median(pts, axis=0)
    radii = np.linalg.norm(pts - center[None, :], axis=1)
    r50 = float(np.quantile(radii, 0.50))
    r80 = float(np.quantile(radii, 0.80))

    connectivity_radius = (
        _graph_connectivity_radius_from_distance_matrix(dmat, target_mass=connectivity_mass)
        if include_connectivity_radius
        else float("nan")
    )
    scales = {
        "median_kNN_distance": knn_median,
        "q90_kNN_distance": knn_q90,
        "median_MST_edge_length": mst_median,
        "q90_MST_edge_length": mst_q90,
        "longest_non_outlier_MST_edge": longest_non_outlier,
        "connectivity_90_radius": connectivity_radius,
        "global_R50": r50,
        "global_R80": r80,
    }
    scales.update({
        "graph_connectivity_radius": connectivity_radius,
        "estimated_R50": r50,
        "estimated_R80": r80,
    })
    return scales


def _graph_connectivity_radius_from_distance_matrix(
    distance_matrix: np.ndarray, *, target_mass: float
) -> float:
    """Union-find sweep over `distance_matrix`'s edges for the smallest radius at
    which `target_mass` fraction of points are mutually connected.

    Takes the distance matrix explicitly (rather than raw points) so callers
    that already built it -- `bandwidth_geometry_scales` always does -- reuse
    it instead of paying for a second O(n^2) `pdist`. `bandwidth_geometry_scales`
    owns pairwise-distance construction; this function owns connectivity
    analysis only, given the matrix.
    """
    dmat = np.asarray(distance_matrix, dtype=float)
    if dmat.ndim != 2 or dmat.shape[0] != dmat.shape[1]:
        raise ValueError(f"distance_matrix must be square 2-D, got shape {dmat.shape}")
    n = dmat.shape[0]
    if n < 2:
        return float("nan")
    target_mass = float(np.clip(target_mass, 1.0 / n, 1.0))
    i_idx, j_idx = np.triu_indices(n, k=1)
    edge_lengths = dmat[i_idx, j_idx]
    order = np.argsort(edge_lengths)

    parent = np.arange(n)
    size = np.ones(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for idx in order:
        i = int(i_idx[idx])
        j = int(j_idx[idx])
        ri = find(i)
        rj = find(j)
        if ri != rj:
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]
        if size[find(i)] / n >= target_mass:
            return float(edge_lengths[idx])
    return float(edge_lengths[order[-1]]) if edge_lengths.size else float("nan")


def propose_bandwidth_candidates(
    points: np.ndarray,
    *,
    rule_names: Sequence[str] | None = None,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    knn_k: int = DEFAULT_KNN_K,
    connectivity_mass: float = DEFAULT_CONNECTIVITY_MASS,
) -> list[BandwidthCandidate]:
    scales = bandwidth_geometry_scales(
        points,
        knn_k=knn_k,
        connectivity_mass=connectivity_mass,
    )
    if rule_names is None:
        rule_names = DEFAULT_RULE_NAMES

    candidates: list[BandwidthCandidate] = []
    for rule_name in rule_names:
        canonical_rule_name = RULE_NAME_ALIASES.get(str(rule_name), str(rule_name))
        geometry_scale = float(scales.get(canonical_rule_name, scales.get(str(rule_name), float("nan"))))
        if not np.isfinite(geometry_scale) or geometry_scale <= 0:
            continue
        for multiplier in multipliers:
            bandwidth = float(geometry_scale * float(multiplier))
            candidates.append(
                BandwidthCandidate(
                    bandwidth_rule=canonical_rule_name,
                    geometry_scale=geometry_scale,
                    bandwidth_multiplier=float(multiplier),
                    bandwidth=bandwidth,
                )
            )
    return candidates


def precompute_squared_distances(
    grid_points: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Pairwise squared Euclidean distances, `(n_grid, n_points)`.

    Uses `scipy.spatial.distance.cdist(..., "sqeuclidean")` -- scipy's C
    implementation beats both a broadcast-subtract-square loop (the original
    implementation) and a hand-rolled |a-b|^2 = |a|^2+|b|^2-2*a.b BLAS-matmul
    expansion (an intermediate version of this function) at canonical-grid
    scale: ~2.7x faster than the matmul version alone (measured: 61x61 grid x
    160 points, ~6.0ms/call matmul vs ~2.2ms/call cdist), identical output to
    float64 noise floor (~1e-14). cdist avoids the matmul expansion's extra
    elementwise passes (separate norm/broadcast/clamp steps) that the BLAS
    call alone doesn't eliminate.
    """
    grid_points = np.asarray(grid_points, dtype=float)
    points = np.asarray(points, dtype=float)
    if grid_points.ndim != 2 or grid_points.shape[1] != 2:
        raise ValueError("grid_points must have shape (n_grid, 2)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    return cdist(grid_points, points, "sqeuclidean")


def evaluate_isotropic_gaussian_kde_from_dist2(
    dist2: np.ndarray,
    bandwidth: float,
    *,
    cell_area: float | None = None,
    normalize_grid: bool = True,
) -> np.ndarray:
    """Dense isotropic Gaussian KDE from precomputed squared distances.

    A sklearn.neighbors.KernelDensity backend was benchmarked and found to be
    numerically identical (ratio 1.0 +/- 2.8e-13) but ~3x slower for the
    current 2-D workload (~160 samples, 61x61 grid) -- dense BLAS matmul wins
    here since a Gaussian kernel's infinite support means tree methods still
    visit most points anyway. Reconsider a tree-based backend if KDE moves to
    substantially higher dimensions or much larger sample counts; see
    v0/benchmark_kde_backends.py to rerun the comparison rather than
    rediscovering it from scratch.
    """
    dist2 = np.asarray(dist2, dtype=float)
    h = max(float(bandwidth), 1e-6)
    h2 = h * h
    # In-place exp() into a scratch buffer (not dist2 itself -- callers may
    # reuse/inspect their own dist2 array) avoids allocating a second
    # full-size temporary beyond the one np.multiply already needs -- ~15%
    # faster than `np.exp(-0.5 * dist2 / h2)` alone, exact to float64 noise
    # floor (verified diff ~1e-17).
    scratch = np.multiply(dist2, -0.5 / h2)
    np.exp(scratch, out=scratch)
    density = scratch.mean(axis=1)
    density /= 2.0 * np.pi * h2
    if normalize_grid and cell_area is not None:
        mass = float(np.sum(density) * float(cell_area))
        if np.isfinite(mass) and mass > 0:
            density = density / mass
    return density
