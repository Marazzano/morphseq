"""
Component-based density composition for the modal-organization simulations.

This module is the V0 density layer described by the simulation spec:

component recipe -> composed density -> sampled observations

It intentionally stays pre-metric. The only jobs here are to define the density
recipe objects, build a true density landscape on a canonical grid, and sample
observations from that landscape for visual QA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import cos, hypot, pi, sin
from typing import Mapping, Sequence

import numpy as np

from .peak_counting import count_mass_significant_modes


@dataclass(frozen=True)
class CanonicalGrid:
    """Rectangular grid used to evaluate a composed density landscape."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_size: int = 161

    @cached_property
    def xs(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.grid_size)

    @cached_property
    def ys(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.grid_size)

    @cached_property
    def xx(self) -> np.ndarray:
        return np.meshgrid(self.xs, self.ys, indexing="xy")[0]

    @cached_property
    def yy(self) -> np.ndarray:
        return np.meshgrid(self.xs, self.ys, indexing="xy")[1]

    @cached_property
    def dx(self) -> float:
        return float((self.x_max - self.x_min) / max(self.grid_size - 1, 1))

    @cached_property
    def dy(self) -> float:
        return float((self.y_max - self.y_min) / max(self.grid_size - 1, 1))

    @cached_property
    def cell_area(self) -> float:
        return float(self.dx * self.dy)


@dataclass(frozen=True)
class DensityGrid:
    """Evaluated density on a canonical grid."""

    xx: np.ndarray
    yy: np.ndarray
    density: np.ndarray
    grid: CanonicalGrid | None = None


@dataclass(frozen=True)
class DensityComponentSpec:
    """One component in the density recipe."""

    component_id: str
    component_type: str
    mass_fraction: float
    geometry_type: str
    anchor_x: float
    anchor_y: float
    orientation_angle: float = 0.0
    local_width: float = 1.0
    anisotropy_ratio: float = 1.0
    density_profile: str = "gaussian"
    parameters: Mapping[str, float] = field(default_factory=dict)
    connected_components: tuple[str, ...] = ()

    @property
    def anchor(self) -> np.ndarray:
        return np.asarray([self.anchor_x, self.anchor_y], dtype=float)


@dataclass(frozen=True)
class DensitySpec:
    """Component recipe plus canonical evaluation grid."""

    density_id: str
    canonical_grid: CanonicalGrid
    components: tuple[DensityComponentSpec, ...]


@dataclass(frozen=True)
class ComponentTruth:
    """Component metadata measured from the evaluated component density."""

    component_id: str
    component_type: str
    mass_fraction: float
    geometry_type: str
    anchor_x: float
    anchor_y: float
    orientation_angle: float
    local_width: float
    anisotropy_ratio: float
    density_profile: str
    parameters: Mapping[str, float]
    component_mass_centroid_x: float
    component_mass_centroid_y: float


@dataclass(frozen=True)
class ComposedDensityTruth:
    """The realized density landscape before finite sampling."""

    density_spec: DensitySpec
    composed_grid: DensityGrid
    component_truth: tuple[ComponentTruth, ...]
    component_fields: dict[str, np.ndarray] | None

    @property
    def true_grid(self) -> DensityGrid:
        """Backward-compatible alias for the composed density grid."""
        return self.composed_grid


@dataclass(frozen=True)
class DensityRealization:
    """Full V0 realization: truth landscape plus sampled observations."""

    truth: ComposedDensityTruth
    points: np.ndarray
    component_labels: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray | None:
        return self.component_labels


@dataclass(frozen=True)
class BridgeRegionTruth:
    """Measured bridge-region summary from the composed density."""

    bridge_component_id: str
    connected_components: tuple[str, str]
    peak_density_pair: float
    bridge_region_density_ratio: float
    bridge_region_mass_fraction: float
    saddle_density_ratio: float
    bridge_region_label: str


@dataclass(frozen=True)
class ComposedDensityValidation:
    """Validation summary for a composed density landscape."""

    density_id: str
    total_mass: float
    resolved_peak_count: int
    expected_peak_count: int | None
    peak_count_matches: bool | None
    bridge_regions: tuple[BridgeRegionTruth, ...]

    @property
    def has_bridge(self) -> bool:
        return len(self.bridge_regions) > 0


def _rotation_matrix(theta: float) -> np.ndarray:
    c = cos(theta)
    s = sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _component_map(components: Sequence[DensityComponentSpec]) -> dict[str, DensityComponentSpec]:
    return {component.component_id: component for component in components}


def _validate_density_spec(spec: DensitySpec) -> None:
    masses = np.asarray([component.mass_fraction for component in spec.components], dtype=float)
    if not np.isclose(float(masses.sum()), 1.0, atol=1e-6):
        raise ValueError(
            f"DensitySpec {spec.density_id!r} has mass fractions summing to {masses.sum():.6f}, not 1.0"
        )

    by_id = _component_map(spec.components)
    for component in spec.components:
        if component.component_type == "bridge":
            if len(component.connected_components) != 2:
                raise ValueError(
                    f"Bridge component {component.component_id!r} must connect exactly two mode components"
                )
            for connected_id in component.connected_components:
                if connected_id not in by_id:
                    raise ValueError(
                        f"Bridge component {component.component_id!r} references unknown component {connected_id!r}"
                    )
                if by_id[connected_id].component_type != "mode":
                    raise ValueError(
                        f"Bridge component {component.component_id!r} must connect mode components; "
                        f"{connected_id!r} is {by_id[connected_id].component_type!r}"
                    )
        elif component.connected_components:
            raise ValueError(
                f"Component {component.component_id!r} of type {component.component_type!r} "
                "may not define connected_components"
            )


def validate_density_spec(
    spec: DensitySpec,
    *,
    expected_peak_count: int | None = None,
    expected_bridge_labels: Mapping[str, str] | None = None,
    bridge_width_multiplier: float = 0.8,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    peak_count_tolerance: int = 0,
) -> ComposedDensityValidation:
    """Compose and validate the density truth implied by a recipe.

    This is the truth-level gate for `F_composed`. It checks the composed mass,
    the resolved peak count, and any bridge-region labels requested by the caller.
    """

    truth = compose_density_truth(spec, keep_component_fields=True)
    return validate_composed_density_truth(
        truth,
        expected_peak_count=expected_peak_count,
        expected_bridge_labels=expected_bridge_labels,
        bridge_width_multiplier=bridge_width_multiplier,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        peak_count_tolerance=peak_count_tolerance,
    )


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


def _bridge_region_mask(
    component: DensityComponentSpec,
    grid: CanonicalGrid,
    by_id: Mapping[str, DensityComponentSpec],
    *,
    width_multiplier: float,
) -> np.ndarray:
    curve, _, _ = _bridge_curve_samples(component, by_id)
    start = curve[0]
    end = curve[-1]
    axis = end - start
    axis_norm2 = float(np.dot(axis, axis))
    if axis_norm2 <= 1e-12:
        raise ValueError(f"Bridge component {component.component_id!r} has degenerate geometry")

    xx, yy = _grid_xy(grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    rel = pts - start[None, :]
    t = np.clip((rel @ axis) / axis_norm2, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * axis[None, :]
    dist2 = np.sum((pts - proj) ** 2, axis=1)
    bridge_radius = max(float(component.local_width), 1e-3) * float(width_multiplier)
    return (dist2 <= bridge_radius ** 2).reshape(xx.shape)


def _local_peak_density(
    density: np.ndarray,
    grid: CanonicalGrid,
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


def validate_composed_density_truth(
    truth: ComposedDensityTruth,
    *,
    expected_peak_count: int | None = None,
    expected_bridge_labels: Mapping[str, str] | None = None,
    bridge_width_multiplier: float = 0.8,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    peak_count_tolerance: int = 0,
) -> ComposedDensityValidation:
    """Validate the composed density landscape itself."""

    spec = truth.density_spec
    density_grid = truth.composed_grid
    if density_grid.grid is None:
        raise ValueError("Composed density grid must carry its canonical grid")

    density = np.asarray(density_grid.density, dtype=float)
    total_mass = float(np.sum(density) * density_grid.grid.cell_area)
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError(f"Composed density for {spec.density_id!r} has non-positive mass")
    if not np.isclose(total_mass, 1.0, atol=1e-6):
        raise ValueError(
            f"Composed density for {spec.density_id!r} does not integrate to 1.0 "
            f"(mass={total_mass:.6f})"
        )

    resolved_peak_count, _ = count_mass_significant_modes(
        density,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
    )
    peak_count_matches: bool | None = None
    if expected_peak_count is not None:
        peak_count_matches = abs(int(resolved_peak_count) - int(expected_peak_count)) <= int(peak_count_tolerance)
        if not peak_count_matches:
            raise ValueError(
                f"Composed density for {spec.density_id!r} resolved {resolved_peak_count} peaks, "
                f"expected {expected_peak_count}"
            )

    by_id = _component_map(spec.components)
    bridge_regions: list[BridgeRegionTruth] = []
    for component in spec.components:
        if component.component_type != "bridge":
            continue
        if len(component.connected_components) != 2:
            continue
        left = by_id[component.connected_components[0]]
        right = by_id[component.connected_components[1]]
        mask = _bridge_region_mask(
            component,
            density_grid.grid,
            by_id,
            width_multiplier=bridge_width_multiplier,
        )
        bridge_mass = float(np.sum(density[mask]) * density_grid.grid.cell_area)
        bridge_region_mass_fraction = bridge_mass
        left_peak = _local_peak_density(
            density,
            density_grid.grid,
            left.anchor,
            search_radius=max(left.local_width, 1e-3) * max(left.anisotropy_ratio, 1.0) * 2.0,
        )
        right_peak = _local_peak_density(
            density,
            density_grid.grid,
            right.anchor,
            search_radius=max(right.local_width, 1e-3) * max(right.anisotropy_ratio, 1.0) * 2.0,
        )
        peak_density_pair = float(min(left_peak, right_peak))
        if peak_density_pair <= 0:
            raise ValueError(
                f"Bridge component {component.component_id!r} in {spec.density_id!r} has non-positive peak reference"
            )

        bridge_values = density[mask]
        bridge_values = bridge_values[np.isfinite(bridge_values)]
        bridge_min = float(np.min(bridge_values)) if bridge_values.size else 0.0
        bridge_mean = float(np.mean(bridge_values)) if bridge_values.size else 0.0
        bridge_region_density_ratio = float(bridge_mean / peak_density_pair)
        saddle_density_ratio = float(bridge_min / peak_density_pair)
        bridge_label = _bridge_label_from_ratios(
            bridge_region_density_ratio,
            bridge_region_mass_fraction,
        )
        if expected_bridge_labels is not None:
            expected_label = expected_bridge_labels.get(component.component_id)
            if expected_label is not None and bridge_label != expected_label:
                raise ValueError(
                    f"Bridge component {component.component_id!r} in {spec.density_id!r} "
                    f"measured as {bridge_label!r}, expected {expected_label!r} "
                    f"(density_ratio={bridge_region_density_ratio:.3f}, "
                    f"mass_fraction={bridge_region_mass_fraction:.3f})"
                )

        bridge_regions.append(
            BridgeRegionTruth(
                bridge_component_id=component.component_id,
                connected_components=tuple(component.connected_components[:2]),
                peak_density_pair=peak_density_pair,
                bridge_region_density_ratio=bridge_region_density_ratio,
                bridge_region_mass_fraction=bridge_region_mass_fraction,
                saddle_density_ratio=saddle_density_ratio,
                bridge_region_label=bridge_label,
            )
        )

    return ComposedDensityValidation(
        density_id=spec.density_id,
        total_mass=total_mass,
        resolved_peak_count=int(resolved_peak_count),
        expected_peak_count=expected_peak_count,
        peak_count_matches=peak_count_matches,
        bridge_regions=tuple(bridge_regions),
    )


def _component_extent_hint(component: DensityComponentSpec, by_id: Mapping[str, DensityComponentSpec]) -> float:
    base = max(component.local_width, 1e-3)
    aniso = max(float(component.anisotropy_ratio), 1.0 / max(float(component.anisotropy_ratio), 1e-6))
    extent = 3.0 * base * aniso

    if component.geometry_type == "flat_core" or component.density_profile == "flat_core":
        extent = max(extent, 2.6 * base * aniso)
    elif component.geometry_type == "spiral" or component.density_profile == "spiral":
        params = component.parameters
        radius = float(params.get("base_radius", 0.0)) + float(params.get("radial_growth", 0.0))
        extent = max(extent, radius + 3.0 * base)
    elif component.component_type == "bridge" and len(component.connected_components) == 2:
        left = by_id[component.connected_components[0]].anchor
        right = by_id[component.connected_components[1]].anchor
        extent = max(extent, 0.5 * hypot(*(right - left)) + 3.0 * base)

    return float(extent)


def infer_canonical_grid(
    components: Sequence[DensityComponentSpec],
    *,
    grid_size: int = 161,
    margin: float = 0.18,
) -> CanonicalGrid:
    """Infer a canonical grid that comfortably bounds the recipe."""

    if not components:
        raise ValueError("infer_canonical_grid requires at least one component")

    by_id = _component_map(components)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for component in components:
        extent = _component_extent_hint(component, by_id)
        xmins.append(component.anchor_x - extent)
        xmaxs.append(component.anchor_x + extent)
        ymins.append(component.anchor_y - extent)
        ymaxs.append(component.anchor_y + extent)

    x_min = float(min(xmins))
    x_max = float(max(xmaxs))
    y_min = float(min(ymins))
    y_max = float(max(ymaxs))
    x_pad = max((x_max - x_min) * margin, 0.5)
    y_pad = max((y_max - y_min) * margin, 0.5)

    return CanonicalGrid(
        x_min=x_min - x_pad,
        x_max=x_max + x_pad,
        y_min=y_min - y_pad,
        y_max=y_max + y_pad,
        grid_size=int(grid_size),
    )


def build_density_spec(
    density_id: str,
    components: Sequence[DensityComponentSpec],
    *,
    grid_size: int = 161,
    margin: float = 0.18,
) -> DensitySpec:
    """Create a density spec and infer a canonical grid from the recipe."""

    components = tuple(components)
    grid = infer_canonical_grid(components, grid_size=grid_size, margin=margin)
    return DensitySpec(density_id=density_id, canonical_grid=grid, components=components)


def _grid_xy(grid: CanonicalGrid) -> tuple[np.ndarray, np.ndarray]:
    return grid.xx, grid.yy


def _normalize_density_field(density: np.ndarray, grid: CanonicalGrid) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=float), 0.0, None)
    mass = float(np.sum(density) * grid.cell_area)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Density field has non-positive mass")
    return density / mass


def _gaussian_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    center: Sequence[float],
    sigma_major: float,
    sigma_minor: float,
    orientation_angle: float = 0.0,
) -> np.ndarray:
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    c = cos(orientation_angle)
    s = sin(orientation_angle)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    sigma_major = max(float(sigma_major), 1e-3)
    sigma_minor = max(float(sigma_minor), 1e-3)
    return np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))


def _flat_core_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    center: Sequence[float],
    width: float,
    anisotropy_ratio: float = 1.0,
    orientation_angle: float = 0.0,
    edge_softness: float = 0.35,
) -> np.ndarray:
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    c = cos(orientation_angle)
    s = sin(orientation_angle)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    major = max(float(width), 1e-3) * max(float(anisotropy_ratio), 1.0)
    minor = max(float(width), 1e-3) / max(float(anisotropy_ratio), 1.0)
    r = np.sqrt((u / major) ** 2 + (v / minor) ** 2)
    shell = np.maximum(r - 1.0, 0.0)
    return np.where(r <= 1.0, 1.0, np.exp(-0.5 * (shell / max(edge_softness, 1e-3)) ** 2))


def _sum_isotropic_gaussians_on_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Evaluate a weighted Gaussian curve on a grid in chunks."""

    centers = np.asarray(centers, dtype=float)
    weights = np.asarray(weights, dtype=float)
    density = np.zeros_like(xx, dtype=float)
    if len(centers) == 0:
        return density

    sigma2 = max(float(sigma), 1e-3) ** 2
    x = xx[None, :, :]
    y = yy[None, :, :]
    for start in range(0, len(centers), max(int(chunk_size), 1)):
        stop = min(start + int(chunk_size), len(centers))
        chunk_centers = centers[start:stop]
        chunk_weights = weights[start:stop]
        dx = x - chunk_centers[:, 0, None, None]
        dy = y - chunk_centers[:, 1, None, None]
        fields = np.exp(-0.5 * (dx * dx + dy * dy) / sigma2)
        density += np.sum(chunk_weights[:, None, None] * fields, axis=0)
    return density


def _mode_r80_radius(component: DensityComponentSpec) -> float:
    """Approximate the radius enclosing 80% of a single mode's mass.

    For Gaussian modes this is a convenient geometric proxy used only to anchor
    the bridge endpoints. It intentionally stays approximate; the exact resolved
        truth is measured from `F_composed` later.
    """

    local = max(component.local_width, 1e-3)
    stretch = max(float(component.anisotropy_ratio), 1.0) ** 0.5
    if component.density_profile == "flat_core":
        return float(0.95 * local * stretch)
    return float(1.794 * local * stretch)


def _spiral_curve_points(
    component: DensityComponentSpec,
    *,
    n_samples: int | None = None,
) -> np.ndarray:
    """Sample a smooth spiral curve densely enough to approximate continuity."""

    params = component.parameters
    turns = float(params.get("turns", 1.5))
    base_radius = float(params.get("base_radius", 0.35))
    radial_growth = float(params.get("radial_growth", 2.8))
    phase = float(params.get("phase", 0.0))
    if n_samples is None:
        n_samples = int(params.get("n_curve_samples", max(256, int(round(turns * 160)))))
    t = np.linspace(0.0, 1.0, int(n_samples))
    theta = component.orientation_angle + phase + 2.0 * pi * turns * t
    radius = base_radius + radial_growth * t
    x = component.anchor_x + radius * np.cos(theta)
    y = component.anchor_y + radius * np.sin(theta)
    return np.column_stack([x, y])


def _bridge_curve_samples(
    component: DensityComponentSpec,
    by_id: Mapping[str, DensityComponentSpec],
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a smooth bridge curve between approximate R80 boundaries."""

    if len(component.connected_components) != 2:
        raise ValueError(f"Bridge component {component.component_id!r} needs two endpoints")

    left = by_id[component.connected_components[0]]
    right = by_id[component.connected_components[1]]
    start_center = left.anchor
    end_center = right.anchor
    axis = end_center - start_center
    center_distance = hypot(*axis)
    if center_distance <= 1e-9:
        raise ValueError(f"Bridge component {component.component_id!r} has coincident endpoints")

    unit = axis / center_distance
    left_r80 = _mode_r80_radius(left)
    right_r80 = _mode_r80_radius(right)
    bridge_length = center_distance - left_r80 - right_r80
    if bridge_length <= 0:
        raise ValueError(
            f"Bridge component {component.component_id!r} has no inter-boundary span; "
            "the connected modes overlap too much for a V0 bridge"
        )

    start = start_center + unit * left_r80
    end = end_center - unit * right_r80
    if n_samples is None:
        n_samples = int(component.parameters.get("n_curve_samples", max(256, int(round(bridge_length * 48)))))
    t = np.linspace(0.0, 1.0, int(n_samples))
    curve = start[None, :] + t[:, None] * (end - start)[None, :]

    axial_sigma = float(component.parameters.get("axial_sigma", 0.28))
    axial_weights = np.exp(-0.5 * ((t - 0.5) / max(axial_sigma, 1e-3)) ** 2)
    axial_weights = axial_weights / float(axial_weights.sum())
    return curve, axial_weights, float(bridge_length)


def _component_density_field(
    component: DensityComponentSpec,
    grid: CanonicalGrid,
    *,
    by_id: Mapping[str, DensityComponentSpec],
) -> np.ndarray:
    xx, yy = _grid_xy(grid)
    if component.density_profile == "gaussian":
        major = max(component.local_width, 1e-3) * max(float(component.anisotropy_ratio), 1.0)
        minor = max(component.local_width, 1e-3) / max(float(component.anisotropy_ratio), 1.0)
        density = _gaussian_field(
            xx,
            yy,
            center=component.anchor,
            sigma_major=major,
            sigma_minor=minor,
            orientation_angle=component.orientation_angle,
        )
        return _normalize_density_field(density, grid)

    if component.density_profile == "flat_core":
        density = _flat_core_field(
            xx,
            yy,
            center=component.anchor,
            width=component.local_width,
            anisotropy_ratio=component.anisotropy_ratio,
            orientation_angle=component.orientation_angle,
        )
        return _normalize_density_field(density, grid)

    if component.density_profile == "spiral" or component.geometry_type == "spiral":
        centers = _spiral_curve_points(component)
        sigma = max(component.local_width, 1e-3)
        weight = np.full(len(centers), 1.0 / max(len(centers), 1), dtype=float)
        density = _sum_isotropic_gaussians_on_grid(xx, yy, centers, weight, sigma)
        return _normalize_density_field(density, grid)

    if component.density_profile == "ridge" or component.geometry_type in {"ridge", "capsule"}:
        centers, weights, _ = _bridge_curve_samples(component, by_id)
        sigma = max(component.local_width, 1e-3)
        density = _sum_isotropic_gaussians_on_grid(xx, yy, centers, weights, sigma)
        return _normalize_density_field(density, grid)

    raise ValueError(
        f"Unknown density profile {component.density_profile!r} for component {component.component_id!r}"
    )


def _component_truth_from_density(
    component: DensityComponentSpec,
    density: np.ndarray,
    grid: CanonicalGrid,
) -> ComponentTruth:
    xx, yy = _grid_xy(grid)
    mass = float(np.sum(density) * grid.cell_area)
    if mass <= 0:
        raise ValueError(f"Component {component.component_id!r} produced zero mass")
    cx = float(np.sum(density * xx) * grid.cell_area / mass)
    cy = float(np.sum(density * yy) * grid.cell_area / mass)
    return ComponentTruth(
        component_id=component.component_id,
        component_type=component.component_type,
        mass_fraction=float(component.mass_fraction),
        geometry_type=component.geometry_type,
        anchor_x=float(component.anchor_x),
        anchor_y=float(component.anchor_y),
        orientation_angle=float(component.orientation_angle),
        local_width=float(component.local_width),
        anisotropy_ratio=float(component.anisotropy_ratio),
        density_profile=component.density_profile,
        parameters=dict(component.parameters),
        component_mass_centroid_x=cx,
        component_mass_centroid_y=cy,
    )


def compose_density_truth(
    spec: DensitySpec,
    *,
    keep_component_fields: bool = True,
) -> ComposedDensityTruth:
    """Evaluate each component and compose the true density landscape."""

    _validate_density_spec(spec)
    by_id = _component_map(spec.components)
    grid = spec.canonical_grid
    xx, yy = _grid_xy(grid)

    component_truth: list[ComponentTruth] = []
    component_fields: dict[str, np.ndarray] | None = {} if keep_component_fields else None
    composed = np.zeros_like(xx, dtype=float)
    for component in spec.components:
        field = _component_density_field(component, grid, by_id=by_id)
        component_truth.append(_component_truth_from_density(component, field, grid))
        weighted = float(component.mass_fraction) * field
        if component_fields is not None:
            component_fields[component.component_id] = weighted
        composed += weighted

    total_mass = float(np.sum(composed) * grid.cell_area)
    if not np.isclose(total_mass, 1.0, atol=1e-6):
        if not np.isfinite(total_mass) or total_mass <= 0:
            raise ValueError(f"Composed density for {spec.density_id!r} has non-positive mass")
        composed = composed / total_mass
    composed_grid = DensityGrid(grid=grid, xx=xx, yy=yy, density=composed)
    return ComposedDensityTruth(
        density_spec=spec,
        composed_grid=composed_grid,
        component_truth=tuple(component_truth),
        component_fields=component_fields,
    )


def sample_points_from_density_grid(
    density_grid: DensityGrid,
    n: int,
    rng: np.random.Generator,
    *,
    component_fields: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample points from an evaluated density grid.

    When `component_fields` is provided, the returned labels are the dominant local
    component contribution at each chosen cell. This keeps the points tied to F_composed
    while still yielding diagnostic component colors.
    """

    grid = density_grid.grid
    if grid is None:
        raise ValueError("DensityGrid.grid is required for grid sampling")

    density = np.asarray(density_grid.density, dtype=float)
    probs = density.ravel()
    probs = probs / float(probs.sum())
    flat_idx = rng.choice(len(probs), size=int(n), replace=True, p=probs)
    iy, ix = np.unravel_index(flat_idx, density.shape)

    xs = grid.xs[ix]
    ys = grid.ys[iy]
    xj = rng.uniform(-0.5 * grid.dx, 0.5 * grid.dx, size=int(n))
    yj = rng.uniform(-0.5 * grid.dy, 0.5 * grid.dy, size=int(n))
    points = np.column_stack([
        np.clip(xs + xj, grid.x_min, grid.x_max),
        np.clip(ys + yj, grid.y_min, grid.y_max),
    ])

    if not component_fields:
        return points, None

    names = np.asarray(list(component_fields.keys()), dtype=object)
    field_stack = np.stack(
        [np.asarray(component_fields[name], dtype=float).ravel()[flat_idx] for name in names],
        axis=1,
    )
    field_totals = field_stack.sum(axis=1)
    labels = names[np.argmax(field_stack, axis=1)]
    if np.any(field_totals <= 0):
        labels = labels.astype(object, copy=True)
        labels[field_totals <= 0] = names[0]
    return points, labels


def _sample_from_gaussian_component(
    component: DensityComponentSpec,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    major = max(component.local_width, 1e-3) * max(float(component.anisotropy_ratio), 1.0)
    minor = max(component.local_width, 1e-3) / max(float(component.anisotropy_ratio), 1.0)
    cov = _rotation_matrix(component.orientation_angle) @ np.diag([major ** 2, minor ** 2]) @ _rotation_matrix(component.orientation_angle).T
    return rng.multivariate_normal(mean=component.anchor, cov=cov, size=count)


def _sample_from_flat_core_component(
    component: DensityComponentSpec,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    major = max(component.local_width, 1e-3) * max(float(component.anisotropy_ratio), 1.0)
    minor = max(component.local_width, 1e-3) / max(float(component.anisotropy_ratio), 1.0)
    radii = np.sqrt(rng.uniform(0.0, 1.0, size=count))
    angles = rng.uniform(0.0, 2.0 * pi, size=count)
    local = np.column_stack([radii * np.cos(angles) * major, radii * np.sin(angles) * minor])
    rot = _rotation_matrix(component.orientation_angle)
    return component.anchor[None, :] + local @ rot.T


def _sample_from_path_component(
    component: DensityComponentSpec,
    count: int,
    rng: np.random.Generator,
    by_id: Mapping[str, DensityComponentSpec],
) -> np.ndarray:
    sigma = max(component.local_width, 1e-3)
    if component.density_profile == "ridge" or component.geometry_type in {"ridge", "capsule"}:
        centers, _, _ = _bridge_curve_samples(component, by_id)
        idx = rng.choice(len(centers), size=count, replace=True)
        return centers[idx] + rng.normal(0.0, sigma, size=(count, 2))

    centers = _spiral_curve_points(component)
    idx = rng.choice(len(centers), size=count, replace=True)
    return centers[idx] + rng.normal(0.0, sigma, size=(count, 2))


def _sample_component_points(
    component: DensityComponentSpec,
    count: int,
    rng: np.random.Generator,
    *,
    by_id: Mapping[str, DensityComponentSpec],
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=float)

    if component.density_profile == "gaussian":
        return _sample_from_gaussian_component(component, count, rng)
    if component.density_profile == "flat_core":
        return _sample_from_flat_core_component(component, count, rng)
    if component.density_profile in {"ridge", "spiral"} or component.geometry_type in {"ridge", "capsule", "spiral"}:
        return _sample_from_path_component(component, count, rng, by_id)

    raise ValueError(
        f"Unknown density profile {component.density_profile!r} for component {component.component_id!r}"
    )


def realize_density(
    spec: DensitySpec,
    n: int,
    rng: np.random.Generator | None = None,
    *,
    sampling_mode: str = "grid",
    keep_component_fields: bool = True,
) -> DensityRealization:
    """Compose the true density and sample finite observations from it."""

    if rng is None:
        rng = np.random.default_rng(0)

    truth = compose_density_truth(spec, keep_component_fields=keep_component_fields)
    return realize_from_truth(truth, n, rng, sampling_mode=sampling_mode)


def realize_from_truth(
    truth: ComposedDensityTruth,
    n: int,
    rng: np.random.Generator,
    *,
    sampling_mode: str = "grid",
) -> DensityRealization:
    """Sample observations from an already composed density truth."""

    spec = truth.density_spec
    by_id = _component_map(spec.components)

    if sampling_mode == "grid":
        points, labels = sample_points_from_density_grid(
            truth.composed_grid,
            int(n),
            rng,
            component_fields=truth.component_fields,
        )
        return DensityRealization(truth=truth, points=points, component_labels=labels)

    if sampling_mode == "component":
        masses = np.asarray([component.mass_fraction for component in spec.components], dtype=float)
        counts = rng.multinomial(int(n), masses)

        points: list[np.ndarray] = []
        labels: list[str] = []
        for component, count in zip(spec.components, counts, strict=True):
            sampled = _sample_component_points(component, int(count), rng, by_id=by_id)
            if len(sampled) == 0:
                continue
            points.append(sampled)
            labels.extend([component.component_id] * len(sampled))

        if points:
            all_points = np.vstack(points)
            labels_arr = np.asarray(labels, dtype=object)
            order = rng.permutation(len(all_points))
            all_points = all_points[order]
            labels_arr = labels_arr[order]
        else:
            all_points = np.empty((0, 2), dtype=float)
            labels_arr = np.empty((0,), dtype=object)

        return DensityRealization(truth=truth, points=all_points, component_labels=labels_arr)

    raise ValueError(f"Unknown sampling_mode {sampling_mode!r}; expected 'grid' or 'component'")
