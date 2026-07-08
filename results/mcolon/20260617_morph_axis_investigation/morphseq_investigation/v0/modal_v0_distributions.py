"""Minimal V0 synthetic distributions for modal-organization visual QA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphseq_investigation.core.density_composition import (
    DensityComponentSpec,
    DensityRealization,
    DensitySpec,
    build_density_spec,
    realize_density,
)


@dataclass(frozen=True)
class V0Distribution:
    distribution_id: str
    density_spec: DensitySpec
    note: str

    def realize(self, n: int, rng: np.random.Generator) -> DensityRealization:
        return realize_density(self.density_spec, n=n, rng=rng)


def _mode(
    component_id: str,
    *,
    mass_fraction: float,
    anchor_x: float,
    anchor_y: float,
    width: float,
    anisotropy_ratio: float = 1.0,
    orientation_angle: float = 0.0,
    density_profile: str = "gaussian",
) -> DensityComponentSpec:
    return DensityComponentSpec(
        component_id=component_id,
        component_type="mode",
        mass_fraction=mass_fraction,
        geometry_type="ellipse" if density_profile == "gaussian" else "spiral",
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        orientation_angle=orientation_angle,
        local_width=width,
        anisotropy_ratio=anisotropy_ratio,
        density_profile=density_profile,
    )


def _bridge(
    component_id: str,
    *,
    mass_fraction: float,
    left_anchor: tuple[float, float],
    right_anchor: tuple[float, float],
    width: float,
    left_id: str,
    right_id: str,
    n_curve_samples: int = 256,
    axial_sigma: float = 0.28,
) -> DensityComponentSpec:
    anchor_x = 0.5 * (float(left_anchor[0]) + float(right_anchor[0]))
    anchor_y = 0.5 * (float(left_anchor[1]) + float(right_anchor[1]))
    return DensityComponentSpec(
        component_id=component_id,
        component_type="bridge",
        mass_fraction=mass_fraction,
        geometry_type="ridge",
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        local_width=width,
        density_profile="ridge",
        connected_components=(left_id, right_id),
        parameters={
            "n_curve_samples": float(n_curve_samples),
            "axial_sigma": float(axial_sigma),
        },
    )


def _make_spec(density_id: str, components: list[DensityComponentSpec]) -> DensitySpec:
    return build_density_spec(density_id, components, grid_size=161, margin=0.16)


COMPACT_MODE_WIDTH = 0.34


ONE_PEAK_COMPACT = _make_spec(
    "one_peak_compact",
    [
        _mode(
            "mode_0",
            mass_fraction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            width=COMPACT_MODE_WIDTH,
            anisotropy_ratio=1.0,
        )
    ],
)

ONE_PEAK_DIFFUSE = _make_spec(
    "one_peak_diffuse",
    [
        DensityComponentSpec(
            component_id="mode_0",
            component_type="mode",
            mass_fraction=1.0,
            geometry_type="region",
            anchor_x=0.0,
            anchor_y=0.0,
            local_width=1.55,
            density_profile="flat_core",
        )
    ],
)

ONE_PEAK_ELONGATED = _make_spec(
    "one_peak_elongated",
    [
        _mode(
            "mode_0",
            mass_fraction=1.0,
            anchor_x=0.0,
            anchor_y=0.0,
            width=0.28,
            anisotropy_ratio=6.0,
            orientation_angle=0.12,
        )
    ],
)

ONE_PEAK_SPIRAL = _make_spec(
    "one_peak_spiral",
    [
        DensityComponentSpec(
            component_id="mode_0",
            component_type="mode",
            mass_fraction=1.0,
            geometry_type="spiral",
            anchor_x=0.0,
            anchor_y=0.0,
            orientation_angle=0.0,
            local_width=0.18,
            density_profile="spiral",
            parameters={
                "turns": 1.55,
                "base_radius": 0.28,
                "radial_growth": 3.10,
                "phase": 0.18,
                "n_curve_samples": 384.0,
            },
        )
    ],
)

TWO_PEAKS_NO_BRIDGE = _make_spec(
    "two_peaks_no_bridge",
    [
        _mode("mode_left", mass_fraction=0.5, anchor_x=-2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
        _mode("mode_right", mass_fraction=0.5, anchor_x=2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
    ],
)

TWO_PEAKS_LOW_BRIDGE = _make_spec(
    "two_peaks_low_bridge",
    [
        _mode("mode_left", mass_fraction=0.475, anchor_x=-2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
        _mode("mode_right", mass_fraction=0.475, anchor_x=2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
        _bridge(
            "bridge_left_right",
            mass_fraction=0.05,
            left_anchor=(-2.35, 0.0),
            right_anchor=(2.35, 0.0),
            width=0.14,
            left_id="mode_left",
            right_id="mode_right",
            n_curve_samples=256,
        ),
    ],
)

TWO_PEAKS_HIGH_BRIDGE = _make_spec(
    "two_peaks_high_bridge",
    [
        _mode("mode_left", mass_fraction=0.32, anchor_x=-2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
        _mode("mode_right", mass_fraction=0.32, anchor_x=2.35, anchor_y=0.0, width=COMPACT_MODE_WIDTH),
        _bridge(
            "bridge_left_right",
            mass_fraction=0.36,
            left_anchor=(-2.35, 0.0),
            right_anchor=(2.35, 0.0),
            width=0.25,
            left_id="mode_left",
            right_id="mode_right",
            n_curve_samples=320,
        ),
    ],
)

THREE_PEAKS_COMPACT = _make_spec(
    "three_peaks_compact",
    [
        _mode("mode_0", mass_fraction=1.0 / 3.0, anchor_x=-1.72, anchor_y=-1.12, width=COMPACT_MODE_WIDTH),
        _mode("mode_1", mass_fraction=1.0 / 3.0, anchor_x=1.72, anchor_y=-1.12, width=COMPACT_MODE_WIDTH),
        _mode("mode_2", mass_fraction=1.0 / 3.0, anchor_x=0.00, anchor_y=1.62, width=COMPACT_MODE_WIDTH),
    ],
)


V0_DISTRIBUTIONS: list[V0Distribution] = [
    V0Distribution("one_peak_compact", ONE_PEAK_COMPACT, "one peak; very compact"),
    V0Distribution("one_peak_diffuse", ONE_PEAK_DIFFUSE, "one broad homogeneous region"),
    V0Distribution("one_peak_elongated", ONE_PEAK_ELONGATED, "one anisotropic strip"),
    V0Distribution("one_peak_spiral", ONE_PEAK_SPIRAL, "one curved trend"),
    V0Distribution("two_peaks_high_bridge", TWO_PEAKS_HIGH_BRIDGE, "two peaks; high bridge"),
    V0Distribution("two_peaks_low_bridge", TWO_PEAKS_LOW_BRIDGE, "two peaks; weak bridge"),
    V0Distribution("two_peaks_no_bridge", TWO_PEAKS_NO_BRIDGE, "two peaks; no bridge"),
    V0Distribution("three_peaks_compact", THREE_PEAKS_COMPACT, "three compact modes"),
]


V0_DISTRIBUTIONS_BY_ID = {spec.distribution_id: spec for spec in V0_DISTRIBUTIONS}
