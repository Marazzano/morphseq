"""Empirical peak-count helper for modal-organization density landscapes.

This module keeps the current valley-sweep peak calling in one place so the same
definition can be reused by visual QA, benchmark scripts, and later validators.
It is intentionally simple and empirical: sweep a super-level threshold down from
the peak, then count the first threshold where two or more mass-significant
components appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as ndi_label


@dataclass(frozen=True)
class PeakCountDetail:
    """Peak-count result for a single density field."""

    n_modes: int
    peak_density: float
    total_mass: float
    split_fraction: float | None
    split_level: float | None
    n_components_at_split: int
    component_mass_fractions: tuple[float, ...]


def peak_count_detail(
    density: np.ndarray,
    *,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> PeakCountDetail:
    """Return the empirical peak-count detail for a density field.

    The rule matches the existing valley visualization logic:

    - sweep a threshold from near-peak down toward zero
    - identify the first level whose super-level set splits into >=2 components
    - require at least two components to carry at least `min_component_mass_frac`
      of the total mass
    - if no such level exists, fall back to a single peak and a half-peak ring
      level for visualization
    """

    dens = np.asarray(density, dtype=float)
    dens = np.where(np.isfinite(dens), dens, 0.0)
    peak = float(np.max(dens)) if dens.size else 0.0
    total = float(np.sum(dens)) if dens.size else 0.0
    if peak <= 0 or total <= 0:
        return PeakCountDetail(
            n_modes=0,
            peak_density=peak,
            total_mass=total,
            split_fraction=None,
            split_level=None,
            n_components_at_split=0,
            component_mass_fractions=(),
        )

    for frac in np.linspace(float(start_frac), float(stop_frac), int(sweep_steps)):
        level = float(frac * peak)
        labels, n_components = ndi_label(dens >= level)
        if n_components < 2:
            continue
        masses = np.asarray(
            [dens[labels == k].sum() / total for k in range(1, n_components + 1)],
            dtype=float,
        )
        n_modes = int(np.sum(masses >= float(min_component_mass_frac)))
        if n_modes >= 2:
            return PeakCountDetail(
                n_modes=n_modes,
                peak_density=peak,
                total_mass=total,
                split_fraction=float(frac),
                split_level=level,
                n_components_at_split=int(n_components),
                component_mass_fractions=tuple(float(m) for m in masses),
            )

    return PeakCountDetail(
        n_modes=1,
        peak_density=peak,
        total_mass=total,
        split_fraction=None,
        split_level=0.5 * peak,
        n_components_at_split=1,
        component_mass_fractions=(1.0,),
    )


def count_mass_significant_modes(
    density: np.ndarray,
    *,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> tuple[int, float | None]:
    """Return the empirical peak count and the split level used for plotting."""

    detail = peak_count_detail(
        density,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )
    return detail.n_modes, detail.split_level

