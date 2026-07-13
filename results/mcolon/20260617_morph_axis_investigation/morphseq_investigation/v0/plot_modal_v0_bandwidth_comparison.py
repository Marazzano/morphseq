"""Render V0 modal QA plots for two bandwidth regimes.

This script emits two figures:
- a conservative local-bandwidth regime
- the current best local-bandwidth regime

Each figure uses the same sampled realizations, but the KDE and peak audit rows
are recomputed under the selected bandwidth rule and show all three peak detectors.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.resolved_peak_analysis import (  # noqa: E402
    DEFAULT_ANALYSIS_SPEC,
    _evaluate_density_for_spec,
)
from morphseq_investigation.core.density_composition import DensityGrid  # noqa: E402
from morphseq_investigation.core.peak_counting import SUPPORTED_METHODS, detect_peaks  # noqa: E402
from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    DistributionVisualSpec,
    plot_v0_distribution_qc_grid,
)
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS  # noqa: E402


DEFAULT_OUT_DIR = RUN_DIR / "plots"
CONSERVATIVE_KDE = ("median_kNN_distance", 1.0)
BEST_KDE = ("longest_non_outlier_MST_edge", 0.75)


@dataclass(frozen=True)
class BandwidthConfig:
    label: str
    bandwidth_rule: str
    bandwidth_multiplier: float
    out_name: str


def build_visual_specs(n: int, seed: int) -> list[DistributionVisualSpec]:
    rng_master = np.random.default_rng(seed)
    out: list[DistributionVisualSpec] = []
    for spec in V0_DISTRIBUTIONS:
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        realization = spec.realize(n, rng)
        out.append(
            DistributionVisualSpec(
                distribution_id=spec.distribution_id,
                points=realization.points,
                component_labels=realization.component_labels,
                composed_grid=realization.truth.composed_grid,
                note=spec.note,
            )
        )
    return out


def _sample_grid_for_spec(spec: DistributionVisualSpec, bandwidth_rule: str, bandwidth_multiplier: float) -> DensityGrid:
    if spec.composed_grid is None or spec.composed_grid.grid is None:
        raise ValueError(f"Distribution {spec.distribution_id!r} is missing a composed grid")
    grid = spec.composed_grid
    points = np.asarray(spec.points, dtype=float)
    analysis_spec = replace(
        DEFAULT_ANALYSIS_SPEC,
        bandwidth_rule=bandwidth_rule,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    density = _evaluate_density_for_spec(points, grid.grid, analysis_spec)
    return DensityGrid(xx=grid.xx, yy=grid.yy, density=density, grid=grid.grid)


def _observed_details_for_spec(
    spec: DistributionVisualSpec,
    sample_grid: DensityGrid,
    *,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 200,
) -> dict[str, object]:
    details: dict[str, object] = {}
    for method_name in SUPPORTED_METHODS:
        details[method_name] = detect_peaks(
            sample_grid.density,
            method=method_name,
            grid=sample_grid,
            sample_points=np.asarray(spec.points, dtype=float),
            min_component_mass_frac=min_component_mass_frac,
            sweep_steps=sweep_steps,
        )
    return details


def render_bandwidth_figure(
    specs: list[DistributionVisualSpec],
    *,
    bandwidth_rule: str,
    bandwidth_multiplier: float,
    out_path: Path,
    title: str,
    seed: int,
    n: int,
) -> Path:
    augmented_specs: list[DistributionVisualSpec] = []
    observed_details_by_method: dict[str, dict[str, object]] = {}
    for spec in specs:
        sample_grid = _sample_grid_for_spec(spec, bandwidth_rule, bandwidth_multiplier)
        augmented_specs.append(
            DistributionVisualSpec(
                distribution_id=spec.distribution_id,
                points=spec.points,
                sampled_grid=sample_grid,
                component_labels=spec.component_labels,
                composed_grid=spec.composed_grid,
                note=spec.note,
            )
        )
        observed_details_by_method[spec.distribution_id] = _observed_details_for_spec(spec, sample_grid)

    out = plot_v0_distribution_qc_grid(
        augmented_specs,
        out_path,
        title=title,
        auto_scale=False,
        include_metric_row=True,
        observed_details_by_method=observed_details_by_method,
        observed_method_order=SUPPORTED_METHODS,
        primary_observed_method="kde_peak_basins_sample_support",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Sample size per V0 distribution.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for output PNGs.")
    args = parser.parse_args()

    specs = build_visual_specs(args.n, args.seed)
    configs = [
        BandwidthConfig(
            label="conservative",
            bandwidth_rule=CONSERVATIVE_KDE[0],
            bandwidth_multiplier=CONSERVATIVE_KDE[1],
            out_name="modal_v0_distribution_qc_conservative.png",
        ),
        BandwidthConfig(
            label="best",
            bandwidth_rule=BEST_KDE[0],
            bandwidth_multiplier=BEST_KDE[1],
            out_name="modal_v0_distribution_qc_best.png",
        ),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        out_path = args.out_dir / config.out_name
        title = (
            f"V0 modal density-composition visual QA ({config.label}; "
            f"n={args.n}, seed={args.seed}, {config.bandwidth_rule} x {config.bandwidth_multiplier:.2f})"
        )
        out = render_bandwidth_figure(
            specs,
            bandwidth_rule=config.bandwidth_rule,
            bandwidth_multiplier=config.bandwidth_multiplier,
            out_path=out_path,
            title=title,
            seed=args.seed,
            n=args.n,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
