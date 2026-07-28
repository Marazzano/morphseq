"""Validate V0 synthetic distributions against Stage 2a `compute_resolved_peaks`
(COMPOSE_single_path_plan.md). This is the ontology-migrated counterpart of the
old `plot_modal_v0_bandwidth_comparison.py`'s direct `detect_peaks` calls: it
runs each V0 distribution through the ONE compositional path (`derive_shared_grid`
-> `DistributionRecord` -> `compute_resolved_peaks`) and checks the foreground
fields (`resolved_peak_count`, `is_reliable`) against the expected peak count
encoded in each distribution's name/note.

Run:
    conda run -n morphseq-env --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/v0/validate_v0_resolved_peaks.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.distribution_records import (  # noqa: E402
    DistributionAnalysisContext,
    DistributionRecord,
    PeakResolutionConfig,
    compute_resolved_peaks,
)
from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec  # noqa: E402
from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    draw_resolved_peak_basins,
    format_density_axis,
    mode_count_label,
    plot_kde_field,
    plot_raw_points,
    square_density_box,
)
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS  # noqa: E402

DEFAULT_PLOT_OUT = RUN_DIR / "plots" / "v0_resolved_peaks_validation.png"


# Expected resolved_peak_count per V0 distribution. `None` marks cases where
# the honest answer is genuinely ambiguous (e.g. a strong bridge may fuse two
# modes into one, or the vote may legitimately split) -- those are reported,
# not asserted.
EXPECTED_PEAK_COUNT: dict[str, int | None] = {
    "one_peak_compact": 1,
    "one_peak_diffuse": 1,
    "one_peak_elongated": 1,
    "one_peak_spiral": 1,
    "two_peaks_no_bridge": 2,
    "two_peaks_low_bridge": 2,
    "two_peaks_high_bridge": None,  # ambiguous by design; report only
    "three_peaks_compact": 3,
}


def _analysis_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="longest_non_outlier_MST_edge",
        bandwidth_multiplier=0.75,
        peak_detector_method="kde_peak_basins_sample_support",
    )


@dataclass(frozen=True)
class V0ValidationResult:
    distribution_id: str
    expected_peak_count: int | None
    resolved_peak_count: int | None
    is_reliable: bool
    mode_frequency: float
    passed: bool | None  # None when expected is ambiguous (report only)
    record: DistributionRecord


def derive_pooled_grid(all_points: list[np.ndarray], *, grid_size: int = 81, margin: float = 0.1):
    """ONE `CanonicalGrid` shared across every V0 distribution, so panels are
    directly comparable on the same box/scale (autoscale-per-panel hides that
    an "elongated"/"spiral" distribution's support is actually much larger
    than a "compact" one's -- see COMPOSE_single_path_plan.md's shared-frame
    invariant). Same union-bbox + margin convention as
    `distribution_records.derive_shared_grid`, generalized from 2 point sets
    to N."""
    from morphseq_investigation.core.density_composition import CanonicalGrid

    pooled = np.concatenate([np.asarray(p, dtype=float) for p in all_points], axis=0)
    x_lo, x_hi = float(pooled[:, 0].min()), float(pooled[:, 0].max())
    y_lo, y_hi = float(pooled[:, 1].min()), float(pooled[:, 1].max())
    x_pad = (x_hi - x_lo) * margin if x_hi > x_lo else 1.0
    y_pad = (y_hi - y_lo) * margin if y_hi > y_lo else 1.0
    x_min, x_max, y_min, y_max = square_density_box(
        (x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad)
    )
    return CanonicalGrid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, grid_size=grid_size)


def validate_one(distribution_id: str, points: np.ndarray, config: PeakResolutionConfig, grid) -> V0ValidationResult:
    record = DistributionRecord(
        distribution_id=distribution_id,
        points=points,
        analysis_context=DistributionAnalysisContext(grid=grid, spec=_analysis_spec()),
    )
    record = compute_resolved_peaks(record, config)
    dist = record.resolved_peaks

    expected = EXPECTED_PEAK_COUNT[distribution_id]
    passed = None if expected is None else (dist.resolved_peak_count == expected)

    return V0ValidationResult(
        distribution_id=distribution_id,
        expected_peak_count=expected,
        resolved_peak_count=dist.resolved_peak_count,
        is_reliable=dist.is_reliable,
        mode_frequency=dist.resolution_evidence.peak_resolution_summary.mode_frequency
        if dist.resolution_evidence is not None
        else float("nan"),
        passed=passed,
        record=record,
    )


def plot_validation_grid(results: list[V0ValidationResult], out_path: Path, *, title: str) -> Path:
    """One panel per V0 distribution: KDE field + raw points + resolved peak
    basins (via `draw_resolved_peak_basins`), titled with the vote-supported
    mode count (via `mode_count_label`) and PASS/FAIL/SKIP status."""
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)

    status_color = {"PASS": "#1B7837", "FAIL": "#B2182B", "SKIP": "#808080"}

    for i, r in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        dist = r.record.resolved_peaks
        grid = r.record.canonical_grid
        box = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)

        plot_kde_field(ax, dist.density_grid, cmap="Blues")
        plot_raw_points(ax, r.record.points)
        draw_resolved_peak_basins(ax, dist, color="#333333", linewidth=1.6, n_modes=dist.resolved_peak_count)
        format_density_axis(ax, box)

        status = "SKIP" if r.passed is None else ("PASS" if r.passed else "FAIL")
        color = status_color[status]
        label = mode_count_label(dist.resolved_peak_count, frequency=r.mode_frequency)
        expected_str = "any" if r.expected_peak_count is None else str(r.expected_peak_count)
        ax.set_title(
            f"{r.distribution_id}\n{label}  (expected {expected_str})  {status}",
            fontsize=8.5, color=color,
        )

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=200, help="Sample size per V0 distribution.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point realization.")
    parser.add_argument("--n-bootstrap-draws", type=int, default=80)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=0.80)
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT_OUT, help="Output PNG path.")
    parser.add_argument("--no-plot", action="store_true", help="Skip rendering the validation grid PNG.")
    args = parser.parse_args()

    config = PeakResolutionConfig(
        n_bootstrap_draws=args.n_bootstrap_draws,
        bootstrap_sample_fraction=args.bootstrap_sample_fraction,
        seed=args.seed,
    )

    rng_master = np.random.default_rng(args.seed)
    realizations = []
    for spec in V0_DISTRIBUTIONS:
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        realizations.append((spec.distribution_id, spec.realize(args.num_samples, rng).points))

    # ONE shared grid across all 8 distributions -- panels must be
    # comparable on the same box/scale, not autoscaled per-panel.
    shared_grid = derive_pooled_grid([points for _, points in realizations])

    results: list[V0ValidationResult] = [
        validate_one(distribution_id, points, config, shared_grid)
        for distribution_id, points in realizations
    ]

    header = f"{'distribution_id':<22} {'expected':>8} {'resolved':>8} {'reliable':>8} {'mode_freq':>9}  status"
    print(header)
    print("-" * len(header))
    n_fail = 0
    for r in results:
        status = "SKIP (ambiguous)" if r.passed is None else ("PASS" if r.passed else "FAIL")
        if r.passed is False:
            n_fail += 1
        print(
            f"{r.distribution_id:<22} {str(r.expected_peak_count):>8} {str(r.resolved_peak_count):>8} "
            f"{str(r.is_reliable):>8} {r.mode_frequency:>9.2f}  {status}"
        )

    print()

    if not args.no_plot:
        out = plot_validation_grid(
            results, args.plot_out,
            title=f"V0 resolved-peak validation (n={args.num_samples}, seed={args.seed}, "
                  f"draws={args.n_bootstrap_draws})",
        )
        print(f"Saved: {out}")

    if n_fail:
        print(f"{n_fail} distribution(s) FAILED expected peak-count recovery.")
        sys.exit(1)
    print("All non-ambiguous distributions recovered their expected peak count.")


if __name__ == "__main__":
    main()
