"""Vertical-slice smoke test for the resolved-peak metrics layer.

Wires points -> KDE -> detect_peaks -> resolve -> summarize on three V0
scenarios (one_peak_compact, two_peaks_no_bridge, three_peaks_compact) --
the 1/2/3-peak regimes, the last being a known-risky case for the current
detector calibration (see MODAL_ORGANIZATION_OPEN_QUESTIONS.md). Emits a
summary table, a per-peak table, a QC overlay figure, and (optionally) one
pooled-label permutation comparison between one_peak_compact and
three_peaks_compact.

This is intentionally narrow -- see PEAK_METRICS_MVP.md and the approved plan
for this pass. Do not extend to the full 8-scenario V0 corpus or an all-by-all
driver here; that is explicitly deferred to a follow-up pass.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from morphseq_investigation.core.density_composition import compose_density_truth, realize_from_truth  # noqa: E402
from morphseq_investigation.core.resolved_peak_analysis import (  # noqa: E402
    DEFAULT_ANALYSIS_SPEC,
    EmpiricalNullSpec,
    ResolvedPeakAnalysisSpec,
    resolve_points_with_analysis_spec,
    resolved_peak_summary_to_row,
    resolved_peak_to_rows,
    run_resolved_peak_permutation_comparison,
)
from morphseq_investigation.core.resolved_peak_metrics import (  # noqa: E402
    ResolvedPeakRunContext,
    summarize_resolved_peak_distribution,
)
from morphseq_investigation.plotting.modal_distribution_plotting import plot_resolved_peak_overlay  # noqa: E402
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS_BY_ID  # noqa: E402


OUT_DIR = RUN_DIR / "morphseq_investigation" / "tables" / "resolved_peak_v0_smoke_test"
PLOT_DIR = RUN_DIR / "morphseq_investigation" / "plots" / "resolved_peak_v0_smoke_test"

SCENARIOS = ("one_peak_compact", "two_peaks_no_bridge", "three_peaks_compact")

# The canonical V0 spec now lives in core (DEFAULT_ANALYSIS_SPEC) so the figure
# reference-readout, the SGE array path, and this smoke test share one config.
ANALYSIS_SPEC = DEFAULT_ANALYSIS_SPEC


def build_smoke_rows(n: int, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows: list[dict] = []
    peak_rows: list[dict] = []
    distributions_by_scenario_seed: dict[tuple[str, int], object] = {}

    for scenario_id in SCENARIOS:
        dist = V0_DISTRIBUTIONS_BY_ID[scenario_id]
        truth = compose_density_truth(dist.density_spec)
        canonical_grid = truth.composed_grid.grid

        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, hash(scenario_id) % (2**31)]))
            realization = realize_from_truth(truth, n=n, rng=rng)

            distribution = resolve_points_with_analysis_spec(
                distribution_id=f"{scenario_id}_seed{seed}",
                points=realization.points,
                canonical_grid=canonical_grid,
                analysis_spec=ANALYSIS_SPEC,
            )
            summary = summarize_resolved_peak_distribution(distribution)

            context = ResolvedPeakRunContext(
                analysis_id="resolved_peak_v0_smoke_test",
                scenario_id=scenario_id,
                replicate_id=f"seed{seed}",
                seed=seed,
                n=n,
                bandwidth_rule=ANALYSIS_SPEC.bandwidth_rule,
                bandwidth_multiplier=ANALYSIS_SPEC.bandwidth_multiplier,
                bandwidth_value=float("nan"),
                peak_detector_method=ANALYSIS_SPEC.peak_detector_method,
                canonical_grid_id=scenario_id,
                assignment_rule=ANALYSIS_SPEC.assignment_rule,
            )
            summary_rows.append(resolved_peak_summary_to_row(summary, context=context))
            peak_rows.extend(resolved_peak_to_rows(distribution))
            distributions_by_scenario_seed[(scenario_id, seed)] = (distribution, summary, canonical_grid)

    return pd.DataFrame(summary_rows), pd.DataFrame(peak_rows), distributions_by_scenario_seed


def build_qc_figure(distributions_by_scenario_seed: dict, seeds: list[int], out_path: Path) -> None:
    n_rows = len(SCENARIOS)
    n_cols = len(seeds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)

    for row_idx, scenario_id in enumerate(SCENARIOS):
        for col_idx, seed in enumerate(seeds):
            ax = axes[row_idx][col_idx]
            distribution, summary, _ = distributions_by_scenario_seed[(scenario_id, seed)]
            plot_resolved_peak_overlay(ax, distribution, summary, title=f"{scenario_id} (seed={seed})")

    fig.suptitle("Resolved-peak QC: one_peak_compact / two_peaks_no_bridge / three_peaks_compact", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def run_permutation_comparison(n: int, seed: int) -> pd.DataFrame:
    reference_dist = V0_DISTRIBUTIONS_BY_ID["one_peak_compact"]
    target_dist = V0_DISTRIBUTIONS_BY_ID["three_peaks_compact"]

    reference_truth = compose_density_truth(reference_dist.density_spec)
    target_truth = compose_density_truth(target_dist.density_spec)

    rng_reference = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    rng_target = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    reference_points = realize_from_truth(reference_truth, n=n, rng=rng_reference).points
    target_points = realize_from_truth(target_truth, n=n, rng=rng_target).points

    # Both scenarios share the same canonical-grid construction convention
    # (build_density_spec with the same grid_size/margin defaults), but the
    # composed grids differ in extent per scenario. Use the reference grid's
    # bounds for the shared permutation-comparison canonical grid, since all
    # V0 distributions are centered near the origin at comparable scales.
    canonical_grid = reference_truth.composed_grid.grid

    context = ResolvedPeakRunContext(
        analysis_id="resolved_peak_v0_smoke_test_permutation",
        scenario_id="one_peak_compact_vs_three_peaks_compact",
        replicate_id=f"seed{seed}",
        seed=seed,
        n=n,
        bandwidth_rule=ANALYSIS_SPEC.bandwidth_rule,
        bandwidth_multiplier=ANALYSIS_SPEC.bandwidth_multiplier,
        bandwidth_value=float("nan"),
        peak_detector_method=ANALYSIS_SPEC.peak_detector_method,
        canonical_grid_id="one_peak_compact",
        assignment_rule=ANALYSIS_SPEC.assignment_rule,
    )
    null_spec = EmpiricalNullSpec(method="pooled_label_permutation", n_draws=500, alternative="two-sided")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))

    return run_resolved_peak_permutation_comparison(
        reference_points=reference_points,
        target_points=target_points,
        analysis_spec=ANALYSIS_SPEC,
        null_spec=null_spec,
        context=context,
        rng=rng,
        canonical_grid=canonical_grid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Sample size per replicate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9], help="Seeds to realize per scenario.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for tables.")
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Output directory for the QC figure.")
    parser.add_argument("--run-permutation", action="store_true", help="Also run the one_peak_compact vs three_peaks_compact permutation comparison.")
    args = parser.parse_args()

    summary_df, peak_df, distributions_by_scenario_seed = build_smoke_rows(args.n, args.seeds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "resolved_peak_distribution_summary.csv"
    peak_path = args.out_dir / "resolved_peak_table.csv"
    summary_df.to_csv(summary_path, index=False)
    peak_df.to_csv(peak_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {peak_path}")

    qc_path = args.plot_dir / "resolved_peak_qc.png"
    build_qc_figure(distributions_by_scenario_seed, args.seeds, qc_path)
    print(f"Saved: {qc_path}")

    if args.run_permutation:
        null_test_df = run_permutation_comparison(args.n, args.seeds[0])
        null_test_path = args.out_dir / "resolved_peak_null_test.csv"
        null_test_df.to_csv(null_test_path, index=False)
        print(f"Saved: {null_test_path}")
        print(null_test_df.to_string(index=False))


if __name__ == "__main__":
    main()
