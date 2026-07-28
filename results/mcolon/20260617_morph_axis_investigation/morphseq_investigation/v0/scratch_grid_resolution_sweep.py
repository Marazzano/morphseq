"""Throwaway sweep script (NOT part of core/v0 API): empirical sensitivity of
Stage-2a resolved-peak resolution to the density-GRID resolution knob
(`CanonicalGrid.grid_size`, default 61 via `derive_shared_grid`), as distinct
from the `sweep_steps` threshold-scan resolution knob.

Runs each V0 synthetic distribution (`v0/modal_v0_distributions.py`) through
`compute_resolved_peaks` at grid_size in {61, 46, 31} (~100%/75%/50% of the
default) and records: resolved_peak_count / is_reliable / PASS-FAIL agreement
vs the grid_size=61 baseline, peak-center displacement (matched nearest
peak), valley/saddle height differences (via detector_detail on the
matched-index peak with the deepest saddle), and wall-clock runtime.

Not an implementation change -- does not touch core/ or v0/ production
files. Delete after use.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/v0/scratch_grid_resolution_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.core.density_composition import CanonicalGrid  # noqa: E402
from morphseq_investigation.core.distribution_records import (  # noqa: E402
    DistributionAnalysisContext,
    DistributionRecord,
    PeakResolutionConfig,
    compute_resolved_peaks,
)
from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec  # noqa: E402
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS  # noqa: E402
from morphseq_investigation.v0.validate_v0_resolved_peaks import (  # noqa: E402
    EXPECTED_PEAK_COUNT,
    derive_pooled_grid,
)

GRID_SIZES = [61, 46, 31]  # ~100%, 75%, 50% of default (61 per derive_shared_grid)
NUM_SAMPLES = 200
SEED = 7
N_BOOTSTRAP_DRAWS = 80
BOOTSTRAP_SAMPLE_FRACTION = 0.80


def _analysis_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="longest_non_outlier_MST_edge",
        bandwidth_multiplier=0.75,
        peak_detector_method="kde_peak_basins_sample_support",
    )


def _grid_at(base_grid: CanonicalGrid, grid_size: int) -> CanonicalGrid:
    """Same box, different resolution -- isolates the grid_size knob."""
    return CanonicalGrid(
        x_min=base_grid.x_min, x_max=base_grid.x_max,
        y_min=base_grid.y_min, y_max=base_grid.y_max,
        grid_size=grid_size,
    )


def _match_and_displace(
    centers_a: list[tuple[float, float]], centers_b: list[tuple[float, float]]
) -> list[float]:
    """Greedy nearest-neighbor matching (a -> b), returns per-match Euclidean
    displacement in physical (data) coordinates. Unmatched (count mismatch)
    entries are simply not included -- count mismatch is reported separately."""
    if not centers_a or not centers_b:
        return []
    a = np.asarray(centers_a, dtype=float)
    b = np.asarray(centers_b, dtype=float)
    used_b: set[int] = set()
    displacements = []
    for pt in a:
        dists = np.linalg.norm(b - pt, axis=1)
        order = np.argsort(dists)
        for idx in order:
            if idx not in used_b:
                used_b.add(int(idx))
                displacements.append(float(dists[idx]))
                break
    return displacements


def _saddle_heights(record: DistributionRecord) -> list[float]:
    heights = []
    for peak in record.resolved_peaks.peaks:
        detail = peak.detector_detail
        if detail is not None and detail.nearest_saddle_or_merge_height is not None:
            heights.append(float(detail.nearest_saddle_or_merge_height))
    return heights


def main() -> None:
    config = PeakResolutionConfig(
        n_bootstrap_draws=N_BOOTSTRAP_DRAWS,
        bootstrap_sample_fraction=BOOTSTRAP_SAMPLE_FRACTION,
        seed=SEED,
    )

    rng_master = np.random.default_rng(SEED)
    realizations = []
    for spec in V0_DISTRIBUTIONS:
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        realizations.append((spec.distribution_id, spec.realize(NUM_SAMPLES, rng).points))

    base_grid = derive_pooled_grid([points for _, points in realizations])
    spec = _analysis_spec()

    # results[grid_size][distribution_id] = dict(...)
    results: dict[int, dict[str, dict]] = {gs: {} for gs in GRID_SIZES}
    runtimes: dict[int, float] = {}

    for grid_size in GRID_SIZES:
        grid = _grid_at(base_grid, grid_size)
        t0 = time.perf_counter()
        for distribution_id, points in realizations:
            record = DistributionRecord(
                distribution_id=distribution_id,
                points=points,
                analysis_context=DistributionAnalysisContext(grid=grid, spec=spec),
            )
            record = compute_resolved_peaks(record, config)
            dist = record.resolved_peaks
            expected = EXPECTED_PEAK_COUNT[distribution_id]
            passed = None if expected is None else (dist.resolved_peak_count == expected)
            centers = [p.geometry.center_coordinate for p in dist.peaks]
            results[grid_size][distribution_id] = {
                "resolved_peak_count": dist.resolved_peak_count,
                "is_reliable": dist.is_reliable,
                "passed": passed,
                "centers": centers,
                "saddle_heights": _saddle_heights(record),
            }
        runtimes[grid_size] = time.perf_counter() - t0

    baseline_gs = GRID_SIZES[0]  # 61

    print(f"{'=' * 100}")
    print("PER-SCENARIO RESULTS (grid_size sweep)")
    print(f"{'=' * 100}")
    header = (
        f"{'distribution_id':<22} {'grid':>5} {'expected':>8} {'resolved':>8} "
        f"{'reliable':>8} {'status':>18} {'match_baseline':>15} {'center_disp(mean/max)':>24} "
        f"{'saddle_delta':>14}"
    )
    print(header)
    print("-" * len(header))

    for distribution_id, _ in realizations:
        baseline = results[baseline_gs][distribution_id]
        expected = EXPECTED_PEAK_COUNT[distribution_id]
        for grid_size in GRID_SIZES:
            r = results[grid_size][distribution_id]
            status = "SKIP" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
            match_baseline = (
                "n/a" if grid_size == baseline_gs
                else str(r["resolved_peak_count"] == baseline["resolved_peak_count"])
            )
            if grid_size == baseline_gs:
                disp_str = "baseline"
                saddle_delta_str = "baseline"
            else:
                disps = _match_and_displace(baseline["centers"], r["centers"])
                if disps:
                    disp_str = f"{np.mean(disps):.4f}/{np.max(disps):.4f}"
                else:
                    disp_str = "n/a (count mismatch)"
                bsad = baseline["saddle_heights"]
                rsad = r["saddle_heights"]
                if bsad and rsad:
                    n = min(len(bsad), len(rsad))
                    deltas = [rsad[i] - bsad[i] for i in range(n)]
                    saddle_delta_str = f"{np.mean(deltas):+.5f}"
                else:
                    saddle_delta_str = "n/a"

            print(
                f"{distribution_id:<22} {grid_size:>5} {str(expected):>8} "
                f"{str(r['resolved_peak_count']):>8} {str(r['is_reliable']):>8} {status:>18} "
                f"{match_baseline:>15} {disp_str:>24} {saddle_delta_str:>14}"
            )
        print()

    print(f"{'=' * 100}")
    print("RUNTIME")
    print(f"{'=' * 100}")
    for grid_size in GRID_SIZES:
        print(f"grid_size={grid_size:>3}: {runtimes[grid_size]:.3f}s total for {len(realizations)} distributions "
              f"({runtimes[grid_size] / len(realizations):.4f}s/dist)")

    print()
    print(f"{'=' * 100}")
    print("SUMMARY: PASS/FAIL AGREEMENT vs grid_size=61 BASELINE")
    print(f"{'=' * 100}")
    n_disagree = 0
    for distribution_id, _ in realizations:
        baseline_status = results[baseline_gs][distribution_id]["passed"]
        for grid_size in GRID_SIZES[1:]:
            status = results[grid_size][distribution_id]["passed"]
            if status != baseline_status:
                n_disagree += 1
                print(f"  DISAGREEMENT: {distribution_id} grid_size={grid_size}: "
                      f"passed={status} vs baseline passed={baseline_status}")
    if n_disagree == 0:
        print("  No PASS/FAIL disagreements across any grid_size vs baseline.")
    else:
        print(f"  {n_disagree} disagreement(s) found.")


if __name__ == "__main__":
    main()
