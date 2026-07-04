"""
Stage 1 of the phenotype-geometry decision tree: DISTRIBUTION SHIFT.

Job: "Does the mutant differ from wildtype at all?" If not, the axis is
"wildtype-like" and Stages 2-4 are skipped -- there is no geometry worth
describing.

The stage's job is a two-sample DISTRIBUTION COMPARISON; it is not any one metric.
This module exposes a pluggable registry of two-sample statistics behind a common
interface so energy distance / MMD can be added later without touching the
pipeline. Ships with Wasserstein-1 and Jensen-Shannon divergence.

Implementation note: the 2-D cloud is projected onto a canonical 1-D axis (the
dominant PC of the two combined clouds) before the 1-D two-sample statistics run.
That projection is ONE implementation of "compare distributions", not the theory --
the theory is the two-sample comparison and the projection can change.

WT roles (Axiom 3): the WT-vs-WT matched-N bootstrap supplies the NULL (how big a
shift finite sampling fakes); the WT distribution itself is the REFERENCE the
mutant is compared against.

No plotting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance


# ---------------------------------------------------------------------------
# Canonical 1-D projection (implementation of "compare distributions")
# ---------------------------------------------------------------------------

def project_to_axis(group_2d: np.ndarray, ref_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project both clouds onto the dominant PC of their combined points.

    Fitting the axis on the COMBINED cloud makes it a shared coordinate both
    distributions live in, so a shift shows up as a location difference along the
    axis. Returns (group_1d, ref_1d).
    """
    group_2d = np.asarray(group_2d, dtype=float)
    ref_2d = np.asarray(ref_2d, dtype=float)
    combined = np.vstack([group_2d, ref_2d])
    combined_centered = combined - combined.mean(axis=0)
    # dominant right singular vector = first PC
    _, _, vt = np.linalg.svd(combined_centered, full_matrices=False)
    axis = vt[0]
    return group_2d @ axis, ref_2d @ axis


# ---------------------------------------------------------------------------
# Two-sample metric registry. Each: (a_1d, b_1d) -> float. Pluggable.
# ---------------------------------------------------------------------------

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    return float(wasserstein_distance(a, b))


def _js_divergence_1d(a: np.ndarray, b: np.ndarray, n_grid: int = 200) -> float:
    """Jensen-Shannon divergence between KDEs of a and b on a shared grid.

    Bounded in [0, ln 2] and symmetric -- safe at small N where KL blows up. If a
    KDE cannot be built (degenerate spread), falls back to 0 (no detectable shift).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi - lo < 1e-12:
        return 0.0
    pad = 0.1 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    try:
        pa = gaussian_kde(a)(grid)
        pb = gaussian_kde(b)(grid)
    except np.linalg.LinAlgError:
        return 0.0
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-12))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


# Metrics that are safe at low N. Registry maps name -> (callable, min_n).
SHIFT_METRICS = {
    "wasserstein": (_wasserstein_1d, 5),
    "js_divergence": (_js_divergence_1d, 5),
}


@dataclass
class ShiftStatResult:
    name: str
    stat: float
    reference_stat: float   # WT-vs-WT null median
    pvalue: float           # P(WT-vs-WT null >= observed)
    percentile: float       # where observed sits in the WT-vs-WT null [0, 100]
    null_dist: np.ndarray = field(repr=False)


@dataclass
class DistributionShiftResult:
    n: int
    results: dict[str, ShiftStatResult]

    def pvalue(self, name: str) -> float:
        return self.results[name].pvalue

    @property
    def differs_from_wt(self) -> bool:
        """True if ANY registered metric exceeds its WT-vs-WT null at p<0.05."""
        return any(r.pvalue < 0.05 for r in self.results.values())


def compute_distribution_shift(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    n_resample: int = 500,
    rng: np.random.Generator | None = None,
    metrics: dict | None = None,
) -> DistributionShiftResult:
    """Test whether `group_pts` differs from the WT `reference_pts`.

    For each registered metric: compute the observed group-vs-WT statistic, then
    build a WT-vs-WT null by repeatedly splitting the reference into two matched-N
    halves (one of size n = len(group), one of the same size) and recomputing the
    statistic. The full null is retained.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if metrics is None:
        metrics = SHIFT_METRICS

    n = len(group_pts)
    group_1d, ref_1d = project_to_axis(group_pts, reference_pts)

    observed = {name: fn(group_1d, ref_1d) for name, (fn, _mn) in metrics.items()}

    # WT-vs-WT null: two independent matched-n draws from the reference.
    n_ref = len(ref_1d)
    null_draws = []
    for _ in range(n_resample):
        idx_a = rng.choice(n_ref, size=n, replace=True)
        idx_b = rng.choice(n_ref, size=n, replace=True)
        null_draws.append((ref_1d[idx_a], ref_1d[idx_b]))

    results: dict[str, ShiftStatResult] = {}
    for name, (fn, _mn) in metrics.items():
        null_stats = np.array([fn(a, b) for a, b in null_draws])
        obs = observed[name]
        pvalue = float(np.mean(null_stats >= obs))
        below = float(np.sum(null_stats < obs))
        equal = float(np.sum(null_stats == obs))
        percentile = float(100.0 * (below + 0.5 * equal) / len(null_stats))
        results[name] = ShiftStatResult(
            name=name,
            stat=float(obs),
            reference_stat=float(np.median(null_stats)),
            pvalue=pvalue,
            percentile=percentile,
            null_dist=null_stats,
        )

    return DistributionShiftResult(n=n, results=results)
