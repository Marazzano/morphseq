"""
Cross-cutting confidence scaffold for the phenotype-geometry framework.

Axiom 2: confidence is ORTHOGONAL to the call. It never changes whether a
distribution is continuous or discrete -- it annotates how much to trust that
conclusion. And it is computed PER STATISTIC, not once per call: N-adequacy differs
by metric (MST works at N=5; a tail index needs N~=20), so reporting a single
confidence for a whole bundle would mislabel a metric that fundamentally can't be
estimated at the available N.

The mathematical object is a CONTINUOUS score in [0, 1]. The named tiers
(insufficient / low / moderate / high) are presentation only -- a binning of the
score. The tier thresholds live in one place (`TIER_BINS`) and can be re-tuned
without touching the score.

Confidence inputs (all continuous):
  - n, min_n         : sample-size adequacy for THIS statistic
  - wt_separation    : how far the observed value sits from the WT null
                       (as a percentile distance from 50, in [0, 1])
  - bootstrap_variance : spread of the WT null (wide null -> less confident)
  - loo_stability    : fraction of leave-one-out refits that keep the same call

No plotting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Tier binning of the continuous score. Presentation only.
TIER_BINS = [
    (0.75, "high"),
    (0.50, "moderate"),
    (0.25, "low"),
    (0.00, "insufficient"),
]


def _tier_for_score(score: float) -> str:
    for threshold, label in TIER_BINS:
        if score >= threshold:
            return label
    return "insufficient"


def n_adequacy(n: int, min_n: int) -> float:
    """Sample-size adequacy for a statistic in [0, 1].

    0 when n < min_n / 2 (fundamentally can't estimate this statistic), ramps to 1
    by n = 2 * min_n. A soft ramp rather than a hard gate so confidence degrades
    gracefully instead of cliff-edging.
    """
    lo = min_n / 2.0
    hi = 2.0 * min_n
    if n <= lo:
        return 0.0
    if n >= hi:
        return 1.0
    return float((n - lo) / (hi - lo))


def separation_from_null(percentile: float) -> float:
    """How far the observed statistic sits from the center of the WT null.

    `percentile` is where the observed value falls in the WT null [0, 100]. A value
    at the 50th percentile is indistinguishable from WT (separation 0); at the 0th
    or 100th it is maximally separated (separation 1). Symmetric because either tail
    is equally "unusual".
    """
    return float(abs(percentile - 50.0) / 50.0)


def null_tightness(null_dist: np.ndarray) -> float:
    """How tight the WT null is, in [0, 1]. A wide null means the statistic is noisy
    at this N and we should trust it less.

    Uses the robust coefficient of variation (IQR / |median|) squashed to [0, 1]:
    tightness = 1 / (1 + robust_cv). Tight null (cv->0) -> 1; diffuse null -> ->0.
    """
    null_dist = np.asarray(null_dist, dtype=float)
    if null_dist.size == 0:
        return 0.0
    med = np.median(null_dist)
    iqr = np.subtract(*np.percentile(null_dist, [75, 25]))
    scale = abs(med) if abs(med) > 1e-9 else (np.std(null_dist) + 1e-9)
    robust_cv = iqr / scale
    return float(1.0 / (1.0 + robust_cv))


@dataclass
class ConfidenceReport:
    """Per-statistic confidence. `score` is the math object in [0, 1]; `tier` is its
    display binning. All raw inputs are retained so the score can be recomputed or
    re-weighted later without rerunning the analysis."""
    statistic: str
    n: int
    min_n: int
    n_adequacy: float
    wt_separation: float
    null_tightness: float
    loo_stability: float
    score: float
    tier: str

    def __repr__(self) -> str:
        return (f"ConfidenceReport({self.statistic}: {self.tier} "
                f"score={self.score:.2f} n={self.n}/{self.min_n})")


# Weights for combining the four continuous inputs into the score. N-adequacy is a
# multiplicative GATE (an unestimable statistic can't be confident no matter how
# separated it looks); the other three are averaged as evidence strength.
def score_confidence(
    statistic: str,
    n: int,
    min_n: int,
    percentile: float,
    null_dist: np.ndarray,
    loo_stability: float = 1.0,
) -> ConfidenceReport:
    """Combine the continuous inputs into a per-statistic confidence report.

    score = n_adequacy * mean(wt_separation, null_tightness, loo_stability)

    N-adequacy multiplies (a gate): if we can't estimate the statistic at this N,
    confidence is low regardless of how extreme the point estimate looks. The other
    three average as evidence strength. All in [0, 1], so the score is too.
    """
    adeq = n_adequacy(n, min_n)
    sep = separation_from_null(percentile)
    tight = null_tightness(null_dist)
    loo = float(np.clip(loo_stability, 0.0, 1.0))

    evidence = np.mean([sep, tight, loo])
    score = float(adeq * evidence)

    return ConfidenceReport(
        statistic=statistic,
        n=n,
        min_n=min_n,
        n_adequacy=adeq,
        wt_separation=sep,
        null_tightness=tight,
        loo_stability=loo,
        score=score,
        tier=_tier_for_score(score),
    )
