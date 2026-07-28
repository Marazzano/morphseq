"""
Stage 3 of the phenotype-geometry decision tree: DENSITY GEOMETRY.

Job: "HOW is probability distributed inside a connected support?" This stage only
runs AFTER Stage 2 has established the support is connected -- skewness of a mixture
is not the same biological object as skewness of a connected continuum, so the
conditioning matters.

Support geometry (Stage 2) asked WHERE probability exists. Density geometry asks how
it's shaped on that support: is it compact or broad, symmetric or skewed,
light- or heavy-tailed? These are the classical distribution descriptors, but each
is reported RELATIVE TO the matched-N WT null (Axiom 1) -- "is this spread unusual
for wildtype at this stage and N", never an absolute number.

Descriptors (on the canonical 1-D projection):
  variance, iqr, skewness, kurtosis, tail_index (Hill), entropy

No plotting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kurtosis, skew

from distribution_shift import project_to_axis


# Minimum n at which each descriptor is considered estimable (feeds confidence).
DENSITY_MIN_N = {
    "variance": 5,
    "iqr": 5,
    "skewness": 10,
    "kurtosis": 15,
    "tail_index": 12,
    "entropy": 10,
}


def _variance(x: np.ndarray) -> float:
    return float(np.var(x))


def _iqr(x: np.ndarray) -> float:
    return float(np.subtract(*np.percentile(x, [75, 25])))


def _skewness(x: np.ndarray) -> float:
    return float(skew(x))


def _kurtosis(x: np.ndarray) -> float:
    return float(kurtosis(x))  # excess kurtosis (0 = gaussian)


def _tail_index(x: np.ndarray, tail_frac: float = 0.10) -> float:
    """Hill estimator on the upper tail -- larger = heavier tail.

    Uses |deviations from the median| so it captures either-sided heavy tails.
    Returns 0 if too few tail points to estimate.
    """
    dev = np.abs(x - np.median(x))
    dev = np.sort(dev)[::-1]
    k = max(2, int(np.ceil(tail_frac * len(dev))))
    k = min(k, len(dev) - 1)
    if k < 2:
        return 0.0
    top = dev[:k]
    thresh = dev[k]
    if thresh < 1e-12:
        return 0.0
    logs = np.log(top / thresh)
    hill = float(np.mean(logs))
    return hill


def _entropy(x: np.ndarray, n_bins: int | None = None) -> float:
    """Shannon entropy of a histogram of x (nats). Broader/flatter -> higher."""
    if n_bins is None:
        n_bins = max(5, int(np.sqrt(len(x))))
    hist, _ = np.histogram(x, bins=n_bins, density=False)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


DENSITY_DESCRIPTORS = {
    "variance": _variance,
    "iqr": _iqr,
    "skewness": _skewness,
    "kurtosis": _kurtosis,
    "tail_index": _tail_index,
    "entropy": _entropy,
}


@dataclass
class DescriptorResult:
    name: str
    value: float
    reference_value: float   # WT null median
    percentile: float        # where observed sits in the WT null [0, 100]
    null_dist: np.ndarray = field(repr=False)


@dataclass
class DensityGeometry:
    """All density descriptors for one connected group, each vs. matched-N WT null."""
    n: int
    results: dict[str, DescriptorResult]

    def describe(self) -> dict[str, str]:
        """Human-readable summary: for each descriptor, whether it's elevated /
        reduced / typical relative to WT (>90th, <10th, else)."""
        out = {}
        for name, r in self.results.items():
            if r.percentile >= 90:
                out[name] = "elevated"
            elif r.percentile <= 10:
                out[name] = "reduced"
            else:
                out[name] = "typical"
        return out


def compute_density_geometry(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    n_resample: int = 500,
    rng: np.random.Generator | None = None,
    descriptors: dict | None = None,
) -> DensityGeometry:
    """Compute density descriptors on the group's 1-D projection, each with a
    matched-N WT bootstrap null. Projection axis is shared (combined-cloud PC),
    same as Stage 1, so the descriptors live on a consistent axis."""
    if rng is None:
        rng = np.random.default_rng(0)
    if descriptors is None:
        descriptors = DENSITY_DESCRIPTORS

    n = len(group_pts)
    group_1d, ref_1d = project_to_axis(group_pts, reference_pts)

    observed = {name: fn(group_1d) for name, fn in descriptors.items()}

    n_ref = len(ref_1d)
    draw_indices = [rng.choice(n_ref, size=n, replace=True) for _ in range(n_resample)]

    results: dict[str, DescriptorResult] = {}
    for name, fn in descriptors.items():
        null_stats = np.array([fn(ref_1d[idx]) for idx in draw_indices])
        obs = observed[name]
        below = float(np.sum(null_stats < obs))
        equal = float(np.sum(null_stats == obs))
        percentile = float(100.0 * (below + 0.5 * equal) / len(null_stats))
        results[name] = DescriptorResult(
            name=name,
            value=float(obs),
            reference_value=float(np.median(null_stats)),
            percentile=percentile,
            null_dist=null_stats,
        )

    return DensityGeometry(n=n, results=results)
