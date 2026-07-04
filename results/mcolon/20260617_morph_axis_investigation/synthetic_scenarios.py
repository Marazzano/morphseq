"""
Synthetic ground-truth scenarios for validating the phenotype-geometry framework.

Every generator returns a 2-D point cloud whose TRUE geometry is known, so we can
check that each stage's statistics behave according to their intended theoretical
interpretation -- NOT to tune thresholds, but to confirm meaning.

Scenario families (spec + review refinements):
  Continuous (support connected):
    unimodal_compact     single tight blob (reference)
    variance_only        WT sigma=1 -> mutant sigma=5: shifted, wider, still ONE blob
    broad_continuum      wide blob (variance only)
    tapered_tail         gaussian core + exponential tail
    crescent             curved 1-D manifold (anti-Gaussianity)
    spiral               strongly non-convex connected manifold (hardest)
    outliers             compact core + a few distant points (must NOT fake a gap)
  Discrete (support broken):
    two_discrete         two well-separated equal blobs
    three_discrete       three well-separated equal blobs
    weak_separation      two blobs, small gap (edge case; disagreement expected)
    small_middle         50 / 5 / 50 three clusters (rare middle must survive)
    continuum_with_hole  one support with an empty interior region (broken support)

Each scenario declares its EXPECTED support-geometry call so the validator can
assert automatically. Sample sizes N=5/8/12/20 exercise the confidence machinery.

No plotting, no I/O here -- pure generators + metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Generators. Each: (n, rng) -> (n, 2) array. Extra shape params are baked via
# closures in the SCENARIOS registry so the validator can call them uniformly.
# ---------------------------------------------------------------------------

def gen_unimodal_compact(n: int, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, sigma, size=(n, 2))


def gen_variance_only(n: int, rng: np.random.Generator, sigma: float = 5.0) -> np.ndarray:
    """Shifted + widened but still ONE connected blob. The key biological control:
    must read Stage 1 = different, Stage 2 = connected, Stage 3 = variance elevated."""
    shift = np.array([4.0, 0.0])
    return rng.normal(0.0, sigma, size=(n, 2)) + shift


def gen_broad_continuum(n: int, rng: np.random.Generator, stretch: float = 4.0) -> np.ndarray:
    return rng.normal(0.0, 1.0 * stretch, size=(n, 2))


def gen_tapered_tail(n: int, rng: np.random.Generator, tail_scale: float = 3.0) -> np.ndarray:
    """Gaussian core with a one-sided exponential tail along x. Connected, skewed."""
    core = rng.normal(0.0, 1.0, size=(n, 2))
    tail_x = rng.exponential(tail_scale, size=n)
    core[:, 0] += tail_x
    return core


def gen_crescent(n: int, rng: np.random.Generator, radius: float = 4.0,
                 arc: float = np.pi, noise: float = 0.35) -> np.ndarray:
    """Points along a curved arc -- a connected 1-D manifold that is NON-Gaussian.
    Exposes methods that secretly assume a blob."""
    t = rng.uniform(-arc / 2, arc / 2, size=n)
    x = radius * np.sin(t)
    y = radius * np.cos(t)
    pts = np.column_stack([x, y])
    pts += rng.normal(0.0, noise, size=(n, 2))
    return pts


def gen_spiral(n: int, rng: np.random.Generator, turns: float = 1.5,
               noise: float = 0.2) -> np.ndarray:
    """A single connected spiral arm -- strongly non-convex connected support.
    The hardest anti-Gaussian test: MST should hold; KDE valley may struggle."""
    t = np.sort(rng.uniform(0, 1, size=n))
    theta = turns * 2 * np.pi * t
    r = 0.5 + 4.0 * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y])
    pts += rng.normal(0.0, noise, size=(n, 2))
    return pts


def gen_outliers(n: int, rng: np.random.Generator, n_out: int | None = None,
                 out_dist: float = 8.0) -> np.ndarray:
    """Compact core plus a few far-flung points. Must NOT be called discrete --
    a handful of outliers is not broken support, it's a heavy tail / noise."""
    if n_out is None:
        n_out = max(1, n // 12)
    n_core = n - n_out
    core = rng.normal(0.0, 1.0, size=(n_core, 2))
    angles = rng.uniform(0, 2 * np.pi, size=n_out)
    outs = np.column_stack([out_dist * np.cos(angles), out_dist * np.sin(angles)])
    outs += rng.normal(0.0, 0.3, size=(n_out, 2))
    return np.vstack([core, outs])


def gen_two_discrete(n: int, rng: np.random.Generator, separation: float = 6.0,
                     mode_sigma: float = 0.6) -> np.ndarray:
    n0 = n // 2
    n1 = n - n0
    b0 = rng.normal([-separation / 2, 0.0], mode_sigma, size=(n0, 2))
    b1 = rng.normal([separation / 2, 0.0], mode_sigma, size=(n1, 2))
    return np.vstack([b0, b1])


def gen_three_discrete(n: int, rng: np.random.Generator, separation: float = 6.0,
                       mode_sigma: float = 0.5) -> np.ndarray:
    thirds = [n // 3, n // 3, n - 2 * (n // 3)]
    centers = [[-separation, 0.0], [0.0, 0.0], [separation, 0.0]]
    blobs = [rng.normal(c, mode_sigma, size=(k, 2)) for c, k in zip(centers, thirds)]
    return np.vstack(blobs)


def gen_weak_separation(n: int, rng: np.random.Generator, separation: float = 2.2,
                        mode_sigma: float = 0.7) -> np.ndarray:
    """Two blobs with a SMALL gap -- deliberate edge case. Statistics may disagree;
    that disagreement is informative, not a failure."""
    return gen_two_discrete(n, rng, separation=separation, mode_sigma=mode_sigma)


def gen_small_middle(n: int, rng: np.random.Generator, separation: float = 6.0,
                     mode_sigma: float = 0.5) -> np.ndarray:
    """Three clusters in 50 / 5 / 50 proportion -- a rare middle phenotype. Some
    methods erase the middle; the framework should preserve broken support."""
    frac = np.array([0.47, 0.06, 0.47])
    counts = np.floor(frac * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    counts = np.maximum(counts, 1)
    centers = [[-separation, 0.0], [0.0, 0.0], [separation, 0.0]]
    blobs = [rng.normal(c, mode_sigma, size=(k, 2)) for c, k in zip(centers, counts)]
    return np.vstack(blobs)


def gen_continuum_with_hole(n: int, rng: np.random.Generator, radius: float = 3.0,
                            hole: float = 1.4) -> np.ndarray:
    """One connected support with an empty interior region (annulus). Support IS
    broken (you cannot cross the middle) even though it's a single ring -- exactly
    where support geometry and density geometry diverge."""
    pts = np.empty((n, 2))
    filled = 0
    while filled < n:
        cand = rng.uniform(-radius, radius, size=(n, 2))
        rr = np.sqrt((cand ** 2).sum(axis=1))
        keep = cand[(rr <= radius) & (rr >= hole)]
        take = min(len(keep), n - filled)
        pts[filled:filled + take] = keep[:take]
        filled += take
    return pts


# ---------------------------------------------------------------------------
# Scenario registry with expected support-geometry calls.
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    generator: Callable[[int, np.random.Generator], np.ndarray]
    expected_support: str          # "connected" | "discrete"
    differs_from_wt: bool          # expected Stage 1 outcome
    note: str = ""


# WT reference for these scenarios is a standard unimodal blob (sigma=1).
def wt_reference(n: int, rng: np.random.Generator) -> np.ndarray:
    return gen_unimodal_compact(n, rng, sigma=1.0)


SCENARIOS: list[Scenario] = [
    Scenario("unimodal_compact", lambda n, r: gen_unimodal_compact(n, r, 1.0),
             "connected", False, "reference-like blob"),
    Scenario("variance_only", lambda n, r: gen_variance_only(n, r, 5.0),
             "connected", True, "shifted + wide, still one blob"),
    Scenario("broad_continuum", lambda n, r: gen_broad_continuum(n, r, 4.0),
             "connected", True, "variance only"),
    Scenario("tapered_tail", lambda n, r: gen_tapered_tail(n, r, 3.0),
             "connected", True, "skewed connected"),
    Scenario("crescent", lambda n, r: gen_crescent(n, r),
             "connected", True, "curved manifold (anti-Gaussian)"),
    Scenario("spiral", lambda n, r: gen_spiral(n, r),
             "connected", True, "non-convex manifold (stress)"),
    Scenario("outliers", lambda n, r: gen_outliers(n, r),
             "connected", True, "core + few outliers, NOT a gap"),
    Scenario("two_discrete", lambda n, r: gen_two_discrete(n, r, 6.0),
             "discrete", True, "two separated blobs"),
    Scenario("three_discrete", lambda n, r: gen_three_discrete(n, r, 6.0),
             "discrete", True, "three separated blobs"),
    Scenario("weak_separation", lambda n, r: gen_weak_separation(n, r, 2.2),
             "discrete", True, "edge case; disagreement expected"),
    Scenario("small_middle", lambda n, r: gen_small_middle(n, r, 6.0),
             "discrete", True, "50/5/50, rare middle must survive"),
    Scenario("continuum_with_hole", lambda n, r: gen_continuum_with_hole(n, r),
             "discrete", True, "annulus: broken support"),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}

SAMPLE_SIZES = [5, 8, 12, 20]
