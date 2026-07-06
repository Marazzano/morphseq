"""Minimal V0 synthetic distributions for modal-organization visual QA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Generator = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]

MODE_WIDTH_COMPACT = 0.22
MODE_WIDTH_TWO_PEAK = 0.76


@dataclass(frozen=True)
class V0Distribution:
    distribution_id: str
    generator: Generator
    note: str


def _labels(n: int, value: str) -> np.ndarray:
    return np.full(n, value, dtype=object)


def one_peak_compact(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pts = rng.normal(0.0, MODE_WIDTH_COMPACT, size=(n, 2))
    return pts, _labels(n, "mode_0")


def one_peak_diffuse(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One broad, near-homogeneous density region.

    This is intentionally not just a wider Gaussian. It should read visually as
    one diffuse support region with comparatively even density.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = 2.25 * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    pts += rng.normal(0.0, 0.05, size=(n, 2))
    return pts, _labels(n, "mode_0")


def one_peak_elongated(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One anisotropic peak: a long strip with narrow local width."""
    x = rng.uniform(-3.8, 3.8, size=n)
    y = rng.normal(0.0, 0.17, size=n)
    pts = np.column_stack([x, y])
    pts[:, 0] += rng.normal(0.0, 0.08, size=n)
    return pts, _labels(n, "mode_0")


def one_peak_spiral(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    theta = 1.45 * 2.0 * np.pi * t
    radius = 0.45 + 4.0 * t
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    pts += rng.normal(0.0, 0.18, size=(n, 2))
    return pts, _labels(n, "mode_0")


def _two_peaks_with_bridge(
    n: int,
    rng: np.random.Generator,
    *,
    bridge_frac: float,
    separation: float = 4.8,
    sigma: float = MODE_WIDTH_TWO_PEAK,
) -> tuple[np.ndarray, np.ndarray]:
    n_bridge = int(round(n * bridge_frac))
    n_modes = n - n_bridge
    n0 = n_modes // 2
    n1 = n_modes - n0
    left = rng.normal([-separation / 2.0, 0.0], sigma, size=(n0, 2))
    right = rng.normal([separation / 2.0, 0.0], sigma, size=(n1, 2))
    labels = ["mode_0"] * n0 + ["mode_1"] * n1
    parts = [left, right]
    if n_bridge > 0:
        x = rng.uniform(-separation / 2.0, separation / 2.0, size=n_bridge)
        y = rng.normal(0.0, sigma * 0.75, size=n_bridge)
        bridge = np.column_stack([x, y])
        parts.append(bridge)
        labels.extend(["bridge"] * n_bridge)
    pts = np.vstack(parts)
    labels_arr = np.asarray(labels, dtype=object)
    order = rng.permutation(len(pts))
    return pts[order], labels_arr[order]


def two_peaks_no_bridge(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return _two_peaks_with_bridge(n, rng, bridge_frac=0.0)


def two_peaks_low_bridge(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return _two_peaks_with_bridge(n, rng, bridge_frac=0.06)


def two_peaks_high_bridge(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return _two_peaks_with_bridge(n, rng, bridge_frac=0.28)


def spiral_beaded(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three compact beads placed along the same spiral geometry."""
    bead_t = np.array([0.16, 0.50, 0.84])
    theta = 1.45 * 2.0 * np.pi * bead_t
    radius = 0.45 + 4.0 * bead_t
    centers = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    counts = np.full(len(centers), n // len(centers), dtype=int)
    counts[-1] += n - int(counts.sum())
    pts = []
    labels = []
    for i, (center, count) in enumerate(zip(centers, counts)):
        pts.append(rng.normal(center, MODE_WIDTH_COMPACT, size=(int(count), 2)))
        labels.extend([f"mode_{i}"] * int(count))
    all_pts = np.vstack(pts)
    labels_arr = np.asarray(labels, dtype=object)
    order = rng.permutation(len(all_pts))
    return all_pts[order], labels_arr[order]


def three_peaks_compact(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three compact modes with a broad global footprint."""
    centers = np.asarray(
        [
            [-1.45, -1.05],
            [1.45, -1.05],
            [0.00, 1.45],
        ]
    )
    counts = np.full(len(centers), n // len(centers), dtype=int)
    counts[-1] += n - int(counts.sum())
    pts = []
    labels = []
    for i, (center, count) in enumerate(zip(centers, counts)):
        pts.append(rng.normal(center, MODE_WIDTH_COMPACT, size=(int(count), 2)))
        labels.extend([f"mode_{i}"] * int(count))
    all_pts = np.vstack(pts)
    labels_arr = np.asarray(labels, dtype=object)
    order = rng.permutation(len(all_pts))
    return all_pts[order], labels_arr[order]


V0_DISTRIBUTIONS: list[V0Distribution] = [
    V0Distribution("one_peak_compact", one_peak_compact, "one peak; very compact"),
    V0Distribution("one_peak_diffuse", one_peak_diffuse, "one broad homogeneous region"),
    V0Distribution("one_peak_elongated", one_peak_elongated, "one anisotropic strip"),
    V0Distribution("one_peak_spiral", one_peak_spiral, "one curved trend"),
    V0Distribution("spiral_beaded", spiral_beaded, "three beads along one spiral geometry"),
    V0Distribution("two_peaks_high_bridge", two_peaks_high_bridge, "two peaks; high bridge"),
    V0Distribution("two_peaks_low_bridge", two_peaks_low_bridge, "two peaks; weak bridge"),
    V0Distribution("two_peaks_no_bridge", two_peaks_no_bridge, "two peaks; no bridge"),
    V0Distribution("three_peaks_compact", three_peaks_compact, "three compact modes"),
]


V0_DISTRIBUTIONS_BY_ID = {spec.distribution_id: spec for spec in V0_DISTRIBUTIONS}
