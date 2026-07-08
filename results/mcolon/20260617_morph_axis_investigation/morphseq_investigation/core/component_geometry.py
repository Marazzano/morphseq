"""
Stage 4 of the phenotype-geometry decision tree: COMPONENT GEOMETRY.

Job: "How many stable components exist?" This stage only runs AFTER Stage 2 has
found the support BROKEN. Counting components is downstream discovery -- it is not
the first question (that was distribution shift), and it is meaningless on a
connected continuum.

Two independent methods estimate the component count so their agreement (or not)
is visible:
  - HDBSCAN: density-based, min_cluster_size scaled to N (floored at 3)
  - GMM + BIC: model-selection over 1..K gaussian components

Disagreement between the two is flagged, not resolved by a vote -- biologically
informative until proven otherwise (a rare middle phenotype, a weak bridge).

HDBSCAN is optional at import time: if the package isn't installed we fall back to
GMM-only and mark hdbscan_n as None.

No plotting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .support_geometry import normalize_shape

try:
    from sklearn.cluster import HDBSCAN  # sklearn >= 1.3 ships HDBSCAN natively
    _HAS_HDBSCAN = True
except Exception:  # pragma: no cover - optional dep
    _HAS_HDBSCAN = False


def _gmm_n_components(pts: np.ndarray, max_k: int = 5) -> tuple[int, list[float]]:
    """Estimate component count by BIC-minimizing GMM over 1..max_k."""
    n = len(pts)
    max_k = max(1, min(max_k, n // 3))
    bics = []
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=0, n_init=2)
            gmm.fit(pts)
            bics.append(gmm.bic(pts))
        except Exception:
            bics.append(np.inf)
    # Prefer the SMALLEST k whose BIC is within a small margin of the global min --
    # BIC's absolute minimum will chase a marginally-better fit and add tiny
    # spurious components (a 2-point third blob for two clean clusters). The margin
    # (1% of the BIC range) enforces parsimony: only add a component when it clearly
    # earns its keep.
    bics_arr = np.asarray(bics, dtype=float)
    finite = bics_arr[np.isfinite(bics_arr)]
    if finite.size == 0:
        return 1, bics
    best_bic = finite.min()
    margin = 0.01 * (finite.max() - finite.min() + 1e-9)
    within = np.where(bics_arr <= best_bic + margin)[0]
    best_k = int(within.min() + 1)
    return best_k, bics


def _hdbscan_n_components(pts: np.ndarray) -> tuple[int | None, int]:
    """Estimate component count with HDBSCAN. Returns (n_clusters, n_noise).
    n_clusters is None if hdbscan is unavailable."""
    if not _HAS_HDBSCAN:
        return None, 0
    n = len(pts)
    min_cluster_size = max(3, n // 8)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    labels = clusterer.fit_predict(pts)
    unique = set(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len([u for u in unique if u != -1])
    return n_clusters, n_noise


@dataclass
class ComponentResult:
    n: int
    gmm_n: int
    gmm_bics: list[float]
    hdbscan_n: int | None
    hdbscan_noise: int
    component_sizes: list[int]

    @property
    def methods_agree(self) -> bool:
        """Whether GMM and HDBSCAN agree on the count (True if HDBSCAN unavailable
        -- nothing to disagree with)."""
        if self.hdbscan_n is None:
            return True
        return self.gmm_n == self.hdbscan_n

    @property
    def n_components(self) -> int:
        """Consensus count for reporting: HDBSCAN when available (density-based,
        no gaussianity assumption), else GMM."""
        return self.hdbscan_n if self.hdbscan_n is not None else self.gmm_n


def compute_component_geometry(group_pts: np.ndarray, max_k: int = 5) -> ComponentResult:
    """Estimate component structure of a group already deemed to have broken support.

    Operates on normalize_shape-d points so scale is not what drives the split.
    """
    pts = normalize_shape(group_pts)
    n = len(pts)

    gmm_n, gmm_bics = _gmm_n_components(pts, max_k=max_k)
    hdbscan_n, hdbscan_noise = _hdbscan_n_components(pts)

    # Component sizes from a GMM hard assignment at the chosen k (always available).
    if gmm_n >= 1 and n >= gmm_n:
        gmm = GaussianMixture(n_components=gmm_n, covariance_type="full",
                              random_state=0, n_init=2).fit(pts)
        labels = gmm.predict(pts)
        sizes = [int(np.sum(labels == c)) for c in range(gmm_n)]
    else:
        sizes = [n]

    return ComponentResult(
        n=n,
        gmm_n=gmm_n,
        gmm_bics=gmm_bics,
        hdbscan_n=hdbscan_n,
        hdbscan_noise=hdbscan_noise,
        component_sizes=sorted(sizes, reverse=True),
    )
