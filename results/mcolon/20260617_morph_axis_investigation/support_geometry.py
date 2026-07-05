"""
Stage 2 of the phenotype-geometry decision tree: SUPPORT GEOMETRY.

Job: "Where does probability exist?" -- is a group's 2-D (length x curvature, or
any 2-D axis) support one connected continuum, or does it fracture into
disconnected modes?

This module answers that question with SEVERAL statistics, each probing support
connectivity from a different angle. They are NOT combined into one number and are
NOT expected to agree on every case -- disagreement is biologically informative
until proven otherwise (a weak bridge, a sparse tail, a density-invisible gap will
show up in one statistic and not another). The caller reports them in parallel.

Statistics (required):
  - valley_depth : KDE super-level density separation (density-visible gaps)
  - mst_max_edge : largest unsupported jump in the minimum spanning tree
                   (local gaps + bridges; works at very small n)
  - fiedler      : algebraic connectivity of the kNN graph (global connectivity)
  - conductance  : sharpness of the spectral-cut bottleneck (hard break vs. taper).
                   Corroborating graph statistic alongside MST + Fiedler (does not
                   solely trigger the discrete call).

WILDTYPE has two distinct roles (Axiom 3):
  - reference: WT's OWN shape defines what "normal geometry" is.
  - null: matched-N WT resampling gives the sampling-variability null for each stat.
This module owns the null (the bootstrap p-value machinery). The reference role
lives in how the caller interprets the result.

No plotting, no I/O. Pure statistic library, imported by
`validate_framework.py` (synthetic validation) and `phenotype_geometry.py`
(the orchestrator that runs the full decision tree on real data).

See docs/todos_scratch/morph_axis_discreteness_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import label as ndi_label
from scipy.sparse.csgraph import connected_components, laplacian, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

GRID_SIZE = 30


@dataclass(frozen=True)
class KDESpec:
    """Density-estimator spec for valley_depth's 2-D KDE."""
    estimator: str = "scipy_gaussian"
    bw_method: object | None = None
    bw_scale: float = 1.0


def scipy_gaussian_kde_spec(
    bw_method: object | None = None,
    bw_scale: float = 1.0,
) -> KDESpec:
    """Spec for scipy.stats.gaussian_kde, optionally scaling its bandwidth factor."""
    return KDESpec(estimator="scipy_gaussian", bw_method=bw_method, bw_scale=float(bw_scale))


DensityEvaluator = Callable[[np.ndarray], np.ndarray]


def _scaled_bw_method(bw_method: object | None, bw_scale: float):
    if bw_scale == 1.0:
        return bw_method

    def _bw(kde_obj):
        if bw_method is None:
            base = kde_obj.scotts_factor()
        elif isinstance(bw_method, str):
            if bw_method == "scott":
                base = kde_obj.scotts_factor()
            elif bw_method == "silverman":
                base = kde_obj.silverman_factor()
            else:
                raise ValueError(f"Unsupported scipy gaussian_kde bw_method: {bw_method!r}")
        elif callable(bw_method):
            base = bw_method(kde_obj)
        else:
            base = float(bw_method)
        return float(base) * bw_scale

    return _bw


def make_kde_evaluator(points_2d: np.ndarray, kde: KDESpec | Callable | None = None) -> DensityEvaluator:
    """Build a callable density estimator for already-normalized 2-D points.

    `kde=None` deliberately preserves the old scipy default: `gaussian_kde(pts.T)`.
    """
    pts = np.asarray(points_2d, dtype=float)
    if kde is None:
        scipy_kde = gaussian_kde(pts.T)
        return lambda grid_points: scipy_kde(grid_points)
    if callable(kde) and not isinstance(kde, KDESpec):
        return kde(pts)
    if not isinstance(kde, KDESpec):
        raise TypeError(f"kde must be None, KDESpec, or a factory callable; got {type(kde)!r}")
    if kde.estimator == "scipy_gaussian":
        scipy_kde = gaussian_kde(
            pts.T,
            bw_method=_scaled_bw_method(kde.bw_method, kde.bw_scale),
        )
        return lambda grid_points: scipy_kde(grid_points)
    if kde.estimator == "knn_adaptive":
        raise NotImplementedError("knn_adaptive KDE is reserved for the post-diagnostic phase")
    raise ValueError(f"Unknown KDE estimator: {kde.estimator!r}")


def evaluate_kde_on_grid(
    points_2d: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    kde: KDESpec | Callable | None = None,
) -> np.ndarray:
    """Evaluate the configured 2-D density estimator on an existing meshgrid."""
    evaluator = make_kde_evaluator(points_2d, kde=kde)
    return evaluator(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)


def _midrank_percentile(obs: float, null: np.ndarray) -> float:
    """Percentile of `obs` within `null` using midranks for ties.

    100 * (#null < obs + 0.5 * #null == obs) / len(null). Midrank handling means an
    observed value equal to the whole (degenerate) null lands at 50 -- "not
    separated" -- instead of 100. This matters for valley_depth, whose WT null is
    frequently all-zero: a connected group (obs=0) must read as CENTERED in that
    null, not maximally separated from it.
    """
    null = np.asarray(null, dtype=float)
    below = float(np.sum(null < obs))
    equal = float(np.sum(null == obs))
    return float(100.0 * (below + 0.5 * equal) / len(null))


# ---------------------------------------------------------------------------
# Shape normalization -- kills variance as a confound (spec: global, non-circular)
# ---------------------------------------------------------------------------

def normalize_shape(points_2d: np.ndarray) -> np.ndarray:
    """Whiten a group's points to unit mass/unit scale. Kills variance as a confound.

    Centers on the median (robust to any lopsided tail) and rescales each axis by
    its MAD-derived sigma. This is the "global" normalization the spec requires --
    it accounts for the group's OWN spread so a wide-but-connected continuum doesn't
    look patchy just for being thin. It must never be based on the density valley
    itself (that would be circular).
    """
    pts = np.asarray(points_2d, dtype=float)
    center = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - center), axis=0)
    sigma = mad * 1.4826  # normal-consistent scale estimate
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (pts - center) / sigma


# ---------------------------------------------------------------------------
# Statistic 1: valley depth (KDE super-level density separation)
# ---------------------------------------------------------------------------

def _kde_grid(points_2d: np.ndarray, grid_size: int = GRID_SIZE, *, kde: KDESpec | Callable | None = None):
    """Evaluate a 2-D gaussian KDE of (already-normalized) points on a padded grid."""
    pts = np.asarray(points_2d, dtype=float)

    pad = 1.0
    x_lo, x_hi = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    y_lo, y_hi = pts[:, 1].min() - pad, pts[:, 1].max() + pad
    xs = np.linspace(x_lo, x_hi, grid_size)
    ys = np.linspace(y_lo, y_hi, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    density = evaluate_kde_on_grid(pts, xx, yy, kde=kde)
    return xx, yy, density


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """4-connectivity connected-component labeling of a boolean grid mask."""
    labels, _ = ndi_label(mask)
    return labels


MIN_COMPONENT_MASS_FRAC = 0.03  # ignore fragments holding < 3% of the total KDE mass

# Number of super-level thresholds to check when sweeping from peak down to near-zero.
# Must be dense enough that a narrow split-window (e.g. 3+ unequal-height modes, where
# the "all components simultaneously separate" band can be thin) isn't stepped over
# between two adjacent samples. Verified against three_discrete/small_middle synthetics:
# 25 steps missed a real split at frac=0.841 (both neighboring steps saw "no split");
# 200 steps reliably lands inside it. See git history / FINDING docs for the diagnosis.
VALLEY_SWEEP_STEPS = 200


def _refine_valley_transition(density, total_mass, frac_hi, frac_lo, peak):
    """STUB: not yet needed at VALLEY_SWEEP_STEPS=200 (dense scan already finds the
    known-hard synthetic cases). If a future case needs finer resolution than a flat
    dense sweep can afford, bisect between `frac_hi` (last frac with <2 real
    components) and `frac_lo` (first frac with >=2) to pin the exact transition
    frac, rather than blanket-increasing VALLEY_SWEEP_STEPS everywhere. Note bisection
    only refines a transition the coarse scan already detected -- it cannot discover a
    split the coarse scan stepped over entirely, so this is a refinement of a known
    transition's location, not a substitute for a dense-enough initial scan.
    """
    raise NotImplementedError("bisection refinement not needed yet -- see stub docstring")


def valley_depth(
    points_2d: np.ndarray,
    grid_size: int = GRID_SIZE,
    *,
    kde: KDESpec | Callable | None = None,
) -> float:
    """Scale-and-mass-invariant relative valley depth of a 2-D point cloud's density.

    Points are assumed already `normalize_shape`-d. Builds a KDE, finds the global
    peak, then sweeps a super-level threshold down from that peak; the statistic is
    the threshold (as a fraction of the peak density) at which the super-level set
    FIRST fractures into >=2 connected components that EACH carry a non-trivial
    share of the total density mass. High (near 1) => deep interior valley =>
    discrete. Low (near 0) => one connected blob all the way down => continuous.
    """
    _, _, density = _kde_grid(points_2d, grid_size=grid_size, kde=kde)
    peak = density.max()
    total_mass = density.sum()
    if peak <= 0 or total_mass <= 0:
        return 0.0

    fracs = np.linspace(0.97, 0.02, VALLEY_SWEEP_STEPS)
    for frac in fracs:
        mask = density >= frac * peak
        if not mask.any():
            continue
        labels = _connected_components(mask)
        n_labels = labels.max()
        if n_labels < 2:
            continue
        component_masses = np.array([
            density[labels == lbl].sum() / total_mass for lbl in range(1, n_labels + 1)
        ])
        n_real_components = int(np.sum(component_masses >= MIN_COMPONENT_MASS_FRAC))
        if n_real_components >= 2:
            return float(frac)
    return 0.0


def valley_detection_detail(
    points_2d: np.ndarray,
    grid_size: int = GRID_SIZE,
    *,
    kde: KDESpec | Callable | None = None,
) -> dict:
    """Same computation as `valley_depth`, but returns everything needed to VISUALIZE
    it: the KDE grid, the density field, the detected valley threshold (fraction of
    peak, 0.0 if none), and the super-level component mask AT that threshold.

    This is the single source of truth for the valley visualization -- it re-runs the
    identical KDE + threshold sweep so the picture matches the statistic exactly.

    Returns dict with keys:
      xx, yy      : meshgrid coords (grid_size x grid_size)
      density     : KDE density on the grid
      peak        : peak density
      valley_frac : the threshold fraction where >=2 real components appear (0.0 if none)
      level       : valley_frac * peak (the absolute density 'waterline'); None if none
      labels      : connected-component labels of the super-level set at that level
                    (all zeros if no valley)
    """
    xx, yy, density = _kde_grid(points_2d, grid_size=grid_size, kde=kde)
    peak = float(density.max())
    total_mass = float(density.sum())
    out = {"xx": xx, "yy": yy, "density": density, "peak": peak,
           "valley_frac": 0.0, "level": None,
           "labels": np.zeros_like(density, dtype=int),
           "n_components_at_split": 0,
           "component_mass_fracs": np.array([], dtype=float)}
    if peak <= 0 or total_mass <= 0:
        return out

    for frac in np.linspace(0.97, 0.02, VALLEY_SWEEP_STEPS):
        mask = density >= frac * peak
        if not mask.any():
            continue
        labels = _connected_components(mask)
        n_labels = labels.max()
        if n_labels < 2:
            continue
        component_masses = np.array([
            density[labels == lbl].sum() / total_mass for lbl in range(1, n_labels + 1)
        ])
        if int(np.sum(component_masses >= MIN_COMPONENT_MASS_FRAC)) >= 2:
            out["valley_frac"] = float(frac)
            out["level"] = float(frac * peak)
            out["labels"] = labels
            out["n_components_at_split"] = int(np.sum(component_masses >= MIN_COMPONENT_MASS_FRAC))
            out["component_mass_fracs"] = component_masses
            return out
    return out


# ---------------------------------------------------------------------------
# Statistic 2: MST max-edge (largest unsupported jump)
# ---------------------------------------------------------------------------

def mst_max_edge(points_2d: np.ndarray) -> float:
    """Largest edge in the Euclidean minimum spanning tree, normalized by the median
    edge length.

    Interpretation: a connected continuum has MST edges of roughly uniform length
    (points are evenly reachable); a broken support forces one long bridge edge to
    span the gap between components. The statistic is
        max_edge / median_edge
    so it is scale-free (already operating on normalize_shape-d points, but the
    ratio makes it robust anyway). A value near 1-2 means uniform spacing
    (continuous); a large value means one jump dominates (a gap).

    Works at very small n: for n points the MST has n-1 edges, so n=5 still gives a
    usable 4-edge distribution. This is the low-n workhorse of Stage 2.
    """
    pts = np.asarray(points_2d, dtype=float)
    n = len(pts)
    if n < 3:
        return 0.0
    dmat = squareform(pdist(pts))
    mst = minimum_spanning_tree(dmat)
    edges = mst.data
    edges = edges[edges > 0]
    if edges.size == 0:
        return 0.0
    median_edge = np.median(edges)
    if median_edge < 1e-12:
        return 0.0
    return float(edges.max() / median_edge)


# ---------------------------------------------------------------------------
# Statistic 3: Fiedler value (global algebraic connectivity of a kNN graph)
# ---------------------------------------------------------------------------

def _knn_adjacency(points_2d: np.ndarray, k: int) -> np.ndarray:
    """Symmetric kNN adjacency with a LOCALLY-scaled Gaussian (heat-kernel) weight.

    Bandwidth is self-tuning (Zelnik-Manor & Perona 2004): each point i gets its own
    sigma_i = distance to its k-th nearest neighbor, and an edge (i,j) is weighted

        w_ij = exp( -d_ij^2 / (sigma_i * sigma_j) ).

    Why local and not a single global sigma (the old median-kNN bandwidth): a global
    sigma is set by the whole cloud's typical spacing, so once a real GAP opens
    between two modes the global scale inflates -- it either over-smooths the two
    modes into one fuzzy blob (spurious high-conductance cut) or collapses the graph
    into disconnected pieces (degenerate Fiedler/conductance). A verified 2-D sweep
    (two clusters at growing separation) showed the global bandwidth makes the cut
    conductance NON-monotonic / nan past a moderate gap, while the local bandwidth
    keeps it monotonically -> 0 as the gap grows, independent of gap size and of
    whether the points were shape-normalized. So the fragility was the bandwidth, not
    the whitening. Local scaling keeps within-mode neighborhoods tight regardless of
    between-mode distance, which is exactly what a support-connectivity probe needs.

    Symmetrized by construction (sigma_i*sigma_j is symmetric); a one-way neighbor
    still connects both directions.
    """
    pts = np.asarray(points_2d, dtype=float)
    n = len(pts)
    dmat = squareform(pdist(pts))
    # per-point local bandwidth: distance to each point's own k-th nearest neighbor
    sorted_d = np.sort(dmat, axis=1)
    sigma = sorted_d[:, k]  # col 0 is self (dist 0), so col k is the k-th neighbor
    sigma = np.where(sigma < 1e-12, 1.0, sigma)

    adj = np.zeros((n, n))
    for i in range(n):
        nbr_idx = np.argsort(dmat[i])[1:k + 1]
        for j in nbr_idx:
            w = np.exp(-(dmat[i, j] ** 2) / (sigma[i] * sigma[j]))
            adj[i, j] = w
            adj[j, i] = w  # symmetrize
    return adj


def fiedler_value(points_2d: np.ndarray, k: int | None = None) -> float:
    """Algebraic connectivity (2nd-smallest Laplacian eigenvalue) of a kNN graph.

    Returned INVERTED and scaled as a "disconnectedness" statistic so it points the
    same direction as valley_depth and mst_max_edge (larger = more broken):
        stat = 1 / (1 + fiedler)
    The raw Fiedler value is ~0 when the graph is (nearly) disconnected -- a true
    gap makes the second eigenvalue collapse toward zero. Inverting maps
    well-connected (large fiedler) -> small stat, and broken (fiedler->0) -> stat->1.

    k defaults to a n-adaptive choice (roughly log2(n), floored at 3) so the graph
    stays sparse enough to reveal a bottleneck but connected enough to be meaningful.
    Fiedler is unstable below n~8; the caller's confidence machinery accounts for
    that via min_n, not this function.
    """
    pts = np.asarray(points_2d, dtype=float)
    n = len(pts)
    if n < 4:
        return 0.0
    if k is None:
        k = max(3, min(n - 1, int(np.ceil(np.log2(n)))))
    adj = _knn_adjacency(pts, k=k)
    lap = laplacian(adj, normed=True)
    eigvals = np.linalg.eigvalsh(lap)
    eigvals = np.sort(eigvals)
    fiedler = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    fiedler = max(fiedler, 0.0)
    return float(1.0 / (1.0 + fiedler))


# ---------------------------------------------------------------------------
# Statistic 4: conductance of the spectral cut (bottleneck sharpness)
# ---------------------------------------------------------------------------

def conductance(points_2d: np.ndarray, k: int | None = None) -> float:
    """Conductance of the graph's sparsest spectral cut, as a "brokenness" statistic.

    Complements Fiedler: Fiedler asks "is there a bottleneck?" (global connectivity
    magnitude); conductance asks "how SHARP is that bottleneck?" -- a graceful taper
    vs. a hard break. It is the classic Cheeger cut quantity:

        phi(S) = (weight of edges crossing S) / min(vol(S), vol(complement))

    evaluated on the cut induced by the sign of the Fiedler eigenvector (the standard
    spectral bisection, so it targets the graph's genuine bottleneck rather than an
    arbitrary split). Low phi = a thin bridge separating two heavy halves (a real,
    hard gap); high phi = the halves are richly interconnected (a taper / continuum).

    Returned INVERTED so it points the same direction as the other statistics
    (larger = more broken):
        stat = 1 / (1 + phi)
    A hard break drives phi -> 0 -> stat -> 1; a well-mixed cloud keeps phi large ->
    stat -> small. Mirrors the fiedler_value convention exactly.

    Same n-adaptive k and same n>=4 floor as fiedler_value; unstable at very small n,
    which the caller's confidence machinery (STATISTIC_MIN_N) accounts for.
    """
    pts = np.asarray(points_2d, dtype=float)
    n = len(pts)
    if n < 4:
        return 0.0
    if k is None:
        k = max(3, min(n - 1, int(np.ceil(np.log2(n)))))
    adj = _knn_adjacency(pts, k=k)

    # If the graph is already disconnected, the support is MAXIMALLY broken: no edge
    # bridges the components at all. This is the large-gap limit -- the spectral cut
    # below is only defined for a connected graph (a disconnected graph has >=2 zero
    # Laplacian eigenvalues, so the "2nd" eigenvector is an arbitrary component
    # indicator, not a bottleneck direction, and its sign-split degenerates to
    # all-one-side). Short-circuit to the fully-broken value rather than letting that
    # degeneracy fall through to a phi that reads as "connected".
    n_components, _ = connected_components(adj > 0, directed=False)
    if n_components > 1:
        return 1.0  # phi -> 0 (no crossing edges) -> inverted stat -> 1 = max broken

    # Fiedler vector of the normalized Laplacian -> sign gives the spectral bisection.
    lap = laplacian(adj, normed=True)
    eigvals, eigvecs = np.linalg.eigh(lap)
    order = np.argsort(eigvals)
    fiedler_vec = eigvecs[:, order[1]] if eigvecs.shape[1] > 1 else eigvecs[:, 0]
    side = fiedler_vec >= 0.0
    if side.all() or (~side).all():
        # Connected graph but the sign-split still put everything on one side (near-
        # degenerate Fiedler vector -- an essentially-disconnected bottleneck). Treat
        # as maximally broken, consistent with the disconnected branch above.
        return 1.0

    degree = adj.sum(axis=1)
    vol_a = float(degree[side].sum())
    vol_b = float(degree[~side].sum())
    denom = min(vol_a, vol_b)
    if denom <= 1e-12:
        return 1.0  # one side carries no volume -> isolated component -> max broken
    cut = float(adj[np.ix_(side, ~side)].sum())  # weight crossing the cut (one direction, symmetric adj)
    phi = cut / denom
    return float(1.0 / (1.0 + phi))


# ---------------------------------------------------------------------------
# The bundle: run all required statistics, each with its own matched-N WT null
# ---------------------------------------------------------------------------

# Registry of required support-geometry statistics. Each maps a name -> callable.
# Adding a statistic here automatically flows it through the bundle, the bootstrap
# null, and the caller.
SUPPORT_STATISTICS = {
    "valley_depth": valley_depth,
    "mst_max_edge": mst_max_edge,
    "fiedler": fiedler_value,
    "conductance": conductance,
}

# Minimum n at which each statistic is considered estimable (drives per-statistic
# confidence downstream; NOT a hard gate here). Conductance shares Fiedler's spectral
# machinery, so it inherits the same estimability floor.
STATISTIC_MIN_N = {
    "valley_depth": 10,
    "mst_max_edge": 5,
    "fiedler": 8,
    "conductance": 8,
}


@dataclass
class StatResult:
    """One support-geometry statistic's observed value + its matched-N WT null."""
    name: str
    stat: float
    reference_stat: float          # WT-as-null median (Axiom 3: the null role)
    pvalue: float                  # P(WT draw >= observed) -- fraction of null >= stat
    percentile: float              # where the observed value sits in the WT null [0,100]
    null_dist: np.ndarray = field(repr=False)


@dataclass
class SupportGeometryBundle:
    """All support-geometry statistics for one group, each reported in parallel.

    The statistics are NOT combined. `results` maps statistic name -> StatResult.
    `n` is the tested group's sample size. Disagreement between statistics is
    surfaced (via `disagreements`), not resolved -- it is data, not error.
    """
    n: int
    results: dict[str, StatResult]

    def pvalue(self, name: str) -> float:
        return self.results[name].pvalue

    @property
    def any_discrete(self) -> bool:
        """True if ANY required statistic calls the support broken at p<0.05.
        (Raw OR -- use `support_call` for the interpreted, false-positive-guarded
        call. This stays available so disagreements remain fully visible.)"""
        return any(r.pvalue < 0.05 for r in self.results.values())

    @property
    def all_discrete(self) -> bool:
        return all(r.pvalue < 0.05 for r in self.results.values())

    @property
    def support_call(self) -> str:
        """Interpreted support-geometry call: "discrete" | "connected".

        A broken-support call requires DENSITY-VISIBLE separation -- valley_depth
        significant -- because that is what distinguishes a true gap (empty region
        between filled regions) from a merely sparse or curved connected manifold.
        The graph statistics (MST, Fiedler) fire on any thin manifold (a crescent,
        a spiral, a few outliers all have weak global connectivity WITHOUT a real
        gap), so they corroborate but do not solely trigger the discrete call.

        Rule: discrete iff valley_depth is significant AND at least one graph
        statistic (mst_max_edge or fiedler) corroborates. This keeps sparse-but-
        connected manifolds continuous while catching real splits and holes.
        Statistics that dissent from this call are surfaced by `disagreements`.
        """
        vd = self.results.get("valley_depth")
        fied = self.results.get("fiedler")
        mst = self.results.get("mst_max_edge")
        cond = self.results.get("conductance")

        valley_p = vd.pvalue if vd is not None else 1.0
        fiedler_p = fied.pvalue if fied is not None else 1.0
        mst_p = mst.pvalue if mst is not None else 1.0
        conductance_p = cond.pvalue if cond is not None else 1.0

        # A true gap has BOTH a density valley (empty interior between filled
        # regions) AND weak global connectivity. A sparse/curved connected manifold
        # (crescent, spiral, outliers) has weak connectivity but NO density valley,
        # so requiring the valley guards those false positives. Because valley_depth
        # sits right at its significance boundary for tight equal blobs, we relax its
        # bar when Fiedler corroborates strongly:
        #   - valley clearly significant (p<0.05) with graph corroboration, OR
        #   - valley marginal (p<0.12) with STRONG global-connectivity loss
        #     (fiedler p<0.02) -- the two-of-a-kind confirmation.
        graph_corroborates = (fiedler_p < 0.05) or (mst_p < 0.05) or (conductance_p < 0.05)
        strong_disconnect = fiedler_p < 0.02

        is_discrete = (
            (valley_p < 0.05 and graph_corroborates)
            or (valley_p < 0.12 and strong_disconnect)
        )
        return "discrete" if is_discrete else "connected"

    @property
    def disagreements(self) -> list[str]:
        """Names of statistics whose call (discrete vs. not, at p<0.05) is the
        minority -- flags a case worth inspecting biologically."""
        calls = {name: (r.pvalue < 0.05) for name, r in self.results.items()}
        n_discrete = sum(calls.values())
        n_total = len(calls)
        if n_discrete == 0 or n_discrete == n_total:
            return []  # unanimous
        minority_is_discrete = n_discrete <= n_total - n_discrete
        return [name for name, is_disc in calls.items()
                if is_disc == minority_is_discrete]


def compute_support_geometry(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    n_resample: int = 500,
    rng: np.random.Generator | None = None,
    statistics: dict | None = None,
    kde: KDESpec | Callable | None = None,
) -> SupportGeometryBundle:
    """Compute every required support-geometry statistic on `group_pts`, each with
    its own matched-N WT (`reference_pts`) bootstrap null.

    Both pools are normalize_shape-d to themselves before any statistic is computed
    (so scale/variance is not what's being tested). For each statistic, the null is
    built by drawing len(group_pts) points from the reference WITH replacement,
    n_resample times, and recomputing the statistic. The full null distribution is
    retained on each StatResult (storage is cheap; re-thresholding later is free).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if statistics is None:
        statistics = SUPPORT_STATISTICS
    else:
        statistics = dict(statistics)
    if kde is not None and "valley_depth" in statistics:
        statistics["valley_depth"] = lambda pts: valley_depth(pts, kde=kde)

    group_norm = normalize_shape(group_pts)
    ref_norm = normalize_shape(reference_pts)
    n = len(group_pts)

    # Observed statistics on the real group.
    observed = {name: fn(group_norm) for name, fn in statistics.items()}

    # One shared set of resample index draws, so every statistic sees the same WT
    # draws (keeps their nulls comparable rather than independently noisy).
    draw_indices = [
        rng.choice(len(ref_norm), size=n, replace=True) for _ in range(n_resample)
    ]

    results: dict[str, StatResult] = {}
    for name, fn in statistics.items():
        null_stats = np.empty(n_resample)
        for j, idx in enumerate(draw_indices):
            null_stats[j] = fn(ref_norm[idx])
        obs = observed[name]
        pvalue = float(np.mean(null_stats >= obs))
        percentile = _midrank_percentile(obs, null_stats)
        results[name] = StatResult(
            name=name,
            stat=float(obs),
            reference_stat=float(np.median(null_stats)),
            pvalue=pvalue,
            percentile=percentile,
            null_dist=null_stats,
        )

    return SupportGeometryBundle(n=n, results=results)


# ---------------------------------------------------------------------------
# Back-compat shim: the old single-statistic API used elsewhere.
# ---------------------------------------------------------------------------

# Old name kept so existing imports (`relative_valley_depth`) don't break.
relative_valley_depth = valley_depth


@dataclass
class ConnectednessResult:
    """Legacy single-statistic result (valley_depth only). Prefer
    SupportGeometryBundle for new code."""
    stat: float
    reference_stat: float
    pvalue: float
    null_dist: np.ndarray


def connectedness_pvalue(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    n_resample: int = 500,
    rng: np.random.Generator | None = None,
    kde: KDESpec | Callable | None = None,
) -> ConnectednessResult:
    """Legacy entry point: valley_depth statistic only, matched-N WT null.

    Retained for the existing real-data script and simulation asserts. New code
    should call `compute_support_geometry` for the full parallel bundle.
    """
    bundle = compute_support_geometry(
        group_pts, reference_pts, n_resample=n_resample, rng=rng,
        statistics={"valley_depth": valley_depth}, kde=kde,
    )
    r = bundle.results["valley_depth"]
    return ConnectednessResult(
        stat=r.stat,
        reference_stat=r.reference_stat,
        pvalue=r.pvalue,
        null_dist=r.null_dist,
    )
