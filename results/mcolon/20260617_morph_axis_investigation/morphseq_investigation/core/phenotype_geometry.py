"""
Orchestrator for the WT-calibrated phenotype-geometry decision tree.

This is the ONLY place the conditioning logic lives -- the rule that each stage's
result gates the next. The stage modules (distribution_shift, support_geometry,
density_geometry, component_geometry) stay independent and independently testable;
this module runs them in the conditional order the framework requires and attaches
per-statistic confidence.

Decision tree (each stage conditions the next):
  Stage 1  distribution_shift : different from WT?
             No  -> "wildtype-like", stop.
             Yes -> Stage 2.
  Stage 2  support_geometry   : where does probability exist? connected or broken?
             connected -> Stage 3 (density geometry).
             discrete  -> Stage 4 (component discovery).
  Stage 3  density_geometry   : how is probability distributed inside the continuum?
  Stage 4  component_geometry : how many stable pieces?
  (+)      confidence         : attached per statistic throughout.

Axioms:
  1. Wildtype defines the reference geometry (every stat vs. matched-N WT null).
  2. Confidence is orthogonal to the call (per-statistic, never changes a call).
  3. Reference and null are distinct WT roles (WT shape = reference; WT resample = null).

No plotting, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .component_geometry import ComponentResult, compute_component_geometry
from .confidence import ConfidenceReport, score_confidence
from .density_geometry import (
    DENSITY_MIN_N,
    DensityGeometry,
    compute_density_geometry,
)
from distribution_shift import DistributionShiftResult, compute_distribution_shift
from .support_geometry import (
    KDESpec,
    STATISTIC_MIN_N,
    SupportGeometryBundle,
    compute_support_geometry,
)


@dataclass
class PhenotypeGeometryResult:
    """Full decision-tree outcome for one (group vs. WT) comparison.

    Only the stages that the conditioning actually reached are populated; the rest
    stay None. `terminal_stage` records where the tree stopped.
    """
    n_group: int
    n_wt: int

    # Stage 1 (always run)
    shift: DistributionShiftResult
    changed_from_wt: bool

    # Stage 2 (run iff changed_from_wt)
    support: Optional[SupportGeometryBundle] = None
    support_call: Optional[str] = None  # "connected" | "discrete"

    # Stage 3 (run iff support connected)
    density: Optional[DensityGeometry] = None

    # Stage 4 (run iff support discrete)
    components: Optional[ComponentResult] = None

    # Per-statistic confidence, keyed "stage:statistic"
    confidence: dict[str, ConfidenceReport] = field(default_factory=dict)

    terminal_stage: str = "stage1"  # stage1 | stage2 | stage3 | stage4

    def summary_label(self) -> str:
        if not self.changed_from_wt:
            return "wildtype-like"
        if self.support_call == "connected":
            desc = self.density.describe() if self.density else {}
            elevated = [k for k, v in desc.items() if v == "elevated"]
            tail = f" (elevated: {', '.join(elevated)})" if elevated else ""
            return f"continuous{tail}"
        n_comp = self.components.n_components if self.components else "?"
        return f"discrete ({n_comp} components)"


def _loo_support_stability(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    full_call: str,
    rng: np.random.Generator,
    n_resample: int,
    max_loo: int = 15,
    kde: KDESpec | None = None,
) -> float:
    """Fraction of leave-one-out refits whose support_call matches the full-sample
    call. Caps the number of LOO refits at `max_loo` (subsampling rows) to keep
    cost bounded for large groups."""
    n = len(group_pts)
    if n <= 3:
        return 0.0
    drop_idx = np.arange(n)
    if n > max_loo:
        drop_idx = rng.choice(n, size=max_loo, replace=False)
    agree = 0
    for i in drop_idx:
        loo = np.delete(group_pts, i, axis=0)
        b = compute_support_geometry(loo, reference_pts, n_resample=max(60, n_resample // 4),
                                     rng=rng, kde=kde)
        agree += (b.support_call == full_call)
    return float(agree / len(drop_idx))


def run_phenotype_geometry(
    group_pts: np.ndarray,
    reference_pts: np.ndarray,
    n_resample: int = 500,
    rng: np.random.Generator | None = None,
    compute_loo: bool = True,
    kde: KDESpec | None = None,
) -> PhenotypeGeometryResult:
    """Run the full conditional decision tree for one group against its WT reference.

    `group_pts`, `reference_pts` are (n, 2) arrays on the SAME axis (hand
    morphometrics or an embedding PC). Returns a PhenotypeGeometryResult whose
    populated stages reflect the path the conditioning took.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    group_pts = np.asarray(group_pts, dtype=float)
    reference_pts = np.asarray(reference_pts, dtype=float)
    n_group = len(group_pts)
    n_wt = len(reference_pts)

    # ---- Stage 1: distribution shift (always) ----------------------------
    shift = compute_distribution_shift(group_pts, reference_pts,
                                       n_resample=n_resample, rng=rng)
    changed = shift.differs_from_wt

    result = PhenotypeGeometryResult(
        n_group=n_group, n_wt=n_wt, shift=shift, changed_from_wt=changed,
        terminal_stage="stage1",
    )

    # Confidence for Stage 1 metrics.
    for name, sr in shift.results.items():
        result.confidence[f"stage1:{name}"] = score_confidence(
            statistic=name, n=n_group, min_n=5,
            percentile=sr.percentile, null_dist=sr.null_dist, loo_stability=1.0,
        )

    if not changed:
        return result  # wildtype-like: stop.

    # ---- Stage 2: support geometry ---------------------------------------
    support = compute_support_geometry(group_pts, reference_pts,
                                       n_resample=n_resample, rng=rng, kde=kde)
    call = support.support_call
    result.support = support
    result.support_call = call
    result.terminal_stage = "stage2"

    loo_stab = 1.0
    if compute_loo:
        loo_stab = _loo_support_stability(group_pts, reference_pts, call, rng, n_resample, kde=kde)

    for name, sr in support.results.items():
        result.confidence[f"stage2:{name}"] = score_confidence(
            statistic=name, n=n_group, min_n=STATISTIC_MIN_N.get(name, 8),
            percentile=sr.percentile, null_dist=sr.null_dist, loo_stability=loo_stab,
        )

    # ---- conditional branch ----------------------------------------------
    if call == "connected":
        # Stage 3: density geometry
        density = compute_density_geometry(group_pts, reference_pts,
                                           n_resample=n_resample, rng=rng)
        result.density = density
        result.terminal_stage = "stage3"
        for name, dr in density.results.items():
            result.confidence[f"stage3:{name}"] = score_confidence(
                statistic=name, n=n_group, min_n=DENSITY_MIN_N.get(name, 10),
                percentile=dr.percentile, null_dist=dr.null_dist, loo_stability=1.0,
            )
    else:
        # Stage 4: component discovery
        components = compute_component_geometry(group_pts)
        result.components = components
        result.terminal_stage = "stage4"

    return result
