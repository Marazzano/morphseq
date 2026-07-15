"""
seam_bridge_init.py
-------------------
Initializer-space seam repair for batch-induced temporal discontinuities.

Motivation
==========
The NaN-aware per-bin UMAP + Procrustes initializer in ``init_embedding.py``
stitches each time bin's layout onto the previous bin's frame. When a new
experiment (batch) first enters at some time bin, its bundle can land at a
*different absolute location* than the biological continuation of the previous
bin — a "cliff" or "seam" in ``x0``. See
``results/mcolon/20260703_raw_latent_clustering_baseline/RELATIVE_GEOMETRY_AUDIT.md``.

The fix is not batch correction of the features. It is a geometric repair of the
*initial layout*: move the observations that have no temporal predecessor
support back onto the manifold of the previous bin, then let ordinary
condensation forces take over.

Diagnostic ladder (this module supports all three rungs via ``mode``)
=====================================================================
- ``mode="off"``   (step 0): ablation / control. Detection + QC still run; no
                             coordinate is moved. This is the ruler for steps 1-2.
- ``mode="init"``  (step 1): apply the repair to ``x0`` before the solver. The
                             corrected layout becomes the fidelity anchor, so no
                             solver force fights the repair.
- ``mode="force"`` (step 2): reserved hook. The *same* shared core (detector +
                             matching + target geometry) is exposed so a decaying
                             solver force can reuse it. Not wired into the solver
                             here; :func:`plan_seam_bridge` returns the moves so a
                             force term can consume them.

The two locus rungs (init vs force) differ ONLY in where the identical planned
move is applied, which is what makes their comparison against step 0 clean.

Key design decisions (resolved with the analyst)
================================================
- **Two-sided support gate.** An observation is eligible for a move only if it is
  unsupported at ``t-1`` *and still supported at* ``t+1``. Unsupported on both
  sides is treated as a genuinely isolated state (novel biology or true outlier)
  and is never moved — only logged. This is the automatic replacement for the
  manual "reject when it may be a novel state" review.
- **Within-genotype / within-group matching.** Predecessor matches are restricted
  to the same label group so a seam repair can never anchor a crispant to a
  control (which would erase the very separation being measured).
- **Rigid slide, not per-point springs.** The cliff is one shared offset. The move
  is a single robust (median) displacement per rigid unit, applied to the whole
  group, so within-bundle geometry (genotype spread, phenotype relationships) is
  carried across untouched. ``rigid_unit`` selects the granularity.
- **Stop at bandwidth h.** The group is slid only until it enters normal
  attraction range ``h`` of the predecessor manifold, then the move stops and
  ordinary condensation takes over. It is never snapped all the way on.

This module is pure numpy and is OFF by default. It does not import the solver
and is not wired into any runner. Labels and batch ids are passed in explicitly;
they are NOT part of the ``aligned_umap_init`` boundary (the initializer contract
forbids condensation from seeing labels), so the repair is a distinct post-init
step that takes ``x0`` and returns a repaired ``x0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["off", "init", "force"]
RigidUnit = Literal["batch_genotype", "batch", "per_point"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SeamBridgeConfig:
    """Configuration for the seam-bridge initializer repair.

    Parameters
    ----------
    mode
        Diagnostic ladder rung. ``"off"`` detects and reports but moves nothing
        (step 0). ``"init"`` applies the move to ``x0`` (step 1). ``"force"``
        plans the move but does not apply it here (step 2 hook for a solver
        force to consume). Default ``"off"`` — the module is inert unless asked.
    attraction_bandwidth
        ``h`` — the normal attraction range. A point counts as "supported" at a
        neighbor time if it has at least one same-group neighbor within ``h``
        there. Also the stop distance for the slide. If ``None`` it is estimated
        as ``bandwidth_quantile`` of within-bin same-group nearest-neighbor
        distances (see :func:`estimate_attraction_bandwidth`).
    bandwidth_quantile
        Quantile used to auto-estimate ``h`` when ``attraction_bandwidth`` is
        None. Uses a robust low quantile so ``h`` reflects genuinely local
        neighborly distance, not the bundle diameter.
    rigid_unit
        Granularity of the rigid slide. ``"batch_genotype"`` (default): one
        displacement per (incoming batch, genotype) group — preserves genotype
        separation and internal shape, safest for the crispant readout.
        ``"batch"``: one displacement for the whole incoming bundle across
        genotypes (assumes an identical batch offset for every genotype).
        ``"per_point"``: each eligible point moves toward its own within-group
        predecessor (most literal, accepts edge noise).
    match_within_label
        If True, predecessor matches are restricted to the same ``labels`` value.
        Independent of ``rigid_unit`` (you can match within genotype but still
        aggregate the displacement per batch).
    min_matched_pairs
        A rigid unit is only moved if it has at least this many valid matched
        predecessor pairs; otherwise it is skipped and logged (too little
        evidence to trust the displacement).
    max_reasonable_slide_mult
        Slides longer than this multiple of ``h`` raise a stronger warning in the
        QC report rather than being executed silently. A very long required slide
        suggests the discontinuity may not be a simple technical offset.
    """

    mode: Mode = "off"
    attraction_bandwidth: float | None = None
    bandwidth_quantile: float = 0.5
    rigid_unit: RigidUnit = "batch_genotype"
    match_within_label: bool = True
    min_matched_pairs: int = 3
    max_reasonable_slide_mult: float = 4.0


# --------------------------------------------------------------------------- #
# Report structures
# --------------------------------------------------------------------------- #
@dataclass
class UnsupportedRecord:
    """One observation (embryo, time) flagged by the support detector."""

    embryo_row: int
    time_index: int
    label: str
    batch: str
    supported_prev: bool  # has a same-group neighbor within h at t-1
    supported_next: bool  # has a same-group neighbor within h at t+1
    nearest_prev_dist: float  # distance to nearest same-group t-1 neighbor (inf if none)
    eligible: bool  # unsupported_prev AND supported_next -> a seam, not an island


@dataclass
class SeamMove:
    """A planned (and possibly applied) rigid move for one unit at one seam."""

    time_index: int
    batch: str
    label: str  # "" when rigid_unit == "batch"
    embryo_rows: list[int]
    n_matched_pairs: int
    displacement: tuple[float, float]  # the rigid vector actually planned
    raw_displacement: tuple[float, float]  # full median match displacement before h-clipping
    initial_gap: float  # group centroid distance to predecessor manifold before move
    final_gap: float  # ... after the planned move (>= 0, target ~ h)
    slide_len: float  # ||displacement||
    applied: bool
    skipped_reason: str | None = None


@dataclass
class SeamBridgeReport:
    """Full diagnostic output of one seam-bridge pass."""

    mode: Mode
    attraction_bandwidth: float
    n_observations: int
    unsupported: list[UnsupportedRecord] = field(default_factory=list)
    moves: list[SeamMove] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- convenience summaries ------------------------------------------- #
    @property
    def n_unsupported_prev(self) -> int:
        return sum(1 for r in self.unsupported if not r.supported_prev)

    @property
    def n_eligible(self) -> int:
        return sum(1 for r in self.unsupported if r.eligible)

    @property
    def n_isolated(self) -> int:
        """Unsupported on BOTH sides — candidate novel state, never moved."""
        return sum(
            1 for r in self.unsupported if not r.supported_prev and not r.supported_next
        )

    @property
    def n_applied_moves(self) -> int:
        return sum(1 for m in self.moves if m.applied)

    def summary(self) -> dict[str, float]:
        n = max(self.n_observations, 1)
        return {
            "mode": self.mode,
            "attraction_bandwidth": self.attraction_bandwidth,
            "n_observations": self.n_observations,
            "n_unsupported_prev": self.n_unsupported_prev,
            "frac_unsupported_prev": self.n_unsupported_prev / n,
            "n_eligible_seam": self.n_eligible,
            "n_isolated_novel_candidates": self.n_isolated,
            "n_planned_moves": len(self.moves),
            "n_applied_moves": self.n_applied_moves,
            "n_warnings": len(self.warnings),
        }


def _group_key(labels: np.ndarray | None, obs: np.ndarray) -> np.ndarray:
    """Integer group key per observed row (all-zeros if labels absent)."""
    if labels is None:
        return np.zeros(obs.size, dtype=int)
    _, inv = np.unique(labels[obs], return_inverse=True)
    return inv


# --------------------------------------------------------------------------- #
# Bandwidth estimation
# --------------------------------------------------------------------------- #
def estimate_attraction_bandwidth(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray | None,
    *,
    quantile: float = 0.5,
    match_within_label: bool = True,
) -> float:
    """Estimate ``h`` from within-bin same-group nearest-neighbor distances.

    For each time bin, compute each observed point's distance to its nearest
    same-group neighbor in that bin, then take ``quantile`` over all such
    distances. A robust low/median quantile reflects genuinely local neighborly
    spacing rather than bundle diameter.
    """
    _, T, _ = x0.shape
    nn_dists: list[float] = []
    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        if obs.size < 2:
            continue
        pos = x0[obs, t, :]
        groups = _group_key(labels, obs) if match_within_label else np.zeros(obs.size)
        for g in np.unique(groups):
            gi = np.flatnonzero(groups == g)
            if gi.size < 2:
                continue
            p = pos[gi]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            nn_dists.extend(d.min(axis=1).tolist())
    if not nn_dists:
        return 1.0
    h = float(np.quantile(np.asarray(nn_dists), quantile))
    return h if np.isfinite(h) and h > 0 else 1.0


# --------------------------------------------------------------------------- #
# Detection: two-sided predecessor support
# --------------------------------------------------------------------------- #
def detect_predecessor_support(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    *,
    attraction_bandwidth: float,
    match_within_label: bool = True,
) -> list[UnsupportedRecord]:
    """Flag observations lacking predecessor support, with the t+1 sibling check.

    An observation ``(i, t)`` is *supported at* ``t-1`` if it has at least one
    same-group observed neighbor within ``attraction_bandwidth`` at ``t-1``.
    Support at ``t+1`` is defined symmetrically. Only observations that are
    unsupported at ``t-1`` are recorded. ``eligible`` (unsupported at ``t-1`` but
    supported at ``t+1``) marks a true seam; unsupported on both sides marks a
    candidate isolated/novel state that must not be moved.
    """
    N_e, T, _ = x0.shape
    records: list[UnsupportedRecord] = []

    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        for i in obs:
            sup_prev, d_prev = _support_at(
                x0, mask, labels, i, t, t - 1,
                attraction_bandwidth, match_within_label,
            )
            if sup_prev:
                continue  # has predecessor support -> not a concern
            sup_next, _ = _support_at(
                x0, mask, labels, i, t, t + 1,
                attraction_bandwidth, match_within_label,
            )
            records.append(
                UnsupportedRecord(
                    embryo_row=int(i),
                    time_index=int(t),
                    label=str(labels[i]) if labels is not None else "",
                    batch=str(batch[i]) if batch is not None else "",
                    supported_prev=False,
                    supported_next=bool(sup_next),
                    nearest_prev_dist=float(d_prev),
                    eligible=bool(sup_next),  # seam iff supported at t+1
                )
            )
    return records


def _support_at(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray | None,
    i: int,
    t: int,
    t_ref: int,
    h: float,
    match_within_label: bool,
) -> tuple[bool, float]:
    """Return (has neighbor within h at t_ref, nearest neighbor distance)."""
    T = x0.shape[1]
    if t_ref < 0 or t_ref >= T:
        return False, np.inf
    ref_obs = np.flatnonzero(mask[:, t_ref])
    if match_within_label and labels is not None:
        ref_obs = ref_obs[labels[ref_obs] == labels[i]]
    if ref_obs.size == 0:
        return False, np.inf
    d = np.linalg.norm(x0[ref_obs, t_ref, :] - x0[i, t, :], axis=-1)
    d_min = float(d.min())
    return d_min <= h, d_min


# --------------------------------------------------------------------------- #
# Ridge scoring: seam severity for any position array
# --------------------------------------------------------------------------- #
def ridge_score(
    positions: np.ndarray,
    mask: np.ndarray,
    *,
    batch: np.ndarray | None = None,
    attraction_bandwidth: float | None = None,
    bandwidth_quantile: float = 0.5,
) -> dict:
    """Score seam severity of a position array. Works on ``x0`` OR condensed output.

    MVP definition (batch- and genotype-agnostic)
    ---------------------------------------------
    Support is a pure manifold-continuity test: an observation ``(i, t)`` is
    *supported* at a neighbor bin if it has ANY observed neighbor within ``h``
    there, regardless of batch or genotype. This measures whether the whole
    manifold is temporally continuous — which is exactly what the 48 hpf ridge
    breaks — without assuming genotype structure.

    This is the comparison ruler for the step 0/1/2 ladder. Run it on ``x0`` (the
    seam is born in the initializer, so it should already show here) and again on
    the condensed positions (did the solver close the seam on its own? = step 0's
    verdict).

    Headline metrics
    ----------------
    - ``frac_one_sided`` — fraction of observations with no neighbor within ``h``
      at ``t-1``. The broad seam indicator.
    - ``frac_two_sided`` — fraction unsupported at BOTH ``t-1`` and ``t+1``
      (isolated on both sides). The honest ridge / genuine-discontinuity signal.

    ``t=0`` observations have no ``t-1`` and ``t=T-1`` have no ``t+1``; those
    boundary directions are treated as "no support available" but such points are
    excluded from the denominator of the side that cannot exist, so the first/last
    bin does not inflate the score.

    Bandwidth ``h`` must be held FIXED across arrays you compare (pass the same
    value for ``x0`` and condensed); an auto-estimated ``h`` shifts with layout
    scale and breaks comparability. When ``None`` it is estimated from
    ``positions`` and returned so it can be reused for the paired call.

    Returns
    -------
    dict with attraction_bandwidth, n_observations, frac_one_sided,
    frac_two_sided, n_one_sided, n_two_sided, per_time, and per_batch
    (per_batch only when ``batch`` is provided).
    """
    N_e, T, _ = positions.shape
    h = attraction_bandwidth
    if h is None:
        h = estimate_attraction_bandwidth(
            positions, mask, labels=None, quantile=bandwidth_quantile,
            match_within_label=False,
        )

    n_obs = int(mask.sum())
    n_one_sided = 0
    n_two_sided = 0
    per_time: dict[int, dict] = {}
    per_batch: dict[str, dict] = {}

    for t in range(T):
        obs = np.flatnonzero(mask[:, t])
        for i in obs:
            sup_prev = _any_neighbor_within(positions, mask, i, t, t - 1, h)
            sup_next = _any_neighbor_within(positions, mask, i, t, t + 1, h)
            has_prev_bin = t - 1 >= 0
            has_next_bin = t + 1 < T
            # one-sided: lacks a predecessor even though a previous bin exists
            unsup_prev = has_prev_bin and not sup_prev
            # two-sided: isolated on both available sides (needs both bins to exist)
            unsup_both = (
                has_prev_bin and has_next_bin and not sup_prev and not sup_next
            )
            if unsup_prev:
                n_one_sided += 1
            if unsup_both:
                n_two_sided += 1

            pt = per_time.setdefault(int(t), {"n": 0, "n_one_sided": 0, "n_two_sided": 0})
            pt["n"] += 1
            pt["n_one_sided"] += int(unsup_prev)
            pt["n_two_sided"] += int(unsup_both)

            if batch is not None:
                b = str(batch[i])
                pb = per_batch.setdefault(b, {"n": 0, "n_one_sided": 0, "n_two_sided": 0})
                pb["n"] += 1
                pb["n_one_sided"] += int(unsup_prev)
                pb["n_two_sided"] += int(unsup_both)

    return {
        "attraction_bandwidth": float(h),
        "n_observations": n_obs,
        "frac_one_sided": n_one_sided / max(n_obs, 1),
        "frac_two_sided": n_two_sided / max(n_obs, 1),
        "n_one_sided": n_one_sided,
        "n_two_sided": n_two_sided,
        "per_time": per_time,
        "per_batch": per_batch,
    }


def _any_neighbor_within(
    positions: np.ndarray, mask: np.ndarray, i: int, t: int, t_ref: int, h: float
) -> bool:
    """True if point (i,t) has ANY observed neighbor within h at bin t_ref."""
    T = positions.shape[1]
    if t_ref < 0 or t_ref >= T:
        return False
    ref = np.flatnonzero(mask[:, t_ref])
    if ref.size == 0:
        return False
    d = np.linalg.norm(positions[ref, t_ref, :] - positions[i, t, :], axis=-1)
    return bool(d.min() <= h)


# --------------------------------------------------------------------------- #
# Planning: rigid slide-to-h moves
# --------------------------------------------------------------------------- #
def plan_seam_bridge(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    records: list[UnsupportedRecord],
    config: SeamBridgeConfig,
    h: float,
) -> tuple[list[SeamMove], list[str]]:
    """Plan rigid slide-to-``h`` moves for eligible (seam) observations.

    The planned displacement is the SAME object step 1 (init) applies and step 2
    (force) would consume. Planning never mutates ``x0``.
    """
    moves: list[SeamMove] = []
    warnings: list[str] = []

    eligible = [r for r in records if r.eligible]
    if not eligible:
        return moves, warnings

    # Group eligible records into rigid units keyed by (time, batch[, label]).
    units: dict[tuple, list[UnsupportedRecord]] = {}
    for r in eligible:
        if config.rigid_unit == "per_point":
            key = (r.time_index, r.batch, r.label, r.embryo_row)
        elif config.rigid_unit == "batch":
            key = (r.time_index, r.batch, "")
        else:  # batch_genotype
            key = (r.time_index, r.batch, r.label)
        units.setdefault(key, []).append(r)

    for key, unit_records in units.items():
        t = key[0]
        batch_id = key[1]
        label_id = key[2] if len(key) > 2 else ""
        rows = [r.embryo_row for r in unit_records]

        pairs = _match_predecessors(
            x0, mask, labels, unit_records, t, h, config.match_within_label
        )
        n_pairs = len(pairs)
        if n_pairs < config.min_matched_pairs:
            moves.append(
                SeamMove(
                    time_index=t, batch=batch_id, label=label_id, embryo_rows=rows,
                    n_matched_pairs=n_pairs, displacement=(0.0, 0.0),
                    raw_displacement=(0.0, 0.0), initial_gap=np.inf, final_gap=np.inf,
                    slide_len=0.0, applied=False,
                    skipped_reason=f"only {n_pairs} matched pairs (< {config.min_matched_pairs})",
                )
            )
            continue

        # Robust full displacement = median of (predecessor - point) match vectors.
        match_vecs = np.array([pv - cv for cv, pv in pairs])  # (n_pairs, 2)
        raw_disp = np.median(match_vecs, axis=0)

        # Clip to slide-to-h: move the group along raw_disp only until its centroid
        # reaches within h of the predecessor manifold. The gap is measured as the
        # centroid-to-nearest-predecessor distance along the displacement direction.
        cur_pos = x0[rows, t, :]
        initial_gap = float(np.linalg.norm(raw_disp))
        if initial_gap <= h:
            # Already within bandwidth after matching accounts for it: no move needed.
            disp = np.zeros(2)
            final_gap = initial_gap
        else:
            scale = (initial_gap - h) / initial_gap  # slide until h remains
            disp = raw_disp * scale
            final_gap = h

        slide_len = float(np.linalg.norm(disp))
        if slide_len > config.max_reasonable_slide_mult * h:
            warnings.append(
                f"[t={t} batch={batch_id} label={label_id}] required slide "
                f"{slide_len:.3g} exceeds {config.max_reasonable_slide_mult}x h={h:.3g}; "
                f"discontinuity may not be a simple technical offset — review before trusting."
            )

        moves.append(
            SeamMove(
                time_index=t, batch=batch_id, label=label_id, embryo_rows=rows,
                n_matched_pairs=n_pairs,
                displacement=(float(disp[0]), float(disp[1])),
                raw_displacement=(float(raw_disp[0]), float(raw_disp[1])),
                initial_gap=initial_gap, final_gap=final_gap, slide_len=slide_len,
                applied=False,  # set True only when actually applied in mode="init"
            )
        )
    return moves, warnings


def _match_predecessors(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray | None,
    unit_records: list[UnsupportedRecord],
    t: int,
    h: float,
    match_within_label: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (current_pos, predecessor_pos) pairs, nearest same-group at t-1.

    Mutual-NN is not required here because we only use these matches to estimate a
    single robust median displacement; outlier matches are down-weighted by the
    median rather than by hard mutual-NN rejection.
    """
    t_prev = t - 1
    if t_prev < 0:
        return []
    prev_obs = np.flatnonzero(mask[:, t_prev])
    if prev_obs.size == 0:
        return []

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for r in unit_records:
        i = r.embryo_row
        cand = prev_obs
        if match_within_label and labels is not None:
            cand = cand[labels[cand] == labels[i]]
        if cand.size == 0:
            continue
        cur = x0[i, t, :]
        d = np.linalg.norm(x0[cand, t_prev, :] - cur, axis=-1)
        j = cand[int(np.argmin(d))]
        pairs.append((cur, x0[j, t_prev, :]))
    return pairs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def bridge_seams(
    x0: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    config: SeamBridgeConfig | None = None,
) -> tuple[np.ndarray, SeamBridgeReport]:
    """Detect and (optionally) repair batch-induced initializer seams.

    Parameters
    ----------
    x0 : (N_e, T, 2)
        Initial positions from ``aligned_umap_init`` (NaN where unobserved).
    mask : (N_e, T) bool
    labels : (N_e,) str
        Per-embryo genotype/group label (constant across time).
    batch : (N_e,) str
        Per-embryo experiment/batch id (constant across time). Used to group
        rigid units and to attribute seams to the incoming batch.
    config : SeamBridgeConfig, optional
        Defaults to an inert config (``mode="off"``): detect + report, move
        nothing.

    Returns
    -------
    x0_out : (N_e, T, 2)
        A copy of ``x0``. Repaired in place only when ``mode="init"``; identical
        to the input for ``mode="off"`` and ``mode="force"``.
    report : SeamBridgeReport
        Detection records, planned/applied moves, and warnings. Always populated,
        regardless of mode — so ``mode="off"`` is a true diagnostic control.
    """
    config = config or SeamBridgeConfig()
    x0_out = np.array(x0, copy=True)

    h = config.attraction_bandwidth
    if h is None:
        h = estimate_attraction_bandwidth(
            x0, mask, labels,
            quantile=config.bandwidth_quantile,
            match_within_label=config.match_within_label,
        )

    n_obs = int(mask.sum())
    report = SeamBridgeReport(mode=config.mode, attraction_bandwidth=float(h), n_observations=n_obs)

    report.unsupported = detect_predecessor_support(
        x0, mask, labels, batch,
        attraction_bandwidth=h,
        match_within_label=config.match_within_label,
    )

    # Warn on large unsupported fractions (report-only; never escalates the fix).
    frac = report.n_unsupported_prev / max(n_obs, 1)
    if frac > 0.25:
        report.warnings.append(
            f"{frac:.0%} of observations lack predecessor support "
            f"({report.n_unsupported_prev}/{n_obs}); large islands may indicate a "
            f"batch-induced discontinuity or a genuinely emerging biological state."
        )
    if report.n_isolated > 0:
        report.warnings.append(
            f"{report.n_isolated} observation(s) unsupported at BOTH t-1 and t+1 "
            f"(isolated on both sides); left in place as candidate novel/outlier "
            f"states and never moved."
        )

    moves, plan_warnings = plan_seam_bridge(
        x0, mask, labels, batch, report.unsupported, config, h
    )
    report.warnings.extend(plan_warnings)

    if config.mode == "init":
        for m in moves:
            if m.skipped_reason is not None or m.slide_len == 0.0:
                continue
            disp = np.array(m.displacement)
            for row in m.embryo_rows:
                x0_out[row, m.time_index, :] += disp
            m.applied = True

    report.moves = moves
    return x0_out, report
