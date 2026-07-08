"""Reference-relative resolved-peak readout for the valley visualization.

This is the real-data front end for the resolved-peak metrics engine (see
PEAK_METRICS_MVP.md): given a target genotype's points and its matched WT
reference points (already normalized onto the same canonical frame as the
`build_distribution_overlay` grid), it runs one pooled-label permutation
comparison per stage and renders a compact "target vs WT" readout -- an up/down
arrow, a plain-English meaning, and significance shading per metric.

It deliberately does NOT re-implement the metrics or the null test. It calls the
same `run_resolved_peak_permutation_comparison` used by the smoke test and the
SGE array path, so the arrows in the figure and the null-test tables agree by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphseq_investigation.core.resolved_peak_analysis import (
    DEFAULT_ANALYSIS_SPEC,
    EmpiricalNullSpec,
    run_resolved_peak_permutation_comparison,
)
from morphseq_investigation.core.resolved_peak_metrics import ResolvedPeakRunContext

# The five resolved-peak axes we surface in the figure, in display order, each
# with the plain-English reading of a POSITIVE observed_difference (target minus
# reference). `up_word`/`down_word` are what an increase / decrease relative to
# WT means biologically, kept short enough to sit under a KDE panel.
@dataclass(frozen=True)
class MetricReadout:
    metric: str
    label: str          # short axis name shown at left of the row
    up_word: str        # meaning when target > WT
    down_word: str      # meaning when target < WT


READOUT_METRICS: tuple[MetricReadout, ...] = (
    MetricReadout("number_of_peaks", "modes",
                  "more modes", "fewer modes"),
    MetricReadout("across_peak_distance_mean", "separation",
                  "modes farther apart", "modes closer together"),
    MetricReadout("across_peak_r80_density_mean", "concentration",
                  "denser modes", "more diffuse modes"),
    MetricReadout("across_peak_radius_mean", "spread",
                  "wider modes", "tighter modes"),
    MetricReadout("assigned_support_fraction", "support",
                  "more mass in modes", "more scattered mass"),
)

SIG_P = 0.05
# Below this |difference| (relative to the metric's null spread) we call it "no
# meaningful shift" and draw a flat marker instead of an arrow, so a technically-
# nonzero but tiny/insignificant delta doesn't read as a direction.


@dataclass(frozen=True)
class MetricCell:
    metric: str
    observed_difference: float
    p_value: float
    significant: bool
    direction: int      # +1 up, -1 down, 0 flat/undefined
    meaning: str        # plain-English reading for THIS direction
    valid: bool         # test_is_valid (structural validity of the null)


def compute_reference_readout(
    *,
    target_points: np.ndarray,
    wt_points: np.ndarray,
    canonical_grid,
    n_draws: int = 200,
    seed: int = 42,
    stage_id: str = "",
    gene: str = "",
    analysis_spec=DEFAULT_ANALYSIS_SPEC,
) -> dict[str, MetricCell]:
    """One pooled-label permutation comparison (target vs WT) for one stage.

    `target_points` / `wt_points` must already be normalized onto the SAME frame
    as `canonical_grid` (i.e. the normalized points that produced the overlay
    grid), so the KDE the engine builds lines up with the density the figure
    draws.

    Returns a metric -> MetricCell map covering READOUT_METRICS. Metrics whose
    structural requirement isn't met (e.g. a separation metric on a single-peak
    distribution) come back with direction 0 and valid=False.
    """
    null_spec = EmpiricalNullSpec(
        method="pooled_label_permutation", n_draws=n_draws, alternative="two-sided",
    )
    context = ResolvedPeakRunContext(
        analysis_id="valley_visualization_reference_readout",
        scenario_id=f"{gene}_{stage_id}",
        replicate_id=stage_id,
        seed=seed,
        n=int(min(len(target_points), len(wt_points))),
        bandwidth_rule=analysis_spec.bandwidth_rule,
        bandwidth_multiplier=analysis_spec.bandwidth_multiplier,
        bandwidth_value=float("nan"),
        peak_detector_method=analysis_spec.peak_detector_method,
        canonical_grid_id=f"{gene}_{stage_id}",
        assignment_rule=analysis_spec.assignment_rule,
    )

    df = run_resolved_peak_permutation_comparison(
        reference_points=np.asarray(wt_points, dtype=float),
        target_points=np.asarray(target_points, dtype=float),
        analysis_spec=analysis_spec,
        null_spec=null_spec,
        context=context,
        rng=np.random.default_rng(seed),
        canonical_grid=canonical_grid,
        metrics=tuple(m.metric for m in READOUT_METRICS),
    )

    by_metric = {row["metric_name"]: row for _, row in df.iterrows()}
    cells: dict[str, MetricCell] = {}
    for spec in READOUT_METRICS:
        row = by_metric.get(spec.metric)
        if row is None:
            cells[spec.metric] = MetricCell(
                spec.metric, float("nan"), float("nan"), False, 0, "n/a", False)
            continue
        diff = float(row["observed_difference"])
        p = float(row["empirical_p_value"])
        valid = bool(row["test_is_valid"])
        sig = valid and np.isfinite(p) and p < SIG_P
        if not valid or not np.isfinite(diff) or diff == 0.0:
            direction = 0
            meaning = "no modes to compare" if not valid else "no shift"
        elif diff > 0:
            direction = 1
            meaning = spec.up_word
        else:
            direction = -1
            meaning = spec.down_word
        cells[spec.metric] = MetricCell(
            spec.metric, diff, p, sig, direction, meaning, valid)
    return cells


# ── rendering ────────────────────────────────────────────────────────────────
_ARROW = {1: "↑", -1: "↓", 0: "–"}   # up / down / en-dash (flat)
_SIG_COLOR = "#B2182B"      # significant -> crimson (matches valley-sig red)
_NS_COLOR = "#9AA0A6"       # not significant -> muted gray
_INVALID_COLOR = "#C9CCD1"  # structurally undefined -> faint gray


def render_readout_cell(ax, cells: dict[str, MetricCell], *, label_fs: int = 11) -> None:
    """Draw one stage's readout column: one row per READOUT_METRIC, each an arrow
    + concise meaning, colored by significance. Called once per stage column."""
    ax.axis("off")
    n = len(READOUT_METRICS)
    for i, spec in enumerate(READOUT_METRICS):
        cell = cells[spec.metric]
        # top-to-bottom
        y = 1.0 - (i + 0.5) / n
        if not cell.valid:
            color = _INVALID_COLOR
        elif cell.significant:
            color = _SIG_COLOR
        else:
            color = _NS_COLOR
        weight = "bold" if cell.significant else "normal"
        arrow = _ARROW[cell.direction]
        ax.text(0.03, y, arrow, transform=ax.transAxes, fontsize=label_fs + 3,
                fontweight="bold", color=color, ha="left", va="center")
        star = " *" if cell.significant else ""
        ax.text(0.20, y, f"{cell.meaning}{star}", transform=ax.transAxes,
                fontsize=label_fs - 2, fontweight=weight, color=color,
                ha="left", va="center")


def readout_metric_labels() -> list[str]:
    """Left-margin axis names, one per metric row, top-to-bottom."""
    return [spec.label for spec in READOUT_METRICS]
