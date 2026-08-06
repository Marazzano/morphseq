"""Figures for the supervised crispant-vs-control morphology contrasts.

Visual grammar, held constant across every panel in this module:

    steel blue   matched control embryos
    crimson      crispant embryos
    dashed grey  the separating hyperplane (s = 0)
    dotted grey  a null/reference level (chance AUC, random-direction angle, ...)

Every statistic here has a null that is not zero -- chance AUC is 0.5, and two arbitrary directions
in 5D sit a median of ~70 degrees apart, not 90. Each figure draws its own reference line so the
numbers are never read against an implicit and wrong baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec

from gene7_config import REFERENCE_TEMPERATURE, TEMPERATURE_SCALE, temperature_limits

CONTROL_COLOUR = "#3B6EA5"
CRISPANT_COLOUR = "#C0392B"
NULL_COLOUR = "0.55"

FONT_FAMILY = "DejaVu Sans"


def use_house_style() -> None:
    """Apply the shared look. Called by the runner and by the notebook's first cell."""
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.3",
            "axes.linewidth": 0.9,
            "xtick.color": "0.3",
            "ytick.color": "0.3",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
            "legend.frameon": False,
        }
    )


def _condition_label(temperature: float, timepoint: float) -> str:
    return f"{int(temperature)}C / {int(timepoint)}hpf"


def _stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.01:
        return "**"
    if q < 0.1:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def auc_heatmap(summary: pd.DataFrame, *, figsize=(11.5, 3.6)):
    """LOO-AUC for every contrast, targets down and conditions across.

    Colour is centred on 0.5 (chance) rather than on the data range, so a panel that looks white is
    a contrast with no generalising morphological difference -- which is the thing worth seeing at a
    glance. Stars mark BH-adjusted permutation significance.
    """
    targets = sorted(summary["target"].unique())
    conditions = sorted(
        summary[["temperature", "timepoint"]].drop_duplicates().itertuples(index=False),
        key=lambda row: (row.temperature, row.timepoint),
    )
    grid = np.full((len(targets), len(conditions)), np.nan)
    annotations = np.empty_like(grid, dtype=object)
    annotations[:] = ""

    lookup = summary.set_index(["target", "temperature", "timepoint"])
    for i, target in enumerate(targets):
        for j, condition in enumerate(conditions):
            key = (target, condition.temperature, condition.timepoint)
            if key not in lookup.index:
                continue
            row = lookup.loc[key]
            grid[i, j] = row["loo_auc"]
            annotations[i, j] = f"{row['loo_auc']:.2f}{_stars(row['q_auc'])}"

    figure, axis = plt.subplots(figsize=figsize)
    limit = float(np.nanmax(np.abs(grid - 0.5))) or 0.1
    image = axis.imshow(
        grid, cmap="RdBu_r", vmin=0.5 - limit, vmax=0.5 + limit, aspect="auto"
    )
    axis.set_xticks(range(len(conditions)))
    axis.set_xticklabels(
        [_condition_label(c.temperature, c.timepoint) for c in conditions],
        rotation=45, ha="right",
    )
    axis.set_yticks(range(len(targets)))
    axis.set_yticklabels(targets)
    for i in range(len(targets)):
        for j in range(len(conditions)):
            if annotations[i, j]:
                axis.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8.5,
                          color="black")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(np.arange(-0.5, len(conditions), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)

    bar = figure.colorbar(image, ax=axis, fraction=0.02, pad=0.015)
    bar.set_label("leave-one-out AUC\n(0.5 = chance)")
    axis.set_title(
        "Does a crispant-vs-control morphology axis generalise?   * q<0.1   ** q<0.01", pad=12
    )
    return figure


def score_panels(scores: pd.DataFrame, summary: pd.DataFrame, *, target: str,
                 figsize=(11, 8.5)):
    """One target's signed-distance distributions across all twelve conditions.

    The panel that answers the dilution question directly: if controls spread as widely along ``s``
    as crispants do, the continuous score is carrying variation the binary label would have
    suppressed, and whether that variation is signal or noise becomes an empirical matter rather
    than an assumption.
    """
    block = scores.loc[scores["contrast_target"] == target]
    temperatures = sorted(block["temperature"].unique())
    timepoints = sorted(block["timepoint"].unique())

    figure, axes = plt.subplots(
        len(temperatures), len(timepoints), figsize=figsize, sharex=False
    )
    axes = np.atleast_2d(axes)
    lookup = summary.set_index(["target", "temperature", "timepoint"])
    rng = np.random.RandomState(0)

    for i, temperature in enumerate(temperatures):
        for j, timepoint in enumerate(timepoints):
            axis = axes[i, j]
            panel = block.loc[
                (block["temperature"] == temperature) & (block["timepoint"] == timepoint)
            ]
            if panel.empty:
                axis.axis("off")
                continue

            for label, colour, offset in (
                (0, CONTROL_COLOUR, 0.0), (1, CRISPANT_COLOUR, 1.0)
            ):
                values = panel.loc[panel["is_crispant"] == label, "s"].to_numpy()
                jitter = offset + rng.uniform(-0.13, 0.13, size=len(values))
                axis.scatter(values, jitter, s=26, color=colour, alpha=0.85,
                             edgecolor="white", linewidth=0.6, zorder=3)
                axis.plot([values.mean()] * 2, [offset - 0.24, offset + 0.24],
                          color=colour, linewidth=2.2, zorder=4)

            axis.axvline(0, color=NULL_COLOUR, linestyle="--", linewidth=1, zorder=1)
            axis.set_yticks([0, 1])
            axis.set_yticklabels(["ctrl", "crispant"] if j == 0 else ["", ""], fontsize=8.5)
            axis.set_ylim(-0.5, 1.5)

            key = (target, temperature, timepoint)
            if key in lookup.index:
                row = lookup.loc[key]
                axis.set_title(
                    f"{_condition_label(temperature, timepoint)}   "
                    f"AUC {row['loo_auc']:.2f}{_stars(row['q_auc'])}",
                    fontsize=9.5,
                )
            if i == len(temperatures) - 1:
                axis.set_xlabel("signed distance $s$", fontsize=9)

    figure.suptitle(
        f"{target} vs matched control — position along the supervised morphology axis",
        fontsize=12.5, y=0.995,
    )
    figure.tight_layout()
    return figure


def dispersion_scatter(scores: pd.DataFrame, summary: pd.DataFrame, *, figsize=(6.2, 5.8)):
    """Within-group spread along ``s``: crispants against their matched controls.

    A direct test of the mosaic-F0 premise. If injected clutches are a mixture of effectively-null
    and escaper embryos, they should be *more* dispersed than controls along the axis that separates
    them, putting points above the diagonal. Points on or below the line say the crispant group is
    as homogeneous as the control group, which would undercut the argument for a graded score.
    """
    rows = []
    for contrast, block in scores.groupby("contrast"):
        control = block.loc[block["is_crispant"] == 0, "s"]
        crispant = block.loc[block["is_crispant"] == 1, "s"]
        rows.append(
            {
                "contrast": contrast,
                "target": block["contrast_target"].iloc[0],
                "temperature": block["temperature"].iloc[0],
                "sd_control": control.std(ddof=1),
                "sd_crispant": crispant.std(ddof=1),
            }
        )
    frame = pd.DataFrame(rows).merge(
        summary[["contrast", "loo_auc", "q_auc", "log_sd_ratio", "perm_p_dispersion",
                 "q_dispersion"]],
        on="contrast", how="left",
    )

    figure, axis = plt.subplots(figsize=figsize)
    markers = {t: m for t, m in zip(sorted(frame["target"].unique()), ("o", "s", "D", "^"))}
    vmin, vmax = temperature_limits(frame["temperature"])
    for target, block in frame.groupby("target"):
        axis.scatter(
            block["sd_control"], block["sd_crispant"],
            c=block["temperature"], cmap=TEMPERATURE_SCALE, vmin=vmin, vmax=vmax,
            s=78, marker=markers[target], edgecolor="0.25", linewidth=0.7, zorder=3,
        )

    limit = float(np.nanmax([frame["sd_control"].max(), frame["sd_crispant"].max()])) * 1.08
    axis.plot([0, limit], [0, limit], color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axis.text(limit * 0.97, limit * 0.90, "equal spread", color=NULL_COLOUR,
              ha="right", fontsize=9)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("SD of $s$ within matched controls")
    axis.set_ylabel("SD of $s$ within crispants")

    # The figure's claim is "points sit above the line", so the readout states that tally directly:
    # how many contrasts land above it, the typical fold difference, and how many clear their OWN
    # refit-label permutation null. No pooled test is reported. A Wilcoxon over the log ratios used
    # to sit here and was deleted: it treats 36 contrasts as 36 independent draws when they are four
    # targets crossed with the same handful of conditions, sharing control wells, so its p-value
    # counts the same clutch several times and is far smaller than the design supports. The
    # permutation null is the honest per-contrast test, and consistency is the honest aggregate one.
    axis.set_title(
        "Are crispant clutches more heterogeneous than controls?\n"
        "(the mosaic-F0 premise, stated as a picture)", fontsize=11,
    )

    handles = [
        mpl.lines.Line2D([], [], marker=markers[t], linestyle="none", color="0.45",
                         markeredgecolor="0.25", markersize=8, label=t)
        for t in sorted(frame["target"].unique())
    ]
    axis.legend(handles=handles, title="target", loc="lower right", fontsize=9)

    mappable = plt.cm.ScalarMappable(cmap=TEMPERATURE_SCALE, norm=plt.Normalize(vmin, vmax))
    bar = figure.colorbar(mappable, ax=axis, fraction=0.04, pad=0.02)
    bar.set_label(f"temperature (C)\n(white = {REFERENCE_TEMPERATURE:.0f}C)")
    bar.ax.axhline(REFERENCE_TEMPERATURE, color="0.35", linewidth=0.9)
    figure.tight_layout()
    return figure, frame


def stability_plot(axes_list, summary: pd.DataFrame, *, figsize=(7.5, 9.5)):
    """Bootstrap angle spread per contrast, against the random-direction reference.

    Separation and stability are different properties: a contrast can have significantly different
    centroids and still recover a near-arbitrary direction when the embryos are resampled. The grey
    band is where an arbitrary 5D direction would land, so anything overlapping it is undetermined
    regardless of its p-value.
    """
    order = np.argsort([np.nanmedian(a.boot_angles) for a in axes_list])
    ordered = [axes_list[i] for i in order]
    lookup = summary.set_index("contrast")

    figure, axis = plt.subplots(figsize=figsize)
    random_p05 = float(summary["random_angle_p05"].iloc[0])
    random_median = float(summary["random_angle_median"].iloc[0])
    axis.axvspan(random_p05, 90, color=NULL_COLOUR, alpha=0.16, zorder=0,
                 label="random 5D direction (5th pctile - 90)")
    axis.axvline(random_median, color=NULL_COLOUR, linestyle=":", linewidth=1.4, zorder=1,
                 label=f"random median ({random_median:.0f}°)")

    for position, contrast_axis in enumerate(ordered):
        angles = contrast_axis.boot_angles
        low, mid, high = np.nanpercentile(angles, [5, 50, 95])
        usable = bool(lookup.loc[contrast_axis.label, "usable"]) \
            if contrast_axis.label in lookup.index else False
        colour = CRISPANT_COLOUR if usable else "0.45"
        axis.plot([low, high], [position, position], color=colour, linewidth=2, alpha=0.75,
                  zorder=2)
        axis.scatter([mid], [position], s=34, color=colour, zorder=3,
                     edgecolor="white", linewidth=0.6)

    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([a.label for a in ordered], fontsize=7.6)
    axis.set_xlabel("angle between bootstrap direction and full-data direction (degrees)")
    axis.set_xlim(0, 90)
    axis.set_ylim(-1, len(ordered))
    axis.invert_yaxis()
    axis.set_title("How well determined is each discriminant direction?\n"
                   "red = passes the usability gate", fontsize=11)
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    return figure


def rank_stability_plot(axes_list, summary: pd.DataFrame, *, figsize=(7.5, 9.5)):
    """Ordinal stability per contrast: does the embryo *ordering* survive resampling?

    The counterpart to :func:`stability_plot`, and the one with downstream consequences -- a
    regression on ``s`` cares where embryos land relative to one another, not where the normal
    vector points.

    Each contrast carries its own floor (open marker), because the floor is a property of that
    contrast's embryo cloud: the 5D subspace is anisotropic (G00 alone carries ~49% of its
    variance), so two arbitrary directions already order these embryos at rho ~ 0.47 with no
    information whatsoever. Significance is the fraction of random pairs reaching the observed
    bootstrap median, BH-adjusted -- median against the full null, not tail against tail.
    """
    order = np.argsort([np.nanmedian(a.boot_rank_rho) for a in axes_list])
    ordered = [axes_list[i] for i in order]
    lookup = summary.set_index("contrast")

    figure, axis = plt.subplots(figsize=figsize)
    for position, contrast_axis in enumerate(ordered):
        low, mid, high = np.nanpercentile(contrast_axis.boot_rank_rho, [5, 50, 95])
        floor_low, floor_mid, floor_high = np.nanpercentile(contrast_axis.random_rank_rho,
                                                            [25, 50, 75])
        stable = bool(lookup.loc[contrast_axis.label, "rank_stable"]) \
            if contrast_axis.label in lookup.index else False
        colour = CRISPANT_COLOUR if stable else "0.45"
        axis.plot([floor_low, floor_high], [position, position], color=NULL_COLOUR,
                  linewidth=4, alpha=0.30, zorder=1)
        axis.scatter([floor_mid], [position], s=26, facecolor="none", edgecolor=NULL_COLOUR,
                     linewidth=1.2, zorder=2)
        axis.plot([low, high], [position, position], color=colour, linewidth=2, alpha=0.8, zorder=3)
        axis.scatter([mid], [position], s=34, color=colour, zorder=4,
                     edgecolor="white", linewidth=0.6)

    axis.scatter([], [], s=26, facecolor="none", edgecolor=NULL_COLOUR, linewidth=1.2,
                 label="random-direction floor (median, IQR)")
    axis.scatter([], [], s=34, color=CRISPANT_COLOUR, label="bootstrap median (q<0.1 vs floor)")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([a.label for a in ordered], fontsize=7.6)
    axis.set_xlabel(r"Spearman $\rho$ between bootstrap ordering and full-data ordering")
    axis.set_xlim(-0.05, 1.02)
    axis.set_ylim(-1, len(ordered))
    axis.invert_yaxis()
    axis.set_title("Ordinal stability: does the embryo RANKING survive resampling?\n"
                   "red = 5th pctile clears this contrast's own random floor", fontsize=11)
    axis.legend(loc="lower left", fontsize=8.5)
    figure.tight_layout()
    return figure


def angle_versus_rank(summary: pd.DataFrame, *, figsize=(11.5, 4.8)):
    """Does angular instability actually predict consequential instability?

    Left: the two stability measures against each other. If they tracked tightly, the angle would
    have been a fine proxy and the ordinal analysis would be redundant. Points sitting high on the
    y-axis at large angles are contrasts whose direction wobbles a lot while their embryo ordering
    barely moves -- direction perturbation lying mostly orthogonal to where the embryos actually are.

    Right: each contrast's ordinal stability against its own random-direction floor. Points near the
    diagonal are contrasts whose apparent reproducibility is just the cloud's anisotropy.
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    markers = {t: m for t, m in zip(sorted(summary["target"].unique()), ("o", "s", "D", "^"))}

    for target, block in summary.groupby("target"):
        axes[0].scatter(block["boot_angle_p95"], block["rank_rho_median"],
                        c=block["loo_auc"], cmap="viridis", vmin=0, vmax=1,
                        s=76, marker=markers[target], edgecolor="0.25", linewidth=0.7, zorder=3)
        axes[1].scatter(block["random_rank_rho_median"], block["rank_rho_median"],
                        c=block["loo_auc"], cmap="viridis", vmin=0, vmax=1,
                        s=76, marker=markers[target], edgecolor="0.25", linewidth=0.7, zorder=3)

    from scipy import stats as _stats
    valid = summary[["boot_angle_p95", "rank_rho_median"]].dropna()
    rho = _stats.spearmanr(valid["boot_angle_p95"], valid["rank_rho_median"]).statistic
    axes[0].set_xlabel("angular instability — bootstrap 95th pctile angle (degrees)")
    axes[0].set_ylabel(r"ordinal stability — median bootstrap $\rho$")
    axes[0].set_title(f"Do the two measures agree?   Spearman = {rho:+.2f}", fontsize=10.5)
    axes[0].axvline(float(summary["random_angle_median"].iloc[0]), color=NULL_COLOUR,
                    linestyle=":", linewidth=1.3)

    limit = 1.02
    axes[1].plot([0, limit], [0, limit], color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axes[1].set_xlim(0, limit)
    axes[1].set_ylim(0, limit)
    axes[1].set_xlabel(r"random-direction floor — median $\rho$")
    axes[1].set_ylabel(r"observed — median bootstrap $\rho$")
    axes[1].set_title("Above the line = ordering is more reproducible\nthan the cloud alone explains",
                      fontsize=10.5)

    handles = [
        mpl.lines.Line2D([], [], marker=markers[t], linestyle="none", color="0.45",
                         markeredgecolor="0.25", markersize=8, label=t)
        for t in sorted(summary["target"].unique())
    ]
    axes[0].legend(handles=handles, title="target", fontsize=8.5, loc="lower left")
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    figure.colorbar(mappable, ax=axes, fraction=0.02, pad=0.015).set_label("LOO-AUC")
    return figure


def rank_churn_figure(axis_obj, *, figsize=(7.2, 6.4), n_show: int = 120):
    """One contrast's bootstrap orderings drawn as a bump chart.

    The most direct picture of "consequential variability": each line is one embryo, tracked from
    its full-data rank to its rank under a bootstrap refit. Flat lines mean the ordering is stable;
    heavy crossing means the score would reshuffle the regression's predictor from one resample to
    the next. Controls blue, crispants red -- so crossing *between* colours is the costly kind.
    """
    from scipy import stats

    scores = axis_obj.scores.sort_values("s").reset_index(drop=True)
    features_order = np.argsort(np.argsort(axis_obj.scores["s"].to_numpy()))
    n = len(scores)

    figure, axis = plt.subplots(figsize=figsize)
    rng = np.random.RandomState(0)
    replicate_ranks = axis_obj.metadata.get("bootstrap_ranks")
    if replicate_ranks is None:
        raise ValueError("axis has no cached bootstrap ranks; refit with store_ranks=True")

    picks = rng.choice(len(replicate_ranks), size=min(n_show, len(replicate_ranks)), replace=False)
    for index in picks:
        ranks = replicate_ranks[index]
        for embryo in range(n):
            colour = (CRISPANT_COLOUR if axis_obj.scores["is_crispant"].iloc[embryo]
                      else CONTROL_COLOUR)
            axis.plot([0, 1], [features_order[embryo], ranks[embryo]],
                      color=colour, alpha=0.045, linewidth=1.0, zorder=2)

    for embryo in range(n):
        colour = (CRISPANT_COLOUR if axis_obj.scores["is_crispant"].iloc[embryo]
                  else CONTROL_COLOUR)
        axis.scatter([0], [features_order[embryo]], s=30, color=colour, zorder=4)

    median_rho = float(np.nanmedian(axis_obj.boot_rank_rho))
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["full-data\nrank", "bootstrap\nrank"])
    axis.set_ylabel("position along $s$ (0 = most control-like)")
    axis.set_title(f"{axis_obj.label}\nrank churn under resampling   "
                   rf"median $\rho$ = {median_rho:.2f}", fontsize=11)
    figure.tight_layout()
    return figure


def confound_scatter(summary: pd.DataFrame, *, figsize=(11, 4.6)):
    """Is a working discriminant just a developmental-delay axis, or the dominant cohort PC?

    Left: angle to the control-derived stage direction. A contrast near 0 degrees has found staging.
    Right: angle to the crispant cohort's own unsupervised PC1. Near 0 means the supervised axis
    recovers what the intra-cohort PCA already found; near the random median means the perturbation
    direction is a *minor* axis of within-cohort variation, which is itself informative.
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    markers = {t: m for t, m in zip(sorted(summary["target"].unique()), ("o", "s", "D", "^"))}
    random_median = float(summary["random_angle_median"].iloc[0])
    vmin, vmax = temperature_limits(summary["temperature"])

    for axis, column, title in (
        (axes[0], "angle_to_stage", "angle to control stage axis"),
        (axes[1], "angle_to_cohort_pc1", "angle to crispant cohort PC1"),
    ):
        for target, block in summary.groupby("target"):
            axis.scatter(
                block[column], block["loo_auc"],
                c=block["temperature"], cmap=TEMPERATURE_SCALE, vmin=vmin, vmax=vmax,
                s=76, marker=markers[target], edgecolor="0.25", linewidth=0.7, zorder=3,
            )
        axis.axhline(0.5, color=NULL_COLOUR, linestyle="--", linewidth=1.1, zorder=1)
        axis.axvline(random_median, color=NULL_COLOUR, linestyle=":", linewidth=1.3, zorder=1)
        axis.set_xlabel(f"{title} (degrees)")
        axis.set_xlim(0, 90)
        axis.text(random_median + 1.5, 0.02, "random", rotation=90,
                  fontsize=8, color=NULL_COLOUR, va="bottom")

    # Left panel only: the zone where a working discriminant is plausibly just a delay score.
    axes[0].axvspan(0, 35, color=CRISPANT_COLOUR, alpha=0.07, zorder=0)
    axes[0].text(1.5, 0.02, "stage-confound zone", fontsize=8, color=CRISPANT_COLOUR,
                 va="bottom", rotation=90)

    axes[0].set_ylabel("leave-one-out AUC")
    handles = [
        mpl.lines.Line2D([], [], marker=markers[t], linestyle="none", color="0.45",
                         markeredgecolor="0.25", markersize=8, label=t)
        for t in sorted(summary["target"].unique())
    ]
    axes[0].legend(handles=handles, title="target", fontsize=9, loc="lower left")
    mappable = plt.cm.ScalarMappable(cmap=TEMPERATURE_SCALE, norm=plt.Normalize(vmin, vmax))
    bar = figure.colorbar(mappable, ax=axes, fraction=0.02, pad=0.015)
    bar.set_label(f"temperature (C)\n(white = {REFERENCE_TEMPERATURE:.0f}C)")
    bar.ax.axhline(REFERENCE_TEMPERATURE, color="0.35", linewidth=0.9)
    figure.suptitle("Interpretation checks: what else is the discriminant parallel to?",
                    fontsize=12, y=1.02)
    return figure


def direction_heatmap(directions: pd.DataFrame, summary: pd.DataFrame, *, figsize=(6.4, 9.5)):
    """Loading of every discriminant on the five shared dimensions.

    Rows grouped by target: if a gene has a consistent morphological signature, its rows should
    share a pattern across temperatures and timepoints.
    """
    wide = directions.pivot_table(
        index=["target", "temperature", "timepoint"], columns="shared_dim", values="loading"
    ).sort_index()
    usable = summary.set_index(["target", "temperature", "timepoint"])["usable"]

    figure, axis = plt.subplots(figsize=figsize)
    limit = float(np.nanmax(np.abs(wide.to_numpy())))
    image = axis.imshow(wide.to_numpy(), cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(wide.shape[1]))
    axis.set_xticklabels(wide.columns)
    axis.set_yticks(range(len(wide)))
    axis.set_yticklabels(
        [
            f"{'* ' if usable.get(index, False) else '  '}"
            f"{index[0]} | {int(index[1])}C | {int(index[2])}hpf"
            for index in wide.index
        ],
        fontsize=7.6,
    )
    for boundary in np.flatnonzero(
        wide.index.get_level_values(0)[1:] != wide.index.get_level_values(0)[:-1]
    ):
        axis.axhline(boundary + 0.5, color="black", linewidth=1.4)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02).set_label("loading")
    axis.set_title("Discriminant directions in the shared subspace\n(* = usable contrast)",
                   fontsize=11)
    figure.tight_layout()
    return figure


def similarity_heatmap(similarity: pd.DataFrame, *, figsize=(10.5, 9)):
    """Pairwise ``|cos|`` between discriminant directions, ordered by target.

    The colour floor is the *random-direction* expectation rather than 0, because in 5D two unrelated
    unit vectors already share ``|cos| = 0.375``. Anything at or below that floor is indistinguishable
    from an arbitrary pair. Blocks along the diagonal would mean a target keeps its signature across
    conditions.
    """
    import lda_contrasts as lc

    figure, axis = plt.subplots(figsize=figsize)
    floor = float(lc.random_cosine_reference().mean())
    image = axis.imshow(similarity.to_numpy(), cmap="magma", vmin=floor, vmax=1.0)
    axis.set_xticks(range(len(similarity)))
    axis.set_xticklabels(similarity.columns, rotation=90, fontsize=6.2)
    axis.set_yticks(range(len(similarity)))
    axis.set_yticklabels(similarity.index, fontsize=6.2)
    bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    bar.set_label(f"|cos| between directions (floor = random 5D expectation, {floor:.3f})")
    axis.set_title("Do contrasts share a morphological direction?", fontsize=11)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Image strips
# ---------------------------------------------------------------------------


def contrast_strip_figure(strip: pd.DataFrame, *, title: str, per_row: int = 12,
                          tile_px: int = 118, cmap: str = "gray"):
    """Every embryo in a contrast, ordered along ``s``, wrapped over rows.

    Border colour carries the class, so where the controls actually fall in the severity ordering is
    visible directly -- controls interleaved among the crispants means the axis is not separating
    them, whatever the AUC says. The whole contrast is shown rather than an even sample, because at
    n~23 the interleaving pattern is the information.
    """
    from matplotlib import image as mpimg

    present = strip.loc[strip["image_exists"]].reset_index(drop=True)
    if present.empty:
        raise ValueError(f"no resolvable images for {title!r}")

    n_rows = int(np.ceil(len(present) / per_row))
    figure = plt.figure(figsize=(per_row * tile_px / 100 * 1.05, n_rows * tile_px / 100 * 1.34))
    grid = gridspec.GridSpec(n_rows, per_row, figure=figure, wspace=0.06, hspace=0.30)

    for position, row in present.iterrows():
        axis = figure.add_subplot(grid[position // per_row, position % per_row])
        axis.imshow(mpimg.imread(row["image_path"]), cmap=cmap)
        axis.set_xticks([])
        axis.set_yticks([])
        colour = CRISPANT_COLOUR if row["is_crispant"] else CONTROL_COLOUR
        for spine in axis.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.4)
        axis.set_title(f"{row['s']:+.2f}", fontsize=7.6, color=colour, pad=2.5)

    figure.suptitle(
        f"{title}\nordered by signed distance $s$   "
        f"(blue = control, red = crispant)",
        fontsize=11.5, y=1.0 + 0.055 / n_rows,
    )
    return figure




def strip_crop_box(paths, *, threshold: float = 0.15, pad: int = 8):
    """Union bounding box of the embryos across a set of snips, as (top, bottom, left, right).

    Shared across every tile that will be drawn, deliberately. Cropping each embryo to its own
    extent would rescale them independently and destroy exactly the size and elongation differences
    the strip exists to show -- and it would make two strips drawn separately (one per class)
    incomparable.
    """
    from matplotlib import image as mpimg

    top, bottom, left, right, shape = np.inf, -np.inf, np.inf, -np.inf, None
    for path in paths:
        frame = mpimg.imread(path)
        frame = frame if frame.ndim == 2 else frame[..., :3].mean(-1)
        shape = frame.shape
        mask = frame > threshold
        rows, columns = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        if not len(rows) or not len(columns):
            continue
        top, bottom = min(top, rows.min()), max(bottom, rows.max())
        left, right = min(left, columns.min()), max(right, columns.max())
    height, width = shape
    return (max(int(top) - pad, 0), min(int(bottom) + pad, height - 1),
            max(int(left) - pad, 0), min(int(right) + pad, width - 1))


def contrast_strip_tight(strip: pd.DataFrame, *, title: str, per_row: int = 12,
                         crop_box=None, threshold: float = 0.15, pad: int = 8,
                         tile_in: float = 1.05, cmap: str = "gray",
                         score_format: str = "+.3f", subtitle: str | None = None):
    """``contrast_strip_figure`` with the dead space cropped out.

    The raw snips are 576 x 256 with the embryo occupying about 19% of the frame, so tiling them
    whole spends most of the figure on background. Cropping first buys roughly a 5x gain in embryo
    area at the same figure size.

    Pass ``crop_box`` from :func:`strip_crop_box` computed over the WHOLE contrast when drawing one
    class at a time, so the per-class strips stay on a common scale and can be read against each
    other. Left to ``None`` the box is computed from whatever rows are passed in.

    Scores are printed to three decimals by default. Two decimals is not enough: this contrast has
    a control at s = -0.2678 and a crispant at s = -0.2695, which both render as "-0.27" and look
    like a duplicate or a mislabelling when they are neither.
    """
    from matplotlib import image as mpimg

    present = strip.loc[strip["image_exists"]].reset_index(drop=True)
    if present.empty:
        raise ValueError(f"no resolvable images for {title!r}")
    if crop_box is None:
        crop_box = strip_crop_box(present["image_path"], threshold=threshold, pad=pad)
    top, bottom, left, right = crop_box

    cropped = []
    for path in present["image_path"]:
        frame = mpimg.imread(path)
        frame = frame if frame.ndim == 2 else frame[..., :3].mean(-1)
        cropped.append(frame[top:bottom + 1, left:right + 1])
    aspect = (bottom - top + 1) / (right - left + 1)

    n_rows = int(np.ceil(len(present) / per_row))
    figure = plt.figure(figsize=(per_row * tile_in, n_rows * tile_in * aspect * 1.16))
    grid = gridspec.GridSpec(n_rows, per_row, figure=figure, wspace=0.02, hspace=0.16)

    for position, row in present.iterrows():
        axis = figure.add_subplot(grid[position // per_row, position % per_row])
        axis.imshow(cropped[position], cmap=cmap, aspect="equal")
        axis.set_xticks([]); axis.set_yticks([])
        colour = CRISPANT_COLOUR if row["is_crispant"] else CONTROL_COLOUR
        for spine in axis.spines.values():
            spine.set_edgecolor(colour); spine.set_linewidth(2.2)
        axis.set_title(format(row["s"], score_format), fontsize=8.2, color=colour, pad=2.0)

    figure.suptitle(f"{title}\n{subtitle if subtitle is not None else 'ordered by signed distance $s$'}",
                    fontsize=12, y=1.0 + 0.05 / n_rows)
    return figure


def save(figure, path) -> Path:
    """Write a figure to PNG and close it."""
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target
def save_inline(figure, path, *, dpi: int = 200, bbox="tight") -> Path:
    """Write a PNG WITHOUT closing the figure, so it still renders in a notebook cell.

    ``save`` closes the figure, which suppresses the inline display -- correct for a headless
    driver script, wrong inside a notebook. Use this one there.

    Pass ``bbox=None`` when two figures are meant to be overlayable. The default "tight" crops each
    one to its own drawn content, so a panel carrying a colourbar and an otherwise identical panel
    without one come out at different scales even when their axes rectangles match exactly.
    """
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    # savefig(bbox_inches=None) does NOT mean "no cropping" -- print_figure falls back to
    # rcParams["savefig.bbox"], which the house style sets to "tight". Override the rcParam too,
    # otherwise bbox=None silently still crops and same-figsize panels come out different widths.
    with mpl.rc_context({"savefig.bbox": bbox}):
        figure.savefig(target, dpi=dpi, bbox_inches=bbox)
    return target

