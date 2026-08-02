"""Figures for the edgeR three-arm comparison.

The organising idea: every regression result is shown *next to the axis that produced it*. A
contrast's T_std is not interpretable without knowing whether its morphology axis separated the
groups at all, how significant that separation was, and whether the ordering was stable -- so the
embryo dot plot and the axis diagnostics travel with the regression outcome in the same panel.

Shared grammar (matching lda_plots):

    steel blue   matched control embryos
    crimson      crispant embryos
    dashed grey  the separating hyperplane (s = 0) / a null reference level
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt

CONTROL_COLOUR = "#3B6EA5"
CRISPANT_COLOUR = "#C0392B"
NULL_COLOUR = "0.55"

# Candidate summaries of "how much morphological phenotype was there to work with". They measure
# different things and are NOT interchangeable: separation and AUC ask how far crispants sit from
# controls, whereas log_sd_ratio asks how heterogeneous the crispants are among themselves -- which
# is the quantity a within-group dose analysis actually consumes.
METRIC_LABELS = {
    "separation": "morphological separation (shrunk Mahalanobis)",
    "loo_auc": "axis quality (leave-one-out AUC)",
    "rank_rho_median": "ordinal stability (bootstrap rank $\\rho$)",
    "log_sd_ratio": "crispant / control spread  (log SD ratio)",
    "top_retention_median": "top-tercile retention",
    "boot_angle_p95": "bootstrap angle p95 (lower = better)",
}

ARM_COLOURS = {"binary_z": "#4A4A4A", "s_z": "#C0392B", "hinge_z": "#E08A2E"}
ARM_LABELS = {"binary_z": "binary", "s_z": "$s$", "hinge_z": "hinge"}
ARM_ORDER = ("binary_z", "s_z", "hinge_z")


def use_house_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.3", "axes.linewidth": 0.9, "xtick.color": "0.3",
        "ytick.color": "0.3", "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 160, "legend.frameon": False,
    })


def _stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "**" if q < 0.01 else ("*" if q < 0.1 else "")


# ---------------------------------------------------------------------------
# The per-contrast panel -- axis on the left, regression outcome on the right
# ---------------------------------------------------------------------------


def contrast_panel(contrast: str, scores: pd.DataFrame, quality: pd.DataFrame,
                   global_stats: pd.DataFrame, *, figsize=(11.5, 2.9)):
    """One contrast: where the embryos sit along ``s``, and what the three regressions did with it.

    Left panel is the dot plot -- every embryo's signed distance, controls below and crispants
    above, with the decision threshold at 0. This is the thing to read first: it shows the
    separation, the overlap, and the within-crispant spread that the graded score is supposed to be
    exploiting.

    Right panel is the standardised global statistic for each arm. Because all three predictors are
    z-scored and carry one degree of freedom, the bars are directly comparable -- taller means the
    cell-type composition moved further from its permutation null.
    """
    block = scores.loc[scores["contrast"] == contrast]
    row = quality.loc[quality["contrast"] == contrast].iloc[0]
    stats_block = global_stats.loc[global_stats["contrast"] == contrast].set_index("arm")

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(1, 2, width_ratios=[2.6, 1.0], wspace=0.28, figure=figure)

    # ---- left: embryo positions relative to the decision threshold ----
    axis = figure.add_subplot(grid[0, 0])
    rng = np.random.RandomState(0)
    for label, colour, offset, name in (
        (0, CONTROL_COLOUR, 0.0, "ctrl"), (1, CRISPANT_COLOUR, 1.0, "crispant")
    ):
        values = block.loc[block["is_crispant"] == label, "s"].to_numpy()
        axis.scatter(values, offset + rng.uniform(-0.15, 0.15, size=len(values)),
                     s=44, color=colour, alpha=0.88, edgecolor="white", linewidth=0.7, zorder=3)
        axis.plot([values.mean()] * 2, [offset - 0.28, offset + 0.28], color=colour,
                  linewidth=2.4, zorder=4)
    axis.axvline(0, color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axis.text(0, 1.62, "decision threshold", fontsize=7.5, color=NULL_COLOUR, ha="center")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["ctrl", "crispant"], fontsize=9)
    axis.set_ylim(-0.55, 1.72)
    axis.set_xlabel("signed distance $s$")

    flags = []
    if bool(row.get("usable", False)):
        flags.append("USABLE")
    if bool(row.get("stage_confounded", False)):
        flags.append("stage-confounded")
    axis.set_title(
        f"{contrast}      " + "   ".join(flags),
        fontsize=10.5, loc="left",
    )
    axis.text(
        0.995, 0.04,
        f"LOO-AUC {row['loo_auc']:.2f}{_stars(row['q_auc'])}   "
        f"boot angle p95 {row['boot_angle_p95']:.0f}°   "
        r"rank $\rho$ " f"{row['rank_rho_median']:.2f}   "
        f"n={int(row['n_crispant'])}+{int(row['n_control'])}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="0.35",
    )

    # ---- right: the three arms' global statistics ----
    bars = figure.add_subplot(grid[0, 1])
    present = [arm for arm in ARM_ORDER if arm in stats_block.index]
    heights = [stats_block.loc[arm, "t_std"] for arm in present]
    bars.bar(range(len(present)), heights,
             color=[ARM_COLOURS[a] for a in present], width=0.66, zorder=3)
    for position, arm in enumerate(present):
        p = stats_block.loc[arm, "p_global"]
        bars.text(position, heights[position], f"  p={p:.3f}", rotation=90,
                  ha="center", va="bottom", fontsize=7.2, color="0.3")
    bars.axhline(0, color="0.3", linewidth=0.9)
    bars.set_xticks(range(len(present)))
    bars.set_xticklabels([ARM_LABELS[a] for a in present], fontsize=9)
    bars.set_ylabel("$T_{std}$", fontsize=9)
    bars.set_ylim(0, max(max(heights) * 1.45, 1))
    bars.set_title("composition shift", fontsize=9.5)
    return figure


def contrast_grid(scores, quality, global_stats, contrasts, *, per_page=6):
    """Yield stacked per-contrast panels, ``per_page`` at a time."""
    for start in range(0, len(contrasts), per_page):
        chunk = contrasts[start:start + per_page]
        figure = plt.figure(figsize=(11.5, 2.9 * len(chunk)))
        outer = gridspec.GridSpec(len(chunk), 1, hspace=0.85, figure=figure)
        for index, contrast in enumerate(chunk):
            block = scores.loc[scores["contrast"] == contrast]
            row = quality.loc[quality["contrast"] == contrast].iloc[0]
            stats_block = global_stats.loc[
                global_stats["contrast"] == contrast].set_index("arm")
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[index], width_ratios=[2.6, 1.0], wspace=0.28
            )
            _draw_dotplot(figure.add_subplot(inner[0]), contrast, block, row)
            _draw_bars(figure.add_subplot(inner[1]), stats_block)
        yield figure


def _draw_dotplot(axis, contrast, block, row):
    rng = np.random.RandomState(0)
    for label, colour, offset in ((0, CONTROL_COLOUR, 0.0), (1, CRISPANT_COLOUR, 1.0)):
        values = block.loc[block["is_crispant"] == label, "s"].to_numpy()
        axis.scatter(values, offset + rng.uniform(-0.15, 0.15, size=len(values)),
                     s=40, color=colour, alpha=0.88, edgecolor="white", linewidth=0.7, zorder=3)
        axis.plot([values.mean()] * 2, [offset - 0.28, offset + 0.28], color=colour,
                  linewidth=2.2, zorder=4)
    axis.axvline(0, color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["ctrl", "crispant"], fontsize=8.5)
    axis.set_ylim(-0.55, 1.6)
    axis.set_xlabel("signed distance $s$", fontsize=9)
    flag = "  USABLE" if bool(row.get("usable", False)) else ""
    axis.set_title(f"{contrast}{flag}", fontsize=10, loc="left")
    axis.text(0.995, 0.04,
              f"AUC {row['loo_auc']:.2f}{_stars(row['q_auc'])}  "
              f"angle {row['boot_angle_p95']:.0f}°  "
              r"$\rho$ " f"{row['rank_rho_median']:.2f}",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=7.6, color="0.35")


def _draw_bars(axis, stats_block):
    present = [arm for arm in ARM_ORDER if arm in stats_block.index]
    heights = [stats_block.loc[arm, "t_std"] for arm in present]
    axis.bar(range(len(present)), heights, color=[ARM_COLOURS[a] for a in present],
             width=0.66, zorder=3)
    axis.axhline(0, color="0.3", linewidth=0.9)
    axis.set_xticks(range(len(present)))
    axis.set_xticklabels([ARM_LABELS[a] for a in present], fontsize=8.5)
    axis.set_ylabel("$T_{std}$", fontsize=8.5)
    axis.set_ylim(0, max(max(heights) * 1.3, 1))


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------


def head_to_head(global_stats: pd.DataFrame, quality: pd.DataFrame, *, figsize=(9.5, 6.2)):
    """Paired slope plot: each contrast's global statistic under each of the three predictors.

    Lines rather than bars because the comparison is *within* contrast -- the absolute level varies
    enormously between contrasts and is not the point. A line sloping down from `binary` says the
    graded morphology score moved the composition less than the plain indicator did.
    """
    wide = global_stats.pivot(index="contrast", columns="arm", values="t_std")
    wide = wide.join(quality.set_index("contrast")[["usable", "loo_auc"]])
    wide = wide.sort_values("binary_z", ascending=False)

    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for axis, subset, title in (
        (axes[0], wide, f"all contrasts (n={len(wide)})"),
        (axes[1], wide[wide["usable"]], f"usable axes only (n={int(wide['usable'].sum())})"),
    ):
        present = [a for a in ARM_ORDER if a in subset.columns]
        for _, row in subset.iterrows():
            values = [row[a] for a in present]
            improved = values[1] > values[0]
            axis.plot(range(len(present)), values,
                      color=CRISPANT_COLOUR if improved else "0.6",
                      alpha=0.85 if improved else 0.45,
                      linewidth=1.8 if improved else 1.0, zorder=3 if improved else 2,
                      marker="o", markersize=4)
        medians = [subset[a].median() for a in present]
        axis.plot(range(len(present)), medians, color="black", linewidth=3, marker="s",
                  markersize=8, zorder=5, label="median")
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels([ARM_LABELS[a] for a in present])
        axis.set_title(title, fontsize=10.5)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("$T_{std}$  (composition shift vs permutation null)")
    figure.suptitle("Does replacing the binary indicator with a graded morphology score help?",
                    fontsize=12, y=1.0)
    figure.tight_layout()
    return figure


def geometry_plot(geometry: pd.DataFrame, quality: pd.DataFrame, *, figsize=(11.5, 4.8)):
    """Direction and magnitude of each arm's coefficient vector relative to the binary arm's.

    The geometric claim was: same direction, bigger magnitude. Left panel tests it -- points should
    sit at high cosine and above the horizontal line if morphology sharpened the same signal.
    Right panel splits each vector into the part parallel to the binary contrast and the part
    orthogonal to it; a large orthogonal share means the morphology score is picking up something
    the label does not, which may be signal or may be noise.
    """
    merged = geometry.merge(quality[["contrast", "usable", "loo_auc"]], on="contrast")
    merged = merged.loc[merged["arm"] != "binary_z"]

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    for arm, marker in (("s_z", "o"), ("hinge_z", "D")):
        block = merged.loc[merged["arm"] == arm]
        axes[0].scatter(block["cosine_to_binary"], block["norm"] / block["norm_binary"],
                        s=70, marker=marker, c=[ARM_COLOURS[arm]] * len(block),
                        alpha=[1.0 if u else 0.35 for u in block["usable"]][0]
                        if len(block) else 1.0,
                        edgecolor="0.25", linewidth=0.7, label=ARM_LABELS[arm], zorder=3)
    axes[0].axhline(1.0, color=NULL_COLOUR, linestyle="--", linewidth=1.2)
    axes[0].text(0.02, 1.02, "equal magnitude to binary", fontsize=8, color=NULL_COLOUR)
    axes[0].set_xlabel("cosine to the binary arm's coefficient vector")
    axes[0].set_ylabel("magnitude relative to binary")
    axes[0].set_title("Same direction, bigger magnitude?", fontsize=10.5)
    axes[0].legend(fontsize=9)

    usable = merged.loc[merged["usable"] & (merged["arm"] == "s_z")].sort_values(
        "parallel_component", ascending=False)
    positions = np.arange(len(usable))
    axes[1].barh(positions, usable["parallel_component"], color=CRISPANT_COLOUR,
                 label="parallel to binary", zorder=3)
    axes[1].barh(positions, usable["orthogonal_component"],
                 left=usable["parallel_component"], color="#E7B7B0",
                 label="orthogonal (new or noise)", zorder=3)
    axes[1].plot(usable["norm_binary"], positions, "k|", markersize=11,
                 label="binary arm magnitude", zorder=5)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(usable["contrast"], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$\|z\|$ decomposition for the $s$ arm")
    axes[1].set_title("Where the $s$ arm's signal sits", fontsize=10.5)
    axes[1].legend(fontsize=8.5, loc="lower right")
    figure.tight_layout()
    return figure


def quality_versus_gain(global_stats: pd.DataFrame, quality: pd.DataFrame, *, figsize=(6.4, 5.6)):
    """Does a better morphology axis buy a bigger regression gain?

    The natural rescue for a negative result: maybe the axis was just bad. If that were the story,
    contrasts with high LOO-AUC would show positive gains. A flat cloud says the axis quality is
    not the limiting factor.
    """
    from scipy import stats as sps

    wide = global_stats.pivot(index="contrast", columns="arm", values="t_std")
    merged = wide.join(quality.set_index("contrast")[["loo_auc", "usable", "target"]])
    merged["gain"] = merged["s_z"] - merged["binary_z"]

    figure, axis = plt.subplots(figsize=figsize)
    markers = {t: m for t, m in zip(sorted(merged["target"].unique()), ("o", "s", "D", "^"))}
    for target, block in merged.groupby("target"):
        axis.scatter(block["loo_auc"], block["gain"], s=78, marker=markers[target],
                     c=[CRISPANT_COLOUR if u else "0.62" for u in block["usable"]],
                     edgecolor="0.25", linewidth=0.7, zorder=3)
    axis.axhline(0, color=NULL_COLOUR, linestyle="--", linewidth=1.3, zorder=1)
    axis.axvline(0.5, color=NULL_COLOUR, linestyle=":", linewidth=1.1, zorder=1)
    axis.text(0.505, axis.get_ylim()[1], " chance AUC", fontsize=8, color=NULL_COLOUR, va="top")

    valid = merged[["loo_auc", "gain"]].dropna()
    rho = sps.spearmanr(valid["loo_auc"], valid["gain"])
    axis.set_xlabel("axis quality — leave-one-out AUC")
    axis.set_ylabel(r"$T_{std}(s) - T_{std}(\mathrm{binary})$")
    axis.set_title("Would a better axis have rescued it?\n"
                   f"Spearman = {rho.statistic:+.2f} (p = {rho.pvalue:.3f})", fontsize=11)
    handles = [mpl.lines.Line2D([], [], marker=markers[t], linestyle="none", color="0.45",
                                markeredgecolor="0.25", markersize=8, label=t)
               for t in sorted(merged["target"].unique())]
    axis.legend(handles=handles, title="target", fontsize=9, loc="lower left")
    figure.tight_layout()
    return figure


def within_slope_plot(global_stats: pd.DataFrame, quality: pd.DataFrame, *, figsize=(6.4, 5.8)):
    """The within-crispant dose effect against its own negative control.

    Both slopes come from ``~ binary + s_within:group``, so the group difference has already been
    removed and each measures only the gradient *inside* one group. The control slope is the
    negative control: control embryos also vary along ``s``, and if that variation is measurement
    noise the slope should sit at zero.

    Points high and to the left are the wanted result -- a real dose effect among crispants with a
    silent control. Points on the diagonal would mean both groups respond alike, which is the
    signature of a shared nuisance (developmental stage being the obvious candidate) rather than a
    perturbation dose.
    """
    wide = global_stats.pivot(index="contrast", columns="arm", values="t_std")
    if not {"s_within_control", "s_within_crispant"} <= set(wide.columns):
        raise ValueError("within-group slope arms are absent from global_stats")
    merged = wide.join(quality.set_index("contrast")[["usable", "target", "loo_auc"]])

    figure, axis = plt.subplots(figsize=figsize)
    markers = {t: m for t, m in zip(sorted(merged["target"].dropna().unique()), ("o", "s", "D", "^"))}
    for target, block in merged.groupby("target"):
        axis.scatter(block["s_within_control"], block["s_within_crispant"],
                     s=80, marker=markers[target],
                     c=[CRISPANT_COLOUR if u else "0.62" for u in block["usable"]],
                     edgecolor="0.25", linewidth=0.7, zorder=3)
    span = float(np.nanmax(np.abs(merged[["s_within_control", "s_within_crispant"]].to_numpy()))) * 1.1
    axis.plot([-span, span], [-span, span], color=NULL_COLOUR, linestyle="--", linewidth=1.2,
              zorder=1)
    axis.axhline(0, color="0.8", linewidth=0.9, zorder=1)
    axis.axvline(0, color="0.8", linewidth=0.9, zorder=1)
    axis.set_xlim(-span, span)
    axis.set_ylim(-span, span)
    axis.set_xlabel("$T_{std}$ — within-CONTROL slope  (negative control)")
    axis.set_ylabel("$T_{std}$ — within-CRISPANT slope  (the dose effect)")
    axis.set_title("Does the gradient inside a crispant group add\nanything beyond the label?",
                   fontsize=11)
    handles = [mpl.lines.Line2D([], [], marker=markers[t], linestyle="none", color="0.45",
                                markeredgecolor="0.25", markersize=8, label=t)
               for t in sorted(merged["target"].dropna().unique())]
    axis.legend(handles=handles, title="target", fontsize=9, loc="upper left")
    return figure


def three_level_plot(global_stats: pd.DataFrame, quality: pd.DataFrame, *, figsize=(9.5, 5.8)):
    """Wildtype-looking crispants and severe crispants, each against the shared controls.

    The direct test of why a graded score would fail. If morphology tracked molecular severity,
    escapers would sit near zero and only the severe group would move. Escapers landing well above
    zero says morphologically normal-looking crispants are substantially perturbed -- morphological
    severity is then a weak proxy for molecular severity, which is exactly the failure mode the
    head-to-head comparison reported.
    """
    wide = global_stats.pivot(index="contrast", columns="arm", values="t_std")
    present = [a for a in ("escaper_vs_control", "severe_vs_control", "binary_z")
               if a in wide.columns]
    subset = wide.loc[wide[["escaper_vs_control", "severe_vs_control"]].notna().all(axis=1), present]
    subset = subset.join(quality.set_index("contrast")[["usable"]]).sort_values(
        "severe_vs_control", ascending=False)

    labels = {"escaper_vs_control": "escaper\nvs ctrl", "severe_vs_control": "severe\nvs ctrl",
              "binary_z": "all crispants\nvs ctrl"}
    order = [a for a in ("escaper_vs_control", "severe_vs_control", "binary_z") if a in present]

    figure, axes = plt.subplots(1, 2, figsize=figsize,
                                gridspec_kw={"width_ratios": [1.15, 1.0]})
    for _, row in subset.iterrows():
        axes[0].plot(range(len(order)), [row[a] for a in order],
                     color=CRISPANT_COLOUR if row["usable"] else "0.65",
                     alpha=0.85 if row["usable"] else 0.4,
                     linewidth=1.8 if row["usable"] else 1.0, marker="o", markersize=4,
                     zorder=3 if row["usable"] else 2)
    axes[0].plot(range(len(order)), [subset[a].median() for a in order], color="black",
                 linewidth=3, marker="s", markersize=8, zorder=5, label="median")
    axes[0].axhline(0, color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([labels[a] for a in order])
    axes[0].set_ylabel("$T_{std}$  (composition shift vs permutation null)")
    axes[0].set_title(f"n = {len(subset)} estimable contrasts\nred = usable axis", fontsize=10.5)
    axes[0].legend(fontsize=9)

    axes[1].scatter(subset["severe_vs_control"], subset["escaper_vs_control"], s=80,
                    c=[CRISPANT_COLOUR if u else "0.62" for u in subset["usable"]],
                    edgecolor="0.25", linewidth=0.7, zorder=3)
    limit = float(np.nanmax(subset[["severe_vs_control", "escaper_vs_control"]].to_numpy())) * 1.1
    axes[1].plot([0, limit], [0, limit], color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1)
    axes[1].axhline(0, color="0.8", linewidth=0.9, zorder=1)
    axes[1].text(limit * 0.97, limit * 0.93, "equally perturbed", color=NULL_COLOUR,
                 ha="right", fontsize=9)
    axes[1].set_xlabel("$T_{std}$ — severe crispants vs control")
    axes[1].set_ylabel("$T_{std}$ — escapers vs control")
    axes[1].set_title("Are wildtype-looking crispants\nactually unperturbed?", fontsize=10.5)
    figure.tight_layout()
    return figure


def power_vs_phenotype(global_stats: pd.DataFrame, quality: pd.DataFrame, *,
                       metric: str = "separation", figsize=(13, 4.4)):
    """Conditional on a *detectable* morphological phenotype, does morphology buy statistical power?

    This is a different and fairer question than "does morphology help uniformly". Contrasts where
    the crispants are barely distinguishable from controls have no morphological signal to exploit,
    and averaging them in dilutes whatever the informative contrasts have to say. Here the
    within-crispant dose signal is plotted against how strong the morphological phenotype actually
    was.

    Each panel carries the **within-control slope** alongside as a matched negative control. That is
    what separates a real dose-response from "some contrasts are simply cleaner than others": a
    generic quality effect would lift both slopes together, whereas a genuine morphology-to-molecular
    relationship should lift only the crispant side.
    """
    crispant = global_stats[global_stats["arm"] == "s_within_crispant"].set_index("contrast")
    control = global_stats[global_stats["arm"] == "s_within_control"].set_index("contrast")
    joined = crispant[["t_std", "p_global"]].join(
        control[["t_std", "p_global"]], rsuffix="_control"
    ).join(quality.set_index("contrast")[[metric, "loo_auc", "usable"]])

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    label = METRIC_LABELS.get(metric, metric)

    # --- panel A: dose-response, with the negative control overlaid ---
    axis = axes[0]
    axis.scatter(joined[metric], joined["t_std_control"], s=52, facecolor="none",
                 edgecolor=NULL_COLOUR, linewidth=1.1, zorder=2,
                 label="within-control slope (neg. control)")
    significant = joined["p_global"] < 0.05
    axis.scatter(joined.loc[~significant, metric], joined.loc[~significant, "t_std"],
                 s=58, color="0.62", edgecolor="0.3", linewidth=0.6, zorder=3,
                 label="within-crispant, n.s.")
    axis.scatter(joined.loc[significant, metric], joined.loc[significant, "t_std"],
                 s=78, color=CRISPANT_COLOUR, edgecolor="0.2", linewidth=0.7, zorder=4,
                 label="within-crispant, p<0.05")
    axis.axhline(0, color="0.8", linewidth=0.9, zorder=1)
    axis.set_xlabel(label)
    axis.set_ylabel("$T_{std}$  — within-group dose signal")
    from scipy import stats as sps
    valid = joined[[metric, "t_std"]].dropna()
    r = sps.spearmanr(valid[metric], valid["t_std"])
    axis.set_title(f"Dose-response   rho = {r.statistic:+.2f} (p = {r.pvalue:.3f})", fontsize=10.5)
    axis.legend(fontsize=8, loc="upper left")

    # --- panels B and C: the same thing binned, which is easier to read off ---
    joined["tercile"] = pd.qcut(joined[metric], 3, labels=["weak", "mid", "strong"])
    grouped = joined.groupby("tercile", observed=True)
    fraction = grouped.apply(
        lambda block: pd.Series({
            "crispant": (block["p_global"] < 0.05).mean(),
            "control": (block["p_global_control"] < 0.05).mean(),
            "median_t_crispant": block["t_std"].median(),
            "median_t_control": block["t_std_control"].median(),
            "n": len(block),
        }), include_groups=False,
    )

    positions = np.arange(len(fraction))
    axes[1].bar(positions - 0.19, fraction["crispant"], width=0.36, color=CRISPANT_COLOUR,
                label="within-crispant", zorder=3)
    axes[1].bar(positions + 0.19, fraction["control"], width=0.36, color="0.68",
                label="within-control", zorder=3)
    axes[1].axhline(0.05, color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=4)
    axes[1].text(len(fraction) - 0.55, 0.062, "chance", fontsize=8, color=NULL_COLOUR)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{t}\n(n={int(fraction['n'][t])})" for t in fraction.index])
    axes[1].set_ylabel("fraction of contrasts at p < 0.05")
    axes[1].set_xlabel(f"{label.split('(')[0].strip()} — tercile")
    axes[1].set_title("Hit rate by phenotype strength", fontsize=10.5)
    axes[1].legend(fontsize=8.5)

    axes[2].bar(positions - 0.19, fraction["median_t_crispant"], width=0.36,
                color=CRISPANT_COLOUR, label="within-crispant", zorder=3)
    axes[2].bar(positions + 0.19, fraction["median_t_control"], width=0.36, color="0.68",
                label="within-control", zorder=3)
    axes[2].axhline(0, color="0.4", linewidth=0.9)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(fraction.index)
    axes[2].set_ylabel("median $T_{std}$")
    axes[2].set_xlabel(f"{label.split('(')[0].strip()} — tercile")
    axes[2].set_title("Effect size by phenotype strength", fontsize=10.5)
    axes[2].legend(fontsize=8.5)

    figure.suptitle(
        "When the morphological phenotype IS detectable, does morphology buy power?",
        fontsize=12, y=1.03,
    )
    figure.tight_layout()
    return figure, fraction


def significance_vs_phenotype(global_stats: pd.DataFrame, quality: pd.DataFrame, *,
                              metric: str = "separation", figsize=(7.2, 7.0)):
    """Morphological phenotype strength against the *significance* of the within-group dose slope.

    The direct statement of the claim: a stronger morphological phenotype should make the
    within-crispant gradient a more significant predictor of composition. Plotted as
    -log10(p_global), so higher is more significant and the p = 0.05 threshold is a single line.

    The companion panel is the within-CONTROL slope on identical axes -- same embryos, same model,
    same permutation machinery, only the group changed. It is the control for "some contrasts are
    simply cleaner than others", which would tilt both panels together. Drawn beneath rather than
    overlaid because at 36 points per series an overlay obscures both.

    Note the permutation floor: with a finite number of draws the smallest attainable p is
    1/(n_perm + 1), so points pile up against a ceiling in -log10 space. That ceiling is marked.
    """
    from scipy import stats as sps

    crispant = global_stats[global_stats["arm"] == "s_within_crispant"].set_index("contrast")
    control = global_stats[global_stats["arm"] == "s_within_control"].set_index("contrast")
    axis_quality = quality.set_index("contrast")[[metric, "loo_auc", "usable", "target"]]

    label = METRIC_LABELS.get(metric, metric)
    ceiling = -np.log10(1.0 / (global_stats["p_global"].min() ** -1)) if False else None

    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": [1.0, 0.62], "hspace": 0.12})

    for axis, block, colour, name in (
        (axes[0], crispant, CRISPANT_COLOUR, "within-CRISPANT slope  (the dose effect)"),
        (axes[1], control, "0.45", "within-CONTROL slope  (negative control)"),
    ):
        joined = block[["p_global"]].join(axis_quality).dropna(subset=[metric])
        significance = -np.log10(joined["p_global"])
        hit = joined["p_global"] < 0.05

        axis.scatter(joined.loc[~hit, metric], significance[~hit], s=62, color="0.72",
                     edgecolor="0.3", linewidth=0.6, zorder=3)
        axis.scatter(joined.loc[hit, metric], significance[hit], s=86, color=colour,
                     edgecolor="0.2", linewidth=0.8, zorder=4)

        axis.axhline(-np.log10(0.05), color=NULL_COLOUR, linestyle="--", linewidth=1.3, zorder=2)
        axis.text(joined[metric].max(), -np.log10(0.05) + 0.04, "p = 0.05", fontsize=8.5,
                  color=NULL_COLOUR, ha="right", va="bottom")

        # A least-squares guide for the eye, with the rank correlation as the actual statistic.
        if len(joined) > 3:
            slope, intercept = np.polyfit(joined[metric], significance, 1)
            grid = np.linspace(joined[metric].min(), joined[metric].max(), 50)
            axis.plot(grid, slope * grid + intercept, color=colour, linewidth=1.6,
                      alpha=0.55, zorder=2)
            r = sps.spearmanr(joined[metric], significance)
            axis.set_title(f"{name}      Spearman = {r.statistic:+.2f}  (p = {r.pvalue:.3f})",
                           fontsize=10.5, loc="left")
        axis.set_ylabel(r"$-\log_{10}$ p")

    top = -np.log10(global_stats["p_global"].replace(0, np.nan).min())
    axes[0].axhline(top, color="0.85", linestyle=":", linewidth=1.1, zorder=1)
    axes[0].text(axes[0].get_xlim()[0], top + 0.03, "permutation floor", fontsize=7.5,
                 color="0.6", va="bottom")
    axes[1].set_xlabel(label)
    figure.suptitle("Does a stronger morphological phenotype make the dose slope\n"
                    "more significant?", fontsize=12, y=0.99)
    return figure


def metric_comparison(global_stats: pd.DataFrame, quality: pd.DataFrame, *,
                      metrics=("log_sd_ratio", "rank_rho_median", "separation", "loo_auc"),
                      figsize=(13.5, 6.4)):
    """Which summary of the morphological phenotype best predicts the dose slope's significance?

    Same scatter as :func:`significance_vs_phenotype`, repeated across candidate metrics so they can
    be compared on one page, each with its matched within-control panel beneath.

    They are not measuring the same thing. ``separation`` and ``loo_auc`` ask how far crispants sit
    from controls; ``log_sd_ratio`` asks how heterogeneous the crispants are *among themselves*. For
    a within-group dose analysis the second is the more relevant question -- a cleanly separated but
    internally uniform crispant group offers no gradient to detect, however striking the phenotype.

    **Read the p-values with the search in mind:** comparing several metrics and quoting the best one
    is a multiple-comparisons problem. The counts are printed by the caller.
    """
    from scipy import stats as sps

    crispant = global_stats[global_stats["arm"] == "s_within_crispant"].set_index("contrast")
    control = global_stats[global_stats["arm"] == "s_within_control"].set_index("contrast")
    columns = [c for c in quality.columns if c != "contrast"]
    axis_quality = quality.set_index("contrast")[columns]

    figure, axes = plt.subplots(2, len(metrics), figsize=figsize, sharey="row",
                                gridspec_kw={"height_ratios": [1.0, 0.6], "hspace": 0.35})
    axes = np.atleast_2d(axes)

    for column, metric in enumerate(metrics):
        for row, (block, colour) in enumerate(
            ((crispant, CRISPANT_COLOUR), (control, "0.45"))
        ):
            axis = axes[row, column]
            joined = block[["p_global"]].join(axis_quality[[metric]]).dropna()
            significance = -np.log10(joined["p_global"])
            hit = joined["p_global"] < 0.05

            axis.scatter(joined.loc[~hit, metric], significance[~hit], s=42, color="0.74",
                         edgecolor="0.35", linewidth=0.5, zorder=3)
            axis.scatter(joined.loc[hit, metric], significance[hit], s=62, color=colour,
                         edgecolor="0.2", linewidth=0.7, zorder=4)
            axis.axhline(-np.log10(0.05), color=NULL_COLOUR, linestyle="--", linewidth=1.1,
                         zorder=2)

            if len(joined) > 3:
                slope, intercept = np.polyfit(joined[metric], significance, 1)
                grid = np.linspace(joined[metric].min(), joined[metric].max(), 40)
                axis.plot(grid, slope * grid + intercept, color=colour, linewidth=1.5,
                          alpha=0.55, zorder=2)
                r = sps.spearmanr(joined[metric], significance)
                marker = "*" if r.pvalue < 0.05 else ""
                axis.set_title(f"rho {r.statistic:+.2f}  p={r.pvalue:.3f}{marker}",
                               fontsize=9.5)
            if row == 1:
                axis.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=8.5)
            axis.tick_params(labelsize=8)

        axes[0, column].set_xticklabels([])

    axes[0, 0].set_ylabel(r"$-\log_{10}$ p" "\nwithin-CRISPANT")
    axes[1, 0].set_ylabel(r"$-\log_{10}$ p" "\nwithin-CONTROL")
    figure.suptitle("Which measure of morphological phenotype best predicts the dose slope?\n"
                    "top = the dose effect, bottom = matched negative control   (* p < 0.05)",
                    fontsize=11.5, y=1.02)
    return figure


def hit_gain_plot(coefficients: pd.DataFrame, quality: pd.DataFrame, *,
                  arm: str = "s_within_crispant", q_threshold: float = 0.10,
                  figsize=(13, 6.2)):
    """How many cell types each contrast gains from the morphology term, contrast by contrast.

    Counts are reported as **set membership**, not as totals: for each contrast the cell types
    passing FDR under ``arm`` are split into those the binary indicator already found and those it
    did not. A median hit count would hide this entirely -- the distribution is zero-inflated and
    heavily skewed, so most contrasts gain nothing and a few gain a great deal.

    The within-CONTROL slope is drawn on the same axis as the negative control. It is the number
    that decides whether the gains are real: the same model on the same embryos with only the group
    changed should find nothing.
    """
    counts = []
    for contrast, block in coefficients.groupby("contrast"):
        sets = {}
        for name in ("binary_z", arm, "s_within_control"):
            rows = block[block["arm"] == name]
            sets[name] = set(rows.loc[rows["q_value"] < q_threshold, "cell_type"])
        counts.append({
            "contrast": contrast,
            "binary": len(sets["binary_z"]),
            "shared": len(sets[arm] & sets["binary_z"]),
            "new": len(sets[arm] - sets["binary_z"]),
            "arm_total": len(sets[arm]),
            "control": len(sets["s_within_control"]),
            "n_tested": block[block["arm"] == arm]["cell_type"].nunique(),
        })
    frame = (pd.DataFrame(counts)
             .merge(quality[["contrast", "usable", "loo_auc", "log_sd_ratio"]], on="contrast")
             .sort_values(["new", "arm_total"], ascending=False)
             .reset_index(drop=True))

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1.0], wspace=0.28, figure=figure)

    # --- left: per contrast, stacked shared/new against the binary baseline ---
    axis = figure.add_subplot(grid[0, 0])
    positions = np.arange(len(frame))
    axis.barh(positions, frame["binary"], height=0.66, color="0.78", zorder=2,
              label="binary indicator (baseline)")
    axis.barh(positions, frame["shared"], height=0.34, color="#8FA9C4", zorder=3,
              label=f"{ARM_LABELS.get(arm, arm)}: also found by binary")
    axis.barh(positions, frame["new"], height=0.34, left=frame["shared"],
              color=CRISPANT_COLOUR, zorder=3, label=f"{ARM_LABELS.get(arm, arm)}: NEW")
    axis.plot(frame["control"], positions, "k|", markersize=9, zorder=5,
              label="within-control slope (neg. control)")
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{'* ' if u else '  '}{c}" for c, u in zip(frame["contrast"], frame["usable"])],
        fontsize=6.6)
    axis.invert_yaxis()
    axis.set_xlabel(f"cell types passing q < {q_threshold:g} (within contrast)")
    axis.set_title(f"Cell types resolved per contrast   (* = usable axis)\n"
                   f"total new: {int(frame['new'].sum())}   "
                   f"negative control total: {int(frame['control'].sum())}", fontsize=10.5)
    axis.legend(fontsize=8, loc="lower right")

    # --- right: the same thing as distributions ---
    box = figure.add_subplot(grid[0, 1])
    series = [frame["binary"], frame["arm_total"], frame["new"], frame["control"]]
    labels = ["binary", ARM_LABELS.get(arm, arm), "NEW only", "neg. ctrl"]
    colours = ["0.78", "#8FA9C4", CRISPANT_COLOUR, "0.45"]
    parts = box.boxplot(series, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.6))
    for patch, colour in zip(parts["boxes"], colours):
        patch.set_facecolor(colour); patch.set_alpha(0.85)
    rng = np.random.RandomState(0)
    for index, values in enumerate(series, start=1):
        box.scatter(index + rng.uniform(-0.16, 0.16, len(values)), values, s=16,
                    color="0.25", alpha=0.55, zorder=4)
    box.set_xticks(range(1, len(labels) + 1))
    box.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    box.set_ylabel(f"cell types at q < {q_threshold:g}")
    box.set_title("Distribution across contrasts", fontsize=10.5)

    figure.suptitle("Does adding morphology resolve cell types the label alone misses?",
                    fontsize=12.5, y=1.0)
    return figure, frame


def volcano_within(coefficients: pd.DataFrame, *, arm: str = "s_within_crispant",
                   q_threshold: float = 0.10, figsize=(11.5, 5.2)):
    """Pooled volcano for the morphology term, with the negative control beside it.

    Every (contrast, cell type) test on one pair of axes. Points are coloured by whether the binary
    indicator had already found that cell type in that contrast, so the red points are precisely the
    gain being claimed.

    The right panel is the within-CONTROL slope drawn on identical axes. Read the two together: the
    claim is only as strong as the emptiness of the right panel.
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    binary = coefficients[coefficients["arm"] == "binary_z"]
    binary_hits = set(zip(binary.loc[binary["q_value"] < q_threshold, "contrast"],
                          binary.loc[binary["q_value"] < q_threshold, "cell_type"]))

    for axis, name, title in (
        (axes[0], arm, ARM_LABELS.get(arm, arm)),
        (axes[1], "s_within_control", "within-control slope (negative control)"),
    ):
        block = coefficients[coefficients["arm"] == name].copy()
        block["significance"] = -np.log10(block["p_value"].clip(lower=1e-12))
        significant = block["q_value"] < q_threshold
        already = [(c, t) in binary_hits for c, t in zip(block["contrast"], block["cell_type"])]
        already = np.asarray(already)

        axis.scatter(block.loc[~significant, "logFC"], block.loc[~significant, "significance"],
                     s=7, color="0.82", edgecolor="none", zorder=2)
        axis.scatter(block.loc[significant & already, "logFC"],
                     block.loc[significant & already, "significance"],
                     s=26, color="#4A7BA8", edgecolor="none", zorder=3,
                     label=f"q<{q_threshold:g}, also found by binary "
                           f"({int((significant & already).sum())})")
        axis.scatter(block.loc[significant & ~already, "logFC"],
                     block.loc[significant & ~already, "significance"],
                     s=34, color=CRISPANT_COLOUR, edgecolor="0.2", linewidth=0.4, zorder=4,
                     label=f"q<{q_threshold:g}, NEW ({int((significant & ~already).sum())})")
        axis.axvline(0, color="0.85", linewidth=0.9, zorder=1)
        axis.set_xlabel("log fold change per SD of the morphology score")
        axis.set_title(title, fontsize=10.5)
        axis.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel(r"$-\log_{10}$ p (uncorrected)")
    figure.suptitle("Every (contrast x cell type) test, pooled — and the same for the control",
                    fontsize=12, y=1.02)
    figure.tight_layout()
    return figure


def cell_type_scatter(coefficients: pd.DataFrame, contrast: str, *, figsize=(5.8, 5.6)):
    """Per-cell-type standardised scores, binary arm against ``s`` arm, for one contrast.

    The contrast-level statistic is a summary; this is what it is summarising. Points on the
    diagonal mean the two predictors agree cell type by cell type; systematic compression toward
    the horizontal means the morphology score is an attenuated version of the same signal.
    """
    block = coefficients.loc[coefficients["contrast"] == contrast]
    binary = block.loc[block["arm"] == "binary_z"].set_index("cell_type")
    graded = block.loc[block["arm"] == "s_z"].set_index("cell_type")
    shared = binary.index.intersection(graded.index)

    figure, axis = plt.subplots(figsize=figsize)
    significant = (binary.loc[shared, "q_value"] < 0.1) | (graded.loc[shared, "q_value"] < 0.1)
    axis.scatter(binary.loc[shared, "score_z"][~significant],
                 graded.loc[shared, "score_z"][~significant],
                 s=26, color="0.72", edgecolor="none", zorder=2, label="q ≥ 0.1 in both")
    axis.scatter(binary.loc[shared, "score_z"][significant],
                 graded.loc[shared, "score_z"][significant],
                 s=48, color=CRISPANT_COLOUR, edgecolor="0.25", linewidth=0.6, zorder=3,
                 label="q < 0.1 in either")

    limit = float(np.nanmax(np.abs(np.r_[binary.loc[shared, "score_z"],
                                         graded.loc[shared, "score_z"]]))) * 1.1
    axis.plot([-limit, limit], [-limit, limit], color=NULL_COLOUR, linestyle="--", linewidth=1.2)
    axis.axhline(0, color="0.85", linewidth=0.8, zorder=1)
    axis.axvline(0, color="0.85", linewidth=0.8, zorder=1)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("standardised score — binary arm")
    axis.set_ylabel("standardised score — $s$ arm")
    axis.set_title(contrast, fontsize=10.5)
    axis.legend(fontsize=8.5, loc="upper left")
    figure.tight_layout()
    return figure


def abundance_profile(index: pd.DataFrame, *, figsize=(8.5, 4.4), top: int = 30):
    """Cell-type abundance, the compositional-risk diagnostic.

    If one type were a large share of the panel, a real change in it would drag every other
    coefficient negative and fill the results with reciprocal artefacts. A flat profile means that
    failure mode is not in play.
    """
    ordered = index.sort_values("mean_fraction", ascending=False).head(top)
    figure, axis = plt.subplots(figsize=figsize)
    axis.barh(np.arange(len(ordered)), ordered["mean_fraction"], color="#4A6FA5", zorder=3)
    axis.set_yticks(np.arange(len(ordered)))
    axis.set_yticklabels(ordered["cell_type"], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("mean fraction of cells per embryo")
    axis.set_title(f"Abundance profile — top {top} of {len(index)} cell types\n"
                   f"largest is {index['mean_fraction'].max():.1%}; "
                   f"top 5 sum to {index['mean_fraction'].nlargest(5).sum():.1%}",
                   fontsize=10.5)
    figure.tight_layout()
    return figure


def save(figure, path) -> Path:
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target
