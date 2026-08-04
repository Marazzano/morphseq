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


def overlap_counts(arm_coefficients: pd.DataFrame, reference_coefficients: pd.DataFrame, *,
                   arm: str, reference: str = "binary_z", q_threshold: float = 0.10,
                   exclude: "list[str] | None" = None) -> pd.DataFrame:
    """Per-contrast split of two arms' hit sets into reference-only / shared / arm-only.

    The two arms may live in DIFFERENT coefficient tables -- that is the point. The unsupervised
    arms are fit by a separate script into ``data/unsupervised/coefficients.csv`` while the binary
    reference stays in ``data/edger/coefficients.csv``, and comparing them means joining on
    (contrast, cell_type) across files. Pass the same frame twice for a within-file comparison.
    """
    exclude = set(exclude or ())
    arm_block = arm_coefficients[arm_coefficients["arm"] == arm]
    reference_block = reference_coefficients[reference_coefficients["arm"] == reference]

    def hit_sets(block):
        return {contrast: set(group.loc[group["q_value"] < q_threshold, "cell_type"])
                for contrast, group in block.groupby("contrast")}

    found, reference_hits = hit_sets(arm_block), hit_sets(reference_block)
    tested = arm_block.groupby("contrast")["cell_type"].nunique()

    rows = []
    for contrast in sorted(set(found) & set(reference_hits)):
        if contrast in exclude:
            continue
        got, want = found[contrast], reference_hits[contrast]
        shared, union = got & want, got | want
        rows.append({
            "contrast": contrast,
            "reference_only": len(want - got), "shared": len(shared), "arm_only": len(got - want),
            "reference_total": len(want), "arm_total": len(got),
            "jaccard": len(shared) / len(union) if union else np.nan,
            "recovery": len(shared) / len(want) if want else np.nan,
            "precision": len(shared) / len(got) if got else np.nan,
            "n_tested": int(tested.get(contrast, 0)),
        })
    return pd.DataFrame(rows)


def contrast_sort_key(contrasts: pd.Series) -> pd.DataFrame:
    """Sort key for contrast labels: target gene, then timepoint, then temperature.

    Labels look like ``"wfs1a,wfs1b vs ctrl | 34C | 36hpf"``. Sorting on the raw string interleaves
    the three factors and makes a 35-row axis unreadable; this groups all of one target together,
    then walks time, then temperature within it.
    """
    parsed = contrasts.str.extract(
        r"^(?P<target>.+?)\s+vs\s+ctrl\s*\|\s*(?P<temperature>[\d.]+)C\s*\|\s*(?P<timepoint>[\d.]+)hpf")
    parsed["temperature"] = pd.to_numeric(parsed["temperature"], errors="coerce")
    parsed["timepoint"] = pd.to_numeric(parsed["timepoint"], errors="coerce")
    parsed["target"] = parsed["target"].fillna(contrasts)
    return parsed


def order_by_design(frame: pd.DataFrame, column: str = "contrast") -> pd.DataFrame:
    """Reorder rows by target gene, then timepoint, then temperature."""
    keys = contrast_sort_key(frame[column])
    return (frame.assign(**{f"_{k}": v for k, v in keys.items()})
            .sort_values(["_target", "_timepoint", "_temperature"])
            .drop(columns=[f"_{k}" for k in keys.columns])
            .reset_index(drop=True))


def hit_overlap_bars(arm_coefficients: pd.DataFrame, quality: pd.DataFrame, *, arm: str,
                     reference_coefficients: pd.DataFrame | None = None,
                     reference: str = "binary_z", arm_label: str | None = None,
                     q_threshold: float = 0.10, exclude: "list[str] | None" = None,
                     title: str | None = None, figsize=(10.4, 7.0),
                     fade_unvalidated: bool = False, fade_alpha: float = 0.30):
    """Bar-only view: who resolves what, contrast by contrast, in three disjoint colours.

    The middle panel of ``hit_overlap_plot`` on its own, with a pooled summary bar underneath, sized
    for a slide. Works for any arm against any reference, in the same table or across two -- so the
    supervised (``s_z``) and unsupervised (``pooled_pc1``, ``within_pc1``) comparisons render
    identically and can be put side by side.

    ``fade_unvalidated=True`` keeps every contrast on the axis but draws the ones whose morphology
    axis failed the §1 validity gate at ``fade_alpha``. This is the honest way to show the full set:
    a contrast whose axis never generalised has an ``arm_only`` count that is not evidence of
    anything, so hiding those rows overstates coverage while giving them equal visual weight
    overstates the result. Faded rows still contribute to the pooled "all" bar underneath.
    """
    reference_coefficients = (arm_coefficients if reference_coefficients is None
                              else reference_coefficients)
    label = arm_label or ARM_LABELS.get(arm, arm)
    frame = overlap_counts(arm_coefficients, reference_coefficients, arm=arm,
                           reference=reference, q_threshold=q_threshold, exclude=exclude)
    frame = frame.merge(quality[["contrast", "usable"]], on="contrast", how="left")
    frame["usable"] = frame["usable"].fillna(False).astype(bool)
    frame = order_by_design(frame)

    REFERENCE_ONLY, SHARED, ARM_ONLY = "0.62", "#8FA9C4", CRISPANT_COLOUR

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(2, 1, height_ratios=[len(frame), 3.4], hspace=0.34, figure=figure)

    axis = figure.add_subplot(grid[0, 0])
    positions = np.arange(len(frame))

    # Per-bar RGBA rather than one colour per series: barh takes a colour LIST, which is the only
    # way to vary alpha row by row within a single stacked series.
    def shade(colour):
        if not fade_unvalidated:
            return colour
        return [mpl.colors.to_rgba(colour, 1.0 if u else fade_alpha) for u in frame["usable"]]

    axis.barh(positions, frame["reference_only"], height=0.74, color=shade(REFERENCE_ONLY),
              zorder=3)
    axis.barh(positions, frame["shared"], height=0.74, left=frame["reference_only"],
              color=shade(SHARED), zorder=3)
    axis.barh(positions, frame["arm_only"], height=0.74,
              left=frame["reference_only"] + frame["shared"], color=shade(ARM_ONLY), zorder=3)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{'* ' if u else '   '}{c}" for c, u in
                          zip(frame["contrast"], frame["usable"])], fontsize=7.4)
    if fade_unvalidated:
        for tick, usable in zip(axis.get_yticklabels(), frame["usable"]):
            tick.set_color("0.2" if usable else "0.62")
    axis.invert_yaxis()
    axis.set_xlabel("cell types per contrast", fontsize=9)
    axis.tick_params(labelbottom=True, labelsize=8.5)

    # Explicit handles: with per-bar colour lists the automatic legend would sample whichever row
    # happened to come first and could show a faded swatch as if it were the series colour.
    handles = [mpl.patches.Patch(facecolor=colour, label=f"{name}  ({int(frame[column].sum())})")
               for column, colour, name in
               (("reference_only", REFERENCE_ONLY, "binary only"),
                ("shared", SHARED, "shared"),
                ("arm_only", ARM_ONLY, f"{label} only"))]
    # Deliberately NO extra "faded = ..." entry. The faded and standard versions are meant to
    # OVERLAY, so the legend has to stay the same size and shape in both; a fourth row shifts
    # everything against the reference figure. The fade is explained in the caption, not the legend.
    axis.legend(handles=handles, fontsize=9.2, loc="lower right")
    axis.set_title(title or f"Cell types resolved: binary indicator versus {label}"
                            f"      (* = validated morphology axis)", fontsize=11.5, pad=10)

    # Independent x-axis: the pooled totals are an order of magnitude longer than any single
    # contrast, so a shared scale flattens the per-contrast rows. Both panels carry visible ticks.
    pooled = figure.add_subplot(grid[1, 0])
    subsets = (("usable", frame[frame["usable"]]), ("all", frame))
    y = np.arange(len(subsets))
    left = np.zeros(len(subsets))
    for column, colour in (("reference_only", REFERENCE_ONLY), ("shared", SHARED),
                           ("arm_only", ARM_ONLY)):
        widths = np.array([float(block[column].sum()) for _, block in subsets])
        pooled.barh(y, widths, left=left, height=0.55, color=colour, edgecolor="white",
                    linewidth=1.0, zorder=3)
        for position, width, offset in zip(y, widths, left):
            if width >= 18:
                pooled.text(offset + width / 2, position, f"{int(width)}", ha="center",
                            va="center", fontsize=9.5,
                            color="white" if colour == ARM_ONLY else "0.15", zorder=4)
        left += widths
    pooled.set_yticks(y)
    pooled.set_yticklabels([f"{name} ({len(block)})" for name, block in subsets], fontsize=9)
    pooled.set_xlabel(f"cell types passing q < {q_threshold:g}   (pooled — note the separate scale)")
    pooled.spines["left"].set_visible(False)
    pooled.tick_params(axis="y", length=0)
    return figure, frame


def _pooled_scale(frame: pd.DataFrame) -> float:
    """Divisor that brings pooled totals onto the per-contrast axis."""
    per_contrast_max = (frame["reference_only"] + frame["shared"] + frame["arm_only"]).max()
    pooled_max = float(frame[["reference_only", "shared", "arm_only"]].sum().sum())
    return max(1.0, round(pooled_max / max(per_contrast_max, 1)))


def hit_overlap_plot(coefficients: pd.DataFrame, quality: pd.DataFrame, *, arm: str = "s_z",
                     q_threshold: float = 0.10, figsize=(14.2, 6.4)):
    """The same comparison as ``geometry_plot``, counted in resolved cell types instead of vectors.

    ``geometry_plot`` asks whether the morphology arm's COEFFICIENT VECTOR points the same way as
    the binary arm's and how long it is. This asks the downstream question a reader actually cares
    about: **which cell types does each one resolve, and how much do those sets overlap?**

    The two can disagree, and here they do. ``T_std`` is a sum of squared standardised scores, so it
    is driven by the magnitude of the strongest effects; a hit count saturates, because a cell type
    at q = 1e-36 counts exactly as much as one at q = 0.09. The binary arm produces far more extreme
    individual effects while resolving a similar number of cell types.

    Three disjoint categories per contrast, which is the whole point of the middle panel:

        binary-only   resolved by the class label, missed by the morphology score
        shared        resolved by both
        morph-only    resolved by the morphology score, missed by the class label

    NO NEGATIVE CONTROL EXISTS FOR THE MORPH-ONLY SET. Unlike the within-crispant slope in §5, which
    has the matched within-control slope beside it, nothing here bounds how many of the morph-only
    hits are FDR noise. At q < 0.10 the expected false count is ~10% of whatever the arm calls. The
    comparison is fair -- same one-predictor design, same df, same cell-type filter, same fixed
    dispersion, same BH family -- but "fair" is not the same as "verified".
    """
    binary_sets, arm_sets, rows = {}, {}, []
    for contrast, block in coefficients.groupby("contrast"):
        b = block[block["arm"] == "binary_z"]
        a = block[block["arm"] == arm]
        if not len(a):
            continue
        binary_sets[contrast] = set(b.loc[b["q_value"] < q_threshold, "cell_type"])
        arm_sets[contrast] = set(a.loc[a["q_value"] < q_threshold, "cell_type"])
        shared = binary_sets[contrast] & arm_sets[contrast]
        union = binary_sets[contrast] | arm_sets[contrast]
        rows.append({
            "contrast": contrast,
            "binary_only": len(binary_sets[contrast] - arm_sets[contrast]),
            "shared": len(shared),
            "morph_only": len(arm_sets[contrast] - binary_sets[contrast]),
            "binary_total": len(binary_sets[contrast]),
            "arm_total": len(arm_sets[contrast]),
            "jaccard": len(shared) / len(union) if union else np.nan,
            "recovery": len(shared) / len(binary_sets[contrast]) if binary_sets[contrast] else np.nan,
            "n_tested": a["cell_type"].nunique(),
        })
    frame = (pd.DataFrame(rows).merge(quality[["contrast", "usable", "loo_auc"]], on="contrast")
             .sort_values(["shared", "binary_total"], ascending=False).reset_index(drop=True))

    BINARY_ONLY, SHARED, MORPH_ONLY = "0.62", "#8FA9C4", CRISPANT_COLOUR
    label = ARM_LABELS.get(arm, arm)

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(1, 3, width_ratios=[0.95, 1.55, 0.6], wspace=0.5, figure=figure)

    # --- left: hit counts head to head, one point per contrast ---
    axis = figure.add_subplot(grid[0, 0])
    top = 1.08 * max(frame["binary_total"].max(), frame["arm_total"].max())
    axis.plot([0, top], [0, top], color=NULL_COLOUR, linestyle="--", linewidth=1.2, zorder=1,
              label="equal counts")
    for usable_flag, colour, name in ((True, CRISPANT_COLOUR, "usable axis"),
                                      (False, "0.66", "not usable")):
        block = frame[frame["usable"] == usable_flag]
        axis.scatter(block["binary_total"], block["arm_total"], s=68,
                     color=colour, edgecolor="0.25", linewidth=0.7, alpha=0.9, zorder=3,
                     label=f"{name} ({len(block)})")
    axis.set_xlim(-1, top); axis.set_ylim(-1, top)
    axis.set_xlabel("cell types resolved by the binary indicator")
    axis.set_ylabel(f"cell types resolved by {label}")
    axis.set_title("Equal footing on cell types resolved", fontsize=10.5)
    axis.legend(fontsize=8.5, loc="upper left")

    # --- middle: the three disjoint sets, per contrast ---
    bars = figure.add_subplot(grid[0, 1])
    positions = np.arange(len(frame))
    bars.barh(positions, frame["binary_only"], height=0.72, color=BINARY_ONLY, zorder=3,
              label=f"binary only ({int(frame['binary_only'].sum())})")
    bars.barh(positions, frame["shared"], height=0.72, left=frame["binary_only"],
              color=SHARED, zorder=3, label=f"shared ({int(frame['shared'].sum())})")
    bars.barh(positions, frame["morph_only"], height=0.72,
              left=frame["binary_only"] + frame["shared"], color=MORPH_ONLY, zorder=3,
              label=f"{label} only ({int(frame['morph_only'].sum())})")
    bars.set_yticks(positions)
    bars.set_yticklabels([f"{'* ' if u else '  '}{c}" for c, u in
                          zip(frame["contrast"], frame["usable"])], fontsize=6.6)
    bars.invert_yaxis()
    bars.set_xlabel(f"cell types passing q < {q_threshold:g}  (union of the two arms)")
    bars.set_title(f"Who resolves what, contrast by contrast   (* = usable axis)", fontsize=10.5)
    bars.legend(fontsize=8.5, loc="lower right")

    # --- right: pooled, all contrasts and the gated subset ---
    pooled = figure.add_subplot(grid[0, 2])
    subsets = (("all 36", frame), (f"usable ({int(frame['usable'].sum())})",
                                   frame[frame["usable"]]))
    y = np.arange(len(subsets))[::-1]
    left = np.zeros(len(subsets))
    for column, colour in (("binary_only", BINARY_ONLY), ("shared", SHARED),
                           ("morph_only", MORPH_ONLY)):
        widths = np.array([float(block[column].sum()) for _, block in subsets])
        pooled.barh(y, widths, left=left, height=0.5, color=colour, edgecolor="white",
                    linewidth=0.9, zorder=3)
        for position, width, offset in zip(y, widths, left):
            if width >= 25:
                pooled.text(offset + width / 2, position, f"{int(width)}", ha="center",
                            va="center", fontsize=9,
                            color="white" if colour == MORPH_ONLY else "0.15", zorder=4)
        left += widths
    for position, (_, block) in zip(y, subsets):
        pooled.text(left[list(y).index(position)] + 12, position,
                    f"{int(block['binary_total'].sum())} vs {int(block['arm_total'].sum())}",
                    va="center", fontsize=8.8, fontweight="bold", color="0.25")
    pooled.set_yticks(y); pooled.set_yticklabels([name for name, _ in subsets], fontsize=9)
    pooled.set_xlim(0, left.max() * 1.32)
    pooled.set_xlabel("pooled cell types")
    pooled.set_title("Pooled\n(binary total vs $s$ total)", fontsize=10)
    pooled.spines["left"].set_visible(False)
    pooled.tick_params(axis="y", length=0)

    figure.suptitle(f"Binary indicator versus {label}, counted in resolved cell types",
                    fontsize=12.5, y=1.01)
    return figure, frame


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


def slope_gain_bars(check: pd.DataFrame, quality: pd.DataFrame, *, q_threshold: float = 0.10,
                    usable_only: bool = False, layers: str = "both",
                    summary: pd.DataFrame | None = None, figsize=None):
    """Gains over the plain binary contrast, split by HOW the gain arrives.

    Everything is measured against one fixed baseline -- ``Model 1``, ``~ binary``, the analysis
    anyone would run without morphology. Adding the within-group slope terms then buys cell types
    through two distinct channels, and they deserve different amounts of trust:

        baseline    B1          significant under ``~ binary`` alone
        indirect    B2 \ B1     NOT significant under Model 1, but significant on the BINARY
                                coefficient of ``~ binary + s_within:group``. Morphology did not
                                detect these -- the slope columns absorbed within-group scatter, the
                                quasi-likelihood dispersion fell, and the group contrast sharpened
                                enough to carry them over FDR.
        direct      S \ (B1uB2) significant ONLY for the within-crispant slope. The discovery claim.

    The three are disjoint and sum to |B1 u B2 u S|, so bar length is total cell types resolved.

    THERE IS DELIBERATELY NO "BOTH" CATEGORY. ``s_within`` is orthogonalised to ``binary`` by
    construction, so the slope is structurally barred from sharing credit for the group difference;
    an intersection of the two hit sets therefore understates the real overlap badly rather than
    measuring it. Overlap between morphology and the label is a question for §2b, where the
    predictor is the full ``s`` and no orthogonalisation has been applied.

    ``layers="binary"`` draws the baseline alone -- the same rows, ordering and x-limits as
    ``layers="both"``, so the two render as a build: the gains appear on top of an unchanged bar.

    Rows are ordered by target gene, then timepoint, then temperature.

    Needs ``binary_coefficient_check.csv`` (from ``check_binary_coefficient.R``), which carries the
    Model 2 binary coefficient that ``coefficients.csv`` discards.
    """
    rows = []
    for contrast, block in check.groupby("contrast"):
        b1 = set(block.loc[block["q_binary_m1"] < q_threshold, "cell_type"])
        b2 = set(block.loc[block["q_binary_m2"] < q_threshold, "cell_type"])
        s_hits = set(block.loc[block["q_slope"] < q_threshold, "cell_type"])
        control = set(block.loc[block.get("q_slope_control", 1.0) < q_threshold, "cell_type"]) \
            if "q_slope_control" in block else set()
        rows.append({
            "contrast": contrast,
            "baseline": len(b1),
            "indirect": len(b2 - b1 - s_hits),
            "direct": len(s_hits - b1),
            "lost": len(b1 - b2),
            "control_slope": len(control),
            "n_tested": block["cell_type"].nunique(),
        })
    frame = pd.DataFrame(rows).merge(quality[["contrast", "usable"]], on="contrast", how="left")
    frame["usable"] = frame["usable"].fillna(False).astype(bool)
    # Axis limits come from the FULL set before any subsetting, so the all-contrast and gated
    # versions -- and the binary-only and binary+morph layers -- all share one scale and can be
    # overlaid or flipped between without anything moving.
    full_total = (frame["baseline"] + frame["indirect"] + frame["direct"])
    row_limit = 1.04 * full_total.max()
    pooled_limit = 1.05 * float(full_total.sum())
    if usable_only:
        frame = frame[frame["usable"]].copy()
    frame = order_by_design(frame)

    BASELINE, INDIRECT, DIRECT = "0.62", "#F3A18E", "#C0392B"   # grey, light coral, dark coral
    show_gains = layers != "binary"
    if figsize is None:
        figsize = (11.0, 0.26 * len(frame) + 3.4)

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(2, 1, height_ratios=[max(len(frame), 6), 2.6], hspace=0.34,
                             figure=figure)
    axis = figure.add_subplot(grid[0, 0])
    positions = np.arange(len(frame))

    axis.barh(positions, frame["baseline"], height=0.76, color=BASELINE, zorder=3,
              label=f"binary contrast, $\\sim$ binary  ({int(frame['baseline'].sum())})")
    if show_gains:
        axis.barh(positions, frame["indirect"], height=0.76, left=frame["baseline"],
                  color=INDIRECT, zorder=3,
                  label=f"indirect — group contrast sharpened  (+{int(frame['indirect'].sum())})")
        axis.barh(positions, frame["direct"], height=0.76,
                  left=frame["baseline"] + frame["indirect"], color=DIRECT, zorder=3,
                  label=f"direct — gradient only  (+{int(frame['direct'].sum())})")
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{'* ' if u else '   '}{c}" for c, u in zip(frame["contrast"], frame["usable"])],
        fontsize=7.2)
    axis.invert_yaxis()
    axis.set_xlim(0, row_limit)
    axis.set_xlabel("cell types per contrast", fontsize=9)
    axis.tick_params(labelbottom=True, labelsize=8.5)
    axis.legend(fontsize=9, loc="lower right")
    scope = "validated morphology axes only" if usable_only else "all contrasts   (* = validated axis)"
    axis.set_title(("Cell types resolved by the binary contrast alone — " + scope) if not show_gains
                   else ("What adding the morphology gradient buys — " + scope), fontsize=11.5, pad=10)

    pooled = figure.add_subplot(grid[1, 0])
    left = 0.0
    segments = [("baseline", BASELINE)] + ([("indirect", INDIRECT), ("direct", DIRECT)]
                                           if show_gains else [])
    for column, colour in segments:
        width = float(frame[column].sum())
        pooled.barh([0], [width], left=[left], height=0.62, color=colour, edgecolor="white",
                    linewidth=1.2, zorder=3)
        if width:
            pooled.text(left + width / 2, 0, f"{int(width)}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if colour == DIRECT else "0.15", zorder=4)
        left += width
    pooled.set_xlim(0, pooled_limit)
    pooled.set_ylim(-0.6, 0.6)
    pooled.set_yticks([0])
    pooled.set_yticklabels([f"pooled\n({len(frame)} contrasts)"], fontsize=9.5)
    pooled.set_xlabel(f"cell types passing q < {q_threshold:g}   (pooled — note the separate scale)")
    pooled.spines["left"].set_visible(False)
    pooled.tick_params(axis="y", length=0)
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


def slope_versus_binary_q(check: pd.DataFrame, *, q_threshold: float = 0.10,
                          near_miss: float = 0.25, figsize=(13.4, 5.6)):
    """The slope hits, scored against the binary contrast's q-value for the same cell type.

    Restricted to the cell types the within-crispant slope calls at ``q_threshold``. The question
    is what the binary indicator made of those same cell types: were they marginal binary hits the
    gradient nudged over the line, or effects the group contrast gives no hint of at all? The
    horizontal position answers it directly, and the strata make the split countable.

    TWO COMPARATORS, because the answer depends on which binary fit you mean and the difference is
    not negligible:

    ``Model 1``  ``~ binary`` on its own -- the analysis anyone would actually run, and the baseline
                 the "NEW" count in the synthesis notebook is defined against.
    ``Model 2``  the binary column of the SAME four-column design the slope comes from. Holding the
                 design fixed isolates the coefficient from the model, so a cell type new here is
                 new because of the gradient rather than because the comparator was refitted.

    Model 2 is the stricter reading and gives the smaller count. Both are shown because they answer
    different questions, and the gap between them is itself the audit result.

    Requires ``data/edger/binary_coefficient_check.csv`` from ``check_binary_coefficient.R``; the
    shipped ``coefficients.csv`` does not carry Model 2's binary coefficient.
    """
    hits = check[check["q_slope"] < q_threshold].copy()
    hits["y"] = -np.log10(hits["q_slope"])

    strata = (
        (f"also a binary hit (q < {q_threshold:g})", "#8FA9C4", 26),
        (f"near miss ({q_threshold:g} $\\leq$ q < {near_miss:g})", "#E08A2E", 34),
        (f"no binary signal (q $\\geq$ {near_miss:g})", CRISPANT_COLOUR, 38),
    )
    comparators = (("q_binary_m1", "Model 1:  $\\sim$ binary  (the standard analysis)"),
                   ("q_binary_m2", "Model 2:  binary column of the slope's own design"))

    figure = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.66], wspace=0.34, figure=figure)
    scatter_axes = [figure.add_subplot(grid[0, index]) for index in (0, 1)]

    # The two axes cover wildly different ranges -- the binary contrast reaches q ~ 1e-40 while the
    # slope tops out near 1e-4 -- so they are scaled independently. Forcing a shared scale (or a 1:1
    # line) would squash every slope hit into a strip and imply a comparison of magnitudes that is
    # not on offer: these are different coefficients, not two estimates of one thing. The binary
    # axis is capped and over-cap points drawn as triangles rather than silently piled on the edge.
    cap = 12.0
    y_top = 1.08 * hits["y"].max()
    counts = {}

    for axis, (column, title) in zip(scatter_axes, comparators):
        raw = -np.log10(hits[column])
        x = raw.clip(upper=cap)
        over = raw > cap
        band = np.select([hits[column] < q_threshold, hits[column] < near_miss],
                         [0, 1], default=2)
        counts[column] = [int((band == index).sum()) for index in range(3)]

        axis.axvline(-np.log10(q_threshold), color=NULL_COLOUR, linestyle="--", linewidth=1.0,
                     zorder=2)
        for index, (label, colour, size) in enumerate(strata):
            scale = 0.55 + 0.45 * hits["logfc_slope"].abs() / hits["logfc_slope"].abs().max()
            for mask, marker, boost in ((band == index) & ~over, "o", 1.0),\
                                       ((band == index) & over, ">", 1.5):
                if not mask.any():
                    continue
                axis.scatter(x[mask], hits.loc[mask, "y"], s=size * scale[mask] * boost,
                             marker=marker, color=colour, edgecolor="0.25", linewidth=0.35,
                             alpha=0.88, zorder=4,
                             label=f"{label}  —  {int((band == index).sum())}"
                                   if marker == "o" else None)
        axis.set_xlim(-0.25, cap + 0.6)
        axis.set_ylim(-np.log10(q_threshold) - 0.1, y_top)
        ticks = [t for t in axis.get_xticks() if 0 <= t < cap] + [cap]
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{t:g}" for t in ticks[:-1]] + [f"$\\geq${cap:g}"])
        axis.set_xlabel(r"$-\log_{10}$ q,  binary indicator")
        axis.set_title(title, fontsize=10.5)
        axis.legend(fontsize=8.2, loc="upper right", title=f"n = {len(hits)} slope hits",
                    title_fontsize=8.2)
        axis.text(-np.log10(q_threshold) - 0.16, y_top, f"q = {q_threshold:g}  ",
                  fontsize=7.6, color=NULL_COLOUR, ha="right", va="top", rotation=90)
    scatter_axes[0].set_ylabel(r"$-\log_{10}$ q,  within-crispant slope")
    scatter_axes[1].tick_params(labelleft=False)

    # --- right: the same three strata as counts, so the two comparators can be read off ---
    bar = figure.add_subplot(grid[0, 2])
    positions = np.arange(len(comparators))[::-1]
    left = np.zeros(len(comparators))
    for index, (label, colour, _) in enumerate(strata):
        widths = np.array([counts[column][index] for column, _ in comparators], dtype=float)
        bar.barh(positions, widths, left=left, height=0.5, color=colour, edgecolor="white",
                 linewidth=0.8, zorder=3)
        for position, width, offset in zip(positions, widths, left):
            if width >= 6:
                bar.text(offset + width / 2, position, f"{int(width)}", ha="center", va="center",
                         fontsize=9, color="white" if index == 2 else "0.15", zorder=4)
        left += widths
    for position, (column, _) in zip(positions, comparators):
        new = counts[column][1] + counts[column][2]
        bar.text(len(hits) + 3, position, f"NEW = {new}", va="center", fontsize=9.5,
                 fontweight="bold", color=CRISPANT_COLOUR)
    bar.set_yticks(positions)
    bar.set_yticklabels(["Model 1", "Model 2"], fontsize=9.5)
    bar.set_xlim(0, len(hits) * 1.34)
    bar.set_xlabel(f"the {len(hits)} slope hits, split by binary q")
    bar.set_title("How many are genuinely new?", fontsize=10.5)
    bar.spines["left"].set_visible(False)
    bar.tick_params(axis="y", length=0)

    figure.suptitle("The within-crispant slope's hits, seen through the binary contrast",
                    fontsize=12.5, y=1.02)
    return figure, hits


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


def save_inline(figure, path, *, dpi: int = 200) -> Path:
    """Write a PNG WITHOUT closing the figure, so it still renders in a notebook cell.

    ``save`` closes the figure, which suppresses the inline display -- fine for a headless driver
    script, wrong inside a notebook. Use this one there.
    """
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target
