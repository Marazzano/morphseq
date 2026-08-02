"""Figures for the label-free discovery analysis.

Shared grammar with the supervised figures: crimson = the analysis arm, steel blue = the supervised
reference, grey = neither. Every panel that reports a recovery fraction also shows what was *missed*,
because a recall number without its denominator is unreadable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt

RECOVERED = "#C0392B"
MISSED = "#D8CFCB"
EXTRA = "#E8A33D"
REFERENCE = "#3B6EA5"
NULL_COLOUR = "0.55"

ARM_LABEL = {"pooled_pc1": "pooled PC1  (no labels at all)",
             "within_pc1": "within-crispant PC1  (crispants only)"}


def use_house_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.3", "axes.linewidth": 0.9, "xtick.color": "0.3",
        "ytick.color": "0.3", "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 160, "legend.frameon": False,
    })


def hit_sets(coefficients: pd.DataFrame, arm: str, q: float = 0.10) -> dict:
    block = coefficients[coefficients["arm"] == arm]
    return {contrast: set(group.loc[group["q_value"] < q, "cell_type"])
            for contrast, group in block.groupby("contrast")}


def recovery_table(unsupervised: pd.DataFrame, supervised: pd.DataFrame, *,
                   arm: str, reference_arm: str = "binary_z", q: float = 0.10) -> pd.DataFrame:
    """Per-contrast recovery of the reference arm's cell types by an unsupervised arm."""
    found = hit_sets(unsupervised, arm, q)
    reference = hit_sets(supervised, reference_arm, q)
    rows = []
    for contrast, reference_set in reference.items():
        got = found.get(contrast, set())
        rows.append({
            "contrast": contrast,
            "reference_hits": len(reference_set),
            "recovered": len(got & reference_set),
            "missed": len(reference_set - got),
            "extra": len(got - reference_set),
            "found_total": len(got),
            "recall": len(got & reference_set) / len(reference_set) if reference_set else np.nan,
            "precision": len(got & reference_set) / len(got) if got else np.nan,
        })
    return pd.DataFrame(rows)


def recovery_waterfall(table: pd.DataFrame, axes_diagnostics: pd.DataFrame, *,
                       arm: str, figsize=(12.5, 6.4)):
    """Per contrast: how much of the supervised result a label-free axis picks up.

    Bars are the reference (binary-contrast) hits, split into recovered and missed, with the
    unsupervised arm's false-positive calls drawn to the left of zero so that both kinds of error
    are visible at once. Sorted by recall.

    Marks on the left edge record whether that contrast had (a) a PC1 significantly more
    concentrated than a column-shuffle null and (b) a validated supervised axis — the two
    preconditions under which recovery could plausibly work at all.
    """
    diagnostics = axes_diagnostics[axes_diagnostics["axis"] == arm].set_index("contrast")
    frame = table.merge(
        diagnostics[["axis_significant", "usable", "angle_to_lda", "variance_ratio"]],
        left_on="contrast", right_index=True, how="left",
    ).sort_values(["recall", "reference_hits"], ascending=False).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=figsize)
    positions = np.arange(len(frame))
    axis.barh(positions, frame["recovered"], color=RECOVERED, zorder=3, label="recovered")
    axis.barh(positions, frame["missed"], left=frame["recovered"], color=MISSED, zorder=3,
              label="missed by the unsupervised axis")
    axis.barh(positions, -frame["extra"], color=EXTRA, zorder=3,
              label="found but not in the reference")
    axis.axvline(0, color="0.3", linewidth=1.0, zorder=4)

    labels = []
    for row in frame.itertuples():
        flags = ("P" if row.axis_significant else "·") + ("S" if row.usable else "·")
        labels.append(f"[{flags}] {row.contrast}")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=6.8)
    axis.invert_yaxis()
    axis.set_xlabel("cell types at q < 0.10   (left of zero: called but not in the reference)")
    total_recall = frame["recovered"].sum() / max(frame["reference_hits"].sum(), 1)
    axis.set_title(
        f"{ARM_LABEL.get(arm, arm)} — recovery of the binary contrast\n"
        f"overall recall {frame['recovered'].sum()}/{frame['reference_hits'].sum()} "
        f"= {total_recall:.1%}    "
        f"[P] PC1 beats its null   [S] supervised axis validated",
        fontsize=11,
    )
    axis.legend(fontsize=8.5, loc="lower right")
    figure.tight_layout()
    return figure, frame


def recall_vs_alignment(table: pd.DataFrame, axes_diagnostics: pd.DataFrame, *,
                        arm: str, figsize=(11.5, 4.6)):
    """Why recovery varies: how close the unsupervised axis landed to the supervised one.

    PC1 has no reason to point at the perturbation. Where it happens to, recovery should be high;
    where it is orthogonal, the perturbation is invisible to it however strong the phenotype. The
    dotted line is the angle two arbitrary directions in this 5D space would average.
    """
    diagnostics = axes_diagnostics[axes_diagnostics["axis"] == arm].set_index("contrast")
    frame = table.merge(
        diagnostics[["angle_to_lda", "variance_ratio", "null_ratio_p95", "axis_significant",
                     "usable", "random_angle_median", "loo_auc"]],
        left_on="contrast", right_index=True, how="left",
    ).dropna(subset=["angle_to_lda", "recall"])

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    random_median = float(frame["random_angle_median"].iloc[0])

    axes[0].scatter(frame["angle_to_lda"], frame["recall"],
                    s=28 + 3.2 * frame["reference_hits"],
                    c=[RECOVERED if u else "0.65" for u in frame["usable"]],
                    edgecolor="0.25", linewidth=0.7, zorder=3)
    axes[0].axvline(random_median, color=NULL_COLOUR, linestyle=":", linewidth=1.3, zorder=1)
    axes[0].text(random_median + 1.2, 0.02, "two random directions", rotation=90, fontsize=8,
                 color=NULL_COLOUR, va="bottom")
    from scipy import stats as sps
    r = sps.spearmanr(frame["angle_to_lda"], frame["recall"])
    axes[0].set_xlabel("angle between PC1 and the supervised discriminant (deg)")
    axes[0].set_ylabel("recall of that contrast's binary hits")
    axes[0].set_title(f"Alignment predicts recovery\nSpearman {r.statistic:+.2f} "
                      f"(p = {r.pvalue:.3f});  marker size = reference hits", fontsize=10.5)

    axes[1].scatter(frame["variance_ratio"], frame["recall"],
                    s=28 + 3.2 * frame["reference_hits"],
                    c=[RECOVERED if s else "0.65" for s in frame["axis_significant"]],
                    edgecolor="0.25", linewidth=0.7, zorder=3)
    axes[1].plot(frame["null_ratio_p95"], frame["recall"], "k|", markersize=7, zorder=4,
                 label="column-shuffle null (95th pct)")
    axes[1].set_xlabel("PC1 explained-variance ratio")
    axes[1].set_title("Axis concentration does not predict recovery\n"
                      "(a dominant axis need not be the right one)", fontsize=10.5)
    axes[1].legend(fontsize=8.5, loc="upper left")
    figure.tight_layout()
    return figure


def direction_scatter(unsupervised: pd.DataFrame, supervised: pd.DataFrame, *,
                      arm: str, q: float = 0.10, figsize=(6.6, 6.2)):
    """Effect sizes from the label-free axis against the supervised binary contrast.

    An eigenvector has no sign, so each contrast is allowed **one** global flip — chosen to maximise
    agreement. That single bit is the only label information entering this panel, and it is why the
    panel speaks to *consistency of direction*, not to whether the direction could be predicted
    ab initio.
    """
    merged = (unsupervised[unsupervised["arm"] == arm][["contrast", "cell_type", "logFC",
                                                        "q_value"]]
              .merge(supervised[supervised["arm"] == "binary_z"][["contrast", "cell_type",
                                                                  "logFC", "q_value"]],
                     on=["contrast", "cell_type"], suffixes=("_unsup", "_binary")))

    flipped = []
    for _, group in merged.groupby("contrast"):
        significant = group[group["q_value_unsup"] < q]
        sign = 1.0
        if len(significant) >= 3:
            agreement = (np.sign(significant["logFC_unsup"])
                         == np.sign(significant["logFC_binary"])).mean()
            sign = 1.0 if agreement >= 0.5 else -1.0
        block = group.copy()
        block["logFC_unsup"] = block["logFC_unsup"] * sign
        flipped.append(block)
    merged = pd.concat(flipped, ignore_index=True)

    both = (merged["q_value_unsup"] < q) & (merged["q_value_binary"] < q)
    only_unsup = (merged["q_value_unsup"] < q) & ~(merged["q_value_binary"] < q)
    neither = ~(merged["q_value_unsup"] < q) & ~(merged["q_value_binary"] < q)

    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(merged.loc[neither, "logFC_binary"], merged.loc[neither, "logFC_unsup"],
                 s=7, color="0.86", edgecolor="none", zorder=2)
    axis.scatter(merged.loc[only_unsup, "logFC_binary"], merged.loc[only_unsup, "logFC_unsup"],
                 s=30, color=EXTRA, edgecolor="0.3", linewidth=0.4, zorder=3,
                 label=f"unsupervised only ({int(only_unsup.sum())})")
    axis.scatter(merged.loc[both, "logFC_binary"], merged.loc[both, "logFC_unsup"],
                 s=38, color=RECOVERED, edgecolor="0.2", linewidth=0.5, zorder=4,
                 label=f"both ({int(both.sum())})")

    limit = float(np.nanpercentile(np.abs(np.r_[merged["logFC_binary"],
                                                merged["logFC_unsup"]]), 99.5)) * 1.1
    axis.plot([-limit, limit], [-limit, limit], color=NULL_COLOUR, linestyle="--", linewidth=1.2,
              zorder=1)
    axis.axhline(0, color="0.88", linewidth=0.8, zorder=1)
    axis.axvline(0, color="0.88", linewidth=0.8, zorder=1)
    axis.set_xlim(-limit, limit); axis.set_ylim(-limit, limit)
    axis.set_xlabel("log fold change — supervised binary contrast")
    axis.set_ylabel(f"log fold change — {ARM_LABEL.get(arm, arm)}\n(one global sign flip per contrast)")

    from scipy import stats as sps
    subset = merged[both]
    if len(subset) > 3:
        r = sps.spearmanr(subset["logFC_binary"], subset["logFC_unsup"])
        agreement = (np.sign(subset["logFC_binary"]) == np.sign(subset["logFC_unsup"])).mean()
        axis.set_title(f"Do label-free effects point the same way?\n"
                       f"among shared hits: Spearman {r.statistic:+.2f}, "
                       f"sign agreement {agreement:.0%}", fontsize=11)
    axis.legend(fontsize=9, loc="upper left")
    figure.tight_layout()
    return figure


def recovery_summary_bars(tables: dict, *, figsize=(9.5, 4.8)):
    """Headline recovery for each arm against each reference, with denominators shown."""
    figure, axis = plt.subplots(figsize=figsize)
    labels, recovered, missed, extra = [], [], [], []
    for label, table in tables.items():
        labels.append(label)
        recovered.append(table["recovered"].sum())
        missed.append(table["missed"].sum())
        extra.append(table["extra"].sum())

    positions = np.arange(len(labels))
    axis.bar(positions, recovered, color=RECOVERED, zorder=3, label="recovered")
    axis.bar(positions, missed, bottom=recovered, color=MISSED, zorder=3, label="missed")
    axis.bar(positions, [-value for value in extra], color=EXTRA, zorder=3,
             label="found, not in reference")
    for index, (r, m) in enumerate(zip(recovered, missed)):
        axis.text(index, r + m + 8, f"{r}/{r + m}\n{r / (r + m):.0%}", ha="center", fontsize=9.5)
    axis.axhline(0, color="0.3", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylabel("cell types at q < 0.10")
    axis.set_title("How much of the supervised result survives without labels?", fontsize=11.5)
    axis.legend(fontsize=9)
    figure.tight_layout()
    return figure


def save(figure, path) -> Path:
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return target
