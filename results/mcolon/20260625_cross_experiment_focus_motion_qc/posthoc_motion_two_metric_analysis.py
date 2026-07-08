#!/usr/bin/env python
"""Post-hoc two-threshold motion QC analysis.

Uses completed motion metrics and classifies not-dead snips by two criteria:
  - ncc_p05 < 0.85
  - bad_pair_frac > 0.10

Outputs a summary table, a stacked category plot, a scatter plot, and a
decile gallery with category border colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"

MOTION_CSV = TABLES / "motion_metrics_by_experiment.csv"
NCC_P05_CUT = 0.85
BAD_PAIR_FRAC_CUT = 0.10
N_DECILES = 10
EXAMPLES_PER_EXPERIMENT_DECILE = 6

CATEGORY_ORDER = [
    "pass_both",
    "ncc_only_reject",
    "bad_pair_only_reject",
    "reject_by_both",
]

CATEGORY_LABELS = {
    "pass_both": "pass both",
    "ncc_only_reject": "NCC only",
    "bad_pair_only_reject": "bad-pair only",
    "reject_by_both": "reject by both",
}

CATEGORY_COLORS = {
    "pass_both": "#1a8820",
    "ncc_only_reject": "#b5651d",
    "bad_pair_only_reject": "#1f77b4",
    "reject_by_both": "#7b1fa2",
}


def classify_motion(row: pd.Series) -> str:
    ncc_fail = bool(row["ncc_p05"] < NCC_P05_CUT)
    bad_pair_fail = bool(row["bad_pair_frac"] > BAD_PAIR_FRAC_CUT)
    if ncc_fail and bad_pair_fail:
        return "reject_by_both"
    if ncc_fail:
        return "ncc_only_reject"
    if bad_pair_fail:
        return "bad_pair_only_reject"
    return "pass_both"


def load_not_dead_motion() -> pd.DataFrame:
    df = pd.read_csv(MOTION_CSV)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["ncc_p05", "bad_pair_frac"]).copy()
    df["motion_tile_status"] = df.apply(classify_motion, axis=1)
    df["motion_pass_fail_status"] = np.where(
        df["motion_tile_status"].eq("reject_by_both"),
        "fail_refined",
        "pass_refined",
    )
    return df


def write_summary(df: pd.DataFrame) -> Path:
    rows = []
    for exp, grp in df.groupby("experiment_id", observed=True):
        cats = grp["motion_tile_status"]
        ncc_fail = grp["ncc_p05"] < NCC_P05_CUT
        bad_pair_fail = grp["bad_pair_frac"] > BAD_PAIR_FRAC_CUT
        row = {
            "experiment_id": exp,
            "n_not_dead_motion_rows": int(len(grp)),
            "ncc_p05_threshold": NCC_P05_CUT,
            "bad_pair_frac_threshold": BAD_PAIR_FRAC_CUT,
            "ncc_only_reject_frac": float((cats == "ncc_only_reject").mean()),
            "bad_pair_only_reject_frac": float((cats == "bad_pair_only_reject").mean()),
            "reject_by_both_frac": float((cats == "reject_by_both").mean()),
            "pass_both_frac": float((cats == "pass_both").mean()),
            "any_reject_frac": float((ncc_fail | bad_pair_fail).mean()),
            "refined_reject_frac": float((ncc_fail & bad_pair_fail).mean()),
        }
        for cat in CATEGORY_ORDER:
            row[f"n_{cat}"] = int((cats == cat).sum())
        rows.append(row)

    out = TABLES / "motion_two_metric_threshold_summary_not_dead_snips.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="none", label=CATEGORY_LABELS[cat])
        for cat in CATEGORY_ORDER
    ]


def plot_category_fractions(df: pd.DataFrame) -> Path:
    counts = (
        df.groupby(["experiment_id", "motion_tile_status"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    fracs = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(fracs), dtype=float)
    x = np.arange(len(fracs))
    for cat in CATEGORY_ORDER:
        vals = fracs[cat].to_numpy()
        ax.bar(x, vals * 100, bottom=bottom * 100, color=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(fracs.index.astype(str))
    ax.set_ylabel("% not-dead motion rows")
    ax.set_title("Motion QC two-threshold categories")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIGURES / "motion_two_metric_category_fractions_not_dead_snips.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_scatter(df: pd.DataFrame) -> Path:
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(8.5, 3.2 * len(experiments)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment_id"].astype(str) == exp]
        for cat in CATEGORY_ORDER:
            grp = sub[sub["motion_tile_status"] == cat]
            ax.scatter(
                grp["ncc_p05"],
                grp["bad_pair_frac"],
                s=8,
                alpha=0.45,
                color=CATEGORY_COLORS[cat],
                linewidths=0,
            )
        ax.axvline(NCC_P05_CUT, color="black", linestyle="--", linewidth=1)
        ax.axhline(BAD_PAIR_FRAC_CUT, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{exp}: ncc_p05 vs bad_pair_frac")
        ax.set_ylabel("bad_pair_frac")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("ncc_p05")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES / "motion_two_metric_scatter_not_dead_snips.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, grp in df.groupby("experiment_id", observed=True):
        ordered = grp.sort_values("ncc_p05", ascending=True).reset_index(drop=True)
        ordered["decile"] = pd.qcut(
            ordered["ncc_p05"].rank(method="first"),
            N_DECILES,
            labels=False,
            duplicates="drop",
        ).astype(int) + 1
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def sample_gallery_manifest(df: pd.DataFrame, figure_path: Path) -> pd.DataFrame:
    with_deciles = assign_deciles(df)
    rows = []
    for exp in sorted(with_deciles["experiment_id"].astype(str).unique()):
        for decile in sorted(with_deciles["decile"].unique()):
            grp = with_deciles[
                (with_deciles["experiment_id"].astype(str) == exp)
                & (with_deciles["decile"].astype(int) == int(decile))
            ].sort_values("ncc_p05", ascending=True)
            if len(grp) <= EXAMPLES_PER_EXPERIMENT_DECILE:
                picks = grp
            else:
                picks = grp.iloc[np.linspace(0, len(grp) - 1, EXAMPLES_PER_EXPERIMENT_DECILE).astype(int)]
            for rank, (_, row) in enumerate(picks.iterrows(), start=1):
                rows.append(
                    {
                        "experiment_id": exp,
                        "metric_type": "motion_two_metric",
                        "decile": int(decile),
                        "sample_rank_within_decile": rank,
                        "well_id": row.get("well_id", row.get("well", "")),
                        "time_index": row.get("time_index", np.nan),
                        "image_path": row.get("image_path", ""),
                        "mask_path": row.get("mask_path", ""),
                        "primary_metric": row.get("ncc_p05", np.nan),
                        "bad_pair_frac": row.get("bad_pair_frac", np.nan),
                        "pass_fail_status": row.get("motion_pass_fail_status", ""),
                        "tile_status": row.get("motion_tile_status", ""),
                        "figure_path": str(figure_path),
                        "snip_id": row.get("snip_id", ""),
                        "image_id": row.get("image_id", ""),
                    }
                )
    return pd.DataFrame(rows)


def plot_gallery(manifest: pd.DataFrame, out_path: Path) -> None:
    experiments = sorted(manifest["experiment_id"].astype(str).unique())
    deciles = sorted(manifest["decile"].astype(int).unique())
    block_rows = 2
    block_cols = 3
    ncols = len(experiments) * block_cols
    nrows = len(deciles) * block_rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for d_i, decile in enumerate(deciles):
        for e_i, exp in enumerate(experiments):
            grp = manifest[
                (manifest["decile"].astype(int) == decile)
                & (manifest["experiment_id"].astype(str) == exp)
            ].sort_values("sample_rank_within_decile")
            for j, (_, row) in enumerate(grp.head(block_rows * block_cols).iterrows()):
                rr = d_i * block_rows + j // block_cols
                cc = e_i * block_cols + j % block_cols
                ax = axes[rr, cc]
                ax.axis("on")
                image_path = Path(str(row.get("image_path", "")))
                if image_path.exists():
                    ax.imshow(np.array(Image.open(image_path).convert("L")), cmap="gray")
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
                cat = str(row.get("tile_status", "pass_both"))
                for spine in ax.spines.values():
                    spine.set_color(CATEGORY_COLORS.get(cat, "#444444"))
                    spine.set_linewidth(3.0 if cat == "reject_by_both" else 2.2)
                ax.set_title(
                    f"{exp} {row.get('well_id', '')} t{row.get('time_index', '')}\n"
                    f"p05={row.get('primary_metric', np.nan):.3g} bad={row.get('bad_pair_frac', np.nan):.2g}",
                    fontsize=5.5,
                )
            axes[d_i * block_rows, e_i * block_cols].set_ylabel(f"D{decile}", fontsize=8)

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Motion deciles by experiment: ncc_p05, colored by two-threshold status",
        fontsize=13,
        y=0.982,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_not_dead_motion()
    summary_path = write_summary(df)
    fractions_path = plot_category_fractions(df)
    scatter_path = plot_scatter(df)
    gallery_path = FIGURES / "motion_deciles_by_experiment_ncc_p05_two_metric_status.png"
    manifest = sample_gallery_manifest(df, gallery_path)
    manifest_path = TABLES / "motion_two_metric_decile_gallery_samples.csv"
    manifest.to_csv(manifest_path, index=False)
    plot_gallery(manifest, gallery_path)

    print(f"Saved summary -> {summary_path}")
    print(f"Saved category fractions -> {fractions_path}")
    print(f"Saved scatter -> {scatter_path}")
    print(f"Saved gallery manifest -> {manifest_path}")
    print(f"Saved gallery -> {gallery_path}")


if __name__ == "__main__":
    main()
