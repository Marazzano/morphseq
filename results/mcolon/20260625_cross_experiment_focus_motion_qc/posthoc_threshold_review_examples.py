#!/usr/bin/env python
"""Generate threshold review example figures from completed QC metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "review_examples"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
MOTION_CSV = TABLES / "motion_metrics_by_experiment.csv"

RANDOM_SEED = 1729
N_REVIEW = 10

FOCUS_ENTROPY_CUT = -0.55
FOCUS_SHARPNESS_RATIO_CUT = 1.5
MOTION_NCC_P05_CUT = 0.85
MOTION_BAD_PAIR_FRAC_CUT = 0.10

FOCUS_COLORS = {
    "accepted": "#1a8820",
    "entropy_only_reject": "#b5651d",
    "sharpness_only_reject": "#1f77b4",
    "rejected_by_both": "#7b1fa2",
}

MOTION_COLORS = {
    "pass_both": "#1a8820",
    "ncc_only_reject": "#b5651d",
    "bad_pair_only_reject": "#1f77b4",
    "reject_by_both": "#7b1fa2",
}

MOTION_LABELS = {
    "pass_both": "pass both",
    "ncc_only_reject": "NCC only",
    "bad_pair_only_reject": "bad-pair only",
    "reject_by_both": "reject by both",
}


def focus_status(row: pd.Series) -> str:
    entropy_fail = bool(row["ff_rel_entropy"] < FOCUS_ENTROPY_CUT)
    sharp_fail = bool(row["ff_lap_abs_ratio"] < FOCUS_SHARPNESS_RATIO_CUT)
    if entropy_fail and sharp_fail:
        return "rejected_by_both"
    if entropy_fail:
        return "entropy_only_reject"
    if sharp_fail:
        return "sharpness_only_reject"
    return "accepted"


def motion_status(row: pd.Series) -> str:
    ncc_fail = bool(row["ncc_p05"] < MOTION_NCC_P05_CUT)
    bad_pair_fail = bool(row["bad_pair_frac"] > MOTION_BAD_PAIR_FRAC_CUT)
    if ncc_fail and bad_pair_fail:
        return "reject_by_both"
    if ncc_fail:
        return "ncc_only_reject"
    if bad_pair_fail:
        return "bad_pair_only_reject"
    return "pass_both"


def sample_n(df: pd.DataFrame, n: int, seed_offset: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=RANDOM_SEED + seed_offset).copy()


def read_image(path: object) -> np.ndarray | None:
    p = Path(str(path))
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L"))


def read_mask(path: object, shape: tuple[int, int]) -> np.ndarray | None:
    p = Path(str(path))
    if not p.exists():
        return None
    mask = np.array(Image.open(p)).astype(bool)
    if mask.shape != shape:
        return None
    return mask


def mask_bbox(mask: np.ndarray, pad: int = 60) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def draw_focus_review(samples: pd.DataFrame, out_path: Path, title: str) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(11, max(2.2 * n, 3.0)), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, (_, row) in enumerate(samples.iterrows()):
        img = read_image(row["image_path"])
        if img is None:
            axes[r, 0].text(0.5, 0.5, "missing image", ha="center", va="center")
            axes[r, 1].text(0.5, 0.5, "missing image", ha="center", va="center")
            continue
        mask = read_mask(row["mask_path"], img.shape)
        status = str(row["focus_tile_status"])
        color = FOCUS_COLORS.get(status, "#444444")

        axes[r, 0].axis("on")
        axes[r, 0].imshow(img, cmap="gray")
        if mask is not None:
            y0, y1, x0, x1 = mask_bbox(mask)
            axes[r, 0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, lw=1.8)
        axes[r, 0].set_title(
            f"{row['experiment_id']} {row.get('well_id', '')} t{row.get('time_index', '')} | {status}\n"
            f"rel_ent={row['ff_rel_entropy']:.3f} cut<{FOCUS_ENTROPY_CUT}; "
            f"lap_ratio={row['ff_lap_abs_ratio']:.3f} cut<{FOCUS_SHARPNESS_RATIO_CUT}",
            fontsize=8,
        )

        axes[r, 1].axis("on")
        if mask is not None:
            y0, y1, x0, x1 = mask_bbox(mask)
            axes[r, 1].imshow(img[y0:y1, x0:x1], cmap="gray")
            axes[r, 1].contour(mask[y0:y1, x0:x1], levels=[0.5], colors=[color], linewidths=1.2)
        else:
            axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].set_title("embryo crop + mask", fontsize=8)
        for ax in axes[r]:
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.4)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in FOCUS_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle(title, fontsize=13, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_focus_reviews() -> pd.DataFrame:
    df = pd.read_csv(FOCUS_CSV)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    df["focus_tile_status"] = df.apply(focus_status, axis=1)
    df["focus_pass_fail_status"] = np.where(df["focus_tile_status"].eq("accepted"), "pass", "reject")

    manifest_rows = []
    seed_offset = 0
    for exp, grp in df.groupby("experiment_id", observed=True):
        exp_s = str(exp)
        rejected = grp[grp["focus_pass_fail_status"] == "reject"].copy()
        accepted = grp[grp["focus_pass_fail_status"] == "pass"].copy()
        both = grp[grp["focus_tile_status"] == "rejected_by_both"].copy()
        cohorts = {
            "worst_any_reject": rejected.sort_values("ff_rel_entropy", ascending=True).head(N_REVIEW),
            "random_any_reject": sample_n(rejected, N_REVIEW, seed_offset),
            "random_pass": sample_n(accepted, N_REVIEW, seed_offset + 100),
            "random_reject_by_both": sample_n(both, N_REVIEW, seed_offset + 200),
        }
        seed_offset += 1
        for cohort, samples in cohorts.items():
            if samples.empty:
                continue
            out = FIGURES / f"focus_{exp_s}_{cohort}_highres.png"
            draw_focus_review(
                samples,
                out,
                f"Focus review: {exp_s} {cohort.replace('_', ' ')}",
            )
            tmp = samples.copy()
            tmp["review_type"] = cohort
            tmp["figure_path"] = str(out)
            manifest_rows.append(tmp)

    if not manifest_rows:
        return pd.DataFrame()
    manifest = pd.concat(manifest_rows, ignore_index=True)
    keep = [
        "experiment_id",
        "review_type",
        "well_id",
        "time_index",
        "snip_id",
        "image_id",
        "image_path",
        "mask_path",
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "focus_tile_status",
        "focus_pass_fail_status",
        "figure_path",
    ]
    return manifest[[c for c in keep if c in manifest.columns]]


def draw_motion_review(samples_by_category: dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    categories = ["ncc_only_reject", "bad_pair_only_reject", "pass_both", "reject_by_both"]
    ncols = len(categories)
    nrows = N_REVIEW
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 1.85), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for c, cat in enumerate(categories):
        grp = samples_by_category.get(cat, pd.DataFrame())
        axes[0, c].set_title(MOTION_LABELS[cat], fontsize=9)
        for r in range(nrows):
            ax = axes[r, c]
            if r >= len(grp):
                if r == 0 and grp.empty:
                    ax.text(0.5, 0.5, "none", ha="center", va="center", fontsize=8)
                continue
            row = grp.iloc[r]
            img = read_image(row["image_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img, cmap="gray")
            ax.axis("on")
            color = MOTION_COLORS[cat]
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.0)
            ax.set_title(
                f"{row.get('well_id', '')} t{row.get('time_index', '')}\n"
                f"p05={row['ncc_p05']:.3g} bad={row['bad_pair_frac']:.2g}",
                fontsize=5.8,
            )

    handles = [
        Patch(facecolor=MOTION_COLORS[cat], edgecolor="none", label=MOTION_LABELS[cat])
        for cat in categories
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle(title, fontsize=13, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_motion_reviews() -> pd.DataFrame:
    df = pd.read_csv(MOTION_CSV)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["ncc_p05", "bad_pair_frac"]).copy()
    df["motion_tile_status"] = df.apply(motion_status, axis=1)

    manifest_rows = []
    seed_offset = 1000
    for exp, grp in df.groupby("experiment_id", observed=True):
        exp_s = str(exp)
        samples_by_category = {}
        for cat in ["ncc_only_reject", "bad_pair_only_reject", "pass_both", "reject_by_both"]:
            sub = grp[grp["motion_tile_status"] == cat].copy()
            samples_by_category[cat] = sample_n(sub, N_REVIEW, seed_offset)
            seed_offset += 1
        out = FIGURES / f"motion_{exp_s}_two_metric_category_examples.png"
        draw_motion_review(
            samples_by_category,
            out,
            f"Motion review: {exp_s} | NCC p05<{MOTION_NCC_P05_CUT}, bad-pair>{MOTION_BAD_PAIR_FRAC_CUT}",
        )
        for cat, samples in samples_by_category.items():
            if samples.empty:
                continue
            tmp = samples.copy()
            tmp["review_type"] = cat
            tmp["figure_path"] = str(out)
            manifest_rows.append(tmp)

    if not manifest_rows:
        return pd.DataFrame()
    manifest = pd.concat(manifest_rows, ignore_index=True)
    keep = [
        "experiment_id",
        "review_type",
        "well_id",
        "time_index",
        "snip_id",
        "image_id",
        "image_path",
        "mask_path",
        "ncc_p05",
        "bad_pair_frac",
        "ncc_min",
        "motion_tile_status",
        "figure_path",
    ]
    return manifest[[c for c in keep if c in manifest.columns]]


def main() -> None:
    focus_manifest = build_focus_reviews()
    motion_manifest = build_motion_reviews()

    focus_out = TABLES / "focus_threshold_review_examples_manifest.csv"
    motion_out = TABLES / "motion_threshold_review_examples_manifest.csv"
    focus_manifest.to_csv(focus_out, index=False)
    motion_manifest.to_csv(motion_out, index=False)

    print(f"Saved focus manifest -> {focus_out} ({len(focus_manifest)} rows)")
    print(f"Saved motion manifest -> {motion_out} ({len(motion_manifest)} rows)")
    print(f"Saved review figures under -> {FIGURES}")


if __name__ == "__main__":
    main()
