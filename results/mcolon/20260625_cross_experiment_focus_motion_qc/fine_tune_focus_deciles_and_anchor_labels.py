#!/usr/bin/env python
"""Fine-tuning review outputs for focus Sobel sweep.

Applies manual label corrections for the review session and builds bottom-half decile galleries for
the local_context, Sobel > 0.02 metric.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, measure


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FT = HERE / "fine_tuning"
FT_TABLES = FT / "tables"
FT_FIGURES = FT / "figures"
ANCHOR_GROUP_DIR = FT_FIGURES / "anchor_groups_highres_corrected"
DECILE_DIR = FT_FIGURES / "local_context_grad002_bottom_half_deciles"

ANCHOR_CSV = TABLES / "anchor_interior_structure_metrics.csv"
SWEEP_CSV = FT_TABLES / "focus_sobel_threshold_sweep_metrics.csv"
POP_CSV = TABLES / "interior_structure_cross_experiment_metrics.csv"

TARGET_MODE = "local_context"
TARGET_SOBEL_THRESHOLD = 0.02
N_DECILES = 10
N_PER_DECILE = 20


LABEL_CORRECTIONS = [
    {
        "experiment_id": "20251125",
        "well_id": "A10",
        "time_index": 224,
        "old_label": "fail_blur_anchor",
        "new_label": "fail_blur_ghost",
        "reason": "manual review 2026-06-30: treat with ghost/structureless failures",
    },
    {
        "experiment_id": "20250912",
        "well_id": "A04",
        "time_index": 100,
        "old_label": "fail_blur_ghost",
        "new_label": "head_only_out_of_focus",
        "reason": "manual review 2026-06-30: not whole-embryo ghost; head-only/partial focus issue",
    },
    {
        "experiment_id": "20251125",
        "well_id": "H04",
        "time_index": 82,
        "old_label": "pass_dorsal_bright_review",
        "new_label": "pass_in_focus",
        "reason": "manual review 2026-06-30: completely fine",
    },
]


COLORS = {
    "fail_blur_ghost": "#8b0000",
    "head_only_out_of_focus": "#c2185b",
    "motion_and_partial_blur": "#e377c2",
    "fail_blur_anchor": "#d62728",
    "fail_blur_candidate": "#d62728",
    "fail_blur_caught": "#d62728",
    "dorsal_bright_review": "#ff7f0e",
    "pass_dorsal_bright_review": "#ff7f0e",
    "gold_dorsal_in_focus": "#1f77b4",
    "gold_in_focus": "#1a8820",
    "pass_in_focus": "#1a8820",
    "pass_anchor": "#1a8820",
    "pass_after_blur_anchor": "#1a8820",
    "pass_or_gray_anchor": "#7f7f7f",
    "dead_negative_control": "#000000",
    "saturation_fail_anchor": "#9467bd",
}

LABEL_ORDER = [
    "fail_blur_ghost",
    "head_only_out_of_focus",
    "fail_blur_anchor",
    "fail_blur_candidate",
    "fail_blur_caught",
    "motion_and_partial_blur",
    "dorsal_bright_review",
    "pass_dorsal_bright_review",
    "gold_dorsal_in_focus",
    "gold_in_focus",
    "pass_in_focus",
    "pass_anchor",
    "pass_after_blur_anchor",
    "pass_or_gray_anchor",
    "dead_negative_control",
    "saturation_fail_anchor",
]


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["experiment_id"] = out["experiment_id"].astype(str)
    out["well_id"] = out["well_id"].astype(str).str.upper()
    out["time_index"] = pd.to_numeric(out["time_index"], errors="coerce").astype("Int64")
    return out


def apply_label_corrections(anchor: pd.DataFrame) -> pd.DataFrame:
    out = normalize_id_columns(anchor)
    out["original_user_label"] = out["user_label"].astype(str)
    out["corrected_user_label"] = out["user_label"].astype(str)
    out["label_correction_reason"] = ""

    corrections = pd.DataFrame(LABEL_CORRECTIONS)
    corrections.to_csv(FT_TABLES / "anchor_label_corrections.csv", index=False)

    for corr in LABEL_CORRECTIONS:
        m = (
            (out["experiment_id"] == corr["experiment_id"])
            & (out["well_id"] == corr["well_id"])
            & (out["time_index"] == corr["time_index"])
        )
        if not m.any():
            raise ValueError(f"label correction matched no rows: {corr}")
        out.loc[m, "corrected_user_label"] = corr["new_label"]
        out.loc[m, "label_correction_reason"] = corr["reason"]

    out["user_label"] = out["corrected_user_label"]
    out.to_csv(FT_TABLES / "anchor_interior_structure_metrics_corrected_labels.csv", index=False)
    return out


def crop_image_and_mask(image_path: str, mask_path: str, pad: int = 80) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(mask_path)).astype(bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return img, mask
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def display_image(img: np.ndarray) -> np.ndarray:
    vals = img[np.isfinite(img)]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo:
        return np.clip(img, 0, 255).astype(np.uint8)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 1))


def draw_crop(ax, image_path: str, mask_path: str, *, pad: int = 80) -> None:
    img, mask = crop_image_and_mask(image_path, mask_path, pad=pad)
    ax.imshow(display_image(img), cmap="gray", interpolation="nearest")
    for contour in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="#00e5ff", linewidth=0.75)
    ax.axis("off")


def corrected_anchor_gallery(anchor: pd.DataFrame) -> None:
    ANCHOR_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SWEEP_CSV, low_memory=False)
    metrics = normalize_id_columns(metrics)
    metric = metrics[
        (metrics["source"] == "anchor")
        & (metrics["normalization_mode"] == "crop_global")
        & (metrics["sobel_threshold"] == 0.04)
    ][["experiment_id", "well_id", "time_index", "interior_strong_edge_frac"]]
    metric = metric.rename(columns={"interior_strong_edge_frac": "crop_global_edge_frac"})
    anchor = normalize_id_columns(anchor).merge(metric, on=["experiment_id", "well_id", "time_index"], how="left")

    groups: list[tuple[str, pd.DataFrame]] = []
    for label in LABEL_ORDER:
        g = anchor[anchor["user_label"] == label].sort_values(["experiment_id", "well_id", "time_index"])
        if not g.empty:
            groups.append((label, g.reset_index(drop=True)))

    max_cols = min(max(len(g) for _, g in groups), 6)
    fig, axes = plt.subplots(len(groups), max_cols, figsize=(3.25 * max_cols + 2.2, 2.75 * len(groups)), squeeze=False)
    for r, (label, g) in enumerate(groups):
        for c in range(max_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(g):
                continue
            row = g.iloc[c]
            draw_crop(ax, row["image_path"], row["mask_path"])
            title = f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}\n"
            title += f"crop={row.get('crop_global_edge_frac', np.nan):.3f} local={row.get('interior_strong_edge_frac', np.nan):.3f}"
            ax.set_title(title, fontsize=7, color=COLORS.get(label, "black"))
        axes[r, 0].text(
            -0.08,
            0.5,
            label.replace("_", " "),
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS.get(label, "black"),
        )
    fig.suptitle("Corrected focus QC anchor review gallery", fontsize=13)
    fig.tight_layout(rect=[0.08, 0.01, 1, 0.985])
    fig.savefig(FT_FIGURES / "focus_anchor_highres_review_gallery_by_group_corrected.png", dpi=220)
    plt.close(fig)

    for label, g in groups:
        cols = min(len(g), 6)
        rows = math.ceil(len(g) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (_, row) in enumerate(g.iterrows()):
            ax = axes[i // cols, i % cols]
            draw_crop(ax, row["image_path"], row["mask_path"])
            ax.set_title(
                f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}\n"
                f"crop={row.get('crop_global_edge_frac', np.nan):.3f} local={row.get('interior_strong_edge_frac', np.nan):.3f}",
                fontsize=8,
            )
        fig.suptitle(label.replace("_", " "), color=COLORS.get(label, "black"), fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(ANCHOR_GROUP_DIR / f"{label}.png", dpi=240)
        plt.close(fig)


def build_decile_galleries() -> None:
    DECILE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SWEEP_CSV, low_memory=False)
    use = metrics[
        (metrics["source"] == "population")
        & (metrics["normalization_mode"] == TARGET_MODE)
        & (metrics["sobel_threshold"] == TARGET_SOBEL_THRESHOLD)
    ].copy()
    paths = pd.read_csv(
        POP_CSV,
        usecols=["experiment_id", "well_id", "time_index", "snip_id", "image_path", "mask_path"],
        low_memory=False,
    )
    paths = normalize_id_columns(paths)
    use = normalize_id_columns(use)
    use = use.merge(
        paths,
        on=["experiment_id", "well_id", "time_index", "snip_id"],
        how="left",
    )
    use = use.dropna(subset=["interior_strong_edge_frac", "image_path", "mask_path"])
    median_value = float(use["interior_strong_edge_frac"].median())
    bottom = use[use["interior_strong_edge_frac"] <= median_value].copy()
    bottom["bottom_half_decile"] = (
        pd.qcut(bottom["interior_strong_edge_frac"].rank(method="first"), N_DECILES, labels=False) + 1
    )

    bottom.to_csv(FT_TABLES / "local_context_grad002_bottom_half_decile_membership.csv", index=False)
    pd.DataFrame(
        [{"normalization_mode": TARGET_MODE, "sobel_threshold": TARGET_SOBEL_THRESHOLD, "median_value": median_value, "n_bottom_half": len(bottom)}]
    ).to_csv(FT_TABLES / "local_context_grad002_bottom_half_decile_summary.csv", index=False)

    build_one_decile_gallery(bottom, DECILE_DIR / "ALL_local_context_grad002_bottom_half_deciles.png", "ALL", median_value)
    for exp in sorted(bottom["experiment_id"].unique()):
        sub = bottom[bottom["experiment_id"] == exp].copy()
        sub["bottom_half_decile"] = (
            pd.qcut(sub["interior_strong_edge_frac"].rank(method="first"), N_DECILES, labels=False) + 1
        )
        build_one_decile_gallery(
            sub,
            DECILE_DIR / f"{exp}_local_context_grad002_bottom_half_deciles.png",
            str(exp),
            float(use[use["experiment_id"] == exp]["interior_strong_edge_frac"].median()),
        )


def crop_for_decile(image_path: str, mask_path: str, pad: int = 60, thumb: int = 220) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        img, mask = crop_image_and_mask(image_path, mask_path, pad=pad)
    except Exception:
        return None, None
    h, w = img.shape
    scale = max(h, w) / thumb
    if scale > 1:
        new_size = (max(1, int(w / scale)), max(1, int(h / scale)))
        img = np.array(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize(new_size, Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize(new_size, Image.NEAREST)).astype(bool)
    return img, mask


def build_one_decile_gallery(df: pd.DataFrame, out: Path, title_prefix: str, median_value: float) -> None:
    fig, axes = plt.subplots(N_DECILES, N_PER_DECILE, figsize=(N_PER_DECILE * 1.45, N_DECILES * 1.55), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    rng = np.random.default_rng(1729)
    manifest_rows = []
    for r, decile in enumerate(range(1, N_DECILES + 1)):
        grp = df[df["bottom_half_decile"] == decile].sort_values("interior_strong_edge_frac")
        if len(grp) > N_PER_DECILE:
            # Spread examples across the decile, not a random clump.
            idx = np.linspace(0, len(grp) - 1, N_PER_DECILE).round().astype(int)
            grp = grp.iloc[idx]
        elif len(grp) > 0:
            grp = grp.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).sort_values("interior_strong_edge_frac")
        for c, (_, row) in enumerate(grp.iterrows()):
            ax = axes[r, c]
            img, mask = crop_for_decile(row["image_path"], row["mask_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
            else:
                ax.imshow(display_image(img), cmap="gray", interpolation="nearest")
                if mask is not None and mask.any():
                    ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.4)
            ax.set_title(
                f"{row['interior_strong_edge_frac']:.3f}\n{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}",
                fontsize=5,
            )
            manifest_rows.append(row.to_dict())
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=9)

    fig.suptitle(
        f"{title_prefix}: local_context Sobel > 0.02, bottom-half deciles only\n"
        f"D1 lowest edge fraction -> D10 near median; median={median_value:.3f}; {N_PER_DECILE}/decile",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    pd.DataFrame(manifest_rows).to_csv(out.with_suffix(".manifest.csv"), index=False)
    print(f"Saved -> {out}", flush=True)


def main() -> None:
    FT_TABLES.mkdir(parents=True, exist_ok=True)
    FT_FIGURES.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(ANCHOR_CSV, low_memory=False)
    anchor = anchor[anchor["struct_error"].fillna("") == ""].copy()
    corrected = apply_label_corrections(anchor)
    corrected_anchor_gallery(corrected)
    build_decile_galleries()


if __name__ == "__main__":
    main()
