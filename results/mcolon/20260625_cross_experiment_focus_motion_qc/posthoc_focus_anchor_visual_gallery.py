#!/usr/bin/env python
"""High-res per-anchor visual gallery: one row per anchor for eyeball QC.

Purpose: let a human directly inspect whether the dorsal-bright reviews actually
carry sharp fine structure (information present, low contrast) or are genuinely
smeared like the blur fails (information erased). Scalar metrics conflate these;
the eye does not. Each row shows raw / locally-normalized / zoomed-center crops
so brightness and resolution can be judged independently.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "sobel_feature_size"
FIGURES.mkdir(parents=True, exist_ok=True)

ANCHORS = TABLES / "focus_anchor_registry_metrics.csv"
OUT = FIGURES / "focus_anchor_highres_rows.png"

# Sort order: blur fails first, then dead/saturation, then dorsal-bright, then passes.
ORDER = {
    "fail_blur_anchor": 1,
    "fail_blur_candidate": 1,
    "fail_blur_caught": 1,
    "dead_negative_control": 2,
    "saturation_fail_anchor": 2,
    "dorsal_bright_review": 3,
    "pass_dorsal_bright_review": 3,
    "pass_or_gray_anchor": 4,
    "pass_in_focus": 5,
    "pass_anchor": 5,
    "pass_after_blur_anchor": 5,
}
LABEL_COLORS = {
    1: "#d62728",  # blur fail
    2: "#9467bd",  # dead / saturation
    3: "#ff7f0e",  # dorsal bright review
    4: "#7f7f7f",  # gray
    5: "#1a8820",  # pass
}


def bbox(mask: np.ndarray, pad: int = 60) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad + 1, mask.shape[0]),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, mask.shape[1]),
    )


def robust_u8(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def center_zoom(img: np.ndarray, mask: np.ndarray, frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the central `frac` of the embryo bbox so fine edges are visible."""
    ys, xs = np.where(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = int((ys.max() - ys.min()) * frac / 2)
    w = int((xs.max() - xs.min()) * frac / 2)
    h, w = max(h, 30), max(w, 30)
    y0, y1 = max(cy - h, 0), min(cy + h, img.shape[0])
    x0, x1 = max(cx - w, 0), min(cx + w, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(ANCHORS)
    df = df[df["metric_error"].fillna("") == ""].copy()
    df["sort_key"] = df["user_label"].map(ORDER).fillna(99).astype(int)
    return df.sort_values(["sort_key", "experiment_id", "well_id", "time_index"]).reset_index(drop=True)


def main() -> None:
    df = load_rows()
    n = len(df)
    fig, axes = plt.subplots(n, 3, figsize=(13, n * 3.0), squeeze=False)

    for r, (_, row) in enumerate(df.iterrows()):
        img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(row["mask_path"])).astype(bool)
        color = LABEL_COLORS.get(int(row["sort_key"]), "#444444")

        if mask.shape != img.shape or not mask.any():
            for c in range(3):
                axes[r, c].text(0.5, 0.5, "bad mask", ha="center", va="center")
                axes[r, c].axis("off")
            continue

        y0, y1, x0, x1 = bbox(mask)
        img_c = img[y0:y1, x0:x1]
        mask_c = mask[y0:y1, x0:x1]
        dil = ndi.binary_dilation(mask_c, iterations=28)
        norm = robust_u8(img_c, mask_c | dil)
        zimg, zmask = center_zoom(norm.astype(np.float32), mask_c, frac=0.45)

        panels = [
            ("raw", img_c, mask_c),
            ("local-norm", norm, mask_c),
            ("center zoom (norm)", zimg, zmask),
        ]
        for c, (title, arr, m) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if m is not None and m.any():
                ax.contour(m.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.5)
            if r == 0:
                ax.set_title(title, fontsize=11)

        label = (
            f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}\n"
            f"{row['user_label']}\n"
            f"IQR={row['emb_iqr_local']:.0f}  ent={row['norm_rel_entropy_local_annulus']:.2f}\n"
            f"Sobel={row['sobel_mean_emb_norm']:.3f}  f245={row['emb_frac_gt245']:.2f}\n"
            f"sobel f/c={row['sobel_fine_to_coarse_ratio']:.2f}  LoG f/c={row['log_fine_to_coarse_ratio']:.2f}"
        )
        axes[r, 0].set_ylabel(label, fontsize=8, color=color, rotation=0, ha="right", va="center", labelpad=70)

    fig.suptitle(
        "Focus anchors, one per row (red=blur fail, purple=dead/sat, orange=dorsal-bright, green=pass)\n"
        "Question: do dorsal-bright (orange) show sharp fine edges in center zoom, or are they smeared like blur (red)?",
        y=0.997,
        fontsize=12,
    )
    fig.tight_layout(rect=[0.04, 0, 1, 0.985])
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"Saved high-res anchor rows -> {OUT}")


if __name__ == "__main__":
    main()
