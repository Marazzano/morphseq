#!/usr/bin/env python
"""Per-experiment emb_iqr_local decile galleries across all experiments.

emb_iqr_local = IQR of raw intensities INSIDE the embryo mask (embryo pixels only,
NOT the annulus). It measures embryo contrast amount: dead / saturated / washed-out
embryos go flat -> low IQR; real contrast-bearing embryos -> high IQR. This is the
proposed FIRST QC filter (dead/contrast axis), separate from the blur axis.

This script samples not-dead snips per experiment, recomputes emb_iqr_local fresh
(the cross-experiment focus_metrics_by_experiment.csv does not carry it), cuts into
10 per-experiment deciles, and renders N_PER_DECILE images per decile so the bottom
deciles can be eyeballed to confirm they are genuinely droppable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "iqr_deciles"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
OUT_METRICS = TABLES / "iqr_cross_experiment_decile_metrics.csv"
OUT_SAMPLES = TABLES / "iqr_cross_experiment_decile_samples.csv"

N_DECILES = 10
N_PER_DECILE = 20
SAMPLE_PER_EXPERIMENT = 2500
RANDOM_SEED = 1729


def bbox(mask: np.ndarray, pad: int = 96) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad + 1, mask.shape[0]),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, mask.shape[1]),
    )


def emb_iqr(row: pd.Series) -> float:
    """IQR of raw intensities inside the embryo mask (embryo pixels only)."""
    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    if mask.shape != img.shape or not mask.any():
        raise ValueError("bad mask")
    y0, y1, x0, x1 = bbox(mask)
    emb_px = img[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    return float(np.percentile(emb_px, 75) - np.percentile(emb_px, 25))


def compute_metrics() -> pd.DataFrame:
    if OUT_METRICS.exists():
        print(f"Loading existing IQR metrics -> {OUT_METRICS}", flush=True)
        return pd.read_csv(OUT_METRICS)

    df = pd.read_csv(FOCUS_CSV, low_memory=False)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["image_path", "mask_path"]).copy()
    df["experiment_id"] = df["experiment_id"].astype(str)

    parts = []
    for exp, grp in df.groupby("experiment_id"):
        if len(grp) > SAMPLE_PER_EXPERIMENT:
            grp = grp.sample(n=SAMPLE_PER_EXPERIMENT, random_state=RANDOM_SEED)
        parts.append(grp)
    sample = pd.concat(parts, ignore_index=True)
    print(f"Sampled {len(sample)} not-dead snips across {sample['experiment_id'].nunique()} experiments", flush=True)

    iqrs, errs = [], []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        try:
            iqrs.append(emb_iqr(row))
            errs.append("")
        except Exception as exc:  # noqa: BLE001
            iqrs.append(np.nan)
            errs.append(str(exc))
        if i % 500 == 0 or i == len(sample):
            print(f"[iqr] {i}/{len(sample)}", flush=True)
    sample["emb_iqr_local"] = iqrs
    sample["iqr_error"] = errs
    sample.to_csv(OUT_METRICS, index=False)
    print(f"Saved IQR metrics -> {OUT_METRICS}", flush=True)
    return sample


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    use = df[df["iqr_error"].fillna("") == ""].dropna(subset=["emb_iqr_local"]).copy()
    out = []
    for exp, grp in use.groupby("experiment_id"):
        grp = grp.copy()
        grp["iqr_decile"] = (
            pd.qcut(grp["emb_iqr_local"].rank(method="first"), N_DECILES, labels=False, duplicates="drop").astype(int)
            + 1
        )
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def sample_for_gallery(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, decile), grp in df.groupby(["experiment_id", "iqr_decile"]):
        grp = grp.sort_values("emb_iqr_local")
        if len(grp) > N_PER_DECILE:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_DECILE).astype(int)]
        else:
            picks = grp
        picks = picks.copy()
        picks["rank_within_decile"] = np.arange(1, len(picks) + 1)
        rows.append(picks)
    manifest = pd.concat(rows, ignore_index=True)
    manifest.to_csv(OUT_SAMPLES, index=False)
    print(f"Saved decile samples -> {OUT_SAMPLES}", flush=True)
    return manifest


def crop_for_gallery(image_path: object, mask_path: object, pad: int = 60):
    ip, mp = Path(str(image_path)), Path(str(mask_path))
    if not ip.exists() or not mp.exists():
        return None, None
    img = np.array(Image.open(ip).convert("L"), dtype=np.uint8)
    mask = np.array(Image.open(mp)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return img, None
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def plot_experiment(manifest: pd.DataFrame, exp: str) -> None:
    sub = manifest[manifest["experiment_id"] == exp]
    fig, axes = plt.subplots(
        N_DECILES, N_PER_DECILE, figsize=(N_PER_DECILE * 1.5, N_DECILES * 1.6), squeeze=False
    )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, decile in enumerate(sorted(sub["iqr_decile"].unique())):
        grp = sub[sub["iqr_decile"] == decile].sort_values("rank_within_decile")
        for c, (_, row) in enumerate(grp.head(N_PER_DECILE).iterrows()):
            ax = axes[r, c]
            img, mask = crop_for_gallery(row["image_path"], row["mask_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                if mask is not None and mask.any():
                    ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.4)
            ax.set_title(f"IQR={row['emb_iqr_local']:.0f}", fontsize=5)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=9)

    fig.suptitle(
        f"{exp}: not-dead embryo IQR deciles (per-experiment, D1 low contrast -> D10 high), {N_PER_DECILE}/decile",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGURES / f"{exp}_iqr_deciles_not_dead.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved -> {out}", flush=True)


def write_summary(df: pd.DataFrame) -> None:
    use = df[df["iqr_error"].fillna("") == ""].dropna(subset=["emb_iqr_local"])
    summary = (
        use.groupby(["experiment_id", "iqr_decile"])
        .agg(n=("emb_iqr_local", "size"),
             iqr_min=("emb_iqr_local", "min"),
             iqr_median=("emb_iqr_local", "median"),
             iqr_max=("emb_iqr_local", "max"))
        .reset_index()
    )
    out = TABLES / "iqr_cross_experiment_decile_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved decile summary -> {out}", flush=True)


def main() -> None:
    metrics = compute_metrics()
    deciled = assign_deciles(metrics)
    manifest = sample_for_gallery(deciled)
    write_summary(deciled)
    for exp in sorted(manifest["experiment_id"].unique()):
        plot_experiment(manifest, exp)


if __name__ == "__main__":
    main()
