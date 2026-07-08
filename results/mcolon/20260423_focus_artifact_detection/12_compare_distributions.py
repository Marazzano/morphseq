"""
12_compare_distributions.py
===========================
Compare the two focus-QC measurement DOMAINS over the full experiment:

  - z-stack domain   : rel_entropy_mean   (per-Z slices, script 10 full scan)
  - projected domain : ff_rel_entropy     (final focus-stacked 2D image, script 11 --all)

This is the comparison that motivated script 11: does measuring focus on the
image that actually goes downstream (the projection) flag different embryos than
measuring it on the raw z-stack? Outputs the overlaid distributions + a paired
scatter that exposes where the two domains decouple (the gray zone).

Run AFTER 11_merge_chunks.py has produced ff_metrics_full.csv.

  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/12_compare_distributions.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ZSTACK_CSV = HERE / "10_scan_output/rel_entropy_summaries.csv"
PROJECTED_CSV = HERE / "outputs/projected/scan/ff_metrics_full.csv"
OUT_DIR = HERE / "outputs/comparison"


def load() -> pd.DataFrame:
    z = pd.read_csv(ZSTACK_CSV)
    z = z[z["has_mask"].astype(bool)][["t", "well", "rel_entropy_mean"]]
    f = pd.read_csv(PROJECTED_CSV)[["t", "well", "ff_rel_entropy", "ff_lap_abs_ratio"]]
    df = z.merge(f, on=["t", "well"], how="inner")
    print(f"z-stack rows: {len(z)} | projected rows: {len(f)} | joined: {len(df)}")
    return df


def overlay_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = float(min(df["rel_entropy_mean"].min(), df["ff_rel_entropy"].min()))
    hi = float(max(df["rel_entropy_mean"].max(), df["ff_rel_entropy"].max()))
    bins = np.linspace(lo, hi, 80)
    ax.hist(df["rel_entropy_mean"], bins=bins, alpha=0.5,
            label="z-stack  rel_entropy_mean", color="#1f77b4")
    ax.hist(df["ff_rel_entropy"], bins=bins, alpha=0.5,
            label="projected  ff_rel_entropy", color="#d62728")
    ax.set_xlabel("relative entropy (embryo - background)")
    ax.set_ylabel("count")
    ax.set_title("Focus QC: z-stack vs projected-image distribution")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "zstack_vs_projected_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def paired_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["rel_entropy_mean"], df["ff_rel_entropy"],
               s=6, alpha=0.3, color="#333333")
    lo = float(min(df["rel_entropy_mean"].min(), df["ff_rel_entropy"].min()))
    hi = float(max(df["rel_entropy_mean"].max(), df["ff_rel_entropy"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1, label="y = x")
    ax.set_xlabel("z-stack  rel_entropy_mean")
    ax.set_ylabel("projected  ff_rel_entropy")
    ax.set_title("Where the two domains decouple")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "zstack_vs_projected_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    df.to_csv(OUT_DIR / "joined_zstack_projected.csv", index=False)
    overlay_hist(df)
    paired_scatter(df)
    corr = df["rel_entropy_mean"].corr(df["ff_rel_entropy"])
    print(f"Pearson r (z-stack vs projected): {corr:.3f}")


if __name__ == "__main__":
    main()
