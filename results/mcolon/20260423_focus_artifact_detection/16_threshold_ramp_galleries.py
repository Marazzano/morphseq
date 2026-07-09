"""
16_threshold_ramp_galleries.py
==============================
Stress-test the two candidate cutoffs by zooming into the embryos that straddle
each threshold -- a fine-binned ramp stepping across the line so you can see
quality change right at the cut and judge whether the threshold is good.

Two thresholds (z-stack entropy intentionally dropped -- not the gate we want):
  1. ff_rel_entropy  cut = -0.55   (projected full-focus, leading candidate)
  2. ff_lap_abs_ratio cut = 1.5    (sharpness ratio emb/bg)

Each gallery: several narrow bins stepping through the cut (rows), COLS embryos
sampled per bin, every tile annotated z/ff/shp/rat. The two bins immediately
below the cut = "would be REJECTED"; immediately above = "would be KEPT".

Run after the full projected scan + merge.
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/16_threshold_ramp_galleries.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
HERE = Path(__file__).resolve().parent
FULL_CSV = HERE / "outputs/projected/scan/ff_metrics_full.csv"
IMAGES_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
OUT_DIR = HERE / "outputs/comparison/threshold_ramps"
DATE = "20250912"
COLS = 8

# (column, label, cut, bin_edges) -- edges span the ramp; cut marks the line.
RAMPS = [
    ("ff_rel_entropy", "projected ff_rel_entropy", -0.55,
     [-0.65, -0.60, -0.575, -0.55, -0.525, -0.50, -0.45]),
    ("ff_lap_abs_ratio", "sharpness ratio (emb/bg)", 1.5,
     [1.0, 1.25, 1.40, 1.50, 1.60, 1.75, 2.0]),
]


def img_path(well: str, t: int) -> Path:
    return IMAGES_DIR / f"{DATE}_{well}" / f"{DATE}_{well}_ch00_t{t:04d}.jpg"


def sample_evenly(sub: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    sub = sub.sort_values(col).reset_index(drop=True)
    if len(sub) <= n:
        return sub
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    return sub.iloc[idx].reset_index(drop=True)


def build_ramp(df: pd.DataFrame, col: str, label: str, cut: float, edges: list[float]) -> None:
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    nrows = len(bins)
    fig, axes = plt.subplots(nrows, COLS, figsize=(COLS * 2.1, nrows * 2.3))
    if nrows == 1:
        axes = axes[None, :]

    for r, (lo, hi) in enumerate(bins):
        sub = df[df[col].between(lo, hi, inclusive="left")]
        picks = sample_evenly(sub, col, COLS)
        side = "REJECT" if hi <= cut else ("ACCEPT" if lo >= cut else "CUT")
        for c in range(COLS):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks):
                continue
            row = picks.iloc[c]
            well, t = str(row["well"]), int(row["t"])
            p = img_path(well, t)
            if p.exists():
                ax.imshow(np.array(Image.open(p).convert("L")), cmap="gray")
            else:
                ax.text(0.5, 0.5, "no img", ha="center", va="center", fontsize=8)
            ax.set_title(
                f"{well} t{t}\n"
                f"z={row['rel_entropy_mean']:.2f} ff={row['ff_rel_entropy']:.2f}\n"
                f"shp={row['ff_lap_abs_mean_emb']:.0f} rat={row['ff_lap_abs_ratio']:.2f}",
                fontsize=6.5,
            )
        # Row label: bin range, side of cut, n. Color rejects red, keeps green.
        color = "#b2182b" if side == "REJECT" else ("#1a8820" if side == "ACCEPT" else "#d08000")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_color(color); sp.set_linewidth(3)
        axes[r, 0].set_ylabel(f"[{lo:.3g}, {hi:.3g})\n{side}  n={len(sub)}",
                              fontsize=8, color=color)

    fig.suptitle(
        f"Threshold ramp: {label}   cut = {cut}\n"
        "rows step across the cut (red=REJECT / green=ACCEPT)   "
        "tile: z / ff / shp / rat",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    safe = col.replace("/", "_")
    out = OUT_DIR / f"ramp_{safe}_cut{cut}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved -> {out}  ({nrows} bins x {COLS})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FULL_CSV)
    df = df[df["has_mask"].astype(bool)] if "has_mask" in df else df
    for col, label, cut, edges in RAMPS:
        build_ramp(df, col, label, cut, edges)


if __name__ == "__main__":
    main()
