"""
15_decile_galleries.py
======================
Three full-distribution decile galleries, one per candidate focus metric, so we
can compare which metric's deciles give the cleanest good->bad visual transition
and pick a threshold:

  1. rel_entropy_mean   (z-stack, stringent)
  2. ff_rel_entropy     (projected full-focus)
  3. ff_lap_abs_ratio   (sharpness ratio, embryo vs background)

Each gallery: 10 decile rows x COLS sampled embryos, every tile annotated with
all four numbers (z / ff / shp / rat) so cross-metric disagreements (e.g. B06
t16: low entropy but high sharpness) are visible.

Reads the full projected scan + raw 2D JPEGs. Run after 11_merge_chunks.py.

  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/15_decile_galleries.py
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
OUT_DIR = HERE / "outputs/comparison/decile_galleries"
DATE = "20250912"

N_DECILES = 10
COLS = 8  # embryos sampled per decile row

# (column, label, higher_is_better) — used both for binning and the row order.
METRICS = [
    ("rel_entropy_mean", "z-stack rel_entropy_mean", True),
    ("ff_rel_entropy",   "projected ff_rel_entropy", True),
    ("ff_lap_abs_ratio", "sharpness ratio (emb/bg)", True),
]


def img_path(well: str, t: int) -> Path:
    return IMAGES_DIR / f"{DATE}_{well}" / f"{DATE}_{well}_ch00_t{t:04d}.jpg"


def sample_evenly(sub: pd.DataFrame, sort_col: str, n: int) -> pd.DataFrame:
    sub = sub.sort_values(sort_col).reset_index(drop=True)
    if len(sub) <= n:
        return sub
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    return sub.iloc[idx].reset_index(drop=True)


def build_gallery(df: pd.DataFrame, metric: str, label: str, higher_better: bool) -> None:
    d = df[df[metric].notna()].copy()
    # decile 0 = worst end. qcut on the metric; if higher is better, decile 0 is the
    # low (bad) end already, so labels run worst->best down the rows.
    d["decile"] = pd.qcut(d[metric], N_DECILES, labels=False, duplicates="drop")
    deciles = sorted(d["decile"].unique())

    nrows = len(deciles)
    fig, axes = plt.subplots(nrows, COLS, figsize=(COLS * 2.0, nrows * 2.2))
    if nrows == 1:
        axes = axes[None, :]

    for r, dec in enumerate(deciles):
        grp = d[d["decile"] == dec]
        lo, hi = grp[metric].min(), grp[metric].max()
        picks = sample_evenly(grp, metric, COLS)
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
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"D{int(dec)+1}\n[{lo:.2f},\n {hi:.2f}]", fontsize=8)

    fig.suptitle(
        f"Decile gallery binned by: {label}\n"
        "rows = deciles (D1 worst -> D10 best)   "
        "tile: z=z-entropy ff=proj-entropy shp=abs-sharp rat=sharp-ratio",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe = metric.replace("/", "_")
    out = OUT_DIR / f"deciles_by_{safe}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out}  ({nrows} deciles x {COLS})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FULL_CSV)
    df = df[df["has_mask"].astype(bool)] if "has_mask" in df else df
    for metric, label, hb in METRICS:
        build_gallery(df, metric, label, hb)


if __name__ == "__main__":
    main()
