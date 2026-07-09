"""
14_band_image_gallery.py
========================
Actual-image gallery for the decision band (z-stack rel_entropy_mean in
[-0.55, -0.35]), so we can eyeball whether the embryos near the threshold are
usable, and how the sharpness numbers track what we see.

Each tile is the 2D focus-stacked JPEG for one (well, t), annotated with:
  z   = z-stack rel_entropy_mean   (the stringent metric)
  ff  = projected ff_rel_entropy
  shp = ff_lap_abs_mean_emb        (ABSOLUTE sharpness: mean |Laplacian| in embryo)
  rat = ff_lap_abs_ratio           (sharpness RATIO: embryo / background)

All four come from the full projected scan; see 11_focus_stacked_metrics.py
metric_row() for where shp/rat are computed (cv2.Laplacian on the stacked image).

Rows are z-stack sub-bins (the stringent axis); columns are embryos sampled
evenly within each sub-bin. Run AFTER 11_merge_chunks.py.

  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/14_band_image_gallery.py
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
OUT = HERE / "outputs/comparison/threshold_band/band_image_gallery.png"
DATE = "20250912"

BAND_LO, BAND_HI = -0.55, -0.35
BIN_WIDTH = 0.05
COLS = 7  # embryos sampled per sub-bin row


def img_path(well: str, t: int) -> Path:
    return IMAGES_DIR / f"{DATE}_{well}" / f"{DATE}_{well}_ch00_t{t:04d}.jpg"


def sample_row(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    sub = sub.sort_values("rel_entropy_mean").reset_index(drop=True)
    if len(sub) <= n:
        return sub
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    return sub.iloc[idx].reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(FULL_CSV)
    df = df[df["has_mask"].astype(bool)] if "has_mask" in df else df
    band = df[df["rel_entropy_mean"].between(BAND_LO, BAND_HI, inclusive="both")].copy()

    edges = np.round(np.arange(BAND_LO, BAND_HI + 1e-9, BIN_WIDTH), 3)
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    nrows = len(bins)

    fig, axes = plt.subplots(nrows, COLS, figsize=(COLS * 2.0, nrows * 2.3))
    if nrows == 1:
        axes = axes[None, :]

    for r, (lo, hi) in enumerate(bins):
        sub = band[band["rel_entropy_mean"].between(lo, hi, inclusive="left")]
        picks = sample_row(sub, COLS)
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
                f"z={row['rel_entropy_mean']:.2f}  ff={row['ff_rel_entropy']:.2f}\n"
                f"shp={row['ff_lap_abs_mean_emb']:.0f}  rat={row['ff_lap_abs_ratio']:.2f}",
                fontsize=7,
            )
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"z in [{lo:.2f}, {hi:.2f})", fontsize=9)

    fig.suptitle(
        "Decision-band gallery (rows = z-stack sub-bin, the stringent axis)\n"
        "z=z-stack rel_entropy  ff=projected rel_entropy  "
        "shp=abs sharpness (mean|Laplacian|)  rat=sharpness ratio emb/bg",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"Saved -> {OUT}  ({nrows} sub-bins x {COLS} cols)")


if __name__ == "__main__":
    main()
