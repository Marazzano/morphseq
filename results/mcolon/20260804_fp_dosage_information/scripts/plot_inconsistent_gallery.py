"""Gallery of wells whose QC verdict CHANGES across timepoints.

WHY THIS IS THE RIGHT POPULATION TO LOOK AT. Carrying a transgene is permanent. So a well that
genuinely has no transgene must fail the gate at EVERY timepoint, and a well that has one should
pass at every timepoint. Consistency over time is therefore a ground-truth check the gate cannot
game -- it is not fitted to anything.

Measured on the full plate:

    genotype                       always-fail   always-pass   INCONSISTENT
    ab (no transgene)                    8            0             0
    tdtomato                             2           32            11
    pbx4_pbx1b_crispant_tdtomato         1           25            12

ALL 8 ab WELLS ALWAYS FAIL AND NONE IS INCONSISTENT -- the gate is temporally stable on known
negatives, which is the strongest evidence it is measuring something real.

The 3 always-failing transgenic wells are probably genuine non-fluorescent embryos (the user noted
some conditions were picked without confirming fluorescence). Those are correct rejections we lack
labels for.

THE 23 INCONSISTENT WELLS CANNOT BE BIOLOGY. A fish does not acquire a transgene between Tuesday
and Wednesday. Each one is either a measurement failure at the timepoints that fail (bad
segmentation, focus, occlusion) or a marginal embryo sitting on the threshold. This gallery shows
each inconsistent well across ALL its timepoints, with the passing and failing frames side by side,
so the failure mode can be read off the images rather than guessed from numbers.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

EXP = "20260624_2x_td_bf_pbx_coll_plate01"
OUT = MORPHSEQ_ROOT / ".pbx_smoke/out"
OUTDIR = Path(__file__).resolve().parents[1] / "output"
PAD_PX = 40


def load_genotypes() -> dict[str, str]:
    sheet = pd.read_excel(
        MORPHSEQ_ROOT / ".pbx_smoke/in/plate_metadata" / f"{EXP}_well_metadata.xlsx",
        sheet_name="genotype").set_index("Unnamed: 0")
    return {f"{r}{int(c):02d}": str(v) for r, s in sheet.iterrows()
            for c, v in s.items() if pd.notna(v)}


def crop(frame_path, mask_row):
    """Crop from the NATIVE frame -- snips exist for only 4 wells, native frames for all."""
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    h, w = image.shape[:2]
    y0, y1 = max(0, int(mask_row.bbox_y_min_px) - PAD_PX), min(h, int(mask_row.bbox_y_max_px) + PAD_PX)
    x0, x1 = max(0, int(mask_row.bbox_x_min_px) - PAD_PX), min(w, int(mask_row.bbox_x_max_px) + PAD_PX)
    return image[y0:y1, x0:x1]


qc = pd.read_csv(OUTDIR / "intensity_qc.csv")
qc["genotype"] = qc.well.map(load_genotypes())

per_well = qc.groupby(["well", "genotype"]).usable_for_pattern.agg(["sum", "count"]).reset_index()
per_well.columns = ["well", "genotype", "n_pass", "n_tp"]
inconsistent = per_well[(per_well.n_pass > 0) & (per_well.n_pass < per_well.n_tp)]
print(f"{len(inconsistent)} inconsistent wells")

masks = pd.concat([pd.read_csv(p) for p in glob.glob(str(
    OUT / "object_extraction" / EXP / "frame_masks/per_well/*/*_frame_masks.csv"))], ignore_index=True)
inv = pd.concat([pd.read_csv(p) for p in glob.glob(str(
    OUT / "acquisition" / EXP / "frame_inventory/per_well/*/*_frame_inventory.csv"))], ignore_index=True)
rfp = inv[inv.channel_id == "RFP"].set_index(["well_id", "time_index"])["image_path"].to_dict()

# One row per well, one column per timepoint. Sorted so the most lopsided wells (1 of 3 passing)
# come first -- those are where a single frame disagrees with the rest and the cause is clearest.
inconsistent = inconsistent.assign(frac=lambda d: d.n_pass / d.n_tp).sort_values("frac")
sel = inconsistent.head(12)
times = sorted(qc.time_index.unique())[:6]

fig, axes = plt.subplots(len(sel), len(times), figsize=(2.0 * len(times), 2.1 * len(sel)))
axes = np.atleast_2d(axes)
for row, (_, w) in enumerate(sel.iterrows()):
    well_id = f"{EXP}_{w.well}"
    for col, t in enumerate(times):
        ax = axes[row, col]
        ax.set_xticks([]); ax.set_yticks([])
        r = qc[(qc.well == w.well) & (qc.time_index == t)]
        if r.empty or (well_id, t) not in rfp:
            ax.axis("off")
            continue
        r = r.iloc[0]
        m = masks[(masks.well_id == well_id) & (masks.time_index == t)]
        m = m[m.area_px == m.area_px.max()] if len(m) else m
        img = crop(Path(rfp[(well_id, t)]), m.iloc[0]) if len(m) else None
        if img is None or img.size == 0:
            ax.axis("off")
            continue
        ax.imshow(img, cmap="magma", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        ok = bool(r.usable_for_pattern)
        # PASS/FAIL is the whole point of the figure, so it is the frame colour, not a caption.
        for spine in ax.spines.values():
            spine.set_edgecolor("#2166AC" if ok else "#B2182B")
            spine.set_linewidth(2.5)
        ax.set_title(f"t{t}  {'PASS' if ok else 'FAIL'}\neff={r.effective_states:.0f}",
                     fontsize=7, color="#2166AC" if ok else "#B2182B")
    axes[row, 0].set_ylabel(f"{w.well}\n{w.genotype[:12]}\n{w.n_pass}/{w.n_tp}", fontsize=7)

fig.suptitle(
    "Wells whose QC verdict CHANGES over time — a transgene cannot appear and disappear\n"
    "blue = passed, red = failed.  These are measurement failures or threshold-marginal embryos.",
    fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
OUTDIR.mkdir(parents=True, exist_ok=True)
png = OUTDIR / "inconsistent_wells_gallery.png"
fig.savefig(png, dpi=130)
print(f"wrote {png}")

per_well["class"] = per_well.apply(
    lambda r: "always_fails" if r.n_pass == 0 else ("always_passes" if r.n_pass == r.n_tp else "inconsistent"),
    axis=1)
per_well.to_csv(OUTDIR / "well_consistency.csv", index=False)
print(f"wrote {OUTDIR / 'well_consistency.csv'}")
print(pd.crosstab(per_well.genotype, per_well["class"]).to_string())
