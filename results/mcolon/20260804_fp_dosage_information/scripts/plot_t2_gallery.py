"""t2 embryo gallery: BF snip + RFP snip + mask area, sampled across the intensity range.

WHAT THIS IS FOR. Picking usable orientations by eye, and checking the premise that embryos in
roughly the same orientation should have roughly the same SURFACE AREA. Area is the confound for
any per-embryo intensity comparison: a mask that caught more of the fish, or caught the fish
side-on rather than dorsal, integrates more signal for the same underlying expression.

So every panel is labelled with mask area alongside the intensity, and the gallery is sampled
ACROSS the intensity range rather than showing the brightest -- the question is whether dim and
bright embryos differ in expression or merely in how much of them the mask found.

AREA ALSO EXPOSES SEGMENTATION FAILURE. At t2 the areas run 82k to 4.6M px. A real embryo here is
~85k; anything above a few hundred thousand is a full-frame or well-sized blob, and those all read
DIM (380-763 DN) for the obvious reason that they are mostly background. Those rows are still
plotted, in a separate band, because seeing them is the point -- they are silently in the
distribution otherwise.
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
TIME_INDEX = 2

# A real embryo at this magnification is ~85k px. Well above that is a segmentation failure, not a
# big fish -- kept in the figure but banded separately so it cannot be mistaken for a dim embryo.
PLAUSIBLE_AREA_MAX_PX = 250_000

BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max__no_change"


def crop_from_native(frame_path: Path, mask_row, pad_px: int = 40):
    """Crop the embryo out of the NATIVE frame using its frame_masks bbox.

    RENDERED FROM NATIVE FRAMES, NOT SNIPS. This gallery only needs to show what the embryo looks
    like, and snips exist for 4 wells while intensity exists for 30 -- channel_intensity reads
    native frames and never needed a snip. Cropping the native raster shows the SAME pixels the
    measurement used, for every well that has been measured, with no dependency on a rendering
    step that has not run.
    """
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    h, w = image.shape[:2]
    y0 = max(0, int(mask_row.bbox_y_min_px) - pad_px)
    y1 = min(h, int(mask_row.bbox_y_max_px) + pad_px)
    x0 = max(0, int(mask_row.bbox_x_min_px) - pad_px)
    x1 = min(w, int(mask_row.bbox_x_max_px) + pad_px)
    return image[y0:y1, x0:x1]


shards = sorted(glob.glob(str(
    OUT / "object_extraction" / EXP / "channel_intensity/per_well/*/RFP__projection__max/channel_intensity.csv"
)))
d = pd.concat([f for f in (pd.read_csv(p) for p in shards) if len(f)], ignore_index=True)
d = d[d.time_index == TIME_INDEX].copy()
d["well"] = d.well_id.str.rsplit("_", n=1).str[-1]
d["mean_dn"] = d.embryo_sum_dn / d.embryo_px
d["plausible"] = d.embryo_px <= PLAUSIBLE_AREA_MAX_PX

# The transform table names each embryo's snip, so the gallery follows the pipeline's own identity
# rather than reconstructing filenames.
transforms = pd.concat([
    pd.read_csv(p) for p in glob.glob(str(
        OUT / "object_extraction" / EXP / "snip_geometry/per_well/*/*_snip_transforms.csv"))
], ignore_index=True)
transforms = transforms[transforms.time_index == TIME_INDEX]
by_mask = transforms.set_index("geometry_source_mask_id")

# frame_masks gives the bbox to crop; frame_inventory gives the native frame paths.
masks = pd.concat([pd.read_csv(p) for p in glob.glob(str(
    OUT / "object_extraction" / EXP / "frame_masks/per_well/*/*_frame_masks.csv"))], ignore_index=True)
masks = masks[masks.time_index == TIME_INDEX].set_index("mask_id")
inv = pd.concat([pd.read_csv(p) for p in glob.glob(str(
    OUT / "acquisition" / EXP / "frame_inventory/per_well/*/*_frame_inventory.csv"))], ignore_index=True)
inv = inv[inv.time_index == TIME_INDEX]
bf_frame = inv[inv.channel_id == "BF"].set_index("well_id")["image_path"].to_dict()
rfp_frame = inv[inv.channel_id == "RFP"].set_index("well_id")["image_path"].to_dict()

rows = []
for _, r in d.iterrows():
    if r.mask_id not in masks.index:
        continue
    m = masks.loc[r.mask_id]
    rows.append({**r, "mask": m,
                 "bf": bf_frame.get(r.well_id), "rfp": rfp_frame.get(r.well_id)})
g = pd.DataFrame(rows)
g = g[g.bf.notna() & g.rfp.notna()]
if g.empty:
    raise SystemExit("no snips found for t2 -- has snip_materialization run for these wells?")

# SAMPLE ACROSS THE RANGE, not the top: take plausible embryos at even intensity quantiles, then
# append the implausible ones so the failure mode is visible in the same figure.
ok = g[g.plausible].sort_values("mean_dn")
picks = ok.iloc[np.linspace(0, len(ok) - 1, min(8, len(ok))).astype(int)] if len(ok) else ok
bad = g[~g.plausible].sort_values("embryo_px", ascending=False).head(3)
sel = pd.concat([picks, bad])

import cv2  # noqa: E402

n = len(sel)
fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 5.2))
axes = np.atleast_2d(axes)
for i, (_, r) in enumerate(sel.iterrows()):
    for row, (key, cmap) in enumerate((("bf", "gray"), ("rfp", "magma"))):
        img = crop_from_native(Path(r[key]), r["mask"])
        ax = axes[row, i]
        if img is None or img.size == 0:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # Per-panel display scaling ONLY -- this is a picture, and the quantitative comparison is
        # the number printed above it, never the rendered brightness.
        ax.imshow(img, cmap=cmap, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            flag = "" if r.plausible else "\nSEG FAIL"
            ax.set_title(f"{r.well}\n{r.embryo_px/1000:.0f}k px\n{r.mean_dn:.0f} DN{flag}",
                         fontsize=7, color="black" if r.plausible else "#B2182B")
axes[0, 0].set_ylabel("BF", fontsize=9)
axes[1, 0].set_ylabel("RFP", fontsize=9)
fig.suptitle(
    f"t2 embryos sampled across the intensity range — area vs intensity\n"
    f"a real embryo is ~85k px; red panels are segmentation failures kept in view",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))

OUTDIR.mkdir(parents=True, exist_ok=True)
png = OUTDIR / "t2_embryo_gallery.png"
fig.savefig(png, dpi=150)
print(f"wrote {png}")
print(f"\nplausible embryos (<= {PLAUSIBLE_AREA_MAX_PX:,} px): {len(ok)}")
if len(ok):
    print(f"  area  {ok.embryo_px.min():,} - {ok.embryo_px.max():,} px "
          f"({ok.embryo_px.max()/ok.embryo_px.min():.2f}x spread)")
    print(f"  CV of area: {ok.embryo_px.std()/ok.embryo_px.mean():.2f}  "
          f"<- if orientation were identical this would be small")
print(f"segmentation failures (> {PLAUSIBLE_AREA_MAX_PX:,} px): {(~g.plausible).sum()}")
