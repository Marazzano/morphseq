"""Box-prompted SAM2 masks for the 3-experiment SeaHub validation set.

Takes the GroundingDINO boxes already in embryo_manifest.csv (crop_x1_px..crop_y2_px, 8 per FOV)
and prompts SAM2's image predictor with them to obtain per-embryo segmentation MASKS (not just
box crops). One SAM2 predict call per FOV using all 8 boxes at once.

Reuses the pipeline's SAM2 install (models/sam2 -> mdcolon checkout) via sys.path injection, the
same mechanism data_pipeline/models/sam2.py documents. CPU is fine for 3 FOVs / 24 embryos.

Outputs (sibling dir sam2_masks_3_experiments/):
  masks/<image_id>__embryo_NN_mask.png       binary mask, full-FOV frame
  mask_snips/<image_id>__embryo_NN.png       mask cropped to its box (per-embryo)
  overlays/<image_id>__overlay.jpg           FOV with all 8 masks tinted + boxes
  mask_contact_sheets/<exp>_<stem>.jpg       8 masked embryo crops
  sam2_mask_manifest.csv                     per-embryo: area_px, box, paths
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SRC_MANIFEST = HERE / "outputs/segmentation_validation_3_experiments/embryo_manifest.csv"
OUT = HERE / "outputs/sam2_masks_3_experiments"
SAM2_ROOT = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/sam2"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"          # relative to the sam2 pkg dir
SAM2_CKPT = f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt"

for sub in ("masks", "mask_snips", "overlays", "mask_contact_sheets"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, SAM2_ROOT)
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}; loading SAM2 hiera_large ...", flush=True)
sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
predictor = SAM2ImagePredictor(sam2)
print("SAM2 loaded.", flush=True)

m = pd.read_csv(SRC_MANIFEST)
box_cols = ["crop_x1_px", "crop_y1_px", "crop_x2_px", "crop_y2_px"]
rows_out = []

for image_id, g in m.groupby("image_id"):
    g = g.sort_values("embryo_position")
    img_path = g["image_path"].iloc[0]
    exp = g["experiment_id"].iloc[0]
    stem = g["stem"].iloc[0]
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    boxes = g[box_cols].to_numpy(dtype=np.float32)  # (8,4) xyxy pixels

    predictor.set_image(img)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=boxes, multimask_output=False,
        )
    # masks shape: (N,1,H,W) or (N,H,W) depending on version -> normalize to (N,H,W)
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    scores = np.asarray(scores).reshape(-1)

    overlay = Image.fromarray(img).convert("RGBA")
    tint = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(tint)
    palette = [(255,80,80),(80,255,80),(80,120,255),(255,220,60),
               (255,80,255),(60,230,230),(255,150,40),(160,120,255)]

    crops = []
    for i, (_, row) in enumerate(g.iterrows()):
        pos = int(row["embryo_position"])
        mask = masks[i] > 0.0
        x1,y1,x2,y2 = [int(v) for v in boxes[i]]
        # full-frame binary mask
        Image.fromarray((mask*255).astype(np.uint8)).save(
            OUT/"masks"/f"{image_id}__embryo_{pos:02d}_mask.png")
        # mask cropped to box, applied to image -> masked snip
        sub_img = img[y1:y2, x1:x2].copy()
        sub_m = mask[y1:y2, x1:x2]
        masked = sub_img.copy(); masked[~sub_m] = 255  # white bg outside mask
        Image.fromarray(masked).save(OUT/"mask_snips"/f"{image_id}__embryo_{pos:02d}.png")
        crops.append(Image.fromarray(masked))
        # overlay tint + box
        col = palette[(pos-1) % 8]
        ys, xs = np.where(mask)
        if len(xs):
            arr = np.array(tint)
            arr[ys, xs] = (*col, 110)
            tint = Image.fromarray(arr)
            td = ImageDraw.Draw(tint)
        td.rectangle([x1,y1,x2,y2], outline=(*col,255), width=3)
        rows_out.append({
            "image_id": image_id, "experiment_id": exp, "stem": stem,
            "embryo_position": pos, "mask_score": float(scores[i]),
            "mask_area_px": int(mask.sum()),
            "box_area_px": int((x2-x1)*(y2-y1)),
            "fill_fraction": float(mask.sum()/max(1,(x2-x1)*(y2-y1))),
            "mask_path": str(OUT/"masks"/f"{image_id}__embryo_{pos:02d}_mask.png"),
            "mask_snip_path": str(OUT/"mask_snips"/f"{image_id}__embryo_{pos:02d}.png"),
        })
    Image.alpha_composite(overlay, tint).convert("RGB").save(
        OUT/"overlays"/f"{image_id}__overlay.jpg", quality=88)

    # contact sheet: 4x2 masked crops
    cw = max(c.width for c in crops); ch = max(c.height for c in crops)
    sheet = Image.new("RGB",(cw*4+10, ch*2+6),(255,255,255))
    for i,c in enumerate(crops):
        r,cc = divmod(i,4)
        sheet.paste(c,(cc*(cw+2), r*(ch+2)))
    sheet.save(OUT/"mask_contact_sheets"/f"{exp}_{stem}.jpg", quality=88)
    print(f"  {exp}/{stem}: {len(crops)} masks, scores {scores.min():.2f}-{scores.max():.2f}", flush=True)

pd.DataFrame(rows_out).to_csv(OUT/"sam2_mask_manifest.csv", index=False)
print(f"\nDONE. {len(rows_out)} embryo masks -> {OUT}", flush=True)
