#!/usr/bin/env python
"""Montage the pipeline's materialized full-field (stitched 3-tile) images.

Reads acquisition/{exp}/materialized_images/{well}/BF/projection/focus_stack/{well}_BF_t0000.png
(after the map_keyence_positions_to_wells tile-count fix these are true 960x2160 mosaics),
labels each by well + temperature + chem_perturbation, and writes one montage per plate plus a
combined sheet. Wide layout (many columns) since each full-field tile is tall.

Run:
  conda run -n segmentation_grounded_sam --no-capture-output python build_ff_montage.py
"""
from __future__ import annotations
import glob, os, math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

O = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline_output"
RESULT = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20260702_hotchem_qc"
EXPERIMENTS = [
    "20260702_hotchem_24hpf_plate01", "20260702_hotchem_24hpf_plate02",
    "20260702_hotchem_30hpf_plate01", "20260702_hotchem_30hpf_plate02",
    "20260702_hotchem_36hpf_plate01", "20260702_hotchem_36hpf_plate02",
]
THUMB_W = 64      # full-field is ~960x2160 (tall) -> thumb ~64x144
COLS = 24         # wide layout to fix the aspect ratio
PAD = 3
LABEL_H = 12

os.makedirs(RESULT, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 8)
except Exception:
    FONT = ImageFont.load_default()


def load_meta(exp):
    p = f"{O}/acquisition/{exp}/ingest_metadata/plate_metadata.csv"
    m = {}
    if os.path.exists(p):
        for _, r in pd.read_csv(p, dtype=str).iterrows():
            m[str(r["well_id"]).split("_")[-1]] = (str(r.get("temperature", "?")),
                                                   str(r.get("chem_perturbation", "?")))
    return m


def cell(path, label):
    im = Image.open(path).convert("L")
    th = max(1, int(THUMB_W * im.size[1] / im.size[0]))
    im = im.resize((THUMB_W, th))
    c = Image.new("RGB", (THUMB_W, th + LABEL_H), (0, 0, 0))
    c.paste(im, (0, 0))
    ImageDraw.Draw(c).text((1, th), label[:15], fill=(255, 255, 100), font=FONT)
    return c


def montage(cells, title, out):
    if not cells:
        print("  [skip]", title); return
    cw = max(c.size[0] for c in cells); ch = max(c.size[1] for c in cells)
    cols = min(COLS, len(cells)); rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * (cw + PAD) + PAD, 18 + rows * (ch + PAD) + PAD), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 3), title, fill=(0, 0, 0), font=FONT)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (PAD + col * (cw + PAD), 18 + r * (ch + PAD)))
    sheet.save(out); print("  wrote", out, f"({len(cells)} wells)")


def main():
    allcells = []
    for exp in EXPERIMENTS:
        meta = load_meta(exp)
        pngs = sorted(glob.glob(
            f"{O}/acquisition/{exp}/materialized_images/*/BF/projection/focus_stack/*_BF_t0000.png"))
        cells = []
        for p in pngs:
            slug = os.path.normpath(p).split(os.sep)[-5].split("_")[-1]
            t, c = meta.get(slug, ("?", "?"))
            cells.append(cell(p, f"{slug} {t}C {c}"))
        if pngs:
            print(f"{exp}: {len(pngs)} wells, first image size = {Image.open(pngs[0]).size}")
        montage(cells, f"FULL-FIELD {exp} (n={len(cells)})", f"{RESULT}/ff_{exp}.png")
        allcells.extend(cells)
    montage(allcells, "FULL-FIELD — all 6 hotchem plates (3-tile stitched, focus-stacked)",
            f"{RESULT}/ff_ALL.png")


if __name__ == "__main__":
    main()
