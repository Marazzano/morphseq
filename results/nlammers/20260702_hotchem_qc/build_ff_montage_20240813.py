#!/usr/bin/env python
"""Montage the pipeline's materialized full-field images for the 20240813* experiments.

Same logic as build_ff_montage.py, but points at the pipeline/output root (where the
20240813* acquisitions live) and writes montages to a dedicated results dir.

Reads acquisition/{exp}/materialized_images/{well}/BF/projection/focus_stack/{well}_BF_t0000.png
(720x2304 portrait; the natural view is landscape, so each cell is rotated 90 deg), labels each by
well, and writes one montage per experiment plus a combined sheet.

Run:
  python build_ff_montage_20240813.py
"""
from __future__ import annotations
import glob, os, math
from PIL import Image, ImageDraw, ImageFont

O = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
RESULT = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20240813_ff_montage"
EXPERIMENTS = [
    "20240813_24hpf", "20240813_30hpf", "20240813_36hpf", "20240813_extras",
]
THUMB_W = 480     # cells are landscape 2304x720, cropped to ~1613x720 -> thumb ~480x214
COLS = 12         # 12 x 8 = a 96-well plate
CROP_FRAC = 0.15  # trim this fraction off each of the left/right edges (mostly well wall)
PAD = 3
LABEL_H = 12

os.makedirs(RESULT, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 10)
except Exception:
    FONT = ImageFont.load_default()


def cell(path, label):
    # On disk the mosaic is portrait (720 wide x 2304 tall); rotate to its natural landscape view.
    im = Image.open(path).convert("L").rotate(-90, expand=True)
    # Trim the left/right edges: the outer tiles are mostly well wall, not embryo.
    w, h = im.size
    dx = int(w * CROP_FRAC)
    im = im.crop((dx, 0, w - dx, h))
    th = max(1, int(THUMB_W * im.size[1] / im.size[0]))
    im = im.resize((THUMB_W, th))
    c = Image.new("RGB", (THUMB_W, th + LABEL_H), (0, 0, 0))
    c.paste(im, (0, 0))
    ImageDraw.Draw(c).text((2, th), label, fill=(255, 255, 100), font=FONT)
    return c


def montage(cells, title, out):
    if not cells:
        print("  [skip]", title); return
    cw = max(c.size[0] for c in cells); ch = max(c.size[1] for c in cells)
    cols = min(COLS, len(cells)); rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * (cw + PAD) + PAD, 20 + rows * (ch + PAD) + PAD), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0), font=FONT)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (PAD + col * (cw + PAD), 20 + r * (ch + PAD)))
    sheet.save(out); print("  wrote", out, f"({len(cells)} wells)")


def main():
    allcells = []
    for exp in EXPERIMENTS:
        pngs = sorted(glob.glob(
            f"{O}/acquisition/{exp}/materialized_images/*/BF/projection/focus_stack/*_BF_t0000.png"))
        cells = []
        for p in pngs:
            well = os.path.normpath(p).split(os.sep)[-5].split("_")[-1]
            cells.append(cell(p, well))
        # Upstream mislabeled even rows and reversed their order L<->R; un-reverse the
        # cell order (labels travel with each cell) so B1<->B12, B2<->B11, etc.
        for row in range(1, math.ceil(len(cells) / COLS), 2):  # rows 2,4,6,8 (1-based)
            lo, hi = row * COLS, min((row + 1) * COLS, len(cells))
            cells[lo:hi] = cells[lo:hi][::-1]
        if pngs:
            print(f"{exp}: {len(pngs)} wells, first size = {Image.open(pngs[0]).size}")
        montage(cells, f"FULL-FIELD {exp} (n={len(cells)})", f"{RESULT}/ff_{exp}.png")
        allcells.extend(cells)
    montage(allcells, "FULL-FIELD — all 20240813 experiments (3-tile stitched, focus-stacked)",
            f"{RESULT}/ff_ALL.png")


if __name__ == "__main__":
    main()
