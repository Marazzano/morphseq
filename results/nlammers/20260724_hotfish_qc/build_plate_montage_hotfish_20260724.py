#!/usr/bin/env python
"""True-position 96-well plate montage for the 6 hotfish plates.

Differences from the legacy build_ff_montage.py, per request:
  (i)   NO flip logic — the even-row reversal hack is gone entirely. It is unnecessary:
        the pipeline labels wells from the instrument's explicit Keyence well marker
        (mapping_method='keyence_xy_well_marker'), NOT from serpentine capture order, so the
        well labels are already physically correct. Applying the old reversal to correct labels
        is what INTRODUCED the every-other-row column flip.
  (ii)  Plate ID printed as the sheet title; well label printed upper-left on each cell.
  (iii) Each image placed at its TRUE plate position, parsed from its well label (A01 -> row A,
        col 1). 8 rows (A-H) x 12 cols. Wells with no image are left blank.
  (iv)  Each image is photometrically inverted.

Reads acquisition/{exp}/materialized_images/{well}/BF/projection/focus_stack/{well}_BF_t0000.png
(portrait on disk; rotated 90 deg to the natural landscape view).

Run:
  conda run -n segmentation_grounded_sam --no-capture-output python build_plate_montage_hotfish_20260724.py
"""
from __future__ import annotations
import glob, os, re
from PIL import Image, ImageDraw, ImageFont, ImageOps

O = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output"
RESULT = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20260724_hotfish_qc"
EXPERIMENTS = [
    "20260724_hotfish_24hpf_plate01", "20260724_hotfish_24hpf_plate02",
    "20260724_hotfish_30hpf_plate01", "20260724_hotfish_30hpf_plate02",
    "20260724_hotfish_36hpf_plate01", "20260724_hotfish_36hpf_plate02",
]
ROWS = "ABCDEFGH"   # 8 plate rows
NCOLS = 12          # 12 plate columns
THUMB_W = 480       # cell width after rotate/crop/resize
CROP_FRAC = 0.15    # trim this fraction off each of the left/right edges (mostly well wall)
PAD = 3
TITLE_H = 20        # top strip for the plate-ID title
HDR = 20            # margin for row-letter / column-number headers
WELL_RE = re.compile(r"^([A-H])(0[1-9]|1[0-2])$")

os.makedirs(RESULT, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 11)
    FONT_BIG = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    FONT = ImageFont.load_default()
    FONT_BIG = FONT


def load_cell(path):
    """Load one well image: rotate to landscape, trim well-wall edges, INVERT, resize. No flip."""
    im = Image.open(path).convert("L").rotate(-90, expand=True)
    w, h = im.size
    dx = int(w * CROP_FRAC)
    im = im.crop((dx, 0, w - dx, h))
    im = ImageOps.invert(im)                      # (iv) photometric inversion
    th = max(1, int(THUMB_W * im.size[1] / im.size[0]))
    im = im.resize((THUMB_W, th))
    return im.convert("RGB")


def label_cell(im, text):
    """Stamp the well label in the upper-left with a dark backing box for legibility."""
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 6, 15], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 100), font=FONT)
    return im


def build_plate(exp):
    pngs = sorted(glob.glob(
        f"{O}/acquisition/{exp}/materialized_images/*/BF/projection/focus_stack/*_BF_t0000.png"))
    wells = {}
    for p in pngs:
        well = os.path.normpath(p).split(os.sep)[-5].split("_")[-1]
        if WELL_RE.match(well):
            wells[well] = p
        else:
            print(f"  [skip non-canonical well label] {well}")
    if not wells:
        print(f"{exp}: no images"); return

    # Uniform cell size from any one image (all wells share the same native size).
    cw, ch = load_cell(next(iter(wells.values()))).size

    sheet_w = HDR + NCOLS * (cw + PAD) + PAD
    sheet_h = TITLE_H + HDR + len(ROWS) * (ch + PAD) + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    # (ii) plate ID printed on the sheet
    draw.text((6, 4), f"{exp}   (n={len(wells)} wells)", fill=(0, 0, 0), font=FONT_BIG)

    # column-number headers (1..12)
    for c in range(NCOLS):
        x = HDR + c * (cw + PAD) + PAD + cw // 2 - 4
        draw.text((x, TITLE_H + 4), str(c + 1), fill=(0, 0, 0), font=FONT)

    # (iii) place every present image at its TRUE (row, col); missing wells stay blank
    for r, rowletter in enumerate(ROWS):
        y = TITLE_H + HDR + r * (ch + PAD)
        draw.text((5, y + ch // 2 - 6), rowletter, fill=(0, 0, 0), font=FONT)   # row-letter header
        for c in range(NCOLS):
            well = f"{rowletter}{c + 1:02d}"
            if well not in wells:
                continue
            x = HDR + c * (cw + PAD) + PAD
            sheet.paste(label_cell(load_cell(wells[well]), well), (x, y))

    out = f"{RESULT}/plate_{exp}.png"
    sheet.save(out)
    print(f"{exp}: {len(wells)} wells -> wrote {out}")


def main():
    for exp in EXPERIMENTS:
        build_plate(exp)


if __name__ == "__main__":
    main()
