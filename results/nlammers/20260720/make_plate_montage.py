#!/usr/bin/env python
"""96-well plate montages of materialized FF images for the 20250612* experiments.

Lays out one 8-row x 12-col plate grid per experiment (A01..H12), each cell labelled with its well
id and genotype. Wells with no materialized image are drawn as an explicit "no image" placeholder
rather than silently skipped, so a hole in the plate is visible instead of shifting the grid.

Two layout facts worth knowing:
  * Keyence FF projections here are TALL strips (1440 x 3420 -- a 3-tile vertical mosaic), not
    squares. Cells preserve that aspect ratio; forcing square thumbnails would squash the embryos.
  * Genotype comes from the `genotype` sheet of the plate metadata workbook, which is stored as a
    literal plate grid (col 0 = row letters A-H, row 0 = column numbers 1-12), so it is read
    positionally and flattened to a {well_id: genotype} map.

Usage:
    python make_plate_montage.py --experiment 20250612_24hpf_ctrl_atf6   # single (test)
    python make_plate_montage.py --all                                   # all 20250612*
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ACQ = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/output/acquisition"
META = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/input/plate_metadata"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

ROWS = "ABCDEFGH"
COLS = range(1, 13)
CELL_W = 200          # per-cell image width in px; height follows the source aspect ratio
LABEL_H = 48          # strip under each image: well id / genotype / temperature (3 lines)
PAD = 6
BG = (255, 255, 255)
FG = (20, 20, 20)
MUTED = (140, 140, 140)


def _font(size: int):
    """Best-effort TrueType; PIL's bitmap default ignores size, so fall back gracefully."""
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_plate_grid(experiment: str, sheet: str) -> dict[str, str]:
    """Read any plate-grid metadata sheet into {well_id: value}.

    The workbook stores these sheets as a literal plate map (col 0 = row letters A-H, row 0 =
    column numbers 1-12), so it is read positionally and flattened. Shared by `genotype` and
    `temperature`, which use the identical layout.
    """
    path = os.path.join(META, f"{experiment}_well_metadata.xlsx")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
    except ValueError:  # sheet absent in this workbook
        return {}
    out: dict[str, str] = {}
    for r in range(1, df.shape[0]):
        row_letter = str(df.iloc[r, 0]).strip()
        if row_letter.upper() not in ROWS:
            continue
        for c in range(1, df.shape[1]):
            try:
                col_num = int(float(str(df.iloc[0, c]).strip()))
            except (ValueError, TypeError):
                continue
            val = df.iloc[r, c]
            if pd.isna(val):
                continue
            # Trim trailing ".0" so 34.0 reads as "34" but 28.5 stays "28.5".
            if isinstance(val, float) and val.is_integer():
                text = str(int(val))
            else:
                text = str(val).strip()
            out[f"{row_letter.upper()}{col_num:02d}"] = text
    return out


def find_ff_images(experiment: str) -> dict[str, str]:
    """Map well_id -> FF projection png (earliest timepoint if several)."""
    pat = os.path.join(
        ACQ, experiment, "materialized_images", "*", "BF", "projection", "focus_stack", "*.png"
    )
    by_well: dict[str, str] = {}
    for p in sorted(glob.glob(pat)):
        m = re.search(r"_([A-H]\d{2})_BF_t(\d+)\.png$", os.path.basename(p))
        if not m:
            continue
        well = m.group(1)
        # keep the earliest timepoint so every cell is the same acquisition stage
        if well not in by_well or m.group(2) < re.search(r"_t(\d+)\.png$", by_well[well]).group(1):
            by_well[well] = p
    return by_well


def build_montage(experiment: str) -> str:
    genos = load_plate_grid(experiment, "genotype")
    temps = load_plate_grid(experiment, "temperature")
    imgs = find_ff_images(experiment)
    if not imgs:
        raise SystemExit(f"No FF images found for {experiment}")

    # Cell height from the true aspect ratio of a real image (these are tall 3-tile strips).
    with Image.open(next(iter(imgs.values()))) as probe:
        src_w, src_h = probe.size
    cell_h = max(1, int(round(CELL_W * src_h / src_w)))

    f_lab = _font(15)
    f_hdr = _font(20)
    f_title = _font(28)

    hdr = 40           # column-number strip
    row_lab = 34       # row-letter gutter
    title_h = 54
    tile_w = CELL_W + PAD
    tile_h = cell_h + LABEL_H + PAD
    W = row_lab + 12 * tile_w + PAD
    H = title_h + hdr + 8 * tile_h + PAD

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    d.text((PAD, PAD + 6), f"{experiment}  —  FF projections (96-well)", font=f_title, fill=FG)
    n_found = sum(1 for r in ROWS for c in COLS if f"{r}{c:02d}" in imgs)
    d.text((PAD, PAD + 34), f"{n_found}/96 wells with images", font=f_lab, fill=MUTED)

    for ci, c in enumerate(COLS):
        x = row_lab + ci * tile_w + CELL_W // 2
        d.text((x - 6, title_h + 10), f"{c}", font=f_hdr, fill=FG)

    for ri, r in enumerate(ROWS):
        y = title_h + hdr + ri * tile_h
        d.text((PAD, y + cell_h // 2), r, font=f_hdr, fill=FG)
        for ci, c in enumerate(COLS):
            well = f"{r}{c:02d}"
            x = row_lab + ci * tile_w
            path = imgs.get(well)
            if path:
                with Image.open(path) as im:
                    canvas.paste(im.convert("RGB").resize((CELL_W, cell_h), Image.LANCZOS), (x, y))
            else:
                d.rectangle([x, y, x + CELL_W, y + cell_h], outline=(210, 210, 210), fill=(245, 245, 245))
                d.text((x + CELL_W // 2 - 26, y + cell_h // 2 - 8), "no image", font=f_lab, fill=MUTED)
            g = genos.get(well, "?")
            if len(g) > 22:
                g = g[:21] + "…"
            t = temps.get(well)
            d.text((x + 2, y + cell_h + 3), well, font=f_lab, fill=FG)
            d.text((x + 2, y + cell_h + 17), g, font=f_lab, fill=MUTED)
            if t is not None:
                d.text((x + 2, y + cell_h + 31), f"{t}°C", font=f_lab, fill=MUTED)

    out = os.path.join(OUTDIR, f"{experiment}_plate_montage.png")
    canvas.save(out, "PNG")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        exps = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(ACQ, "20250612*")) if os.path.isdir(p)
        )
    elif a.experiment:
        exps = [a.experiment]
    else:
        raise SystemExit("pass --experiment <id> or --all")
    for e in exps:
        out = build_montage(e)
        print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
