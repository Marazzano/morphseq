#!/usr/bin/env python
"""Full-field montages built DIRECTLY from raw Keyence tiles (bypasses the broken pipeline stitch).

For each well: read its 3 raw tiles (z-stacks), max-project each over z, and stitch them per the
experiment's keyence_stitch_map coords into one full-field image. Then montage per plate + combined.

This is a VISUALIZATION workaround for the single-tile pipeline bug — max-projection, not the LoG
focus-stack, but interpretable for eyeballing well quality/content.

Run:  conda run -n segmentation_grounded_sam --no-capture-output python build_raw_fullfield_montage.py
"""
from __future__ import annotations
import glob, os, json, math, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from skimage import io as skio

RAW = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/Keyence"
O = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline_output"
RESULT = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20260702_hotchem_qc"
EXPERIMENTS = [
    "20260702_hotchem_24hpf_plate01", "20260702_hotchem_24hpf_plate02",
    "20260702_hotchem_30hpf_plate01", "20260702_hotchem_30hpf_plate02",
    "20260702_hotchem_36hpf_plate01", "20260702_hotchem_36hpf_plate02",
]
Z_STEP = 3          # use every 3rd z-plane for the max projection (speed)
THUMB_W = 70        # full-field is ~960x2160 (tall); thumbnail width
COLS = 16
os.makedirs(RESULT, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 9)
except Exception:
    FONT = ImageFont.load_default()
_TILE_RE = re.compile(r"_(\d{5})_Z(\d+)_CH1\.tif$", re.I)


def load_meta(exp):
    p = f"{O}/acquisition/{exp}/ingest_metadata/plate_metadata.csv"
    m = {}
    if os.path.exists(p):
        df = pd.read_csv(p, dtype=str)
        for _, r in df.iterrows():
            m[str(r["well_id"]).split("_")[-1]] = (str(r.get("temperature", "?")), str(r.get("chem_perturbation", "?")))
    return m


def load_coords(exp):
    p = f"{O}/acquisition/{exp}/ingest_metadata/keyence_stitch_map__keyence.json"
    c = json.load(open(p))["coords"]                 # {"1":[x,y], ...}
    return {int(k): (int(round(v[0])), int(round(v[1]))) for k, v in c.items()}


def well_of_xy(xydir):
    for f in os.listdir(xydir):
        m = re.match(r"^_([A-H]\d{1,2})$", f)
        if m:
            return m.group(1)
    return None


def maxproj_tile(xydir, tile):
    paths = sorted(p for p in glob.glob(os.path.join(xydir, "*_CH1.tif"))
                   if (mm := _TILE_RE.search(p)) and int(mm.group(1)) == tile)
    if not paths:
        return None
    paths = paths[::Z_STEP] or paths
    stack = np.stack([skio.imread(p) for p in paths], axis=0)
    return stack.max(axis=0)


def stitch_well(xydir, coords):
    tiles = {}
    for t in coords:
        img = maxproj_tile(xydir, t)
        if img is not None:
            tiles[t] = img
    if not tiles:
        return None
    h, w = next(iter(tiles.values())).shape[:2]
    W = max(coords[t][0] for t in tiles) + w
    H = max(coords[t][1] for t in tiles) + h
    canvas = np.zeros((H, W), dtype=np.float32)
    for t, img in tiles.items():
        x, y = coords[t]
        canvas[y:y + h, x:x + w] = img[:h, :w]
    lo, hi = np.percentile(canvas, (2, 98))            # contrast stretch
    canvas = np.clip((canvas - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas)


def cell(im, label):
    th = int(THUMB_W * im.size[1] / im.size[0])
    im = im.convert("L").resize((THUMB_W, th))
    canvas = Image.new("RGB", (THUMB_W, th + 14), (0, 0, 0))
    canvas.paste(im, (0, 0))
    ImageDraw.Draw(canvas).text((1, th), label[:16], fill=(255, 255, 100), font=FONT)
    return canvas


def montage(cells, title, out):
    if not cells:
        print("  [skip]", title); return
    cw = max(c.size[0] for c in cells); ch = max(c.size[1] for c in cells)
    cols = min(COLS, len(cells)); rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * (cw + 3) + 3, 20 + rows * (ch + 3) + 3), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0), font=FONT)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (3 + col * (cw + 3), 20 + r * (ch + 3)))
    sheet.save(out); print("  wrote", out, f"({len(cells)} wells)")


def main():
    allcells = []
    for exp in EXPERIMENTS:
        coords = load_coords(exp); meta = load_meta(exp)
        cells = []
        for xydir in sorted(glob.glob(f"{RAW}/{exp}/XY*")):
            if not os.path.isdir(xydir):
                continue
            slug = well_of_xy(xydir)
            if slug is None:
                continue
            im = stitch_well(xydir, coords)
            if im is None:
                continue
            t, c = meta.get(slug, ("?", "?"))
            cells.append(cell(im, f"{slug} {t}C {c}"))
        montage(cells, f"RAW FULL-FIELD {exp} (n={len(cells)})", f"{RESULT}/fullfield_{exp}.png")
        allcells.extend(cells)
    montage(allcells, "RAW FULL-FIELD — all 6 hotchem plates (3-tile stitch, max-proj)",
            f"{RESULT}/fullfield_ALL.png")


if __name__ == "__main__":
    main()
