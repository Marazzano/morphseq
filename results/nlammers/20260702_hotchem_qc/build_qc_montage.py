#!/usr/bin/env python
"""Montages of hotchem embryo snips AND raw projected well images, keyed to snip_qc.

Produces, under RESULT_DIR:
  montage_snip_<exp>.png    per-experiment snip (cropped embryo) montage
  montage_raw_<exp>.png     per-experiment raw BF-projection montage (one per materialized well)
  montage_snip_ALL.png      combined snip montage across all six
  montage_raw_ALL.png       combined raw montage across all six
  cohort_pass_rates.csv     pass/fail counts per (temperature, chem_perturbation) cohort

Each tile is labelled  "<well> <temp>C <chem>"  and border-colored by QC:
  green = passing (use_snip True), red = failed QC, gray = no snip produced / no verdict.
Raw tiles are colored by the WELL's snip outcome, so a red/gray RAW tile lets you judge whether
a bad result is a computational miss vs. a genuinely low-quality well image.

Run with the pipeline env:
  conda run -n segmentation_grounded_sam --no-capture-output python build_qc_montage.py
"""
from __future__ import annotations
import glob, os, math, csv
from collections import defaultdict
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT_ROOT = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline_output"
RESULT_DIR = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20260702_hotchem_qc"
EXPERIMENTS = [
    "20260702_hotchem_24hpf_plate01", "20260702_hotchem_24hpf_plate02",
    "20260702_hotchem_30hpf_plate01", "20260702_hotchem_30hpf_plate02",
    "20260702_hotchem_36hpf_plate01", "20260702_hotchem_36hpf_plate02",
]

THUMB_W, THUMB_H = 96, 216
BORDER, PAD, LABEL_H, TITLE_H, COLS = 4, 2, 16, 22, 16
GREEN, RED, GRAY = (40, 170, 60), (200, 40, 40), (140, 140, 140)

os.makedirs(RESULT_DIR, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 9)
except Exception:
    FONT = ImageFont.load_default()


def load_meta(exp):
    p = f"{OUT_ROOT}/acquisition/{exp}/ingest_metadata/plate_metadata.csv"
    m = {}
    if os.path.exists(p):
        df = pd.read_csv(p, dtype=str)
        for _, r in df.iterrows():
            m[str(r["well_id"])] = (str(r.get("temperature", "?")), str(r.get("chem_perturbation", "?")))
    else:
        print(f"  [warn] no plate_metadata for {exp}")
    return m


def load_verdicts(exp):
    """Return (snip_map, well_status). snip_map[snip_id]=(well_id,use,reasons);
    well_status[well_id] in {'pass','fail'} for wells that produced a snip."""
    # Prefer the merged table; fall back to per-well shards so PARTIAL/in-progress runs work.
    merged = f"{OUT_ROOT}/quality_control/{exp}/snip_qc/{exp}_snip_qc.parquet"
    parts = [merged] if os.path.exists(merged) else sorted(
        glob.glob(f"{OUT_ROOT}/quality_control/{exp}/snip_qc/per_well/*/*_snip_qc.parquet"))
    snip_map, well_use = {}, defaultdict(list)
    if not parts:
        print(f"  [warn] no snip_qc yet for {exp}")
    for p in parts:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        for _, r in df.iterrows():
            sid, wid, use, why = str(r["snip_id"]), str(r["well_id"]), bool(r["use_snip"]), str(r.get("qc_fail_reasons") or "")
            snip_map[sid] = (wid, use, why)
            well_use[wid].append(use)
    well_status = {w: ("pass" if any(v) else "fail") for w, v in well_use.items()}
    return snip_map, well_status


def make_cell(img_path, color, label):
    cw, ch = THUMB_W + 2 * BORDER, THUMB_H + 2 * BORDER + LABEL_H
    canvas = Image.new("RGB", (cw, ch), color)
    try:
        im = Image.open(img_path).convert("RGB").resize((THUMB_W, THUMB_H))
    except Exception:
        im = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(im, (BORDER, BORDER))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, ch - LABEL_H, cw, ch], fill=(0, 0, 0))
    d.text((2, ch - LABEL_H), label[:22], fill=(255, 255, 255), font=FONT)
    return canvas


def montage(cells, title, out_path):
    if not cells:
        print(f"  [skip] {title}: no cells")
        return
    cw, ch = cells[0].size
    cols = min(COLS, len(cells)); rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * (cw + PAD) + PAD, TITLE_H + rows * (ch + PAD) + PAD), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0), font=FONT)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        sheet.paste(c, (PAD + col * (cw + PAD), TITLE_H + PAD + r * (ch + PAD)))
    sheet.save(out_path); print(f"  wrote {out_path} ({len(cells)} tiles)")


def well_of(path, depth):
    return os.path.normpath(path).split(os.sep)[depth]


def main():
    all_snip, all_raw = [], []
    cohort = defaultdict(lambda: {"wells": 0, "well_pass": 0, "well_fail": 0, "well_nosnip": 0,
                                  "snips": 0, "snip_pass": 0, "snip_fail": 0})
    for exp in EXPERIMENTS:
        meta = load_meta(exp)
        snip_map, well_status = load_verdicts(exp)

        def lbl(wid):
            t, c = meta.get(wid, ("?", "?"))
            return f"{wid.split('_')[-1]} {t}C {c}"

        # ---- snip tiles ----
        snip_cells = []
        for p in sorted(g for g in glob.glob(
                f"{OUT_ROOT}/object_extraction/{exp}/snips/per_well/*/snips/*/*.png") if not g.endswith("_embryo.png")):
            sid = os.path.basename(p)[:-4]
            wid, use, why = snip_map.get(sid, (None, None, ""))
            if wid is None:
                wid = well_of(p, -3)
            color = GRAY if use is None else (GREEN if use else RED)
            lab = lbl(wid) + ("" if use else f" !{why.split('|')[0][:8]}")
            snip_cells.append(make_cell(p, color, lab))
        montage(snip_cells, f"SNIP {exp}", f"{RESULT_DIR}/montage_snip_{exp}.png")
        all_snip.extend(snip_cells)

        # ---- raw projection tiles (all materialized wells) ----
        raw_cells = []
        raws = sorted(glob.glob(f"{OUT_ROOT}/acquisition/{exp}/materialized_images/*/BF/projection/focus_stack/*_BF_t0000.png"))
        for p in raws:
            wid = well_of(p, -5)                       # .../materialized_images/{well_id}/BF/projection/focus_stack/file
            status = well_status.get(wid, "nosnip")
            color = {"pass": GREEN, "fail": RED, "nosnip": GRAY}[status]
            raw_cells.append(make_cell(p, color, lbl(wid)))
            # cohort accounting is well-level, driven by the raw (full-well) set
            t, c = meta.get(wid, ("?", "?")); key = f"{t}C_{c}"
            k = cohort[key]; k["wells"] += 1
            k[{"pass": "well_pass", "fail": "well_fail", "nosnip": "well_nosnip"}[status]] += 1
        montage(raw_cells, f"RAW {exp}", f"{RESULT_DIR}/montage_raw_{exp}.png")
        all_raw.extend(raw_cells)

        # snip-level cohort counts
        for sid, (wid, use, why) in snip_map.items():
            t, c = meta.get(wid, ("?", "?")); k = cohort[f"{t}C_{c}"]
            k["snips"] += 1; k["snip_pass" if use else "snip_fail"] += 1

    montage(all_snip, "ALL SNIP — green=use, red=fail, gray=no verdict", f"{RESULT_DIR}/montage_snip_ALL.png")
    montage(all_raw, "ALL RAW — border=well QC outcome (green pass / red fail / gray no snip)", f"{RESULT_DIR}/montage_raw_ALL.png")

    # ---- cohort pass-rate table ----
    csv_path = f"{RESULT_DIR}/cohort_pass_rates.csv"
    cols = ["cohort", "wells", "well_pass", "well_fail", "well_nosnip", "snips", "snip_pass", "snip_fail"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for key in sorted(cohort):
            r = cohort[key]; w.writerow([key] + [r[c] for c in cols[1:]])
    print(f"\n  wrote {csv_path}")
    print("\n=== cohort pass rates (temp-chem) ===")
    print(f"{'cohort':<16}{'wells':>6}{'pass':>6}{'fail':>6}{'nosnip':>7}   {'snips':>6}{'spass':>6}{'sfail':>6}")
    for key in sorted(cohort):
        r = cohort[key]
        print(f"{key:<16}{r['wells']:>6}{r['well_pass']:>6}{r['well_fail']:>6}{r['well_nosnip']:>7}   {r['snips']:>6}{r['snip_pass']:>6}{r['snip_fail']:>6}")


if __name__ == "__main__":
    main()
