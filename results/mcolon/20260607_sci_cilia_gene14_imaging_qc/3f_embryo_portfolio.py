"""
3f - Per-embryo image portfolio (the QC canvas / greenlight spot-check artifact).

A per-embryo image grid: each cell is an embryo's SNAPSHOT image with its true genotype /
predicted genotype / predicted phenotype / QC status above it. The primary visual-QC tool
for (a) confirming good image data goes into the model, (b) eyeballing model calls vs truth,
and (c) catching embryos that were wrongly excluded or silently dropped.

DATA SCOPE — SNAPSHOTS ONLY (non-SKY / non-`_sci_`). PLAN.md LOUD rule: `_t01`/`_t02`
snapshots are for portfolio / QC spot-checking ONLY, never the model; you cannot put a
timeseries in a contact sheet. The `_sci_` timeseries plates are excluded here. The model's
predicted-label badge for a plate01 embryo is still the timeseries-based call (script 2
applied timeseries-priority dedup upstream) — we just show it on the snapshot image.

SPINE = the 3a audit (`tables/embryo_loss_map.csv`), one row per sequenced snapshot well,
keyed on `embryo_id`. ONE AUDIT ROW = ONE CARD. No sequenced embryo disappears: rows with no
model output (EXCLUDED / ABSENT_IMAGED / OK-but-no-prediction) become labeled placeholder /
raw-well cards, never dropped.

LABELS come from TWO model files, joined on `embryo_id`:
  - predicted genotype  <- predictions/sequenced_genotype_qc_cross_bin.csv
  - predicted phenotype <- predictions/sequenced_homozygous_phenotype_cross_bin.csv
    (homozygous-only by design; non-homozygous embryos legitimately get no phenotype call).

IMAGE per cell, tried in order:
  1. embryo snip  : training_data/bf_embryo_snips/<exp>/<embryo_id>_t*.jpg
  2. raw FF well  : built_image_data/stitched_FF_images/<exp>/<well>_t*_stitch.jpg  (fallback
                    for ABSENT/EXCLUDED — lets us SEE why the well failed, e.g. out of focus)
  3. none         : gray placeholder.
Any image is fit into a fixed snip-shaped frame (rotate longest-side -> longest-side, then
scale-to-fit, no crop) so the full QC region stays visible and future image types just work.

VIEWS (like the legacy make_sequenced_portfolio_views.py):
  - by_plate     : one page per experiment (plate).
  - by_gene_time : one page per (gene, stratum, collection-stage).
Plus per-(gene, stratum, stage) standalone PNGs for slide drop-in.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3f_embryo_portfolio.py
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))  # local plot_config

from plot_config import STATUS_COLORS, color_for, snap_to_design_stage  # noqa: E402

# Reuse script 0's single source of truth for collection-time + plate parsing so the
# reconstructed physical_embryo_id matches the predictions exactly (the module name starts
# with a digit, so import it via importlib rather than a bare `import`).
_spec = importlib.util.spec_from_file_location(
    "load_and_clean_datasets", RUN_DIR / "0_load_and_clean_datasets.py")
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
COLLECTION_TIME_HPF = _loader.COLLECTION_TIME_HPF
plate_token = _loader.plate_token

TABLE_DIR = RUN_DIR / "tables"
PRED_DIR = RUN_DIR / "predictions"
OUT_DIR = RUN_DIR / "plots" / "portfolio"
SNIP_ROOT = PROJECT_ROOT / "morphseq_playground" / "training_data" / "bf_embryo_snips"
FF_ROOT = PROJECT_ROOT / "morphseq_playground" / "built_image_data" / "stitched_FF_images"

# Fixed cell frame = the normal snip geometry (portrait, long side vertical). Any image is
# rotated so its long side matches this frame's long side, then scaled to fit.
FRAME_W, FRAME_H = 256, 576

NCOLS = 6
GENES = ["b9d2", "cep290", "crispant"]

STRATUM_ORDER = {
    "AB": 0, "wildtype_sibling": 1, "heterozygous": 2,
    "homozygous": 3, "mutant_unresolved": 4, "unknown": 99,
}

# Scored truth labels per gene (mirrors 3b GENO_LABELS) — the only truths the genotype-QC
# model is evaluated against. Truths outside this set (e.g. crispant `injection_control`)
# are not "wrong calls"; don't flag them with the red mismatch border.
SCORED_TRUTHS = {
    "b9d2": {"wildtype", "heterozygous", "homozygous"},
    "cep290": {"wildtype", "heterozygous", "homozygous"},
    "crispant": {"ab_wildtype", "foxj1a_crispant", "ift88_crispant",
                 "ift88_ift74_crispant", "sspo_crispant"},
}

SHORT = {
    "wildtype": "WT", "heterozygous": "HET", "homozygous": "HOM",
    "High_to_Low": "HTL", "Low_to_High": "LTH", "HTA": "HTA", "CE": "CE",
    "ab_wildtype": "AB", "injection_control": "inj-ctrl",
    "foxj1a_crispant": "foxj1a", "ift88_crispant": "ift88",
    "ift88_ift74_crispant": "ift88/74", "sspo_crispant": "sspo",
}


# ---------------------------------------------------------------------------- fonts

def _font(size: int, bold: bool = False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(30, True)
FONT_HEADER = _font(15, True)
FONT_SMALL = _font(13)
FONT_TINY = _font(12)
FONT_ID = _font(10)        # long embryo_id strings — small so the full plate name fits
FONT_WELL = _font(14, True)  # well + collection time, the readable identity line


# ---------------------------------------------------------------------------- helpers

def clean(value, empty: str = "NA") -> str:
    if pd.isna(value):
        return empty
    s = str(value).strip()
    return empty if s in ("", "nan") else s


def short(label: str) -> str:
    return SHORT.get(label, label)


def gene_from_exp(exp: str) -> str:
    for g in GENES:
        if g in exp:
            return g
    return "unknown"


def age_from_exp(exp: str):
    if "30to48" in exp:
        return 48
    m = re.findall(r"(\d+)hpf", exp)
    return int(m[-1]) if m else None


def physical_id(exp: str, well: str) -> str:
    """Reconstruct script-0's physical_embryo_id (biology-keyed, collection-time NOT exp).

    Mirrors 0_load_and_clean_datasets.py: {gene}_{ct}hpf_{plate}_{well}. This is the key that
    bridges a plate01 `_t02` snapshot to its `_sci_` timeseries sibling, so the snapshot card
    can borrow the model call that was filed under the timeseries embryo_id.
    """
    gene = gene_from_exp(exp)
    ct = COLLECTION_TIME_HPF.get(exp)
    tok = f"{int(ct)}hpf" if ct is not None else "naHpf"
    return f"{gene}_{tok}_{plate_token(exp)}_{well}"


def resolve_snip(embryo_id: str, exp: str) -> Path | None:
    """Cropped embryo snip <exp>/<embryo_id>_t*.<ext> (the normal portfolio image)."""
    base = SNIP_ROOT / exp
    if not base.is_dir():
        return None
    hits = sorted(base.glob(f"{embryo_id}_t*.jpg")) or sorted(base.glob(f"{embryo_id}.*"))
    return hits[0] if hits else None


def resolve_ff(well: str, exp: str) -> Path | None:
    """Raw stitched full-focus well image — fallback so we can SEE why a well failed."""
    base = FF_ROOT / exp
    if not base.is_dir():
        return None
    hits = sorted(base.glob(f"{well}_t*_stitch.jpg")) or sorted(base.glob(f"{well}_*stitch*"))
    return hits[0] if hits else None


def fit_into_frame(img: Image.Image, frame_w: int, frame_h: int) -> Image.Image:
    """Snap the image's longest side to the frame's longest side, then scale-to-fit (no crop).

    Image-source-agnostic: rotate 90 deg only if the long-axis orientations disagree, then
    letterbox-center inside a fixed frame. Adding other image types later needs no new logic.
    """
    src = img.convert("RGB")
    frame_portrait = frame_h >= frame_w
    img_portrait = src.height >= src.width
    if img_portrait != frame_portrait:
        src = src.rotate(90, expand=True)
    src = ImageOps.contain(src, (frame_w, frame_h), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (frame_w, frame_h), "#111111")
    canvas.paste(src, ((frame_w - src.width) // 2, (frame_h - src.height) // 2))
    return canvas


# ---------------------------------------------------------------------------- manifest

def build_manifest() -> pd.DataFrame:
    audit = pd.read_csv(TABLE_DIR / "embryo_loss_map.csv", low_memory=False)
    geno_all = pd.read_csv(PRED_DIR / "sequenced_genotype_qc_cross_bin.csv", low_memory=False)
    phen_all = pd.read_csv(PRED_DIR / "sequenced_homozygous_phenotype_cross_bin.csv", low_memory=False)

    # SNAPSHOTS feed the cards, but the _sci_ TIMESERIES predictions are kept aside: a plate01
    # `_t02` snapshot has no prediction under its own embryo_id (the call was filed under the
    # timeseries sibling), so we bridge it back via physical_embryo_id below.
    is_sci_g = geno_all["experiment"].str.contains("_sci_")
    is_sci_p = phen_all["experiment"].str.contains("_sci_")
    geno, sci_geno = geno_all[~is_sci_g].copy(), geno_all[is_sci_g].copy()
    phen, sci_phen = phen_all[~is_sci_p].copy(), phen_all[is_sci_p].copy()

    geno = geno.rename(columns={"predicted_label": "pred_genotype",
                                "top_probability": "pred_genotype_prob"})
    phen_small = phen[["embryo_id", "predicted_label"]].rename(
        columns={"predicted_label": "pred_phenotype"})

    pred = geno.merge(phen_small, on="embryo_id", how="left")
    pred_cols = ["embryo_id", "pred_genotype", "pred_genotype_prob", "pred_phenotype",
                 "zygosity", "genotype_clean", "gene", "sequenced_stratum",
                 "collection_time_hpf", "data_source"]
    pred = pred[[c for c in pred_cols if c in pred.columns]]

    # Audit is the SPINE. Left-join predictions on embryo_id (audit rows with no model output
    # keep NaN labels and become placeholder / raw-well cards).
    man = audit.merge(pred, on="embryo_id", how="left")
    man["physical_embryo_id"] = man.apply(
        lambda r: physical_id(r["exp"], r["well"]), axis=1)
    man["pred_source"] = man["pred_genotype"].notna().map({True: "direct", False: ""})

    # BRIDGE: for plate01 `_t02` snapshots that got no direct prediction, borrow the label
    # filed under the `_sci_` timeseries sibling, joined on physical_embryo_id.
    sci_geno_lab = sci_geno.set_index("physical_embryo_id")
    sci_phen_lab = sci_phen.set_index("physical_embryo_id")["predicted_label"]
    sci_geno_lab = sci_geno_lab[~sci_geno_lab.index.duplicated(keep="first")]
    sci_phen_lab = sci_phen_lab[~sci_phen_lab.index.duplicated(keep="first")]

    need = ~man["pred_genotype"].notna()
    pid = man["physical_embryo_id"]
    bridged = need & pid.isin(sci_geno_lab.index)
    man.loc[bridged, "pred_genotype"] = pid[bridged].map(sci_geno_lab["predicted_label"])
    man.loc[bridged, "pred_genotype_prob"] = pid[bridged].map(sci_geno_lab["top_probability"])
    man.loc[bridged, "pred_phenotype"] = pid[bridged].map(sci_phen_lab)
    # Borrow the timeseries truth labels too (the snapshot well's zygosity wasn't resolved).
    for col in ("zygosity", "genotype_clean", "sequenced_stratum"):
        if col in sci_geno_lab.columns:
            man.loc[bridged, col] = pid[bridged].map(sci_geno_lab[col])
    man.loc[bridged, "pred_source"] = "timeseries(plate01)"

    man["has_prediction"] = man["pred_genotype"].notna()
    n_bridged = int(bridged.sum())

    # Fill the three page-group keys for EVERY row (audit-only rows lack prediction metadata;
    # recover gene + stage from the experiment name — verified 0 unknown / 0 null).
    man["gene"] = man["gene"].where(man["gene"].notna(), man["exp"].map(gene_from_exp))

    def stage_for(row):
        if pd.notna(row.get("collection_time_hpf")):
            s = snap_to_design_stage(row["collection_time_hpf"])
            if s is not None:
                return s
        a = age_from_exp(row["exp"])
        return a if a is not None else "missing"

    man["collection_stage"] = man.apply(stage_for, axis=1)

    # stratum: prediction stratum if present, else coarse from seq_code, else unknown.
    seq_stratum = {1: "wildtype_sibling", 2: "mutant_unresolved"}
    man["stratum"] = man["sequenced_stratum"].where(
        man["sequenced_stratum"].notna() & (man["sequenced_stratum"].astype(str) != ""),
        man["seq_code"].map(seq_stratum),
    ).fillna("unknown")

    # Truth genotype: zygosity for b9d2/cep290, genotype_clean for crispant (no zygosity).
    def truth_label(row):
        if row["gene"] == "crispant":
            return clean(row.get("genotype_clean"), "")
        return clean(row.get("zygosity"), "")
    man["truth_genotype"] = man.apply(truth_label, axis=1)

    # Resolve image: snip first, FF well fallback, else none.
    snip_paths, ff_paths, sources = [], [], []
    for _, r in man.iterrows():
        sp = resolve_snip(r["embryo_id"], r["exp"])
        if sp is not None:
            snip_paths.append(str(sp)); ff_paths.append(""); sources.append("snip")
            continue
        fp = resolve_ff(r["well"], r["exp"])
        if fp is not None:
            snip_paths.append(""); ff_paths.append(str(fp)); sources.append("ff")
        else:
            snip_paths.append(""); ff_paths.append(""); sources.append("none")
    man["snip_path"] = snip_paths
    man["ff_path"] = ff_paths
    man["image_source"] = sources
    man["image_path"] = man["snip_path"].where(man["snip_path"] != "", man["ff_path"])

    return man, len(audit), pred["embryo_id"].nunique(), n_bridged


# ---------------------------------------------------------------------------- card

def draw_badge(d: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str,
               color: str, max_w: int) -> None:
    d.rectangle([x, y, x + 10, y + 14], fill=color, outline="#222222")
    d.text((x + 14, y - 1), f"{label}: {value}", fill="#111111", font=FONT_TINY)


def card(row: pd.Series, card_w: int, card_h: int, head_h: int) -> Image.Image:
    status = clean(row.get("status"), "OK")
    excluded = status == "EXCLUDED"
    ff = row.get("image_source") == "ff"
    none_img = row.get("image_source") == "none"
    bg = "#fff4f4" if excluded else ("#f4f7ff" if ff else "#ffffff")
    im = Image.new("RGB", (card_w, card_h), bg)
    d = ImageDraw.Draw(im)
    # red border if the genotype call is wrong (both known); FF border to flag raw-well cells.
    border, bw = "#beb6aa", 1
    truth, pred_g = clean(row.get("truth_genotype"), ""), clean(row.get("pred_genotype"), "")
    scored = truth in SCORED_TRUTHS.get(row.get("gene"), set())
    if scored and pred_g not in ("", "NA") and truth != pred_g:
        border, bw = "#B2182B", 3
    elif ff:
        border, bw = "#3b6fb5", 2
    d.rectangle([0, 0, card_w - 1, card_h - 1], outline=border, width=bw)

    y = 5
    # Identity line: well (big, readable) + collection time. Then the full embryo_id small
    # underneath so the long `..._30to48hpf_plate01_t02_E07_e01` name fits without clipping.
    ct = row.get("collection_time_hpf")
    if pd.isna(ct):
        ct = age_from_exp(row["exp"])
    ct_lbl = f"{int(ct)} hpf" if pd.notna(ct) else "hpf?"
    d.text((8, y), f"{clean(row.get('well'), '?')} · {ct_lbl} collection",
           fill="#111111", font=FONT_WELL)
    y += 16
    d.text((8, y), clean(row.get("embryo_id"), ""), fill="#555555", font=FONT_ID)
    y += 14

    # truth genotype (or seq-only coarse / unknown)
    if truth:
        kind = "genotype"
        draw_badge(d, 8, y, "truth", short(truth), color_for(truth, kind), card_w)
    else:
        coarse = {1: "wt-side", 2: "mutant"}.get(row.get("seq_code"), "?")
        draw_badge(d, 8, y, "truth", f"{coarse} (seq only)", "#808080", card_w)
    y += 17

    # predicted genotype (or no-model reason)
    if row.get("has_prediction"):
        draw_badge(d, 8, y, "pred", short(pred_g), color_for(pred_g, "genotype"), card_w)
    else:
        draw_badge(d, 8, y, "pred", f"— (no model: {status})", "#777777", card_w)
    y += 17

    # predicted phenotype: present / not-homozygous / no-model
    phen = clean(row.get("pred_phenotype"), "")
    if phen:
        draw_badge(d, 8, y, "phen", short(phen), color_for(phen, "phenotype"), card_w)
    elif row.get("has_prediction"):
        draw_badge(d, 8, y, "phen", "n/a (not homozygous)", "#cccccc", card_w)
    else:
        draw_badge(d, 8, y, "phen", "— (no model output)", "#777777", card_w)
    y += 17

    # QC status (+ reason)
    qc_txt = status
    if excluded:
        qc_txt = f"EXCLUDED ({clean(row.get('exclusion_flags'), 'qc')})"
    elif status == "ABSENT_IMAGED":
        qc_txt = "ABSENT_IMAGED (imaged, not detected)"
    draw_badge(d, 8, y, "qc", qc_txt, STATUS_COLORS.get(status, "#777777"), card_w)
    y += 17

    # notes line: image source (when not a normal snip) + bridged-prediction provenance.
    notes = []
    if ff:
        notes.append("FF raw well")
    elif none_img:
        notes.append("no image")
    if row.get("pred_source") == "timeseries(plate01)":
        notes.append("pred via timeseries")
    if notes:
        d.text((8, y), " · ".join(notes), fill="#3b6fb5", font=FONT_TINY)

    # image
    box = (6, head_h, card_w - 6, card_h - 6)
    bw_img, bh_img = box[2] - box[0], box[3] - box[1]
    if none_img:
        d.rectangle(box, fill="#eeeeee", outline="#bbbbbb")
        d.text((box[0] + 14, box[1] + bh_img // 2 - 8),
               clean(row.get("status"), "no image"), fill="#777777", font=FONT_HEADER)
    else:
        try:
            framed = fit_into_frame(Image.open(row["image_path"]), bw_img, bh_img)
            im.paste(framed, (box[0], box[1]))
        except Exception:
            d.rectangle(box, fill="#eeeeee", outline="#bbbbbb")
            d.text((box[0] + 12, box[1] + 20), "image read error",
                   fill="#777777", font=FONT_HEADER)
    return im


# ---------------------------------------------------------------------------- pages

def render_page(sub: pd.DataFrame, title: str, cols: int) -> Image.Image:
    # Card geometry: header band + the fixed image frame.
    head_h = 122
    card_w = FRAME_W + 12
    card_h = head_h + FRAME_H + 12
    rows = max(1, math.ceil(len(sub) / cols))
    margin, gap, title_h = 24, 8, 52
    page_w = margin * 2 + cols * card_w + (cols - 1) * gap
    page_h = title_h + rows * card_h + (rows - 1) * gap + margin
    page = Image.new("RGB", (page_w, page_h), "#f5f1e9")
    d = ImageDraw.Draw(page)
    d.text((margin, 12), title, fill="#111111", font=FONT_TITLE)
    for i, (_, r) in enumerate(sub.iterrows()):
        x = margin + (i % cols) * (card_w + gap)
        y = title_h + (i // cols) * (card_h + gap)
        page.paste(card(r, card_w, card_h, head_h), (x, y))
    return page


def stratum_sort(s: str) -> int:
    return STRATUM_ORDER.get(str(s), 50)


def save_pages(pages: list[Image.Image], pdf_path: Path) -> None:
    if not pages:
        return
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])


def view_by_plate(man: pd.DataFrame) -> None:
    pages, page_dir = [], OUT_DIR / "by_plate_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(man["exp"].unique(),
                  key=lambda e: (gene_from_exp(e), age_from_exp(e) or 0, e))
    total = 0
    for idx, exp in enumerate(keys, 1):
        sub = man[man["exp"] == exp].sort_values(
            ["stratum", "well", "embryo_id"], key=lambda c: c.map(stratum_sort)
            if c.name == "stratum" else c, kind="stable")
        total += len(sub)
        title = f"{exp} | n={len(sub)} sequenced"
        page = render_page(sub, title, NCOLS)
        pages.append(page)
        page.save(page_dir / f"{idx:02d}_{exp}.png")
        print(f"    by_plate: {exp}  n={len(sub)}")
    save_pages(pages, OUT_DIR / "portfolio_by_plate.pdf")
    print(f"  wrote portfolio_by_plate.pdf ({len(pages)} pages, {total} cards)")
    return total


def view_by_gene_time(man: pd.DataFrame) -> None:
    pages, page_dir = [], OUT_DIR / "by_gene_time_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale standalone PNGs from prior runs so the directory reflects only this run's
    # groups (group membership shifts when bridged predictions move embryos between strata).
    for old in OUT_DIR.glob("*__*__*.png"):
        old.unlink()
    grp = man.groupby(["gene", "stratum", "collection_stage"], dropna=False)
    keys = sorted(grp.groups.keys(),
                  key=lambda k: (str(k[0]), stratum_sort(k[1]),
                                 float(k[2]) if str(k[2]) != "missing" else 1e9))
    total = 0
    for idx, key in enumerate(keys, 1):
        gene, stratum, stage = key
        sub = grp.get_group(key).sort_values(
            ["well", "embryo_id"], kind="stable")
        total += len(sub)
        stage_lbl = f"{int(stage)} hpf" if str(stage) != "missing" else "stage missing"
        title = f"{gene} | {stratum} | {stage_lbl} | n={len(sub)}"
        page = render_page(sub, title, NCOLS)
        pages.append(page)
        safe = f"{gene}__{stratum}__{stage}".replace("/", "_").replace(" ", "_")
        page.save(page_dir / f"{idx:02d}_{safe}.png")
        # also the standalone per-(gene,stratum,stage) PNG for slide drop-in
        page.save(OUT_DIR / f"{safe}.png")
        print(f"    by_gene_time: {gene} | {stratum} | {stage_lbl}  n={len(sub)}")
    save_pages(pages, OUT_DIR / "portfolio_by_gene_time.pdf")
    print(f"  wrote portfolio_by_gene_time.pdf ({len(pages)} pages, {total} cards)")
    return total


# ---------------------------------------------------------------------------- main

def main() -> None:
    print("3f - per-embryo image portfolio (SNAPSHOTS ONLY; audit is the spine)")
    print("  LOUD: _sci_ timeseries plates are EXCLUDED; plate01 badge = timeseries-based call.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    man, n_audit, n_pred, n_bridged = build_manifest()
    n_man = len(man)
    n_no_pred = int((~man["has_prediction"]).sum())
    n_no_snip = int((man["image_source"] != "snip").sum())
    n_none = int((man["image_source"] == "none").sum())

    man.to_csv(OUT_DIR / "portfolio_manifest.csv", index=False)

    # ---- reconciliation (live, no hard-coded totals) ----
    print("\n=== reconciliation ===")
    print(f"  n_audit_rows       = {n_audit}")
    print(f"  n_prediction_rows  = {n_pred}  (non-sci genotype preds)")
    print(f"  n_manifest_rows    = {n_man}")
    print(f"  n_bridged (plate01 _t02 borrowed timeseries call via physical_embryo_id) = {n_bridged}")
    print(f"  n_missing_prediction (silent-drop candidates, after bridge) = {n_no_pred}")
    print(f"  n_missing_snip (snip absent) = {n_no_snip}  | FF fallback = "
          f"{int((man['image_source']=='ff').sum())}  | no image = {n_none}")
    print("  silent-drops by audit status:")
    print(man[~man["has_prediction"]]["status"].value_counts().to_string())

    # ---- assertions ----
    assert man["embryo_id"].is_unique, "duplicate embryo_id in manifest"
    assert n_man == n_audit, f"left-join changed row count ({n_man} != {n_audit})"

    # ---- views ----
    print("\n=== view: by plate ===")
    t_plate = view_by_plate(man)
    print("\n=== view: by gene x stratum x time ===")
    t_gene = view_by_gene_time(man)

    assert t_plate == n_man, f"by_plate pages dropped/dup embryos ({t_plate} != {n_man})"
    assert t_gene == n_man, f"by_gene_time pages dropped/dup embryos ({t_gene} != {n_man})"

    print(f"\nDone. {n_man} cards across both views (== audit rows). "
          f"Outputs in {OUT_DIR.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
