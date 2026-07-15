"""SAM3 open-vocabulary text-prompt benchmark + visual gallery.

SAM3 has no true "no prompt" mode -- every call needs a text or geometric
prompt. This script treats the text prompt as the thing under test: for each
candidate prompt, run SAM3 (no boxes, no exemplars) across a fixed image set,
and report two things:

1. A visual gallery -- one grid per prompt, prompt text as the big header,
   boxes drawn with LARGE score labels. No per-image files (grid only).

2. A GT-conditioned calibration table -- for the 4 wells with reference boxes
   (hatched_F01_t0040, egg_stage_E03_t0000, older_narrow_C12_t0172 have real
   ground truth; zfpm_F05_t0009_multi has legacy grounding-DINO boxes), match
   each GT box to its best-IoU SAM3 detection and report that detection's
   confidence score. This answers: "conditional on an embryo really being
   there, what score does SAM3 give it?" -- which is what the 0.1 threshold
   choice should be calibrated against, independent of how many extra
   (unmatched / no-GT-available) detections a prompt also produces.

Run with: .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_prompt_benchmark.py
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VENDOR_PATH = os.path.expanduser("~/.cache/morphseq/vendor/sam3")
if VENDOR_PATH not in sys.path:
    sys.path.insert(0, VENDOR_PATH)
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/morphseq/models/sam3/hf"))

import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Low so we can see the full score distribution and calibrate the real
# operating threshold (0.1) against it, rather than have it pre-filtered.
CONFIDENCE_THRESHOLD = 0.05
DISPLAY_THRESHOLD = 0.1  # boxes below this are dropped entirely from the gallery

PROMPTS = {
    "embryo": "embryo",
    "zebrafish_embryo": "zebrafish embryo",
    "individual_zebrafish_embryo": "individual zebrafish embryo",
    "zebrafish_embryo_individual": "zebrafish embryo individual",
    "zebrafish_embryo_individual_embryo": "zebrafish embryo individual embryo",
    "zebrafish_embryo_with_yolk": "zebrafish embryo with yolk",
    "just_yolk": "yolk",
}

RAW_ROOT = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized"
)
SEQ_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260408_segmenting_sequence_images"
)

IMAGES = {
    "hatched_F01_t0040": RAW_ROOT / "20260202/images/20260202_F01/20260202_F01_ch00_t0040.jpg",
    "egg_stage_E03_t0000": RAW_ROOT / "20260202/images/20260202_E03/20260202_E03_ch00_t0000.jpg",
    "older_narrow_C12_t0172": RAW_ROOT / "20251121/images/20251121_C12/20251121_C12_ch00_t0172.jpg",
    "zfpm_F05_t0009_multi": RAW_ROOT
    / "20260502_zfpm_pilot/images/20260502_zfpm_pilot_F05/20260502_zfpm_pilot_F05_ch00_t0009.jpg",
    "20260202_A01_t0000": RAW_ROOT / "20260202/images/20260202_A01/20260202_A01_ch00_t0000.jpg",
    "20260202_B06_t0060": RAW_ROOT / "20260202/images/20260202_B06/20260202_B06_ch00_t0060.jpg",
    "20251121_D08_t0100": RAW_ROOT / "20251121/images/20251121_D08/20251121_D08_ch00_t0100.jpg",
    "20251121_G03_t0050": RAW_ROOT / "20251121/images/20251121_G03/20251121_G03_ch00_t0050.jpg",
    "zfpm_C02_t0020": RAW_ROOT
    / "20260502_zfpm_pilot/images/20260502_zfpm_pilot_C02/20260502_zfpm_pilot_C02_ch00_t0020.jpg",
    "zfpm_H10_t0040": RAW_ROOT
    / "20260502_zfpm_pilot/images/20260502_zfpm_pilot_H10/20260502_zfpm_pilot_H10_ch00_t0040.jpg",
    "mounted_image6_OOD": SEQ_DIR / "test_sequence_image" / "image (6).png",
    "mounted_image7_OOD": SEQ_DIR / "test_sequence_image" / "image (7).png",
}

# GT / reference boxes, xyxy pixel coords. Used ONLY for calibration scoring,
# never shown to the model.
GT_BOXES = {
    "hatched_F01_t0040": [
        (409.0, 699.0, 824.0, 1219.0),
        (1385.0, 961.0, 1974.0, 1277.0),
        (577.0, 1274.0, 901.0, 1919.0),
        (1002.0, 999.0, 1434.0, 1445.0),
    ],
    "egg_stage_E03_t0000": [
        (937.0, 1001.0, 1150.0, 1251.0),
        (474.0, 1445.0, 710.0, 1667.0),
        (1420.0, 1489.0, 1664.0, 1705.0),
        (943.0, 1252.0, 1188.0, 1466.0),
        (1216.0, 972.0, 1437.0, 1207.0),
    ],
    "older_narrow_C12_t0172": [
        (1088.0, 486.0, 1307.0, 1672.0),
    ],
    "zfpm_F05_t0009_multi": [  # legacy grounding-DINO boxes, not full GT (only 3/5 embryos)
        (575.0, 1372.0, 1300.0, 1864.0),
        (364.0, 446.0, 615.0, 1234.0),
        (1357.0, 970.0, 1556.0, 1807.0),
    ],
}

OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_prompt_benchmark"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0),
    (200, 0, 200), (0, 200, 200), (255, 255, 0), (150, 75, 0),
]


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(image, dets, display_threshold):
    """dets: list of (box_xyxy, score). Boxes below display_threshold are dropped entirely."""
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = load_font(46)
    shown = [(box, score) for box, score in dets if score >= display_threshold]
    for i, (box, score) in enumerate(shown):
        x0, y0, x1, y1 = box
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        label = f"{score:.2f}"
        tx, ty = x0, max(0, y0 - 52)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb, fill="white")
        draw.text((tx, ty), label, fill=color, font=font)
    return im


def make_grid(panels, titles, prompt_text, out_path, ncols=4, thumb=(560, 560)):
    header_h, title_h, pad = 70, 34, 8
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    cell_w, cell_h = thumb[0] + pad * 2, thumb[1] + pad * 2 + title_h
    sheet = Image.new("RGB", (cell_w * ncols, header_h + cell_h * nrows), "white")
    draw = ImageDraw.Draw(sheet)
    header_font = load_font(40)
    title_font = load_font(22)
    draw.text((pad, 12), f'SAM3 text prompt: "{prompt_text}"  (display threshold {DISPLAY_THRESHOLD})', fill="black", font=header_font)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        r, c = divmod(i, ncols)
        x0, y0 = c * cell_w, header_h + r * cell_h
        p = panel.copy()
        p.thumbnail(thumb)
        draw.text((x0 + pad, y0 + 4), title, fill="black", font=title_font)
        sheet.paste(p, (x0 + pad, y0 + title_h))
    sheet.save(out_path)
    print(f"wrote {out_path}")


def run_prompt(processor, prompt_name, prompt_text, calibration_rows, raw_detections):
    panels, titles = [], []
    for name, path in IMAGES.items():
        if not path.exists():
            print(f"  SKIP {name}: missing {path}")
            continue
        image = Image.open(path).convert("RGB")
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=prompt_text)

        n_obj = len(state["scores"])
        det_boxes = [tuple(state["boxes"][i].cpu().tolist()) for i in range(n_obj)]
        det_scores = [state["scores"][i].item() for i in range(n_obj)]
        dets = list(zip(det_boxes, det_scores))

        raw_detections.append({
            "prompt": prompt_name,
            "image": name,
            "scores": det_scores,
        })

        n_shown = sum(1 for _, s in dets if s >= DISPLAY_THRESHOLD)
        det_img = draw_boxes(image, dets, DISPLAY_THRESHOLD)
        panels.append(det_img)
        titles.append(f"{name} (n≥{DISPLAY_THRESHOLD}: {n_shown} / raw: {n_obj})")

        if name in GT_BOXES:
            for gt_i, gt_box in enumerate(GT_BOXES[name]):
                best_iou, best_score = 0.0, None
                for box, score in dets:
                    iou = iou_xyxy(gt_box, box)
                    if iou > best_iou:
                        best_iou, best_score = iou, score
                calibration_rows.append({
                    "prompt": prompt_name,
                    "image": name,
                    "gt_idx": gt_i,
                    "best_iou": best_iou,
                    "matched_score": best_score,
                })

    make_grid(panels, titles, prompt_text, OUT_DIR / f"gallery_{prompt_name}.png")


def print_calibration_table(calibration_rows):
    print("\n" + "=" * 100)
    print("GT-conditioned calibration: for each reference box, best-IoU SAM3 detection's score")
    print("(hatched_F01/egg_stage_E03/older_narrow_C12 = real GT; zfpm_F05 = legacy grounding-DINO boxes)")
    print("=" * 100)
    header = f"{'prompt':8} {'image':26} {'gt#':4} {'best_iou':9} {'matched_score':13} {'matched?(iou>=0.5)':18}"
    print(header)
    print("-" * len(header))
    for row in calibration_rows:
        matched = "YES" if row["best_iou"] >= 0.5 else "NO"
        score_str = f"{row['matched_score']:.3f}" if row["matched_score"] is not None else "n/a"
        print(f"{row['prompt']:8} {row['image']:26} {row['gt_idx']:<4} {row['best_iou']:.3f}     {score_str:13} {matched:18}")

    print("\nSummary by prompt (matched GT boxes only, iou>=0.5):")
    for prompt_name in PROMPTS:
        matched_scores = [
            r["matched_score"] for r in calibration_rows
            if r["prompt"] == prompt_name and r["best_iou"] >= 0.5 and r["matched_score"] is not None
        ]
        total_gt = sum(1 for r in calibration_rows if r["prompt"] == prompt_name)
        n_matched = len(matched_scores)
        if matched_scores:
            mean_score = sum(matched_scores) / len(matched_scores)
            min_score = min(matched_scores)
            print(f"  {prompt_name:8}: recall={n_matched}/{total_gt}  mean_matched_score={mean_score:.3f}  min_matched_score={min_score:.3f}")
        else:
            print(f"  {prompt_name:8}: recall=0/{total_gt}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"Loading SAM3 image model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    calibration_rows = []
    raw_detections = []
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n### prompt='{prompt_name}': \"{prompt_text}\" ###")
        run_prompt(processor, prompt_name, prompt_text, calibration_rows, raw_detections)

    print_calibration_table(calibration_rows)

    results_path = OUT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "prompts": PROMPTS,
            "display_threshold": DISPLAY_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "calibration_rows": calibration_rows,
            "raw_detections": raw_detections,
        }, f, indent=2)
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
