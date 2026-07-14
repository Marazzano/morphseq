"""Manifest-driven SAM3 vs legacy comparison. Pick a "prompt" case to display at
the top (clearly labeled PROMPT), then run SAM3's open-vocabulary text prompt
"zebrafish embryo" independently on every case in test_images/manifest.json
(SAM3 has no cross-image box-exemplar-transfer API -- text prompting is the only
mechanism that does open-set exhaustive detection across different images), and
show legacy-pipeline-detected (left) vs SAM3-detected (right) for each.

Run with:
  .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_prompt_sweep.py --prompt-case f01_t0040
  .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_prompt_sweep.py --prompt-case seq_image_7
"""
import argparse
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

TEST_IMAGES_ROOT = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/test_images"
)
OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_prompt_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0),
    (200, 0, 200), (0, 200, 200), (150, 75, 0), (255, 105, 180),
]

TEXT_PROMPT = "zebrafish embryo"


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes_xyxy, labels, colors, label_font_size=28):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = load_font(label_font_size)
    for (x0, y0, x1, y1), label, color in zip(boxes_xyxy, labels, colors):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        text_bbox = draw.textbbox((x0, max(0, y0 - label_font_size - 6)), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))
        draw.text((x0, max(0, y0 - label_font_size - 6)), label, fill=color, font=font)
    return im


def load_case(case):
    img_path = TEST_IMAGES_ROOT / case["source_image"]
    image = Image.open(img_path).convert("RGB")
    legacy_path = TEST_IMAGES_ROOT / case["legacy_boxes"]
    with open(legacy_path) as f:
        legacy_data = json.load(f)
    legacy_boxes = [b["box"] for b in legacy_data["boxes"]]
    legacy_labels = [b["label"] for b in legacy_data["boxes"]]
    return image, legacy_boxes, legacy_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-case", required=True, help="case name from manifest.json to show as the PROMPT row")
    args = parser.parse_args()

    with open(TEST_IMAGES_ROOT / "manifest.json") as f:
        manifest = json.load(f)
    cases_by_name = {c["name"]: c for c in manifest["cases"]}
    if args.prompt_case not in cases_by_name:
        raise SystemExit(f"unknown case {args.prompt_case!r}; choices: {list(cases_by_name)}")

    prompt_case = cases_by_name[args.prompt_case]
    other_cases = [c for c in manifest["cases"] if c["name"] != args.prompt_case]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"Loading SAM3 image model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    # PROMPT row: show the legacy boxes for the chosen prompt case as reference
    prompt_image, prompt_legacy_boxes, prompt_legacy_labels = load_case(prompt_case)
    prompt_colors = [COLORS[i % len(COLORS)] for i in range(len(prompt_legacy_boxes))]
    prompt_img_annotated = draw_boxes(prompt_image, prompt_legacy_boxes, prompt_legacy_labels, prompt_colors)

    rows = []
    for case in other_cases:
        name = case["name"]
        print(f"\n=== {name} ===")
        image, legacy_boxes, legacy_labels = load_case(case)

        legacy_scores_labels = [f"L:{lbl}" for lbl in legacy_labels]
        legacy_colors = [COLORS[i % len(COLORS)] for i in range(len(legacy_boxes))]
        legacy_img = draw_boxes(image, legacy_boxes, legacy_scores_labels, legacy_colors)

        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)
        n_obj = len(state["scores"])
        det_boxes = [state["boxes"][i].cpu().tolist() for i in range(n_obj)]
        det_scores = [state["scores"][i].item() for i in range(n_obj)]
        print(f"  legacy: {len(legacy_boxes)} boxes | SAM3 text-prompt '{TEXT_PROMPT}': {n_obj} boxes")
        for i, (b, s) in enumerate(zip(det_boxes, det_scores)):
            print(f"    det{i+1}: box={b} score={s:.3f}")

        det_labels = [f"S{i+1} {s:.2f}" for i, s in enumerate(det_scores)]
        det_colors = [COLORS[i % len(COLORS)] for i in range(n_obj)]
        sam3_img = draw_boxes(image, det_boxes, det_labels, det_colors)

        rows.append(
            (
                name,
                legacy_img,
                sam3_img,
                f"{name}: legacy ({len(legacy_boxes)})",
                f"{name}: SAM3 text-prompt ({n_obj})",
            )
        )

    # assemble combined panel
    thumb_size = (650, 650)
    prompt_thumb = prompt_img_annotated.copy()
    prompt_thumb.thumbnail((850, 850))

    row_thumbs = []
    for name, legacy_img, sam3_img, legacy_title, sam3_title in rows:
        lt = legacy_img.copy()
        lt.thumbnail(thumb_size)
        st = sam3_img.copy()
        st.thumbnail(thumb_size)
        row_thumbs.append((lt, st, legacy_title, sam3_title))

    pad, title_h, gutter, row_gap = 15, 40, 25, 18
    panel_w = max(max(lt.width, st.width) for lt, st, _, _ in row_thumbs)
    row_h_list = [max(lt.height, st.height) for lt, st, _, _ in row_thumbs]

    sheet_w = pad * 2 + gutter + panel_w * 2
    sheet_h = (
        pad * 3
        + title_h * 2
        + prompt_thumb.height
        + sum(title_h + rh + row_gap for rh in row_h_list)
    )
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(30)
    prompt_font = load_font(36)

    y = pad
    draw.text((pad, y), f"PROMPT: {args.prompt_case} ({len(prompt_legacy_boxes)} boxes)  [text concept: \"{TEXT_PROMPT}\"]", fill=(180, 0, 0), font=prompt_font)
    y += title_h + 10
    prompt_x = pad + (sheet_w - pad * 2 - prompt_thumb.width) // 2
    sheet.paste(prompt_thumb, (prompt_x, y))
    y += prompt_thumb.height + pad
    draw.line([(pad, y), (sheet_w - pad, y)], fill=(0, 0, 0), width=3)
    y += pad

    for (lt, st, legacy_title, sam3_title), row_h in zip(row_thumbs, row_h_list):
        x0, x1 = pad, pad + panel_w + gutter
        draw.text((x0, y), legacy_title, fill="black", font=title_font)
        draw.text((x1, y), sam3_title, fill="black", font=title_font)
        y += title_h
        sheet.paste(lt, (x0, y))
        sheet.paste(st, (x1, y))
        y += row_h + row_gap

    out_path = OUT_DIR / f"prompt_{args.prompt_case}_sweep.png"
    sheet.save(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
