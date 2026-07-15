"""Out-of-distribution generalization test: does SAM3 work "out of the box" on a
completely different imaging setup -- individually mounted embryos on a plain
lavender background, numbered 1-8 (NOT the petri-dish overhead shots the other
exemplars came from)?

Layout: one combined panel image --
  top:     prompt/reference image (20260202_F01_ch00_t0040, 4 GT boxes) for context
  row N:   target image N  |  legacy grounding-DINO pipeline detected (left)  |  SAM3 text-prompt detected (right)

SAM3 has no cross-image box-exemplar-transfer API (see sam3_image_processor.py --
only set_text_prompt and add_geometric_prompt-on-current-image exist), so "the
prompt performs on the right" means SAM3's open-vocabulary text prompt
"zebrafish embryo" run independently on each target image -- the actual
mechanism SAM3 exposes for exhaustive open-set detection. The F01 image is shown
at top purely as the reference exemplar this whole test thread started from.

Run with: .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_sequence_grid_test.py
"""
import csv
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
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor


def normalize_bbox_cxcywh(bbox_cxcywh: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    normalized = bbox_cxcywh.clone()
    normalized[..., 0] /= img_w
    normalized[..., 1] /= img_h
    normalized[..., 2] /= img_w
    normalized[..., 3] /= img_h
    return normalized


PROMPT_IMAGE_PATH = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
    "raw_data_organized/20260202/images/20260202_F01/20260202_F01_ch00_t0040.jpg"
)
PROMPT_BOXES_XYXY = [
    (409.0, 699.0, 824.0, 1219.0),
    (1385.0, 961.0, 1974.0, 1277.0),
    (577.0, 1274.0, 901.0, 1919.0),
    (1002.0, 999.0, 1434.0, 1445.0),
]

SEQ_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260408_segmenting_sequence_images"
)
DETECTIONS_CSV = SEQ_DIR / "output" / "detections.csv"
TARGET_IMAGES = ["image (6).png", "image (7).png"]

OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_sequence_grid_test"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0),
    (200, 0, 200), (0, 200, 200), (150, 75, 0), (255, 105, 180),
]

FONT_LABEL_SIZE = 30
FONT_TITLE_SIZE = 32


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def load_legacy_boxes(image_name: str, img_w: int, img_h: int):
    boxes = []
    scores = []
    with open(DETECTIONS_CSV) as f:
        for row in csv.DictReader(f):
            if row["image"] != image_name:
                continue
            x1 = float(row["box_x1"]) * img_w
            y1 = float(row["box_y1"]) * img_h
            x2 = float(row["box_x2"]) * img_w
            y2 = float(row["box_y2"]) * img_h
            boxes.append((x1, y1, x2, y2))
            scores.append(float(row["confidence"]))
    return boxes, scores


def draw_boxes(image: Image.Image, boxes_xyxy, labels, colors, label_font_size=FONT_LABEL_SIZE):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = load_font(label_font_size)
    for (x0, y0, x1, y1), label, color in zip(boxes_xyxy, labels, colors):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        text_bbox = draw.textbbox((x0, max(0, y0 - label_font_size - 6)), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))
        draw.text((x0, max(0, y0 - label_font_size - 6)), label, fill=color, font=font)
    return im


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"Loading SAM3 image model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    # prompt/reference panel (context only -- not literally reused on other images)
    prompt_image = Image.open(PROMPT_IMAGE_PATH).convert("RGB")
    prompt_labels = [f"exemplar{i+1}" for i in range(len(PROMPT_BOXES_XYXY))]
    prompt_img_annotated = draw_boxes(
        prompt_image, PROMPT_BOXES_XYXY, prompt_labels, COLORS[: len(PROMPT_BOXES_XYXY)]
    )

    rows = []  # list of (target_name, legacy_img, sam3_img, legacy_title, sam3_title)

    for image_name in TARGET_IMAGES:
        print(f"\n=== {image_name} ===")
        img_path = SEQ_DIR / "test_sequence_image" / image_name
        target_image = Image.open(img_path).convert("RGB")
        tw, th = target_image.size

        legacy_boxes, legacy_scores = load_legacy_boxes(image_name, tw, th)
        legacy_labels = [f"L{i+1} {s:.2f}" for i, s in enumerate(legacy_scores)]
        legacy_colors = [COLORS[i % len(COLORS)] for i in range(len(legacy_boxes))]
        legacy_img = draw_boxes(target_image, legacy_boxes, legacy_labels, legacy_colors)
        print(f"  legacy grounding-DINO: {len(legacy_boxes)} boxes")

        state = processor.set_image(target_image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt="zebrafish embryo")
        n_obj = len(state["scores"])
        det_boxes = [state["boxes"][i].cpu().tolist() for i in range(n_obj)]
        det_scores = [state["scores"][i].item() for i in range(n_obj)]
        print(f"  SAM3 text-prompt 'zebrafish embryo': {n_obj} boxes")
        for i, (b, s) in enumerate(zip(det_boxes, det_scores)):
            print(f"    det{i+1}: box={b} score={s:.3f}")

        det_labels = [f"S{i+1} {s:.2f}" for i, s in enumerate(det_scores)]
        det_colors = [COLORS[i % len(COLORS)] for i in range(n_obj)]
        sam3_img = draw_boxes(target_image, det_boxes, det_labels, det_colors)

        rows.append(
            (
                image_name,
                legacy_img,
                sam3_img,
                f"{image_name}: legacy grounding-DINO ({len(legacy_boxes)})",
                f"{image_name}: SAM3 text-prompt ({n_obj})",
            )
        )

    # assemble combined panel
    thumb_size = (750, 750)
    prompt_thumb = prompt_img_annotated.copy()
    prompt_thumb.thumbnail((950, 950))

    row_thumbs = []
    for name, legacy_img, sam3_img, legacy_title, sam3_title in rows:
        lt = legacy_img.copy()
        lt.thumbnail(thumb_size)
        st = sam3_img.copy()
        st.thumbnail(thumb_size)
        row_thumbs.append((lt, st, legacy_title, sam3_title))

    pad, title_h, gutter, row_gap = 15, 44, 25, 20
    panel_w = max(max(lt.width, st.width) for lt, st, _, _ in row_thumbs)
    row_h = max(max(lt.height, st.height) for lt, st, _, _ in row_thumbs)

    sheet_w = pad * 2 + gutter + panel_w * 2
    sheet_h = (
        pad * 2
        + title_h
        + prompt_thumb.height
        + row_gap
        + len(row_thumbs) * (title_h + row_h + row_gap)
    )
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(FONT_TITLE_SIZE)

    y = pad
    draw.text((pad, y), f"Prompt / reference: F01_t0040 ({len(PROMPT_BOXES_XYXY)} exemplar boxes)", fill="black", font=title_font)
    y += title_h
    prompt_x = pad + (sheet_w - pad * 2 - prompt_thumb.width) // 2
    sheet.paste(prompt_thumb, (prompt_x, y))
    y += prompt_thumb.height + row_gap

    for (lt, st, legacy_title, sam3_title) in row_thumbs:
        x0, x1 = pad, pad + panel_w + gutter
        draw.text((x0, y), legacy_title, fill="black", font=title_font)
        draw.text((x1, y), sam3_title, fill="black", font=title_font)
        y += title_h
        sheet.paste(lt, (x0, y))
        sheet.paste(st, (x1, y))
        y += row_h + row_gap

    out_path = OUT_DIR / "sequence_grid_legacy_vs_sam3.png"
    sheet.save(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
