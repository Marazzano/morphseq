"""Prompt SAM3 (CPU dev checkpoint) with the 4 ground-truth boxes from
20260202_F01_ch00_t0040 as positive box exemplars, run detection on the SAME
image, and render a ground-truth vs. detected side-by-side.

Run with: pixi run -e sam3 python tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_exemplar_prompt.py
"""
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

IMAGE_PATH = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
    "raw_data_organized/20260202/images/20260202_F01/20260202_F01_ch00_t0040.jpg"
)
OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_run"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ground-truth per-embryo boxes (x_min, y_min, x_max, y_max) in pixels, from
# candidate_exemplars_multi_embryo_wells.csv
GT_BOXES_XYXY = {
    "e01": (409.0, 699.0, 824.0, 1219.0),
    "e02": (1385.0, 961.0, 1974.0, 1277.0),
    "e03": (577.0, 1274.0, 901.0, 1919.0),
    "e04": (1002.0, 999.0, 1434.0, 1445.0),
}

COLORS = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0)]


def draw_boxes(image: Image.Image, boxes_xyxy, labels, colors):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    for (x0, y0, x1, y1), label, color in zip(boxes_xyxy, labels, colors):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0, max(0, y0 - 32)), label, fill=color, font=font)
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

    image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = image.size
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)

    print("Prompting with 4 ground-truth boxes as positive exemplars...")
    for eid, (x0, y0, x1, y1) in GT_BOXES_XYXY.items():
        box_xywh = torch.tensor([x0, y0, x1 - x0, y1 - y0]).view(-1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox_cxcywh(box_cxcywh, width, height).flatten().tolist()
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=norm_box, label=True
        )

    # inference_state IS the results dict after prompting (see plot_results usage
    # in sam3_image_predictor_example.ipynb): keys "scores", "masks", "boxes" (XYXY, absolute px)
    n_objects = len(inference_state["scores"])
    print(f"SAM3 found {n_objects} object(s)")
    detected_boxes = [inference_state["boxes"][i].cpu().tolist() for i in range(n_objects)]
    detected_scores = [inference_state["scores"][i].item() for i in range(n_objects)]
    for i, (box, score) in enumerate(zip(detected_boxes, detected_scores)):
        print(f"  det{i+1}: box={box} score={score:.3f}")

    gt_labels = list(GT_BOXES_XYXY.keys())
    gt_boxes = list(GT_BOXES_XYXY.values())
    gt_img = draw_boxes(image, gt_boxes, gt_labels, COLORS)

    det_labels = [f"det{i+1} ({s:.2f})" for i, s in enumerate(detected_scores)]
    det_colors = [COLORS[i % len(COLORS)] for i in range(len(detected_boxes))]
    det_img = draw_boxes(image, detected_boxes, det_labels, det_colors)

    gt_img.save(OUT_DIR / "ground_truth_boxes.png")
    det_img.save(OUT_DIR / "sam3_detected_boxes.png")

    # side-by-side via PIL: thumbnail both panels, stack with title bar + gutter
    pad, title_h, gutter = 10, 40, 20
    thumb_size = (900, 900)
    gt_thumb = gt_img.copy()
    gt_thumb.thumbnail(thumb_size)
    det_thumb = det_img.copy()
    det_thumb.thumbnail(thumb_size)

    panel_w = max(gt_thumb.width, det_thumb.width)
    panel_h = max(gt_thumb.height, det_thumb.height)
    sheet_w = pad * 2 + gutter + panel_w * 2
    sheet_h = pad * 2 + title_h + panel_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    x0 = pad
    x1 = pad + panel_w + gutter
    y_img = pad + title_h
    draw.text((x0, pad), f"Ground truth ({len(gt_boxes)} exemplar boxes)", fill="black", font=font)
    draw.text((x1, pad), f"SAM3 detected ({len(detected_boxes)} boxes)", fill="black", font=font)
    sheet.paste(gt_thumb, (x0, y_img))
    sheet.paste(det_thumb, (x1, y_img))

    side_by_side_path = OUT_DIR / "gt_vs_sam3_side_by_side.png"
    sheet.save(side_by_side_path)
    print(f"wrote {side_by_side_path}")


if __name__ == "__main__":
    main()
