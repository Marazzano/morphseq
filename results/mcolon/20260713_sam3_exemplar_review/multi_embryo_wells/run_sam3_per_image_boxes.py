"""Run SAM3 box-prompted detection independently on three images spanning
different embryo morphology: hatched larvae (F01_t0040), round egg-stage
embryos (E03_t0000), and an older/narrower elongated larva (C12_t0172).

Each image is prompted with ITS OWN ground-truth boxes as positive geometric
exemplars (SAM3's Sam3Processor has no cross-image exemplar-transfer API --
only per-image text or geometric prompts -- so this tests detection quality
per morphology rather than transfer from a single source image).

Run with: .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_per_image_boxes.py
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


CASES = {
    "hatched_F01_t0040": {
        "image_path": Path(
            "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
            "raw_data_organized/20260202/images/20260202_F01/20260202_F01_ch00_t0040.jpg"
        ),
        "gt_boxes": [
            (409.0, 699.0, 824.0, 1219.0),
            (1385.0, 961.0, 1974.0, 1277.0),
            (577.0, 1274.0, 901.0, 1919.0),
            (1002.0, 999.0, 1434.0, 1445.0),
        ],
    },
    "egg_stage_E03_t0000": {
        "image_path": Path(
            "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
            "raw_data_organized/20260202/images/20260202_E03/20260202_E03_ch00_t0000.jpg"
        ),
        "gt_boxes": [
            (937.0, 1001.0, 1150.0, 1251.0),
            (474.0, 1445.0, 710.0, 1667.0),
            (1420.0, 1489.0, 1664.0, 1705.0),
            (943.0, 1252.0, 1188.0, 1466.0),
            (1216.0, 972.0, 1437.0, 1207.0),
        ],
    },
    "older_narrow_C12_t0172": {
        "image_path": Path(
            "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/morphseq_playground/sam2_pipeline_files/"
            "raw_data_organized/20251121/images/20251121_C12/20251121_C12_ch00_t0172.jpg"
        ),
        "gt_boxes": [
            (1088.0, 486.0, 1307.0, 1672.0),
        ],
    },
}

OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_per_image_run"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0), (200, 0, 200), (0, 200, 200)]


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


def side_by_side(left_img, right_img, left_title, right_title, out_path):
    pad, title_h, gutter = 10, 40, 20
    thumb_size = (900, 900)
    lt = left_img.copy()
    lt.thumbnail(thumb_size)
    rt = right_img.copy()
    rt.thumbnail(thumb_size)

    panel_w = max(lt.width, rt.width)
    panel_h = max(lt.height, rt.height)
    sheet = Image.new("RGB", (pad * 2 + gutter + panel_w * 2, pad * 2 + title_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    x0, x1 = pad, pad + panel_w + gutter
    y_img = pad + title_h
    draw.text((x0, pad), left_title, fill="black", font=font)
    draw.text((x1, pad), right_title, fill="black", font=font)
    sheet.paste(lt, (x0, y_img))
    sheet.paste(rt, (x1, y_img))
    sheet.save(out_path)
    print(f"wrote {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"Loading SAM3 image model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    for name, case in CASES.items():
        print(f"\n=== {name} ===")
        image = Image.open(case["image_path"]).convert("RGB")
        width, height = image.size
        state = processor.set_image(image)
        processor.reset_all_prompts(state)

        for (x0, y0, x1, y1) in case["gt_boxes"]:
            box_xywh = torch.tensor([x0, y0, x1 - x0, y1 - y0]).view(-1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox_cxcywh(box_cxcywh, width, height).flatten().tolist()
            state = processor.add_geometric_prompt(state=state, box=norm_box, label=True)

        n_obj = len(state["scores"])
        det_boxes = [state["boxes"][i].cpu().tolist() for i in range(n_obj)]
        det_scores = [state["scores"][i].item() for i in range(n_obj)]
        print(f"  GT boxes: {len(case['gt_boxes'])}  ->  SAM3 detected: {n_obj}")
        for i, (b, s) in enumerate(zip(det_boxes, det_scores)):
            print(f"    det{i+1}: box={b} score={s:.3f}")

        gt_labels = [f"gt{i+1}" for i in range(len(case["gt_boxes"]))]
        gt_colors = [COLORS[i % len(COLORS)] for i in range(len(case["gt_boxes"]))]
        gt_img = draw_boxes(image, case["gt_boxes"], gt_labels, gt_colors)

        det_labels = [f"det{i+1} ({s:.2f})" for i, s in enumerate(det_scores)]
        det_colors = [COLORS[i % len(COLORS)] for i in range(n_obj)]
        det_img = draw_boxes(image, det_boxes, det_labels, det_colors)

        gt_img.save(OUT_DIR / f"{name}_ground_truth.png")
        det_img.save(OUT_DIR / f"{name}_sam3_detected.png")
        side_by_side(
            gt_img,
            det_img,
            f"{name}: ground truth ({len(case['gt_boxes'])} boxes)",
            f"{name}: SAM3 detected ({n_obj} boxes)",
            OUT_DIR / f"{name}_gt_vs_sam3.png",
        )


if __name__ == "__main__":
    main()
