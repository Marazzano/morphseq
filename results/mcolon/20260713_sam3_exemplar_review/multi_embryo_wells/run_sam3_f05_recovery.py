"""F05_t0009 recovery test: the legacy grounded-SAM pipeline only tracked 3 of the
5 visible larvae in this frame. Test whether SAM3, prompted with the SAME image
via open-vocabulary text prompting ("zebrafish embryo"), recovers all 5 -- i.e.
exhaustive detection rather than just refining given boxes.

Layout (3-panel, stacked):
  top:           prompt image (the 3 legacy per-embryo boxes drawn as reference)
  bottom-left:   legacy pipeline detections (3 boxes, from grounded_sam_segmentations json)
  bottom-right:  SAM3 detected (text-prompted "zebrafish embryo", exhaustive open-vocab)

Run with: .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_f05_recovery.py
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
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
    "raw_data_organized/20260502_zfpm_pilot/images/20260502_zfpm_pilot_F05/20260502_zfpm_pilot_F05_ch00_t0009.jpg"
)

# legacy grounded_sam pipeline boxes (only 3 of 5 visible larvae tracked)
LEGACY_BOXES_XYXY = {
    "e01": (575.0, 1372.0, 1300.0, 1864.0),
    "e02": (364.0, 446.0, 615.0, 1234.0),
    "e03": (1357.0, 970.0, 1556.0, 1807.0),
}

OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_f05_recovery"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0), (200, 0, 200), (0, 200, 200)]

FONT_LABEL_SIZE = 34
FONT_TITLE_SIZE = 30


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes_xyxy, labels, colors, label_font_size=FONT_LABEL_SIZE):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = load_font(label_font_size)
    for (x0, y0, x1, y1), label, color in zip(boxes_xyxy, labels, colors):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # background box behind text for legibility at larger font size
        text_bbox = draw.textbbox((x0, max(0, y0 - label_font_size - 8)), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))
        draw.text((x0, max(0, y0 - label_font_size - 8)), label, fill=color, font=font)
    return im


def build_panel_layout(prompt_img, legacy_img, sam3_img, prompt_title, legacy_title, sam3_title, out_path):
    thumb_size = (900, 900)
    pt = prompt_img.copy()
    pt.thumbnail(thumb_size)
    lt = legacy_img.copy()
    lt.thumbnail(thumb_size)
    st = sam3_img.copy()
    st.thumbnail(thumb_size)

    pad, title_h, gutter = 15, 50, 25
    row_w = lt.width + gutter + st.width
    top_w = max(pt.width, row_w)

    title_font = load_font(FONT_TITLE_SIZE)

    sheet_w = pad * 2 + top_w
    sheet_h = pad * 3 + title_h * 2 + pt.height + max(lt.height, st.height)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    # top: prompt image, centered
    top_x = pad + (top_w - pt.width) // 2
    draw.text((pad, pad), prompt_title, fill="black", font=title_font)
    y_top_img = pad + title_h
    sheet.paste(pt, (top_x, y_top_img))

    # bottom row: legacy (left), sam3 (right)
    y_bottom_title = y_top_img + pt.height + pad
    y_bottom_img = y_bottom_title + title_h
    bl_x = pad
    br_x = pad + lt.width + gutter
    draw.text((bl_x, y_bottom_title), legacy_title, fill="black", font=title_font)
    draw.text((br_x, y_bottom_title), sam3_title, fill="black", font=title_font)
    sheet.paste(lt, (bl_x, y_bottom_img))
    sheet.paste(st, (br_x, y_bottom_img))

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

    image = Image.open(IMAGE_PATH).convert("RGB")

    # prompt panel: legacy boxes drawn as the reference prompt
    legacy_labels = list(LEGACY_BOXES_XYXY.keys())
    legacy_boxes = list(LEGACY_BOXES_XYXY.values())
    prompt_img = draw_boxes(image, legacy_boxes, legacy_labels, COLORS[: len(legacy_boxes)])

    # legacy panel: same as prompt (this IS what the legacy pipeline found)
    legacy_img = prompt_img

    # SAM3 panel: open-vocabulary text-prompt exhaustive detection on the same image
    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt="zebrafish embryo")

    n_obj = len(state["scores"])
    det_boxes = [state["boxes"][i].cpu().tolist() for i in range(n_obj)]
    det_scores = [state["scores"][i].item() for i in range(n_obj)]
    print(f"Legacy pipeline: {len(legacy_boxes)} boxes")
    print(f"SAM3 text-prompt 'zebrafish embryo': {n_obj} boxes")
    for i, (b, s) in enumerate(zip(det_boxes, det_scores)):
        print(f"  det{i+1}: box={b} score={s:.3f}")

    det_labels = [f"det{i+1} {s:.2f}" for i, s in enumerate(det_scores)]
    det_colors = [COLORS[i % len(COLORS)] for i in range(n_obj)]
    sam3_img = draw_boxes(image, det_boxes, det_labels, det_colors)

    prompt_img.save(OUT_DIR / "prompt_legacy_boxes.png")
    sam3_img.save(OUT_DIR / "sam3_text_prompt_detected.png")

    build_panel_layout(
        prompt_img,
        legacy_img,
        sam3_img,
        f"Prompt: legacy pipeline boxes ({len(legacy_boxes)} of 5 visible larvae)",
        f"Legacy pipeline detected ({len(legacy_boxes)})",
        f"SAM3 detected ({n_obj})",
        OUT_DIR / "f05_recovery_panel.png",
    )


if __name__ == "__main__":
    main()
