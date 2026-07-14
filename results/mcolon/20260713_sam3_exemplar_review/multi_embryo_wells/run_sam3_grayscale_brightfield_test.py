"""Mini test: does converting the OOD mounted-embryo images (lavender bg,
individually mounted) to grayscale/brightfield-style improve SAM3 zero-shot
text-prompt detection?

mounted_image6_OOD was a near-total miss for every winning prompt except
"individual zebrafish embryo" (fired at 0.30+, everything else <0.1 or 0).
Hypothesis: the color/background cue is throwing off the text grounding
since SAM3's training distribution and our best-performing prompts were
tuned against grayscale brightfield petri-dish images. Convert both mounted
images (6 and 7) to grayscale (L mode, replicated to RGB so the model still
gets a 3-channel input) and re-run the two best prompts plus the OOD-rescuer
prompt, compare raw scores color vs. grayscale.

Run with: .pixi/envs/sam3/bin/python3 tmp/sam3_exemplar_review/multi_embryo_wells/run_sam3_grayscale_brightfield_test.py
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

CONFIDENCE_THRESHOLD = 0.05
DISPLAY_THRESHOLD = 0.1

PROMPTS = {
    "zebrafish_embryo_individual": "zebrafish embryo individual",
    "individual_zebrafish_embryo": "individual zebrafish embryo",
    "zebrafish_embryo": "zebrafish embryo",
}

SEQ_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260408_segmenting_sequence_images"
)
IMAGES = {
    "mounted_image6_OOD": SEQ_DIR / "test_sequence_image" / "image (6).png",
    "mounted_image7_OOD": SEQ_DIR / "test_sequence_image" / "image (7).png",
}

OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/sam3_grayscale_test"
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


def draw_boxes(image, dets, display_threshold):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = load_font(30)
    shown = [(box, score) for box, score in dets if score >= display_threshold]
    for i, (box, score) in enumerate(shown):
        x0, y0, x1, y1 = box
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = f"{score:.2f}"
        tx, ty = x0, max(0, y0 - 34)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb, fill="white")
        draw.text((tx, ty), label, fill=color, font=font)
    return im


def make_grid(panels, titles, header, out_path, ncols=3, thumb=(500, 500)):
    header_h, title_h, pad = 50, 30, 8
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    cell_w, cell_h = thumb[0] + pad * 2, thumb[1] + pad * 2 + title_h
    sheet = Image.new("RGB", (cell_w * ncols, header_h + cell_h * nrows), "white")
    draw = ImageDraw.Draw(sheet)
    header_font = load_font(28)
    title_font = load_font(18)
    draw.text((pad, 10), header, fill="black", font=header_font)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        r, c = divmod(i, ncols)
        x0, y0 = c * cell_w, header_h + r * cell_h
        p = panel.copy()
        p.thumbnail(thumb)
        draw.text((x0 + pad, y0 + 4), title, fill="black", font=title_font)
        sheet.paste(p, (x0 + pad, y0 + title_h))
    sheet.save(out_path)
    print(f"wrote {out_path}")


def run_variant(processor, variant_name, image_variants):
    """image_variants: dict name -> PIL Image (already prepped, RGB)."""
    panels, titles = [], []
    results = {}
    for name, image in image_variants.items():
        for prompt_name, prompt_text in PROMPTS.items():
            state = processor.set_image(image)
            state = processor.set_text_prompt(state=state, prompt=prompt_text)
            n_obj = len(state["scores"])
            det_boxes = [tuple(state["boxes"][i].cpu().tolist()) for i in range(n_obj)]
            det_scores = [state["scores"][i].item() for i in range(n_obj)]
            dets = list(zip(det_boxes, det_scores))
            n_shown = sum(1 for _, s in dets if s >= DISPLAY_THRESHOLD)
            results[(name, prompt_name)] = sorted(det_scores, reverse=True)

            det_img = draw_boxes(image, dets, DISPLAY_THRESHOLD)
            panels.append(det_img)
            titles.append(f"{name} | {prompt_text} (n≥0.1: {n_shown}/{n_obj})")

    make_grid(panels, titles, f"SAM3 on {variant_name} images", OUT_DIR / f"gallery_{variant_name}.png", ncols=3)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"Loading SAM3 image model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    color_images = {name: Image.open(path).convert("RGB") for name, path in IMAGES.items()}
    gray_images = {name: Image.open(path).convert("L").convert("RGB") for name, path in IMAGES.items()}

    print("\n### COLOR (original) ###")
    color_results = run_variant(processor, "color", color_images)

    print("\n### GRAYSCALE (brightfield-style) ###")
    gray_results = run_variant(processor, "grayscale", gray_images)

    print("\n" + "=" * 90)
    print("Top-8 raw scores per (image, prompt): color vs grayscale")
    print("=" * 90)
    for name in IMAGES:
        for prompt_name in PROMPTS:
            c = color_results[(name, prompt_name)][:8]
            g = gray_results[(name, prompt_name)][:8]
            print(f"{name:20} {prompt_name:28} color={[f'{s:.2f}' for s in c]}")
            print(f"{'':20} {'':28} gray ={[f'{s:.2f}' for s in g]}")


if __name__ == "__main__":
    main()
