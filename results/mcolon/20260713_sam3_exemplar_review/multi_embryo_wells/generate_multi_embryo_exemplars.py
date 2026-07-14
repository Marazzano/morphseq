import json
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEG_JSON = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
    "segmentation/grounded_sam_segmentations_20260202.json"
)
RAW_IMG_ROOT = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/"
    "raw_data_organized/20260202/images"
)
OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/older_stage_t0070"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# well -> frame indices chosen for clean, non-overlapping per-embryo boxes
WELL_FRAMES = {
    "20260202_E03": [70],
    "20260202_F01": [70],
    "20260202_E02": [70],
}

COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 120, 255),
    (255, 165, 0),
    (200, 0, 200),
    (0, 200, 200),
]


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def main():
    with open(SEG_JSON) as f:
        data = json.load(f)
    videos = data["experiments"]["20260202"]["videos"]

    rows = []
    overlay_paths = []

    for well, frame_indices in WELL_FRAMES.items():
        v = videos[well]
        imgs = v["image_ids"]
        by_frame_idx = {img["frame_index"]: img_id for img_id, img in imgs.items()}

        for fidx in frame_indices:
            img_id = by_frame_idx.get(fidx)
            if img_id is None:
                print(f"WARNING: {well} missing frame_index {fidx}")
                continue
            img_entry = imgs[img_id]
            embryos = img_entry["embryos"]
            raw_boxes = {eid: e["segmentation"]["bbox"] for eid, e in embryos.items()}
            seg_w, seg_h = next(iter(embryos.values()))["segmentation"]["size"]
            # bboxes are normalized [0,1] fractions of (width, height); convert to pixels
            boxes = {
                eid: [b[0] * seg_w, b[1] * seg_h, b[2] * seg_w, b[3] * seg_h]
                for eid, b in raw_boxes.items()
            }

            # sanity check: flag any duplicate/near-duplicate boxes
            vals = list(boxes.items())
            bad = False
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    if iou(vals[i][1], vals[j][1]) > 0.5:
                        bad = True
            if bad:
                print(f"SKIP degenerate frame {img_id} (overlapping boxes)")
                continue

            img_path = RAW_IMG_ROOT / well / f"{img_id}.jpg"
            if not img_path.exists():
                print(f"WARNING: missing raw image {img_path}")
                continue

            im = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
                )
            except Exception:
                font = ImageFont.load_default()

            for i, (eid, bbox) in enumerate(sorted(boxes.items())):
                x0, y0, x1, y1 = bbox
                color = COLORS[i % len(COLORS)]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                draw.text((x0, max(0, y0 - 32)), eid, fill=color, font=font)

                rows.append(
                    {
                        "exemplar_id": f"zf_{img_id}_{eid.split('_')[-1]}",
                        "concept_label": "zebrafish embryo",
                        "prompt_role": "positive",
                        "prompt_type": "box",
                        "reference_image_path": str(img_path),
                        "bbox_x_min_px": x0,
                        "bbox_y_min_px": y0,
                        "bbox_x_max_px": x1,
                        "bbox_y_max_px": y1,
                        "source_image_id": img_id,
                        "source_embryo_id": eid,
                        "source_note": (
                            "grounded_sam_segmentations_20260202.json; "
                            f"video {well}; frame {fidx}; per-embryo bbox"
                        ),
                    }
                )

            out_path = OUT_DIR / f"{img_id}_multi_embryo_boxes.png"
            im.save(out_path)
            overlay_paths.append(out_path)
            print(f"wrote {out_path} ({len(boxes)} embryos)")

    csv_path = OUT_DIR / "candidate_exemplars_multi_embryo_wells.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} exemplar rows)")

    # contact sheet
    thumbs = []
    max_w = 0
    total_h = 0
    for p in overlay_paths:
        im = Image.open(p)
        im.thumbnail((700, 700))
        thumbs.append((p.name, im))
        max_w = max(max_w, im.width)
        total_h += im.height + 30

    sheet = Image.new("RGB", (max_w * 3 + 40, (total_h // 3) + 200), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    cols = 3
    x_pad, y_pad = 10, 10
    col_w = max_w + x_pad
    row_h = max((t[1].height for t in thumbs), default=700) + 40
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (col_w * cols + x_pad, row_h * rows_n + y_pad), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (name, im) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = x_pad + c * col_w
        y = y_pad + r * row_h
        sheet.paste(im, (x, y))
        draw.text((x, y + im.height + 4), name, fill="black", font=font)

    sheet_path = OUT_DIR / "multi_embryo_wells_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}")


if __name__ == "__main__":
    main()
