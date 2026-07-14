import json
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PLAYGROUND_ROOT = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files"
)
OUT_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/tmp/sam3_exemplar_review/multi_embryo_wells/new_batch"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (seg_json_name, experiment_key, well, frame_index)
TARGETS = [
    ("grounded_sam_segmentations_20260502_zfpm_pilot.json", "20260502_zfpm_pilot", "20260502_zfpm_pilot_A04", 79),
    ("grounded_sam_segmentations_20260502_zfpm_pilot.json", "20260502_zfpm_pilot", "20260502_zfpm_pilot_F05", 9),
    ("grounded_sam_segmentations_20251121.json", "20251121", "20251121_C12", 172),
]

COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 120, 255),
    (255, 165, 0),
    (200, 0, 200),
    (0, 200, 200),
]


def main():
    rows = []
    overlay_paths = []
    seg_cache = {}

    for seg_json, exp_key, well, frame in TARGETS:
        if seg_json not in seg_cache:
            with open(PLAYGROUND_ROOT / "segmentation" / seg_json) as f:
                seg_cache[seg_json] = json.load(f)
        data = seg_cache[seg_json]
        v = data["experiments"][exp_key]["videos"][well]
        imgs = v["image_ids"]
        by_idx = {img["frame_index"]: iid for iid, img in imgs.items()}
        img_id = by_idx.get(frame)
        if img_id is None:
            print(f"WARNING: {well} missing frame_index {frame}")
            continue
        img_entry = imgs[img_id]
        embryos = img_entry["embryos"]
        if not embryos:
            print(f"WARNING: {well} frame {frame} has no embryos in JSON")
            continue

        seg_w, seg_h = next(iter(embryos.values()))["segmentation"]["size"]
        boxes = {}
        for eid, e in embryos.items():
            bx = e["segmentation"]["bbox"]
            boxes[eid] = [bx[0] * seg_w, bx[1] * seg_h, bx[2] * seg_w, bx[3] * seg_h]

        img_path = PLAYGROUND_ROOT / "raw_data_organized" / exp_key / "images" / well / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"WARNING: missing raw image {img_path}")
            continue

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
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
                        f"{seg_json}; video {well}; frame {frame}; per-embryo bbox; "
                        "user-selected exemplar frame"
                    ),
                }
            )

        out_path = OUT_DIR / f"{img_id}_boxes.png"
        im.save(out_path)
        overlay_paths.append(out_path)
        print(f"wrote {out_path} ({len(boxes)} embryos in JSON)")

    if rows:
        csv_path = OUT_DIR / "candidate_exemplars_new_batch.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path} ({len(rows)} exemplar rows)")

    # contact sheet
    thumbs = []
    for p in overlay_paths:
        im = Image.open(p)
        im.thumbnail((700, 700))
        thumbs.append((p.name, im))

    cols = 3
    x_pad, y_pad = 10, 10
    max_w = max((t[1].width for t in thumbs), default=700)
    row_h = max((t[1].height for t in thumbs), default=700) + 40
    rows_n = (len(thumbs) + cols - 1) // cols
    col_w = max_w + x_pad
    sheet = Image.new("RGB", (col_w * cols + x_pad, row_h * rows_n + y_pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for idx, (name, im) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = x_pad + c * col_w
        y = y_pad + r * row_h
        sheet.paste(im, (x, y))
        draw.text((x, y + im.height + 4), name, fill="black", font=font)

    sheet_path = OUT_DIR / "new_batch_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}")


if __name__ == "__main__":
    main()
