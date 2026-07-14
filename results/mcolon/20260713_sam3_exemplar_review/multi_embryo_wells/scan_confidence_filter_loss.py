"""Scan all gdino_detections_*.json files (raw, pre-filter GroundingDINO output) and count
how many individual detections fall in the "filtered zone": passed the base detector's
box_threshold (0.35, per pipeline_config.yaml) but got cut by the stricter high-quality
confidence_threshold=0.45 default in 03_gdino_detection.py, BEFORE SAM2 seeding ever runs.

This quantifies how many real embryos may be silently missing from seed_frame_info across
the whole dataset due to that threshold gap, not just the one F05 well already found.
"""
import json
import glob
from pathlib import Path

DETECTIONS_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/detections"
)
BOX_THRESHOLD = 0.35   # base GroundingDINO detector gate (pipeline_config.yaml)
HQ_THRESHOLD = 0.45    # stricter high-quality filter default (03_gdino_detection.py CLI default)

PROMPT = "individual embryo"


def main():
    files = sorted(DETECTIONS_DIR.glob("gdino_detections_*.json"))
    files = [f for f in files if ".backup." not in f.name]

    total_raw = 0
    total_in_filtered_zone = 0  # 0.35 <= conf < 0.45
    total_kept = 0              # conf >= 0.45
    images_with_loss = 0
    total_images = 0

    per_experiment = []

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"SKIP {fp.name}: {e}")
            continue

        exp_raw = 0
        exp_filtered_zone = 0
        exp_kept = 0
        exp_images_with_loss = 0
        exp_images = 0

        images = data.get("images", {})
        for image_id, image_data in images.items():
            for ann in image_data.get("annotations", []):
                if ann.get("prompt") != PROMPT:
                    continue
                dets = ann.get("detections", [])
                if not dets:
                    continue
                exp_images += 1
                confs = [d["confidence"] for d in dets]
                n_in_zone = sum(1 for c in confs if BOX_THRESHOLD <= c < HQ_THRESHOLD)
                n_kept = sum(1 for c in confs if c >= HQ_THRESHOLD)
                exp_raw += len(confs)
                exp_filtered_zone += n_in_zone
                exp_kept += n_kept
                if n_in_zone > 0:
                    exp_images_with_loss += 1

        total_raw += exp_raw
        total_in_filtered_zone += exp_filtered_zone
        total_kept += exp_kept
        images_with_loss += exp_images_with_loss
        total_images += exp_images

        if exp_raw > 0:
            per_experiment.append(
                {
                    "experiment": fp.stem.replace("gdino_detections_", ""),
                    "images_with_annotations": exp_images,
                    "images_with_filtered_loss": exp_images_with_loss,
                    "raw_detections": exp_raw,
                    "kept_ge_0.45": exp_kept,
                    "lost_0.35_to_0.45": exp_filtered_zone,
                }
            )

    per_experiment.sort(key=lambda r: r["lost_0.35_to_0.45"], reverse=True)

    print(f"Scanned {len(files)} experiment files\n")
    print(f"{'experiment':<45} {'images':>8} {'img_w_loss':>10} {'raw':>7} {'kept':>7} {'lost(0.35-0.45)':>16}")
    for r in per_experiment[:30]:
        print(
            f"{r['experiment']:<45} {r['images_with_annotations']:>8} {r['images_with_filtered_loss']:>10} "
            f"{r['raw_detections']:>7} {r['kept_ge_0.45']:>7} {r['lost_0.35_to_0.45']:>16}"
        )

    print("\n=== TOTALS ===")
    print(f"total images with '{PROMPT}' annotations: {total_images}")
    print(f"images with at least one detection in the 0.35-0.45 lost zone: {images_with_loss}")
    print(f"total raw detections (>= 0.35): {total_raw}")
    print(f"total kept (>= 0.45): {total_kept}")
    print(f"total LOST to the 0.45 filter (0.35 <= conf < 0.45): {total_in_filtered_zone}")
    if total_raw:
        print(f"fraction of raw detections lost to this gap: {total_in_filtered_zone/total_raw:.1%}")


if __name__ == "__main__":
    main()
