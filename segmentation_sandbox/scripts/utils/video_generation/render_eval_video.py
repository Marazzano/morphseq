#!/usr/bin/env python3
"""
CLI: Render a SAM2 evaluation video using VideoGenerator.

Example:
  python render_eval_video.py \
    --json /path/to/sam2_pipeline_files/segmentation/grounded_sam_segmentations.json \
    --exp 20240418_D0 \
    --video 20240418_D0_A01 \
    --out ./20240418_D0_A01_eval.mp4 \
    --show-qc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Robust import: allow running this file directly without PYTHONPATH set
try:
    from video_generation.video_generator import VideoGenerator
except ModuleNotFoundError:
    # Add the package root (…/scripts/utils) to sys.path
    _pkg_root = Path(__file__).resolve().parents[1]
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from video_generation.video_generator import VideoGenerator


def main() -> int:
    p = argparse.ArgumentParser(description="Render SAM2 evaluation video(s)")
    p.add_argument("--json", required=True, help="Path to SAM2 results JSON (grounded_sam_segmentations.json)")
    p.add_argument("--exp", required=False, help="Experiment ID (auto-derived from video IDs if omitted)")
    p.add_argument("--video", action="append", help="Video ID; may be specified multiple times")
    p.add_argument("--videos", help="Comma-separated list of video IDs")
    p.add_argument("--all-in-exp", action="store_true", help="Process all videos under --exp")
    p.add_argument("--out", help="Output MP4 path (only when processing a single video)")
    p.add_argument("--out-dir", default=".", help="Output directory for MP4 files (batch or default single)")
    p.add_argument("--suffix", default="_eval", help="Suffix to append to video_id for output filenames")
    p.add_argument("--show-bbox", action="store_true", help="Show bounding boxes")
    p.add_argument("--no-mask", action="store_true", help="Disable mask overlay")
    p.add_argument("--no-metrics", action="store_true", help="Disable metrics overlay")
    p.add_argument("--show-qc", action="store_true", help="Show QC flags overlay (if present)")
    args = p.parse_args()

    # Build video list from inputs
    videos: list[str] = []
    if args.video:
        videos.extend(args.video)
    if args.videos:
        videos.extend([v.strip() for v in args.videos.split(',') if v.strip()])

    # Load JSON once; possibly discover all videos in an experiment
    results_json = Path(args.json)
    data = None
    if args.all_in_exp:
        if not args.exp:
            print("ERROR: --all-in-exp requires --exp to be set")
            return 1
        try:
            import json
            data = json.loads(results_json.read_text())
        except Exception as e:
            print(f"❌ Failed to load JSON: {e}")
            return 1
        exp_data = data.get("experiments", {}).get(args.exp, {})
        videos_dict = exp_data.get("videos", {})
        discovered = sorted(videos_dict.keys())
        if not discovered:
            print(f"❌ No videos found under experiment {args.exp}")
            return 1
        videos = discovered

    if not videos:
        print("ERROR: No videos specified. Use --video/--videos or --all-in-exp with --exp.")
        return 1

    # Helper to derive experiment id from video id
    import re
    def derive_exp_id(vid: str) -> str:
        if args.exp:
            return args.exp
        m = re.match(r"^(.+)_([A-H][0-9]{2})$", vid)
        if m:
            return m.group(1)
        return vid.rsplit('_', 1)[0] if '_' in vid else vid

    # Decide output strategy
    multi = len(videos) > 1
    if multi and args.out:
        print("⚠️  Ignoring --out because multiple videos were provided; using --out-dir and --suffix")

    vg = VideoGenerator()
    ok_all = True
    from pathlib import Path as _P
    out_dir = _P(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in videos:
        exp_id = derive_exp_id(vid)
        if not multi and args.out:
            out_path = _P(args.out)
        else:
            out_name = f"{vid}{args.suffix}.mp4"
            out_path = out_dir / out_name

        print(f"🎬 Rendering {vid} (exp={exp_id}) → {out_path}")
        ok = vg.create_sam2_eval_video_from_results(
            results_json_path=results_json,
            experiment_id=exp_id,
            video_id=vid,
            output_video_path=out_path,
            show_bbox=args.show_bbox,
            show_mask=not args.no_mask,
            show_metrics=not args.no_metrics,
            show_qc_flags=args.show_qc,
            verbose=True,
        )
        ok_all = ok_all and ok

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
