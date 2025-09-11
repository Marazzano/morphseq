#!/usr/bin/env bash
# Simple wrapper to render a SAM2 evaluation video using VideoGenerator.
# Edit the defaults below and run: ./make_eval_video.sh

set -euo pipefail

# --- Defaults (edit these) -------------------------------------------------
RESULTS_JSON="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations.json"
# EXP_ID is optional; if empty, it will be derived from VIDEO_ID
EXP_ID=""
# Provide either a single VIDEO_ID or a comma-separated list in VIDEOS
VIDEO_ID="20240418_D04"
VIDEOS=""  # e.g., `"20240418_D04,20240418_D05" (overrides VIDEO_ID if set)

# Output policy: use directory + suffix to form filenames
OUT_DIR="./net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/videos/20250907"
OUT_SUFFIX="_eval"

# Overlay options
SHOW_BBOX=true
SHOW_MASK=true
SHOW_METRICS=true
SHOW_QC=true
# --------------------------------------------------------------------------

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

# Ensure Python can import the video_generation package
export PYTHONPATH="${script_dir}/..:${PYTHONPATH:-}"

# Build flag list based on booleans
flags=( )
[[ "${SHOW_BBOX}" == "true" ]] && flags+=("--show-bbox")
[[ "${SHOW_MASK}" != "true" ]] && flags+=("--no-mask")
[[ "${SHOW_METRICS}" != "true" ]] && flags+=("--no-metrics")
[[ "${SHOW_QC}" == "true" ]] && flags+=("--show-qc")

# Build base args
args=( --json "${RESULTS_JSON}" --out-dir "${OUT_DIR}" --suffix "${OUT_SUFFIX}" )

# Pass video ids: either comma list or single
if [[ -n "${VIDEOS}" ]]; then
  args+=( --videos "${VIDEOS}" )
else
  args+=( --video "${VIDEO_ID}" )
fi

# Only pass --exp if set; otherwise the CLI derives it from video_id
if [[ -n "${EXP_ID}" ]]; then
  args+=(--exp "${EXP_ID}")
fi

python3 "${script_dir}/render_eval_video.py" "${args[@]}" "${flags[@]}"
echo "✅ Done. Outputs written under: ${OUT_DIR}"
