#!/usr/bin/env bash
#$ -N morphseq_sam2_onwards             # job name
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=8G
#$ -l h_rt=120:00:00                    # walltime (adjust as needed)
#$ -j y                                 # merge stdout/stderr
#$ -pe serial 1
#$ -cwd
#$ -V
#$ -o logs
#$ -e logs

# SAM2-onwards pipeline with optional metadata rebuild.
# Usage with array jobs:
#   qsub -t 1-14 -tc 3 -v EXP_FILE=/path/to/experiment_list.txt run_build03_onwards_force.sh
# Override behaviour with env vars:
#   RUN_METADATA_REBUILD=0   # skip Build01 metadata-only refresh
#   RUN_SAM2=0               # skip re-running SAM2
#   RUN_BUILD03=0            # skip Build03 action
#   RUN_SNIP_EXPORT=1        # export snips only (use existing Build03 metadata)
#   RUN_BUILD04=0            # skip Build04 action
#   RUN_BUILD06=0            # skip Build06 action
#   SAM2_WORKERS=8           # SAM2 worker count (default 8)
#   SAM2_CONFIDENCE=0.45     # SAM2 confidence threshold
#   SAM2_IOU=0.5             # SAM2 IoU threshold

set -euo pipefail

# --- CONFIGURATION -----------------------------------------------------------
REPO_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
DATA_ROOT="${REPO_ROOT}/morphseq_playground"
MODEL_NAME="20241107_ds_sweep01_optimum"
ENV_NAME="segmentation_grounded_sam"

# Default experiment list (used if not running as array job)
DEFAULT_EXPERIMENTS="20250305"

# Tunable defaults — override by exporting the variable before invoking this script.
# Example: RUN_SAM2=0 SAM2_WORKERS=2 EXP_LIST=20250305 bash run_build03_onwards_force.sh
: "${METADATA_MICROSCOPE:=YX1}"

# SAM2 inference knobs
: "${SAM2_WORKERS:=8}"
: "${SAM2_CONFIDENCE:=0.45}"
: "${SAM2_IOU:=0.5}"

# Pipeline stage toggles (1=run, 0=skip)
: "${RUN_METADATA_REBUILD:=1}"
: "${RUN_SAM2:=1}"
: "${RUN_BUILD03:=1}"
: "${BUILD03_SKIP_GEOMETRY_QC:=0}"
: "${RUN_BUILD04:=1}"
: "${RUN_BUILD06:=1}"
: "${RUN_SNIP_EXPORT:=1}"

# Snip export knobs (outscale fixed at 6.5 to match embedding expectations)
: "${SNIP_WORKERS:=1}"
: "${SNIP_DL_RAD_UM:=50}"
: "${SNIP_OVERWRITE:=0}"
# ----------------------------------------------------------------------------

if [[ "${SNIP_OVERWRITE}" == "1" ]]; then
  SNIP_OVERWRITE_PY="True"
else
  SNIP_OVERWRITE_PY="False"
fi

echo "[sam2-onwards] JOB_ID=${JOB_ID:-unknown} TASK=${SGE_TASK_ID:-0}"
echo "[sam2-onwards] Repo root : ${REPO_ROOT}"
echo "[sam2-onwards] Data root : ${DATA_ROOT}"

echo "[sam2-onwards] Run flags - metadata:${RUN_METADATA_REBUILD} sam2:${RUN_SAM2} b03:${RUN_BUILD03} snip:${RUN_SNIP_EXPORT} b04:${RUN_BUILD04} b06:${RUN_BUILD06}"
echo "[sam2-onwards] Build03 flags - skip_geometry_qc:${BUILD03_SKIP_GEOMETRY_QC}"
echo "[sam2-onwards] SAM2 params - workers:${SAM2_WORKERS} conf:${SAM2_CONFIDENCE} iou:${SAM2_IOU}"
echo "[sam2-onwards] Snip params - workers:${SNIP_WORKERS} dl_rad:${SNIP_DL_RAD_UM} overwrite:${SNIP_OVERWRITE}"

# Support SGE array jobs: select one experiment per task using SGE_TASK_ID
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  if [[ "${SGE_TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "[sam2-onwards] Array task ID: ${SGE_TASK_ID}"
    if [[ -n "${EXP_FILE:-}" && -f "${EXP_FILE}" ]]; then
      mapfile -t _EXPS < "${EXP_FILE}"
    elif [[ -n "${EXP_LIST:-}" ]]; then
      IFS=',' read -r -a _EXPS <<< "${EXP_LIST}"
    else
      IFS=',' read -r -a _EXPS <<< "${DEFAULT_EXPERIMENTS}"
    fi
    _IDX=$(( SGE_TASK_ID - 1 ))
    if (( _IDX < 0 || _IDX >= ${#_EXPS[@]} )); then
      echo "[sam2-onwards] ERROR: Task index $_IDX out of range (len=${#_EXPS[@]})" >&2
      exit 1
    fi
    EXPERIMENT="${_EXPS[_IDX]}"
  else
    echo "[sam2-onwards] SGE_TASK_ID='${SGE_TASK_ID}' is non-numeric; ignoring array selection"
    EXPERIMENT="${DEFAULT_EXPERIMENTS}"
  fi
else
  # Running interactively or non-array job
  EXPERIMENT="${DEFAULT_EXPERIMENTS}"
fi

echo "[sam2-onwards] Processing experiment(s): ${EXPERIMENT}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[sam2-onwards] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a SELECTED_EXPERIMENTS <<< "${EXPERIMENT}"

# Create logs dir if running interactively
mkdir -p logs

# Activate conda environment (robust to libmamba issues)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    export CONDA_SOLVER=classic
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}" || echo "[sam2-onwards] WARNING: failed to activate ${ENV_NAME}; continuing"
  fi
else
  echo "[sam2-onwards] WARNING: conda not found in PATH; proceeding without activation"
fi

# Ensure Python can import the repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "🚀 Starting SAM2 onwards pipeline for ${EXPERIMENT}..."

if [[ "${RUN_METADATA_REBUILD}" == "1" ]]; then
  for exp_name in "${SELECTED_EXPERIMENTS[@]}"; do
    echo "🔄 Pre-step: Build01 metadata-only for ${exp_name}"
    python -m src.run_morphseq_pipeline.cli build01 \
      --data-root "${DATA_ROOT}" \
      --exp "${exp_name}" \
      --microscope "${METADATA_MICROSCOPE}" \
      --metadata-only \
      --overwrite \
      || echo "[sam2-onwards] WARNING: Build01 metadata-only failed for ${exp_name}"
  done
fi

if [[ "${RUN_SAM2}" == "1" ]]; then
  echo "🔄 Step 1: Running SAM2 for ${EXPERIMENT}..."
  python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action sam2 \
    --sam2-workers "${SAM2_WORKERS}" \
    --sam2-confidence "${SAM2_CONFIDENCE}" \
    --sam2-iou "${SAM2_IOU}" \
    --force
fi

if [[ "${RUN_BUILD03}" == "1" ]]; then
  echo "🔄 Step 2: Running Build03 for ${EXPERIMENT}..."
  # Export flag for Python to read
  export BUILD03_SKIP_GEOMETRY_QC
  python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build03 \
    --model-name "${MODEL_NAME}" \
    --force
fi

if [[ "${RUN_SNIP_EXPORT}" == "1" ]]; then
  echo "🖼️  Step 2b: Exporting BF snips for ${EXPERIMENT}..."
  for exp_name in "${SELECTED_EXPERIMENTS[@]}"; do
    python - <<PY
from pathlib import Path
import pandas as pd
from src.build.build03A_process_images import extract_embryo_snips

data_root = Path("${DATA_ROOT}")
exp = "${exp_name}"
candidates = [
    data_root / "metadata" / "build03_output" / f"expr_embryo_metadata_{exp}.csv",
    data_root / "metadata" / "build03" / "per_experiment" / f"expr_embryo_metadata_{exp}.csv",
]
for csv_path in candidates:
    if csv_path.exists():
        stats_df = pd.read_csv(csv_path)
        break
else:
    checked = "\n   - ".join(str(p) for p in candidates)
    raise SystemExit(f"❌ Build03 metadata not found for {exp}. Checked paths:\\n   - {checked}")

extract_embryo_snips(
    root=data_root,
    stats_df=stats_df,
    outscale=6.5,
    dl_rad_um=float("${SNIP_DL_RAD_UM}"),
    overwrite_flag=${SNIP_OVERWRITE_PY},
    n_workers=int("${SNIP_WORKERS}"),
)
print(f"✅ Snip export complete for {exp}")
PY
  done
fi

if [[ "${RUN_BUILD04}" == "1" ]]; then
  echo "🔄 Step 3: Running Build04 for ${EXPERIMENT}..."
  python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build04 \
    --model-name "${MODEL_NAME}" \
    --force
fi

if [[ "${RUN_BUILD06}" == "1" ]]; then
  echo "🔄 Step 4: Running Build06 for ${EXPERIMENT}..."
  python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build06 \
    --model-name "${MODEL_NAME}" \
    --force
fi

echo "🎉 SAM2 onwards pipeline completed for ${EXPERIMENT}!"

# Example usage with array jobs:
#
# Run for all experiments in list:
# qsub -t 1-30 -tc 3 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh

# Run for single experiment (test):
# qsub -t 1 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   run_build03_onwards_force.sh
# qsub -t 1-2 -tc 2 \
#   /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_build03_onwards_force.sh
