#!/usr/bin/env bash
#$ -N morphseq_sam2_onwards             # job name
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=8G
#$ -l h_rt=120:00:00                    # walltime (adjust as needed)
#$ -j y                                 # merge stdout/stderr
#$ -pe serial 12
#$ -cwd
#$ -V
#$ -o logs
#$ -e logs

# SAM2 onwards pipeline with SGE array job support
# Usage with array jobs:
# qsub -t 1-14 -tc 3 -v EXP_FILE=/path/to/experiment_list.txt run_sam2_onwards.sh

set -euo pipefail

# --- CONFIGURATION -----------------------------------------------------------
REPO_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
DATA_ROOT="${REPO_ROOT}/morphseq_playground" 
MODEL_NAME="20241107_ds_sweep01_optimum"
ENV_NAME="segmentation_grounded_sam"

# Default experiment list (used if not running as array job)
DEFAULT_EXPERIMENTS="20250305"

# ----------------------------------------------------------------------------

echo "[sam2-onwards] JOB_ID=${JOB_ID:-unknown} TASK=${SGE_TASK_ID:-0}"
echo "[sam2-onwards] Repo root : ${REPO_ROOT}"
echo "[sam2-onwards] Data root : ${DATA_ROOT}"

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

echo "[sam2-onwards] Processing experiment: ${EXPERIMENT}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[sam2-onwards] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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

# Step 2: Build03  
echo "🔄 Step 2: Running Build03 for ${EXPERIMENT}..."
python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build03 \
    --model-name "${MODEL_NAME}" \
    --force

# Step 3: Build04
echo "🔄 Step 3: Running Build04 for ${EXPERIMENT}..."
python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build04 \
    --model-name "${MODEL_NAME}" \
    --force

# Step 4: Build06
echo "🔄 Step 4: Running Build06 for ${EXPERIMENT}..."
python -m src.run_morphseq_pipeline.cli pipeline \
    --data-root "${DATA_ROOT}" \
    --experiments "${EXPERIMENT}" \
    --action build06 \
    --model-name "${MODEL_NAME}" \
    --force

echo "🎉 SAM2 onwards pipeline completed for ${EXPERIMENT}!"

# Example usage with array jobs:
#
# Run for all experiments in list:
# qsub -t 1-18 -tc 3 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   run_sam2_onwards_force.sh
#
# Run for single experiment (test):
# qsub -t 1 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   run_sam2_onwards_force.sh
