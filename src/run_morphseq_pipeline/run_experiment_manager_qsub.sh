#!/usr/bin/env bash
#$ -N morphseq_experiment_mngr            # job name
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=8G
#$ -l h_rt=120:00:00           # walltime (adjust as needed)
#$ -j y                       # merge stdout/stderr
#$ -pe serial 12
#$ -cwd
#$ -V
#$ -o logs
#$ -e logs

set -euo pipefail

# --- EDIT THESE DEFAULTS (simple assignments) -------------------------------
REPO_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
DATA_ROOT="${REPO_ROOT}/morphseq_playground"
# EXPERIMENTS="all"
EXPERIMENTS="20250512,20250711,20250515_part2,20250519,20250520"
DRY_RUN="0"                 # set to 1 to enable --dry-run
ENV_NAME="segmentation_grounded_sam"
MSEQ_OVERWRITE_BUILD01="0"   # force FF recompute (both Keyence/YX1)
MSEQ_OVERWRITE_STITCH="0"    # force restitch (Keyence)
# MSEQ_BUILD01_DEBUG="1"       # enable verbose logging in build01 step
# MSEQ_TRACE="1"
# MSEQ_YX1_DEBUG="1"
# export MSEQ_OVERWRITE_BUILD01 MSEQ_OVERWRITE_STITCH MSEQ_BUILD01_DEBUG

# ----------------------------------------------------------------------------

echo "[morphseq] JOB_ID=${JOB_ID:-unknown} TASK=${SGE_TASK_ID:-0}"
echo "[morphseq] Repo root : ${REPO_ROOT}"
echo "[morphseq] Data root : ${DATA_ROOT}"

# Support SGE array jobs: select one experiment per task using SGE_TASK_ID

# qsub -t 1 -tc 3   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list.txt   src/run_morphseq_pipeline/run_experiment_manager_qsub.sh
if [[ -n "${SGE_TASK_ID:-}" ]]; then
  if [[ "${SGE_TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "[morphseq] Array task ID: ${SGE_TASK_ID}"
    if [[ -n "${EXP_FILE:-}" && -f "${EXP_FILE}" ]]; then
      mapfile -t _EXPS < "${EXP_FILE}"
    elif [[ -n "${EXP_LIST:-}" ]]; then
      IFS=',' read -r -a _EXPS <<< "${EXP_LIST}"
    else
      IFS=',' read -r -a _EXPS <<< "${EXPERIMENTS}"
    fi
    _IDX=$(( SGE_TASK_ID - 1 ))
    if (( _IDX < 0 || _IDX >= ${#_EXPS[@]} )); then
      echo "[morphseq] ERROR: Task index $_IDX out of range (len=${#_EXPS[@]})" >&2
      exit 1
    fi
    EXPERIMENTS="${_EXPS[_IDX]}"
  else
    echo "[morphseq] SGE_TASK_ID='${SGE_TASK_ID}' is non-numeric; ignoring array selection"
  fi
fi

echo "[morphseq] Experiments: ${EXPERIMENTS}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "[morphseq] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Create logs dir if running interactively without SGE stdout redirection
mkdir -p logs

# Activate conda environment (robust to libmamba issues)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # Work around libmamba solver issues by forcing classic solver
    export CONDA_SOLVER=classic
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}" || echo "[morphseq] WARNING: failed to activate ${ENV_NAME}; continuing"
  fi
else
  echo "[morphseq] WARNING: conda not found in PATH; proceeding without activation"
fi

# Ensure Python can import the repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CMD=( python -m src.run_morphseq_pipeline.cli pipeline \
      --data-root "${DATA_ROOT}" \
      --experiments "${EXPERIMENTS}" \
      --action e2e \
      --model-name 20241107_ds_sweep01_optimum )

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  CMD+=( --dry-run )
fi

echo "[morphseq] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[morphseq] Done."



# qsub -t 1-14 -tc 3 \
#   -v EXP_FILE=/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_lists/20250905_list_all.txt \
#   /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_manager_qsub.sh
