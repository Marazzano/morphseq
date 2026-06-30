#!/bin/bash
set -euo pipefail

MORPHSEQ_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq"
BASE="$MORPHSEQ_ROOT/results/mcolon/20260625_cross_experiment_focus_motion_qc"
SCRIPT="$BASE/cross_experiment_qc_audit.py"
QSUB_SCRIPT="$BASE/run_cross_experiment_qc.qsub"
PY="/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python"

EXPERIMENTS="${1:-20250305,20251125,20260206}"
METRIC_TYPE="${METRIC_TYPE:-both}"

mkdir -p "$BASE/logs" "$BASE/tables" "$BASE/figures"

echo "Running preflight for experiments=$EXPERIMENTS"
"$PY" "$SCRIPT" \
  --experiments "$EXPERIMENTS" \
  --metric-type "$METRIC_TYPE" \
  --out-dir "$BASE" \
  --preflight-only

echo "Submitting one-GPU qsub job"
SUBMIT_OUTPUT=$(EXPERIMENTS="$EXPERIMENTS" METRIC_TYPE="$METRIC_TYPE" qsub "$QSUB_SCRIPT")
echo "$SUBMIT_OUTPUT"
JOB_ID=$(printf "%s\n" "$SUBMIT_OUTPUT" | sed -n 's/.* job \([0-9][0-9]*\) .*/\1/p')
if [ -n "$JOB_ID" ]; then
  echo "Monitor:"
  echo "  qstat -j $JOB_ID"
  echo "  tail -f $BASE/logs/cross_exp_qc.o$JOB_ID"
  echo "  tail -f $BASE/logs/cross_exp_qc.e$JOB_ID"
fi
