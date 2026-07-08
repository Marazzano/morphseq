#!/bin/bash
set -euo pipefail

BASE="/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260625_cross_experiment_focus_motion_qc"
JOB_ID="${1:-}"

if [ -z "$JOB_ID" ]; then
  latest=$(ls -t "$BASE"/logs/cross_exp_qc.o* 2>/dev/null | head -n 1 || true)
  if [ -z "$latest" ]; then
    echo "No cross_exp_qc logs found under $BASE/logs"
    exit 1
  fi
  JOB_ID="${latest##*.o}"
fi

echo "Job: $JOB_ID"
echo "qstat:"
qstat -j "$JOB_ID" 2>/dev/null | sed -n '1,80p' || echo "  job not currently visible to qstat"

echo
echo "stdout tail: $BASE/logs/cross_exp_qc.o$JOB_ID"
tail -n 80 "$BASE/logs/cross_exp_qc.o$JOB_ID" 2>/dev/null || echo "  stdout log not available yet"

echo
echo "stderr tail: $BASE/logs/cross_exp_qc.e$JOB_ID"
tail -n 80 "$BASE/logs/cross_exp_qc.e$JOB_ID" 2>/dev/null || echo "  stderr log not available yet"
