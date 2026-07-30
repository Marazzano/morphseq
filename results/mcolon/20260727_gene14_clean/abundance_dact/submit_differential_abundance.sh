#!/usr/bin/env bash

# Submit one resumable array task per planned contrast. When every fit finishes,
# a final dependent job assembles the raw checkpoints into one result table.

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
ABUNDANCE_DIR="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/abundance_dact"
PLAN_FILE="$ABUNDANCE_DIR/contrast_plan.R"
JOB_SCRIPT="$ABUNDANCE_DIR/qsub_differential_abundance.sh"
LOG_DIR="$ABUNDANCE_DIR/output/logs"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

mkdir -p "$LOG_DIR"

NUMBER_OF_CONTRASTS=$(
  PLAN_FILE="$PLAN_FILE" "$RSCRIPT" -e \
    'source(Sys.getenv("PLAN_FILE")); cat(nrow(contrast_plan))'
)

FIT_JOB=$(
  qsub -terse \
    -N gene14_dact_fit \
    -t "1-$NUMBER_OF_CONTRASTS" \
    -tc 3 \
    -o "$LOG_DIR" \
    -v DACT_MODE=fit_one,PLAN_FILE="$PLAN_FILE" \
    "$JOB_SCRIPT"
)

# qsub reports an array job as JOB_ID.1-N:1. The assembly dependency needs the
# parent JOB_ID.
FIT_JOB_ID="${FIT_JOB%%.*}"

ASSEMBLY_JOB=$(
  qsub -terse \
    -N gene14_dact_assemble \
    -hold_jid "$FIT_JOB_ID" \
    -o "$LOG_DIR" \
    -v DACT_MODE=assemble,PLAN_FILE="$PLAN_FILE" \
    "$JOB_SCRIPT"
)

echo "Submitted $NUMBER_OF_CONTRASTS contrast tasks: $FIT_JOB"
echo "Submitted dependent assembly job: $ASSEMBLY_JOB"
echo "Logs: $LOG_DIR"
