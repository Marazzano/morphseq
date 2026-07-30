#!/usr/bin/env bash
#$ -q trapnell-login.q
#$ -l mfree=24G
#$ -l h_rt=12:00:00
#$ -j y
#$ -pe serial 8
#$ -cwd
#$ -V

# Run one contrast, or assemble all completed contrasts, by rendering the
# readable differential-abundance notebook with explicit parameters.

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
ABUNDANCE_DIR="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/abundance_dact"
RMD_FILE="$ABUNDANCE_DIR/differential_abundance.Rmd"
PLAN_FILE="$ABUNDANCE_DIR/contrast_plan.R"
REPORT_DIR="$ABUNDANCE_DIR/output/reports"
LOG_DIR="$ABUNDANCE_DIR/output/logs"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

mkdir -p "$REPORT_DIR" "$LOG_DIR"

# BPCells needs HDF5 on every execution node.
HDF5_LIB=/net/gs/vol3/software/modules-sw/hdf5/1.14.3-cxxenabled/Linux/Ubuntu22.04/x86_64/lib
export LD_LIBRARY_PATH="${HDF5_LIB}:${LD_LIBRARY_PATH:-}"

# Give each array task its own BPCells and R Markdown scratch directory.
TASK_LABEL="${JOB_ID:-local}_${SGE_TASK_ID:-assemble}"
export TMPDIR="$PROJECT_DIR/tmp/bpcells_scratch/$TASK_LABEL"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$TMPDIR"

DACT_MODE="${DACT_MODE:-fit_one}"
export DACT_MODE RMD_FILE PLAN_FILE REPORT_DIR PROJECT_DIR

if [[ "$DACT_MODE" == "fit_one" ]]; then
  if [[ -z "${SGE_TASK_ID:-}" ]]; then
    echo "fit_one mode requires an SGE array task ID." >&2
    exit 1
  fi

  export DACT_CONTRAST_NAME
  DACT_CONTRAST_NAME=$(
    "$RSCRIPT" -e \
      'source(Sys.getenv("PLAN_FILE")); cat(contrast_plan$contrast_name[[as.integer(Sys.getenv("SGE_TASK_ID"))]])'
  )
  export DACT_REPORT_NAME="fit_${DACT_CONTRAST_NAME}.html"
else
  export DACT_CONTRAST_NAME=""
  export DACT_REPORT_NAME="differential_abundance.html"
fi

echo "mode=$DACT_MODE"
echo "contrast=${DACT_CONTRAST_NAME:-all}"
echo "host=$(hostname) slots=${NSLOTS:-NA} start=$(date)"

"$RSCRIPT" -e '
  contrast_name <- Sys.getenv("DACT_CONTRAST_NAME")
  if (!nzchar(contrast_name)) {
    contrast_name <- NULL
  }

  rmarkdown::render(
    input = Sys.getenv("RMD_FILE"),
    params = list(
      mode = Sys.getenv("DACT_MODE"),
      contrast_name = contrast_name
    ),
    output_file = Sys.getenv("DACT_REPORT_NAME"),
    output_dir = Sys.getenv("REPORT_DIR"),
    intermediates_dir = Sys.getenv("TMPDIR"),
    knit_root_dir = Sys.getenv("PROJECT_DIR"),
    envir = new.env(),
    quiet = FALSE
  )
'

echo "done=$(date)"
