#!/usr/bin/env bash
#$ -N cilia_diff
#$ -q trapnell-short.q
#$ -l mfree=16G
#$ -l h_rt=02:00:00
#$ -j y
#$ -pe serial 1
#$ -cwd
#$ -V

# Fit one requested embryo-level cilia-module contrast and render its report.
#
# Example:
#   CILIA_CONTRAST=foxj1a_48hpf qsub qsub_differential_cilia_modules.sh

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
MODULE_DIR="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/cilia_gene_module"
RMD_FILE="$MODULE_DIR/differential_cilia_modules.Rmd"
REPORT_DIR="$MODULE_DIR/output/reports"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

mkdir -p "$REPORT_DIR"

export TMPDIR="$PROJECT_DIR/tmp/cilia_module_models/${JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT

CILIA_CONTRAST="${CILIA_CONTRAST:-all}"
REPORT_NAME="differential_cilia_modules_${CILIA_CONTRAST}.html"
export PROJECT_DIR RMD_FILE REPORT_DIR CILIA_CONTRAST REPORT_NAME

echo "contrast=$CILIA_CONTRAST host=$(hostname) start=$(date)"

"$RSCRIPT" -e '
  rmarkdown::render(
    input = Sys.getenv("RMD_FILE"),
    params = list(contrast_name = Sys.getenv("CILIA_CONTRAST")),
    output_file = Sys.getenv("REPORT_NAME"),
    output_dir = Sys.getenv("REPORT_DIR"),
    intermediates_dir = Sys.getenv("TMPDIR"),
    knit_root_dir = Sys.getenv("PROJECT_DIR"),
    envir = new.env(),
    quiet = FALSE
  )
'

echo "done=$(date)"
