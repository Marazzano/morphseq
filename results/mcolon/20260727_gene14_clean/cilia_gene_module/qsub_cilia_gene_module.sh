#!/usr/bin/env bash
#$ -N cilia_module
#$ -q trapnell-long.q
#$ -l mfree=96G
#$ -l h_rt=24:00:00
#$ -j y
#$ -pe serial 4
#$ -cwd
#$ -V

# Render one explicit mode of the readable cilia-module notebook.
#
# Examples:
#   CILIA_MODE=prepare_modules qsub qsub_cilia_gene_module.sh
#   CILIA_MODE=score_reference qsub qsub_cilia_gene_module.sh
#   CILIA_MODE=score_gene14 qsub qsub_cilia_gene_module.sh
#   CILIA_MODE=assemble qsub qsub_cilia_gene_module.sh
#
# For the simplest one-job run:
#   CILIA_MODE=all qsub qsub_cilia_gene_module.sh

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
MODULE_DIR="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/cilia_gene_module"
RMD_FILE="$MODULE_DIR/cilia_gene_module.Rmd"
REPORT_DIR="$MODULE_DIR/output/reports"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

mkdir -p "$REPORT_DIR"

# BPCells needs HDF5 on every execution node.
HDF5_LIB=/net/gs/vol3/software/modules-sw/hdf5/1.14.3-cxxenabled/Linux/Ubuntu22.04/x86_64/lib
export LD_LIBRARY_PATH="${HDF5_LIB}:${LD_LIBRARY_PATH:-}"

# Each job gets an isolated BPCells and R Markdown scratch directory.
TASK_LABEL="${JOB_ID:-local}_${CILIA_MODE:-prepare_modules}"
export TMPDIR="$PROJECT_DIR/tmp/bpcells_scratch/$TASK_LABEL"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$TMPDIR"

CILIA_MODE="${CILIA_MODE:-prepare_modules}"
REPORT_NAME="cilia_gene_module_${CILIA_MODE}.html"
export CILIA_MODE PROJECT_DIR RMD_FILE REPORT_DIR REPORT_NAME

echo "mode=$CILIA_MODE"
echo "host=$(hostname) slots=${NSLOTS:-NA} start=$(date)"

"$RSCRIPT" -e '
  rmarkdown::render(
    input = Sys.getenv("RMD_FILE"),
    params = list(mode = Sys.getenv("CILIA_MODE")),
    output_file = Sys.getenv("REPORT_NAME"),
    output_dir = Sys.getenv("REPORT_DIR"),
    intermediates_dir = Sys.getenv("TMPDIR"),
    knit_root_dir = Sys.getenv("PROJECT_DIR"),
    envir = new.env(),
    quiet = FALSE
  )
'

echo "done=$(date)"
