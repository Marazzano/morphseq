#!/usr/bin/env bash
#$ -N foxj1a_inspect
#$ -q trapnell-short.q
#$ -l mfree=16G
#$ -l h_rt=02:00:00
#$ -j y
#$ -pe serial 1
#$ -cwd
#$ -V

# Build the focused foxj1a 48-hpf inspection report and all of its figures.

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
HERE="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/cilia_gene_module/foxj1a_48hpf_inspection"
RMD_FILE="$HERE/inspect_foxj1a_48hpf.Rmd"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

HDF5_LIB=/net/gs/vol3/software/modules-sw/hdf5/1.14.3-cxxenabled/Linux/Ubuntu22.04/x86_64/lib
export LD_LIBRARY_PATH="${HDF5_LIB}:${LD_LIBRARY_PATH:-}"

export TMPDIR="$PROJECT_DIR/tmp/foxj1a_cilia_inspection/${JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT

export PROJECT_DIR HERE RMD_FILE

echo "host=$(hostname) start=$(date)"

"$RSCRIPT" -e '
  rmarkdown::render(
    input = Sys.getenv("RMD_FILE"),
    output_file = "report.html",
    output_dir = Sys.getenv("HERE"),
    intermediates_dir = Sys.getenv("TMPDIR"),
    knit_root_dir = Sys.getenv("PROJECT_DIR"),
    envir = new.env(),
    quiet = FALSE
  )
'

echo "done=$(date)"
