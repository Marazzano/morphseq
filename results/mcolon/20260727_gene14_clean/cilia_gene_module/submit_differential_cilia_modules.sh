#!/usr/bin/env bash

# Submit every unfinished direct perturbation contrast in contrast_plan.R.
# Existing result tables are skipped, so this command is safe to run again.

set -euo pipefail

PROJECT_DIR=/net/trapnell/vol1/home/mdcolon/proj/morphseq
MODULE_DIR="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/cilia_gene_module"
CONTRAST_PLAN="$PROJECT_DIR/results/mcolon/20260727_gene14_clean/abundance_dact/contrast_plan.R"
RESULT_DIR="$MODULE_DIR/output/differential_module_results"
JOB_SCRIPT="$MODULE_DIR/qsub_differential_cilia_modules.sh"
RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript

mkdir -p "$RESULT_DIR"

# The R contrast plan remains the single source of truth for contrast names.
mapfile -t CONTRASTS < <(
  "$RSCRIPT" -e '
    suppressPackageStartupMessages(library(dplyr))
    source(commandArgs(trailingOnly = TRUE)[[1]])
    cat(perturbation_contrast_plan$contrast_name, sep = "\n")
  ' "$CONTRAST_PLAN"
)

n_submitted=0
n_complete=0

for contrast_name in "${CONTRASTS[@]}"; do
  result_file="$RESULT_DIR/${contrast_name}.tsv"

  if [[ -s "$result_file" ]]; then
    echo "Already complete: $contrast_name"
    n_complete=$((n_complete + 1))
    continue
  fi

  job_id=$(
    qsub -terse \
      -v "CILIA_CONTRAST=$contrast_name" \
      "$JOB_SCRIPT"
  )

  echo "Submitted $contrast_name: $job_id"
  n_submitted=$((n_submitted + 1))
done

echo "Submitted: $n_submitted"
echo "Already complete: $n_complete"
