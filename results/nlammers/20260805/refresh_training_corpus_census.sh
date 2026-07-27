#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CENSUS_PYTHON_BIN="${CENSUS_PYTHON_BIN:-/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python}"

exec "${CENSUS_PYTHON_BIN}" -u "${SCRIPT_DIR}/build_training_corpus_census.py" "$@"
