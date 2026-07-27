#!/bin/bash
set -euo pipefail

AUDIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python"

exec "${PYTHON}" "${AUDIT_DIR}/refresh_pipeline_audit.py" "$@"
