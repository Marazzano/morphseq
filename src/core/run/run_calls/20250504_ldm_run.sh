#!/usr/bin/env bash
set -euo pipefail

echo >&2 "Retired: the referenced src.run.training_ldm_cluster entrypoint and its Hydra config no longer exist."
echo >&2 "This script is retained as provenance only; use src.core.run.training with a supported config."
exit 2
