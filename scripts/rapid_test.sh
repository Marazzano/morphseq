#!/bin/bash
# Quick pipeline test on subset data

EXPERIMENT=${1:-test_yx1_001}
TARGET_RULE=${2:-all}

echo "Testing: $EXPERIMENT → $TARGET_RULE"

snakemake \
    --config experiments=$EXPERIMENT \
    --cores 2 \
    --printshellcmds \
    $TARGET_RULE

# Auto-validate if hitting checkpoints
case $TARGET_RULE in
    "rule_generate_image_manifest")
        python scripts/validate_phase2_outputs.py $EXPERIMENT
        ;;
    "all")
        python scripts/validate_full_pipeline.py $EXPERIMENT
        ;;
esac
