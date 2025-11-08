#!/bin/bash
# Test Phase 1 Pipeline for 20250912 YX1 Dataset

echo "========================================="
echo "Phase 1 Test: 20250912 YX1 Dataset"
echo "========================================="

# Change to pipeline directory
cd src/data_pipeline/pipeline_orchestrator

echo ""
echo "Step 1: Dry run to check DAG"
echo "-----------------------------------------"
snakemake -n phase1_complete

echo ""
echo "Step 2: Run Phase 1 (if dry run succeeded)"
echo "-----------------------------------------"
read -p "Continue with actual run? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    snakemake --cores 1 phase1_complete
fi

echo ""
echo "Step 3: Check outputs"
echo "-----------------------------------------"
cd ../../..
ls -lh data_pipeline_output/experiment_metadata/20250912/

echo ""
echo "Done!"
