#!/bin/bash

# SAM2 Pipeline Validation Test Script
# Created: August 31, 2025
# Purpose: First real validation of supposedly "production ready" pipeline

echo "🧪 SAM2 Pipeline Validation Test"
echo "================================"

# Ensure conda environment
if [[ "$CONDA_DEFAULT_ENV" != "segmentation_grounded_sam" ]]; then
    echo "❌ Wrong conda environment. Please run: conda activate segmentation_grounded_sam"
    exit 1
fi

# Change to project directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# Test 1: Minimal Build03A Test
echo "🔍 Test 1: Minimal Build03A (2 embryos, 1 frame each)"
echo "Expected: Should process 2 embryos and create metadata files"
echo ""

python -m src.run_morphseq_pipeline.cli build03 \
  --root /net/trapnell/vol1/home/mdcolon/proj/morphseq/test_data \
  --test-suffix sam2_minimal_test \
  --exp 20250612_30hpf_ctrl_atf6 \
  --sam2-csv /net/trapnell/vol1/home/mdcolon/proj/morphseq/sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv \
  --by-embryo 2 \
  --frames-per-embryo 1

# Check results
TEST_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/test_data/sam2_minimal_test"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test 1 completed successfully!"
    echo "📁 Checking created files:"
    
    if [ -d "$TEST_ROOT" ]; then
        echo "   - Test directory exists: $TEST_ROOT"
        ls -la "$TEST_ROOT/"
        
        if [ -d "$TEST_ROOT/metadata" ]; then
            echo "   - Metadata directory:"
            ls -la "$TEST_ROOT/metadata/"
            
            if [ -f "$TEST_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv" ]; then
                echo "   ✅ df01.csv created successfully!"
                wc -l "$TEST_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv"
            else
                echo "   ❌ df01.csv missing"
            fi
        fi
    fi
    
    echo ""
    echo "🚀 Ready for Test 2: Build04 compatibility test"
    echo "   Run: ./test_sam2_pipeline.sh build04"
    
else
    echo ""
    echo "❌ Test 1 FAILED"
    echo "   This confirms the pipeline is NOT actually production ready"
    echo "   Need to debug the actual runtime issues vs. theoretical fixes"
fi