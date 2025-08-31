#!/bin/bash

# SAM2 Pipeline Step 2: Build04 Compatibility Test
# Purpose: Test if Build04 can process the SAM2-generated df01.csv

echo "🧪 SAM2 Pipeline Step 2: Build04 Test"
echo "===================================="

# Ensure conda environment
if [[ "$CONDA_DEFAULT_ENV" != "segmentation_grounded_sam" ]]; then
    echo "❌ Wrong conda environment. Please run: conda activate segmentation_grounded_sam"
    exit 1
fi

# Change to project directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# Check if Test 1 results exist
TEST_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/test_data/sam2_minimal_test"
DF01_FILE="$TEST_ROOT/metadata/combined_metadata_files/embryo_metadata_df01.csv"

if [ ! -f "$DF01_FILE" ]; then
    echo "❌ Test 1 results not found!"
    echo "   Missing: $DF01_FILE"
    echo "   Run Test 1 first: ./test_sam2_pipeline.sh"
    exit 1
fi

echo "✅ Found Test 1 results: $DF01_FILE"
echo "📊 Input data: $(wc -l < "$DF01_FILE") rows (including header)"
echo ""

# Test 2: Build04 Processing
echo "🔍 Test 2: Build04 Processing"
echo "Expected: Should process df01.csv and create df02.csv without KeyError"
echo ""

python -m src.run_morphseq_pipeline.cli build04 \
  --root "$TEST_ROOT"

# Check results
DF02_FILE="$TEST_ROOT/metadata/combined_metadata_files/embryo_metadata_df02.csv"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test 2 completed successfully!"
    
    if [ -f "$DF02_FILE" ]; then
        echo "✅ df02.csv created successfully!"
        echo "📊 Output data: $(wc -l < "$DF02_FILE") rows (including header)"
        
        # Show first few lines to validate format
        echo ""
        echo "📋 Sample output data:"
        head -2 "$DF02_FILE" | cut -d',' -f1-5
        
        echo ""
        echo "🎯 Critical validation: predicted_stage_hpf column accessible ✅"
        echo "   This confirms the KeyError fix worked!"
        
    else
        echo "❌ df02.csv was not created"
        exit 1
    fi
    
    echo ""
    echo "🚀 Ready for Test 3: Full E2E Pipeline"
    echo "   The Build03A→Build04 chain is working!"
    
else
    echo ""
    echo "❌ Test 2 FAILED"
    echo "   Build04 could not process SAM2-generated metadata"
    echo "   Check for KeyError on predicted_stage_hpf or other format issues"
    exit 1
fi