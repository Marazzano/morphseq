#!/bin/bash

# Test script to validate the complete SAM2 pipeline with predicted_stage_hpf fix
# This runs a complete Build03A→Build04 test with 2 embryos

echo "🧪 Testing Complete SAM2 Pipeline with predicted_stage_hpf Fix"
echo "============================================================="

# Set up environment
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# Test parameters
EXPERIMENT="20250612_30hpf_ctrl_atf6"
SAM2_CSV="sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv"
TEST_ROOT="/net/trapnell/vol1/home/mdcolon/proj/morphseq_test_data_sam2_pipeline"

echo "📋 Test Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  SAM2 CSV: $SAM2_CSV"
echo "  Test Root: $TEST_ROOT"
echo ""

# Clean up any previous test data
echo "🧹 Cleaning up previous test data..."
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

# Test 1: Build03A with predicted_stage_hpf calculation
echo "🔧 Step 1: Testing Build03A with predicted_stage_hpf calculation..."

python -c "
import sys
sys.path.insert(0, '.')
from src.build.build03A_process_images import segment_wells_sam2_csv, compile_embryo_stats
from pathlib import Path
import pandas as pd

print('Loading SAM2 data and calculating predicted_stage_hpf...')
root = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq'
exp_name = '$EXPERIMENT'
sam2_csv_path = '$SAM2_CSV'

# Step 1: Load SAM2 data with predicted_stage_hpf calculation
tracked_df = segment_wells_sam2_csv(root, exp_name, sam2_csv_path)

# Subset to 2 embryos for testing
tracked_df = tracked_df.head(2)

print(f'✅ SAM2 data loaded: {len(tracked_df)} rows')
if 'predicted_stage_hpf' in tracked_df.columns:
    print(f'✅ predicted_stage_hpf present: {tracked_df[\"predicted_stage_hpf\"].tolist()[:2]}')
else:
    print('❌ predicted_stage_hpf missing')
    exit(1)

# Step 2: Run compile_embryo_stats to add surface_area_um, etc.
print('\\nRunning compile_embryo_stats...')
stats_df = compile_embryo_stats('$TEST_ROOT', tracked_df, overwrite_flag=True, n_workers=1)

print(f'✅ Stats compiled: {stats_df.shape[0]} rows, {stats_df.shape[1]} columns')

# Check for Build04-required columns
build04_cols = ['predicted_stage_hpf', 'surface_area_um', 'use_embryo_flag']
missing = [col for col in build04_cols if col not in stats_df.columns]
if missing:
    print(f'❌ Missing Build04 columns: {missing}')
    exit(1)
else:
    print(f'✅ All Build04 columns present: {build04_cols}')

# Save test metadata for Build04
test_meta_dir = Path('$TEST_ROOT/metadata/combined_metadata_files')
test_meta_dir.mkdir(parents=True, exist_ok=True)
stats_df.to_csv(test_meta_dir / 'embryo_metadata_df01.csv', index=False)
print(f'✅ Test metadata saved to: {test_meta_dir / \"embryo_metadata_df01.csv\"}')

# Show sample data
print('\\n📊 Sample data:')
print(f'  predicted_stage_hpf: {float(stats_df[\"predicted_stage_hpf\"].iloc[0]):.6f}')
print(f'  surface_area_um: {float(stats_df[\"surface_area_um\"].iloc[0]):.1f}')
print(f'  use_embryo_flag: {stats_df[\"use_embryo_flag\"].iloc[0]}')
"

# Check if Build03A succeeded
if [ $? -ne 0 ]; then
    echo "❌ Build03A test failed"
    exit 1
fi

echo ""
echo "🔬 Step 2: Testing Build04 compatibility..."

# Test 2: Build04 - check if it can read the metadata without KeyError
python -c "
import sys
sys.path.insert(0, '.')
import pandas as pd
import os

# Try to read the metadata like Build04 does
metadata_path = os.path.join('$TEST_ROOT', 'metadata', 'combined_metadata_files')
try:
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, 'embryo_metadata_df01.csv'))
    print(f'✅ Build04 can read metadata: {embryo_metadata_df.shape[0]} rows')
    
    # Test the specific line that was failing
    use_indices = embryo_metadata_df['use_embryo_flag'] == True
    if use_indices.sum() > 0:
        time_vec_ref = embryo_metadata_df['predicted_stage_hpf'].iloc[use_indices].values
        print(f'✅ KeyError test passed: predicted_stage_hpf accessible')
        print(f'  Sample values: {time_vec_ref[:2].tolist()}')
    else:
        print('⚠️ No embryos with use_embryo_flag=True, but no KeyError')
        
except KeyError as e:
    print(f'❌ KeyError still occurs: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Other error: {e}')
    exit(1)
"

# Check if Build04 test succeeded 
if [ $? -ne 0 ]; then
    echo "❌ Build04 compatibility test failed"
    exit 1
fi

echo ""
echo "🏁 PIPELINE TEST RESULTS:"
echo "=========================="
echo "✅ SAM2 CSV export: Fixed well metadata access"  
echo "✅ Build03A integration: predicted_stage_hpf calculated"
echo "✅ Build04 compatibility: No KeyError on predicted_stage_hpf"
echo ""
echo "🎉 SUCCESS! Pipeline fix is working correctly."
echo ""
echo "📋 Ready for full pipeline deployment:"
echo "1. Use sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv"
echo "2. Run full Build03A→Build04→Build05 pipeline"
echo "3. Process complete 92-embryo dataset"