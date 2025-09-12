#!/usr/bin/env python3
"""
Debug script to investigate SAM2 QC flag values and identify the parsing issue.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

def debug_sam2_qc_flags():
    """Debug what's actually in the sam2_qc_flags field."""
    
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/sam2_expr_files/sam2_metadata_20250622_chem_28C_T00_1425.csv")
    
    print("🔍 Debugging SAM2 QC flag values...")
    print(f"📄 CSV: {csv_path}")
    print()
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    print(f"📊 Total records: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    print()
    
    # Check if sam2_qc_flags column exists
    if 'sam2_qc_flags' not in df.columns:
        print("❌ sam2_qc_flags column not found!")
        return
    
    # Analyze the sam2_qc_flags column
    qc_flags = df['sam2_qc_flags']
    
    print("🔍 SAM2 QC Flags Analysis:")
    print(f"   • Column type: {qc_flags.dtype}")
    print(f"   • Non-null count: {qc_flags.count()}")
    print(f"   • Null count: {qc_flags.isnull().sum()}")
    print()
    
    # Check unique values
    unique_values = qc_flags.unique()
    print(f"🎯 Unique values ({len(unique_values)}):")
    for i, val in enumerate(unique_values):
        count = (qc_flags == val).sum()
        val_repr = repr(val)  # Shows hidden characters
        val_type = type(val).__name__
        print(f"   {i+1}. {val_repr} (type: {val_type}, count: {count})")
    print()
    
    # Test our current logic
    print("🧪 Testing current QC flag detection logic:")
    flagged_count = 0
    sample_flagged = []
    sample_clean = []
    
    for idx, val in enumerate(qc_flags):
        # Current logic from _set_final_use_embryo_flag
        sam2_qc_flags_str = str(val).strip() if pd.notna(val) else ""
        has_sam2_flags = len(sam2_qc_flags_str) > 0
        
        if has_sam2_flags:
            flagged_count += 1
            if len(sample_flagged) < 3:  # Keep first 3 examples
                sample_flagged.append((idx, repr(val), sam2_qc_flags_str))
        else:
            if len(sample_clean) < 3:  # Keep first 3 examples
                sample_clean.append((idx, repr(val), sam2_qc_flags_str))
    
    print(f"   • Current logic flags: {flagged_count}/{len(qc_flags)} records")
    print()
    
    print("🔍 Sample flagged records:")
    for idx, orig, processed in sample_flagged:
        embryo_id = df.iloc[idx]['embryo_id'] if 'embryo_id' in df.columns else f"row_{idx}"
        print(f"   • {embryo_id}: {orig} → '{processed}'")
    print()
    
    print("🔍 Sample clean records:")
    for idx, orig, processed in sample_clean:
        embryo_id = df.iloc[idx]['embryo_id'] if 'embryo_id' in df.columns else f"row_{idx}"
        print(f"   • {embryo_id}: {orig} → '{processed}'")
    print()
    
    # Test improved logic
    print("🧪 Testing improved QC flag detection logic:")
    improved_flagged_count = 0
    
    for val in qc_flags:
        # Improved logic
        if pd.isna(val):
            has_sam2_flags = False
        else:
            sam2_qc_flags_str = str(val).strip()
            has_sam2_flags = sam2_qc_flags_str and sam2_qc_flags_str.lower() not in ["", "nan", "none", "null"]
        
        if has_sam2_flags:
            improved_flagged_count += 1
    
    print(f"   • Improved logic flags: {improved_flagged_count}/{len(qc_flags)} records")
    print()
    
    # Show specific records mentioned by user
    print("🎯 Checking specific records mentioned by user:")
    f04_records = df[df['embryo_id'].str.contains('F04', na=False)]
    f05_records = df[df['embryo_id'].str.contains('F05', na=False)]
    
    if not f04_records.empty:
        f04_qc = f04_records.iloc[0]['sam2_qc_flags']
        print(f"   • F04 QC flag: {repr(f04_qc)} (should be 'MASK_ON_EDGE')")
    
    if not f05_records.empty:
        f05_qc = f05_records.iloc[0]['sam2_qc_flags']
        print(f"   • F05 QC flag: {repr(f05_qc)} (should be empty)")
    
    print()
    print("✅ Debug analysis complete!")

if __name__ == "__main__":
    debug_sam2_qc_flags()