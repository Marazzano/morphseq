#!/usr/bin/env python3
"""
Test script for DLMA integration module.

Run this after installing dependencies to verify the integration works correctly.

Usage:
    python test_dlma_integration.py

Requirements:
    pip install numpy pandas scikit-image

Author: morphseq team
Date: 2025-11-15
"""

import sys
from pathlib import Path

# Test imports
print("=" * 80)
print("DLMA Integration Module Tests")
print("=" * 80)

print("\n[1/5] Testing imports...")
try:
    from zebrafish_morphometric_analysis import (
        compute_mask_area_percentages,
        extract_morphometric_features_batch,
        summarize_morphometric_features,
        get_all_class_names,
        get_organ_names,
        get_phenotype_names,
        DLMA_ALL_CLASSES,
        DLMA_ORGAN_CLASSES,
        DLMA_PHENOTYPE_CLASSES,
    )
    import numpy as np
    import pandas as pd
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease install required packages:")
    print("  pip install numpy pandas scikit-image")
    sys.exit(1)


# Test class definitions
print("\n[2/5] Testing class definitions...")
try:
    organs = get_organ_names()
    phenotypes = get_phenotype_names()
    all_classes = get_all_class_names()

    assert len(organs) == 8, f"Expected 8 organs, got {len(organs)}"
    assert len(phenotypes) == 8, f"Expected 8 phenotypes, got {len(phenotypes)}"
    assert len(all_classes) == 16, f"Expected 16 total classes, got {len(all_classes)}"

    print(f"✓ Class definitions correct")
    print(f"  - Organs: {len(organs)}")
    print(f"  - Phenotypes: {len(phenotypes)}")
    print(f"  - Total: {len(all_classes)}")

except AssertionError as e:
    print(f"✗ Class definition test failed: {e}")
    sys.exit(1)


# Test compute_mask_area_percentages
print("\n[3/5] Testing compute_mask_area_percentages...")
try:
    np.random.seed(42)
    H, W = 256, 256

    # Test case 1: Multiple instances
    n_instances = 3
    masks = (np.random.rand(n_instances, H, W) > 0.85).astype(np.uint8)
    class_ids = np.array([1, 2, 3])  # eye, heart, yolk

    result = compute_mask_area_percentages(masks, class_ids)

    # Check return type
    assert isinstance(result, dict), "Result should be a dictionary"

    # Check required keys
    assert 'total_embryo_area_px' in result, "Missing total_embryo_area_px"
    assert 'eye_area_pct' in result, "Missing eye_area_pct"
    assert 'heart_area_pct' in result, "Missing heart_area_pct"
    assert 'yolk_area_pct' in result, "Missing yolk_area_pct"

    # Check values are reasonable
    assert result['total_embryo_area_px'] > 0, "Total area should be > 0"
    assert 0 <= result['eye_area_pct'] <= 100, "Percentages should be in [0, 100]"

    print(f"✓ Test case 1 passed (multiple instances)")
    print(f"  - Total embryo area: {result['total_embryo_area_px']} pixels")
    print(f"  - Eye area: {result['eye_area_pct']:.2f}%")
    print(f"  - Heart area: {result['heart_area_pct']:.2f}%")
    print(f"  - Yolk area: {result['yolk_area_pct']:.2f}%")

    # Test case 2: With total embryo mask
    total_mask = (np.random.rand(H, W) > 0.7).astype(np.uint8)
    result2 = compute_mask_area_percentages(masks, class_ids, total_mask)

    assert result2['total_embryo_area_px'] == np.sum(total_mask > 0), \
        "Total area should match provided mask"

    print(f"✓ Test case 2 passed (with embryo mask)")

    # Test case 3: Empty mask handling
    empty_masks = np.zeros((1, H, W), dtype=np.uint8)
    result3 = compute_mask_area_percentages(empty_masks, np.array([1]))

    assert result3['total_embryo_area_px'] == 0, "Empty mask should have 0 area"

    print(f"✓ Test case 3 passed (empty mask handling)")

except Exception as e:
    print(f"✗ compute_mask_area_percentages test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test batch extraction (with dummy data)
print("\n[4/5] Testing extract_morphometric_features_batch...")
try:
    # Create dummy tracking DataFrame
    tracking_df = pd.DataFrame({
        'snip_id': ['test_001', 'test_002', 'test_003'],
        'image_id': ['test_001', 'test_002', 'test_003'],
        'micrometers_per_pixel': [1.5, 1.5, 1.5],
    })

    # This will return NaN for missing files, which is expected
    # Just testing the function signature and structure
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        result_df = extract_morphometric_features_batch(
            tracking_df=tracking_df,
            dlma_predictions_dir=tmpdir_path,
            prediction_format='detectron2',
        )

        # Check output structure
        assert isinstance(result_df, pd.DataFrame), "Should return DataFrame"
        assert 'snip_id' in result_df.columns, "Should have snip_id column"
        assert len(result_df) == len(tracking_df), "Should have same length as input"

        # Check for expected columns
        expected_cols = ['eye_area_pct', 'heart_area_pct', 'yolk_area_pct']
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

    print(f"✓ Batch extraction test passed")
    print(f"  - Output shape: {result_df.shape}")
    print(f"  - Columns: {len(result_df.columns)}")

except Exception as e:
    print(f"✗ Batch extraction test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Test summarize_morphometric_features
print("\n[5/5] Testing summarize_morphometric_features...")
try:
    # Create dummy features DataFrame
    np.random.seed(42)
    n_samples = 20

    features_df = pd.DataFrame({
        'snip_id': [f'test_{i:03d}' for i in range(n_samples)],
        'eye_area_pct': np.random.rand(n_samples) * 5,
        'heart_area_pct': np.random.rand(n_samples) * 3,
        'yolk_area_pct': np.random.rand(n_samples) * 20,
        'treatment': ['control'] * 10 + ['treated'] * 10,
    })

    # Test overall summary
    summary1 = summarize_morphometric_features(features_df)
    assert isinstance(summary1, pd.DataFrame), "Should return DataFrame"
    assert 'n_samples' in summary1.columns, "Should have n_samples"
    assert 'eye_mean_pct' in summary1.columns, "Should have mean percentages"

    print(f"✓ Overall summary test passed")
    print(f"  - Summary shape: {summary1.shape}")

    # Test grouped summary
    summary2 = summarize_morphometric_features(features_df, group_by=['treatment'])
    assert len(summary2) == 2, "Should have 2 groups (control + treated)"
    assert 'treatment' in summary2.columns, "Should have grouping column"

    print(f"✓ Grouped summary test passed")
    print(f"  - Groups: {len(summary2)}")

except Exception as e:
    print(f"✗ Summary test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# All tests passed
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nThe DLMA integration module is working correctly.")
print("\nNext steps:")
print("  1. Install DLMA and Detectron2 (see DLMA_INSTALLATION.md)")
print("  2. Download model weights")
print("  3. Run inference on your zebrafish images")
print("  4. Use extract_morphometric_features_batch() in your pipeline")
print("\nSee README_DLMA_INTEGRATION.md for usage examples.")
