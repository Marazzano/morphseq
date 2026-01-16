#!/usr/bin/env python3
"""
Phase 2 Import Tests

Tests that all Phase 2 subpackages (distance, utilities, io) import correctly.
"""

import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq/src')

def test_distance_imports():
    """Test distance subpackage imports"""
    print("Testing distance imports...")
    from analyze.trajectory_analysis.distance import (
        compute_dtw_distance,
        compute_dtw_distance_matrix,
        prepare_multivariate_array,
        compute_md_dtw_distance_matrix,
        compute_trajectory_distances,
        dba,
    )
    print("✓ Distance imports OK")
    return True

def test_utilities_imports():
    """Test utilities subpackage imports"""
    print("Testing utilities imports...")
    from analyze.trajectory_analysis.utilities import (
        extract_trajectories_df,
        interpolate_to_common_grid_df,
        compute_trend_line,
        fit_pca_on_embeddings,
        transform_embeddings_to_pca,
        test_anticorrelation,
    )
    print("✓ Utilities imports OK")
    return True

def test_io_imports():
    """Test io subpackage imports"""
    print("Testing io imports...")
    from analyze.trajectory_analysis.io import (
        load_experiment_dataframe,
        extract_trajectory_dataframe,
        compute_dtw_distance_from_df,
        load_phenotype_file,
        save_phenotype_file,
    )
    print("✓ I/O imports OK")
    return True

def test_cross_imports():
    """Test that io correctly imports from distance"""
    print("Testing cross-subpackage imports...")
    # This import chain: io.data_loading -> distance.dtw_distance
    from analyze.trajectory_analysis.io.data_loading import compute_dtw_distance_from_df
    print("✓ Cross-subpackage imports OK")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2 Import Tests")
    print("=" * 60)

    results = []
    results.append(("distance", test_distance_imports()))
    results.append(("utilities", test_utilities_imports()))
    results.append(("io", test_io_imports()))
    results.append(("cross-imports", test_cross_imports()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All Phase 2 tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
