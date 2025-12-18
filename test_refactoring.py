#!/usr/bin/env python3
"""
Quick test to verify the permutation testing refactoring works.
Tests both new API and backwards compatibility.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

print("=" * 70)
print("Testing Permutation Testing Refactoring")
print("=" * 70)

# Test 1: Import new utilities
print("\n1. Testing new permutation_utils imports...")
try:
    from analyze.difference_detection.permutation_utils import (
        compute_pvalue,
        pool_shuffle,
        label_shuffle,
        PermutationResult
    )
    print("   ✓ permutation_utils imports successful")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Import new statistics
print("\n2. Testing new statistics imports...")
try:
    from analyze.difference_detection.statistics import (
        compute_energy_distance,
        compute_mmd,
        compute_mean_distance
    )
    print("   ✓ statistics imports successful")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Import new distribution_test
print("\n3. Testing new distribution_test imports...")
try:
    from analyze.difference_detection.distribution_test import (
        permutation_test_energy,
        permutation_test_mmd,
        permutation_test_distribution
    )
    print("   ✓ distribution_test imports successful")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Test new API from package level
print("\n4. Testing package-level imports (new API)...")
try:
    from analyze.difference_detection import (
        compute_pvalue,
        PermutationResult,
        compute_energy_distance,
        permutation_test_energy
    )
    print("   ✓ Package-level imports successful")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Test backwards compatibility (old API)
print("\n5. Testing backwards compatibility (old classification API)...")
try:
    from analyze.difference_detection.classification import (
        predictive_signal_test,
        compute_embryo_penetrance
    )
    print("   ✓ Backwards compatible imports work")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Test compute_pvalue function
print("\n6. Testing compute_pvalue function...")
try:
    observed = 0.75
    null_dist = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    pval = compute_pvalue(observed, null_dist, alternative="greater", pseudo_count=True)
    expected_pval = 1.0 / 6.0  # (0 + 1) / (5 + 1)
    assert np.isclose(pval, expected_pval), f"Expected {expected_pval}, got {pval}"
    print(f"   ✓ compute_pvalue works correctly (p={pval:.4f})")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 7: Test PermutationResult class
print("\n7. Testing PermutationResult class...")
try:
    result = PermutationResult(
        statistic_name='energy',
        observed=0.25,
        pvalue=0.03,
        null_distribution=np.array([0.1, 0.12, 0.11, 0.13, 0.09]),
        n_permutations=5
    )
    assert result.is_significant(alpha=0.05), "Result should be significant"
    assert not result.is_significant(alpha=0.01), "Result should not be significant at 0.01"
    result_dict = result.to_dict()
    assert 'energy' in result_dict
    assert 'pvalue' in result_dict
    print(f"   ✓ PermutationResult works correctly: {result}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 8: Test energy distance computation
print("\n8. Testing energy distance computation...")
try:
    X1 = np.random.randn(20, 5)
    X2 = np.random.randn(20, 5) + 0.5
    energy = compute_energy_distance(X1, X2)
    assert energy >= 0, "Energy distance should be non-negative"
    assert energy > 0, "Energy distance should be positive for different distributions"
    print(f"   ✓ Energy distance computed: {energy:.4f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 9: Test MMD computation
print("\n9. Testing MMD computation...")
try:
    X1 = np.random.randn(20, 5)
    X2 = np.random.randn(20, 5) + 0.3
    mmd = compute_mmd(X1, X2)
    assert mmd >= 0, "MMD should be non-negative"
    print(f"   ✓ MMD computed: {mmd:.4f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 10: Test full permutation test
print("\n10. Testing full permutation test...")
try:
    rng = np.random.default_rng(42)
    X1 = rng.normal(0, 1, (30, 5))
    X2 = rng.normal(0.5, 1, (30, 5))
    result = permutation_test_energy(X1, X2, n_permutations=50, random_state=42)
    assert isinstance(result, PermutationResult)
    assert result.observed > 0
    assert 0 <= result.pvalue <= 1
    print(f"   ✓ Full permutation test works: {result}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nRefactoring successful! Key improvements:")
print("  - DRY: Unified permutation test logic in permutation_utils")
print("  - KISS: Simple, flat module structure")
print("  - Extensible: Easy to add new test statistics")
print("  - Backwards compatible: Old imports still work")
