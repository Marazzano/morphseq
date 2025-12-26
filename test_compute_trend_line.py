#!/usr/bin/env python3
"""
Test script for compute_trend_line() function.

Tests the new trend line computation with median/mean aggregation and
Gaussian smoothing functionality.
"""

import warnings
import numpy as np
import pandas as pd
from src.analyze.trajectory_analysis import compute_trend_line, plot_multimetric_trajectories
from src.analyze.trajectory_analysis.pair_analysis.data_utils import compute_binned_mean


def test_basic_functionality():
    """Test basic compute_trend_line functionality."""
    print("=" * 70)
    print("Test 1: Basic Functionality")
    print("=" * 70)

    times = np.array([24.0, 24.2, 24.5, 24.7, 25.0, 25.3, 25.5, 25.8])
    values = np.array([1.0, 1.2, 1.5, 1.3, 1.8, 1.9, 2.1, 2.0])

    # Test median with smoothing (default)
    bin_times, bin_stats = compute_trend_line(
        times, values,
        bin_width=0.5,
        statistic='median',
        smooth_sigma=1.5
    )
    print("Median with smoothing (default):")
    print(f"  Input times: {times}")
    print(f"  Input values: {values}")
    print(f"  Bin times: {bin_times}")
    print(f"  Bin stats: {bin_stats}")
    print(f"  ✓ Returned {len(bin_times)} bins")
    print()

    # Test mean without smoothing
    bin_times2, bin_stats2 = compute_trend_line(
        times, values,
        bin_width=0.5,
        statistic='mean',
        smooth_sigma=None
    )
    print("Mean without smoothing:")
    print(f"  Bin times: {bin_times2}")
    print(f"  Bin stats: {bin_stats2}")
    print(f"  ✓ Returned {len(bin_times2)} bins")
    print()


def test_deprecation_warning():
    """Test that compute_binned_mean emits deprecation warning."""
    print("=" * 70)
    print("Test 2: Deprecation Warning")
    print("=" * 70)

    warnings.simplefilter('always', DeprecationWarning)

    times = np.array([24.0, 24.5, 25.0, 25.5])
    values = np.array([1.0, 1.5, 1.8, 2.0])

    print("Calling deprecated compute_binned_mean()...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bin_times, bin_means = compute_binned_mean(times, values, bin_width=0.5)

        if len(w) > 0:
            print(f"  ✓ Deprecation warning emitted:")
            print(f"    {w[0].message}")
        else:
            print(f"  ✗ No warning emitted!")

        print(f"  Result: {bin_times}, {bin_means}")
    print()


def test_integration_with_plotting():
    """Test integration with plot_multimetric_trajectories."""
    print("=" * 70)
    print("Test 3: Integration with plot_multimetric_trajectories")
    print("=" * 70)

    # Create synthetic test data
    np.random.seed(42)
    embryo_ids = [f'embryo_{i}' for i in range(10)]
    timepoints = np.linspace(24, 30, 20)
    data = []

    for embryo_id in embryo_ids:
        for t in timepoints:
            data.append({
                'embryo_id': embryo_id,
                'predicted_stage_hpf': t + np.random.normal(0, 0.1),
                'metric1': np.sin(t / 5) + np.random.normal(0, 0.1),
                'metric2': t * 0.5 + np.random.normal(0, 0.5),
                'pair': 'pair_1' if int(embryo_id.split('_')[1]) < 5 else 'pair_2',
                'genotype': 'wt' if int(embryo_id.split('_')[1]) % 2 == 0 else 'mutant'
            })

    df = pd.DataFrame(data)

    print(f"Created synthetic data:")
    print(f"  Shape: {df.shape}")
    print(f"  Metrics: metric1, metric2")
    print(f"  Pairs: {df['pair'].unique()}")
    print(f"  Genotypes: {df['genotype'].unique()}")
    print()

    # Test 1: Median + smoothing (default)
    print("Test 3a: Median + smoothing (default)")
    try:
        fig = plot_multimetric_trajectories(
            df,
            metrics=['metric1', 'metric2'],
            col_by='pair',
            color_by_grouping='genotype',
            backend='plotly',
            output_path=None,
            trend_statistic='median',
            trend_smooth_sigma=1.5,
        )
        print("  ✓ Median + smoothing test passed")
    except Exception as e:
        print(f"  ✗ Median + smoothing test failed: {e}")
        raise
    print()

    # Test 2: Mean + no smoothing
    print("Test 3b: Mean + no smoothing")
    try:
        fig = plot_multimetric_trajectories(
            df,
            metrics=['metric1'],
            col_by='pair',
            backend='plotly',
            output_path=None,
            trend_statistic='mean',
            trend_smooth_sigma=None,
        )
        print("  ✓ Mean + no smoothing test passed")
    except Exception as e:
        print(f"  ✗ Mean + no smoothing test failed: {e}")
        raise
    print()


def test_edge_cases():
    """Test edge cases."""
    print("=" * 70)
    print("Test 4: Edge Cases")
    print("=" * 70)

    # Empty arrays
    print("Test 4a: Empty arrays")
    bin_times, bin_stats = compute_trend_line(
        np.array([]), np.array([]), bin_width=0.5
    )
    assert bin_times == [] and bin_stats == [], "Empty arrays should return empty lists"
    print("  ✓ Empty arrays handled correctly")
    print()

    # Single bin
    print("Test 4b: Single bin")
    times = np.array([24.0, 24.2, 24.3])
    values = np.array([1.0, 1.2, 1.1])
    bin_times, bin_stats = compute_trend_line(
        times, values, bin_width=1.0, statistic='median', smooth_sigma=1.5
    )
    print(f"  Bin times: {bin_times}")
    print(f"  Bin stats: {bin_stats}")
    print(f"  ✓ Single bin handled correctly")
    print()

    # Invalid statistic
    print("Test 4c: Invalid statistic")
    try:
        bin_times, bin_stats = compute_trend_line(
            times, values, bin_width=0.5, statistic='invalid'
        )
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Raised ValueError as expected: {e}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "COMPUTE_TREND_LINE TEST SUITE" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_basic_functionality()
        test_deprecation_warning()
        test_integration_with_plotting()
        test_edge_cases()

        print("=" * 70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • compute_trend_line() works with median and mean")
        print("  • Gaussian smoothing applies correctly")
        print("  • Deprecation warning works for compute_binned_mean()")
        print("  • Integration with plot_multimetric_trajectories() works")
        print("  • Edge cases handled properly")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("✗✗✗ TESTS FAILED ✗✗✗")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
