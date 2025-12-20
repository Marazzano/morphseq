#!/usr/bin/env python3
"""
Test script to verify the color_by_grouping refactoring works correctly.
"""

import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq/src')

from analyze.trajectory_analysis.facetted_plotting_refactored import (
    plot_trajectories_faceted,
    plot_multimetric_trajectories,
)

print("=" * 70)
print("Testing color_by_grouping refactoring")
print("=" * 70)

# Create minimal test data
np.random.seed(42)
n_embryos = 20
timepoints = np.linspace(0, 10, 50)

data = []
for embryo_id in range(n_embryos):
    genotype = 'wt' if embryo_id < 10 else 'mutant'
    cluster = 'cluster_A' if embryo_id % 2 == 0 else 'cluster_B'
    pair = f'pair_{embryo_id // 4}'

    for t in timepoints:
        value = np.sin(t + embryo_id * 0.1) + np.random.normal(0, 0.1)
        data.append({
            'embryo_id': f'emb_{embryo_id:03d}',
            'predicted_stage_hpf': t,
            'baseline_deviation_normalized': value,
            'genotype': genotype,
            'cluster': cluster,
            'pair': pair,
        })

df = pd.DataFrame(data)

print(f"\nCreated test dataset: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
print(f"  Genotypes: {df['genotype'].unique()}")
print(f"  Clusters: {df['cluster'].unique()}")
print(f"  Pairs: {df['pair'].unique()}")

# Test 1: Basic plot with col_by only
print("\n" + "-" * 70)
print("Test 1: Basic plot (col_by only, no grouping)")
print("-" * 70)
try:
    fig = plot_trajectories_faceted(
        df,
        col_by='genotype',
        color_by_grouping=None,
        backend='matplotlib',
        title='Test 1: Genotypes (no grouping)',
    )
    print("✓ Test 1 PASSED")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Plot with color_by_grouping
print("\n" + "-" * 70)
print("Test 2: Plot with color_by_grouping")
print("-" * 70)
try:
    fig = plot_trajectories_faceted(
        df,
        col_by='pair',
        color_by_grouping='genotype',
        backend='matplotlib',
        title='Test 2: Pairs with Genotype Grouping',
    )
    print("✓ Test 2 PASSED")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multimetric plot
print("\n" + "-" * 70)
print("Test 3: Multimetric plot with color_by_grouping")
print("-" * 70)

# Add second metric
df['metric2'] = df['baseline_deviation_normalized'] * 2 + 1

try:
    fig = plot_multimetric_trajectories(
        df,
        metrics=['baseline_deviation_normalized', 'metric2'],
        col_by='cluster',
        color_by_grouping='genotype',
        backend='matplotlib',
        title='Test 3: Multi-metric with Genotype Grouping',
    )
    print("✓ Test 3 PASSED")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Ensure old parameters are removed
print("\n" + "-" * 70)
print("Test 4: Verify old parameters (overlay, color_by) are removed")
print("-" * 70)
try:
    # This should fail
    fig = plot_trajectories_faceted(
        df,
        col_by='genotype',
        overlay='cluster',  # This parameter no longer exists
        backend='matplotlib',
    )
    print("✗ Test 4 FAILED: overlay parameter should have been removed!")
except TypeError as e:
    if 'overlay' in str(e):
        print("✓ Test 4 PASSED: overlay parameter correctly removed")
    else:
        print(f"✗ Test 4 FAILED with unexpected error: {e}")
except Exception as e:
    print(f"✗ Test 4 FAILED with unexpected error: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
