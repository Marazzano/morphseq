#!/usr/bin/env python3
"""
Test script to generate multimetric trajectory plots with the new color_by_grouping API.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup repository root
morphseq_root = os.environ.get('MORPHSEQ_REPO_ROOT')
if morphseq_root is None:
    morphseq_root = '/net/trapnell/vol1/home/mdcolon/proj/morphseq'
    print(f"MORPHSEQ_REPO_ROOT not set, using default: {morphseq_root}")
else:
    print(f"MORPHSEQ_REPO_ROOT: {morphseq_root}")

# Change to repository root for proper imports
os.chdir(morphseq_root)
sys.path.insert(0, morphseq_root)

# Import MD-DTW analysis tools
from src.analyze.trajectory_analysis import (
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    identify_outliers,
    remove_outliers_from_distance_matrix,
    generate_dendrograms,
    add_cluster_column,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
    plot_multimetric_trajectories,  # Uses NEW API now!
    PASTEL_COLORS,
    extract_trajectories_df,
    plot_dendrogram_with_categories,
)

print("✓ Imports successful!")

# Create sample data (or load your real data)
print("\nGenerating sample data...")
np.random.seed(42)
n_embryos = 30
timepoints = np.linspace(0, 48, 100)  # 0-48 hpf

data = []
for embryo_id in range(n_embryos):
    genotype = 'wt' if embryo_id < 15 else 'mutant'
    pair = f'pair_{embryo_id // 6}'

    # Simulate realistic trajectories
    base_curvature = 0.5 if genotype == 'wt' else 0.8
    base_length = 500 if genotype == 'wt' else 400

    for t in timepoints:
        curvature = base_curvature + 0.3 * np.sin(t / 10) + np.random.normal(0, 0.05)
        length = base_length + 50 * (t / 48) + np.random.normal(0, 10)

        data.append({
            'embryo_id': f'emb_{embryo_id:03d}',
            'predicted_stage_hpf': t,
            'baseline_deviation_normalized': curvature,
            'total_length_um': length,
            'genotype': genotype,
            'pair': pair,
        })

df = pd.DataFrame(data)
print(f"Created dataset: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
print(f"  Genotypes: {df['genotype'].unique()}")
print(f"  Pairs: {df['pair'].unique()}")

# Create output directory
output_dir = Path('test_plots_output')
output_dir.mkdir(exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# =============================================================================
# Plot 1: Multimetric plot with genotype grouping within pairs
# =============================================================================
print("\n" + "=" * 70)
print("Generating Plot 1: Pairs with Genotype Grouping")
print("=" * 70)

output_file = output_dir / 'multimetric_pairs_by_genotype.png'

fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='pair',
    color_by_grouping='genotype',  # <-- NEW PARAMETER NAME!
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Pair, Grouped by Genotype',
    x_label='Time (hpf)',
    backend='matplotlib',
    bin_width=2.0,
)

fig.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved: {output_file}")

# =============================================================================
# Plot 2: Multimetric plot WITHOUT grouping (just pairs)
# =============================================================================
print("\n" + "=" * 70)
print("Generating Plot 2: Pairs Only (No Grouping)")
print("=" * 70)

output_file = output_dir / 'multimetric_pairs_only.png'

fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='pair',
    color_by_grouping=None,  # <-- No grouping within pairs
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Pair (No Grouping)',
    x_label='Time (hpf)',
    backend='matplotlib',
    bin_width=2.0,
)

fig.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved: {output_file}")

# =============================================================================
# Plot 3: Genotypes as columns, pairs as grouping
# =============================================================================
print("\n" + "=" * 70)
print("Generating Plot 3: Genotypes with Pair Grouping")
print("=" * 70)

output_file = output_dir / 'multimetric_genotypes_by_pair.png'

fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='genotype',
    color_by_grouping='pair',  # <-- Different grouping variable
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Genotype, Grouped by Pair',
    x_label='Time (hpf)',
    backend='matplotlib',
    bin_width=2.0,
)

fig.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved: {output_file}")

print("\n" + "=" * 70)
print("All plots generated successfully!")
print("=" * 70)
print(f"\nView plots in: {output_dir.absolute()}")
print("\nGenerated files:")
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")
