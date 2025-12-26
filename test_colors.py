"""Quick test to verify cluster colors are working correctly."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.analyze.trajectory_analysis import plot_multimetric_trajectories

# Create minimal test data
np.random.seed(42)
n_embryos = 3  # Just a few embryos per cluster for clarity
n_timepoints = 20
time_grid = np.linspace(20, 40, n_timepoints)

data_rows = []
for cluster_id in [0, 1, 2]:
    for embryo_idx in range(n_embryos):
        embryo_id = f"c{cluster_id}_e{embryo_idx}"

        # Different patterns per cluster
        curvature = cluster_id + np.random.normal(0, 0.1, n_timepoints)
        length = 100 * (cluster_id + 1) + np.random.normal(0, 5, n_timepoints)

        for t_idx, t in enumerate(time_grid):
            data_rows.append({
                'embryo_id': embryo_id,
                'predicted_stage_hpf': t,
                'baseline_deviation_normalized': curvature[t_idx],
                'total_length_um': length[t_idx],
                'cluster': cluster_id,
                'genotype': f'wildtype' if cluster_id == 0 else f'homozygous' if cluster_id == 1 else 'heterozygous',
            })

df = pd.DataFrame(data_rows)

print("Test data:")
print(f"  Cluster 0: {df[df['cluster']==0]['embryo_id'].nunique()} embryos")
print(f"  Cluster 1: {df[df['cluster']==1]['embryo_id'].nunique()} embryos")
print(f"  Cluster 2: {df[df['cluster']==2]['embryo_id'].nunique()} embryos")
print()

# Test coloring by cluster (should show 3 different colors: 0, 1, 2)
print("Creating plot colored by cluster (0, 1, 2 should each have distinct colors)...")
fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    cluster_col='cluster',
    color_by='cluster',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature',
        'total_length_um': 'Length (μm)'
    },
    backend='plotly',
    output_path='test_multimetric_output/color_test_by_cluster.html',
    title='Color Test: By Cluster (0=?, 1=?, 2=?)'
)
print("✓ Saved to test_multimetric_output/color_test_by_cluster.html")
print()

# Test coloring by genotype (should show wildtype=green, homo=red, het=orange)
print("Creating plot colored by genotype (should show wildtype, homozygous, heterozygous)...")
fig2 = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    cluster_col='cluster',
    color_by='genotype',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature',
        'total_length_um': 'Length (μm)'
    },
    backend='plotly',
    output_path='test_multimetric_output/color_test_by_genotype.html',
    title='Color Test: By Genotype (WT=Green, Homo=Red, Het=Orange)'
)
print("✓ Saved to test_multimetric_output/color_test_by_genotype.html")
print()

print("Open the HTML files in a browser to verify:")
print("  1. color_test_by_cluster.html - All 3 clusters should have different colors")
print("  2. color_test_by_genotype.html - WT=green, Homo=red, Het=orange")
