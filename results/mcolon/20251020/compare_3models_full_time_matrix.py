#!/usr/bin/env python3
"""
Compare 3 Models (WT, Het, Homo) Full Time Matrix Results

Loads saved results from all three full time matrix models and creates
comprehensive side-by-side comparisons with shared color scales.

Layout: 3 rows (models) × 3 columns (test genotypes)
- Row 1: WT Model predictions on [WT, Het, Homo]
- Row 2: Het Model predictions on [WT, Het, Homo]
- Row 3: Homo Model predictions on [WT, Het, Homo]

Uses LOEO (Leave-One-Embryo-Out) cross-validation when model's training
genotype matches the test genotype (diagonal cells).

Color scales use 5th-95th percentile clipping for improved dynamic range.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib.patches import Rectangle

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

MODEL_DATA_DIRS = {
    'WT': DATA_DIR / 'wt_full_time_matrix',
    'Het': DATA_DIR / 'het_full_time_matrix',
    'Homo': DATA_DIR / 'homo_full_time_matrix'
}

OUTPUT_DATA_DIR = DATA_DIR / '3model_comparison'
OUTPUT_PLOT_DIR = PLOT_DIR / '3model_comparison'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Test genotypes (consistent order)
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

# Model to genotype mapping (for LOEO identification)
MODEL_TRAINING_GENOTYPE = {
    'WT': 'cep290_wildtype',
    'Het': 'cep290_heterozygous',
    'Homo': 'cep290_homozygous'
}

# ============================================================================
# Load Results
# ============================================================================

def load_all_model_results() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load results from all three models.

    Returns
    -------
    dict
        {model_name: {genotype: df}}
    """
    all_results = {}

    for model_name, data_dir in MODEL_DATA_DIRS.items():
        file_path = data_dir / 'full_time_matrix_metrics.csv'

        if not file_path.exists():
            print(f"  WARNING: {file_path} not found")
            all_results[model_name] = {}
            continue

        print(f"  Loading {model_name} model from {file_path}...")
        df = pd.read_csv(file_path)

        # Split by genotype
        model_results = {}
        for genotype in GENOTYPES:
            genotype_df = df[df['genotype'] == genotype].copy()
            if len(genotype_df) > 0:
                model_results[genotype] = genotype_df
                print(f"    {genotype}: {len(genotype_df)} timepoint pairs")

        all_results[model_name] = model_results

    return all_results


# ============================================================================
# Create Comparison Matrices
# ============================================================================

def create_matrices_for_all_models(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
    metric: str = 'mae'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create matrices for all models and genotypes.

    Returns
    -------
    dict
        {model_name: {genotype: matrix_df}}
    """
    # Get union of all start and target times across ALL models
    all_start_times = set()
    all_target_times = set()

    for model_results in all_results.values():
        for genotype_df in model_results.values():
            all_start_times.update(genotype_df['start_time'].unique())
            all_target_times.update(genotype_df['target_time'].unique())

    all_start_times = sorted(all_start_times)
    all_target_times = sorted(all_target_times)

    # Create matrices for each model-genotype combination
    matrices = {}

    for model_name, model_results in all_results.items():
        matrices[model_name] = {}

        for genotype in GENOTYPES:
            if genotype not in model_results:
                continue

            genotype_df = model_results[genotype]

            # Create matrix
            matrix = pd.DataFrame(
                index=all_start_times,
                columns=all_target_times,
                dtype=float
            )

            # Fill matrix
            for _, row in genotype_df.iterrows():
                matrix.loc[row['start_time'], row['target_time']] = row[metric]

            matrices[model_name][genotype] = matrix

    return matrices


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_3model_comparison(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
    metric: str = 'mae',
    percentile_clip: bool = True,
    save_path: Optional[Path] = None
):
    """
    Create 3×3 grid comparing all models on all genotypes.

    Parameters
    ----------
    all_results : dict
        {model_name: {genotype: df}}
    metric : str
        'mae', 'r2', or 'error_std'
    percentile_clip : bool
        If True, use 5th-95th percentile for color scale
    save_path : Path, optional
        Where to save the plot
    """
    models = ['WT', 'Het', 'Homo']
    n_models = len(models)
    n_genotypes = len(GENOTYPES)

    fig, axes = plt.subplots(n_models, n_genotypes, figsize=(6 * n_genotypes, 5 * n_models))

    # Get matrices
    matrices = create_matrices_for_all_models(all_results, metric)

    # Determine shared color scale
    all_values = []
    for model_name in models:
        if model_name not in matrices:
            continue
        for genotype in GENOTYPES:
            if genotype not in matrices[model_name]:
                continue
            matrix = matrices[model_name][genotype]
            all_values.extend(matrix.values.flatten())

    all_values = [x for x in all_values if not np.isnan(x)]

    if len(all_values) == 0:
        print(f"  WARNING: No data for metric {metric}")
        return None

    if percentile_clip:
        vmin = np.percentile(all_values, 5)
        vmax = np.percentile(all_values, 95)
        clip_note = f" (5th-95th percentile: {vmin:.3f}-{vmax:.3f})"
    else:
        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)
        clip_note = ""

    # Handle R² which can be negative
    if metric == 'r2':
        vmin = max(vmin, -1.0)
        vmax = min(vmax, 1.0)

    print(f"\n  {metric.upper()} color scale: {vmin:.3f} - {vmax:.3f}{clip_note}")

    # Create heatmaps
    for model_idx, model_name in enumerate(models):
        for genotype_idx, genotype in enumerate(GENOTYPES):
            ax = axes[model_idx, genotype_idx]

            if model_name not in matrices or genotype not in matrices[model_name]:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            matrix = matrices[model_name][genotype]

            # Create heatmap
            sns.heatmap(
                matrix,
                ax=ax,
                cmap='viridis' if metric != 'r2' else 'RdYlGn',
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                cbar_kws={'label': metric.upper()},
                linewidths=0.5,
                linecolor='white'
            )

            # Labels
            genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()
            model_label = f'{model_name} Model'

            # Add LOEO indicator if this is a diagonal cell
            is_loeo = (MODEL_TRAINING_GENOTYPE[model_name] == genotype)
            loeo_marker = ' (LOEO)' if is_loeo else ''

            ax.set_title(f'{model_label} → {genotype_label}{loeo_marker}', fontsize=11, fontweight='bold')

            # Only show x-labels on bottom row
            if model_idx == n_models - 1:
                ax.set_xlabel('Target Time (hpf)', fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            # Only show y-labels on left column
            if genotype_idx == 0:
                ax.set_ylabel('Start Time (hpf)', fontsize=9)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            # Add border for LOEO cells
            if is_loeo:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

    metric_labels = {
        'mae': 'Mean Absolute Error',
        'r2': 'R² Score',
        'error_std': 'Error Std Deviation'
    }

    title = f'3-Model Comparison: {metric_labels[metric]}'
    if percentile_clip:
        title += '\n(Color scale: 5th-95th percentile clipping for improved dynamic range)'

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    # Add LOEO legend
    fig.text(
        0.5, 0.005,
        'Red borders indicate LOEO (Leave-One-Embryo-Out) cross-validation',
        ha='center',
        fontsize=10,
        style='italic',
        color='red'
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def create_best_model_heatmap(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
    metric: str = 'mae',
    save_path: Optional[Path] = None
):
    """
    Create heatmap showing which model performs best for each genotype.

    For MAE/error_std: Lower is better
    For R²: Higher is better
    """
    models = ['WT', 'Het', 'Homo']
    n_genotypes = len(GENOTYPES)

    fig, axes = plt.subplots(1, n_genotypes, figsize=(6 * n_genotypes, 5))

    if n_genotypes == 1:
        axes = [axes]

    # Get matrices
    matrices = create_matrices_for_all_models(all_results, metric)

    # For each genotype, find which model is best at each cell
    for genotype_idx, genotype in enumerate(GENOTYPES):
        ax = axes[genotype_idx]

        # Collect matrices for this genotype from all models
        genotype_matrices = []
        available_models = []

        for model_name in models:
            if model_name in matrices and genotype in matrices[model_name]:
                genotype_matrices.append(matrices[model_name][genotype])
                available_models.append(model_name)

        if len(genotype_matrices) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Stack matrices and find best model per cell
        stacked = np.stack([m.values for m in genotype_matrices], axis=-1)

        # Initialize with NaN
        best_model_idx = np.full(stacked.shape[:2], np.nan)

        # Only compute argmin/argmax for cells with at least one valid value
        has_data = ~np.all(np.isnan(stacked), axis=-1)

        if metric in ['mae', 'error_std']:
            # Lower is better
            best_model_idx[has_data] = np.nanargmin(stacked[has_data], axis=-1)
        else:  # r2
            # Higher is better
            best_model_idx[has_data] = np.nanargmax(stacked[has_data], axis=-1)

        # Create matrix with model indices
        best_matrix = pd.DataFrame(
            best_model_idx,
            index=genotype_matrices[0].index,
            columns=genotype_matrices[0].columns
        )

        # Plot
        sns.heatmap(
            best_matrix,
            ax=ax,
            cmap='Set2',
            vmin=0,
            vmax=len(available_models) - 1,
            cbar=True,
            cbar_kws={'label': 'Best Model', 'ticks': range(len(available_models))},
            linewidths=0.5,
            linecolor='white'
        )

        # Update colorbar labels
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticklabels(available_models)

        genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()
        ax.set_title(f'{genotype_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Time (hpf)', fontsize=10)
        ax.set_ylabel('Start Time (hpf)', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    metric_labels = {
        'mae': 'Mean Absolute Error',
        'r2': 'R² Score',
        'error_std': 'Error Std Deviation'
    }

    better_text = 'lower is better' if metric in ['mae', 'error_std'] else 'higher is better'

    plt.suptitle(
        f'Best Model per Timepoint: {metric_labels[metric]} ({better_text})',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def create_summary_statistics(
    all_results: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Create summary statistics comparing all models.

    Returns
    -------
    pd.DataFrame
        Summary statistics for each model-genotype combination
    """
    summary_data = []

    for model_name, model_results in all_results.items():
        for genotype in GENOTYPES:
            if genotype not in model_results:
                continue

            df = model_results[genotype]

            for metric in ['mae', 'r2', 'error_std']:
                is_loeo = (MODEL_TRAINING_GENOTYPE[model_name] == genotype)

                summary_data.append({
                    'model': model_name,
                    'test_genotype': genotype,
                    'metric': metric,
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'median': df[metric].median(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'n_timepoints': len(df),
                    'uses_loeo': is_loeo
                })

    return pd.DataFrame(summary_data)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("COMPARING 3 MODELS (WT, HET, HOMO) FULL TIME MATRIX RESULTS")
    print("="*80)

    # ------------------------------------------------------------------------
    # Step 1: Load results
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING SAVED RESULTS FROM ALL MODELS")
    print("="*80)

    all_results = load_all_model_results()

    if sum(len(r) for r in all_results.values()) == 0:
        print("\nERROR: Missing results files. Make sure you've run all three:")
        print("  - run_wt_full_time_matrix.py")
        print("  - run_het_full_time_matrix.py")
        print("  - run_homo_full_time_matrix.py")
        return

    # ------------------------------------------------------------------------
    # Step 2: Create summary statistics
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: COMPUTING SUMMARY STATISTICS")
    print("="*80)

    summary_df = create_summary_statistics(all_results)

    summary_file = OUTPUT_DATA_DIR / '3model_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")

    # Print key comparisons
    print("\nMean MAE by Model and Test Genotype:")
    mae_pivot = summary_df[summary_df['metric'] == 'mae'].pivot_table(
        index='model',
        columns='test_genotype',
        values='mean'
    )
    print(mae_pivot.to_string())

    print("\nMean R² by Model and Test Genotype:")
    r2_pivot = summary_df[summary_df['metric'] == 'r2'].pivot_table(
        index='model',
        columns='test_genotype',
        values='mean'
    )
    print(r2_pivot.to_string())

    # ------------------------------------------------------------------------
    # Step 3: Create 3×3 comparison plots
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: CREATING 3×3 COMPARISON PLOTS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating {metric.upper()} comparison...")

        save_path = OUTPUT_PLOT_DIR / f'3model_comparison_{metric}.png'
        plot_3model_comparison(
            all_results,
            metric=metric,
            percentile_clip=True,
            save_path=save_path
        )

    # ------------------------------------------------------------------------
    # Step 4: Create "best model" heatmaps
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: CREATING 'BEST MODEL' HEATMAPS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating best model heatmap for {metric.upper()}...")

        save_path = OUTPUT_PLOT_DIR / f'best_model_{metric}.png'
        create_best_model_heatmap(
            all_results,
            metric=metric,
            save_path=save_path
        )

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nKey Features:")
    print(f"  - 3×3 grid showing all model-genotype combinations")
    print(f"  - Red borders indicate LOEO cross-validation")
    print(f"  - Color scales use 5th-95th percentile clipping")
    print(f"  - Best model heatmaps show which model performs best where")
    print(f"\nOutputs saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
