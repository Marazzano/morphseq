"""
K Selection Pipeline for Consensus Clustering

Evaluates multiple k values BEFORE filtering to help decide optimal k.
This addresses the workflow issue where filtering was done before knowing
the right k, potentially removing embryos that would form good clusters.

New Workflow:
1. Compute distance matrix (no filtering)
2. Run bootstrap clustering for k in [2, 3, 4, 5, ...]
3. For each k, compute quality metrics:
   - % Core (high confidence assignments)
   - % Outlier (low confidence)
   - Mean max_p
   - Mean entropy
   - Silhouette score
4. Plot comparison → Pick best k
5. Re-run consensus pipeline with chosen k + filtering

Created: 2025-12-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


def add_membership_column(
    df: pd.DataFrame,
    classification: Dict[str, Any],
    column_name: str = 'membership'
) -> pd.DataFrame:
    """
    Add membership category column to DataFrame based on classification results.
    
    Maps each embryo_id to its membership category (core/uncertain/outlier).
    This enables using plot_multimetric_trajectories with color_by_grouping='membership'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with 'embryo_id' column
    classification : Dict
        Output from classify_membership_2d() with 'embryo_ids' and 'category' keys
    column_name : str
        Name for the new column (default: 'membership')
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with new membership column added
        
    Example
    -------
    >>> classification = classify_membership_2d(posteriors['max_p'], ...)
    >>> df = add_membership_column(df, classification)
    >>> fig = plot_multimetric_trajectories(df, ..., color_by_grouping='membership')
    """
    # Create mapping from embryo_id to category
    embryo_to_cat = dict(zip(classification['embryo_ids'], classification['category']))
    
    # Map to DataFrame
    df = df.copy()
    df[column_name] = df['embryo_id'].map(embryo_to_cat)
    
    return df


def evaluate_k_range(
    D: np.ndarray,
    embryo_ids: List[str],
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate multiple k values for consensus clustering WITHOUT filtering.
    
    This helps decide the optimal k before applying any outlier filtering.
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    embryo_ids : List[str]
        Embryo identifiers
    k_range : List[int]
        K values to evaluate (default: [2, 3, 4, 5, 6])
    n_bootstrap : int
        Bootstrap iterations per k (default: 100)
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict
        - 'k_values': list of k tested
        - 'metrics': Dict[k] → quality metrics for each k
        - 'summary_df': DataFrame with comparison
        - 'best_k': recommended k based on core %
    """
    from src.analyze.trajectory_analysis import (
        run_bootstrap_hierarchical,
        analyze_bootstrap_results,
    )
    from src.analyze.trajectory_analysis.cluster_classification import (
        classify_membership_2d,
    )
    from sklearn.metrics import silhouette_score
    
    results_by_k = {}
    
    for k in k_range:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating k={k}")
            print('='*60)
        
        # Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            D=D,
            k=k,
            embryo_ids=embryo_ids,
            n_bootstrap=n_bootstrap,
            verbose=verbose
        )
        
        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)
        
        # Classify membership quality
        classification = classify_membership_2d(
            max_p=posteriors['max_p'],
            log_odds_gap=posteriors['log_odds_gap'],
            modal_cluster=posteriors['modal_cluster'],
            embryo_ids=posteriors['embryo_ids']
        )
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score(D, posteriors['modal_cluster'], metric='precomputed')
        except:
            sil_score = np.nan
        
        # Compute summary metrics
        categories = classification['category']
        n_total = len(categories)
        n_core = np.sum(categories == 'core')
        n_uncertain = np.sum(categories == 'uncertain')
        n_outlier = np.sum(categories == 'outlier')
        
        metrics = {
            'n_embryos': n_total,
            'n_core': n_core,
            'n_uncertain': n_uncertain,
            'n_outlier': n_outlier,
            'pct_core': 100.0 * n_core / n_total,
            'pct_uncertain': 100.0 * n_uncertain / n_total,
            'pct_outlier': 100.0 * n_outlier / n_total,
            'mean_max_p': posteriors['max_p'].mean(),
            'mean_entropy': posteriors['entropy'].mean(),
            'silhouette': sil_score,
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors,
            'classification': classification,
        }
        
        results_by_k[k] = metrics
        
        if verbose:
            print(f"\nk={k} Summary:")
            print(f"  Core: {n_core} ({metrics['pct_core']:.1f}%)")
            print(f"  Uncertain: {n_uncertain} ({metrics['pct_uncertain']:.1f}%)")
            print(f"  Outlier: {n_outlier} ({metrics['pct_outlier']:.1f}%)")
            print(f"  Mean max_p: {metrics['mean_max_p']:.3f}")
            print(f"  Mean entropy: {metrics['mean_entropy']:.3f}")
            print(f"  Silhouette: {sil_score:.3f}")
    
    # Create summary DataFrame
    summary_data = []
    for k in k_range:
        m = results_by_k[k]
        summary_data.append({
            'k': k,
            'pct_core': m['pct_core'],
            'pct_uncertain': m['pct_uncertain'],
            'pct_outlier': m['pct_outlier'],
            'mean_max_p': m['mean_max_p'],
            'mean_entropy': m['mean_entropy'],
            'silhouette': m['silhouette'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Find best k (highest core %)
    best_k = summary_df.loc[summary_df['pct_core'].idxmax(), 'k']
    
    if verbose:
        print(f"\n{'='*60}")
        print("K SELECTION SUMMARY")
        print('='*60)
        print(summary_df.to_string(index=False))
        print(f"\nRecommended k: {best_k} (highest % core assignments)")
    
    return {
        'k_values': k_range,
        'metrics': results_by_k,
        'summary_df': summary_df,
        'best_k': int(best_k),
    }


def plot_k_selection(
    k_results: Dict[str, Any],
    figsize: tuple = (16, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot quality metrics across k values to help select optimal k.
    
    Creates a 2x2 grid:
    - Top-left: Membership % (core/uncertain/outlier) vs k
    - Top-right: Mean max_p vs k  
    - Bottom-left: Mean entropy vs k
    - Bottom-right: Silhouette score vs k
    
    Parameters
    ----------
    k_results : Dict
        Output from evaluate_k_range()
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    summary_df = k_results['summary_df']
    k_values = summary_df['k'].values
    best_k = k_results['best_k']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Membership % vs k
    ax = axes[0, 0]
    ax.plot(k_values, summary_df['pct_core'], 'o-', color='green', 
            linewidth=2.5, markersize=10, label='Core')
    ax.plot(k_values, summary_df['pct_uncertain'], 's-', color='orange', 
            linewidth=2.5, markersize=10, label='Uncertain')
    ax.plot(k_values, summary_df['pct_outlier'], '^-', color='red', 
            linewidth=2.5, markersize=10, label='Outlier')
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5, 
               label=f'Best k={best_k}')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Quality vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Mean max_p vs k
    ax = axes[0, 1]
    ax.plot(k_values, summary_df['mean_max_p'], 'o-', color='steelblue', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Threshold (0.5)')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Max Posterior', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Confidence vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean entropy vs k
    ax = axes[1, 0]
    ax.plot(k_values, summary_df['mean_entropy'], 'o-', color='coral', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Assignment Ambiguity vs K (lower = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    # 4. Silhouette score vs k
    ax = axes[1, 1]
    ax.plot(k_values, summary_df['silhouette'], 'o-', color='purple', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Separation vs K (higher = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'K Selection Analysis\nRecommended k = {best_k}', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def run_k_selection_pipeline(
    D: np.ndarray,
    embryo_ids: List[str],
    df: pd.DataFrame,
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete K selection workflow.
    
    1. Evaluate all k values (NO filtering)
    2. Plot comparison metrics
    3. Recommend best k
    4. Return results for chosen k
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    embryo_ids : List[str]
        Embryo IDs
    df : pd.DataFrame
        Trajectory data (for plotting)
    k_range : List[int]
        K values to test
    n_bootstrap : int
        Bootstrap iterations
    output_dir : Path, optional
        Directory for output files
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict
        - 'k_results': full evaluation results
        - 'best_k': recommended k
        - 'best_results': bootstrap results for best k
        - 'summary_df': comparison DataFrame
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("K SELECTION PIPELINE (No Filtering)")
    print("="*70)
    print(f"\nTesting k values: {k_range}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Embryos: {len(embryo_ids)}")
    print("="*70)
    
    # Step 1: Evaluate all k values
    k_results = evaluate_k_range(
        D=D,
        embryo_ids=embryo_ids,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )
    
    # Step 2: Plot comparison
    fig = plot_k_selection(
        k_results,
        save_path=output_dir / 'k_selection_metrics.png' if output_dir else None
    )
    plt.show()
    
    # Step 3: Save summary
    if output_dir:
        k_results['summary_df'].to_csv(output_dir / 'k_selection_summary.csv', index=False)
        print(f"\n✓ Saved summary: {output_dir / 'k_selection_summary.csv'}")
    
    # Step 4: Return best k results
    best_k = k_results['best_k']
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: k = {best_k}")
    print(f"{'='*70}")
    print(f"\nNext step: Run consensus pipeline with k={best_k} and filtering enabled")
    print(f"  results = run_consensus_pipeline(D, embryo_ids, k={best_k}, ...)")
    
    return {
        'k_results': k_results,
        'best_k': best_k,
        'best_bootstrap_results': k_results['metrics'][best_k]['bootstrap_results'],
        'best_posteriors': k_results['metrics'][best_k]['posteriors'],
        'best_classification': k_results['metrics'][best_k]['classification'],
        'summary_df': k_results['summary_df'],
    }


# ============================================================================
# ALTERNATIVE: TWO-PHASE PIPELINE
# ============================================================================

def run_two_phase_pipeline(
    D: np.ndarray,
    embryo_ids: List[str],
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    iqr_multiplier: float = 1.5,
    posterior_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Two-phase consensus clustering pipeline.
    
    Phase 1: K Selection (no filtering)
    - Evaluate k_range with bootstrap clustering
    - Select best k based on quality metrics
    
    Phase 2: Final Clustering (with filtering)
    - Run consensus pipeline with best k
    - Apply Stage 1 (IQR distance) + Stage 2 (posterior) filtering
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    embryo_ids : List[str]
        Embryo IDs
    k_range : List[int]
        K values to test in Phase 1
    n_bootstrap : int
        Bootstrap iterations
    iqr_multiplier : float
        IQR multiplier for Stage 1 filtering (Phase 2)
    posterior_threshold : float
        Posterior threshold for Stage 2 filtering (Phase 2)
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict
        - 'phase1_results': K selection results
        - 'phase2_results': Final consensus pipeline results
        - 'best_k': Selected k value
    """
    from src.analyze.trajectory_analysis import run_consensus_pipeline
    
    print("="*70)
    print("TWO-PHASE CONSENSUS CLUSTERING PIPELINE")
    print("="*70)
    
    # =========================================================================
    # PHASE 1: K Selection (no filtering)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: K SELECTION (No Filtering)")
    print("="*70)
    
    phase1_results = evaluate_k_range(
        D=D,
        embryo_ids=embryo_ids,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )
    
    best_k = phase1_results['best_k']
    
    # Plot k selection
    fig = plot_k_selection(phase1_results)
    plt.show()
    
    print(f"\n✓ Phase 1 complete. Best k = {best_k}")
    
    # =========================================================================
    # PHASE 2: Final Clustering (with filtering)
    # =========================================================================
    print("\n" + "="*70)
    print(f"PHASE 2: FINAL CLUSTERING (k={best_k}, with filtering)")
    print("="*70)
    
    phase2_results = run_consensus_pipeline(
        D=D,
        embryo_ids=embryo_ids,
        k=best_k,
        n_bootstrap=n_bootstrap,
        enable_stage1_filtering=True,
        enable_stage2_filtering=True,
        stage1_method='iqr',  # Use IQR distance filtering
        iqr_multiplier=iqr_multiplier,
        posterior_threshold=posterior_threshold,
        k_highlight=[best_k - 1, best_k, best_k + 1],
        verbose=verbose
    )
    
    print(f"\n✓ Phase 2 complete.")
    print(f"  Initial embryos: {len(embryo_ids)}")
    print(f"  After Stage 1: {len(phase2_results['embryo_ids_after_stage1'])}")
    print(f"  Final embryos: {len(phase2_results['final_embryo_ids'])}")
    
    return {
        'phase1_results': phase1_results,
        'phase2_results': phase2_results,
        'best_k': best_k,
    }
