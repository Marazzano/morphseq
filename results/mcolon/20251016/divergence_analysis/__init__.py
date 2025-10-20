"""
Morphological divergence analysis module.

Computes distance-based metrics to quantify how much individual embryos
diverge from a reference distribution. The reference can be any genotype
(wildtype, heterozygous, etc.).

Main interface: compute_divergence_scores()
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import warnings

from .distances import (
    compute_mahalanobis_distance,
    compute_euclidean_distance,
    compute_standardized_distance,
    compute_cosine_distance,
    detect_outliers_mahalanobis
)
from .reference import (
    compute_reference_statistics,
    validate_reference_stats,
    get_reference_for_time
)


def compute_divergence_scores(
    df_binned: pd.DataFrame,
    test_genotype: str,
    reference_genotype: str,
    metrics: List[str] = ["mahalanobis", "euclidean"],
    time_col: str = "time_bin",
    z_cols: Optional[List[str]] = None,
    min_reference_samples: int = 10,
    outlier_alpha: float = 0.001,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute divergence of each test embryo from reference distribution.
    
    The reference can be ANY genotype, not just wildtype. This allows
    flexible comparisons:
    - Mutant vs wildtype (standard)
    - Mutant vs heterozygous
    - One mutant vs another mutant
    - Treated vs untreated
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data with latent features
    test_genotype : str
        Genotype to compute divergence for (e.g., "cep290_homozygous")
    reference_genotype : str
        Genotype to use as reference (e.g., "cep290_wildtype")
        Can be ANY genotype in the data
    metrics : list of str, default=["mahalanobis", "euclidean"]
        Distance metrics to compute. Options:
        - "mahalanobis": Mahalanobis distance (accounts for correlations)
        - "euclidean": Euclidean distance (simple L2 distance)
        - "standardized": Standardized Euclidean (z-scored features)
        - "cosine": Cosine distance (directional)
    time_col : str, default="time_bin"
        Column name for time bins
    z_cols : list of str, optional
        Latent feature columns. Auto-detected if None.
    min_reference_samples : int, default=10
        Minimum reference samples needed per time bin
    outlier_alpha : float, default=0.001
        Significance level for Mahalanobis outlier detection
    verbose : bool, default=True
        Print progress and summary
    
    Returns
    -------
    pd.DataFrame
        One row per test embryo-timepoint with columns:
        - embryo_id: Embryo identifier
        - time_bin: Time bin
        - genotype: Test genotype
        - mahalanobis_distance: Mahalanobis distance (if requested)
        - euclidean_distance: Euclidean distance (if requested)
        - standardized_distance: Standardized distance (if requested)
        - cosine_distance: Cosine distance (if requested)
        - is_outlier: Statistical outlier flag (if Mahalanobis computed)
        - n_reference_samples: Number of reference embryos at this time
    
    Examples
    --------
    >>> # Standard: mutant vs wildtype
    >>> df_div = compute_divergence_scores(
    ...     df_binned,
    ...     test_genotype="cep290_homozygous",
    ...     reference_genotype="cep290_wildtype"
    ... )
    
    >>> # Mutant vs heterozygous
    >>> df_div = compute_divergence_scores(
    ...     df_binned,
    ...     test_genotype="cep290_homozygous",
    ...     reference_genotype="cep290_heterozygous"
    ... )
    
    >>> # All metrics
    >>> df_div = compute_divergence_scores(
    ...     df_binned,
    ...     test_genotype="cep290_homozygous",
    ...     reference_genotype="cep290_wildtype",
    ...     metrics=["mahalanobis", "euclidean", "standardized", "cosine"]
    ... )
    """
    if verbose:
        print(f"\nComputing divergence: {test_genotype} vs {reference_genotype}")
        print(f"Metrics: {', '.join(metrics)}")
    
    # Auto-detect latent columns if needed
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError("No latent columns found. Specify z_cols explicitly.")
        if verbose:
            print(f"Using {len(z_cols)} latent features")
    
    # Compute reference statistics
    if verbose:
        print(f"\nComputing reference statistics for {reference_genotype}...")
    
    reference_stats = compute_reference_statistics(
        df_binned,
        reference_genotype=reference_genotype,
        time_col=time_col,
        z_cols=z_cols,
        min_samples=min_reference_samples
    )
    
    if verbose:
        validate_reference_stats(reference_stats, verbose=True)
    
    # Filter to test genotype
    df_test = df_binned[df_binned['genotype'] == test_genotype].copy()
    
    if df_test.empty:
        raise ValueError(
            f"Test genotype '{test_genotype}' not found in data. "
            f"Available: {df_binned['genotype'].unique().tolist()}"
        )
    
    if verbose:
        print(f"\nComputing distances for {len(df_test)} test embryo-timepoints...")
    
    # Compute distances for each time bin
    results = []
    
    for time_bin, group in df_test.groupby(time_col):
        # Get reference stats for this time bin
        ref_stats = get_reference_for_time(reference_stats, time_bin)
        
        if ref_stats is None:
            warnings.warn(f"No reference data for time bin {time_bin}, skipping")
            continue
        
        # Extract feature matrix
        X = group[z_cols].values
        
        # Get reference statistics
        mu_ref = ref_stats['mean']
        cov_ref = ref_stats['cov']
        std_ref = ref_stats['std']
        n_ref = ref_stats['n_samples']
        
        # Compute requested metrics
        distances = {}
        
        if "mahalanobis" in metrics:
            distances['mahalanobis_distance'] = compute_mahalanobis_distance(
                X, mu_ref, cov_ref
            )
        
        if "euclidean" in metrics:
            distances['euclidean_distance'] = compute_euclidean_distance(
                X, mu_ref
            )
        
        if "standardized" in metrics:
            distances['standardized_distance'] = compute_standardized_distance(
                X, mu_ref, std_ref
            )
        
        if "cosine" in metrics:
            distances['cosine_distance'] = compute_cosine_distance(
                X, mu_ref
            )
        
        # Detect outliers if Mahalanobis was computed
        is_outlier = None
        if "mahalanobis" in metrics:
            is_outlier = detect_outliers_mahalanobis(
                distances['mahalanobis_distance'],
                n_features=len(z_cols),
                alpha=outlier_alpha
            )
        
        # Store results for this time bin
        for i, (idx, row) in enumerate(group.iterrows()):
            result = {
                'embryo_id': row.get('embryo_id', f'embryo_{idx}'),
                'time_bin': time_bin,
                'genotype': test_genotype,
                'n_reference_samples': n_ref
            }
            
            # Add distance metrics
            for metric_name, metric_values in distances.items():
                result[metric_name] = metric_values[i]
            
            # Add outlier flag
            if is_outlier is not None:
                result['is_outlier'] = is_outlier[i]
            
            results.append(result)
    
    # Create DataFrame
    df_divergence = pd.DataFrame(results)
    
    if verbose:
        print(f"\nResults: {len(df_divergence)} embryo-timepoints")
        if 'mahalanobis_distance' in df_divergence.columns:
            print(f"Mahalanobis distance: {df_divergence['mahalanobis_distance'].mean():.3f} ± {df_divergence['mahalanobis_distance'].std():.3f}")
            if 'is_outlier' in df_divergence.columns:
                n_outliers = df_divergence['is_outlier'].sum()
                print(f"Outliers detected: {n_outliers} ({100*n_outliers/len(df_divergence):.1f}%)")
    
    return df_divergence


def compare_to_multiple_references(
    df_binned: pd.DataFrame,
    test_genotype: str,
    reference_genotypes: List[str],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Compute divergence from multiple reference genotypes.
    
    Useful for comparing a test genotype to multiple references
    (e.g., compare homozygous to both wildtype and heterozygous).
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    test_genotype : str
        Test genotype
    reference_genotypes : list of str
        Multiple reference genotypes to compare against
    **kwargs
        Additional arguments passed to compute_divergence_scores()
    
    Returns
    -------
    dict
        Keys are reference genotypes, values are divergence DataFrames
    
    Examples
    --------
    >>> results = compare_to_multiple_references(
    ...     df_binned,
    ...     test_genotype="cep290_homozygous",
    ...     reference_genotypes=["cep290_wildtype", "cep290_heterozygous"]
    ... )
    >>> df_vs_wt = results["cep290_wildtype"]
    >>> df_vs_het = results["cep290_heterozygous"]
    """
    results = {}
    
    for ref_genotype in reference_genotypes:
        results[ref_genotype] = compute_divergence_scores(
            df_binned,
            test_genotype=test_genotype,
            reference_genotype=ref_genotype,
            **kwargs
        )
    
    return results


__all__ = [
    'compute_divergence_scores',
    'compare_to_multiple_references',
    'compute_reference_statistics',
    'validate_reference_stats',
]
