"""
Deprecated compatibility wrappers for binary classification tests.

Use src.analyze.difference_detection.classification_test instead.
"""

import warnings
from typing import Dict, List, Any, Optional, Union

from ..classification_test import (
    assign_group_labels,
    run_binary_classification_test,
    compute_timeseries_divergence,
)

warnings.warn(
    "difference_detection.compat.comparison is deprecated. "
    "Use difference_detection.classification_test instead.",
    DeprecationWarning,
    stacklevel=2,
)


def add_group_column(
    df,
    # Mode 1: Manual group assignment
    groups: Optional[Dict[str, List[str]]] = None,
    # Mode 2: From k_results
    k_results: Optional[Dict] = None,
    k: Optional[int] = None,
    cluster_names: Optional[Dict[int, str]] = None,
    membership: Optional[str] = None,
    # Common params
    column_name: str = 'group',
    embryo_id_col: str = 'embryo_id',
    inplace: bool = False,
):
    """
    Deprecated wrapper for assign_group_labels().
    """
    return assign_group_labels(
        df,
        groups=groups,
        k_results=k_results,
        k=k,
        cluster_names=cluster_names,
        membership=membership,
        group_col=column_name,
        embryo_id_col=embryo_id_col,
        inplace=inplace,
    )


def compare_groups(
    df,
    group_col: str,
    group1: str,
    group2: str,
    features: Union[str, List[str]] = 'z_mu_b',
    morphology_metric: Optional[str] = 'total_length_um',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_bin: int = 5,
    within_bin_time_stratification: bool = True,
    within_bin_time_strata_width: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Deprecated wrapper for run_binary_classification_test().
    """
    return run_binary_classification_test(
        df=df,
        group_col=group_col,
        group1=group1,
        group2=group2,
        features=features,
        morphology_metric=morphology_metric,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        bin_width=bin_width,
        n_splits=n_splits,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        min_samples_per_bin=min_samples_per_bin,
        within_bin_time_stratification=within_bin_time_stratification,
        within_bin_time_strata_width=within_bin_time_strata_width,
        random_state=random_state,
        verbose=verbose,
    )


def compute_metric_divergence(
    df,
    group_col: str,
    group1: str,
    group2: str,
    metric_col: str,
    time_col: str,
    embryo_id_col: str,
):
    """
    Deprecated wrapper for compute_timeseries_divergence().
    """
    return compute_timeseries_divergence(
        df=df,
        group_col=group_col,
        group1=group1,
        group2=group2,
        metric_col=metric_col,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
    )


__all__ = [
    "add_group_column",
    "compare_groups",
    "compute_metric_divergence",
]
