"""
Deprecated compatibility wrappers for multiclass classification tests.

Use src.analyze.difference_detection.classification_test_multiclass instead.
"""

import warnings
from typing import Dict, List, Any, Union

from ..classification_test_multiclass import run_multiclass_classification_test

warnings.warn(
    "difference_detection.compat.comparison_multiclass is deprecated. "
    "Use difference_detection.classification_test_multiclass instead.",
    DeprecationWarning,
    stacklevel=2,
)


def compare_groups_multiclass(
    df,
    groups: Dict[str, List[str]],
    features: Union[str, List[str]] = 'z_mu_b',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_class: int = 3,
    within_bin_time_stratification: bool = True,
    within_bin_time_strata_width: float = 0.5,
    skip_bin_if_not_all_present: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Deprecated wrapper for run_multiclass_classification_test().
    """
    return run_multiclass_classification_test(
        df=df,
        groups=groups,
        features=features,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        bin_width=bin_width,
        n_splits=n_splits,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        min_samples_per_class=min_samples_per_class,
        within_bin_time_stratification=within_bin_time_stratification,
        within_bin_time_strata_width=within_bin_time_strata_width,
        skip_bin_if_not_all_present=skip_bin_if_not_all_present,
        random_state=random_state,
        verbose=verbose,
    )


__all__ = [
    "compare_groups_multiclass",
]
