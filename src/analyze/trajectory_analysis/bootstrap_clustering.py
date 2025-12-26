"""
Bootstrap Clustering

Bootstrap resampling methods for consensus clustering with quality assessment.

Functions
---------
- run_bootstrap_hierarchical: Bootstrap hierarchical clustering with consensus labels
- compute_consensus_labels: Compute consensus cluster labels from bootstrap iterations
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from .config import N_BOOTSTRAP, BOOTSTRAP_FRAC, RANDOM_SEED


def run_bootstrap_hierarchical(
    D: np.ndarray,
    k: int,
    embryo_ids: List[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap hierarchical clustering with label alignment.

    Performs repeated hierarchical clustering on random subsamples of the data,
    storing the resulting cluster labels for posterior probability computation.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    k : int
        Number of clusters
    embryo_ids : list of str
        Embryo identifiers (required for tracking). Should encode experiment/run info.
        Example: ['cep290_wt_run1_emb_01', 'cep290_wt_run1_emb_02', ...]
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    frac : float, default=0.8
        Fraction of samples per bootstrap
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    bootstrap_results_dict : dict
        - 'embryo_ids': list of str (copy of input)
        - 'reference_labels': np.ndarray, consensus labels from full data
        - 'bootstrap_results': list of dicts
            - 'labels': np.ndarray (-1 for unsampled)
            - 'indices': np.ndarray of sampled indices
            - 'silhouette': float
        - 'n_clusters': int

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03', ...]
    >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
    >>> reference_labels = results['reference_labels']
    >>> # Lookup embryo by ID
    >>> idx = results['embryo_ids'].index('emb_02')
    >>> label = results['reference_labels'][idx]
    """
    np.random.seed(random_state)
    n_samples = len(D)

    # Compute reference labels from full data
    clusterer = AgglomerativeClustering(
        n_clusters=k,
        linkage='average',
        metric='precomputed'
    )
    reference_labels = clusterer.fit_predict(D)

    # Bootstrap iterations
    bootstrap_results = []
    n_to_sample = max(int(np.ceil(frac * n_samples)), 1)

    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations...")
        print(f"  Sampling {n_to_sample}/{n_samples} samples per iteration")

    for iter_idx in range(n_bootstrap):
        if verbose and (iter_idx + 1) % 10 == 0:
            print(f"  Progress: {iter_idx + 1}/{n_bootstrap}")

        # Random sample
        sampled_indices = np.random.choice(n_samples, size=n_to_sample, replace=False)
        sampled_indices = np.sort(sampled_indices)

        # Create submatrix
        D_subset = D[np.ix_(sampled_indices, sampled_indices)]

        # Cluster subset
        try:
            clusterer_boot = AgglomerativeClustering(
                n_clusters=k,
                linkage='average',
                metric='precomputed'
            )
            labels_subset = clusterer_boot.fit_predict(D_subset)

            # Create full-size label array with -1 for unsampled
            labels_full = np.full(n_samples, -1, dtype=int)
            labels_full[sampled_indices] = labels_subset

            # Compute silhouette if possible
            try:
                silhouette = silhouette_score(D_subset, labels_subset, metric='precomputed')
            except:
                silhouette = np.nan

            bootstrap_results.append({
                'labels': labels_full,
                'indices': sampled_indices,
                'silhouette': silhouette
            })
        except Exception as e:
            if verbose:
                print(f"    Warning: Bootstrap iteration {iter_idx} failed: {e}")
            continue

    if verbose:
        print(f"\nCompleted {len(bootstrap_results)} successful bootstrap iterations")

    return {
        'embryo_ids': list(embryo_ids),
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': k,
        'distance_matrix': D,
        'n_samples': n_samples
    }


def compute_consensus_labels(
    bootstrap_results: Dict[str, Any]
) -> np.ndarray:
    """
    Compute consensus labels from bootstrap results.

    Uses majority voting (argmax of assignment frequency matrix) to determine
    the most frequently assigned cluster for each sample across all bootstrap
    iterations.

    Parameters
    ----------
    bootstrap_results : dict
        Output from run_bootstrap_hierarchical()

    Returns
    -------
    consensus_labels : np.ndarray
        Consensus cluster assignments
    """
    n_samples = bootstrap_results['n_samples']
    bootstrap_iter_results = bootstrap_results['bootstrap_results']

    # Count cluster assignments
    assignment_matrix = np.full((n_samples, bootstrap_results['n_clusters']), 0, dtype=int)

    for boot_result in bootstrap_iter_results:
        labels = boot_result['labels']
        for i in range(n_samples):
            if labels[i] >= 0:
                assignment_matrix[i, labels[i]] += 1

    # Compute consensus: most frequent cluster per sample
    consensus_labels = np.argmax(assignment_matrix, axis=1)

    return consensus_labels


def get_cluster_assignments(
    distance_matrix: np.ndarray,
    embryo_ids: List[str],
    k_values: List[int] = [3, 4, 5, 6],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_frac: float = BOOTSTRAP_FRAC,
    random_seed: int = RANDOM_SEED,
    verbose: bool = False
) -> tuple:
    """
    Run clustering for multiple k values and return consolidated assignments.

    Wraps run_bootstrap_hierarchical() + analyze_bootstrap_results() for multiple k.
    This is a convenience function for testing multiple cluster counts.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    embryo_ids : list of str
        List of embryo identifiers
    k_values : list of int, default=[3, 4, 5, 6]
        K values to test
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    bootstrap_frac : float, default=0.8
        Fraction to sample per iteration
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    df_assignments : DataFrame
        Columns: embryo_id, cluster_k3, cluster_k4, ..., max_p_k3, max_p_k4, ...
    all_results : dict
        Full results keyed by k: {3: {'bootstrap_results': ..., 'posteriors': ...}, ...}

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> df_assignments, all_results = get_cluster_assignments(D, embryo_ids, k_values=[3,4,5])
    >>> # Access cluster assignments for k=3
    >>> cluster_3_labels = df_assignments['cluster_k3']
    >>> # Access full posterior analysis for k=3
    >>> posteriors_k3 = all_results[3]['posteriors']
    """
    import pandas as pd
    from .cluster_posteriors import analyze_bootstrap_results

    all_results = {}
    assignment_dfs = []

    for k in k_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running clustering for k={k}")
            print(f"{'='*60}")

        # Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            distance_matrix, k, embryo_ids,
            n_bootstrap=n_bootstrap,
            frac=bootstrap_frac,
            random_state=random_seed,
            verbose=verbose
        )

        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)

        # Store full results
        all_results[k] = {
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors
        }

        # Extract assignments for this k
        df_k = pd.DataFrame({
            'embryo_id': posteriors['embryo_ids'],
            f'cluster_k{k}': posteriors['modal_cluster'],
            f'max_p_k{k}': posteriors['max_p'],
            f'entropy_k{k}': posteriors['entropy']
        })
        assignment_dfs.append(df_k)

    # Merge all k values
    df_assignments = assignment_dfs[0]
    for df_k in assignment_dfs[1:]:
        df_assignments = df_assignments.merge(df_k, on='embryo_id')

    if verbose:
        print(f"\n{'='*60}")
        print(f"Clustering complete for k={k_values}")
        print(f"{'='*60}")

    return df_assignments, all_results


def compute_coassociation_matrix(
    bootstrap_results_dict: Dict[str, Any],
    *,
    verbose: bool = True
) -> np.ndarray:
    """
    Evidence Accumulation: Compute co-association matrix from bootstrap results.

    M[i,j] = fraction of bootstrap iterations where embryos i and j
             were assigned to the same cluster

    Uses RAW bootstrap labels (no Hungarian alignment needed) - this is simpler
    and alignment-agnostic. The co-clustering frequency is invariant to cluster
    numbering.

    This matrix can be used to build a consensus dendrogram where merge heights
    reflect clustering stability across bootstrap iterations.

    Parameters
    ----------
    bootstrap_results_dict : Dict[str, Any]
        Output from run_bootstrap_hierarchical()
        Must contain 'embryo_ids' and 'bootstrap_results'
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    M : np.ndarray
        Co-association matrix (n_embryos × n_embryos)
        - M[i,j] ∈ [0, 1]: fraction of co-clustering
        - Symmetric: M[i,j] = M[j,i]
        - Diagonal = 1: M[i,i] = 1.0
        - M[i,j] = 0.5 if i and j were never co-sampled (neutral)

    Examples
    --------
    >>> # After bootstrap clustering
    >>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=ids)
    >>> M = compute_coassociation_matrix(bootstrap_results)
    >>>
    >>> # High co-association means stable clustering
    >>> print(f"Mean co-association: {M[np.triu_indices(len(M), k=1)].mean():.3f}")
    >>>
    >>> # Build consensus dendrogram
    >>> D_consensus = coassociation_to_distance(M)
    >>> fig, info = generate_dendrograms(D, ids, coassociation_matrix=M)

    Notes
    -----
    - Uses raw bootstrap labels (no Hungarian alignment)
    - Handles unsampled embryos (labels = -1) correctly
    - Co-association = co-cluster count / co-sample count
    - If i and j never co-sampled, M[i,j] = 0.5 (neutral prior)
    - Consensus distance = 1 - M (use coassociation_to_distance())

    References
    ----------
    Fred, A.L.N., & Jain, A.K. (2005). Combining multiple clusterings using
    evidence accumulation. IEEE TPAMI, 27(6), 835-850.
    """
    n_embryos = len(bootstrap_results_dict['embryo_ids'])
    bootstrap_results = bootstrap_results_dict['bootstrap_results']

    # Count co-clustering and co-sampling
    coassoc_count = np.zeros((n_embryos, n_embryos), dtype=int)
    cosample_count = np.zeros((n_embryos, n_embryos), dtype=int)

    for boot_result in bootstrap_results:
        labels = boot_result['labels']  # RAW labels (no alignment!)

        for i in range(n_embryos):
            for j in range(i, n_embryos):
                # Both sampled? (labels >= 0)
                if labels[i] >= 0 and labels[j] >= 0:
                    cosample_count[i, j] += 1
                    cosample_count[j, i] += 1

                    # Same cluster?
                    if labels[i] == labels[j]:
                        coassoc_count[i, j] += 1
                        coassoc_count[j, i] += 1

    # Compute co-association frequency
    M = np.zeros((n_embryos, n_embryos), dtype=float)
    for i in range(n_embryos):
        for j in range(n_embryos):
            if cosample_count[i, j] > 0:
                M[i, j] = coassoc_count[i, j] / cosample_count[i, j]
            else:
                # Never co-sampled: use neutral prior (0.5)
                M[i, j] = 0.5 if i != j else 1.0

    # Ensure diagonal = 1.0
    np.fill_diagonal(M, 1.0)

    if verbose:
        # Compute statistics on upper triangle (exclude diagonal)
        upper_tri_indices = np.triu_indices(n_embryos, k=1)
        upper_tri_values = M[upper_tri_indices]

        print(f"\nCo-association matrix computed (Evidence Accumulation):")
        print(f"  Size: {n_embryos} × {n_embryos}")
        print(f"  Bootstrap iterations: {len(bootstrap_results)}")
        print(f"  Mean co-association: {upper_tri_values.mean():.3f}")
        print(f"  Std co-association: {upper_tri_values.std():.3f}")
        print(f"  Range: [{upper_tri_values.min():.3f}, {upper_tri_values.max():.3f}]")

    return M


def coassociation_to_distance(M: np.ndarray) -> np.ndarray:
    """
    Convert co-association matrix to distance matrix for consensus dendrogram.

    Consensus distance = 1 - co-association frequency

    This transformation allows using the co-association matrix with hierarchical
    clustering algorithms that expect distance matrices.

    Interpretation of consensus distances:
    - D[i,j] = 0.0: always clustered together (100% co-clustering)
    - D[i,j] = 0.5: neutral (50% co-clustering, or never co-sampled)
    - D[i,j] = 1.0: never clustered together (0% co-clustering)

    Parameters
    ----------
    M : np.ndarray
        Co-association matrix (n × n), values in [0, 1]
        Output from compute_coassociation_matrix()

    Returns
    -------
    D : np.ndarray
        Consensus distance matrix (n × n)
        - D[i,j] = 1 - M[i,j]
        - Symmetric, zero diagonal
        - Values in [0, 1]

    Examples
    --------
    >>> M = compute_coassociation_matrix(bootstrap_results)
    >>> D_consensus = coassociation_to_distance(M)
    >>> fig, info = generate_dendrograms(D, ids, coassociation_matrix=M)

    Notes
    -----
    - This is a simple transformation: D = 1 - M
    - The resulting distance matrix is suitable for hierarchical clustering
    - Merge heights in dendrogram = 1 - co-clustering frequency
    """
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    return D
