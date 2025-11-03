# 1_cluster.py
"""Core clustering and bootstrap stability functions."""

import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score, silhouette_score
from typing import Dict, List, Tuple

# ============ CORE FUNCTIONS ============

def cluster_kmedoids(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-medoids clustering."""
    km = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = km.fit_predict(D)
    return labels, km.medoid_indices_


def bootstrap_once(D: np.ndarray, k: int, frac: float = 0.8) -> Dict:
    """Single bootstrap: sample, cluster, return results."""
    n = len(D)
    idx = np.random.choice(n, int(n * frac), replace=False)
    
    # Submatrix and cluster
    D_sub = D[np.ix_(idx, idx)]
    labels, medoids = cluster_kmedoids(D_sub, k)
    
    # Map back to full indices
    full_labels = np.full(n, -1)
    full_labels[idx] = labels
    
    return {
        'labels': full_labels,
        'indices': idx,
        'silhouette': silhouette_score(D_sub, labels, metric='precomputed')
    }


def compute_coassoc(bootstrap_results: List[Dict]) -> np.ndarray:
    """Compute co-association matrix from bootstrap results."""
    n = len(bootstrap_results[0]['labels'])
    C = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    for res in bootstrap_results:
        labels = res['labels']
        idx = res['indices']
        
        # Count co-occurrences
        for i in idx:
            for j in idx:
                if i < j:
                    counts[i,j] += 1
                    counts[j,i] += 1
                    if labels[i] == labels[j]:
                        C[i,j] += 1
                        C[j,i] += 1
    
    # Normalize
    C = np.divide(C, counts, where=(counts > 0))
    np.fill_diagonal(C, 1.0)
    return C


# ============ WRAPPER FUNCTIONS ============

def run_baseline(D: np.ndarray, k_values: List[int] = [2,3,4]) -> Dict:
    """Run baseline clustering for multiple k."""
    results = {}
    for k in k_values:
        labels, medoids = cluster_kmedoids(D, k)
        results[k] = {
            'labels': labels,
            'medoids': medoids,
            'silhouette': silhouette_score(D, labels, metric='precomputed')
        }
    return results


def run_bootstrap(D: np.ndarray, k: int, n_boot: int = 100) -> Dict:
    """Run full bootstrap stability analysis."""
    # Reference labels
    ref_labels, ref_medoids = cluster_kmedoids(D, k)
    
    # Bootstrap
    boot_results = []
    ari_scores = []
    
    for _ in range(n_boot):
        res = bootstrap_once(D, k)
        boot_results.append(res)
        
        # ARI for sampled points
        idx = res['indices']
        if len(idx) > 0:
            ari = adjusted_rand_score(ref_labels[idx], res['labels'][idx])
            ari_scores.append(ari)
    
    # Co-association matrix
    C = compute_coassoc(boot_results)
    
    return {
        'reference_labels': ref_labels,
        'reference_medoids': ref_medoids,
        'coassoc': C,
        'ari_scores': np.array(ari_scores),
        'bootstrap_results': boot_results
    }


# ============ PLOTTING FUNCTIONS (signatures only) ============

def plot_clustering(D, labels, title="Clustering Result"):
    """Plot clustering result (implement with MDS/PCA projection)."""
    pass

def plot_coassoc_matrix(C, labels=None, title="Co-association Matrix"):
    """Plot co-association matrix as heatmap."""
    pass

def plot_stability_scores(ari_scores, title="Bootstrap Stability (ARI)"):
    """Plot distribution of ARI scores."""
    pass