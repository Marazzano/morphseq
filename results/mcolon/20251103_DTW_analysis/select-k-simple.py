# 2_select_k.py
"""Functions for selecting optimal number of clusters."""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple

# ============ CORE FUNCTIONS ============

def compute_elbow(D: np.ndarray, labels_dict: Dict) -> Dict:
    """Compute within-cluster distances for elbow method."""
    wcss = {}
    for k, data in labels_dict.items():
        labels = data['labels']
        wc_dist = 0
        for cluster in np.unique(labels):
            mask = labels == cluster
            cluster_D = D[np.ix_(mask, mask)]
            wc_dist += cluster_D.sum() / 2  # Sum of pairwise distances
        wcss[k] = wc_dist
    return wcss


def consensus_clustering(C: np.ndarray, k: int) -> np.ndarray:
    """Cluster the co-association matrix to find consensus."""
    # Convert to distance and cluster
    dist = 1 - C
    condensed = dist[np.triu_indices(len(dist), k=1)]
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, k, criterion='maxclust') - 1
    return labels


def eigengap_analysis(D: np.ndarray, sigma: float = None) -> np.ndarray:
    """Compute eigengaps from affinity matrix."""
    if sigma is None:
        sigma = np.median(D[D > 0])
    
    # Create affinity matrix
    A = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(A, 0)
    
    # Normalized Laplacian
    D_diag = np.sum(A, axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D_diag + 1e-10))
    L = np.eye(len(A)) - D_sqrt_inv @ A @ D_sqrt_inv
    
    # Eigenvalues
    eigenvals = np.linalg.eigvalsh(L)
    eigengaps = np.diff(eigenvals)
    
    return eigenvals, eigengaps


def gap_statistic_simple(D: np.ndarray, labels: np.ndarray, n_refs: int = 10) -> float:
    """Simplified gap statistic."""
    # Observed within-cluster sum
    obs_wss = 0
    for k in np.unique(labels):
        mask = labels == k
        cluster_D = D[np.ix_(mask, mask)]
        obs_wss += cluster_D.sum() / (2 * mask.sum())
    
    # Reference distribution (uniform random)
    ref_wss = []
    n = len(D)
    for _ in range(n_refs):
        rand_labels = np.random.randint(0, len(np.unique(labels)), n)
        ref_w = 0
        for k in np.unique(rand_labels):
            mask = rand_labels == k
            cluster_D = D[np.ix_(mask, mask)]
            ref_w += cluster_D.sum() / (2 * mask.sum())
        ref_wss.append(ref_w)
    
    gap = np.mean(ref_wss) - obs_wss
    gap_std = np.std(ref_wss)
    return gap, gap_std


# ============ WRAPPER FUNCTIONS ============

def evaluate_all_k(D: np.ndarray, baseline_results: Dict, 
                   coassoc_matrices: Dict = None) -> Dict:
    """Evaluate all k values with multiple metrics."""
    metrics = {}
    
    for k, data in baseline_results.items():
        labels = data['labels']
        metrics[k] = {
            'silhouette': silhouette_score(D, labels, metric='precomputed'),
            'wcss': compute_elbow(D, {k: data})[k]
        }
        
        # Gap statistic
        gap, gap_std = gap_statistic_simple(D, labels)
        metrics[k]['gap'] = gap
        metrics[k]['gap_std'] = gap_std
        
        # If we have co-association matrix, check consensus quality
        if coassoc_matrices and k in coassoc_matrices:
            C = coassoc_matrices[k]
            consensus_labels = consensus_clustering(C, k)
            # How well do consensus labels match original?
            from sklearn.metrics import adjusted_rand_score
            metrics[k]['consensus_ari'] = adjusted_rand_score(labels, consensus_labels)
    
    return metrics


def suggest_k(metrics: Dict, prior_k: int = 3) -> int:
    """Simple heuristic to suggest best k."""
    scores = {}
    
    for k, m in metrics.items():
        score = 0
        
        # Higher silhouette is better
        score += m['silhouette']
        
        # Higher gap is better
        if 'gap' in m:
            score += m['gap'] / (m['gap_std'] + 1)
        
        # Higher consensus ARI is better
        if 'consensus_ari' in m:
            score += m['consensus_ari']
        
        # Small penalty for complexity
        score -= 0.1 * k
        
        # Bonus for prior
        if k == prior_k:
            score += 0.2
            
        scores[k] = score
    
    return max(scores, key=scores.get)


# ============ PLOTTING FUNCTIONS (signatures only) ============

def plot_elbow(wcss_dict, title="Elbow Plot"):
    """Plot elbow curve for WCSS."""
    pass

def plot_silhouettes(metrics, title="Silhouette Scores"):
    """Bar plot of silhouette scores by k."""
    pass

def plot_eigengaps(eigenvals, eigengaps, title="Eigengap Analysis"):
    """Plot eigenvalues and gaps."""
    pass

def plot_consensus_blocks(C, labels, title="Consensus Matrix"):
    """Plot reordered co-association matrix showing block structure."""
    pass

def plot_metric_comparison(metrics, title="K Selection Metrics"):
    """Multi-panel plot comparing all metrics."""
    pass