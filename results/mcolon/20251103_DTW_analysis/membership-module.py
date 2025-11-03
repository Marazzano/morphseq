# 3_membership.py
"""Classify embryos as core, uncertain, or outlier."""

import numpy as np
from sklearn.metrics import silhouette_samples
from typing import Dict, Tuple

# ============ CORE FUNCTIONS ============

def compute_membership_scores(C: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute membership strength for each embryo."""
    n = len(labels)
    scores = {}
    
    for i in range(n):
        cluster = labels[i]
        # Get co-associations with cluster mates
        cluster_mask = labels == cluster
        cluster_coassoc = C[i, cluster_mask]
        
        # Median co-association with cluster
        median_intra = np.median(cluster_coassoc[cluster_coassoc < 1])  # Exclude self
        
        # Mean co-association with other clusters
        other_mask = labels != cluster
        if other_mask.any():
            mean_inter = np.mean(C[i, other_mask])
        else:
            mean_inter = 0
        
        scores[i] = {
            'cluster': cluster,
            'intra_coassoc': median_intra,
            'inter_coassoc': mean_inter,
            'total_coassoc': np.mean(C[i, :])
        }
    
    return scores


def classify_members(C: np.ndarray, labels: np.ndarray, D: np.ndarray,
                     core_thresh: float = 0.7, outlier_thresh: float = 0.3) -> Dict:
    """Classify each embryo as core, uncertain, or outlier."""
    membership = compute_membership_scores(C, labels)
    silhouettes = silhouette_samples(D, labels, metric='precomputed')
    
    classification = {}
    for i, scores in membership.items():
        # Adaptive threshold based on bootstrap variance
        cluster = scores['cluster']
        cluster_mask = labels == cluster
        cluster_coassocs = C[cluster_mask][:, cluster_mask]
        variance = np.var(cluster_coassocs[np.triu_indices_from(cluster_coassocs, k=1)])
        
        # Adjust threshold if high variance
        adj_thresh = core_thresh - 0.1 if variance > 0.1 else core_thresh
        
        # Classify
        if scores['total_coassoc'] < outlier_thresh:
            category = 'outlier'
        elif scores['intra_coassoc'] >= adj_thresh and silhouettes[i] >= 0.2:
            category = 'core'
        else:
            category = 'uncertain'
        
        classification[i] = {
            'category': category,
            'cluster': cluster,
            'intra_coassoc': scores['intra_coassoc'],
            'silhouette': silhouettes[i],
            'threshold_used': adj_thresh
        }
    
    return classification


def get_core_indices(classification: Dict) -> np.ndarray:
    """Extract indices of core members."""
    return np.array([i for i, c in classification.items() if c['category'] == 'core'])


def get_uncertain_indices(classification: Dict) -> np.ndarray:
    """Extract indices of uncertain members."""
    return np.array([i for i, c in classification.items() if c['category'] == 'uncertain'])


def get_outlier_indices(classification: Dict) -> np.ndarray:
    """Extract indices of outliers."""
    return np.array([i for i, c in classification.items() if c['category'] == 'outlier'])


# ============ WRAPPER FUNCTIONS ============

def analyze_membership(D: np.ndarray, labels: np.ndarray, C: np.ndarray,
                       core_thresh: float = 0.7) -> Dict:
    """Full membership analysis."""
    classification = classify_members(C, labels, D, core_thresh)
    
    # Summary stats
    n_total = len(labels)
    n_core = sum(1 for c in classification.values() if c['category'] == 'core')
    n_uncertain = sum(1 for c in classification.values() if c['category'] == 'uncertain')
    n_outlier = sum(1 for c in classification.values() if c['category'] == 'outlier')
    
    # Per-cluster breakdown
    cluster_stats = {}
    for k in np.unique(labels):
        mask = labels == k
        cluster_members = [i for i in np.where(mask)[0]]
        cluster_stats[k] = {
            'total': len(cluster_members),
            'core': sum(1 for i in cluster_members if classification[i]['category'] == 'core'),
            'uncertain': sum(1 for i in cluster_members if classification[i]['category'] == 'uncertain'),
            'outlier': sum(1 for i in cluster_members if classification[i]['category'] == 'outlier')
        }
    
    return {
        'classification': classification,
        'summary': {
            'n_core': n_core,
            'n_uncertain': n_uncertain, 
            'n_outlier': n_outlier,
            'core_fraction': n_core / n_total
        },
        'cluster_stats': cluster_stats,
        'core_indices': get_core_indices(classification),
        'uncertain_indices': get_uncertain_indices(classification),
        'outlier_indices': get_outlier_indices(classification)
    }


# ============ PLOTTING FUNCTIONS (signatures only) ============

def plot_membership_distribution(classification, title="Membership Categories"):
    """Bar chart of core/uncertain/outlier counts."""
    pass

def plot_membership_scatter(D, classification, title="Membership Visualization"):
    """2D projection colored by membership category."""
    pass

def plot_cluster_breakdown(cluster_stats, title="Per-Cluster Membership"):
    """Stacked bar chart showing membership by cluster."""
    pass