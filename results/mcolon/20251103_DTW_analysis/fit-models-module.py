# 4_fit_models.py
"""Fit mixed-effects models to trajectory clusters."""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from typing import Dict, Tuple, List
import warnings

# ============ CORE FUNCTIONS ============

def fit_spline(t: np.ndarray, y: np.ndarray, smoothing: float = 1.0, 
               k: int = 3) -> UnivariateSpline:
    """Fit penalized spline to trajectory."""
    # Sort by time
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]
    
    # Fit spline
    spline = UnivariateSpline(t_sorted, y_sorted, s=smoothing, k=k)
    return spline


def compute_dba_average(trajectories: List[Tuple], max_iter: int = 10):
    """
    Simplified DBA - compute average aligned trajectory.
    Assumes you have DTW alignment paths available.
    """
    # This is a placeholder - real DBA needs DTW alignment paths
    # For now, return simple average
    t_common = trajectories[0][0]  # Use first trajectory's time
    y_avg = np.mean([np.interp(t_common, t, y) for t, y in trajectories], axis=0)
    return t_common, y_avg


def fit_random_effects(t: np.ndarray, y: np.ndarray, 
                      mean_curve: callable) -> Dict:
    """Estimate random intercept and slope for individual."""
    # Compute residuals from mean
    y_mean = mean_curve(t)
    residuals = y - y_mean
    
    # Simple linear regression on residuals
    X = np.column_stack([np.ones_like(t), t])
    coeffs = np.linalg.lstsq(X, residuals, rcond=None)[0]
    
    b0 = coeffs[0]  # Random intercept
    b1 = coeffs[1]  # Random slope
    
    # Residual variance
    y_pred = y_mean + b0 + b1 * t
    sigma2 = np.var(y - y_pred)
    
    return {'b0': b0, 'b1': b1, 'sigma2': sigma2}


def estimate_variance_components(random_effects: List[Dict]) -> Dict:
    """Estimate variance-covariance of random effects."""
    b0_vals = [re['b0'] for re in random_effects]
    b1_vals = [re['b1'] for re in random_effects]
    
    var_b0 = np.var(b0_vals)
    var_b1 = np.var(b1_vals)
    cov_b0_b1 = np.cov(b0_vals, b1_vals)[0, 1]
    
    # Pooled residual variance
    sigma2_pooled = np.mean([re['sigma2'] for re in random_effects])
    
    return {
        'var_b0': var_b0,
        'var_b1': var_b1,
        'cov_b0_b1': cov_b0_b1,
        'sigma2': sigma2_pooled,
        'cov_matrix': np.array([[var_b0, cov_b0_b1], 
                                [cov_b0_b1, var_b1]])
    }


# ============ WRAPPER FUNCTIONS ============

def fit_cluster_model(trajectories: List[Tuple], core_mask: np.ndarray = None,
                     use_dba: bool = False) -> Dict:
    """
    Fit mixed-effects model to a cluster.
    
    Args:
        trajectories: List of (t, y) tuples
        core_mask: Boolean mask for core members
        use_dba: Whether to compute DBA average
    """
    if core_mask is not None:
        core_trajectories = [traj for i, traj in enumerate(trajectories) if core_mask[i]]
    else:
        core_trajectories = trajectories
    
    # Center time
    all_t = np.concatenate([t for t, _ in core_trajectories])
    t_mean = np.mean(all_t)
    trajectories_centered = [(t - t_mean, y) for t, y in trajectories]
    core_centered = [(t - t_mean, y) for t, y in core_trajectories]
    
    # Fit mean curve (two methods)
    # Method 1: Spline on pooled data
    t_pooled = np.concatenate([t for t, _ in core_centered])
    y_pooled = np.concatenate([y for _, y in core_centered])
    mean_spline = fit_spline(t_pooled, y_pooled, smoothing=10.0)
    
    # Method 2: DBA (if requested)
    if use_dba:
        t_dba, y_dba = compute_dba_average(core_centered)
        dba_spline = fit_spline(t_dba, y_dba, smoothing=1.0)
    else:
        dba_spline = None
    
    # Fit random effects for all members
    random_effects = []
    for t, y in trajectories_centered:
        re = fit_random_effects(t, y, mean_spline)
        random_effects.append(re)
    
    # Estimate variance components
    var_components = estimate_variance_components(random_effects)
    
    # Compute fit statistics
    r2_values = []
    for (t, y), re in zip(trajectories_centered, random_effects):
        y_pred = mean_spline(t) + re['b0'] + re['b1'] * t
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        r2_values.append(r2)
    
    return {
        'mean_spline': mean_spline,
        'dba_spline': dba_spline,
        't_center': t_mean,
        'random_effects': random_effects,
        'variance_components': var_components,
        'r2_values': np.array(r2_values),
        'mean_r2': np.mean(r2_values)
    }


def predict_trajectory(t_new: np.ndarray, cluster_model: Dict, 
                      b0: float = 0, b1: float = 0) -> np.ndarray:
    """Predict trajectory from fitted model."""
    t_centered = t_new - cluster_model['t_center']
    y_pred = cluster_model['mean_spline'](t_centered) + b0 + b1 * t_centered
    return y_pred


def compute_confidence_bands(t: np.ndarray, cluster_model: Dict, 
                            alpha: float = 0.05) -> Tuple:
    """Compute confidence bands for mean curve."""
    t_centered = t - cluster_model['t_center']
    y_mean = cluster_model['mean_spline'](t_centered)
    
    # Standard error from variance components
    var_b0 = cluster_model['variance_components']['var_b0']
    var_b1 = cluster_model['variance_components']['var_b1']
    cov = cluster_model['variance_components']['cov_b0_b1']
    sigma2 = cluster_model['variance_components']['sigma2']
    
    # Variance at each time point
    var_t = var_b0 + 2*cov*t_centered + var_b1*t_centered**2 + sigma2
    se = np.sqrt(var_t)
    
    # Normal approximation
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    
    lower = y_mean - z * se
    upper = y_mean + z * se
    
    return y_mean, lower, upper


# ============ MODEL COMPARISON ============

def compare_spline_dba(cluster_model: Dict, t_eval: np.ndarray) -> float:
    """Compare spline and DBA mean curves."""
    if cluster_model['dba_spline'] is None:
        return np.nan
    
    t_centered = t_eval - cluster_model['t_center']
    y_spline = cluster_model['mean_spline'](t_centered)
    y_dba = cluster_model['dba_spline'](t_centered)
    
    rmse = np.sqrt(np.mean((y_spline - y_dba)**2))
    return rmse


# ============ PLOTTING FUNCTIONS (signatures only) ============

def plot_cluster_fit(t, trajectories, cluster_model, title="Cluster Fit"):
    """Plot trajectories with fitted mean and confidence bands."""
    pass

def plot_random_effects(random_effects, title="Random Effects Distribution"):
    """Scatter plot of random intercepts vs slopes."""
    pass

def plot_residuals(t, y, cluster_model, random_effect, title="Residual Analysis"):
    """Q-Q plot and residuals over time."""
    pass

def plot_spline_vs_dba(t, cluster_model, title="Spline vs DBA"):
    """Compare two mean curve methods."""
    pass