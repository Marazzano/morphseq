        return X, Y, self_indices


## Hyperparameter Tuning Guide

### Key Parameters

1. **Global confidence (λ₀)**
   - Range: [0.1, 0.9]
   - Higher = more trust in tracks
   - **Start with**: 0.5
   - Adjust based on tracking quality

2. **Hard threshold (z_hard)**
   - Range: [1.0, 3.0] or None
   - Lower = more aggressive locking
   - **Start with**: None (soft mode)
   - Use 2.0 for moderately aggressive

3. **Quality threshold for calibration**
   - Range: [0.7, 0.95]
   - Higher = use only best tracks for calibration
   - **Start with**: 0.8
   - Lower if too few calibration pairs

4. **Minimum confidence**
   - Range: [0.05, 0.3]
   - Below this, don't reserve any mass
   - **Start with**: 0.1

### Tuning Strategy

```python
def analyze_reservation_behavior(
    data_subset,
    param_grid,
    m_hat,
    s_hat
):
    """
    Analyze how parameters affect mass reservation.
    """
    results = []
    
    for params in param_grid:
        # Run UOT with parameters
        uot_result = apply_lower_bounds_to_uot(
            X, Y, dt, self_indices,
            m_hat, s_hat,
            **params
        )
        
        # Analyze reservation pattern
        results.append({
            'params': params,
            'reservation_rate': uot_result['n_reserved'] / len(X),
            'mean_reserved_mass': uot_result['lower_bounds'].mean(),
            'mean_confidence': uot_result['confidence_scores'].mean(),
            'self_edge_fraction': compute_self_edge_mass_fraction(uot_result['plan'], self_indices)
        })
    
    return pd.DataFrame(results)
```

## Common Issues and Solutions

### Issue 1: Too few tracks reserved
**Symptom**: n_reserved ≈ 0 despite having tracks  
**Solution**: 
- Increase λ₀
- Check calibration (m_hat, s_hat might be off)
- Lower quality_threshold in calibration

### Issue 2: Bad tracks being locked
**Symptom**: Implausible jumps preserved  
**Solution**:
- Decrease λ₀
- Use z_hard threshold
- Improve calibration quality threshold

### Issue 3: All mass gets reserved
**Symptom**: No residual transport  
**Solution**:
- Decrease λ₀ significantly
- Check if tracks are actually good (visualize)

### Issue 4: Calibration fails
**Symptom**: No consecutive pairs found  
**Solution**:
- Check data hierarchy (use snip_ids if available)
- Increase time tolerance for "consecutive"
- Verify embryo_ids are correct

## Testing Suite

```python
# File: tests/test_self_edge_bounds.py

import numpy as np
from src.flux.transport.trajectory_calibration import (
    calibrate_speed_statistics,
    compute_track_confidence,
    compute_lower_bounds,
    apply_lower_bounds_to_uot
)

def test_calibration_on_synthetic():
    """Test calibration on synthetic data with known speeds."""
    # Create synthetic tracks with known speed
    n_embryos = 50
    n_timepoints = 10
    true_speed = 1.0
    
    embeddings_list = []
    times_list = []
    embryo_ids_list = []
    
    for emb_id in range(n_embryos):
        # Random starting point
        pos = np.random.randn(5)
        
        for t in range(n_timepoints):
            # Move with constant speed + noise
            embeddings_list.append(pos)
            times_list.append(float(t))
            embryo_ids_list.append(emb_id)
            
            # Update position
            direction = np.random.randn(5)
            direction /= np.linalg.norm(direction)
            pos = pos + direction * true_speed + np.random.randn(5) * 0.1
    
    embeddings = np.array(embeddings_list)
    times = np.array(times_list)
    embryo_ids = np.array(embryo_ids_list)
    
    # Calibrate
    m_hat, s_hat = calibrate_speed_statistics(
        embeddings, times, embryo_ids, verbose=True
    )
    
    # Check calibration is close to true speed
    assert abs(m_hat - true_speed) < 0.2, f"Calibration off: {m_hat} vs {true_speed}"
    assert s_hat < 0.5, f"Scale too large: {s_hat}"

def test_confidence_function():
    """Test confidence scoring for different speeds."""
    m_hat, s_hat = 1.0, 0.2
    
    # Typical speed → high confidence
    x = np.zeros(5)
    y = np.ones(5) * 1.0  # displacement = 1.0
    dt = 1.0  # speed = 1.0 = median
    
    confidence = compute_track_confidence(x, y, dt, m_hat, s_hat, mode="soft")
    assert confidence > 0.9, f"Typical speed should have high confidence: {confidence}"
    
    # Fast speed → low confidence
    y_fast = np.ones(5) * 3.0  # displacement = 3.0, speed = 3.0
    confidence_fast = compute_track_confidence(x, y_fast, dt, m_hat, s_hat, mode="soft")
    assert confidence_fast < 0.1, f"Fast speed should have low confidence: {confidence_fast}"

def test_lower_bounds():
    """Test lower bound computation."""
    # Create simple scenario
    X = np.array([[0, 0], [1, 0], [2, 0]])
    Y = np.array([[0, 1], [1, 1], [5, 1]])  # Last one is far
    self_indices = np.array([0, 1, 2])
    
    a = np.ones(3) / 3
    b = np.ones(3) / 3
    
    m_hat, s_hat = 1.0, 0.2
    
    lower_bounds = compute_lower_bounds(
        X, Y, self_indices, dt=1.0,
        a=a, b=b,
        m_hat=m_hat, s_hat=s_hat,
        lambda0=0.5
    )
    
    # First two should have substantial reservation
    assert lower_bounds[0] > 0.1
    assert lower_bounds[1] > 0.1
    # Last one should have very little (too fast)
    assert lower_bounds[2] < 0.01

def test_full_pipeline():
    """Test complete UOT with lower bounds."""
    # Create tracked data
    X = np.random.randn(20, 5)
    Y = X + np.random.randn(20, 5) * 0.1  # Small displacement
    self_indices = np.arange(20)  # Perfect tracking
    
    # Calibrate on same data (for testing)
    embeddings = np.vstack([X, Y])
    times = np.concatenate([np.zeros(20), np.ones(20)])
    embryo_ids = np.concatenate([np.arange(20), np.arange(20)])
    
    m_hat, s_hat = calibrate_speed_statistics(
        embeddings, times, embryo_ids
    )
    
    # Run UOT with lower bounds
    result = apply_lower_bounds_to_uot(
        X, Y, dt=1.0,
        self_indices=self_indices,
        m_hat=m_hat, s_hat=s_hat,
        lambda0=0.5,
        verbose=True
    )
    
    # Check that most mass is on diagonal
    Pi = result['plan']
    diag_mass = np.diag(Pi).sum()
    total_mass = Pi.sum()
    
    assert diag_mass / total_mass > 0.7, f"Not enough mass on diagonal: {diag_mass/total_mass:.2%}"
    assert result['n_reserved'] >= 15, f"Too few reservations: {result['n_reserved']}/20"

def test_missing_tracks():
    """Test handling of missing tracks."""
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 5)
    self_indices = np.full(10, -1)  # No tracks
    
    m_hat, s_hat = 1.0, 0.2
    
    result = apply_lower_bounds_to_uot(
        X, Y, dt=1.0,
        self_indices=self_indices,
        m_hat=m_hat, s_hat=s_hat,
        lambda0=0.5
    )
    
    # Should fall back to pure UOT
    assert result['n_reserved'] == 0
    assert np.allclose(result['fixed_plan'], 0)
    
def test_mixed_quality_tracks():
    """Test with mixture of good and bad tracks."""
    n = 30
    X = np.random.randn(n, 5)
    Y = np.zeros((n, 5))
    self_indices = np.arange(n)
    
    # First 10: good tracks (small displacement)
    Y[:10] = X[:10] + np.random.randn(10, 5) * 0.1
    
    # Next 10: bad tracks (huge displacement)
    Y[10:20] = X[10:20] + np.random.randn(10, 5) * 5.0
    
    # Last 10: moderate tracks
    Y[20:] = X[20:] + np.random.randn(10, 5) * 1.0
    
    # Calibrate on synthetic good data
    m_hat, s_hat = 0.5, 0.2
    
    result = apply_lower_bounds_to_uot(
        X, Y, dt=1.0,
        self_indices=self_indices,
        m_hat=m_hat, s_hat=s_hat,
        lambda0=0.5
    )
    
    # Check selective reservation
    lower_bounds = result['lower_bounds']
    assert lower_bounds[:10].mean() > 0.1, "Good tracks should have high bounds"
    assert lower_bounds[10:20].mean() < 0.01, "Bad tracks should have low bounds"
    assert 0.01 < lower_bounds[20:].mean() < 0.1, "Moderate tracks should be in between"


## Integration Examples

### Complete Training Script

```python
# File: scripts/train_ode_with_calibrated_uot.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import all components
from src.flux.utils.spectral_time_clustering import (
    time_weighted_spectral_clustering,
    ClusteredMinibatchSampler
)
from src.flux.transport.minibatch_uot import compute_biased_uot
from src.flux.transport.trajectory_calibration import (
    calibrate_speed_statistics,
    apply_lower_bounds_to_uot
)

def main():
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv('data/embryo_data.csv')
    
    # 2. Spectral clustering
    print("Running spectral clustering...")
    labels = time_weighted_spectral_clustering(
        df,
        n_clusters=100,
        sigma_space=2.0,
        sigma_time=1.5,
        knn_sparsify=200,
        verbose=True
    )
    
    # 3. Calibrate speed statistics
    print("Calibrating speed statistics...")
    feature_cols = [col for col in df.columns if col.startswith('bc_') or col.startswith('z_')]
    embeddings = df[feature_cols].values
    times = df['predicted_stage_hpf'].values
    embryo_ids = df['embryo_id'].values
    
    m_hat, s_hat = calibrate_speed_statistics(
        embeddings, times, embryo_ids,
        quality_threshold=0.8,
        verbose=True
    )
    
    print(f"Calibration complete: m_hat={m_hat:.3f}, s_hat={s_hat:.3f}")
    
    # 4. Create minibatch sampler
    sampler = ClusteredMinibatchSampler(
        embeddings=embeddings,
        times=times,
        embryo_ids=embryo_ids,
        labels=labels,
        batch_size=256,
        time_gap_range=(0.5, 2.0)
    )
    
    # 5. Initialize model and optimizer
    from src.flux.models.ode import NeuralODE  # Your ODE model
    
    model = NeuralODE(
        input_dim=embeddings.shape[1],
        hidden_dim=128,
        time_encoding_dim=32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 6. Training loop
    print("Starting training...")
    for epoch in range(100):
        epoch_loss = 0
        n_batches = 0
        n_reserved_total = 0
        
        for _ in range(50):  # 50 batches per epoch
            # Sample minibatch
            X, Y, self_indices, dt = sampler.sample_batch()
            
            if len(X) == 0 or len(Y) == 0:
                continue
            
            # Apply calibrated UOT
            uot_result = apply_lower_bounds_to_uot(
                X, Y, dt, self_indices,
                m_hat, s_hat,
                eps=0.05,
                reg_m=1.0,
                lambda0=0.5,
                z_hard=None
            )
            
            n_reserved_total += uot_result['n_reserved']
            
            # Convert to torch
            X_torch = torch.from_numpy(X).float()
            t_torch = torch.full((len(X),), times[0])
            
            # Forward pass
            pred_velocities = model(X_torch, t_torch)
            
            # Compute loss
            row_masses = uot_result['row_masses']
            mask = row_masses > 0.1
            
            if mask.sum() > 0:
                target_velocities = torch.from_numpy(
                    uot_result['velocities'][mask]
                ).float()
                pred_masked = pred_velocities[mask]
                weights = torch.from_numpy(row_masses[mask]).float()
                
                loss = (weights * (pred_masked - target_velocities).pow(2).sum(1)).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
        
        # Report
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_reserved = n_reserved_total / n_batches
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Avg reserved={avg_reserved:.1f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'calibration': {'m_hat': m_hat, 's_hat': s_hat}
            }, f'checkpoints/ode_epoch_{epoch}.pt')

if __name__ == "__main__":
    main()
```

### Visualization and Analysis

```python
def visualize_reservation_patterns(uot_result, self_indices):
    """
    Visualize how mass reservation affects transport plans.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fixed plan (reserved mass)
    ax = axes[0]
    im = ax.imshow(uot_result['fixed_plan'], cmap='Blues', aspect='auto')
    ax.set_title('Reserved Mass (Self-edges)')
    ax.set_xlabel('Target index')
    ax.set_ylabel('Source index')
    plt.colorbar(im, ax=ax)
    
    # Residual plan
    ax = axes[1]
    im = ax.imshow(uot_result['residual_plan'], cmap='Reds', aspect='auto')
    ax.set_title('Residual Transport')
    ax.set_xlabel('Target index')
    ax.set_ylabel('Source index')
    plt.colorbar(im, ax=ax)
    
    # Total plan
    ax = axes[2]
    im = ax.imshow(uot_result['plan'], cmap='hot', aspect='auto')
    ax.set_title('Total Transport Plan')
    ax.set_xlabel('Target index')
    ax.set_ylabel('Source index')
    plt.colorbar(im, ax=ax)
    
    # Mark tracked edges
    for i, j in enumerate(self_indices):
        if j >= 0:
            for ax in axes:
                ax.plot(j, i, 'g*', markersize=5)
    
    plt.tight_layout()
    plt.show()

def analyze_speed_distribution(embeddings, times, embryo_ids, m_hat, s_hat):
    """
    Analyze the distribution of speeds and calibration quality.
    """
    import matplotlib.pyplot as plt
    
    # Compute all speeds
    speeds = []
    z_scores = []
    
    unique_embryos = np.unique(embryo_ids)
    for emb_id in unique_embryos:
        mask = embryo_ids == emb_id
        if mask.sum() < 2:
            continue
            
        emb_embeddings = embeddings[mask]
        emb_times = times[mask]
        
        # Sort by time
        order = np.argsort(emb_times)
        emb_embeddings = emb_embeddings[order]
        emb_times = emb_times[order]
        
        for i in range(len(emb_times) - 1):
            dt = emb_times[i+1] - emb_times[i]
            if dt <= 0 or dt > 2.0:
                continue
                
            displacement = np.linalg.norm(emb_embeddings[i+1] - emb_embeddings[i])
            speed = displacement / dt
            speeds.append(speed)
            z_scores.append((speed - m_hat) / s_hat)
    
    speeds = np.array(speeds)
    z_scores = np.array(z_scores)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Speed distribution
    ax = axes[0]
    ax.hist(speeds, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(m_hat, color='red', linestyle='--', label=f'Median={m_hat:.3f}')
    ax.axvline(m_hat - s_hat, color='orange', linestyle=':', label=f'±1 std')
    ax.axvline(m_hat + s_hat, color='orange', linestyle=':')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Count')
    ax.set_title('Speed Distribution')
    ax.legend()
    
    # Z-score distribution
    ax = axes[1]
    ax.hist(z_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.axvline(-2, color='orange', linestyle=':', label='±2 std')
    ax.axvline(2, color='orange', linestyle=':')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Count')
    ax.set_title('Normalized Speed Distribution')
    ax.legend()
    
    # Confidence function
    ax = axes[2]
    z_range = np.linspace(-3, 5, 100)
    confidence = np.exp(-0.5 * np.maximum(z_range, 0)**2)
    ax.plot(z_range, confidence, linewidth=2)
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Confidence φ(z)')
    ax.set_title('Confidence Function')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Summary

This implementation provides:

1. **Robust calibration** of speed statistics from high-quality tracks
2. **Adaptive confidence** scoring based on displacement plausibility
3. **Flexible reservation** modes (soft exponential decay or hard threshold)
4. **Graceful degradation** when tracks are missing or unreliable
5. **Full integration** with the minibatch UOT pipeline

The key insight is that we use the tracked data when it's good, but don't blindly trust it when embryos appear to teleport. The calibration ensures we adapt to the actual data quality rather than using fixed thresholds.

## Next Steps

1. **Run calibration** on your actual data to get m_hat and s_hat
2. **Visualize** speed distributions to verify calibration quality
3. **Test** different λ₀ values to find optimal reservation strength
4. **Compare** ODE training with and without self-edge bounds
5. **Monitor** reservation rates during training as a diagnostic# Task 03: Self-Edge Lower Bound Implementation

## Overview
Implement intelligent biasing of UOT toward known embryo trajectories using calibrated lower bounds based on displacement speed statistics.

## Motivation
With sparse, noisy tracking data, we need to:
1. **Preserve good tracks**: When an embryo is reliably tracked, use that information
2. **Reject bad tracks**: When displacement is implausible, let UOT find better matches
3. **Handle missing tracks**: Gracefully fall back to pure UOT when no tracking exists

## Mathematical Foundation

### Calibration Heuristic
For each tracked edge (x_i → y_{s(i)}):

1. **Compute normalized speed**:
   ```
   d_i = ||y_{s(i)} - x_i||  (displacement)
   v_i = d_i / Δt_i          (speed)
   z_i = (v_i - m̂) / ŝ      (z-score)
   ```

2. **Confidence function**:
   ```
   φ(z_i) = exp(-0.5 * max(z_i, 0)²)
   ```
   - φ ≈ 1 when speed is typical or slower
   - φ → 0 when speed is unusually fast

3. **Lower bound**:
   ```
   l_i = min(a_i, b_{s(i)}) * λ_0 * φ(z_i)
   ```
   - λ_0 ∈ (0,1]: global confidence factor
   - min(a_i, b_{s(i)}): can't reserve more than available mass

## Implementation Guide

### Step 1: Speed Statistics Calibration

```python
# File: src/flux/transport/trajectory_calibration.py

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats

def calibrate_speed_statistics(
    embeddings: np.ndarray,
    times: np.ndarray,
    embryo_ids: np.ndarray,
    experiment_ids: Optional[np.ndarray] = None,
    snip_ids: Optional[np.ndarray] = None,
    quality_threshold: float = 0.8,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Compute robust speed statistics from high-quality tracked pairs.
    
    This function identifies consecutive frames of the same embryo and
    computes their displacement speeds to establish a baseline for
    what constitutes "normal" developmental motion.
    
    Parameters
    ----------
    embeddings : np.ndarray (n, d)
        Feature vectors (VAE latents or biological components)
    times : np.ndarray (n,)
        Time values (predicted_stage_hpf)
    embryo_ids : np.ndarray (n,)
        Embryo identifiers
    experiment_ids : np.ndarray (n,), optional
        For filtering to specific experiments (good quality ones)
    snip_ids : np.ndarray (n,), optional
        For hierarchical filtering (see parsing utils)
    quality_threshold : float
        Percentile of speeds to keep for robust estimation
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    m_hat : float
        Median speed (robust location estimate)
    s_hat : float
        Robust scale estimate (MAD-based standard deviation)
        
    Notes
    -----
    The hierarchical data structure is:
    experiment_id → embryo_id → snip_id → image_id
    
    We compute speeds at the snip_id level if available, otherwise
    at the embryo_id level.
    """
    speeds = []
    
    # Group by embryo (and snip if available)
    unique_embryos = np.unique(embryo_ids)
    
    for emb_id in unique_embryos:
        mask = embryo_ids == emb_id
        
        if mask.sum() < 2:
            continue  # Need at least 2 points
        
        emb_embeddings = embeddings[mask]
        emb_times = times[mask]
        
        # Sort by time
        time_order = np.argsort(emb_times)
        emb_embeddings = emb_embeddings[time_order]
        emb_times = emb_times[time_order]
        
        # Compute speeds between consecutive frames
        for i in range(len(emb_times) - 1):
            dt = emb_times[i+1] - emb_times[i]
            
            if dt <= 0:
                continue  # Skip if times are not increasing
            
            # Check if this is truly consecutive (not a big gap)
            if dt > 2.0:  # Arbitrary threshold for "consecutive"
                continue
                
            displacement = np.linalg.norm(emb_embeddings[i+1] - emb_embeddings[i])
            speed = displacement / dt
            speeds.append(speed)
    
    if len(speeds) == 0:
        raise ValueError("No valid consecutive pairs found for calibration")
    
    speeds = np.array(speeds)
    
    if verbose:
        print(f"Calibration using {len(speeds)} tracked pairs")
        print(f"Speed range: [{speeds.min():.3f}, {speeds.max():.3f}]")
    
    # Filter outliers using percentile
    if quality_threshold < 1.0:
        cutoff = np.percentile(speeds, quality_threshold * 100)
        speeds_filtered = speeds[speeds <= cutoff]
        if verbose:
            print(f"Filtered to {len(speeds_filtered)} pairs (removed {len(speeds)-len(speeds_filtered)} outliers)")
    else:
        speeds_filtered = speeds
    
    # Compute robust statistics
    m_hat = np.median(speeds_filtered)
    mad = np.median(np.abs(speeds_filtered - m_hat))
    s_hat = mad / 0.6745  # Convert MAD to standard deviation estimate
    
    if verbose:
        print(f"Calibrated statistics: median={m_hat:.3f}, robust_std={s_hat:.3f}")
    
    return m_hat, s_hat


def compute_track_confidence(
    x_i: np.ndarray,
    y_si: np.ndarray,
    dt_i: float,
    m_hat: float,
    s_hat: float,
    mode: str = "soft"
) -> float:
    """
    Compute confidence score for a tracked edge.
    
    Parameters
    ----------
    x_i : np.ndarray (d,)
        Source embedding
    y_si : np.ndarray (d,)
        Tracked target embedding
    dt_i : float
        Time gap
    m_hat : float
        Median speed from calibration
    s_hat : float
        Robust scale from calibration
    mode : str
        "soft": smooth confidence decay
        "hard": binary threshold
        
    Returns
    -------
    float
        Confidence score in [0, 1]
    """
    # Compute speed
    d_i = np.linalg.norm(y_si - x_i)
    v_i = d_i / max(dt_i, 1e-8)
    
    # Normalize
    z_i = (v_i - m_hat) / max(s_hat, 1e-8)
    
    if mode == "soft":
        # Smooth exponential decay for outliers
        phi = np.exp(-0.5 * max(z_i, 0)**2)
    elif mode == "hard":
        # Binary: trust if within 2 standard deviations
        phi = 1.0 if z_i <= 2.0 else 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return phi


### Step 2: Lower Bound Computation

```python
def compute_lower_bounds(
    X: np.ndarray,
    Y: np.ndarray,
    self_indices: np.ndarray,
    dt: float,
    a: np.ndarray,
    b: np.ndarray,
    m_hat: float,
    s_hat: float,
    lambda0: float = 0.5,
    z_hard: Optional[float] = None,
    min_confidence: float = 0.1
) -> np.ndarray:
    """
    Compute lower bounds for self-edges based on track quality.
    
    Parameters
    ----------
    X : np.ndarray (m, d)
        Source embeddings
    Y : np.ndarray (n, d)
        Target embeddings
    self_indices : np.ndarray (m,)
        Tracking indices (-1 for no track)
    dt : float
        Time gap
    a : np.ndarray (m,)
        Source mass distribution
    b : np.ndarray (n,)
        Target mass distribution
    m_hat : float
        Median speed from calibration
    s_hat : float
        Robust scale from calibration
    lambda0 : float
        Global confidence factor (0 = no reservation, 1 = full reservation)
    z_hard : float, optional
        If provided, use hard threshold: full reservation if z <= z_hard
    min_confidence : float
        Minimum confidence to reserve any mass
        
    Returns
    -------
    np.ndarray (m,)
        Lower bounds l_i for each source point
    """
    m = len(X)
    lower_bounds = np.zeros(m)
    
    for i in range(m):
        j = self_indices[i]
        
        if j < 0 or j >= len(Y):
            continue  # No valid track
        
        # Compute displacement and speed
        d_i = np.linalg.norm(Y[j] - X[i])
        v_i = d_i / max(dt, 1e-8)
        z_i = (v_i - m_hat) / max(s_hat, 1e-8)
        
        # Compute confidence
        if z_hard is not None and z_i <= z_hard:
            # Hard mode: full reservation for good tracks
            phi = 1.0
        else:
            # Soft mode: exponential decay
            phi = np.exp(-0.5 * max(z_i, 0)**2)
        
        # Skip if confidence too low
        if phi < min_confidence:
            continue
        
        # Compute lower bound
        max_reservable = min(a[i], b[j])
        lower_bounds[i] = max_reservable * lambda0 * phi
    
    return lower_bounds


def apply_lower_bounds_to_uot(
    X: np.ndarray,
    Y: np.ndarray,
    dt: float,
    self_indices: np.ndarray,
    m_hat: float,
    s_hat: float,
    eps: float = 0.05,
    reg_m: float = 1.0,
    lambda0: float = 0.5,
    z_hard: Optional[float] = None,
    verbose: bool = False
) -> Dict:
    """
    Complete UOT with self-edge lower bounds.
    
    This is the main entry point that combines:
    1. Lower bound computation based on track quality
    2. Mass reservation on self-edges
    3. Residual UOT for remaining mass
    
    Parameters
    ----------
    X : np.ndarray (m, d)
        Source embeddings
    Y : np.ndarray (n, d)
        Target embeddings
    dt : float
        Time gap
    self_indices : np.ndarray (m,)
        Tracking indices
    m_hat : float
        Calibrated median speed
    s_hat : float
        Calibrated speed scale
    eps : float
        Entropic regularization
    reg_m : float
        Mass penalty (unbalanced regularization)
    lambda0 : float
        Global reservation strength
    z_hard : float, optional
        Hard threshold for z-scores
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    dict
        Same as compute_biased_uot, plus:
        - 'lower_bounds': array of computed lower bounds
        - 'confidence_scores': array of track confidences
        - 'n_reserved': number of edges with reservations
    """
    import ot
    
    m, n = len(X), len(Y)
    
    # Default uniform masses
    a = np.ones(m) / m
    b = np.ones(n) / n
    
    # Compute lower bounds
    lower_bounds = compute_lower_bounds(
        X, Y, self_indices, dt, a, b,
        m_hat, s_hat, lambda0, z_hard
    )
    
    # Reserve mass on self-edges
    Pi_fixed = np.zeros((m, n))
    a_residual = a.copy()
    b_residual = b.copy()
    confidence_scores = np.zeros(m)
    
    n_reserved = 0
    total_reserved_mass = 0
    
    for i in range(m):
        j = self_indices[i]
        
        if j >= 0 and j < n and lower_bounds[i] > 0:
            # Reserve mass
            mass_to_fix = lower_bounds[i]
            Pi_fixed[i, j] = mass_to_fix
            a_residual[i] -= mass_to_fix
            b_residual[j] -= mass_to_fix
            
            # Track statistics
            n_reserved += 1
            total_reserved_mass += mass_to_fix
            
            # Store confidence
            d_i = np.linalg.norm(Y[j] - X[i])
            v_i = d_i / max(dt, 1e-8)
            z_i = (v_i - m_hat) / max(s_hat, 1e-8)
            confidence_scores[i] = np.exp(-0.5 * max(z_i, 0)**2)
    
    if verbose:
        print(f"Reserved mass on {n_reserved}/{m} edges")
        print(f"Total reserved mass: {total_reserved_mass:.3f} ({total_reserved_mass/a.sum():.1%} of total)")
        if n_reserved > 0:
            print(f"Mean confidence: {confidence_scores[confidence_scores > 0].mean():.3f}")
    
    # Solve UOT for residual mass
    C = ot.dist(X, Y) ** 2
    
    # Check if there's any residual mass to transport
    if a_residual.sum() > 1e-10 and b_residual.sum() > 1e-10:
        Pi_residual = ot.unbalanced.sinkhorn_unbalanced(
            a_residual, b_residual, C, eps, reg_m
        )
    else:
        Pi_residual = np.zeros((m, n))
        if verbose:
            print("Warning: No residual mass to transport")
    
    # Combine fixed and residual
    Pi = Pi_fixed + Pi_residual
    
    # Extract velocities
    row_masses = Pi.sum(axis=1)
    velocities = np.zeros_like(X)
    
    for i in range(m):
        if row_masses[i] > 1e-10:
            weighted_displacement = Pi[i, :] @ (Y - X[i])
            velocities[i] = weighted_displacement / (dt * row_masses[i])
    
    return {
        'plan': Pi,
        'velocities': velocities,
        'row_masses': row_masses,
        'cost': np.sum(Pi * C),
        'lower_bounds': lower_bounds,
        'confidence_scores': confidence_scores,
        'n_reserved': n_reserved,
        'fixed_plan': Pi_fixed,
        'residual_plan': Pi_residual
    }


### Step 3: Integration with Training Pipeline

```python
class CalibratedUOTTrainer:
    """
    Trainer that uses calibrated self-edge bounds in UOT.
    """
    
    def __init__(
        self,
        model,  # Neural ODE model
        data_loader,
        optimizer,
        calibration_data: Optional[Dict] = None,
        uot_config: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize trainer with calibration.
        
        Parameters
        ----------
        model : torch.nn.Module
            Neural ODE model
        data_loader : DataLoader
            Provides (embeddings, times, embryo_ids)
        optimizer : torch.optim.Optimizer
            Optimizer for model
        calibration_data : dict, optional
            Pre-computed calibration statistics
        uot_config : dict, optional
            UOT hyperparameters
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        
        # Default UOT config
        self.uot_config = uot_config or {
            'eps': 0.05,
            'reg_m': 1.0,
            'lambda0': 0.5,
            'z_hard': None,
            'batch_size': 256,
            'dt_range': (0.5, 2.0),
            'min_row_mass': 0.1
        }
        
        # Calibration statistics
        if calibration_data:
            self.m_hat = calibration_data['m_hat']
            self.s_hat = calibration_data['s_hat']
            self.calibrated = True
        else:
            self.m_hat = None
            self.s_hat = None
            self.calibrated = False
    
    def calibrate(self, embeddings, times, embryo_ids):
        """
        Calibrate speed statistics from data.
        """
        print("Calibrating speed statistics...")
        self.m_hat, self.s_hat = calibrate_speed_statistics(
            embeddings, times, embryo_ids, verbose=True
        )
        self.calibrated = True
        
        # Save calibration for reuse
        return {'m_hat': self.m_hat, 's_hat': self.s_hat}
    
    def train_epoch(self):
        """
        Train one epoch with calibrated UOT.
        """
        import torch
        
        if not self.calibrated:
            raise ValueError("Must calibrate before training")
        
        epoch_loss = 0
        num_batches = 0
        total_reserved = 0
        total_tracked = 0
        
        for batch_data in self.data_loader:
            embeddings, times, embryo_ids = batch_data
            
            # Sample time gap
            dt = np.random.uniform(*self.uot_config['dt_range'])
            
            # Get minibatch pair (using clustered sampler from Task 02)
            X, Y, self_indices = self._sample_minibatch(
                embeddings, times, embryo_ids, dt
            )
            
            if len(X) == 0 or len(Y) == 0:
                continue
            
            # Track statistics
            n_tracked = (self_indices >= 0).sum()
            total_tracked += n_tracked
            
            # Compute UOT with calibrated lower bounds
            uot_result = apply_lower_bounds_to_uot(
                X, Y, dt, self_indices,
                self.m_hat, self.s_hat,
                eps=self.uot_config['eps'],
                reg_m=self.uot_config['reg_m'],
                lambda0=self.uot_config['lambda0'],
                z_hard=self.uot_config.get('z_hard'),
                verbose=False
            )
            
            total_reserved += uot_result['n_reserved']
            
            # Convert to torch tensors
            X_torch = torch.from_numpy(X).float().to(self.device)
            t_torch = torch.full((len(X),), times[0]).to(self.device)
            
            # Get ODE predictions
            with torch.enable_grad():
                pred_velocities = self.model(X_torch, t_torch)
            
            # Compute loss (mass-weighted MSE)
            row_masses = uot_result['row_masses']
            mask = row_masses > self.uot_config['min_row_mass']
            
            if mask.sum() > 0:
                target_velocities = torch.from_numpy(
                    uot_result['velocities'][mask]
                ).float().to(self.device)
                pred_velocities_masked = pred_velocities[mask]
                weights = torch.from_numpy(row_masses[mask]).float().to(self.device)
                
                # Weighted MSE loss
                loss = (weights * (pred_velocities_masked - target_velocities).pow(2).sum(1)).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        # Report statistics
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            reservation_rate = total_reserved / max(total_tracked, 1)
            print(f"Loss: {avg_loss:.4f} | "
                  f"Reservation rate: {reservation_rate:.2%} ({total_reserved}/{total_tracked})")
            return avg_loss
        else:
            print("Warning: No valid batches in epoch")
            return float('inf')
    
    def _sample_minibatch(self, embeddings, times, embryo_ids, dt):
        """
        Sample minibatch pair (placeholder - use ClusteredMinibatchSampler from Task 02).
        """
        # This is simplified - in practice use the clustered sampler
        n = len(embeddings)
        source_mask = times < times.max() - dt
        target_mask = times > times.min() + dt
        
        source_idx = np.where(source_mask)[0]
        target_idx = np.where(target_mask)[0]
        
        # Sample
        if len(source_idx) > self.uot_config['batch_size']:
            source_idx = np.random.choice(source_idx, self.uot_config['batch_size'], replace=False)
        if len(target_idx) > self.uot_config['batch_size']:
            target_idx = np.random.choice(target_idx, self.uot_config['batch_size'], replace=False)
        
        X = embeddings[source_idx]
        Y = embeddings[target_idx]
        
        # Build tracking
        self_indices = np.full(len(source_idx), -1)
        for i, emb_id in enumerate(embryo_ids[source_idx]):
            matches = np.where(embryo_ids[target_idx] == emb_id)[0]
            if len(matches) > 0:
                self_indices[i] = matches[0]
        
        return X, Y, self_indices