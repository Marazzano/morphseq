# Task 04 — `lower_bound` Biasing in `compute_minibatch_uot`

**Reference:** `self_edge_bounds.md`

## Intent
Encourage self/near‑self transport mass via analytic lower bounds using statistics `(m_hat, s_hat)` estimated from local neighborhoods.

## API Changes
```python
def compute_minibatch_uot(..., bias_mode: str = "none", bias_kwargs: dict | None = None):
    # when bias_mode == "lower_bound": expect keys in bias_kwargs:
    #  - m_hat: (B,) or (B,B) lower bound means
    #  - s_hat: (B,) or (B,B) dispersion terms
    #  - strength: float in [0,1] blending weight
```

## Implementation Sketch
1. Compute unbiased `C` and corresponding Gibbs kernel `K = exp(-C/eps)`.
2. Construct lower‑bound matrix `L` from `(m_hat, s_hat)`; ensure `0 ≤ L ≤ 1` and `L` respects sparsity.
3. Blend kernel: `K' = (1 - α) * K + α * L` with `α = strength`.
4. Run the same unbalanced Sinkhorn on `K'` (or equivalently adjust costs by `-eps*log` blend, mind stability).
5. Return `P`, `v_teacher` unchanged downstream.

## Tests
- With `strength=0`: identical to MVP.
- With synthetic drift, `lower_bound` should improve short‑range mass.
- No NaNs; convergence iterations comparable to MVP.

## Reference / Current Code to Preserve
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
```

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
```