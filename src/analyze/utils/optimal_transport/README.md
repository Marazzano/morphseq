# Optimal Transport Utilities

Reusable optimal transport infrastructure for mask-based temporal analyses.

## Purpose

This module provides **general-purpose optimal transport utilities** that can be used across different analyses, not just embryo morphometrics. The code here is agnostic to the specific biological application.

## What's Included

### Core Data Structures (`config.py`)
- `UOTConfig`: Configuration for UOT solvers
- `UOTFrame`, `UOTFramePair`: Frame containers
- `UOTSupport`: Point cloud representation (coords + weights)
- `UOTProblem`: Full problem specification
- `UOTResult`: Solver output (cost, coupling, mass maps, velocity field)
- `SamplingMode`, `MassMode`: Configuration enums

### Backends (`backends/`)
- **`base.py`**: Abstract backend interface
- **`pot_backend.py`**: CPU implementation using POT library
- Pluggable design allows swapping solvers (e.g., GPU via JAX/ott-jax)

### Density Transforms (`density_transforms.py`)
- `mask_to_density_uniform()`: Uniform mass (0A mode)
- `mask_to_density_boundary_band()`: Boundary-weighted mass (0B mode)
- `mask_to_density_distance_transform()`: Distance-transform mass (0C mode)
- `enforce_min_mass()`: Fallback for near-zero mass

### Multiscale & Sampling (`multiscale_sampling.py`)
- `downsample_density()`: Sum-pooling for area-preserving downsampling
- `pad_to_divisible()`: Pad arrays to match divisor requirements
- `build_support()`: Extract point cloud from density with optional sampling

### Transport Maps (`transport_maps.py`)
- `compute_transport_maps()`: Convert coupling matrix to:
  - Mass created/destroyed maps
  - Velocity field (barycentric projection)

### Metrics (`metrics.py`)
- `summarize_metrics()`: Compute summary statistics from UOT results

## Example Usage

```python
from src.analyze.utils.optimal_transport import (
    UOTConfig,
    POTBackend,
    mask_to_density,
    build_support,
    compute_transport_maps,
)

# Configure solver
config = UOTConfig(
    downsample_factor=4,
    mass_mode="uniform",
    epsilon=1e-2,
    marginal_relaxation=10.0,
)

# Convert masks to densities
density_src = mask_to_density(mask_src, config.mass_mode)
density_tgt = mask_to_density(mask_tgt, config.mass_mode)

# Build support (point clouds)
support_src, _ = build_support(
    density_src,
    max_points=config.max_support_points,
    sampling_mode=config.sampling_mode,
    sampling_strategy=config.sampling_strategy,
    random_seed=config.random_seed,
)
support_tgt, _ = build_support(density_tgt, ...)

# Solve UOT
backend = POTBackend()
result = backend.solve(support_src, support_tgt, config)

# Compute transport maps
mass_created, mass_destroyed, velocity_field = compute_transport_maps(
    result.coupling,
    support_src.coords_yx,
    support_tgt.coords_yx,
    support_src.weights,
    support_tgt.weights,
    work_shape_hw=density_src.shape,
)
```

## Design Principles

1. **Backend-agnostic preprocessing**: All density transforms, downsampling, and sampling work with NumPy/SciPy and don't depend on the solver backend.

2. **Pluggable solvers**: Only the `solve()` method differs between backends. This allows easy GPU upgrades (JAX/ott-jax) without rewriting preprocessing.

3. **Physical mass preservation**: Densities are not normalized per-frame. The solver normalizes internally, then rescales outputs back to source mass units.

4. **Explicit coordinates**: All coordinates use `(y, x)` convention internally. Visualization code handles conversion to `(x, y)` for plotting.

## Potential Use Cases Beyond Morphometrics

- **Cell tracking**: Track cell populations across frames
- **Melanocyte migration**: Analyze pigment cell movement patterns
- **Tissue dynamics**: Compare tissue masks over time
- **Any temporal mask analysis**: Where you want to quantify morphological change

## Related

For embryo-specific analysis code (I/O, preprocessing, visualization), see:
- `src/analyze/optimal_transport_morphometrics/uot_masks/`
