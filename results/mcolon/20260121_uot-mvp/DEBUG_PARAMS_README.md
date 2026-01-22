# UOT Parameter Debugging and Sensitization

## Overview

This script implements a **sequential testing approach** to diagnose and sensitize UOT (Unbalanced Optimal Transport) parameters. It identifies numerical instability issues and finds viable parameter ranges through synthetic tests.

## Key Features

### 1. Sequential Testing Strategy

Tests build on each other - each test only uses parameters that passed the previous test:

1. **Test 1: Identity (Null Test)** - Circle → same circle
   - Expected: No mass created/destroyed, zero transport cost
   - **Pass criteria**: cost < 1e-6, created_mass < 1e-6, K_healthy = True

2. **Test 2: Non-overlapping Circles** - Translation without shape change
   - Expected: Transport cost > 0, NO mass created/destroyed
   - **Pass criteria**: cost > 0, created_mass < 1e-3, sparsity > 0.8

3. **Test 3: Shape Change** - Circle → oval (same centroid)
   - Expected: Mass created/destroyed at shape boundaries
   - **Pass criteria**: created_mass > 0, destroyed_mass > 0, numerical_stable = True

4. **Test 4: Combined** - Translation + shape change
   - Expected: Combined transport + creation/destruction
   - **Pass criteria**: cost > 0, created_mass > 0, numerical_stable = True

### 2. Critical Diagnostics

#### Gibbs Kernel Analysis
```python
K = exp(-C/epsilon)
```
The solver operates on K, not C. If K underflows to zero, the solver fails silently.

**Health indicators:**
- `K_min > 1e-10` - No underflow
- `K_max < 1 - 1e-10` - Epsilon not too large
- `K_healthy = True` - Kernel has proper variation

#### Coupling Sparsity
Biological motion is local → coupling should be sparse (> 80% zeros).
- Low sparsity = mass diffusion = epsilon too high

#### Surface Area Tracking
Tracks `created_area_um2` and `destroyed_area_um2` in physical units (μm²).

### 3. Standardized Visualization

**ALL plots use the CANONICAL GRID coordinate system:**
- **Shape**: 256H × 576W pixels
- **Resolution**: 7.8 μm/pixel
- **Physical dimensions**: 2.0 mm × 4.5 mm

This ensures all results are **directly comparable** regardless of input resolution.

## Usage

### Run Individual Tests

```bash
# Test 1 only (full parameter grid)
python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test 1

# Test 2 only (requires Test 1 viable params)
python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test 2

# Run all tests sequentially
python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test all
```

### Customize Test Parameters

```bash
# Change circle radius (default: 40 px)
python debug_uot_params.py --test all --radius 50

# Change separation for non-overlap test (default: 120 px)
python debug_uot_params.py --test all --separation 100

# Change shift for combined test (default: 10 px)
python debug_uot_params.py --test all --shift 15
```

## Parameter Grid

**Default sweep:**
- **Epsilon**: [1e-3, 1e-2, 1e-1, 1.0, 10.0]
- **Marginal relaxation**: [0.1, 1.0, 10.0, 100.0]
- **Coord scale**: 1/256 (fixed)
- **Metric**: sqeuclidean (fixed)

**Total combinations**: 5 × 4 = 20 parameter sets per test

## Output Structure

```
results/mcolon/20260121_uot-mvp/debug_params/
├── test1_identity/
│   ├── results.csv                           # All metrics for all param combos
│   ├── viable_params.json                    # Params that passed criteria
│   ├── sensitivity_created_mass.png          # 2D heatmap
│   ├── sensitivity_cost.png                  # 2D heatmap
│   ├── parameter_comparison_grid.png         # Visual grid of all combos
│   └── eps_{eps}_regm_{regm}/                # Per-param diagnostics
│       ├── input_masks.png                   # Source/target with metrics
│       ├── cost_and_gibbs.png                # Cost matrix + Gibbs kernel
│       ├── flow_field.png                    # Velocity magnitude + quiver
│       └── creation_destruction.png          # Mass created/destroyed maps
├── test2_nonoverlap/
│   └── ... (same structure)
├── test3_shape_change/
│   └── ... (same structure)
├── test4_combined/
│   └── ... (same structure)
└── recommended_params.json                   # Final recommendations
```

## Key Metrics Recorded

| Metric | Description | Good Value |
|--------|-------------|------------|
| `cost` | Total transport cost | Test-dependent |
| `cost_is_nan` | Numerical failure | False |
| `numerical_stable` | Overall health | True |
| `K_healthy` | Gibbs kernel variation | True |
| `K_zeros` | Underflow count | 0 |
| `sparsity` | Coupling sparsity | > 0.8 |
| `created_mass` | Mass created (normalized) | Test-dependent |
| `destroyed_mass` | Mass destroyed (normalized) | Test-dependent |
| `created_area_um2` | Created surface area | Test-dependent |
| `destroyed_area_um2` | Destroyed surface area | Test-dependent |
| `mean_velocity_px` | Average displacement | Test-dependent |
| `velocity_has_nan` | Velocity field NaNs | False |

## Diagnostic Indicators

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Cost is NaN | Gibbs kernel underflow (K→0) | Increase epsilon |
| All mass created/destroyed | reg_m too high | Decrease reg_m |
| Coupling not sparse | epsilon too high | Decrease epsilon |
| Gibbs kernel all ones | epsilon >> cost values | Decrease epsilon |
| Velocity chaotic | Numerical instability | Increase epsilon, check K |
| Identity test fails | Fundamental param issue | Start with larger epsilon |
| K_zeros > 0 | Underflow in kernel | Increase epsilon |
| K_healthy = False | No variation in kernel | Adjust epsilon |

## Example Results Interpretation

### Test 1 Results (Identity)
```csv
epsilon,marginal_relaxation,cost,created_mass,K_healthy,numerical_stable
1e-3,10.0,1.234,0.5678,False,False    # FAIL: K underflow
1e-2,10.0,0.0012,0.0001,True,True     # PASS: Near-zero as expected
1.0,10.0,0.0001,0.0,True,True         # PASS: Perfect
```

### Sensitivity Heatmap
- **Blue/cold colors** = low values (good for identity test)
- **Red/hot colors** = high values (bad for identity test)
- Identify parameter "sweet spots" visually

### Parameter Comparison Grid
- **Green border** = numerical_stable = True
- **Red border** = numerical_stable = False
- Shows status of all 20 combinations at a glance

## Physical Units

All surface areas are in **micrometers squared (μm²)**:
- `created_area_um2 = created_mass * (um_per_px)²`
- `destroyed_area_um2 = destroyed_mass * (um_per_px)²`

Velocities are in **pixels per frame**:
- For same embryo, different timepoints → physical movement rate
- For different embryos, same timepoint → morphological difference (not movement)

## Canonical Grid Standardization

**Critical for reproducibility:**
- All plots have **identical axis limits**: x ∈ [0, 576], y ∈ [0, 256]
- All data is **mapped to canonical coordinates**
- Enables **direct visual comparison** across tests and parameter combinations
- Matches the **snip export pipeline** exactly

## Next Steps After Running

1. **Review `recommended_params.json`** for viable parameter ranges
2. **Inspect failing parameter combinations** to understand failure modes
3. **Check sensitivity heatmaps** to see parameter trade-offs
4. **Examine Gibbs kernel diagnostics** for underflow/overflow issues
5. **Use viable params for real embryo data** testing

## Scope

**This script is for SYNTHETIC DATA ONLY.**

Real embryo mask testing requires:
- Loading actual embryo masks
- Handling real morphological complexity
- Time-series analysis
- Cross-embryo comparisons

This diagnostic framework establishes baseline parameters before moving to real data.

## Implementation Details

- **Framework**: Uses `src.analyze.optimal_transport_morphometrics` pipeline
- **Backend**: POT (Python Optimal Transport) library
- **Solver**: `ot.unbalanced.sinkhorn_unbalanced`
- **Mass mode**: UNIFORM (constant density on mask)
- **Align mode**: none (no pre-alignment)
- **Metric**: sqeuclidean (squared Euclidean distance)

## References

- Plan document: UOT Parameter Debugging and Sensitization Plan
- Canonical grid: `canonical_grid_implementation.md`
- UOT implementation: `src/analyze/utils/optimal_transport/`
- POT documentation: https://pythonot.github.io/
