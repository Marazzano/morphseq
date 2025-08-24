# TODO: Neural ODE with Unbalanced Optimal Transport Integration

## 🎯 Project Goal
Enhance the existing neural ODE in `src/flux/` that models developmental dynamics by integrating **Unbalanced Optimal Transport (UOT)** with intelligent minibatching and trajectory-aware biasing.

## 📊 System Overview
```
Embryo Data (130k images, sparse tracking)
    ↓
Time-Weighted Spectral Clustering [Task 02]
    ↓
Smart Minibatch Sampling (adjacent time clusters)
    ↓
Trajectory-Biased UOT [Tasks 01 + 03]
    ↓
Teacher Velocities (v̂ᵢ)
    ↓
Neural ODE Training (ż = f_θ(z,t))
```

## 🔍 Phase 0: Codebase Discovery (CRITICAL - Do This First!)

**Agent Instructions**: Before implementing ANYTHING, spend time understanding the existing system:

### A. Find and Document the ODE Implementation
```bash
# Search in src/flux/ for ODE-related files
find src/flux -name "*.py" | xargs grep -l "ode\|ODE\|neural_ode\|dynamics"
find src/flux -name "*ode*.py"
find src/flux -name "*train*.py"
```

Document:
- [ ] Location of ODE model class (likely `models/ode.py` or similar)
- [ ] Current loss function implementation
- [ ] Training script entry point
- [ ] Data loading pipeline (how embeddings + timestamps are loaded)
- [ ] Current hyperparameter handling (config files? argparse? hydra?)

### B. Understand Data Format and Hierarchy
```bash
# Look for data loading and parsing utilities
find src/flux -name "*utils*.py" | xargs grep -l "parse\|embryo\|snip"
find . -name "*.csv" | head -5  # Find example data files
```

**Critical**: Look at parsing utils to understand:
- [ ] Data hierarchy: experiment_id → embryo_id → snip_id → image_id
- [ ] How embryo trajectories are represented
- [ ] CSV schema (columns for embeddings, time, IDs)
- [ ] Which columns contain VAE latents (z_*) vs biological components (bc_*)

### C. Analyze Example Data
```python
# Agent should run this to understand data structure
import pandas as pd
import glob

# Find an embryo data CSV
csv_files = glob.glob("**/embryo*.csv", recursive=True)
if csv_files:
    df = pd.read_csv(csv_files[0], nrows=100)
    print("Columns:", df.columns.tolist())
    print("\nData types:", df.dtypes)
    print("\nTime column values:", df['predicted_stage_hpf'].describe() if 'predicted_stage_hpf' in df else "No time column found")
```

## 📋 Implementation Tasks

### [Task 01: Minibatch UOT](01_minibatch_uot.md)
**Priority**: High | **Depends on**: Phase 0  
**Goal**: Implement minibatch UOT to generate teacher velocities

Key features:
- Minibatch size: 256 (configurable parameter)
- Support variable time gaps (Δt): consecutive and skip-frame
- Three biasing modes for tracked trajectories
- Mass-weighted and masked loss variants

### [Task 02: Time-Weighted Spectral Clustering](02_time_weighted_spectral_clustering.md)
**Priority**: High | **Depends on**: Phase 0  
**Goal**: Create meaningful space-time clusters for minibatch sampling

Key features:
- Dual affinity: morphology (σ_z) + time (σ_t)
- Default to biological components (bc_*), fallback to VAE (z_*)
- Efficient handling of 130k points (consider kNN sparsification)

### [Task 03: Self-Edge Lower Bounds](03_self_edge_lower_bound.md)
**Priority**: Medium | **Depends on**: Tasks 01, 02  
**Goal**: Intelligently bias UOT toward known good trajectories

Key features:
- Global calibration from high-quality tracks
- Adaptive confidence based on displacement speed
- Graceful handling of missing tracks (pure UOT fallback)

## 🔄 Integration Sequence

### Step 1: Exploration and Setup (Day 1)
1. Complete Phase 0 discovery
2. Install dependencies: `pip install pot scipy scikit-learn`
3. Create test data subset (1000 points) for development

### Step 2: Core Implementation (Days 2-3)
1. Implement spectral clustering utility (Task 02)
2. Implement basic minibatch UOT (Task 01, without self-edge bias)
3. Verify on synthetic data

### Step 3: Self-Edge Integration (Day 4)
1. Implement trajectory calibration (Task 03)
2. Add all three biasing modes to UOT
3. Compare performance with/without biasing

### Step 4: Training Integration (Day 5)
1. Modify ODE training loop to use UOT velocities
2. Add configurable hyperparameters
3. Run convergence tests

## ⚠️ Critical Considerations

### Data Quality Handling
- **Sparse tracking**: Many embryos have only 1 image
- **Noisy tracks**: Need robust statistics for calibration
- **Missing targets**: Some source clusters may have no valid targets

### Biological Constraints
- **Irreversibility**: Embryos only progress forward in development
- **No teleportation**: Displacement speeds should be biologically plausible
- **Cluster coherence**: Ensure time gaps respect developmental stages

### Performance Notes
- 130k images is manageable, don't over-optimize initially
- Start with dense matrices, optimize later if needed
- Profile before optimizing

## 🧪 Testing Strategy

### Unit Tests (Lightweight)
1. **Synthetic straight lines**: Perfect tracks → UOT should follow them
2. **Synthetic bifurcation**: One source → two targets
3. **Missing data**: Source with no targets → pure deletion
4. **Speed outlier**: Fast displacement → reduced self-edge weight

### Integration Tests
1. Small real data subset (100 embryos, 10 timepoints)
2. Compare velocities: naive (Δx/Δt) vs UOT vs biased UOT
3. Visualize transport plans as heatmaps

### Validation Metrics
- Transport plan entropy (lower = more confident)
- Self-edge mass fraction (when tracks exist)
- Velocity smoothness over time
- ODE reconstruction error

## 📝 Documentation Requirements

Each implementation should include:
1. **Docstrings**: Full numpy-style with examples
2. **Inline comments**: Explain non-obvious choices
3. **Test file**: `test_[module].py` with clear descriptions
4. **Config example**: How to tune hyperparameters

## 🚀 Success Criteria

- [ ] Agent successfully finds and documents existing ODE code
- [ ] Spectral clustering produces meaningful space-time groups
- [ ] UOT generates smooth velocity fields
- [ ] Self-edge bias improves trajectory consistency
- [ ] ODE training converges with UOT supervision
- [ ] All tests pass with documented results

## 💡 Pro Tips for the Agent

1. **Start simple**: Get basic UOT working before adding complexity
2. **Visualize early**: Plot transport plans to verify correctness
3. **Log everything**: Track which hyperparameters you try
4. **Test incrementally**: Don't wait until the end to test
5. **Ask about parsing**: The data hierarchy is complex - study parsing utils carefully

---

**Next Step**: Agent should begin with Phase 0 discovery, then read the three detailed task files in order.