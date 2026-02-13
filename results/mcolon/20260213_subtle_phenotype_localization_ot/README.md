# Subtle-Phenotype Localization via WT-Referenced OT

**Project Start Date**: 2026-02-13
**Pilot Dataset**: cep290 (simple curvature phenotype)
**HPF Bin**: 48 hpf (47-49 hpf window for initial implementation)

---

## Overview

This project builds a time-consistent, interpretable spatial localization pipeline for subtle phenotypes using:
- **Unbalanced optimal transport** (UOT) to map mutant masks onto WT reference
- **Rostral-caudal (S) coordinates** for embryo-intrinsic spatial localization
- **AUROC-by-region analysis** to identify discriminative spatial regions
- **Automated ROI discovery** via patch search on S axis

**Goal**: Answer "where along the embryo does the phenotype manifest?" with stable, interpretable spatial signals.

---

## Key Documents

- **`PLAN.md`**: Full specification document (DO NOT MODIFY)
- **`IMPLEMENTATION_PLAN.md`**: Detailed implementation roadmap with scripts, functions, and dependencies
- **`config.yaml`**: Configuration parameters for the analysis

---

## Infrastructure Leveraged

### From Existing OT Work

1. **Reference Embryo Selection** (Stream D):
   - Pre-existing cohort selection pipeline
   - 3 reference WT embryos for 24-48 hpf range
   - Selection criteria: maximize coverage, minimize curvature

2. **UOT Grid Configuration**:
   - Canonical grid: 256×576 pixels at 10 μm/pixel
   - Yolk-based alignment
   - Parameters: `epsilon=1e-4`, `marginal_relaxation=10.0` (already tuned)

3. **Plotting Utilities** (`viz.py`):
   - Cost heatmaps, quiver plots, mass creation/destruction
   - NaN masking enforced
   - Consistent vmin/vmax for cross-run comparisons

4. **Bootstrap Utilities** (`spline_fitting/bootstrap.py`):
   - Bootstrap spline fitting with uncertainty estimation
   - Stratified resampling by embryo_id for AUROC confidence intervals

5. **Centerline Extraction** (`body_axis_analysis/centerline_extraction.py`):
   - Geodesic skeletonization method (same as data pipeline)
   - Gaussian blur preprocessing (sigma=15.0) - also provides density measure
   - B-spline smoothing for curvature measurement
   - Basis for S coordinate assignment

---

## Directory Structure

```
.
├── PLAN.md                    # Specification (DO NOT MODIFY)
├── IMPLEMENTATION_PLAN.md     # Implementation roadmap
├── README.md                  # This file
├── config.yaml                # Configuration parameters
├── scripts/                   # Analysis scripts (Sections 0-6)
├── utils/                     # Utility modules
├── tests/                     # Unit and integration tests
├── data/                      # Input data manifests
├── ot_results/                # OT outputs (c(x), d(x), Δm(x))
├── s_bin_masks/               # S bin masks
├── outputs/                   # CSV/JSON results by section
├── figures/                   # Plots by section
└── logs/                      # Execution logs
```

---

## Implementation Sections

### Section 0: Data Preparation ✨ NEXT
- Select reference WT embryo
- Load masks for 26-28 hpf bin (WT + cep290)
- **Script**: `scripts/00_select_reference_and_load_data.py`

### Section 1: OT Mapping + Outlier Filtering
- Run UOT for all embryos → reference
- IQR-based outlier filtering
- Cost heatmaps, displacement fields
- **Script**: `scripts/01_run_ot_mapping_and_filtering.py`

### Section 2: Spline Centerline + S Profiles
- Fit spline to reference mask
- Assign S ∈ [0,1] to pixels (head→tail)
- Compute along-S profiles (cost, displacement, divergence)
- **Script**: `scripts/02_compute_spline_and_s_profiles.py`

### Section 3: Feature Table Construction
- Consolidate S-bin features
- Organize into feature sets A, B, C
- **Script**: `scripts/03_build_feature_table.py`

### Section 4: AUROC-by-Region Analysis
- Compute AUROC per S bin (univariate + 2D directional)
- Bootstrap confidence intervals
- Heat kernel smoothed AUROC profiles
- **Script**: `scripts/04_compute_auroc_by_s_bin.py`

### Section 5: Automated ROI Discovery (1D Patch Search)
- Search over contiguous S intervals
- Patch ablation for importance
- Bootstrap stability testing
- **Script**: `scripts/05_patch_search_on_s.py`

### Section 6: 2D Sparse Mask Learning (OPTIONAL)
- L1 + TV regularization
- Pareto optimization over (λ, μ)
- **Script**: `scripts/06_sparse_mask_learning.py`

---

## Key Design Decisions

1. **Single HPF bin (48 hpf, i.e., 47-49 hpf)** for pilot → extends to multiple bins trivially
2. **Single WT reference embryo** from Stream D cohort at 48 hpf
3. **Fixed OT parameters** (epsilon, marginal_relaxation) → already tuned
4. **K=10 S bins initially** → validate, then try K=20
5. **Gaussian kernel smoothing for visualization only** → stats on unsmoothed features; also provides density measure
6. **Bootstrap by embryo_id** (not snip) → avoid leakage
7. **Geodesic skeletonization** (same method as data pipeline) → consistent with existing curvature calculations

---

## Success Criteria

### Section 1
- ✅ Outlier removal stabilizes cost distributions
- ✅ Mean vector fields show coherent structure
- ✅ Heat kernel smoothing highlights spatial patterns

### Section 2
- ✅ S profiles are smooth and sensible
- ✅ Robust to K (10 vs 20 bins)
- ✅ Curvature phenotype shows expected head/tail emphasis

### Section 4
- ✅ AUROC localizes to interpretable S regions
- ✅ Smoothed AUROC profiles show clear peaks
- ✅ Stable under bootstrap resampling

### Section 5
- ✅ Selected interval aligns with known phenotype
- ✅ Stable across bootstrap resamples
- ✅ Ablation confirms interval importance

---

## Next Steps

1. ✅ **DONE**: Copy PLAN.md to dated folder
2. ✅ **DONE**: Create IMPLEMENTATION_PLAN.md
3. ✅ **DONE**: Create directory structure
4. ✅ **DONE**: Create config.yaml
5. **TODO**: Implement utility modules (`utils/*.py`)
6. **TODO**: Implement Section 0 (data loading)
7. **TODO**: Implement Section 1 (OT mapping)
8. **TODO**: Validate Section 1 before proceeding

---

## Dependencies

**Python Packages**:
- numpy, pandas, scipy, scikit-image, scikit-learn
- matplotlib, seaborn, tqdm, pyyaml

**MorphSeq Modules**:
- `src.analyze.optimal_transport_morphometrics.uot_masks.*`
- `src.analyze.spline_fitting.*`
- `src.analyze.utils.optimal_transport.*`

---

## Notes

- All statistical tests on **unsmoothed** features
- Heat kernel smoothing is **visualization only**
- CV always by `embryo_id` (GroupKFold)
- Bootstrap resampling by `embryo_id` (not snip)
- Reference embryo and OT params **fixed** for pilot

---

**Status**: Ready for implementation. Begin with Section 0 data loading.
