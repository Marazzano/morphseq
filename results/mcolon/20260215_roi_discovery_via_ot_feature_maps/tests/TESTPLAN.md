# ROI Discovery Test Plan — Phases 1 through 2.5

**Date:** 2026-02-16
**Status:** Draft — execute sequentially; each task has its own test file.

## How to use this plan

Each task below maps to a test file in this `tests/` directory. Run them in order.
Files are structured as standalone pytest modules with synthetic data fixtures.
No real data or GPU required — all tests use tiny grids (16x16 or 32x32) and NumPy fallbacks where possible.

**Key biological prior (cep290):** The cep290 mutant phenotype produces signal
predominantly in the **tail region** of the embryo. Planted-ROI tests embed signal
in the bottom rows of the grid to simulate this. Sanity checks verify that
discovered ROIs concentrate there, not at the head/top.

---

## Phase 1: Core Pipeline

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 1.1 | Config construction + validation | `test_p1_01_config.py` | Enums, presets, frozen dataclasses, resolve_lambda/mu |
| 1.2 | FeatureDataset build + validate | `test_p1_02_feature_dataset.py` | Zarr/Parquet round-trip, manifest schema, QC outlier flagging |
| 1.3 | Loader + CV splits | `test_p1_03_loader.py` | Group-aware splits, no embryo leakage, class weights fold-local |
| 1.4 | TV edge construction + computation | `test_p1_04_tv.py` | Edge list correctness, mask-aware boundaries, boundary fraction |
| 1.5 | Trainer (logistic + L1 + TV) | `test_p1_05_trainer.py` | Convergence on planted signal, weight map localization, objective logging |
| 1.6 | ROI extraction | `test_p1_06_roi_extraction.py` | Quantile thresholding, area_fraction, n_components, tail concentration |
| 1.7 | Lambda/mu sweep + selection | `test_p1_07_sweep.py` | Pareto knee, epsilon-best, sweep table completeness, determinism |
| 1.8 | Permutation null (NULL 1) | `test_p1_08_null_permutation.py` | Selection-aware p-value, null AUROC distribution, embryo-level shuffle |
| 1.9 | Bootstrap stability (NULL 3) | `test_p1_09_null_bootstrap.py` | IoU distribution, fixed-(lam,mu), group-aware resampling |
| 1.10 | API integration (fit/plot/report) | `test_p1_10_api.py` | End-to-end smoke test through roi_api.fit() |

## Phase 2.0: Occlusion Validation

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 2.1 | Perturbation + baseline | `test_p2_01_perturbation.py` | Spatial baseline from WT only, preserve/delete operator shapes, fold safety |
| 2.2 | Occlusion evaluation | `test_p2_02_occlusion_eval.py` | Logit gaps, AUROC deltas, single-class NaN handling |
| 2.3 | Bootstrap occlusion (OOB) | `test_p2_03_occlusion_bootstrap.py` | OOB-only evaluation, degenerate OOB handling, threshold sensitivity |
| 2.4 | Resampling helpers | `test_p2_04_resampling.py` | iter_bootstrap_groups, stratification, OOB empty/single-class flags |
| 2.5 | Integration fixes (ADDENDUM A) | `test_p2_05_integration_fixes.py` | compute_logits shared utility, channel_names in TrainResult, config import |

## Phase 2.5: Learned Mask (Fixed Model)

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 2.5.1 | Mask parameterization | `test_p25_01_mask_param.py` | sigmoid conversion, upsample, TV loss, jitter shift |
| 2.5.2 | Mask objective (dual) | `test_p25_02_mask_objective.py` | Preserve > delete on planted ROI, score monotonicity |
| 2.5.3 | Mask trainer (fixed model) | `test_p25_03_mask_trainer.py` | Mask recovers planted ROI, objective decreases, jitter stability |

---

## Execution checklist

```
Phase 1:
  [ ] 1.1  Config
  [ ] 1.2  FeatureDataset
  [ ] 1.3  Loader
  [ ] 1.4  TV
  [ ] 1.5  Trainer
  [ ] 1.6  ROI extraction
  [ ] 1.7  Sweep
  [ ] 1.8  Permutation null
  [ ] 1.9  Bootstrap stability
  [ ] 1.10 API integration

Phase 2.0:
  [ ] 2.1  Perturbation + baseline
  [ ] 2.2  Occlusion evaluation
  [ ] 2.3  Bootstrap occlusion
  [ ] 2.4  Resampling helpers
  [ ] 2.5  Integration fixes

Phase 2.5:
  [ ] 2.5.1  Mask parameterization
  [ ] 2.5.2  Mask objective
  [ ] 2.5.3  Mask trainer
```

## Running

```bash
# From the roi_discovery directory:
cd results/mcolon/20260215_roi_discovery_via_ot_feature_maps

# Run all tests:
pytest tests/ -v

# Run a single phase:
pytest tests/test_p1_*.py -v

# Run a single task:
pytest tests/test_p1_04_tv.py -v
```
