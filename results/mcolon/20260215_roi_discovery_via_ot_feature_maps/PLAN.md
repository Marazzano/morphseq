# Implementation Plan (MorphSeq) — ROI Discovery via WT-Referenced OT Feature Maps

**(Updated: Phase 1 commits to FIXED (λ,μ) bootstrap; removes compare(); clarifies "computationally stable" vs "biology-dependent" tuning)**

## Purpose

- Learn contiguous, interpretable regions of importance (ROIs) on a 512×512 canonical embryo grid that explain genotype discriminability (WT vs cep290) using OT-derived template-space feature maps.
- Build a scalable backend for large datasets and a simple, biologist-friendly front end.
- Ensure statistical validity via selection-aware null distributions and stability resampling.
- Explicitly separate:
  - **(A)** computationally stable defaults (robust + fast to run)
  - **(B)** biology-dependent tuning (signal-to-noise, phenotype type, dataset size)

## Inspiration (principles, not full reproduction)

- Fong & Vedaldi 2017 "Meaningful Perturbations":
  https://arxiv.org/pdf/1704.03296
- Fong et al. 2019 "Extremal Perturbations":
  https://arxiv.org/abs/1910.08485

## Core Model (Linear, small-sample friendly; ROI from weights)

- Weight-map regularized logistic regression:

      J(w,b) = logistic_loss(y, <w, X> + b) + λ||w||_1 + μ TV(w)

- ROI derived from |w| after training (quantile threshold default).
- TV neighborhood: 4-neighborhood restricted to valid in-mask edges (no padding outside embryo).

## Scaling Pattern (default)

- Learn w_low at low resolution; upsample inside the model:
  - learn_res=128 (default), optional 256 later
  - output_res=512 always (canonical)
  - penalties (L1, TV) applied in low-res space

---

## A) DATA CONTRACT (finalize before model code)

### FeatureDataset = standardized on-disk source of truth

**Format:** Zarr (arrays) + Parquet (metadata) + JSON manifest (provenance + rules)

### Directory

    results/<date>/roi_feature_dataset_<tag>/
      manifest.json
      metadata.parquet
      features.zarr/
        X/            (N, 512, 512, C) float32
        y/            (N,) int {0,1}
        mask_ref/     (512,512) bool/uint8
        qc/
          total_cost_C/    (N,) float32
          outlier_flag/    (N,) bool

### Manifest (must include; hard validation)

- canonical_grid: 512×512
- channel_schema: names + definitions + units
- QC rules:
  - IQR outlier filter on total_cost_C (1.5×IQR), logged, never deleted
- split_policy:
  - group_key = embryo_id (MANDATORY; prevents leakage)
- class_balance_strategy (MANDATORY; declared here):
  - use existing MorphSeq "balance method" already used in prior classification models
  - weights computed from TRAIN fold only (recommended)
- chunking/compression spec for X

---

## B) DATA INGESTION (streamable + deterministic)

- Chunk X along N; keep 512×512 tiles contiguous:
  chunk (8,512,512,C) or (16,512,512,C)
- Loader supports full-batch (small N) and minibatch (large N).
- Filtering is deterministic via qc/outlier_flag (default exclude; never drop silently).
- CV splits grouped by embryo_id; fold-local class weights (from training fold only).

---

## C) TV DEFINITION (explicit boundary behavior)

- TV defined over edges; include edge (p,q) only if p and q are inside mask_ref.
- No zero-padding outside embryo.
- Boundary pixels therefore have fewer valid neighbors ("reduced-degree boundary").
- Diagnostics:
  - report boundary_fraction of ROI (ROI overlap with thin boundary band) to detect registration-driven edge artifacts.
- Optional later: TV normalization by edge count if mask geometry varies across datasets; not required in Phase 1.

---

## D) MODEL TRAINING (JAX backend)

### Requirements

- JAX + Optax; jit train_step
- objective logs must include raw + weighted components:
  - logistic_loss_raw
  - l1_raw, tv_raw
  - l1_weighted=λ*l1_raw, tv_weighted=μ*tv_raw
  - total_objective
  - class_weights used

### Default training mode

- params: w_low (learn_res×learn_res×C), b
- w_full = bilinear_upsample(w_low → 512×512×C)
- logits: <X, w_full> + b
- loss: class-weighted logistic + λL1(w_low) + μTV(w_low)

---

## E) λ/μ SWEEP + DETERMINISTIC SELECTION

### Important nuance (explicit)

- There is no universal "perfect (λ,μ)".
- (λ,μ) depends on:
  - phenotype type + expected spatial scale
  - signal-to-noise and sample size
  - feature channel choice (cost vs displacement vs div vs Δm)
- Therefore we treat sweeps as discovery + documentation, not one-time "solve and forget".

### Phase 1: minimal sweep for a stable starting point

- Coarse λ×μ grid (small) to identify a sensible region of operation.
- Deterministic selection rule (pick one and log):
  - **Option A (recommended):** knee on Pareto front (AUROC vs complexity)
  - **Option B:** smallest complexity within ε of best AUROC

### Complexity metrics (recorded for selection + reporting)

- area_fraction, n_components, boundary_fraction

### Outputs

- Store the full sweep table and the selected (λ,μ) with selection score and rule parameters (beta/ε).
- These become "notes" for future biology-specific runs.

---

## F) NULLS + STABILITY (Phase 1 commitment)

### NULL 1 (required): label permutation significance (selection-aware)

- Permute labels at embryo_id unit.
- For each permuted run:
  - run the same sweep + same deterministic selection rule
  - record selected AUROC (and optionally max across sweep)
- p-value computed against the selection-aware null.

### NULL 3 (required): bootstrap stability at FIXED (λ,μ)

**Phase 1 commitment:** bootstrap uses the selected (λ,μ) fixed.

**Reason:**
- answers the interpretable question "is THIS ROI stable?"
- cheaper and more scalable than full-sweep bootstrap
- supports quick iteration and easier debugging

### Bootstrap protocol

- Resample embryos with replacement within each class.
- Fit model at fixed (λ,μ), same learn_res.
- Extract ROI via fixed thresholding rule (quantile q) and compute:
  - IoU distribution across bootstrap ROIs
  - variability of ROI area/components/boundary_fraction
- Report stability summary alongside AUROC and permutation p-value.

### Deferred upgrade (Phase 2 / if needed)

- full-sweep bootstrap ("is the selection procedure stable?") — optional later.

### NULL 2 (spatial structured null)

- Deprioritized; revisit only if reviewers demand.

---

## G) BIOLOGIST-FACING FRONT END (scope-limited for Phase 1)

### Phase 1 API (minimal; no compare())

```python
morphseq.roi.fit(
    genotype="cep290",
    features="cost" | "cost+disp" | "all_ot",
    learn_res=128|256,
    roi_size="small|medium|large",      # maps to λ grid presets
    smoothness="low|medium|high",       # maps to μ grid presets
    class_balance="morphseq_balance_method",
    null="permute|bootstrap|both|none",
    n_permute=100, n_boot=200,
    out_dir=...,
)

morphseq.roi.plot(run_id, style="filled_contours", overlays=["outline", "optional_S_isolines"])
morphseq.roi.report(run_id)  # -> AUROC, selected (λ,μ), area/components, permutation p, bootstrap IoU, boundary_fraction
```

- Optional diagnostic overlay: S isolines (not required for core ROI training)

---

## Immediate next actions (ready to build)

1. **Build FeatureDataset builder + validator (contract-first)**
   - Write Zarr/Parquet/manifest including QC rules, split policy, class_balance_strategy.
   - Implement fail-fast schema checks with informative errors.

2. **Implement streaming loader**
   - Deterministic filtering, embryo-grouped CV splits, fold-local class weights.

3. **Implement JAX trainer (learn_res=128 default)**
   - Weighted logistic + λL1 + μTV with explicit TV boundary rule.
   - Store raw + weighted objective terms and class weights.

4. **Implement small λ×μ sweep + deterministic selection**
   - Log Pareto set + knee/ε selection metadata.
   - Export sweep table and selected (λ,μ).

5. **Implement NULL 1 permutation (selection-aware) + NULL 3 bootstrap (fixed λ/μ)**
   - Parallel, resume-safe artifacts.
   - Report p-value and bootstrap IoU.

## Definition of "done" (Phase 1)

- Validated, streamable FeatureDataset
- Reproducible sweep with deterministic selection and documented (λ,μ) behavior
- Selection-aware permutation p-value
- Fixed-(λ,μ) bootstrap ROI stability (IoU + ROI complexity diagnostics)
- Minimal front end that a non-coder can run and interpret
