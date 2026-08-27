# Plan — core-model refactor and pipeline reconciliation

**Updated:** 2026-08-25 · **Location:** `docs/refactors/core-model/PLAN.md`
**Companions:** `DECISIONS.md` (ledger) · `STATUS.md` (generated, do not edit) ·
`contracts/MANIFEST_SCHEMA.md` (binding interface) · `AGENTS.md` (agent rules)
**Evidence:** `reports/` (measured) · `pipeline/` (pipeline state, pre-merge) · `_archive/` (historical)

Every factual claim below carries its source or is marked as inference.

---

## 1. Where we stand

**Phase 0 is done. Phase 1 was never built. The corpus is not ready.**

*Corrected 2026-08-27. The previous version of this section claimed "the code is essentially done —
58/58 core tests pass at `0431e6d9`". That was false and is the reason this plan misdirected work
for a week. See `reports/GROUND_TRUTH_2026-08-27.md`.*

- **Phase 0: done.** Import paths, Hydra paths, run scripts, `_pixel_scale` geometry, the inert
  metric margin, `accumulate_grad_batches`, the `metricVAE` logvar clamp, and the metric
  `dataconfig` target were all fixed by `6f6e0f3f` (2026-08-18). `src/core/data/` is tracked.
  *Verified: `reports/GROUND_TRUTH_2026-08-27.md`, Claims 3–8.*
- **Core tests: 5, all passing** — not 58. `tests/core/` holds three files, all Phase 0.
  *Command and output: same report, Claim 2.*
- **Phase 1: not started.** Commits `3c0986e6`, `96f3991d`, `c3ca21a2`, `0431e6d9` do not exist in
  any object store or reflog. No manifest adapter, no pipeline-backed dataset, no Hydra data group,
  no provenance bundle. The only surviving artifact is one untracked 348-line
  `src/core/data/pipeline_contracts.py` in the `morphseq-phase1a` worktree.
  *Same report, Claim 9.*
- **One live Phase 0 bug remains:** `contrastive_transform(target_size=...)` accepts the argument
  and ignores it (`src/core/data/data_transforms.py:12`), while `run_utils.py:484,491` pass it.
  Any metric run at a non-native input size silently trains on unresized images. Untested.

The spec to build Phase 1 against is `contracts/MANIFEST_SCHEMA.md` and
`plans/AGENT_BRIEFS_PHASE1.md` — a valid brief that was never executed.

What is also not ready is the images. The stored corpus was written before the 2026-08-02 rendering fix
and carries no provenance proving otherwise. The 176,466-row metric cohort is almost certainly
176,466 rows of **regressed** images.

**The code track and the corpus track are independent.** A full integration acceptance can run today
on the existing corpus to prove the plumbing, while regeneration proceeds in parallel. The only rule:
no run on the current corpus is a science result.

## 2. The corpus problem

| | legacy | regressed | code now |
|---|---:|---:|---:|
| snip scale | 6.5 µm/px | 7.8 | 6.5 (restored `37aeb639`, 2026-08-02) |
| blend radius | 75 µm | 20 | 75 (restored) |

Blend radius caused the clipped appearance — sigma is `blend_radius / pixel_size`, so the taper
collapsed 11.54 px → 2.56 px and saturation rose **3.63×**, more saturated in 89 of 93 paired wells.
Scale shrank embryos 16.7% per linear dimension; predicted area ratio `(6.5/7.8)² = 0.694`, measured
0.700. Embedding effect from the *same* checkpoint: per-dimension correlation 0.687, pairwise-distance
correlation 0.660, temperature R² 0.582 → 0.353. *Source: `pipeline/SNIP_IMAGE_REGRESSION_STATUS.md` (pre-merge; refresh before dispatch).*

The fix touched no stored snip. Every one of the 699,505 audited snips predates it, and all 133
readable inventories lack both scale fields — *verified 2026-08-19, `reports/PIPELINE_RECON.md`;
figures independently re-derived from `reports/recon_tables/*.csv` on 2026-08-27*.

**Ancestry correction (2026-08-27).** `37aeb639` is on both `origin/main` and this branch. But
before the 2026-08-27 merge, `5976f8d2` (the PR-31 snip-rendering merge) was **not** an ancestor of
`core-model-refactor` — `git merge-base --is-ancestor` exited 1, and the merge base was `14c8ab10`.
An earlier note dismissed this because the commit *resolved*; resolving is not the same as being an
ancestor. The `origin/main` merge (`19061cbf`, 2026-08-27) closed the gap, landing 116
`src/data_pipeline` commits including a snip-rendering rewrite. Any "the fix is in our ancestry"
claim predating that merge must be re-derived per-commit.

**Scale columns: current code is fine; the stored corpus is what lacks them.**
`run_snip_processing.py:463` initialises `source_micrometers_per_pixel` to `None`, then `:520`
populates it from the frame inventory. Verified by rendering: the 2026-08-27 gate run emitted
`source_micrometers_per_pixel = 1.887` and `snip_micrometers_per_pixel = 6.5` on all 97 rows. The
133 inventories that lack both fields were written *before* those columns existed — regenerating
fixes them.

*(Corrected 2026-08-27: an earlier revision of this section claimed the renderer still writes
`None`. It does not; that line is placeholder initialisation.)*

What `794adf46` still adds, and what remains unmerged, is the **rendering-contract sidecar** — the
eight-parameter record of target um/px, blend radius, mask source, orientation policy, CLAHE,
background model, resampling kernels, and encoding. `origin/main` has no `provenance.py` under
`snip_processing/`. That is what lets a regenerated corpus prove *how* it was rendered, beyond the
two scale numbers.

**Restriction instead of regeneration probably isn't available.** *Inference, 2026-08-25:* across 130
experiments, `log10(saturated_255_fraction_p95)` is unimodal and continuous (p05 −4.42, median −3.73,
p95 −2.71); the largest gap isolates one experiment at the zero floor. No two-era structure. Caveat:
that measure is pixels at exactly 255, while the root-cause analysis used fraction ≥250 per image —
recomputing properly would make it conclusive. Snip file **mtimes** would settle it faster still.

**A separate intensity finding, not the same thing.** η² by experiment: `min` 0.477, `mean` 0.264,
`zero_fraction` 0.242, `std` 0.222, versus `max` 0.048 and `saturated_255` 0.047 — *measured
2026-08-25 over 17,768 sampled images*. The signature is at the **black end**, not saturation. May be
a rendering artifact (the background is synthesised) rather than acquisition — **defer this decision
until after regeneration.**

## 3. Open decisions — Nick only

| # | Decision | Blocks |
|---|---|---|
| O1 | **Metric mapping table + relation matrix**, versioned. No plate column maps to the 48-class vocabulary; exact matching gives 0–1, order-insensitive token matching recovers 11 | all metric science |
| O2 | **QC policy.** `sa_outlier_flag` is in 69.6% of failures and is the only morphometric criterion in an otherwise acquisition/segmentation set. Study S1 running | cohort composition |
| O4 | **Corpus route.** Regenerate (SeaHub fully; everything else from snip processing onward) vs restrict — see §2 | everything downstream |
| O5 | **Black-level batch effect.** Deterministic normalization, additive-offset-with-clipping augmentation (implemented, off), or conditioning. Defer past regeneration | science validity |
| O6 | **Frozen cohort** — a checked-in ordered experiment list per run. Only 105 of 148 have all four artifacts. Freeze *after* regeneration, since QC verdicts move | reproducibility |
| O7 | **Neighbour contamination** in the 75 µm halo. Filling non-target pixels before blending changes the input distribution again and needs an A/B gate | rendering contract |

## 4. Issue register

**P0 — blocks an integrated trainable model**

- Snip corpus acceptance exercise on `20250612_30hpf_ctrl_atf6` under a fresh immutable root at a named commit *(T2b)*
- **Preprocessing provenance absent** — no scale, blend, mask source, orientation policy, CLAHE, background model, resampling kernel recorded. **Deadline: before bulk regeneration, or it must be redone** *(T2a)*
- Frozen cohort artifact *(O6)* · metric mapping + matrix *(O1)*
- Real end-to-end acceptance: manifest → dataloaders → Lightning → `Trainer.fit` → provenance + W&B → checkpoint reload → DDP repeat *(T3)*

**P1**

Cohort-wide legal-positive preflight · split-ratio tolerance · decode robustness *(D25)* · named
basic/metric presets · real-host I/O measurement · pipeline must emit optical covariates and
`image_product_type` as columns · `stage_prediction_status` backfill (recovers 8,601 QC-passing
snips, 176,466 → 185,067) · 28 experiments with no QC artifact (167,603 snips) · SeaHub segmentation
signoff · SeaHub µm/px placeholder 7.8 uncalibrated · genotype vocabulary normalization (99 values,
~20 collapsible) · background/strain 63.66% null · pair reproducibility across workers and DDP ranks
· W&B offline/credentials (`train_vae` constructs `WandbLogger` unconditionally)

**P2**

~2.5% 180° orientation-flip tail; yolk masks unwired · pipeline golden-image regression suite · QC
schema Q02 lacks per-flag columns · `is_valid_snip` True for all 699,505 rows (no-op) · 38 plate
schemas · `chem_perturbation` × `start_age_hpf` composite never tested · per-embryo z-slice snips
(needs product/z identity axis — see `docs/data_pipeline/specs/target/specs/tech_debt/projection_method_not_in_image_id.md`)

**P3**

Resident model server toggles default off · GPU budget in SGE scripts not a profile · SeaHub
calibration/routing/single-z QC semantics · stale `NotImplementedError` text for Keyence z-stacks ·
`analysis_ready/__init__.py` stale docstring · dead plate columns 100% null

## 5. Threads in flight

See `_archive/THREAD_BRIEFS.md` (superseded 2026-08-27; do not dispatch from it). T1 status system · T2a pipeline provenance · T3 metric unblock +
hardenings + acceptance · S1 QC/`sa_outlier` study · S2 stage lineage study · A1 regeneration scope
and timing. **T2b (rerender + legacy comparison) is gated on T2a.** SeaHub front-half work
(calibration, segmentation validation) has no dependencies and can start any time; SeaHub
reprocessing waits on the accepted contract from T2b.

## 6. Not in scope for the first pipeline-backed run

Encoder/decoder redesign · latent partitioning changes · native 576×256 training · optical
conditioning (covariates absent) · segmentation-error mimicry · per-embryo z-slice snips and
focus-delta conditioning · SeaHub inclusion unless deliberately selected · general pipeline
packaging cleanup.
