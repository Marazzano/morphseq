# Plan — core-model refactor and pipeline reconciliation

**Updated:** 2026-08-25 · **Location:** `docs/refactors/core-model/PLAN.md`
**Companions:** `DECISIONS.md` (ledger) · `STATUS.md` (generated, do not edit) ·
`contracts/MANIFEST_SCHEMA.md` (binding interface) · `AGENTS.md` (agent rules)
**Evidence:** `evidence/` (narrative state docs, audits) · `reports/` (generated study output)

Every factual claim below carries its source or is marked as inference.

---

## 1. Where we stand

**The code is essentially done. The corpus is not.**

Phase 0 and Phase 1 are implemented and unit-tested — 58/58 core tests pass at `0431e6d9`, and
`python -m src.core.run.training --help` composes — *per `evidence/TRAINING_READINESS_REMAINDER.md`,
2026-08-24*. The manifest adapter, datasets, transforms, loaders, Hydra data group, and provenance
bundle all exist. What remains on the code side is four small hardenings, named run configs, and one
real end-to-end acceptance run.

What is not ready is the images. The stored corpus was written before the 2026-08-02 rendering fix
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
correlation 0.660, temperature R² 0.582 → 0.353. *Source: `evidence/SNIP_IMAGE_REGRESSION_STATUS.md`.*

The fix touched no stored snip. Every one of the 699,505 audited snips predates it, and all 133
readable inventories lack both scale fields — *verified 2026-08-19, `reports/PIPELINE_RECON.md`*.

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

See `plans/THREAD_BRIEFS.md`. T1 status system · T2a pipeline provenance · T3 metric unblock +
hardenings + acceptance · S1 QC/`sa_outlier` study · S2 stage lineage study · A1 regeneration scope
and timing. **T2b (rerender + legacy comparison) is gated on T2a.** SeaHub front-half work
(calibration, segmentation validation) has no dependencies and can start any time; SeaHub
reprocessing waits on the accepted contract from T2b.

## 6. Not in scope for the first pipeline-backed run

Encoder/decoder redesign · latent partitioning changes · native 576×256 training · optical
conditioning (covariates absent) · segmentation-error mimicry · per-embryo z-slice snips and
focus-delta conditioning · SeaHub inclusion unless deliberately selected · general pipeline
packaging cleanup.
