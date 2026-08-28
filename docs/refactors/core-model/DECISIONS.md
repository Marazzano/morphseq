# Decision ledger — core-model refactor

**Updated:** 2026-08-28 · **Location:** `docs/refactors/core-model/DECISIONS.md`

This is the ledger of things a human decided. It contains no computed facts — those live in
`STATUS.md`, which is script-generated. Plan, issues, and next steps live in `PLAN.md`.

**Every claim in a written document carries its source inline** — a command, a `file:line`, or a
measurement date — or is explicitly marked as inference. Claims that cannot carry a source belong in
generated `STATUS.md`.

⚠️ **Before replacing the previous version of this file, diff it and report anything it contained
that is not carried over here.** This ledger was reconciled outside the repo and may not include
decisions recorded only in the prior copy.

---

## Ratified

| # | Decision | Verified by |
|---|---|---|
| D1 | `analysis_ready` is **not** the training boundary; the adapter builds from product-aware `snip_inventory` + `frame_inventory` + `stage_predictions` + `snip_qc` + `plate_metadata`, plus collection provenance when required for source-specific start age | — |
| D2 | Replace `metric_array` with a **rule-based relation function** at class×class granularity; curation revision pending | — |
| D4 | Carry explicit product and plane columns; vanilla phase one selects one configured BF projection asset. **No ID-format change** — product key and z belong in columns, never in `snip_id` | — |
| D5 | Fix both metric mis-wiring causes: the `src/data` shim import at `model_configs.py:8`, **and** `dataconfig.target: "BasicDataset"` in the metric Hydra config | — |
| D6 | Keep `[1, 288, 128]` as the default model input; image size becomes a config parameter | — |
| D7 | Derive `_pixel_scale` from `input_dim` rather than the literal `128*288` at `loss_functions.py:185-187`. Forced by D6 | — |
| D8 | **On-the-fly** downsampling at the dataset boundary; no exported training set. A cache only if a real-host I/O measurement demands it | — |
| D9 | Delete the margin term from the metric loss entirely — confirmed inert (shared across all logits, cancels in the softmax) | — |
| D10 | Delete `accumulate_grad_batches` — never passed to Trainer, ignored under manual optimization | — |
| D11 | Splits group-disjoint at `physical_embryo_id`, **content-hash based** (`blake2b`), stable under cohort growth, persisted by ID. Ratio tolerance asserted; empty required split fails unconditionally | — |
| D12 | Decoder δ-pathway: thin, injected late | — |
| D13 | Predictive z-slice **reconstruction as an auxiliary objective** is excluded ("not worth the lift"). This does not exclude carrying z-plane assets or later using z slices as model inputs | — |
| D15 | Metric path wired now; metric **policy** revisited separately | — |
| D16 | Run provenance = one directory per run: `resolved_config.yaml`, ordered observation IDs and selected asset keys, `split_assignments.csv` keyed by `physical_embryo_id`, `metric_group_map.csv` or explicit stub record, `sources.json` (path/size/mtime/row-count per source artifact), `cohort_report.json`, plus the adapter's git SHA. **Not** the full tables — keys + config regenerate them, `sources.json` detects when they cannot | — |
| D17 | The provenance bundle is a **W&B Artifact**; the local directory is staging, addressed by one config key (`run_artifacts_dir`). No coupling to Hydra's cwd or Lightning's internals | — |
| D18 | The manifest carries the **individual QC flag columns**, not just `use_snip`; cohort policy is a named, versioned predicate over flags | — |
| D19 | Do **not** hash image content. Hash source artifacts plus the selected observation/asset key lists | — |
| D20 | `ImageFolder` is **deleted**, not deprecated. Datasets are plain `torch.utils.data.Dataset` over the configured resolved sample view; view generation lives in `__getitem__`, which has the selected asset and parent observation metadata. No separate image-discovery abstraction layer | — |
| D21 | **6.5 µm/px and 75 µm blend radius are the compatibility baseline.** Pre-August stored snips are not checkpoint-compatible inputs | — |
| D22 | **Absence is never failure.** `qc_status` and `stage_status` are three-state. `no_artifact` ≠ failed QC (28 experiments, 167,603 snips); `unavailable` ≠ failed staging (2 experiments, 20,219 snips with finite stage and no status column) | — |
| D23 | Adopt SupCon **`L_out`** (sum over positives outside the log) rather than `L_in`. Decided previously and relitigated through documentation entropy — treat as settled. Requires a λ_metric/τ retune | — |
| D24 | Test holdout supports an explicit `test_experiments` list: listed experiments go wholesale to test, the remainder hash-assigns into train/eval. Tolerance targets computed over the unpinned pool | — |
| D25 | Decode failures are **non-fatal**: log the `snip_id`, resample from the same split, count, and fail the run only above a configured threshold. Counts and IDs go into the cohort report | — |
| D26 | The model data boundary has **two tables**. The observation table has one row per `snip_id`. The asset table has one row per `(snip_id, snip_product_key, z_index)`, with `z_index` null for projections. `snip_id` remains the embryo-time observation identity; product and plane remain explicit columns | — |
| D27 | The vanilla phase selects exactly one configured BF projection asset per observation. The data layer must never assume that `snip_id` maps intrinsically to one file; future z-aware selectors may return one plane, several planes, or an ordered stack without changing biological identity | — |
| D28 | Core batches carry acquisition covariates even when the vanilla model ignores them: `incubation_temperature_c`, `elapsed_time_s`, `time_index`, stage value/status/version, and the applicable start-age provenance. Contrastive-loss temperature must use a distinct name | — |
| D29 | **Track A exits at the vanilla end-to-end gate**: real pipeline-backed rows → deterministic dataset/loaders → vanilla `Trainer.fit` train+validation → finite losses and parameter update → checkpoint save/reload → held-out encode/reconstruct. Online W&B is not required | — |
| D30 | Corpus rendering and regeneration are an **external input**, reported active by Nick on 2026-08-27. The core refactor owns schema preflight of arriving artifacts, not renderer acceptance, rerun orchestration, or pixel-derived product generation | — |
| D31 | Metric grouping, relation semantics, and developmental-age constraints are explicitly **test-only stubs** for Track A plumbing. Stub configurations must be unmistakably named and cannot be represented as scientific runs | — |
| D32 | Cohort-selection machinery is built before the final policy is chosen. The final ordered experiment list, QC policy, stage policy, and exact selected IDs are frozen only after regenerated pixel-dependent QC is available. A named temporary smoke-test policy is allowed | — |
| D33 | Run provenance is part of acceptance, not later cleanup: resolved config, selected observation/asset keys, identity-keyed splits, source fingerprints, cohort counts, mapping policy, and adapter revision are written locally even when W&B is disabled | — |
| D34 | **Pre-A3 Track C mechanism exception.** C1's pure mapping/relation mechanism and C3's pure SupCon `L_out` kernel may be implemented with synthetic fixtures before real A3 acceptance, on an isolated metric feature branch. C2, dataset/pair-loader integration, scientific presets, real metric runs, and weight/temperature tuning remain gated on A3 | Nick, 2026-08-28 |
| D35 | **Metric compatibility candidate, not final science policy.** `uncertain × anything` is excluded, including `uncertain × uncertain`; the same non-uncertain group is positive; different crispant groups retain the legacy exclusion pending explicit review; a positive class relation outside the configured loss-age window is negative. Uncovered labels or class relations fail preflight instead of silently defaulting to negative. Before scientific ratification, inspect oddly labeled controls and the same chemical inhibitor represented at different application times | Nick, 2026-08-28 |
| D36 | **One lead controls metric delegation and integration.** The lead publishes one exact base commit, owns one metric integration branch/worktree, dispatches at most one worker for each non-overlapping slice, and alone integrates worker commits. Nick does not independently commission duplicate workers for the same slice | Nick, 2026-08-28 |

`verified by` is filled in by the pytest `decision` marker. An empty cell means **asserted but
unproven** — the count of empty cells is the headline line in generated `STATUS.md`.

## Superseded

**D3 / D14 — add explicit optical conditioning.** Still not actionable. The current snip writer now
carries explicit source/snip scale
(`src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/snip_identity_contract.py:278-282`),
but the accepted vanilla model does not add optical-conditioning architecture. `microscope_id`,
`objective_magnification`, and physical `z_position` remain unavailable at the model boundary; never
derive, default, or impute them. D28's temperature and elapsed time are acquisition covariates, not a
revival of optical conditioning.

**O9 — "FF is done" convergence criterion.** Deleted permanently.

## Corrections to prior records

**The 7.8 µm/px figure was struck in error on 2026-08-18 and is reinstated.** 7.8 was the value the
replacement pipeline was actually writing; the 6.5→7.8 drift together with the 75→20 blend-radius
drift *is* the snip regression. The earlier strike would have caused a reader to dismiss the most
important open issue in the project.

**Optical covariates were recorded inconsistently.** The 2026-08-19 stored-corpus audit found no
scale fields (`reports/PIPELINE_RECON.md:979-983`); current writer contracts now carry two explicit
scale fields. Stored old artifacts and current writer capability are different claims. See
Superseded.

**Two closed items sat in the open list**: the ImageFolder boundary (closed by D20) and run-artifact
layout (closed by D16/D17).

**The former corpus-regeneration track is no longer owned by this refactor.** Nick reported on
2026-08-27 that all pixel-dependent quantities are being regenerated. D30 records the resulting
scope boundary; schema preflight remains Track A work.
