# AGENTS.md — MorphSeq latent morphology model

Location: `docs/refactors/core-model/AGENTS.md`. Binding on every agent.
Companions in the same folder: `DECISIONS.md` (ledger), `PLAN.md` (scope, issues, threads),
`STATUS.md` (generated — do not edit), `contracts/MANIFEST_SCHEMA.md` (binding interface).

**Evidence convention.** Every factual claim in a written document carries its source inline — a
command, a `file:line`, or a measurement date — or is explicitly marked as inference. A claim that
cannot carry a source belongs in generated `STATUS.md`, not in prose.

## Current state

We are migrating `src/core` off a legacy `ImageFolder` + CSV-bundle training layout onto a
**manifest-backed dataset** built from `src/data_pipeline` outputs. Design rationale:
`NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md`, in this folder. Ratified decisions: `DECISIONS.md`.
If this file and another document disagree, this file wins — and report the disagreement.

## Invariants

1. **IDs are opaque strings.** Never parse, slice, or reconstruct one. Use the pipeline's explicit
   `physical_embryo_id`; never derive it as `snip_id[:-6]`. Never encode new semantics into an ID —
   image product type, focus, and channel belong in **columns**, not identifiers.
2. **Never glob the pipeline output tree.** Experiments come from an explicit ordered config list.
3. **One ordering.** The resolved manifest's row order is the dataset order. No second glob, no
   `ImageFolder` index, no filename parsing, no parallel array indexed alongside it. Datasets are
   plain `torch.utils.data.Dataset` over manifest rows — `ImageFolder` is deleted, not deprecated.
4. **Splits are group-disjoint at `physical_embryo_id`**, persisted by ID, and stable both under
   manifest reordering and under cohort growth.
5. **`is_valid_snip` ≠ `use_snip`.** Materialised vs passed-QC. Surface both, conflate neither.
6. **Cohort filters are explicit and required, never defaulted.** Channel and `image_product_type`
   both take a list. Phase one is FF-only; a mixed cohort must fail loudly, not silently average
   focal planes with projections.
6b. **Absence is not failure.** QC status and stage status are **three-state**, never boolean.
   `no_artifact` ≠ failed QC (28 experiments, 167,603 snips). `unavailable` ≠ failed staging
   (2 experiments, 20,219 snips with finite stages and no status column). Collapsing either into a
   boolean silently deletes good data.
7. **Filtering policy is configuration.** Any non-default cohort rule is a named, versioned policy
   saved with the run.
8. **`src/data_pipeline` imports are confined to the manifest adapter**, limited to path-resolution
   and contract helpers. Everything else consumes the resolved table.
9. **Preserve the `DatasetOutput` keys and the `self_stats`/`other_stats` metric batch contract**
   during the compatibility phase.
10. **Model input defaults to `[1, 288, 128]` and is configurable.** Pipeline PNGs are 576×256 and
    are deterministically resized at the dataset boundary — on the fly, never via an exported
    training set. Any normalisation tied to image size must derive from `input_dim`, never a literal.
11. **Carry metadata through even when phase one ignores it**: `image_product_type`, the mask path
    (`embryo_mask_snip_path`), and **every individual QC flag column** — not just the `use_snip`
    verdict. Cohort policy must be changeable without re-plumbing the manifest.
11b. **The optical covariates do not exist.** No `micrometers_per_pixel`, `microscope_id`,
    `objective_magnification`, or `z_position` in any pipeline schema. Reserve the names; never
    derive, default, or impute them.
12. **Fail loudly and specifically.** Missing artifact, uncovered relation label, unreadable Parquet
    engine, snip with no mask, anchor with no legal positive — name the offending ID and the policy.
    Never degrade silently; never drop QC because its reader is unavailable.

## Do not touch in phases 0–1

Encoder/decoder architecture · latent biological/nuisance partitioning · reconstruction / KL / GAN /
LPIPS / NT-Xent formulas (the ratified margin-term deletion is the sole exception) · the pipeline's
canonical artifact schemas · the pipeline's legacy-embedding and analysis-ready branches.

## Discuss-first — do not implement unprompted

Segmentation-error mimicry augmentation · the metric relation policy itself · switching the default
brightness augmentation from multiplicative to additive-with-clipping. The *mechanism* for relations
may be built; its *content* is Nick's.

**Resolved 2026-08-19:** the ImageFolder boundary. View generation lives inside the manifest
dataset's `__getitem__`, which has the whole row including `embryo_mask_snip_path`. No separate
abstraction layer.

## Working agreements

- Stay inside your slice's file-ownership fence. Slices run in parallel on separate branches.
- **No `sys.path` / `PYTHONPATH` manipulation.** An unresolvable import is a packaging bug to report.
- Prefer failing a test to weakening it.
- When a spec is ambiguous, take the most conservative reading, implement it, and record the
  ambiguity in your summary. Do not block waiting for an answer.
- Report what you did *not* do. Unfinished items and assumptions beat a clean-looking diff.

## Known traps

- `src/data/` is an empty downstream-analysis shim, not the training data package. `src.core.data` is
  the real one. An import of top-level `data` that "works" resolved to the wrong package.
- `src/vae` is dead for `src/core` but **live for `src/analyze`**. Do not remove it.
- The metric path has **two** mis-wiring causes (`model_configs.py:8` shim import, and
  `dataconfig.target: "BasicDataset"` in the metric hydra config). Fixing one looks like progress.
- `_pixel_scale` is a literal `128*288` at `loss_functions.py:185-187`, independent of `input_dim`.
- `contrastive_transform(target_size=...)` accepts and ignores `target_size`.
- `snip_qc` is Parquet. `pyarrow` 25.0.1 is now installed in `morphseq-env`, but preflight must still
  fail with a specific message if no engine is present — never skip QC silently.
- Booleans may arrive as the strings `"True"` / `"False"`. `bool("False")` is `True`.
- The pipeline's ID grammar has **no focus axis**. If both FF and z-slice products ever coexist,
  `snip_id` may collide. Check rather than assume. `image_product_type` is currently *inferred from
  path segments* — it is not a column yet.
- `is_valid_snip` is **True for all 699,505 rows**. The strict gate reduces to `use_snip`. Keep the
  check; do not assume it filters anything.
- QC schema Q02 (2 experiments) carries only `use_snip` and `qc_fail_reasons` — no per-flag columns.
  A per-flag policy must fail loudly there, never fall back silently.
- The default cohort yields **176,466 rows** from 699,505. A materially different number is a bug.
- Metric pair sampling uses `delta <= time_window`; the loss's target matrix uses
  `delta <= time_window + 1.5`. Existing behaviour — pin it in a test, do not "fix" it.
- `initialize_model` does **not** throw on a missing `metric_array` — it is `hasattr`-guarded
  (`run_utils.py:339-340`), so absence is silent.
