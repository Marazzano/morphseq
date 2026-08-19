# AGENTS.md — MorphSeq latent morphology model

Binding on every agent. Read `DECISIONS.md` alongside this file, in the same folder.

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
   `ImageFolder` index, no filename parsing, no parallel array indexed alongside it.
4. **Splits are group-disjoint at `physical_embryo_id`**, persisted by ID, and stable both under
   manifest reordering and under cohort growth.
5. **`is_valid_snip` ≠ `use_snip`.** Materialised vs passed-QC. Surface both, conflate neither.
6. **Cohort filters are explicit and required, never defaulted.** Channel and `image_product_type`
   both take a list. Phase one is FF-only; a mixed cohort must fail loudly, not silently average
   focal planes with projections.
7. **Filtering policy is configuration.** Any non-default cohort rule is a named, versioned policy
   saved with the run.
8. **`src/data_pipeline` imports are confined to the manifest adapter**, limited to path-resolution
   and contract helpers. Everything else consumes the resolved table.
9. **Preserve the `DatasetOutput` keys and the `self_stats`/`other_stats` metric batch contract**
   during the compatibility phase.
10. **Model input defaults to `[1, 288, 128]` and is configurable.** Pipeline PNGs are 576×256 and
    are deterministically resized at the dataset boundary — on the fly, never via an exported
    training set. Any normalisation tied to image size must derive from `input_dim`, never a literal.
11. **Carry metadata through even when phase one ignores it**: optical covariates
    (`micrometers_per_pixel`, `objective_magnification`, `microscope_id`, `z_position`),
    `image_product_type` / `projection_method`, and the mask path. Re-plumbing later is the expensive
    outcome; an unused column costs nothing.
12. **Fail loudly and specifically.** Missing artifact, uncovered relation label, unreadable Parquet
    engine, snip with no mask, anchor with no legal positive — name the offending ID and the policy.
    Never degrade silently; never drop QC because its reader is unavailable.

## Do not touch in phases 0–1

Encoder/decoder architecture · latent biological/nuisance partitioning · reconstruction / KL / GAN /
LPIPS / NT-Xent formulas (the ratified margin-term deletion is the sole exception) · the pipeline's
canonical artifact schemas · the pipeline's legacy-embedding and analysis-ready branches.

## Discuss-first — do not implement unprompted

The ImageFolder boundary (where view-generation sits, how the mask reaches it) · segmentation-error
mimicry augmentation · the metric relation policy itself. The *mechanism* for relations may be built;
its *content* is Nick's.

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
- `snip_qc` is Parquet; `morphseq-env` has no Parquet engine, `vae-env-cluster` does.
- Booleans may arrive as the strings `"True"` / `"False"`. `bool("False")` is `True`.
- The pipeline's ID grammar has **no focus axis**. If both FF and z-slice products ever coexist,
  `snip_id` may collide. Check rather than assume.
- Metric pair sampling uses `delta <= time_window`; the loss's target matrix uses
  `delta <= time_window + 1.5`. Existing behaviour — pin it in a test, do not "fix" it.
- `initialize_model` does **not** throw on a missing `metric_array` — it is `hasattr`-guarded
  (`run_utils.py:339-340`), so absence is silent.
