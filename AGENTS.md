# AGENTS.md — MorphSeq latent morphology model

Location: repo root `AGENTS.md`, symlinked as `CLAUDE.md`. Binding on every agent.
Companions in `docs/refactors/core-model/`: `README.md` (index), `DECISIONS.md` (ledger),
`PLAN.md` (scope, issues, order of operations), `STATUS.md` (generated — do not edit),
`contracts/MANIFEST_SCHEMA.md` (binding interface).

**Nothing in `docs/refactors/core-model/_archive/` describes current state.** Do not cite an
archived document as evidence for a claim about code.

**Evidence convention.** Every factual claim in a written document carries its source inline — a
command, a `file:line`, or a measurement date — or is explicitly marked as inference. A claim that
cannot carry a source belongs in generated `STATUS.md`, not in prose.

## Current state

We are migrating `src/core` off a legacy `ImageFolder` + CSV-bundle training layout onto a
**manifest-backed dataset** built from `src/data_pipeline` outputs. Design rationale:
`docs/refactors/core-model/_archive/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md` (archived — rationale
still sound, state claims stale). Binding interface: `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md`.
Ratified decisions: `docs/refactors/core-model/DECISIONS.md`.

**Phase 1 (the manifest data layer) is not built.** A prior status report claimed otherwise across
four commits that do not exist. Verify before building on any claim of implementation state:
`docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md`.
If this file and another document disagree, this file wins — and report the disagreement.

## Invariants

1. **IDs are opaque strings.** Never parse, slice, or reconstruct one. Use the pipeline's explicit
   `physical_embryo_id`; never derive it as `snip_id[:-6]`. Never encode new semantics into an ID —
   image product type, focus, and channel belong in **columns**, not identifiers.
2. **Never glob the pipeline output tree.** Experiments come from an explicit ordered config list.
3. **One ordering.** The observation table and asset table have deterministic order; a configured
   resolved sample view defines dataset order. No second glob, no `ImageFolder` index, no filename
   parsing, no parallel array indexed alongside it. Datasets are plain `torch.utils.data.Dataset`
   over resolved sample rows — `ImageFolder` is deleted, not deprecated. Binding grains:
   `docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md` v2.0.
4. **Splits are group-disjoint at `physical_embryo_id`**, persisted by ID, and stable both under
   manifest reordering and under cohort growth.
5. **`is_valid_snip` ≠ `use_snip`.** Materialised vs passed-QC. Surface both, conflate neither.
6. **Cohort and asset selectors are explicit and required, never guessed.** Experiment, product key,
   z mode, QC, stage, and covariate policies are configuration. Vanilla phase one selects exactly
   one configured BF projection asset per observation; it never silently averages or chooses among
   products/planes.
6b. **Absence is not failure.** QC status and stage status are **three-state**, never boolean.
   In the audited pre-regeneration corpus, `no_artifact` ≠ failed QC (28 experiments, 167,603
   snips) and `unavailable` ≠ failed staging (2 experiments, 20,219 snips with finite stages and no
   status column; `docs/refactors/core-model/reports/PIPELINE_RECON.md:893-932`). Collapsing either
   into a boolean silently deletes good data.
7. **Filtering policy is configuration.** Any non-default cohort rule is a named, versioned policy
   saved with the run.
8. **`src/data_pipeline` imports are confined to the manifest adapter**, limited to path-resolution
   and contract helpers. Everything else consumes the resolved table.
9. **Preserve the `DatasetOutput` keys and the `self_stats`/`other_stats` metric batch contract**
   during the compatibility phase.
10. **Model input defaults to `[1, 288, 128]` and is configurable.** Pipeline PNGs are 576×256 and
    are deterministically resized at the dataset boundary — on the fly, never via an exported
    training set. Any normalisation tied to image size must derive from `input_dim`, never a literal.
11. **Carry metadata through even when phase one ignores it**: product key, nullable `z_index`, the
    mask path, temperature, elapsed time, stage value/status/version, and **every individual QC flag
    column** — not just `use_snip`. Cohort policy and future z selection must be changeable without
    re-plumbing the adapter.
11b. **Do not fabricate optical covariates.** Current snip inventory explicitly carries source and
    snip scale (`src/data_pipeline/object_extraction/segmentation/physical_embryo_registry/snip_identity_contract.py:278-282`).
    Carry those declared fields. `microscope_id`, `objective_magnification`, and physical
    `z_position` are not model-boundary inputs; never derive, default, or impute them. `z_index` is a
    plane index, not a physical z position.
12. **Fail loudly and specifically.** Missing artifact, uncovered relation label, unreadable Parquet
    engine, snip with no mask, anchor with no legal positive — name the offending ID and the policy.
    Never degrade silently; never drop QC because its reader is unavailable.

## Do not touch in Track A

Encoder/decoder architecture · latent biological/nuisance partitioning · reconstruction / KL / GAN /
LPIPS / NT-Xent formulas · the pipeline's canonical artifact schemas · the pipeline's
legacy-embedding and analysis-ready branches. After Track A3 is accepted, Track C may implement the
ratified D23 `L_out` change and no unrelated loss/architecture change.

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
- The old import and metric-Hydra target mis-wirings were fixed in Phase 0
  (`docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md:182-271,509-530`). Do not redo them.
  The live data blocker is the legacy `ImageFolder`/positional path.
- `contrastive_transform(target_size=...)` accepts and ignores `target_size`.
- `snip_qc` is Parquet. `pyarrow` 25.0.1 is now installed in `morphseq-env`, but preflight must still
  fail with a specific message if no engine is present — never skip QC silently.
- Booleans may arrive as the strings `"True"` / `"False"`. `bool("False")` is `True`.
- `snip_id` is intentionally not an asset key. One observation may have several products/planes.
  Observation key: `snip_id`; asset key: `(snip_id, snip_product_key, z_index)`. Product and plane
  are explicit columns; never deduplicate on bare `snip_id` or take the first asset row.
- `is_valid_snip` was **True for all 699,505 rows** in the audited pre-regeneration corpus
  (`docs/refactors/core-model/reports/PIPELINE_RECON.md:899-905`). Keep the check; do not assume either
  that it filters nothing in regenerated data or that it is equivalent to QC.
- QC schema Q02 (2 experiments) carries only `use_snip` and `qc_fail_reasons` — no per-flag columns.
  A per-flag policy must fail loudly there, never fall back silently.
- The audited pre-regeneration default cohort yielded **176,466 rows** from 699,505
  (`docs/refactors/core-model/reports/PIPELINE_RECON.md:922-932`). It is a historical adapter fixture,
  not the regenerated-corpus acceptance count.
- Legacy metric pair sampling uses `delta <= time_window`; its loss target uses
  `delta <= time_window + 1.5`. Preserve it only for the Track A test stub. Track C must configure and
  validate the two windows explicitly; the new stage estimand does not inherit 1.5 by assumption.
- `initialize_model` does **not** throw on a missing `metric_array` — it is `hasattr`-guarded
  (`run_utils.py:339-340`), so absence is silent.
