# Phase 1 agent briefs — Codex 5.6

Three slices, parallel, branch each. All three are written against `docs/refactor/MANIFEST_SCHEMA.md`,
which is the interface between them — read it first, treat it as binding, and report any place it is
wrong rather than deviating silently.

**Prerequisite.** Phase 0 must be merged: the `src.core.*` packaging fix, the Hydra config group, and
image size as a config parameter. If any is missing, stop and report — do not work around it.

| slice | owns | must not touch |
|---|---|---|
| **1A** | new `src/core/data/pipeline_manifest.py`, new `src/core/data/pipeline_contracts.py`, `src/core/data/dataset_configs.py`, new Hydra data-config group, `tests/core/test_manifest*.py` | `dataset_classes.py`, `data_transforms.py`, `pl_wrappers.py`, `callbacks.py` |
| **1B** | `src/core/data/dataset_classes.py`, `src/core/data/data_transforms.py`, `src/core/lightning/pl_wrappers.py`, deletion of the legacy loader, `tests/core/test_dataset*.py` | `pipeline_manifest.py`, `pipeline_contracts.py`, `callbacks.py` |
| **1C** | `src/core/lightning/callbacks.py`, new `src/core/run/provenance.py`, `tests/core/test_provenance.py` | everything else |

Fill `<PIPELINE_OUTPUT_ROOT>` and `<EXPERIMENT_IDS>` before sending.

---

## Slice 1A — the manifest adapter

> Read `docs/refactor/AGENTS.md`, `docs/refactor/DECISIONS.md`, and
> `docs/refactor/MANIFEST_SCHEMA.md`. The schema document is your specification; this brief is how to
> build it.
>
> **Goal.** A module that turns `pipeline_output_root` plus an explicit ordered experiment list into
> one resolved training table, and a structured validation report. It is the sole authority for image
> paths, metadata, splits, and row order. Nothing downstream reaches back into `src/data_pipeline`.
>
> **Build two files.**
>
> `src/core/data/pipeline_contracts.py` — declare, per source (`snip_inventory`, `stage_predictions`,
> `snip_qc`, `plate_metadata`), the required columns, optional columns, expected dtypes, and a
> contract-version marker. Derive these from the pipeline's own writers and validators; cite the
> source file and symbol in a comment for each. The real cohort has **2 inventory schemas** (same 22
> columns, order differs), **2 stage schemas** (S02 lacks `stage_prediction_status`), **3 QC schemas**
> (Q02 lacks all per-flag columns), and **38 plate schemas** — so validation must be tolerant and
> *reporting*, never fail-on-first-difference. Return a structured report, don't raise.
>
> `src/core/data/pipeline_manifest.py` — the adapter:
>
> 1. accept `pipeline_output_root`, explicit ordered `experiment_ids`, channel list, product-type
>    list, QC policy name, stage policy, metric-group mapping path, split ratios;
> 2. resolve sources through the pipeline path registry
>    (`pipeline_orchestrator/orchestration/paths.py`) or a thin adapter over it. Imports from
>    `src/data_pipeline` are confined to this module and limited to path/contract helpers;
> 3. read each experiment's merged inventory as the base table; **fail per-experiment with a named
>    error** when an artifact is missing — 15 of 148 experiments have no inventory and 28 more have no
>    QC artifact, so this path is exercised immediately, not hypothetically;
> 4. normalise booleans safely (`bool("False")` is `True` — the strings appear in real data);
> 5. join stage and QC one-to-one on `snip_id` with coverage reports; join plate many-to-one on
>    `well_id`, asserting exactly one plate row per selected well;
> 6. materialise the three-state `stage_status` and `qc_status` exactly as the schema document
>    specifies — these are the two places where a boolean silently destroys real data;
> 7. apply explicit filters (channel, product type, validity, QC policy, stage policy) with per-filter,
>    per-experiment, per-reason counts;
> 8. assign splits by `blake2b(physical_embryo_id)`, group-disjoint, stable under cohort growth;
> 9. resolve `metric_group` from the required config mapping, failing loudly and naming any uncovered
>    values;
> 10. return the table plus a structured validation summary, and expose the source-artifact inventory
>     (path, size, mtime, row count per file) for slice 1C to consume.
>
> **Design constraint that will bite later if you miss it:** the manifest builder must be usable for
> *inference* cohorts too, not just training. QC filtering, split assignment, and metric-group
> resolution must each be independently switchable off. Do not bake train-only assumptions in.
>
> **Environment.** `pyarrow` 25.0.1 is now installed in `morphseq-env`. Preflight must still fail with
> a specific, actionable message if no Parquet engine is present — silently skipping QC is not an
> acceptable degradation.
>
> **Tests.** Contract tolerance across all observed schema variants; per-experiment missing-artifact
> errors; boolean string parsing; one-to-one join coverage reporting both missing and extra IDs;
> three-state status materialisation; split group-disjointness; **split stability when the cohort grows
> by one experiment** (build for {A,B}, then {A,B,C}, assert no shared embryo changed split); uncovered
> `metric_group` raises naming the values.
>
> **Regression check against real data.** With the default policy over `<EXPERIMENT_IDS>`, expect
> **176,466 rows**. A materially different number means a bug — investigate before proceeding.
>
> **In your summary:** which pipeline source files each contract declaration came from; every
> discrepancy between declared contract and real artifacts; the per-filter counts from a real run; and
> whether the `src.data_pipeline` path import is clean in the training environment.

---

## Slice 1B — manifest-backed datasets, transforms, loaders

> Read `docs/refactor/AGENTS.md` and `docs/refactor/MANIFEST_SCHEMA.md`. You consume the resolved
> table; you do not build it. Slice 1A owns the adapter — write against the schema document and use a
> synthetic table conforming to it until 1A lands.
>
> **Goal.** Replace `ImageFolder` with plain `torch.utils.data.Dataset` classes indexed by manifest
> row, fix the transform contract, make loaders split-local, and delete the legacy path.
>
> **1. Datasets.** A basic dataset and a metric dataset, both plain `torch.utils.data.Dataset` over
> manifest rows. Row `i` is sample `i`; `__getitem__` opens exactly the path in that row. Preserve the
> existing `DatasetOutput` keys and the `self_stats`/`other_stats` metric batch contract so the model
> and losses stay untouched.
>
> The row carries `embryo_mask_snip_path`, so `__getitem__` can open the snip *and* its mask together.
> That is deliberate: it is where mask-aware augmentation will live later. Load the mask lazily —
> phase one does not use it, and doubling I/O for an unused array is not free.
>
> **2. Worker memory.** The manifest is ~700k rows. A pandas object-dtype DataFrame is copied into
> every fork-based worker. Convert model-facing columns to numpy arrays or categorical codes at
> dataset construction and hold no DataFrame reference in `__getitem__`.
>
> **3. Pair sampling must not be O(N) per item.** The current implementation builds full-length boolean
> arrays for every sample. At 176k+ rows inside dataloader workers that dominates wall-clock. Build,
> once per split, an index keyed by `(metric_group, age_bin)` plus a same-`physical_embryo_id` index;
> sample from buckets, checking neighbouring age bins. Age bins sized to `time_window`. When no legal
> positive exists, raise naming the anchor `snip_id` and the selection policy — never a bare NumPy
> sampling exception.
>
> Preserve, and pin in a test, the existing asymmetry: pair sampling uses `delta <= time_window` while
> the loss's batch-wide target matrix uses `delta <= time_window + 1.5`. This is existing behaviour.
> Do not "fix" it.
>
> **4. Transforms.** One shared deterministic decoder:
> `open -> convert("L") -> resize to model (H, W) -> float32 tensor in [0, 1]`.
> **Pin the resize explicitly** — name the library, the interpolation enum, and the antialias flag, and
> state whether resize happens on the PIL image or the tensor. 576×256 → 288×128 is an exact 2×
> downsample and PIL and torchvision do not default to the same filter. Match
> `src/data_pipeline/feature_extraction/legacy_embeddings/transforms.py` and say so in a comment.
> Fix `contrastive_transform` currently accepting and **ignoring** `target_size`.
>
> Basic mode stops at the decoder. Metric mode applies augmentation *around* that size contract, and
> each of the two views is augmented independently — verify whether the current code does this and
> report what you find.
>
> **5. Brightness augmentation — add, do not switch on.** Measured across the cohort, the experiment
> signature in intensity is concentrated at the black end (η² by experiment: `min` 0.477,
> `zero_fraction` 0.242) rather than saturation (0.047). Multiplicative brightness jitter does not span
> that axis; **additive offset followed by clipping to [0, 1]** does. Implement it as a configurable
> augmentation, **defaulted off**, alongside the existing brightness jitter. Switching the default is
> Nick's call, not yours — flag it in your summary.
>
> **6. Loaders.** Use split-specific dataset views rather than a full-dataset positional sampler. That
> makes candidate selection naturally split-local and lets Lightning apply a distributed sampler
> safely. Add a real test dataloader, or explicitly declare test inference out of phase-one scope —
> do not let test rows silently reuse eval pairing state.
>
> **7. Delete the legacy path** in one reviewable commit: `make_seq_key`, the `ImageFolder` subclasses,
> the positional split generation, and `split_indices.pkl` writing. No config switch, no fallback. Do
> not touch `src/data` or `src/vae` — `src/vae` is live for `src/analyze`.
>
> **Tests.** Manifest row `i`, loaded pixel path, and returned metadata all refer to the same
> `snip_id`; basic tensors `[1, 288, 128]` float32 finite in `[0,1]`; metric items `[2, 1, 288, 128]`;
> deterministic output matches the legacy-embedding transform (justify any tolerance with the observed
> max deviation); augmentation applied after the size contract and per-view; positives never cross
> splits; missing-positive raises actionably; pair selection complexity documented and not O(N).
>
> **In your summary:** the exact interpolation/antialias configuration and what you matched it against;
> the observed max deviation from the legacy transform; whether per-view augmentation was already
> correct; your pair-index structure and its complexity; and confirmation the legacy path is
> unreachable.

---

## Slice 1C — run provenance

> Read `docs/refactor/AGENTS.md` and `docs/refactor/DECISIONS.md` (D16, D17, D19).
>
> **Goal.** Replace `split_indices.pkl` — positional integer arrays that become meaningless the moment
> the manifest changes — with an identity-based provenance bundle, logged to W&B.
>
> **Write one directory per run**, at a path given by a single config key `run_artifacts_dir`:
>
> - `resolved_config.yaml` — the fully resolved run config
> - `snip_ids.txt.gz` — the selected snip IDs, sorted
> - `split_assignments.csv` — keyed by `physical_embryo_id`
> - `metric_group_map.csv` — the exact mapping table used
> - `sources.json` — path, size, mtime, and row count for every source artifact consumed
> - `cohort_report.json` — counts by filter and exclusion reason, per experiment
> - the adapter's git SHA
>
> **Do not save the full manifest.** ~700k rows is roughly 350 MB per run. The snip ID list plus the
> config regenerates it, `sources.json` detects when it can't, and the git SHA covers the case where
> the adapter's own logic changed.
>
> **Hashing.** Hash the source artifacts (~500 MB total, seconds) and the sorted snip ID list (~18 MB,
> sub-second) with blake2b. **Do not hash image content** — 699,505 files at ~28 GB is minutes to hours
> and buys nothing the source hashes don't. Make that a comment so nobody adds it later.
>
> **W&B.** The bundle is logged as a **W&B Artifact**; the local directory is a staging area. Nothing
> may be written relative to Hydra's cwd or reach into Lightning's `default_root_dir` internals — the
> local run-directory layout is going to be redesigned, and provenance code must not need to change
> when it is. One config key in, no structural assumptions.
>
> **Tests.** A run snapshot reconstructs the exact selected snip-ID set without positional indices;
> `sources.json` detects a changed source artifact; the bundle round-trips; nothing is written outside
> `run_artifacts_dir`; W&B logging is mockable and the bundle is still written when W&B is disabled.
>
> **In your summary:** the bundle size for a full-cohort run; hashing wall-clock; and any place you had
> to reach into Hydra or Lightning internals despite the constraint.
