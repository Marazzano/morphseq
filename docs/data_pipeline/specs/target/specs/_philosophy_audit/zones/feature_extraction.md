# Philosophy Audit — feature_extraction

**Auditor:** claude-sonnet-4-6, 2026-06-24
**Rubric:** `pipeline_file_philosophy.md` (two hard constraints + 14-pt checklist) +
`feature_world.md` (Shared Feature And QC Stage Pattern / Code Organization Pattern)
**Zone:** `src/data_pipeline/feature_extraction/` (~66 files)

---

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 5 |
| MAJOR | 10 |
| MINOR | 7 |

**Verdict:** The per-product subfolder packages (`mask_geometry/`, `curvature_metrics/`,
`pose_kinematics/`, `stage_predictions/`, `fraction_alive/`, `consolidated_features/`,
`legacy_embeddings/`) range from good to excellent. The blocking problems are concentrated in
(1) the surviving legacy layout — `core/`, `entrypoints/`, `pipelines/`, and root-level `.py`
files — which were supposed to be deleted when their per-product successors went live, and
(2) a dead import in the legacy flat entrypoint plus a broken path-resolution call in the
active `fraction_alive/compute.py`. The new product contracts correctly import spine columns
from the minting site; the one internal re-declaration in `mask_geometry/contract.py` is a
private alias, not a public duplicate, but it still warrants cleanup.

---

## Findings

---

### BLOCKER

---

#### B1 — `fraction_alive/compute.py:20,51-58` — imports deprecated `path_contracts`, calls will raise at runtime

```python
from data_pipeline.shared.path_contracts import resolve_data_root_relative_path
...
resolved = resolve_data_root_relative_path(path)
```

**Violated rule:** Hard Constraint 1 (paths from `paths.py`, not hidden globals); philosophy
"Explicit signatures, no haunted globals" — `resolve_data_root_relative_path` is now a
**raise-on-call tripwire** (see `shared/path_contracts.py:26-28`). Importing it emits a
`RuntimeWarning` on module import; calling it raises `RuntimeError` at runtime. Every invocation
of `compute_fraction_alive_features` will crash.

**Severity:** BLOCKER — the active compute path is broken.

**Fix:** Remove the import and the `_resolve` inner function entirely. The `embryo_mask_snip_path`
stored in `snip_inventory` is already documented as relative to `output_root`; the entrypoint
passes `output_root` explicitly — use it directly with `Path(output_root) / Path(path)` and
remove the dead `resolve_data_root_relative_path` call. Alternatively use the path column
directly as an absolute path if `snip_processing` already stored it absolute.

---

#### B2 — `entrypoints/compute_fraction_alive.py:8,31` — imports a deleted function (`load_auxiliary_masks_manifest`)

```python
from data_pipeline.feature_extraction.io.loaders import (
    load_auxiliary_masks_manifest,   # ← this function was deleted; only a NOTE comment remains
    ...
)
...
auxiliary_masks_df = load_auxiliary_masks_manifest(args.auxiliary_masks_manifest)
```

**Violated rule:** Dead legacy module that was explicitly retired. `io/loaders.py:76` carries a
comment stating the function was retired when auxiliary masks moved to per-snip grain.

**Severity:** BLOCKER — this entire legacy entrypoint raises `ImportError` on load.

**Fix:** The whole `entrypoints/compute_fraction_alive.py` file should be deleted (per
`feature_world.md` Legacy Domain Retirement: `feature_extraction/entrypoints/` must be empty
once per-product subfolders are live). The active path is `fraction_alive/entrypoint.py`.

---

#### B3 — `pipelines/compute_stage_predictions.py:56-137` — raw artifact-path strings, no registry

```python
out_dir = output_root / "computed_features" / exp / "per_well" / well / "contracts"
...
out_pq = out_dir / "stage_predictions.parquet"
out_csv = out_dir / "stage_predictions.csv"
flag = out_dir / ".stage_predictions.computed"
flag.write_text("computed\n")
```

**Violated rule:** Hard Constraint 1 — raw path strings never in rules, tasks, or modules.
`"computed_features"`, `"per_well"`, `"contracts"`, `".stage_predictions.computed"` are all
hardcoded. The sentinel file uses the `write_text("computed\n")` pattern instead of the
`validated_path(...)` helper from orchestration.

**Severity:** BLOCKER — this is the exact anti-pattern the registry constraint prohibits. The
path this module writes will silently diverge from any Snakemake rule that reads from the registry.

**Fix:** Delete this whole `pipelines/` directory. The target implementation is
`stage_predictions/entrypoint.py` + `stage_predictions/compute.py` which already exist and are
correct. If a merge step is still needed, it belongs under the registry pattern (a
`per_well_then_merge` fanout entry in `PIPELINE_STEPS`).

---

#### B4 — `pipelines/merge_stage_predictions.py:17-46` — raw path strings, disk-scan without registry

```python
exp_root = output_root / "computed_features" / exp
per_well_root = exp_root / "per_well"
contracts_dir = exp_root / "contracts"
...
p = per_well_root / w / "contracts" / "stage_predictions.parquet"
```

**Violated rule:** Hard Constraint 1 (raw path strings, not via `paths.py` helpers). Also
violates "Planning-time helpers stay pure; runtime helpers may read disk — and say so": this
function scans `per_well_root.iterdir()` without being named a runtime/disk-scan helper and
without a docstring announcing its moment.

**Severity:** BLOCKER — same root cause as B3; part of the same non-registry `pipelines/` tree.

**Fix:** Delete alongside B3.

---

#### B5 — `core/`, `entrypoints/` (legacy flat layout), `consolidate_features.py`, `mask_geometry_metrics.py`, `stage_inference.py`, `pose_kinematics_metrics.py`, `config.py` (root-level) — dead legacy modules not deleted

Per `feature_world.md` (Legacy Domain Retirement):

> `feature_extraction/core/` and `feature_extraction/entrypoints/` are the legacy layout for
> features. Each per-product subfolder that is implemented retires its counterpart in those old
> directories. The old flat modules at `feature_extraction/` package root must also be deleted
> once their product subfolders are live.

All per-product subfolders are live. The surviving dead modules are:

| Dead module | Status |
|---|---|
| `core/__init__.py`, `core/mask_geometry.py`, `core/fraction_alive.py`, `core/stage_inference.py`, `core/consolidate_features.py`, `core/pose_kinematics.py`, `core/curvature_metrics.py` | 1-line re-export shims or full implementations; should be deleted |
| `entrypoints/compute_mask_geometry.py`, `entrypoints/compute_fraction_alive.py`, `entrypoints/consolidate_features.py`, `entrypoints/compute_stage_predictions.py`, `entrypoints/compute_curvature_metrics.py`, `entrypoints/compute_pose_kinematics.py` | old flat entrypoints; should be deleted |
| `consolidate_features.py` (root) | old flat consolidate logic with metadata join smell (see M3); should be deleted |
| `mask_geometry_metrics.py` (root) | old flat pure function; is still imported by `mask_geometry/compute.py` and `pose_kinematics/compute.py` as a delegate — keep ONLY if left as a pure-function leaf, but move into a product-private helper |
| `stage_inference.py` (root) | still imported by `stage_predictions/compute.py` (`predict_stage_hpf`) — keep only the pure function, delete legacy batch helpers |
| `pose_kinematics_metrics.py` (root) | still imported by `pose_kinematics/compute.py` — same disposition |
| `config.py` (root) | domain-level config with old `DEFAULT_FEATURE_EXTRACTION_CONFIG` dict; still imported by `entrypoints/compute_stage_predictions.py` which itself should be deleted |
| `pipelines/` directory | three files, all wrong (see B3, B4, and M4) |

**Severity:** BLOCKER — leaving dead modules alongside the new product folders creates competing
sources of truth. The spec is explicit: retirement is part of done, not a separate pass.

**Mitigating note:** The `core/curvature_metrics.py` is a real implementation (not a shim)
despite living in `core/`. It would need to be moved to `curvature_metrics/` private helpers or
kept as a temporary delegate. All others in `core/` are one-line shims pointing elsewhere.

---

### MAJOR

---

#### M1 — `mask_geometry/contract.py:29` — local `_SPINE_COLUMNS` alias duplicates the concept

```python
_SPINE_COLUMNS: tuple[str, ...] = SNIP_ID_SPINE_COLUMNS + _FRAME_DERIVED_COLUMNS
```

**Violated rule:** "One concept, built in exactly one place" — `_SPINE_COLUMNS` in
`mask_geometry/contract.py` is a private alias that combines `SNIP_ID_SPINE_COLUMNS` (imported
correctly from the minting site) with `_FRAME_DERIVED_COLUMNS`. It is then used to build
`MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS`. `shared/feature_table_utils.py` already defines
`SNIP_FEATURE_TABLE_ID_COLUMNS` as exactly this same combination (line 30), and every other
product contract imports from there.

**Severity:** MAJOR — not yet drifted (both definitions agree today), but it is a second home for
the same concept. When `_FRAME_DERIVED_COLUMNS` changes in one place, the other will silently lag.

**Fix:** Remove `_SPINE_COLUMNS` and `_FRAME_DERIVED_COLUMNS` from `mask_geometry/contract.py`.
Import `SNIP_FEATURE_TABLE_ID_COLUMNS` from `shared/feature_table_utils` instead (same as
`curvature_metrics/contract.py`, `fraction_alive/contract.py`, etc.).

---

#### M2 — `mask_geometry/compute.py:46` — private `_pixel_size_for_image` duplicates shared helper

```python
def _pixel_size_for_image(frame_inventory_by_image: pd.DataFrame, image_id: str, snip_id: str) -> float:
```

`shared/feature_table_utils.py:102` already defines `pixel_size_for_image` with identical logic.
`mask_geometry/compute.py` has its own private copy that was not removed when the shared version
was created. `curvature_metrics/compute.py` and `pose_kinematics/compute.py` both correctly
import the shared helper; `mask_geometry/compute.py` does not.

**Violated rule:** "One concept, built in exactly one place (no drift)" — two implementations will
diverge on edge-case handling.

**Severity:** MAJOR — correctness risk when one is fixed and the other isn't.

**Fix:** Delete `_pixel_size_for_image` from `mask_geometry/compute.py` and import
`pixel_size_for_image` from `shared/feature_table_utils`.

---

#### M3 — `consolidate_features.py (root):86-102` — analysis-ready join of metadata into feature table

```python
if metadata_df is not None:
    if 'well_id' in metadata_df.columns:
        merge_key = 'well_id'
    ...
    consolidated = consolidated.merge(metadata_df, on=merge_key, ...)
```

**Violated rule:** `feature_world.md` — "Do not join genotype, condition, perturbation, or
`use_snip` into feature products. Those joins belong in `analysis_ready`." Merging
`plate_metadata` (which carries biological condition/treatment context) into the consolidated
feature table violates the feature/analysis-ready boundary.

**Severity:** MAJOR — this is the exact contamination the boundary rule prevents. The root-level
`consolidate_features.py` should be deleted (see B5), but the pattern must not re-appear in
`consolidated_features/compute.py`.

**Fix:** The active `consolidated_features/compute.py` correctly does NOT do this merge. Delete
the root-level legacy file.

---

#### M4 — `pipelines/validate_stage_predictions.py:54-55` — handcrafted sentinel, not via `validated_path()`

```python
output_flag.write_text("validated\n")
```

**Violated rule:** Philosophy "Sidecars are derived, never first-class" — sentinels must be
derived via `validated_path(artifact_path)`, not written as arbitrary text files at arbitrary
paths passed on the CLI. The sentinel naming convention (`artifact.validated`) is part of the
registry contract; `output_flag` could be any path the caller chooses.

**Severity:** MAJOR — this is in the `pipelines/` directory which is slated for deletion, but the
pattern clarification matters: **new entrypoints must use `validated_path(output_csv)` from the
orchestration registry, not their own `write_text` calls**.

---

#### M5 — New product entrypoints do NOT write `.validated` sentinels at all

All five new-product entrypoints (`mask_geometry/entrypoint.py`,
`stage_predictions/entrypoint.py`, `curvature_metrics/entrypoint.py`,
`pose_kinematics/entrypoint.py`, `fraction_alive/entrypoint.py`) write the CSV and return,
but do NOT write the `.validated` sentinel via `validated_path(output_csv)`.

```python
# mask_geometry/entrypoint.py:39-40
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)
# <-- no validated_path(output_csv).write_text("ok\n") call
```

**Violated rule:** `feature_world.md` — "The entrypoint calls `validate_*` and only then writes
the artifact and its `validated_path(...)` marker — an unvalidated write is a contract violation."
A Snakemake rule that depends on the `.validated` sentinel (which is how the registry wires
downstream dependencies) will never find it.

**Severity:** MAJOR — downstream rules depending on the sentinel will block or silently skip.

**Fix:** Each entrypoint's final line should be:
```python
from data_pipeline.pipeline_orchestrator.orchestration.paths import validated_path
validated_path(output_csv).write_text("ok\n")
```
(The `io/writers.py` `write_feature_table` already does this via `_sentinel_path`, but the new
product entrypoints bypass `io/writers.py` entirely.)

---

#### M6 — `io/writers.py:13` — private `_sentinel_path` duplicates `validated_path()` from orchestration

```python
def _sentinel_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".validated")
```

**Violated rule:** "Sidecars are derived via helpers, never hardcoded or registry rows" — the
sentinel path must come from `validated_path(...)` in `orchestration/paths.py`, not a
domain-local reimplementation. If the sentinel naming convention ever changes in orchestration,
`_sentinel_path` will silently diverge.

**Severity:** MAJOR — one concept, two homes.

**Fix:** Import and use `validated_path` from `pipeline_orchestrator.orchestration.paths`.
Delete `_sentinel_path` and `_schema_sidecar_path` from `io/writers.py`.

---

#### M7 — `io/loaders.py` — legacy `time_int` / `frame_index` compat shim and stale join key

```python
# line 38-39
if "time_int" not in df.columns and "frame_index" in df.columns:
    df["time_int"] = pd.to_numeric(df["frame_index"], errors="raise").astype(int)

# line 86
preferred_keys = ["experiment_id", "well_id", "channel_id", "time_int"]
```

**Violated rule:** `pipeline_file_philosophy.md` — "Controlled vocabulary, defined once as
constants — never magic strings"; `feature_world.md` — time axis is `time_index` (not `time_int`
or `frame_index`). The `time_index` standardization memory entry (`project_time_index_standardization.md`)
locks `time_index` as the canonical per-frame axis name. These shims keep the old aliased names
alive in the shared loader.

**Severity:** MAJOR — the legacy compat shims in the shared loader leak stale vocabulary into all
downstream consumers. The loader is supposed to be "boring shared mechanics," not a compatibility
translation layer.

**Fix:** After all callers use `time_index`, delete the `frame_index` shim and update the join
key list. These loaders are also in the legacy flat layout that should eventually be retired when
the new product entrypoints no longer rely on the old loader / merge pattern.

---

#### M8 — `fraction_alive/_legacy_compute.py:67` — inline `snip_id` split to recover `image_id`

```python
image_id = row.get('image_id', snip_id.rsplit('_', 1)[0])
```

**Violated rule:** Hard Constraint 1 — "No id is ever minted or split with an inline f-string or
`.split('_')`." `snip_id.rsplit('_', 1)[0]` is an inline decomposition of a compound identifier
that should come from `shared/identifiers/` parsers.

**Severity:** MAJOR — this is in `_legacy_compute.py` (the file that `compute.py` delegates to
for the core viability calculation). The function `extract_fraction_alive_batch` is itself a
legacy batch function that should be removed, but the id-split pattern is worth calling out
explicitly.

**Fix:** This whole `extract_fraction_alive_batch` function (and the wrapping
`entrypoints/compute_fraction_alive.py`) should be deleted. The active path
(`fraction_alive/compute.py`) does not do this split.

---

#### M9 — `stage_inference.py (root):68` — undeclared `stage_confidence` column in legacy batch

```python
results.append({
    'snip_id': snip_id,
    'predicted_stage_hpf': predicted_hpf,
    'stage_confidence': 1.0,   # ← not in stage_predictions/contract.py
})
```

**Violated rule:** The active `stage_predictions/contract.py` does not declare
`stage_confidence`. The legacy batch adds a column the product contract does not own, which means
any downstream consumer relying on this column from the legacy path will find it absent from the
new path. Also, a constant `1.0` is not a measured feature — it's a placeholder that should
either be in the contract (with meaning) or dropped entirely.

**Severity:** MAJOR — schema mismatch between legacy and new paths. When the new entrypoint
supersedes the old, callers expecting `stage_confidence` will break silently.

**Fix:** Delete `stage_inference.py` (root) once `stage_predictions/` is confirmed live. The
`stage_confidence` column should either be added to `stage_predictions/contract.py` with a
documented meaning or dropped.

---

#### M10 — `features/` package — ghost package creating naming confusion

`src/data_pipeline/features/` exists as a near-empty package (only `__init__.py` and a
`legacy_embeddings/` subdirectory with only `__pycache__`). The spec (`feature_world.md:142-144`)
explicitly states:

> Code package and output stage intentionally differ: source code lives under
> `feature_extraction/`, while feature artifacts land under the `features/` output stage. Do not
> add a new `data_pipeline.features` package for future feature work.

The `features/__init__.py` is empty; the `features/legacy_embeddings/` subdirectory has no `.py`
source files — only compiled `.pyc` caches of what appears to be a prior location of
`legacy_embeddings/`. This is a stale ghost of a previous package layout.

**Severity:** MAJOR — a developer importing `from data_pipeline.features.legacy_embeddings import
...` will get a stale cached module (or ImportError) instead of the correct
`feature_extraction.legacy_embeddings` module. The ghost also contradicts the spec's explicit
"do not add a new `data_pipeline.features` package."

**Fix:** Delete `src/data_pipeline/features/` entirely (including the stale `__pycache__` dirs).
Clear `.pyc` caches. The spec already forbids this package.

---

### MINOR

---

#### m1 — `io/loaders.py` — no module docstring; multi-concern file without section banners

The file mixes: (a) generic table loading, (b) per-product schema validation, (c) path
resolution, (d) a cross-table join function. It has no module docstring and no section banners
separating these concerns.

**Violated rule:** "One file reads top-to-bottom in flow order, with section banners marking each
concern."

**Severity:** MINOR — navigability.

**Fix:** Add a module docstring. Add banners: `# Loaders`, `# Path resolution`, `# Cross-table
joins`.

---

#### m2 — `io/writers.py` — no module docstring; uses domain-local schema `REQUIRED_COLUMNS_FEATURES`

```python
from data_pipeline.schemas.features import REQUIRED_COLUMNS_FEATURES
```

The domain-level `schemas/features.py` is not the per-product contract; `REQUIRED_COLUMNS_FEATURES`
is separate from `CONSOLIDATED_FEATURES_REQUIRED_COLUMNS` owned by
`consolidated_features/contract.py`. Two homes for what should be one authoritative list.

**Severity:** MINOR — potential drift.

**Fix:** `write_consolidated_features_contract` should validate using the product contract's
`validate_consolidated_features` (which already calls the spine validator), then write — not by
embedding a separate schema sidecar from a different source.

---

#### m3 — `fraction_alive/entrypoint.py:25-27` — `output_root: Path | None = None` optional and effectively undocumented

```python
output_root: Path | None = None,
```

`output_root` is accepted but may be `None`. The `compute.py` function then treats `None` as "do
not relativize paths." This is a leaky abstraction: the caller (Snakemake rule) should always
pass `output_root`; making it optional creates a path where the runtime silently skips path
resolution (and hits the `path_contracts` crash — see B1).

**Severity:** MINOR — once B1 is fixed, this becomes a documentation gap rather than a bug.

**Fix:** Make `output_root` required (remove `= None`) once the B1 path-resolution fix is applied.

---

#### m4 — `pipelines/compute_stage_predictions.py:31-35` — `_pipeline_version()` via `git rev-parse`

```python
def _pipeline_version() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[4]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], ...)
```

**Violated rule:** "Explicit signatures, no haunted globals" — path derived from `__file__` is
module-level path construction that depends on the file's physical location. The repository SHA
should come from an environment variable or pipeline config, not subprocess inspection at feature
compute time.

**Severity:** MINOR — this is in the `pipelines/` directory which is slated for deletion (B3).

---

#### m5 — `consolidated_features/inputs.py:47-50` — disk check (`.exists()`) inside what could be a planning-time helper

```python
if not Path(path).exists():
    raise ValueError(f"... feature shard for step {step!r} not found at {path}.")
```

`load_feature_shards` is a runtime loader (called from `run_consolidated_features`), so disk
inspection here is correct and named appropriately (`load_`). However, it is not called from a
DAG-planning context, so this is not a violation. Still worth noting for clarity: if this
function's name were ever changed to `resolve_feature_shard_paths`, the disk check would be
wrong. The current name is fine.

**Severity:** MINOR — name clarity note only, not a violation.

---

#### m6 — `mask_geometry/entrypoint.py` — no module-level docstring orientation

The file has a one-line docstring but lacks the "jobs + why + boundaries" orientation the
philosophy requires for entrypoints:

```python
"""mask_geometry entrypoint — the thin filesystem adapter."""
```

Compare to `legacy_embeddings/entrypoint.py` which names jobs, the batch-write pattern, and the
env boundary upfront. The single line passes for a simple entrypoint, but the absence of the
`inputs + what it does not do` statement makes it harder to audit quickly.

**Severity:** MINOR — style gap.

---

#### m7 — `legacy_embeddings/contract.py:22` — `REQUIRED_COLUMNS: list[str]` local name is generic per-module but public

```python
REQUIRED_COLUMNS: list[str] = ["snip_id", EMBEDDING_MODEL_NAME_COL]
```

`feature_world.md` (Contract Naming Pattern): "Avoid exported generic names like
`REQUIRED_COLUMNS` from product contracts." The public export should be
`LATENT_EMBEDDINGS_REQUIRED_COLUMNS` (or similar) to stay readable when multiple contracts are in
scope. The validator `validate_latent_embeddings` is correctly named.

**Severity:** MINOR — naming convention violation only; no runtime impact.

**Fix:** Rename to `LATENT_EMBEDDINGS_REQUIRED_COLUMNS` and update internal use.

---

## Strengths

**`mask_geometry/` — the exemplar, mostly confirmed.** `contract.py` imports
`SNIP_ID_SPINE_COLUMNS` from the minting site and `validate_snip_grain_identity_columns` — spine
first, then features, no `_flag` columns. `compute.py` is pure (no file IO, no paths). The
entrypoint is thin: load → compute → validate → write. Section banners exist in `compute.py` and
are in flow order.

**`shared/feature_table_utils.py` — correctly earned.** The shared scaffold (`compute_per_snip_mask_features`,
`pixel_size_for_image`, `validate_feature_table`, `SNIP_FEATURE_TABLE_ID_COLUMNS`) is DRY,
imports spine from the minting site, and is reused by `curvature_metrics/` and `pose_kinematics/`
correctly. This is the right shape for shared mechanics.

**`legacy_embeddings/` — well-structured special case.** The 3.9 env boundary is documented
upfront in the entrypoint docstring. Batch-write pattern is explicit (load once, iterate wells).
`model_paths.py` is correctly scope-bounded: path-pure, no orchestration import, `models_root`
passed explicitly, fail-loud with actionable error message. `snip_source.py` is exemplary:
fail-loud at the contract boundary, names the fix in each error, clearly docstrings its moment.

**`consolidated_features/inputs.py` — correct registry usage.** The only place in the zone that
touches orchestration paths — and it does so correctly via `artifact_path()` with
`PATH_MODE_PER_WELL`, with explicit `output_root`, `experiment_id`, and `well_id` parameters.

**`fraction_alive/` (new path) — correct structure, one active blocker.** The product folder has
the right shape: `compute.py` delegates to `_legacy_compute.py` for the pure viability math,
`via_masks.py` uses `build_mask_id` from `shared/identifiers` correctly (no inline id
construction), `contract.py` uses `validate_feature_table` from shared. The blocker is the
deprecated import path; the structure around it is correct.

**`stage_predictions/`, `curvature_metrics/`, `pose_kinematics/` — structurally clean.** All
three follow the recipe: `contract.py` + `compute.py` + `entrypoint.py`, spine imported from
minting site, no `_flag` columns, no inline id splits. The one gap shared by all three is the
missing `.validated` sentinel write (M5).

**Tests correctly in the parallel tree.** All new product tests are under
`tests/data_pipeline/feature_extraction/` with matching sub-package structure. The
`_legacy_drift/` subdirectory is a well-named benchmark group (not a product test), correctly
scoped under the feature_extraction test tree.

---

## Files Reviewed

```
src/data_pipeline/feature_extraction/
  __init__.py  (not read — trivial)
  config.py                              — legacy, delete (B5)
  consolidate_features.py                — legacy, delete (B5, M3)
  mask_geometry_metrics.py               — legacy pure fn, keep only the pure function (B5)
  stage_inference.py                     — legacy, delete batch helpers (B5, M9)
  pose_kinematics_metrics.py             — legacy pure fn, keep only pure helpers (B5)

  mask_geometry/
    contract.py                          — GOOD with one alias cleanup (M1)
    compute.py                           — GOOD with one duplicate helper (M2)
    entrypoint.py                        — GOOD, missing sentinel write (M5)
    __init__.py  (not read)

  consolidated_features/
    contract.py                          — GOOD
    compute.py                           — GOOD
    entrypoint.py                        — GOOD (output_root passed explicitly)
    inputs.py                            — GOOD (registry usage correct)
    __init__.py  (not read)

  fraction_alive/
    contract.py                          — GOOD
    compute.py                           — BLOCKER (deprecated import, B1)
    entrypoint.py                        — GOOD structure, output_root optional (m3)
    _legacy_compute.py                   — has inline id split (M8); keep compute_fraction_alive only
    via_masks.py                         — GOOD (uses shared/identifiers)
    __init__.py  (not read)

  stage_predictions/
    contract.py                          — GOOD
    compute.py                           — GOOD
    entrypoint.py                        — GOOD, missing sentinel write (M5)
    __init__.py  (not read)

  curvature_metrics/
    contract.py                          — GOOD
    compute.py                           — GOOD
    entrypoint.py                        — GOOD, missing sentinel write (M5)
    __init__.py  (not read)

  pose_kinematics/
    contract.py                          — GOOD
    compute.py                           — GOOD
    entrypoint.py                        — GOOD, missing sentinel write (M5)
    __init__.py  (not read)

  legacy_embeddings/
    contract.py                          — GOOD, generic REQUIRED_COLUMNS name (m7)
    entrypoint.py                        — GOOD
    model_paths.py                       — GOOD
    snip_source.py                       — GOOD
    (encode.py, transforms.py, legacy_vae_inference_loader.py, load_model_smoke.py — not read)

  shared/
    feature_table_utils.py               — GOOD

  io/
    loaders.py                           — MAJOR issues (M7, m1)
    writers.py                           — MAJOR issues (M6, m2)

  core/                                  — dead shims/implementations, delete (B5)
    __init__.py, mask_geometry.py, fraction_alive.py, stage_inference.py,
    consolidate_features.py, pose_kinematics.py
    curvature_metrics.py                 — real implementation, move before deleting (B5)
    curvature_skeletonization.py  (not read — referenced by core/curvature_metrics.py)

  entrypoints/                           — dead flat entrypoints, delete (B5)
    compute_mask_geometry.py, compute_fraction_alive.py (broken import, B2),
    consolidate_features.py, compute_stage_predictions.py,
    compute_curvature_metrics.py, compute_pose_kinematics.py
    __init__.py

  pipelines/                             — delete entire directory (B3, B4, M4)
    compute_stage_predictions.py, merge_stage_predictions.py, validate_stage_predictions.py
    __init__.py

src/data_pipeline/features/             — ghost package, delete (M10)
  __init__.py
  legacy_embeddings/ (only __pycache__)
```
