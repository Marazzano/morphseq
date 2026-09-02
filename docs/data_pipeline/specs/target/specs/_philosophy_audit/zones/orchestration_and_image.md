# Philosophy Audit — orchestration + image_materialization + image_building

Auditor: automated review against `pipeline_file_philosophy.md` (2026-06-07).
Files reviewed: 2026-06-24.

---

## Summary

| Severity  | Count |
|-----------|-------|
| BLOCKER   | 2     |
| MAJOR     | 6     |
| MINOR     | 4     |

**Overall verdict:** `pipeline_orchestrator/` and `image_materialization/` together are a strong
implementation of the doctrine — the reference-implementation claim in the doc is largely
justified. Most findings are concentrated in `image_building/`, which is a **confirmed
duplicate-package smell**: it performs the same pixel-writing job as `image_materialization/`
under a different name, with a pre-doctrine interface, stale vocabulary, and no test coverage.
That is Finding #1.

### Duplicate-package verdict

`image_building/` and `image_materialization/` are NOT complementary. `image_building/scope/yx1/`
does YX1 ND2 focus-stacking and pixel writing. `image_materialization/scope/yx1/` does the same.
`image_materialization/` is the doctrine-compliant successor; `image_building/scope/yx1/` is the
pre-doctrine predecessor. The `materialize_well_yx1.py` backend imports two private helpers
(`_determine_bf_channel`, `_get_stack`) from `image_building/scope/yx1/stitched_ff_builder.py`,
making it a partial extraction rather than a clean replacement. The duplicate-package smell is real
and must be resolved by completing the migration.

---

## Findings

### BLOCKER

---

#### B-1: `image_building/` is a pre-doctrine parallel to `image_materialization/` with stale path contracts and no registry coupling — both packages write the same pixel product to different paths

**Files:**
- `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py:139`
- `src/data_pipeline/image_building/scope/keyence/stitched_ff_builder.py:226`

**Snippet (yx1):**
```python
output_dir = output_root / exp_name / "stitched_ff_images"
```
**Snippet (keyence):**
```python
output_dir = output_root / exp_name / "stitched_ff_images"
```

**Violated rules:**
- HARD CONSTRAINT 1: No artifact path as a raw string — `"stitched_ff_images"` is a hardcoded
  path literal in both builders, never resolved through `PIPELINE_STEPS`.
- Image materialization doctrine: scope-specific code must end at canonical acquired image tiles;
  the whole `compile_yx1_data` / `compile_keyence_data` entry-point writes pixel files under a
  layout (`stitched_ff_images/{exp}/{well}/{channel}/`) that is NOT the layout
  `materialized_image_paths.py` defines (`materialized_images/{exp}/{well}/projection/{channel}/`).
  Two packages, two layouts — files produced by one are invisible to the other.
- Pixel path helpers must take `built_image_data_dir` explicitly: both `compile_*` functions take
  `output_root` and construct the path internally via string literal concatenation, not through
  `materialized_image_paths.py`.

**Severity:** BLOCKER

**Fix:** Complete the migration. `image_building/scope/yx1/` and `image_building/scope/keyence/`
are pre-doctrine predecessors. The `_determine_bf_channel` and `_get_stack` helpers from
`image_building/scope/yx1/stitched_ff_builder.py` that `materialize_well_yx1.py` imports should
be promoted into `image_materialization/scope/yx1/` (or a private shared primitives module under
`image_materialization/`) and the `compile_yx1_data` entry-point retired. Track via the
front-half roadmap.

---

#### B-2: `image_building/scope/keyence/stitched_ff_builder.py` imports from `src.build.export_utils` — a root-relative path that bypasses the data_pipeline package boundary

**File:** `src/data_pipeline/image_building/scope/keyence/stitched_ff_builder.py:27`

**Snippet:**
```python
from src.build.export_utils import trim_to_shape
```

**Violated rules:**
- Package boundary: `data_pipeline` packages must not reach outside into `src.build` (a sibling
  package). This is a kingdom-mixing import that makes the Keyence builder non-importable in any
  environment where `src` is not on `sys.path` as a top-level package (e.g. after packaging or in
  test isolation).
- Controlled vocabulary / one concept in one place: `trim_to_shape` duplicates functionality
  already present inside `frame_tiler.py` as `_trim_to_shape` (line 377 of `utils/frame_tiler.py`).
  Two implementations of trim-to-shape in the same logical stage.

**Severity:** BLOCKER

**Fix:** Replace `from src.build.export_utils import trim_to_shape` with the local `_trim_to_shape`
already defined in `utils/frame_tiler.py` (or promote it to a public name and import from there).

---

### MAJOR

---

#### M-1: `image_building/scope/yx1/stitched_ff_builder.py` uses `well_series_mapping: dict[str, int]` (series_number vocabulary) — stale ND2 jargon the doc forbids downstream of the scope boundary

**File:** `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py:115`

**Snippet:**
```python
def compile_yx1_data(
    raw_data_root: Path,
    output_root: Path,
    exp_name: str,
    well_series_mapping: dict[str, int],  # well_name -> series_number (1-based)
```

**Violated rule:** Image materialization doctrine: "NOTHING after the bundle is microscope-aware
(`no YX1/Keyence/ND2/series_number` jargon downstream)." `series_number` is explicitly called out
in the doc as "stale ND2 vocabulary and is not part of that contract."

**Severity:** MAJOR

**Note:** This is inside the legacy `image_building/` package (already a BLOCKER-B-1 candidate for
retirement), but the vocabulary leak is a separate doctrine violation to track.

---

#### M-2: `tasks.py:cmd_validate_snip_inventory` embeds a column manifest and uniqueness check inline — violates the thin-dispatcher rule

**File:** `src/data_pipeline/pipeline_orchestrator/tasks.py:281–294`

**Snippet:**
```python
def cmd_validate_snip_inventory(args: argparse.Namespace) -> None:
    import pandas as pd
    df = pd.read_csv(args.input_csv)
    required = [
        "snip_id", "embryo_id", "physical_embryo_id", ...
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"snip_inventory missing required columns: {missing}")
    if df["snip_id"].duplicated().any():
        raise ValueError("snip_inventory has duplicate snip_id values")
```

**Violated rules:**
- `tasks.py` must be a **thin dispatcher**: parse args and call the stage module; it holds no stage
  logic. This function is the entire validator — no delegate.
- One authoritative validator per contract, owned where the product lives: a `snip_inventory`
  validator belongs in `snip_processing/` (or wherever the contract is defined), not in `tasks.py`.
  There is no matching module-level validator to import from.

**Severity:** MAJOR

**Fix:** Extract the column-check + uniqueness check into a `validate_snip_inventory(df)` function
in the `snip_processing/` (or snip identity contract) package and have
`cmd_validate_snip_inventory` call it. The same pattern used by every other `cmd_validate_*` in
this file.

---

#### M-3: `tasks.py` imports `materialize_stitched_images` at top-level and exposes a `materialize-stitched` verb that leaks pre-doctrine orchestration (stitched_index, scope-aware arguments) into the dispatcher

**File:** `src/data_pipeline/pipeline_orchestrator/tasks.py:20` and `tasks.py:117–133`

**Snippet:**
```python
from data_pipeline.metadata_ingest.stitched_index.materialize_stitched_images import materialize_stitched_images
...
def cmd_materialize_stitched(args: argparse.Namespace) -> None:
    materialize_stitched_images(
        ...
        keyence_projection_method=args.keyence_projection_method,
        keyence_ff_filter_res_um=args.keyence_ff_filter_res_um,
```

**Violated rules:**
- Image materialization doctrine: "The producer is named for the STAGE (`materialize_well`), NOT
  one operation (`stitch_well`)." The verb `materialize-stitched` names the method (stitching),
  and `stitched_index` jargon in the module path leaks the pre-doctrine naming.
- Scope-specific args (`keyence_projection_method`, `keyence_ff_filter_res_um`) surface in the
  shared dispatcher, violating the scope-boundary requirement: scope jargon must not appear in
  shared downstream code.

**Severity:** MAJOR — the command is plumbed and in-use, but it does not conform to the target
doctrine. Flag for replacement when the Keyence route is reconstructed.

---

#### M-4: `materialize_well_yx1.py` imports private helpers (`_determine_bf_channel`, `_get_stack`) from the pre-doctrine `image_building/scope/yx1/` package — cross-boundary dependency on a package that should be retired

**File:** `src/data_pipeline/image_materialization/scope/yx1/materialize_well_yx1.py:37–38`

**Snippet:**
```python
from data_pipeline.image_building.scope.yx1.stitched_ff_builder import (
    _determine_bf_channel,
    _get_stack,
)
```

**Violated rule:** Importing private (`_`-prefixed) symbols from another package couples the
doctrine-conformant `image_materialization/` to the pre-doctrine `image_building/` package. The
imported functions (`_determine_bf_channel`, `_get_stack`) are ND2-specific image-reading
primitives that logically belong to the YX1 materialization backend, not to `image_building/`.
This is the residue of an incomplete migration.

**Severity:** MAJOR

**Fix:** Move `_determine_bf_channel` and `_get_stack` into `image_materialization/scope/yx1/`
(as private helpers of that module or a new `_yx1_nd2_primitives.py` leaf beside it). Then
`image_building/scope/yx1/stitched_ff_builder.py` can import from there (or be retired). The
dependency direction should be `image_materialization/` → shared image primitives, not
`image_materialization/` → `image_building/`.

---

#### M-5: `image_building/yx1/stitched_ff_builder.py` and `image_building/keyence/stitched_ff_builder.py` are bare star-import shims with no docstring explaining the deprecation direction

**Files:**
- `src/data_pipeline/image_building/yx1/stitched_ff_builder.py`
- `src/data_pipeline/image_building/keyence/stitched_ff_builder.py`

**Snippet:**
```python
"""Backward-compatible shim for YX1 stitched FF builder."""
from data_pipeline.image_building.scope.yx1.stitched_ff_builder import *  # noqa: F401,F403
```

**Violated rules:**
- Module docstring must orient (jobs + why + boundaries). "Backward-compatible shim" says what it
  is but not: (a) where callers should import from after the migration, (b) when this shim is
  expected to be deleted, (c) what the migration path is.
- Star-import from a module with `logging.basicConfig(...)` at module level means importing the
  shim triggers root-logger reconfiguration as a side-effect (MAJOR smell even without the
  doctrine).

**Severity:** MAJOR

**Fix:** Add a `# TODO: retire with image_building/scope/yx1/` note and point callers to
`image_materialization/`. Suppress the side-effecting `logging.basicConfig` call in the
`image_building/scope/yx1/` builder (move it behind `if __name__ == "__main__"` or remove it;
library modules must not reconfigure the root logger at import time).

---

#### M-6: `image_building/scope/yx1/stitched_ff_builder.py` calls `logging.basicConfig(...)` at module scope — root-logger reconfiguration as a library import side-effect

**File:** `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py:31–35`

**Snippet:**
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
```

Same pattern exists in `image_building/scope/keyence/stitched_ff_builder.py:29–33`.

**Violated rule:** Explicit signatures, no haunted globals. A library module that reconfigures the
root logger at import time is a module-level mutable-state side effect that surprises every test
and caller. The star-import shims propagate this: importing `image_building.yx1.stitched_ff_builder`
(which is itself imported by `metadata_ingest.stitched_index.materialize_stitched_images`) silently
reconfigures logging for the entire process.

**Severity:** MAJOR

**Fix:** Remove both `logging.basicConfig` calls from the module level. Move them inside
`if __name__ == "__main__"` blocks in the example usage sections, or to the `tasks.py`
`main()` entrypoint where root-logger configuration is appropriate.

---

### MINOR

---

#### m-1: `validated_path()` in `paths.py` emits trailing-suffix form but the live `apply_position_to_well_mapping` step writes a leading-dot form — a known mismatch acknowledged by comment but not yet blocked

**File:** `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:793–800`

**Snippet (from docstring):**
```python
⚠️ Today ``apply_position_to_well_mapping`` writes a LEADING-dot form (``.scope_metadata_mapped.validated``)
while this helper assumes the TARGET trailing form (``scope_metadata_mapped.csv.validated``).
```

**Violated rule:** One concept, built in exactly one place. Two sentinel formats in production is
a diverged contract. The doc says "Sidecars are derived, never first-class" — when two
derivations produce different names, one will be wrong at the consume boundary.

**Severity:** MINOR (acknowledged as open audit item; the comment is appropriate; no code silently
works with the wrong form today — it is a pending on-disk normalization)

**Fix:** Normalize the on-disk sentinel when `apply_position_to_well_mapping` is reconstructed in
Scope 5.

---

#### m-2: `materialize_well_yx1.py` has `candidate: bool = True` as a keyword default — a dangerous default that writes under `candidate/` unless the caller explicitly opts out

**File:** `src/data_pipeline/image_materialization/scope/yx1/materialize_well_yx1.py:118`

**Snippet:**
```python
def materialize_yx1_well(
    ...
    candidate: bool = True,
```

**Violated rule:** Fail loud at contract boundaries — a default that silently opts into the
candidate tree is the opposite of fail-loud. Production callers that forget to pass `candidate=False`
will write files to the wrong subtree with no warning. (Note: `run_materialize_well` in
`run_materialize_well.py` passes `candidate` through correctly from the sequencer's parameter,
which has `candidate: bool = False` as its default — so the sequencer and backend defaults
disagree. A future caller of the backend directly would get `candidate=True` unexpectedly.)

**Severity:** MINOR

**Fix:** Change the backend default to `candidate: bool = False` (matching the sequencer) and add
a note: "Pass `candidate=True` to write under `materialized_images/candidate/` (isolated from
live)." Make the safe production behavior the default, not the other way round.

---

#### m-3: `image_building/scope/yx1/stitched_ff_builder.py:compile_yx1_data` silently swallows per-frame exceptions — hides pipeline failures

**File:** `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py:200–203`

**Snippet:**
```python
except Exception as e:
    log.error("Failed processing well=%s, t=%d: %s", well_name, t, e)
    continue
```

**Violated rule:** "Fail loud at contract boundaries; the message names the fix." A bare
`except Exception: continue` silently produces a partial shard (some frames written, some
skipped) with no mechanism to surface the failure to Snakemake. The rule body will succeed and
mark the shard as complete even though frames are missing.

**Severity:** MINOR (would be MAJOR if this code path were wired into the live pipeline; given
that `compile_yx1_data` is effectively superseded by `materialize_yx1_well`, flag for removal
rather than repair)

---

#### m-4: No tests exist for `image_building/` (no parallel `tests/data_pipeline/image_building/` tree)

**Violated rule:** "Tests for `src/data_pipeline/...` live in the parallel `tests/data_pipeline/...`
tree." The `tests/data_pipeline/image_building/` directory does not exist. The `utils/frame_tiler.py`
module has substantial logic (QC, fallback policy, alignment) with no test coverage. The
`scope/yx1/` and `scope/keyence/` builders are also untested.

**Severity:** MINOR (acceptable for a package marked for retirement, but `frame_tiler.py` is a
live dependency of `metadata_ingest/stitched_index/materialize_stitched_images.py`)

**Fix:** Add at least unit tests for `frame_tiler.stitch_frame_tiles` (the QC path, single-tile
identity case, and concat fallback) under `tests/data_pipeline/image_building/utils/`.

---

## Strengths (reference-impl conformance)

The following is an affirmative statement of where `pipeline_orchestrator/` and
`image_materialization/` genuinely embody the doctrine.

### `orchestration/paths.py` — exemplary on all 14 checklist items

- **HARD CONSTRAINT 1 (no raw paths):** Zero raw path strings. Every artifact path flows through
  `PIPELINE_STEPS` → `artifact_path()`. The `NO-LEAKAGE BOUNDARY` section and the comment block
  above `_resolve_filename` actively enforce this in code.
- **HARD CONSTRAINT 1 (no inline id mint/split):** The `NO-LEAKAGE BOUNDARY` docstring explicitly
  states the rule, and the implementation never constructs an id: `{experiment_id}` and `{well_id}`
  are substituted from caller-supplied values, never built here.
- **Controlled vocabulary:** `PER_WELL_DIRNAME`, `EXPERIMENT`, `PER_WELL_THEN_MERGE`,
  `PATH_MODE_*`, `EXECUTION_*` are all named constants with comments. No magic string is repeated
  in the helpers.
- **`paths.py` is pure:** Zero `.exists()` calls, zero directory listings, zero file reads in the
  entire file. Confirmed clean.
- **Fail loud with fix:** `_normalize_path_mode` names the two legal choices in the error message;
  `_lookup_step` lists known steps; `_resolve_filename` names the missing token and the template.
- **Sidecars derived via helpers:** `validated_path` and `provenance_path` are composites of
  `artifact_path` + suffix. They appear nowhere as raw strings.
- **`fanout` is executable:** `_normalize_path_mode` enforces the fanout→allowed-modes constraint
  at every call, making the registry metadata load-bearing.
- **Section banners in flow order:** Registry → Private lookups → Directory bricks → Public
  composers. The file reads like a pipeline.
- **Module docstring orients before code:** Vocabulary, layout diagram, no-leakage boundary, and
  a forward-declaration status note all appear before line 1 of code.

### `orchestration/well_runner.py` — the run/collect split is correctly implemented

- **`run_well_shard_paths` is pure:** No `.exists()` call, no directory scan. Only
  `validate_well_id` (pure shape check) and `paths.artifact_path` (pure path math).
- **`collect_well_shard_paths` is a disk scan and says so:** Docstring opens with "scans the
  per_well/ dir", the name is `collect_*`, and the behavior matches.
- **No haunted globals:** `output_root` is always a parameter; `PROJECT_ROOT` never appears.
- **Identity via `shared.identifiers`:** `build_well_id`, `validate_well_id` are the only way ids
  are constructed or checked. No `.split("_")` in the module.
- **Fail-loud taxonomy in `collect_well_shard_paths`:** Bad directory name, corrupt shard (sentinel
  without artifact), and mid-flight shard each get distinct handling with clear error messages.
- **Module docstring orientation:** The 2a/2b split is laid out in prose before any code, making
  the `run_*` vs `collect_*` distinction self-teaching.

### `image_materialization/` — microscope boundary correctly implemented

- **`materialized_image_paths.py` is the pixel-path sibling of `paths.py`:** First positional arg
  is `built_image_data_dir`; no `PROJECT_ROOT` fallback; no import of
  `pipeline_orchestrator.orchestration.paths`. Pure path math.
- **`frame_inventory_contract.py` is microscope-agnostic:** Imports only
  `shared/identifiers/constructors` and `validators`. No YX1/Keyence jargon anywhere.
- **`materialization_plan.py` separates request vs resolved vocabulary:** `ImageProductRequest`
  carries `xy_composition='auto'`; `ResolvedImageProduct` never carries `'auto'`. The seam is
  structural, not a comment.
- **`scope/` is the only subfolder:** Scope-specific code forks into `scope/yx1/` and the
  (reserved sketch) Keyence path in the resolver. Everything else is flat.
- **Producer named for the stage:** The entrypoint is `run_materialize_well` / `materialize_well`
  in the registry; `materialize_yx1_well` is the backend executor, never the exposed verb.
- **Consume-boundary contract check:** `materialize_yx1_well` calls
  `validate_yx1_acquisition_inventory(..., check_sources=True)` before reading any ND2 tensor —
  the doc's worked example of "source/disk checks fire at the consume boundary."
- **`scope_resolver_for_materialization_plan.py` explains why in comments:** The "REQUIRED axes vs
  BYPASSABLE axes" distinction is stated and the reasoning ("one tile, so mosaic == identity")
  appears as code + log sentences, not in a policy dict — exactly as the doc prescribes.
- **Parallel test tree exists and is well-populated:** `tests/data_pipeline/image_materialization/`
  covers all six public modules.

### `orchestration/__init__.py` — curated re-export

- All public symbols explicitly listed in `__all__`. The vocabulary exposed is exactly the one the
  rest of the pipeline needs; nothing internal leaks.

---

## Files reviewed

| File | Zone | Lines |
|------|------|-------|
| `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` | orchestration | ~819 |
| `src/data_pipeline/pipeline_orchestrator/orchestration/well_runner.py` | orchestration | ~406 |
| `src/data_pipeline/pipeline_orchestrator/orchestration/__init__.py` | orchestration | ~63 |
| `src/data_pipeline/pipeline_orchestrator/tasks.py` | orchestration | ~1200 |
| `src/data_pipeline/image_materialization/__init__.py` | image_materialization | 1 |
| `src/data_pipeline/image_materialization/frame_inventory_contract.py` | image_materialization | ~332 |
| `src/data_pipeline/image_materialization/materialization_plan.py` | image_materialization | ~231 |
| `src/data_pipeline/image_materialization/materialized_image_paths.py` | image_materialization | ~174 |
| `src/data_pipeline/image_materialization/run_materialize_well.py` | image_materialization | ~130 |
| `src/data_pipeline/image_materialization/select_well_acquisition_rows.py` | image_materialization | ~89 |
| `src/data_pipeline/image_materialization/scope/__init__.py` | image_materialization | 1 |
| `src/data_pipeline/image_materialization/scope/scope_resolver_for_materialization_plan.py` | image_materialization | ~156 |
| `src/data_pipeline/image_materialization/scope/yx1/__init__.py` | image_materialization | 1 |
| `src/data_pipeline/image_materialization/scope/yx1/materialize_well_yx1.py` | image_materialization | ~318 |
| `src/data_pipeline/image_building/__init__.py` | image_building | 1 |
| `src/data_pipeline/image_building/yx1/stitched_ff_builder.py` | image_building | 3 (shim) |
| `src/data_pipeline/image_building/keyence/stitched_ff_builder.py` | image_building | 3 (shim) |
| `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py` | image_building | ~280 |
| `src/data_pipeline/image_building/scope/keyence/stitched_ff_builder.py` | image_building | ~323 |
| `src/data_pipeline/image_building/shared/log_focus.py` | image_building | (shared primitive) |
| `src/data_pipeline/image_building/utils/frame_tiler.py` | image_building | ~420 |
| `tests/data_pipeline/pipeline_orchestrator/test_paths.py` | tests | reviewed (exists) |
| `tests/data_pipeline/pipeline_orchestrator/test_well_runner.py` | tests | reviewed (exists) |
| `tests/data_pipeline/image_materialization/` (6 test files) | tests | reviewed (exists) |
| No `tests/data_pipeline/image_building/` | tests | MISSING |

_Conventions doc:_ `docs/data_pipeline/specs/target/specs/pipeline_file_philosophy.md`
