# Philosophy Audit — quality_control

**Zone:** `src/data_pipeline/quality_control/`
**Auditor:** Claude Sonnet 4.6
**Date:** 2026-06-24
**Rubric:** `pipeline_file_philosophy.md` — two hard constraints + 14-point conformance checklist

---

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 2 |
| MAJOR | 5 |
| MINOR | 4 |

**Verdict:** The **new per-product packages** (`death_detection/`, `surface_area_qc/`, `mask_quality_qc/`, `snip_qc/`) are architecturally strong — they follow the recipe almost perfectly. The BLOCKERs and most MAJORs are concentrated in the **two surviving top-level legacy modules** (`death_detection.py`, `surface_area_outlier_detection.py`) and in a cluster of **orphaned pre-recipe infrastructure** (`io/`, `validators.py`, `config.py`) that was never retired when the per-product packages landed. These need to be cleanly separated: the legacy modules are kept live intentionally for a legacy-drift gate (noted in project memory), but their current state contains broken runtime lookups that will crash if called and no retirement markers. The orphan cluster continues to export validators for products that no longer use them, creating a forked-validator situation.

---

## Findings

### BLOCKERS

---

#### BLOCKER-1 — Legacy `death_detection.py` crashes at runtime on wrong config key lookups

**File:** `src/data_pipeline/quality_control/death_detection.py` lines 172–174, 251, 159/162

```python
persistence_threshold = QC_DEFAULTS['persistence_threshold']   # line 172
min_decline_rate = QC_DEFAULTS['min_decline_rate']             # line 174
dead_lead_time = QC_DEFAULTS['dead_lead_time_hours']           # line 251
```

`QC_DEFAULTS` is the alias for `DEFAULT_QC_CONFIG`, which is a **nested dict** keyed by stage name (`death_detection`, `segmentation_qc`, etc.). The flat keys `'persistence_threshold'`, `'min_decline_rate'`, and `'dead_lead_time_hours'` do **not exist** at the top level. Confirmed via Python:

```
KeyError: 'persistence_threshold'
KeyError: 'min_decline_rate'
KeyError: 'dead_lead_time_hours'
```

Additionally `'min_decline_rate'` does not exist even inside `QC_DEFAULTS['death_detection']` (the correct key is `'decline_rate_threshold'`), and `'dead_lead_time_hours'` does not exist (the correct key is `'lead_time_hr'`). Any call to `detect_persistent_death_inflection` or `compute_dead_flag2_persistence` with `None` defaults will raise `KeyError` at runtime.

**Violated rules:**
- Controlled tokens named as constants, defined once — the config keys are inconsistently named between `config.py` and the caller in `death_detection.py`.
- Fail loud with a message that names the fix — this silently fails with a bare `KeyError`.

**Concrete fix:** Either (a) add a deprecation/retirement header to `death_detection.py` marking it as non-executable legacy-drift reference (since the real logic lives in `death_detection/persistence.py` and `death_detection/compute.py`), or (b) correct all three key names to use the nested lookup `QC_DEFAULTS['death_detection']['persistence_threshold']`, etc., and replace `'min_decline_rate'` → `'decline_rate_threshold'` and `'dead_lead_time_hours'` → `'lead_time_hr'`.

Given the project memory note that legacy modules are kept for a drift gate, option (a) is preferred: add `# LEGACY: not importable from pipeline rules — kept as benchmark for drift comparison only` prominently and guard with `raise RuntimeError(...)` in `main()` if that function would be called in production.

---

#### BLOCKER-2 — Legacy `death_detection.py` uses `from src.data_pipeline...` absolute import (hard failure when packaged)

**File:** `src/data_pipeline/quality_control/death_detection.py` line 33

```python
from src.data_pipeline.quality_control.config import QC_DEFAULTS
```

This is a `src`-prefixed import, which violates the project's import convention (`PYTHONPATH=src:$PYTHONPATH`; all imports are `data_pipeline.*`, never `src.data_pipeline.*`). Confirmed via grep: this is the only `from src.` import in the entire `quality_control/` zone. This import works only if the caller sets `PYTHONPATH` to the repo root rather than to `src/`, making the module unreliable regardless of the KeyError in BLOCKER-1.

**Violated rules:**
- No haunted globals / explicit signatures — using a non-standard path root is a cousin to `PROJECT_ROOT`.
- Controlled vocabulary: the import path convention is `data_pipeline.*` throughout the codebase.

**Concrete fix:** Change to `from data_pipeline.quality_control.config import QC_DEFAULTS` (or add the retirement marker from BLOCKER-1 so the import never executes in production).

---

### MAJOR

---

#### MAJOR-1 — Two top-level legacy modules (`death_detection.py`, `surface_area_outlier_detection.py`) have no retirement/status markers

**Files:**
- `src/data_pipeline/quality_control/death_detection.py` lines 1–26
- `src/data_pipeline/quality_control/surface_area_outlier_detection.py` lines 1–27

Both modules present themselves as **active, authoritative implementations** with standard docstrings describing their algorithm as current ("This module implements...", "Key Features:"). Neither carries any marker that the canonical implementation now lives in `death_detection/` or `surface_area_qc/`. A developer reading either file in isolation has no way to know it is a legacy benchmark, not the production path.

Evidence that `build04_perform_embryo_qc.py` (line 16–17) still imports `compute_dead_flag2_persistence` and `compute_sa_outlier_flag` from these legacy top-level files, meaning they are not purely dead — they are the legacy-drift gate baseline. But that role is not stated in the files.

**Violated rules:**
- Module docstring orients (jobs + why + boundaries) — the docstring actively misleads by presenting as the live implementation.
- One authoritative validator per contract — `surface_area_outlier_detection.py:validate_sa_reference` (lines 167–234) is a second, diverged validator for the same surface-area reference contract, parallel to the authoritative `surface_area_qc/reference_contract.py:validate_surface_area_reference`. Two priests, different scrolls.

**Concrete fix:** Add a prominent `# LEGACY BENCHMARK — kept for drift comparison; not called by pipeline rules` header block to both files. For `surface_area_outlier_detection.py`, mark `validate_sa_reference` as a draft that was superseded by `surface_area_qc/reference_contract.py` and add a delegation comment pointing there.

---

#### MAJOR-2 — `io/paths.py:qc_sentinel_path` duplicates the orchestration `validated_path()` sentinel helper

**File:** `src/data_pipeline/quality_control/io/paths.py` lines 6–7

```python
def qc_sentinel_path(table_path: Path) -> Path:
    return table_path.with_suffix(table_path.suffix + ".validated")
```

`orchestration/paths.py:validated_path()` (line 778 confirmed) already computes `.validated` sentinel paths from the same logic. `io/paths.py` invents a second, package-local sentinel helper that produces identical results. This violates "one concept, built in exactly one place (no drift)" — if the sentinel naming scheme ever changes (e.g., to `".ok"` or `".sentinel"`), it has to be changed in both places.

Additionally, `io/writers.py` (which uses `qc_sentinel_path`) creates the sentinel with `.write_text("ok\n")` directly rather than through the orchestration helper's validated_path resolution, meaning the sentinel is written at a different layer from where it is usually expected.

**Violated rules:**
- One concept, built in exactly one place.
- Sidecars derived via helpers, never hardcoded — this is a parallel derivation of the sidecar, not an import of the canonical helper.

**Concrete fix:** Either (a) delete `io/paths.py` and `io/writers.py` and replace all callers with `from data_pipeline.pipeline_orchestrator.orchestration.paths import validated_path`, or (b) if `io/paths.py` must stay for backward compat, have it import and re-export `validated_path` rather than redefining the suffix logic.

---

#### MAJOR-3 — `validators.py` is a forked second validator for products that now have authoritative contracts

**File:** `src/data_pipeline/quality_control/validators.py` lines 65–138

This module defines validators for seven per-product flag tables:
- `validate_segmentation_qc_flags`
- `validate_viability_qc_flags`
- `validate_death_detection_flags`
- `validate_surface_area_qc_flags`
- `validate_focus_qc_flags`
- `validate_motion_qc_flags`
- `validate_qc_flags`

The per-product packages already have authoritative contracts:
- `death_detection/contract.py` owns `validate_death_detection_qc`
- `surface_area_qc/contract.py` owns `validate_surface_area_qc`
- `mask_quality_qc/contract.py` owns `validate_mask_quality_qc`
- `snip_qc/contract.py` owns `validate_snip_qc`

`validators.py` is now a second validator for these same products, using the **old column names** (e.g. `dead_flag`, `death_inflection_time_int`, `death_predicted_stage_hpf`) that differ from the authoritative contracts' new names (`viability_dead_flag`, `persistence_dead_flag`, `death_event_time_index`, `death_event_stage_hpf`). These two sets of validators are actively describing different schemas, which is the "two priests reading from different scrolls" failure mode.

The **only current caller** of `validators.py` is `io/writers.py`, which itself appears to be an orphaned pre-recipe writer (no callers found in the production path). So `validators.py` is effectively dead code — but it is not marked as such.

**Violated rules:**
- One authoritative validator per contract, owned where the product lives.
- Module docstring orients — no docstring explains what relationship this module has to the per-product contracts.

**Concrete fix:** Deprecate/remove `validators.py` (after confirming `io/writers.py` has no remaining active callers). If needed as a legacy schema record for build04, add a `# LEGACY SCHEMA: build04 column names, not conformant with the new per-product contracts` header.

---

#### MAJOR-4 — `io/loaders.py` contains stub loaders that do no validation (no-op functions)

**File:** `src/data_pipeline/quality_control/io/loaders.py` lines 22–43

```python
def load_segmentation_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)

def load_viability_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)

def load_death_detection_flags(path: Path) -> pd.DataFrame:
    return load_table(path)

def load_surface_area_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)

def load_focus_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)

def load_motion_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)
```

Six named loaders are pure delegation to `load_table(path)` with zero validation or schema enforcement. Each has a function name that promises a validated product but delivers a bare `pd.read_csv`. The contract is hollow. No callers were found for these six functions in the production codebase (grep found zero callers outside `io/__init__.py` itself), confirming they are orphaned.

**Violated rules:**
- Names state position-in-flow / litmus test — these names imply contract-enforced loading but deliver nothing.
- One authoritative validator per contract — these suggest validators should exist here but don't, while real validators exist in the per-product contracts.

**Concrete fix:** Remove `io/loaders.py` (the six stub loaders are unused). If `load_features_table` and `load_qc_table` are still called, retain them in a trimmed file; otherwise remove the file entirely and clean up `io/__init__.py`.

---

#### MAJOR-5 — Four directories exist as pycache-only stubs (`segmentation_qc/`, `auxiliary_mask_qc/`, `core/`, `entrypoints/`)

**Directories:**
- `src/data_pipeline/quality_control/segmentation_qc/` — pycache contains `__init__`, `segmentation_quality_qc`
- `src/data_pipeline/quality_control/auxiliary_mask_qc/` — pycache contains `imaging_quality_qc`
- `src/data_pipeline/quality_control/core/` — pycache contains `_shared`, `consolidate_qc`, `death_detection`, `focus_qc`, `motion_qc`, `segmentation_quality_qc`, `surface_area_outlier_detection`, `viability_qc`
- `src/data_pipeline/quality_control/entrypoints/` — pycache contains seven compute entrypoints and `consolidate_qc`

Each directory has Python source files that have been **deleted** but whose `__pycache__` entries remain, meaning the modules were compiled and used at some point. The directory structure implies Python packages (importable namespaces) but the packages are empty. No active callers were found for any of these directories' modules.

`core/` is the most concerning: it held a parallel compute library (`core/death_detection.py`, `core/surface_area_outlier_detection.py`, `core/segmentation_quality_qc.py`, etc.) that was the pre-recipe computation home. These are now gone, but the directory continues to exist as a ghost, and there is no record of whether they were properly superseded or just deleted.

**Violated rules:**
- One concept, built in exactly one place — ghost directories create the illusion of alternative paths.
- Module docstring orients — there is no docstring because there are no modules, which is its own orientation failure for a reader navigating the tree.

**Concrete fix:** Remove the four empty directories (with their `__pycache__` subdirectories) entirely. If `segmentation_qc` and `auxiliary_mask_qc` are intended future product packages (their names appear in `SNIP_QC_EXCLUSION_REASONS` comments as upcoming), add a placeholder `__init__.py` with a comment `# PLACEHOLDER: not yet implemented` so the directory's intent is self-documenting.

---

### MINOR

---

#### MINOR-1 — `generate_references/build_sa_reference.py` hardcodes absolute production paths

**File:** `src/data_pipeline/quality_control/generate_references/build_sa_reference.py` lines 16–18

```python
BUILD04_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output")
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata")
```

Hard-coded absolute paths to user home directories in a script that lives in the source tree. This is a reference-generation script (not a pipeline rule), so it is a lower severity than a rule entrypoint, but it still violates the principle that paths are never hardcoded as raw strings in code.

**Violated rules:**
- No raw artifact-path string anywhere.

**Concrete fix:** Accept `BUILD04_DIR` and `OUTPUT_DIR` as CLI arguments (argparse), defaulting to `None` with a clear error if not provided. The script already documents its usage in the module docstring — just make the paths parameters.

---

#### MINOR-2 — `io/paths.py:qc_sentinel_path` appends `.validated` via `with_suffix(suffix + ".validated")` rather than using an existing convention

**File:** `src/data_pipeline/quality_control/io/paths.py` lines 6–7

See MAJOR-2 above for the duplication issue. The additional minor concern here is that `.with_suffix(table_path.suffix + ".validated")` on a `Path("foo.csv")` produces `foo.csv.validated`, while a convention using `.with_suffix(".validated")` would produce `foo.validated`. The current approach preserves the original extension in the sentinel name, which is good — but the implementation is fragile if `table_path` has no suffix (edge case: produces `foo.validated` accidentally). Not a crash risk in practice but worth noting.

**Concrete fix:** Resolve by the MAJOR-2 fix (import from orchestration helpers).

---

#### MINOR-3 — `config.py` exports `QC_DEFAULTS` alias alongside `DEFAULT_QC_CONFIG` without deprecating either

**File:** `src/data_pipeline/quality_control/config.py` line 36

```python
QC_DEFAULTS = DEFAULT_QC_CONFIG
```

Two names for the same object with no indication which is canonical. The comment says "backwards-compatible alias for callers already using the old name" — which is appropriate for a transition, but `DEFAULT_QC_CONFIG` (the new name) is also not used by any production code found in this audit (the per-product packages all have their own `DEATH_DETECTION_DEFAULTS`, `SURFACE_AREA_QC_DEFAULTS`, etc.). This file is only used by the legacy modules (via `QC_DEFAULTS`) and nowhere else in the production path.

**Violated rules:**
- One concept, built in exactly one place — dual names for one dict signals unfinished cleanup.

**Concrete fix:** If `config.py` is kept for backward compat with legacy `build04` scripts, add a docstring noting it is legacy infrastructure and that production code uses per-product `*_DEFAULTS` dicts. If legacy build scripts are removed, retire `config.py` entirely.

---

#### MINOR-4 — `snip_qc/entrypoint.py` uses `output_root` (correct) but `death_detection/entrypoint.py` and `surface_area_qc/entrypoint.py` do not

**Files:**
- `src/data_pipeline/quality_control/death_detection/entrypoint.py` lines 28–38
- `src/data_pipeline/quality_control/surface_area_qc/entrypoint.py` lines 21–49
- `src/data_pipeline/quality_control/mask_quality_qc/entrypoint.py` lines 20–41

These three entrypoints accept explicit `output_csv: Path` as their output location parameter, while `snip_qc/entrypoint.py` takes `output_root: Path` and computes the output path via `inputs.py`. Both patterns are explicit (no haunted globals), so this is not a hard constraint violation. However, the inconsistency in interface — some products take a pre-resolved path, others take the root — creates cognitive overhead when wiring Snakemake rules. The `output_root` + `experiment_id` + `well_id` pattern in `snip_qc` is more tightly aligned with the recipe's "paths come from paths.py" principle since the rule can resolve and validate the path via the registry before passing it.

**Violated rules (style-level):**
- Consistent naming and position-in-flow — mixing path-passing conventions across sibling entrypoints.

**Concrete fix:** Not a required fix for current MVP, but note for next entrypoint: prefer `output_root + ids` over pre-resolved `output_csv` where the step is a registry-tracked artifact. This aligns with the `snip_qc` pattern and makes the rule body a template.

---

## Strengths

1. **New per-product packages follow the recipe correctly.** `death_detection/`, `surface_area_qc/`, `mask_quality_qc/`, and `snip_qc/` each have exactly one `config.py` (frozen dataclass + resolver), one `contract.py` (one validator + `check_sources=` mode flag), one `compute.py` (pure logic, no IO), and one `entrypoint.py` (thin filesystem adapter). This is the recipe working as intended.

2. **One validator per contract with lifecycle mode flag.** All four per-product validators correctly implement `check_sources: bool = False` — cheaper at build time, full registry check at consume boundary. The fail-loud pattern is consistent: every missing column produces a message naming the expected columns and the validator scope.

3. **Identity spine imported from the minting site, never re-typed.** All contracts import `SNIP_ID_SPINE_COLUMNS` and `PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` from `data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract`. No QC module re-declares its own version of the spine.

4. **`snip_qc/inputs.py` uses `paths.py` helpers correctly.** The module imports `artifact_path` and `PATH_MODE_PER_WELL` from `data_pipeline.pipeline_orchestrator.orchestration.paths` and resolves every source artifact path through the registry — no raw strings. It also fails loud (with the fix name) if a source artifact is missing, if a flag column is absent, or if snip_id sets diverge.

5. **`tasks.py` verbs are correctly thin.** The seven QC-related `cmd_*` verbs in `tasks.py` each contain a one-line import and a delegate call to the product entrypoint. Zero domain logic lives in `tasks.py` for these products.

6. **Fail-loud messages name the fix.** Error messages throughout the new packages cite the affected `snip_id`(s), expected columns, and what was wrong — meeting the "a validation error is a teaching moment" bar.

7. **Tests exist in the parallel tree.** `tests/data_pipeline/quality_control/` has four product subdirectories with substantive test files for `death_detection`, `mask_quality_qc`, `surface_area_qc`, and `snip_qc`. This matches the required `tests/data_pipeline/...` mirror structure.

8. **`death_detection/` correctly separates two grains.** The two-grain architecture (per-snip `death_detection_qc` + per-animal `death_event`) is cleanly separated in `contract.py`, `grain_reconciliation.py`, and the entrypoint. The per-animal grain explicitly prohibits `embryo_id` to prevent channel over-specification — a nice fail-loud boundary.

9. **`surface_area_qc/reference.py` is a clean packaged-asset loader.** The reference CSV is resolved relative to `__file__` (never via `paths.py` or a caller-supplied raw string), validated on every load, and never read by `compute.py` directly. The pattern is sound.

---

## Files Reviewed

| File | Status |
|---|---|
| `__init__.py` | empty — fine |
| `config.py` | active — minor issues (MINOR-3) |
| `validators.py` | orphaned legacy — MAJOR-3 |
| `death_detection.py` | legacy benchmark — BLOCKER-1, BLOCKER-2, MAJOR-1 |
| `surface_area_outlier_detection.py` | legacy benchmark — MAJOR-1 |
| `death_detection/__init__.py` | clean |
| `death_detection/compute.py` | clean |
| `death_detection/config.py` | clean |
| `death_detection/contract.py` | clean |
| `death_detection/death_event.py` | clean |
| `death_detection/entrypoint.py` | clean (MINOR-4 note) |
| `death_detection/grain_reconciliation.py` | clean |
| `death_detection/persistence.py` | clean |
| `surface_area_qc/__init__.py` | clean |
| `surface_area_qc/compute.py` | clean |
| `surface_area_qc/config.py` | clean |
| `surface_area_qc/contract.py` | clean |
| `surface_area_qc/entrypoint.py` | clean (MINOR-4 note) |
| `surface_area_qc/reference.py` | clean |
| `surface_area_qc/reference_contract.py` | clean |
| `mask_quality_qc/__init__.py` | clean |
| `mask_quality_qc/compute.py` | clean |
| `mask_quality_qc/config.py` | clean |
| `mask_quality_qc/contract.py` | clean |
| `mask_quality_qc/entrypoint.py` | clean (MINOR-4 note) |
| `snip_qc/__init__.py` | clean |
| `snip_qc/build.py` | clean |
| `snip_qc/contract.py` | clean |
| `snip_qc/entrypoint.py` | clean |
| `snip_qc/inputs.py` | clean |
| `io/__init__.py` | orphaned legacy re-export |
| `io/loaders.py` | orphaned legacy stubs — MAJOR-4 |
| `io/paths.py` | duplicate sentinel helper — MAJOR-2, MINOR-2 |
| `io/writers.py` | orphaned legacy writer — MAJOR-2 |
| `reporting/__init__.py` | empty — fine |
| `generate_references/build_sa_reference.py` | one-off script — MINOR-1 |
| `segmentation_qc/` | empty (pycache only) — MAJOR-5 |
| `auxiliary_mask_qc/` | empty (pycache only) — MAJOR-5 |
| `core/` | empty (pycache only) — MAJOR-5 |
| `entrypoints/` | empty (pycache only) — MAJOR-5 |
