# Philosophy Audit — `metadata_ingest`

**Audited against:** `docs/data_pipeline/specs/target/specs/pipeline_file_philosophy.md`
**Date:** 2026-06-24
**Zone:** `src/data_pipeline/metadata_ingest/` (~49 files)

---

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 3 |
| MAJOR | 8 |
| MINOR | 7 |

The zone is structurally split well — separate `scope/yx1/`, `scope/keyence/`, and `scope/shared/` mirrors the stitch-split decision, and the acquisition inventory modules (both scopes) are exemplary: rich docstrings, section banners, clean import direction, `check_sources`-style validator. The blocking issues are concentrated in two places: (1) `stitched_index/materialize_stitched_images.py` — a pre-split fat materializer that has not yet been broken per-scope and constructs paths inline; (2) `scope/yx1/generate_xy_reference.py` — an exploration script with hardcoded absolute paths and module-level globals that has leaked into the zone; (3) two validator functions (`validate_stitched_image_index`, `validate_frame_contract`) that always do live disk I/O with no `check_sources` mode flag, making them unsafe at Snakemake planning time.

---

## Findings

### BLOCKER

---

**B1 — Fat shared materializer: microscope branching past the canonical-tile boundary**
`src/data_pipeline/metadata_ingest/stitched_index/materialize_stitched_images.py` lines 347–436

```python
# line 350
yx1_nd2_files = sorted(raw_images_dir.glob("*.nd2")) if microscope == "YX1" else []
keyence_lookup = _infer_keyence_stack_lookup(raw_images_dir) if microscope == "Keyence" else {}
keyence_orientation = _keyence_orientation(experiment)
keyence_master_params = _keyence_master_params_path(raw_images_dir, experiment) if microscope == "Keyence" else None
...
# line 412
if microscope == "YX1" and nd is not None and dask_arr is not None:
    ...
elif microscope == "Keyence":
    tile_stacks = keyence_lookup.get((well_index, time_int), {})
```

**Violated rules:** Microscope leakage (smell 7) — `if YX1 / elif Keyence` branching in shared materialization code; also Hard Constraint 1 (path string below). The doc states: "Once the acquired tiles are canonical, no code is microscope-aware." The LOCKED 2026-06-16 stitch-split decision says to have per-scope materializers, not a shared branching one.

**Fix:** Split into `scope/yx1/materialize_well.py` and `scope/keyence/materialize_well.py`. The shared post-composition logic (row construction, CSV emit) can live in a shared helper those two call. The shared entry point should disappear.

---

**B2 — Hardcoded absolute paths and haunted globals in a live module**
`src/data_pipeline/metadata_ingest/scope/yx1/generate_xy_reference.py` lines 24–31

```python
REFERENCE_EXP_ID = "20251112"
BASE_PATH = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
PLAYGROUND_PATH = BASE_PATH / "morphseq_playground"
RAW_DATA_PATH = PLAYGROUND_PATH / "raw_image_data" / "YX1"
METADATA_PATH = BASE_PATH / "metadata" / "plate_metadata"
OUTPUT_PATH = PLAYGROUND_PATH / "metadata" / "YX1_nd2_ref_plate_xy_coordinates.csv"
```

**Violated rules:** Hard Constraint 1 (raw path strings) and "Explicit signatures, no haunted globals." Every function in this file reads `METADATA_PATH`, `RAW_DATA_PATH`, or `OUTPUT_PATH` from module-level globals instead of accepting `output_root`/`deps` as parameters. Functions `load_series_number_map`, `find_nd2_file`, and `generate_reference_coordinates` all silently depend on the caller being on the correct machine.

Additionally, `load_series_number_map` (line 97) reads the now-retired `series_number_map` sheet by name — the sheet the `position_well_mapping_contract` explicitly retires as stale ND2 vocabulary.

**Fix:** If this is a one-off setup script, move it to `tools/` or `scripts/` with a prominent header comment; do not leave it in the importable `src/` tree. If it must remain, all paths must be taken as explicit CLI arguments / function parameters. The `series_number_map` sheet reference should be retired.

---

**B3 — Validators with unconditional disk I/O — unsafe at planning time**
`src/data_pipeline/metadata_ingest/stitched_index/validate_stitched_image_index.py` lines 33–47
`src/data_pipeline/metadata_ingest/frame_contract/validate_frame_contract.py` lines 37–51

```python
# both files, same pattern
for rel_path in df["stitched_image_path"].astype(str):
    path = Path(rel_path)
    if not path.is_absolute():
        path = input_csv.parent.parent.parent / path  # implicit 3-level path arithmetic
    if not path.exists():
        missing_files.append(str(path))
```

**Violated rules:** "One authoritative validator per contract … lifecycle differences are a mode flag (`check_sources=`), not a forked second validator." And "Path helpers construct paths; they do not inspect disk." The `path.exists()` call always fires — there is no `check_sources=False` mode. The path reconstruction (`input_csv.parent.parent.parent`) encodes implicit knowledge of the three-level directory structure as a magic `.parent` chain, which is a raw-path smell.

**Fix:** Add a `check_sources: bool = False` parameter (default `False` for planning-safe invocation, `True` at execute time). The path reconstruction must also use the orchestration `paths.py` helpers rather than walking `.parent`.

---

### MAJOR

---

**M1 — Inline well_index f-string duplicated four times across Keyence modules**

```python
# extract_scope_metadata.py line 234-235
row = (xy_idx - 1) // 12
col = (xy_idx - 1) % 12 + 1
return f"{chr(65 + row)}{col:02d}"

# map_keyence_positions_to_wells.py line 54-56 (identical)
# map_keyence_positions_to_wells.py line 73-76 (W0 variant)
# generate_xy_reference.py line 131 (load_series_number_map)
```

**Violated rules:** Hard Constraint 1 — "No id is ever minted or split with an inline f-string." `well_index` is a canonical identity token; its construction formula must live once in `shared/identifiers/`. The identical arithmetic appears at minimum four times across three files, guaranteeing drift (and already has: the XY01a legacy-suffix variant in `extract_scope_metadata.py` line 241 produces `f"{well_letter}{well_num}"` with a different zero-padding assumption than the W0 path).

Note: `raw_plane_parsing.py` already centralizes the W-index formula as `_well_from_w_index` and is imported by the acquisition inventory — that module is the right direction. The fix is to promote that function into `shared/identifiers/` and delete the duplicates.

**Fix:** Add `well_index_from_linear_position(n: int, n_cols: int = 12) -> str` to `shared/identifiers/constructors.py`. Replace all four inline occurrences.

---

**M2 — Raw artifact path assembled inline in materializer**
`src/data_pipeline/metadata_ingest/stitched_index/materialize_stitched_images.py` lines 347, 393

```python
stitched_root = output_root / experiment / "stitched_ff_images"   # line 347
...
output_path = stitched_root / well_index / channel_id / f"{image_id}.{image_extension}"  # line 393
```

**Violated rules:** Hard Constraint 1 — raw path string `"stitched_ff_images"` typed inline, not resolved through `PIPELINE_STEPS` + `step_dir`/`artifact_path`. The per-frame path is also assembled with an f-string; the image extension is embedded inline.

**Fix:** After the per-scope split (B1), register `stitched_ff_images` in `PIPELINE_STEPS` and resolve all paths through `step_dir`/`artifact_path`. The per-frame path should be a pixel-path helper that takes `built_image_data_dir` explicitly.

---

**M3 — `data_root = output_root.parent` — implicit tree-level assumption**
`src/data_pipeline/metadata_ingest/stitched_index/materialize_stitched_images.py` line 348

```python
data_root = output_root.parent
```

**Violated rules:** "One concept, built in exactly one place (no drift)" — stripping `.parent` is implicit knowledge about where `output_root` sits in the directory tree. If the tree ever gains or loses a level, this silently produces the wrong root. The convention doc explicitly warns: "no caller ever strips a level with `.parent`."

**Fix:** Pass `data_root` explicitly as a parameter, or derive it from the `PIPELINE_STEPS` registry.

---

**M4 — `validate_frame_contract` and `validate_stitched_image_index` are structurally forked**

The two functions in B3 above are not just missing a `check_sources` flag — they are near-identical 50-line functions validating the same kind of artifact (an image-path CSV). The conventions doc says: "One contract → one authoritative validator." These two should be unified behind a shared base or merged into one parameterized function.

**Violated rules:** "One authoritative validator per contract … lifecycle differences are a mode flag, not a forked second validator."

**Fix:** Extract the shared schema-check + uniqueness-check + optional-existence-check logic into one base validator in `data_pipeline/io/validators.py` or the respective contract module, parameterized by schema constants and `check_sources=`.

---

**M5 — Module-level `logging.basicConfig` side-effects at import time**

```python
# extract_yx1_scope_metadata.py line 23
# scope/yx1/map_yx1_positions_to_wells.py line 23
logging.basicConfig(level=logging.INFO)
```

**Violated rules:** "No module-level mutable state a function secretly reads" (generalized to: no side-effects at import time). `logging.basicConfig` reconfigures the root logger for every process that imports these modules. In a Snakemake context where multiple workers share a process, this silently overrides any logging configuration the caller set.

**Fix:** Remove `logging.basicConfig` calls from module level. Let the application entrypoint (or a test fixture) configure the root logger.

---

**M6 — `REQUIRED_COLUMNS_POSITION_MAPPING` duplicated in Keyence mapper**
`src/data_pipeline/metadata_ingest/scope/keyence/map_keyence_positions_to_wells.py` lines 19–25

```python
REQUIRED_COLUMNS_POSITION_MAPPING = [
    'experiment_id',
    'position_index',
    'well_index',
    'well_id',
    'mapping_method',
]
```

**Violated rules:** "One concept, built in exactly one place." The canonical column list already lives in `position_well_mapping_contract.py`. This local constant is unused dead weight that diverges silently.

**Fix:** Delete the local constant; import `REQUIRED_POSITION_WELL_MAPPING_COLUMNS` from `position_well_mapping_contract.py` if a local reference is needed.

---

**M7 — Tests under `src/` instead of the parallel `tests/` tree**
`src/data_pipeline/metadata_ingest/scope/tests/test_acquisition_inventory.py` line 3
`src/data_pipeline/metadata_ingest/scope/tests/test_canonical_mapping.py` line 1

```python
# Run with: PYTHONPATH=src pytest src/data_pipeline/metadata_ingest/scope/tests/...
```

**Violated rules:** "Tests for `src/data_pipeline/...` live in the parallel `tests/data_pipeline/...` tree." The project memory explicitly flags `--import-mode=importlib` as required precisely because `tests/` shadows `src/`. Keeping tests inside `src/` works around this but forfeits the import-mode protection and breaks the reviewers' expectation of where to look.

**Fix:** Move both test files to `tests/data_pipeline/metadata_ingest/scope/`. Update the run instruction in the docstrings.

---

**M8 — `process_plate_layout` — vague `process_*` name**
`src/data_pipeline/metadata_ingest/plate/plate_processing.py` line 28

```python
def process_plate_layout(
    input_file: Path,
    experiment_id: str,
    output_csv: Path,
) -> pd.DataFrame:
```

**Violated rules:** Naming convention — "paste the name into a chat" litmus test. `process_plate_layout` does not signal its position in the flow (is it loading? validating? minting identity?). The module docstring explains it does all three, which is actually the right function design — the name just doesn't say so.

**Fix:** Rename to `ingest_plate_metadata` (mirrors `ingest_*` family already used for front-end steps) or `run_plate_metadata_ingest`.

---

### MINOR

---

**m1 — `time_int` key appears twice in the dict literal — silent dead key**
`src/data_pipeline/metadata_ingest/frame_contract/build_frame_contract.py` lines 104 and 107

```python
contract = pd.DataFrame({
    ...
    "time_int": merged["time_int"],    # line 104
    "channel_id": merged["channel_id"],
    "image_id": merged["image_id"],
    "time_int": merged["time_int"],    # line 107  ← duplicate
    "microscope_id": merged["microscope_id"],
    ...
})
```

Python silently takes the last definition; the first is dead code. One of the two was almost certainly intended to be `"time_index"` to match the `time_index` standardization convention. Same pattern occurs in `materialize_stitched_images.py` lines 512 and 516.

**Fix:** Audit both dict literals and align the column names with the `time_index` standardization (see project memory: "per-frame axis is time_index (T dimension)").

---

**m2 — `_find_nd2_file` duplicates live disk scan already done in `build_yx1_acquisition_inventory`**
`src/data_pipeline/metadata_ingest/scope/yx1/extract_yx1_scope_metadata.py` lines 27–34

```python
def _find_nd2_file(raw_data_dir: Path) -> Path:
    nd2_files = list(raw_data_dir.glob("*.nd2"))
```

A `raw_data_dir.glob("*.nd2")` pattern also appears inside `build_yx1_acquisition_inventory`. Two separate discovery scans for the same file. Minor because each is in a different module with a clear purpose, but the duplication means two different error messages for the same failure condition.

**Fix:** Deduplicate into one helper owned by the YX1 acquisition inventory module.

---

**m3 — `extract_keyence_scope_metadata` is 167-line monolith fusing five concerns**
`src/data_pipeline/metadata_ingest/scope/keyence/extract_scope_metadata.py` lines 272–438

The function body sequentially does: (1) file discovery via `_discover_keyence_files`, (2) per-file TIFF metadata loop, (3) DataFrame sort/groupby/compute_intervals algebra, (4) schema validation, (5) CSV emit as a side-effect, (6) optional acquisition inventory emit as a second side-effect. No section banners.

**Violated rules:** "Section banners mark each concern" and the implicit "banners organize co-living concerns but do not excuse mixing." The five concerns here do not all legitimately co-live: the disk-write side-effect and the acquisition inventory side-effect could be split to separate entrypoints.

**Fix:** Add section banners at minimum. Separate the CSV-emit side-effect and acquisition-inventory emit into the calling entrypoint; the core function should return a DataFrame.

---

**m4 — `_print_page_summary` has an untyped parameter**
`src/data_pipeline/metadata_ingest/plate/plate_processing.py` line 82

```python
def _print_page_summary(pages) -> None:
```

Minor type annotation gap — `pages` should be typed as the return type of `load_plate_metadata_pages` (a named tuple / dataclass).

---

**m5 — Module docstring missing from package `__init__.py`**
`src/data_pipeline/metadata_ingest/__init__.py` (empty)

The package root has no docstring. A reader navigating to `metadata_ingest/` has no orientation before opening submodule files.

**Fix:** Add a 3–5 sentence docstring naming the sub-packages and their responsibilities.

---

**m6 — `scope/keyence/extract_scope_metadata.py` has a module-level `logging.basicConfig` (same as M5)**
`src/data_pipeline/metadata_ingest/scope/keyence/extract_scope_metadata.py` line 23

```python
logging.basicConfig(level=logging.INFO)
```

Third occurrence of the import-time side-effect (Keyence extractor + both YX1 files). Grouped with M5 but noted separately for completeness.

---

**m7 — `generate_xy_reference.py` references retired `series_number_map` sheet**
`src/data_pipeline/metadata_ingest/scope/yx1/generate_xy_reference.py` line 105

```python
sm_raw = pd.read_excel(metadata_path, sheet_name='series_number_map', header=None)
```

`series_number_map` is the stale ND2 vocabulary the conventions doc explicitly says must not appear past the canonical-tile boundary. Even if this script is moved to `tools/` (see B2 fix), the sheet reference should be noted as retired.

---

## Strengths

**`scope/yx1/acquisition_inventory.py` and `scope/keyence/acquisition_inventory.py` — exemplary structure.** Both files open with multi-paragraph module docstrings that name the model, cite the design decisions, state the import direction constraint, and orient the reader before line 1 of logic. Both use section banners in dependency/flow order (Contract → Builders → Validators). Both correctly use `check_sources=` in `validate_yx1_acquisition_inventory` (the `assert_acquisition_sources_readable` call), which is the exact pattern the conventions doc holds up as the reference.

**`scope/shared/` correctly centralizes cross-scope logic.** `acquisition_checks.py`, `canonical_mapper.py`, and `validate_physical_well_mapping.py` contain no microscope branching. The import direction is clean: `shared/` imports nothing from `yx1/` or `keyence/`.

**`scope/keyence/raw_plane_parsing.py` — correct kernel isolation.** The module docstring states its import constraint explicitly ("MUST NOT import stages, Snakemake/tasks, stitch, or scope-inventory logic"). The `_well_from_w_index` helper is a first step toward the deduplication M1 calls for.

**`well_discovery/discover_wells_from_scope_metadata.py` — scope-blind at exactly the right moment.** The function reads `scope_metadata_mapped.csv` (the post-join canonical artifact) and never needs to know which microscope produced it. The docstring explicitly calls this out ("never needs to know which microscope produced it"). This is the microscope-boundary doctrine in action.

**`position_well_mapping/position_well_mapping_contract.py` — correct single source of truth.** The contract constants live here once; callers import them. The module is a genuine single-concern leaf (no banners needed, none added).

**`plate/plate_metadata_contract.py`, `plate/plate_metadata_loader.py`, `plate/plate_processing.py` — clean three-layer split.** L1 ingest, identity mint, and L2 validation are in separate files with clear names. `plate_processing.py`'s module docstring names what it calls and what it deliberately does not do.

---

## Files reviewed

| File | Notes |
|---|---|
| `__init__.py` | empty |
| `experiment_identity.py` | clean |
| `time_helpers.py` | clean |
| `frame_contract/build_frame_contract.py` | duplicate dict key (m1) |
| `frame_contract/validate_frame_contract.py` | BLOCKER B3 |
| `frame_inventory/frame_inventory.py` | reviewed |
| `frame_inventory/frame_inventory_validation.py` | reviewed |
| `frame_inventory/frame_inventory_validation_rules.py` | reviewed |
| `frame_inventory/scaffold_dropin_inventory.py` | reviewed |
| `plate/plate_metadata_contract.py` | STRENGTH |
| `plate/plate_metadata_loader.py` | reviewed |
| `plate/plate_processing.py` | M8 (name) |
| `plate/validate_plate_metadata.py` | reviewed |
| `position_well_mapping/position_well_mapping_contract.py` | STRENGTH |
| `scope/acquisition_inventory_contract.py` | reviewed |
| `scope/keyence/acquisition_inventory.py` | STRENGTH |
| `scope/keyence/extract_scope_metadata.py` | M1, m3, m6 |
| `scope/keyence/map_keyence_positions_to_wells.py` | M1, M6 |
| `scope/keyence/mappings.py` | reviewed |
| `scope/keyence/raw_plane_parsing.py` | STRENGTH (partial); note M1 centralization opportunity |
| `scope/shared/acquisition_checks.py` | STRENGTH |
| `scope/shared/apply_position_to_well_mapping.py` | reviewed |
| `scope/shared/canonical_mapper.py` | STRENGTH |
| `scope/shared/validate_physical_well_mapping.py` | reviewed |
| `scope/yx1/acquisition_inventory.py` | STRENGTH |
| `scope/yx1/extract_yx1_scope_metadata.py` | M5 (logging.basicConfig) |
| `scope/yx1/generate_xy_reference.py` | BLOCKER B2 |
| `scope/yx1/map_yx1_positions_to_wells.py` | M5 (logging.basicConfig) |
| `scope/yx1/mappings.py` | reviewed |
| `scope/yx1/validate_xy_reference_grid.py` | reviewed |
| `scope/tests/test_acquisition_inventory.py` | M7 (wrong tree) |
| `scope/tests/test_canonical_mapping.py` | M7 (wrong tree) |
| `stitched_index/materialize_stitched_images.py` | BLOCKERS B1, B2 partial; M2, M3; m1 |
| `stitched_index/validate_stitched_image_index.py` | BLOCKER B3 |
| `well_discovery/discover_wells_from_handoff.py` | reviewed |
| `well_discovery/discover_wells_from_scope_metadata.py` | STRENGTH |
| `well_discovery/discovered_wells_contract.py` | reviewed |
| `well_discovery/split_dropin_inventory.py` | reviewed |
