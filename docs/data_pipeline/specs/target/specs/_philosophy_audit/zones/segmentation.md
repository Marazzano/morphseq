# Philosophy Audit — segmentation

**Audited against:** `pipeline_file_philosophy.md` (2026-06-07)
**Zone:** `src/data_pipeline/segmentation/` — 46 files, ~6250 lines
**Audit date:** 2026-06-24

---

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 4 |
| MAJOR | 6 |
| MINOR | 5 |

**Verdict:** The `physical_embryo_registry/` package and the new `backends/` packages are strong conformant work. The legacy `grounded_sam2/` subtree is the principal problem zone — it contains the zone's worst violations (inline id construction, stale schema vocabulary, raw path strings, inline id parsing) and is structurally misaligned with the target. Fixes are localized; the registry spine itself is clean.

---

## Findings

### BLOCKER

---

**B-1 — Inline `.split("_")` to extract `well_id` from `snip_id`**
`src/data_pipeline/segmentation/backends/unet_snip/run_unet_snip.py:58`

```python
/ snip_id.split("_")[1]  # well_id = second token of snip_id (e.g. B01 from 20250912_B01_...)
```

**Violated rules:** Hard Constraint 1 ("No id is ever split with an inline `.split('_')`"); Checklist item "No inline id mint/split — all via `shared/identifiers/`".

**Why it's a blocker:** `snip_id`'s grammar is owned by `shared/identifiers`; any change to the `snip_id` token order (e.g. during the `time_index` standardization) silently produces wrong `well_id` values in output paths. The `snip_id` is already available; the shared parser `parse_snip_id` returns the parent `embryo_id`, and `parse_embryo_id` can then yield the `physical_embryo_id` which embeds `well_id` via `parse_physical_embryo_id`. Or pass `well_id` as a separate explicit parameter rather than re-deriving it.

**Fix:** Replace with `well_id` passed explicitly into `_mask_output_path(...)` as a parameter (the `run_unet_for_snip_inventory` loop already has `snip_row["well_id"]`), or use `parse_physical_embryo_id(physical_embryo_id)[0]` which returns the embedded `well_id` from the already-available `physical_embryo_id` argument.

---

**B-2 — Raw stage-name path tokens baked into a path constructor**
`src/data_pipeline/segmentation/backends/unet_snip/run_unet_snip.py:55-58`

```python
output_dir
/ experiment_id
/ "snip_auxiliary_masks"
/ "per_well"
```

**Violated rules:** Hard Constraint 1 ("No artifact path is ever typed as a raw string"); Checklist item "No raw artifact-path string anywhere — all via `paths.py` helpers".

**Why it's a blocker:** `"snip_auxiliary_masks"` and `"per_well"` are stage/fanout vocabulary that belongs in `PIPELINE_STEPS` and `paths.py`. If the registry changes the step key or fanout structure, these strings silently produce the wrong path — creating the exact disagreement the constraint is designed to prevent. The path must come from `paths.py` helpers.

**Fix:** Route the output directory through `per_well_step_dir(output_root, "snip_auxiliary_masks", well_id=...)` or the equivalent `paths.py` helper. The Snakefile rule and the Python entrypoint must read from the same `PIPELINE_STEPS` row.

---

**B-3 — Inline `_derive_experiment_id_from_video` and `_derive_video_id_from_image_stem`: private regex id parsers duplicating `shared/identifiers/`**
`src/data_pipeline/segmentation/video_generation/results_adapter.py:27-54`

```python
def _derive_experiment_id_from_video(video_id: str) -> str:
    m = re.match(r"^(.+)_([A-H][0-9]{2})$", video_id)
    ...
def _derive_video_id_from_image_stem(image_stem: str) -> str | None:
    m = re.search(r"_[ft](\d+)$", image_stem)
    ...
```

**Violated rules:** Hard Constraint 1 (id decomposition must use `shared/identifiers/` parsers); Checklist item "No inline id mint/split".

**Why it's a blocker:** These functions re-implement the `image_id` grammar in isolation. The `shared/identifiers` module already owns `parse_image_id(image_id) → (well_id, channel_id, time_index)`. Separate regex copies will drift from the canonical grammar and produce parsing disagreements. The `_[ft](\d+)` pattern also uses the deprecated `time_int` frame suffix (`_t`) alongside the canonical `_f` suffix — propagating a stale convention.

**Fix:** Delete both private derivation functions; replace calls with `parse_image_id` (and `parse_well_id`/`parse_experiment_id` if those parsers exist) from `shared.identifiers`.

---

**B-4 — `generate_image_id` minting `image_id` outside `shared/identifiers/`**
`src/data_pipeline/segmentation/video_generation/video_generator.py:206-207`

```python
@staticmethod
def generate_image_id(video_id: str, frame_number: int) -> str:
    return f"{video_id}_t{str(frame_number).zfill(4)}"
```

**Violated rules:** Hard Constraint 1 ("No id is ever minted with an inline f-string"); Checklist item "No inline id mint/split — all via `shared/identifiers/`".

**Why it's a blocker:** `shared/identifiers/constructors.py` already exports `build_image_id(well_id, channel_id, time_int)`. This static method encodes the `_t{frame:04d}` suffix inline — using the deprecated `_t` convention instead of the canonical `_f` suffix that `build_image_id` produces. This also uses `video_id` (a grounded_sam2-internal concept) where `well_id` is the pipeline-canonical term.

**Fix:** Delete this method and replace any call sites with `build_image_id(well_id, channel_id, time_index)` imported from `shared.identifiers`.

---

### MAJOR

---

**M-1 — `grounded_sam2/csv_formatter.py` uses `time_int` as a column name, duplicates `extract_time_int` already in `shared/identifiers/`**
`src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:116-140, 255-256`

```python
def extract_time_int(image_id: str) -> int:
    m = re.search(r"_[ft](\d+)$", image_id)
    ...
    "time_int": time_int,
    "time_int": time_int,   # duplicate key in same dict literal
```

**Violated rules:** "Names carry their position in the flow" — `time_int` is the deprecated name; `time_index` is the canonical per-frame axis per the `time_index standardization` doctrine; Checklist "Controlled tokens are named constants, defined once". There is also a stray duplicate dict key `"time_int"` on line 256.

**Fix:** Rename the column to `time_index` throughout `csv_formatter.py` and delete `extract_time_int`; use `parse_image_id` from `shared/identifiers` to derive it. Remove the duplicate dict key.

---

**M-2 — `grounded_sam2/csv_formatter.py` imports from `...schemas.segmentation` (foreign schema dependency outside the target contract pattern)**
`src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:66-69`

```python
from ...schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING
REQUIRED_CSV_COLUMNS = REQUIRED_COLUMNS_SEGMENTATION_TRACKING
```

**Violated rules:** "One concept, built in exactly one place (no drift)"; Checklist "No raw artifact-path string anywhere" — schema constants that govern the output product's column contract should live with the contract (in `frame_masks_contract.py` or a dedicated csv contract), not imported from a separate `schemas/` package that is invisible to consumers of `segmentation/`.

**Fix:** Move the `REQUIRED_COLUMNS_SEGMENTATION_TRACKING` constant into a contract file within `segmentation/` (or align it with `FRAME_MASKS_REQUIRED_COLUMNS`), and have `csv_formatter.py` import from there.

---

**M-3 — `entrypoint.py` uses `.parent.parent` path arithmetic to derive the output root**
`src/data_pipeline/segmentation/backends/unet_snip/entrypoint.py:70`

```python
masks_dir = Path(output_csv).parent.parent  # contracts/ -> per-well step root
```

**Violated rules:** "One concept, built in exactly one place (no drift)" — `.parent.parent` strips levels from a path to recover a root that should have been passed explicitly; Checklist "No raw artifact-path string anywhere". The docstring comment `# contracts/ -> per-well step root` reveals a leaking assumption about the internal directory layout.

**Fix:** Pass `masks_dir` (= the per-well step root) as an explicit parameter to `run_snip_auxiliary_masks`; the caller (Snakemake rule or CLI) already has access to it via `paths.py`.

---

**M-4 — `grounded_sam2/csv_formatter.py`: `extract_well_index` re-implements well-grid arithmetic that belongs in `shared/identifiers/`**
`src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:72-113`

```python
def extract_well_index(well_id: str) -> int:
    row_letter = well_id[0].upper()
    col_str = well_id[1:]
    row = ord(row_letter) - ord('A')
    ...
    return row * 12 + col
```

**Violated rules:** Hard Constraint 1 (id grammar lives in `shared/identifiers/`); "No inline id mint/split". The `well_id` token structure (letter-row + numeric column) is the grammar of `well_id`. Inline arithmetic on the characters is parsing the id without using the shared parser.

**Fix:** Move well-index arithmetic into `shared/identifiers/` (or accept that a `well_index` column is not needed in the target schema if it is a derived convenience). If kept, call a shared helper; never parse `well_id` characters inline.

---

**M-5 — `prompt_seeds.py:58`: inline f-string minting a `seed_id` not governed by `shared/identifiers/`**
`src/data_pipeline/segmentation/prompt_seeds.py:58`

```python
"seed_id": f"{seed_image_id}_seed{seed_idx:04d}",
```

**Violated rules:** Hard Constraint 1 ("No id is ever minted with an inline f-string"); the `seed_id` grammar is embedded here with no constructor or parser in `shared/identifiers/`.

**Context / fix:** `seed_id` may be a prompt-local identifier (not in the canonical spine), but the convention requires even ancillary IDs to be minted by named constructors — both so the grammar lives in one place and so the ID can be parsed back. Add `build_seed_id(image_id, seed_index)` to `shared/identifiers/constructors.py` and call it here. If `seed_id` genuinely does not appear in any cross-module contract, a named function `_build_seed_id` local to `prompt_seeds.py` is the minimum fix.

---

**M-6 — `grounded_sam2/csv_formatter.py` exposes a `validate_csv_schema` that is a thin shadow of a real contract validator**
`src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:377-423`

```python
def validate_csv_schema(df: pd.DataFrame) -> None:
    missing_cols = set(REQUIRED_CSV_COLUMNS) - set(df.columns)
    ...
    if "is_seed_frame" in df.columns:
        if df["is_seed_frame"].dtype != bool:
            df["is_seed_frame"] = df["is_seed_frame"].astype(bool)  # mutates in-place during validation
```

**Violated rules:** "One authoritative validator per contract, owned where the product lives"; Checklist "One authoritative validator per contract... lifecycle differences are a mode flag, not a forked second validator". The real `frame_masks` contract is validated by `validate_frame_masks.py`; this is a second, weaker, diverging validator for what appears to be the same product in a legacy JSON-derived form.

**Fix:** Once `csv_formatter.py` outputs to the `frame_masks` contract (which it should, once M-1 and M-2 are fixed), delete `validate_csv_schema` and call `validate_frame_mask_block` instead.

---

### MINOR

---

**m-1 — `video_generation/models.py:FrameRecord` uses `time_int` field name**
`src/data_pipeline/segmentation/video_generation/models.py:36`

```python
@dataclass
class FrameRecord:
    time_int: int | None = None
```

**Violated rules:** `time_int` is deprecated vocabulary; the canonical per-frame axis is `time_index`. Checklist "Names carry their position in the flow".

**Fix:** Rename to `time_index`.

---

**m-2 — `grounded_sam2/csv_formatter.py`: raw artifact path string for PNG mask filename**
`src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:232`

```python
exported_mask_path = f"{image_id}_masks.png"
```

**Violated rules:** Checklist "No raw artifact-path string anywhere". This inline template bakes a file-naming convention with no helper. The path is not derived from `paths.py` and is not tied to any registered step artifact.

**Fix:** Either remove `exported_mask_path` from the contract if it is not needed by the target pipeline (the `frame_masks` contract does not include it), or derive it from a named path helper.

---

**m-3 — `grounded_sam2/frame_organization_for_sam2.py` uses `/tmp` implicitly via `tempfile.mkdtemp`**
`src/data_pipeline/segmentation/grounded_sam2/frame_organization_for_sam2.py:82`

```python
temp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
```

**Context / violated rules:** This is a runtime temporary directory (legitimate disk use). The issue is convention: on a SLURM cluster with `TMPDIR` set per-job, `tempfile.mkdtemp()` should respect `TMPDIR`. This is minor but can cause cross-node temp leaks. Not a hard-constraint violation.

**Fix:** Use `tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))` or pass an explicit `tmp_root` parameter from the Snakemake rule so temp dir location is externally controlled.

---

**m-4 — `video_generation/results_adapter.py` `_extract_time_int` uses `_[ft]` regex matching both canonical and deprecated frame suffixes**
`src/data_pipeline/segmentation/video_generation/results_adapter.py:57-62`

```python
def _extract_time_int(image_id: str, fallback: int) -> int:
    m = re.search(r"_[ft](\d+)$", image_id)
```

**Violated rules:** Embedding id grammar inline, and mixing the deprecated `_t` suffix with the canonical `_f` suffix. This is a conformance smell even in a legacy adapter.

**Fix:** Delete this function; use `parse_image_id` from `shared/identifiers` to extract `time_index`.

---

**m-5 — No tests for `backends/unet_snip/` in the parallel test tree**

`tests/data_pipeline/segmentation/` exists and has `backends/` and `masks/` sub-trees, but there are no tests for:
- `backends/unet_snip/run_unet_snip.py` (`_mask_output_path`, `run_unet_for_snip_inventory`)
- `backends/unet_snip/snip_auxiliary_masks_contract.py` validators

**Violated rules:** Checklist "Tests for `src/data_pipeline/...` live in the parallel `tests/data_pipeline/...` tree"; "Tests pin contracts, not implementation details". The `snip_auxiliary_masks_contract` has meaningful validators that should be pinned (especially `validate_snip_auxiliary_masks_against_snip_inventory`).

**Fix:** Add `tests/data_pipeline/segmentation/backends/unet_snip/test_snip_auxiliary_masks_contract.py` with contract-pinning tests (required columns, dim mismatch, identity cross-check).

---

## Strengths

1. **`physical_embryo_registry/` is the exemplar of the zone.** `build_physical_embryo_registry.py` mints exactly one row per `(well_id, track_id)` using `build_physical_embryo_id` from `shared/identifiers/`, validates at the boundary, and explicitly documents the mint chain in its module docstring. The contract/validator split is clean.

2. **`physical_embryo_registry/snip_identity_contract.py` implements the `check_sources=` lifecycle mode flag correctly.** One validator, two moments — exactly the pattern the conventions doc prescribes. The docstring explicitly explains the two moments.

3. **`validate_frame_masks.py` follows the contract/validator separation faithfully.** `frame_masks_contract.py` owns the schema; `validate_frame_masks.py` owns the logic. The three-function decomposition (`validate_frame_mask_block`, `validate_frame_masks`, `validate_frame_masks_against_prompt_detections`) cleanly separates validation concerns at different lifecycle moments.

4. **`backends/sam2_video/adapt_sam2_output.py` is clean at the canonical boundary.** It imports only `build_mask_id` and `build_track_id` from `shared/identifiers/` for id minting, does no `.split()` id parsing, uses named constructors throughout, and has a clear module docstring that names the moment and return type.

5. **`masks/mask_resize.py` is a textbook one-concern module.** Single topic, named helpers with a documented law ("no resize inline"), explicit shape validation with loud failure messages naming the fix, correct `(H, W)` vs OpenCV `(W, H)` handling, and the `align_binary_masks` "honest shrink" policy documented.

6. **`backends.py` backend config is well-structured.** `SegmentationBackendsConfig` is a frozen dataclass; `_normalize_choice` validates with a loud message listing allowed values; the backend vocabulary is in named sets (`SUPPORTED_DETECTOR_BACKENDS`). Zero magic strings in dispatch.

7. **Parallel test tree is populated.** `tests/data_pipeline/segmentation/` has `backends/`, `masks/`, `physical_embryo_registry/` sub-directories and meaningful coverage, matching the convention.

---

## Files reviewed

| File | Notes |
|---|---|
| `physical_embryo_registry/build_physical_embryo_registry.py` | Clean |
| `physical_embryo_registry/physical_embryo_registry_contract.py` | Clean |
| `physical_embryo_registry/validate_physical_embryo_registry.py` | Clean |
| `physical_embryo_registry/snip_identity_contract.py` | Clean — model for check_sources= pattern |
| `frame_masks_contract.py` | Clean |
| `validate_frame_masks.py` | Clean |
| `valid_frame_masks.py` | Clean (thin boolean wrapper) |
| `backends.py` | Clean |
| `masks/__init__.py` | Clean |
| `masks/mask_resize.py` | Clean |
| `masks/mask_rle.py` | (Not read in detail; no violations surfaced in grep) |
| `masks/mask_geometry.py` | (Not read in detail; no violations surfaced in grep) |
| `prompt_seeds.py` | MAJOR M-5 (inline seed_id mint) |
| `backends/unet_snip/entrypoint.py` | MAJOR M-3 (.parent.parent) |
| `backends/unet_snip/run_unet_snip.py` | BLOCKER B-1, B-2 |
| `backends/unet_snip/snip_auxiliary_masks_contract.py` | Clean |
| `backends/unet_snip/model_loader.py` | (Not read in detail; no violations surfaced) |
| `backends/sam2_video/adapt_sam2_output.py` | Clean |
| `backends/sam2_video/prompt_detections.py` | (Not read in detail) |
| `backends/sam2_video/fake_predictor.py` | (Not read in detail) |
| `grounded_sam2/csv_formatter.py` | BLOCKER B-3 (partial), MAJOR M-1, M-2, M-4, M-6; MINOR m-2 |
| `grounded_sam2/frame_organization_for_sam2.py` | MINOR m-3 |
| `grounded_sam2/gdino_detection.py` | Clean for its scope (backend-local, legitimate disk checks) |
| `grounded_sam2/mask_export.py` | Clean for its scope (backend-local PNG export) |
| `grounded_sam2/propagation.py` | Clean for its scope (backend-local SAM2 runner) |
| `video_generation/models.py` | MINOR m-1 (time_int field) |
| `video_generation/results_adapter.py` | BLOCKER B-3 (_derive_* functions), MINOR m-4 |
| `video_generation/video_generator.py` | BLOCKER B-4 (generate_image_id) |
| `video_generation/render_eval_video.py` | (Not read in detail) |
| `video_generation/service.py` | (Not read in detail) |
| `video_generation/video_config.py` | (Not read in detail) |
| `video_generation/overlay_manager.py` | (Not read in detail) |
| `video_generation/mask_decoding.py` | (Not read in detail) |
| `sam2_video/sam2_frame_view.py` | Clean (disk check in context manager is legitimate runtime use) |
| `sam2_video/run_sam2_video.py` | (Not read in detail) |
| `sam2_video/model_loader.py` | (Not read in detail) |
| `unet/inference.py` | Stub — no violations; note `load_unet_models` arg names use raw checkpoint paths (ok for stub) |
