# WIP — External Dataset Handoff: decisions log (NOT the spec yet)

Scratch doc capturing locked decisions as mdcolon + assistant talk through the external-data
on-ramp. The real target spec gets written ONLY after all threads close. Date: 2026-06-23.

---

## Thread 1 — Scope of the promise — ✅ DECIDED

**Who is "someone else with their own data"?** Another **researcher** — a scientist with **little to
mild comp-bio expertise**, likely **using AI tools** (Claude etc.) to help them drive it.

**Implications (these shape every later thread):**
- **Self-documenting code + strong contracts are the deliverable**, not hand-holding prose. The
  validators must fail loud with the fix named in the message (the user's AI assistant reads the
  error and fixes the input). This is already the house style — lean into it.
- Errors are the UX. "Missing column X — add it as Y" beats a separate tutorial.
- NOT targeting "anyone on earth / zero-setup." Assume they can run the pipeline (envs, cluster-ish).
  The bridge is data-shape, not infrastructure.

---

## Thread 2 — Time / age requirement — ✅ DECIDED (core idea), one semantic nit open

**The conflict:** `predicted_stage_hpf = start_age_hpf + (elapsed_time_s/3600)*rate(temperature)`
splits across both roots; a snapshot user may have no timing.

**The resolution (mdcolon):** the check is **grain-aware and coherent for BOTH our pipeline and an
external user**, and the invariant is **temporal**, so:

> **If a well has more than one distinct `time_index` / timepoint, `elapsed_time_s` is REQUIRED.
> A single-timepoint well does not require it.**

This is honest: a multi-timepoint well without timing = a real error (you can't order/Δ them); a
single-timepoint well = elapsed is trivially zero/irrelevant. **The rule is temporal, NOT an image_id
count** — a single timepoint with multiple channels (BF + fluorescence) has multiple image_ids but is
still single-timepoint. The check lives where it can see the grain (per-well), works identically for
native and drop-in.

**Age naming (mdcolon):** canonical output column is **`start_age_hpf`**; accept `age_hpf` and
`start_age_hpf` as INPUT aliases, normalize to `start_age_hpf`.

**Single-timepoint elapsed (D3a):** `0.0` for a true single-timepoint snapshot (it IS zero elapsed),
NA for multi-timepoint-without-timing (forces a loud failure rather than a silent wrong Δ).

---

## Thread 3 — Biology contract (long OR plate.xlsx) — ✅ DECIDED

**The long-table ingester IS the linchpin** (it's the no-plate path; the L2 contract already
validates its output). Finishing it is the cheapest, highest-value build item.

**Well key — accept + normalize.** Accept any of `{well_index, well, well_name}` (matches the
loader's existing `_could_be_long` sniff) and normalize to canonical `well_index` via
`normalize_well_index` (`A1`→`A01`). Most forgiving for a non-expert; fail loud only if NONE present.
*(Did NOT add `well_id` global as an accepted key — keep it to the local well label; promotion to
`well_id` happens via `build_well_id(experiment_id, well_index)` exactly as native does.)*

**Required vs optional fields — only the well key is mandatory at ingest.** Each biology field
(`genotype`, `start_age_hpf`/`age_hpf`, `temperature`, `medium`) is required **only when a downstream
stage that consumes it runs**, and is enforced **at that consumer**, fail-loud, naming the field +
fix. This matches the existing three-layer doctrine in `plate_metadata_ingest_and_entity_qc.md`
(L2 allows nulls; L3/consumer enforces completeness conditional on a registered entity).

> **LOCKED (option a):** long-ingest **emits the canonical biology columns even when absent**
> (`genotype`, `start_age_hpf`, `temperature`, `medium` → all-NA if the user omitted them). L2
> **requires the columns to exist but allows NA values**; consumer gates enforce completeness. L2
> column-shape is stable regardless of which fields the user supplied. *(Superseded the earlier
> "decide at build time" lean — see correction #3.)*

---

## Thread 4 — frame_inventory link + layout helper — ✅ DECIDED

**`materialized_image_paths.py` = recommended target layout, NOT a gate.** Confirmed. Manifest is
truth (`source_image_path` resolves anywhere); the canonical tree is the friendly self-describing
option, never required.

**Helper: phased — scaffold-script NOW, reorganize-images LATER (mdcolon: "1 then 3").**
- **Phase 1 (build with the spec):** document the manifest format precisely (so a researcher's AI
  assistant can author it) **+ ship a small, well-documented scaffold helper** that reads an image
  dir + reads each image header for real dims and **emits a starter `dropin_frame_inventory.csv`** the
  user edits. Best fit for "researcher + AI tools": a runnable starting point, not a blank CSV.
- **Phase 3 (later):** extend the helper to also copy/symlink images into the canonical
  `materialized_image_paths` tree → a fully self-describing dataset. More I/O surface; deferred.

---

## Thread 5 — completeness gate (SAM2 all-frames-present) — ✅ DECIDED

**The rule = BF contiguous `0..N-1` + all present channels rectangular (same `time_index` set).**
Confirmed as the formalization of "a well must have all its images."

**Location: promote the existing `validate_frame_inventory` to the STRICT per-well file-level gate
at the seam** — schema (kept) + paths exist + images open + real dims == declared + µm/px>0 +
BF-contiguity + rectangularity. ONE gate, native + drop-in identical, fires before segmentation.
Matches the handoff contract's design. (Additive on top of today's weak check; resolves its
`TODO(Scope 2)`.)

> Ties to Thread 2: the **multi-frame ⇒ elapsed_time_s required** check also lives here (it's a
> per-well grain check, same node) — natural home alongside contiguity.

---

## Thread 6 — Validator wiring + path policy + layout shape — ✅ DECIDED

### The two existing layers (don't conflate)
- `io/validators.py::validate_dataframe_schema` — generic layer-0 primitive (columns + nulls).
  Stage-agnostic. **Stays as-is.**
- `frame_inventory.py::validate_frame_inventory` — the frame-inventory-specific gate. **THIS is what
  we grow** (resolves its `TODO(Scope 2)`). Wired by the `validate-frame-inventory` task verb at TWO
  `.smk` nodes: per-well (`validate_frame_inventory_for_well`) + merged (`validate_frame_inventory`).

### The strict validator = ordered, composable checks (NOT a monolith)
```
validate_frame_inventory(input_csv, output_flag, *, image_root=None, check_sources=...):
  [L0] validate_dataframe_schema       — existing (columns + nulls)
  [L1] _validate_unique_keys           — existing (identity-anchored image_id uniqueness)
  [L2] assert_derived_ids_consistent   — existing
  [L3] per-well grain (ALWAYS runs)    — NEW: one experiment_id, one well_index; BF contiguous 0..N-1;
                                          channels rectangular; multi-frame ⇒ elapsed_time_s required (Thread 2)
  [L4] disk / source (check_sources)   — NEW: the IMAGE/FOLDER CONTRACT (below)
```

### `check_sources` is a MODE FLAG (one validator, two modes — matches validate_yx1_acquisition_inventory)
- **Drop-in / external frame inventory: `check_sources=True` by DEFAULT.** (The whole point — prove
  the claimed images are real.)
- **Internal merged/aggregate validation: `check_sources=False` allowed** (don't re-open every image
  for the aggregate view; L0–L3 still run).
- L3 grain checks ALWAYS run regardless of the flag.

### L4 = the IMAGE / FOLDER CONTRACT (no separate folder-contract object)
For each row, gated by `check_sources=True`:
- `source_image_path` resolves (policy below)
- file **exists**
- image **opens** (TIFF/PNG/JPEG)
- real pixel dims **==** declared `image_width_px` / `image_height_px`
- `source_micrometers_per_pixel` **> 0`

> **"well_id folder contract" is NOT a new mayor — it's this optional stricter inspection mode on the
> frame_inventory validator.** Ownership split, locked:
> - **`paths.py` owns WHERE pipeline artifacts live** (shard / sentinel layout under `per_well/{well_id}/`).
> - **`frame_inventory` validator owns WHAT acquired/materialized images CLAIM to be** (the table).
> - **L4 source checks PROVE the claimed image paths are real.**
> Tiny doctrine: *The table makes the claim. The image root gives relative claims a home. The strict
> validator makes the claim touch disk.*

### Path-resolution policy (absolute OR relative; locked)
```
if path.is_absolute():
    resolved = path                       # validate directly — do NOT force inside image_root
else:
    if image_root is None:                # relative + check_sources=True + no root → FAIL LOUD
        raise ValueError("relative source_image_path requires image_root when check_sources=True")
    resolved = image_root / path
    assert resolved.resolve().is_relative_to(image_root.resolve())   # no .. escape
```
- Accept both. Lean: encourage relative (portable); validator canonicalizes to absolute in the shard.
- Absolute paths are NOT forced inside `image_root` (external users keep their own absolute trees).
- Relative paths must not escape `image_root` via `..` (safety).
- Thread `--image-root` through the `validate-frame-inventory` task verb as OPTIONAL.

### Layout (`materialized_image_paths.py`) — NO CLASS YET
- Keep pure functions (atoms → Path, no state). A class with no instance data = the "mayor not
  clipboard" anti-pattern.
- **No `LayoutSpec`** unless we later support user-configurable tree templates (then a small frozen
  clipboard dataclass holding template + suffixes, passed to the functions — never methods).
- If we need **inverse discovery** ("given images, what atoms?" — the Thread-4 scaffold helper), add
  **separate free functions** for the read direction; do NOT turn path construction into a class.

---

## CORRECTIONS to apply before writing the spec (mdcolon review, 2026-06-23) — ✅ LOCKED

1. **Time rule is TEMPORAL, not id-count.** Replace "more than one image_id per well_id" with:
   **"If a well has more than one distinct `time_index` / timepoint, `elapsed_time_s` is required."**
   (image_id may include channel — BF+fluorescence at one timepoint = multiple image_ids but still a
   single-timepoint snapshot. The requirement is temporal multiplicity, not id multiplicity.)
2. **Canonical age column = `start_age_hpf`.** Accept `age_hpf` AND `start_age_hpf` as INPUT aliases;
   **normalized output contains `start_age_hpf`** (keeps the stage formula honest:
   `predicted_stage_hpf = start_age_hpf + elapsed_h * rate(temperature)`).
3. **Long-ingest behavior LOCKED (was deferred → now decided): option (a).** Long-ingest **emits the
   canonical nullable biology columns even when absent** (`genotype`, `start_age_hpf`, `temperature`,
   `medium` → all-NA if the user omitted them). L2 column SHAPE stays stable regardless of which
   fields the user supplied; consumer completeness gates decide if NA is allowed. *Shape stability is
   pipeline oxygen.* (Supersedes the Thread-3 "lean (a), decide at build time" note.)
4. **Pixel-size column name must match the ACTUAL contract — no ghost column.** frame_inventory uses
   `source_micrometers_per_pixel` (verified in `frame_inventory_contract.py`); acquisition inventory
   uses `micrometers_per_pixel`. The spec must name each where it actually lives and NOT introduce a
   rename. The validator RULE is "pixel-size column > 0" — applied to whichever column the contract
   already defines at that seam.
5. **Validator is NON-MUTATING.** Do NOT say "validator canonicalizes to absolute in the shard."
   Correct wording: **"relative paths are resolved against `image_root` DURING validation; the source
   CSV is NOT rewritten by the validator."** (validate writes only the `.validated` sentinel / errors
   report — consistent with the one-file model.) A normalized-absolute-path CSV, if ever wanted, is a
   separate explicit build/normalize step, not the validator.

## CARRY-INTO-SPEC DOCTRINE (mdcolon)
> The external handoff does NOT create a second pipeline. It creates a **stricter entrance into the
> same pipeline.** No "external mode" goblin kingdom.
>
> The manifest makes claims. The metadata gives biological meaning. The validator makes the claims
> touch disk. The consumers decide which biology fields are required.

## FILE PLAN — what we ADD / CHANGE / leave UNTOUCHED (mdcolon-ruled, 2026-06-23) — ✅ LOCKED

Naming doctrine (mdcolon): **validator = the gate; rules = the contract clauses the gate enforces.**
"checks" rejected as too vague. L2 doctrine: *columns are the skeleton, values are the blood — L2
checks the skeleton, consumers check whether the blood is needed.*

### A — Frame inventory validator (SPLIT the gate out + add rules) — FLAT modules, NO subpackage
| File | Action | Role |
|---|---|---|
| `metadata_ingest/frame_inventory/frame_inventory.py` | **CHANGE** | Becomes product/table operations ONLY (keep `merge_frame_inventory_shards`, the reader). The validator MOVES OUT. |
| `metadata_ingest/frame_inventory/frame_inventory_validation.py` | **ADD** | The public gate `validate_frame_inventory(input_csv, output_flag, *, image_root=None, check_sources=True)` — orchestrates which rules apply (L0 schema → L1 uniqueness → L2 derived-ids → L3 grain → L4 sources). Moved/renamed from today's in-`frame_inventory.py` validator. |
| `metadata_ingest/frame_inventory/frame_inventory_validation_rules.py` | **ADD** | The product-specific contract RULES (clauses, not incidental helpers): BF-contiguity, rectangular-channels, elapsed-time (multi-timepoint), source-image readability, declared-dimensions, path-resolution (abs/rel + image_root + no-`..`-escape). **NOT** `_checks.py`. |

> 🔒 **DECIDED (mdcolon, hard-ass review 2026-06-23): FLAT modules now — do NOT create a `validation/`
> subpackage.** Reversed an earlier permissive lean. Reasoning that must not be re-litigated:
> - A `validation/` package is justified when validation becomes a **subsystem**. Right now it is ONE
>   public gate + ONE rules module — not a subsystem. Two files don't earn a package boundary.
> - The promotion trigger is **structural multiplicity, NOT importance and NOT filename length.** The
>   rules are load-bearing CONTRACT CLAUSES (that's why "rules" not "helpers") — but being important
>   does not make them a kingdom. "Do not create a package because two filenames are long."
> - Flat costs: longer filenames. Flat buys: obvious imports, no `__init__.py` re-export semantics,
>   lower [[pytest-import-mode-importlib]] fragility, easier grep/tests. Acceptable trade.
> - **PROMOTE to `frame_inventory/validation/{validate_frame_inventory,grain_rules,source_rules}.py`
>   LATER, only if:** the rules module exceeds ~400–500 lines; OR source validation needs multiple
>   backends; OR grain/source/schema rules need independent test fixtures that become painful; OR a
>   second public validation gate appears; OR external drop-in validation becomes a public API surface
>   separate from native. NONE of these hold today.

> Validator NON-MUTATING (correction #5): resolves relative paths against `image_root` DURING
> validation; never rewrites the source CSV. Writes only the `.validated` sentinel / errors report.

### B — Biology long-table ingest (finish in place)
| File | Action | Role |
|---|---|---|
| `metadata_ingest/plate/plate_metadata_loader.py` | **CHANGE** | Implement `ingest_plate_metadata_long_sheet` (today raises). Age-alias `age_hpf`→`start_age_hpf`. **Emit canonical nullable biology columns even when absent.** |
| `metadata_ingest/plate/plate_metadata_contract.py` | **CHANGE (confirm/minor)** | L2 keeps the 4 biology columns REQUIRED-AS-COLUMNS but **allows NA values** (no shape relaxation). Confirm nullability is expressed; do NOT relax to variable shape. |

### C — Drop-in discovery + split (in well_discovery/, beside the twin)
| File | Action | Role |
|---|---|---|
| `metadata_ingest/well_discovery/discover_wells_from_handoff.py` | **ADD** | Twin of `discover_wells_from_scope_metadata.py`: read big `dropin_frame_inventory.csv`, assert one `experiment_id`, derive `well_id`, reuse `discovered_wells_contract.py`. |
| `metadata_ingest/well_discovery/split_dropin_inventory.py` | **ADD** | `split_dropin_inventory_by_well()` → per-well shard at the `paths.py` frame_inventory path. |

> **NOT** creating `stitched_handoff/` or `metadata_ingest/dropin/` — no external-data kingdom yet.
> Grouped by FUNCTION (discovery), reusing the existing contract.

### D — Scaffold helper (Phase 1: emit starter CSV, no reorg)
| File | Action | Role |
|---|---|---|
| `metadata_ingest/frame_inventory/scaffold_dropin_inventory.py` | **ADD** | Inverse-direction free functions: read image dir, read headers for real dims, emit a starter `dropin_frame_inventory.csv` the user edits. NOT a class; does NOT reorganize files in Phase 1. |

### E — Orchestration wiring
| File | Action | Role |
|---|---|---|
| `pipeline_orchestrator/tasks.py` | **CHANGE** | `validate-frame-inventory` gains `--image-root` + `--check-sources`; new thin verbs: `discover-wells-from-handoff`, `split-dropin-inventory`, `scaffold-dropin-inventory`, `ingest-plate-metadata` already exists (long path is internal to loader). |
| `pipeline_orchestrator/rules/frame_inventory.smk` | **CHANGE** | per-well node → `check_sources=True` + image_root; merged node → `check_sources=False`. |
| `pipeline_orchestrator/rules/dropin_handoff.smk` | **ADD** | External-entry orchestration rules (discover-from-handoff, split, scaffold). Orchestration grouped by WORKFLOW ENTRANCE; product logic stays in the `well_discovery/` + `frame_inventory/` functional homes. Routes artifacts through `paths.py`. NOT a new source-code package. |

> 🔒 **`.smk` ruling (mdcolon, 2026-06-23) — LOCKED, no escape hatch:** drop-in entry rules go in a
> sibling `pipeline_orchestrator/rules/dropin_handoff.smk`, NOT in `frame_inventory.smk`. Source code
> is grouped by FUNCTION; orchestration is grouped by WORKFLOW ENTRANCE. Both files stay readable; this
> is explicitly NOT an external-data source kingdom (no new package).

### F — UNTOUCHED (explicit)
- `image_materialization/materialized_image_paths.py` — stays PURE FUNCTIONS, no class, no edit.
- `image_materialization/frame_inventory_contract.py` — the column manifests/derived-id helpers are
  already correct (the rules MODULE imports from here; does not duplicate it).
- `io/validators.py::validate_dataframe_schema` — the generic L0 primitive; reused, unchanged.

### G — Tests (mirror tree)
`tests/data_pipeline/metadata_ingest/frame_inventory/`: `test_frame_inventory_validation.py`,
`test_frame_inventory_validation_rules.py`, `test_scaffold_dropin_inventory.py`.
`.../plate/test_plate_metadata_loader.py` (extend: long path).
`.../well_discovery/`: `test_discover_wells_from_handoff.py`, `test_split_dropin_inventory.py`.

**Tally:** **6 ADD code/orchestration files** (`frame_inventory_validation.py`,
`frame_inventory_validation_rules.py`, `discover_wells_from_handoff.py`, `split_dropin_inventory.py`,
`scaffold_dropin_inventory.py`, `dropin_handoff.smk`) **+ mirror tests**;
**5 CHANGE** (`frame_inventory.py` split, `plate_metadata_loader.py`, `plate_metadata_contract.py`,
`tasks.py`, `frame_inventory.smk`); **3 UNTOUCHED-by-design**. No new module, no class.

## STATUS: 6 threads + 5 corrections + file plan all LOCKED. Spec NOT yet written.
