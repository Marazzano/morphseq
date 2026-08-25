# Front-half file organization audit — 2026-06-18

**What this is:** a one-pass audit (mdcolon-requested) of the front-half pipeline files for
organization quality and **concern-mixing**, against the section-banner convention now codified in
`pipeline_file_philosophy.md` ("One file reads top-to-bottom in flow order, with section banners
marking each concern"). The reference exemplar is `metadata_ingest/scope/yx1/acquisition_inventory.py`
(Contract → Validation → Builder).

**How to read severity:**
- **HIGH** — genuine concern-mixing; the fix is a SPLIT (separate functions/files), not a banner.
- **MEDIUM** — banners + a small reorder / one helper extraction.
- **LOW** — already clean (or a single-concern leaf that needs nothing).

**Scope note:** this is a findings list, not an executed refactor. The two HIGH items already map to
existing planned work — do them there, not as drive-by edits:
- `materialize_stitched_images.py` split = the locked "split materialization per-scope" open decision
  (`current_state_and_next_steps.md` 2026-06-16 block) + Step 7 strangler territory.
- `tasks.py::cmd_materialize_well` slimming = the thin-dispatcher immutable anchor; pairs naturally
  with Step 7.

---

## HIGH — split, don't banner

| File | The mix | Fix |
|---|---|---|
| `metadata_ingest/stitched_index/materialize_stitched_images.py` (662 lines) | TWO microscopes in one mega-function: `if microscope=="YX1" / elif "Keyence"` (~L439–458, L501–594) + scope-specific helpers (YX1 L107–159, Keyence L161–387) feeding one row-builder (L596–650). | Split into `materialize_stitched_images_yx1` / `_keyence` + a thin dispatcher. This is the **locked per-scope-split decision** + Step 7 — do it there. (The live spine already moved to `image_materialization/`; this legacy file is a Step-7 strangler target anyway.) |
| ~~`pipeline_orchestrator/tasks.py::cmd_materialize_well`~~ ✅ DONE 2026-06-18 | Dispatcher did dataframe algebra: CSV read + position→well `merge`, row filter, empty-check. Violated the thin-dispatcher anchor. | **FIXED:** the join + per-well row slice moved to a pure domain function `image_materialization/select_well_acquisition_rows.py::select_well_acquisition_rows(acq_df, mapping_df, *, experiment_id, well_id) -> df`. `cmd_materialize_well` is now read CSVs + validate mapping (file boundary) → adapter → `run_materialize_well` → write. See placement note below. |
| `image_materialization/scope/yx1/materialize_well_yx1.py::materialize_yx1_well` (~187-line fn) | One function spans entry-guard + ND2 tensor I/O + per-time materialize loop + inventory assembly. | Add INTERNAL banners (Entry guard → ND2 source/tensor setup → Materialization loop → Inventory assembly); optionally extract the loop body to `_materialize_frame_and_record`. (Lower-risk than the two above — banners + optional extract, not a hard split.) |

## MEDIUM — banners + small reorder / one extraction

| File | The mix | Fix |
|---|---|---|
| `image_materialization/frame_inventory_contract.py` | Contract constants ↔ derived-id helpers ↔ consistency-guard logic ↔ dataclasses intermingle (L31–193). | Banners: Contract constants → Contract shapes (dataclasses) → Derived-id helpers → Validation guards; move dataclasses up after constants. |
| `image_materialization/materialization_plan.py` | Global vocab ↔ exceptions ↔ shapes ↔ config-loader/validators (nouns vs verbs interleaved). | Banners: Global Vocabulary → Exceptions → Contract Shapes → Loading/Validation. |
| `metadata_ingest/frame_inventory/frame_inventory.py` | Live validator + live merger + LEGACY frame-contract adapter (strangler debt) co-live (L43–163). | Banner the legacy block as "Legacy Frame Contract Adapter (strangler; Step-7 target)" so live vs dead reads at a glance; full removal is Step 7. |
| `metadata_ingest/scope/yx1/map_yx1_positions_to_wells.py` | Main fn mixes load → validate → map → fallback → **diagnostics/provenance packing** (L212–264) → output. | Extract the provenance/diagnostics packing into `_build_provenance(...)`. |
| `metadata_ingest/scope/shared/apply_position_to_well_mapping.py` | Identity join + id construction + **time-column enrichment** (L82–109) in one function. | Banner / extract the time-enrichment step as its own labeled section or helper. |

## LOW — already clean (no action, or optional cosmetic banners)

`image_materialization/run_materialize_well.py` (exemplary sequencer) ·
`image_materialization/scope/scope_resolver_for_materialization_plan.py` (router → YX1 → reserved
Keyence) · `image_materialization/materialized_image_paths.py` (pure path constructor) ·
`metadata_ingest/scope/yx1/extract_yx1_scope_metadata.py` (concerns isolated into functions) ·
`generate_xy_reference.py` · `validate_xy_reference_grid.py` · `scope/shared/acquisition_checks.py` ·
`scope/shared/validate_physical_well_mapping.py` · `well_discovery/*` ·
`position_well_mapping/position_well_mapping_contract.py` · `orchestration/paths.py` (REFERENCE) ·
`orchestration/well_runner.py` (REFERENCE) · `pipeline_orchestrator/targets.py` (single-concern leaf).

---

## Placement note — where the extracted adapter lives, and why (the `well_runner` boundary)

The adapter is a **flat file** in `image_materialization/` (the package convention is flat — filenames
carry the moment; only `scope/` is a subfolder, because the code genuinely forks by microscope there).
A `well_materialization/` subpackage was considered and REJECTED: the whole `image_materialization/`
kingdom already *is* "materialize one well," so a `well_materialization/` subfolder is a tautology
(`image_materialization/image_materialization/`), and a subfolder is earned only when the code *forks*
(like `scope/`), not when it merely *sequences*.

It is named `select_well_acquisition_rows`, NOT `…_shard`, deliberately. A "shard" is
**`well_runner`'s** vocabulary for the per-well frame_inventory **OUTPUT** file. Our adapter produces
the acquisition **INPUT rows** for one well — the opposite side of the materialize step:

    well_runner (orchestration):  which well_ids run? → per-well frame_inventory SHARD paths (OUTPUT)
    select_well_acquisition_rows: acquisition_inventory rows → the ROWS for one well (INPUT)

Verified: `image_materialization/` imports `well_runner` **zero** times, and the only repo-wide
importer of `well_runner` is `orchestration/__init__.py` (its home). The adapter keeps it that way —
it takes DataFrames + a `well_id` it was handed and returns rows; it never computes a run set or
resolves a path. **A domain kingdom never reaches up into the scheduler.** That import boundary
(domain ↛ orchestration) is the real reason the two concepts must not share vocabulary.

## Principles extracted from the reference files (now in `pipeline_file_philosophy.md`)

`paths.py` and `well_runner.py` are the project's organization exemplars. What makes them clean,
distilled into the convention this audit measured against:
1. **Section banners in flow/dependency order** — the file reads like the pipeline it serves.
2. **Docstring orients before code** — jobs + why + boundaries + hard rules, before line 1 of logic.
3. **Every concept built in exactly one place** (compose downward; no `.parent` stripping).
4. **Look-alikes disambiguated by moment AND return type** (`run_*` planning/pure vs `collect_*`
   runtime/disk).
5. **Thin dispatchers stay thin** — a `tasks.py` verb that contains a `pd.merge` has too many jobs.
6. **Banners separate co-living concerns; a split fixes a real mix** (two microscopes, two kingdoms).
