# Plate Metadata Ingest + Entity Metadata Completeness — three-layer validation (🟢 TARGET)

**Status:** planning spec, mdcolon 2026-06-22. This doc owns how **plate metadata** (the authored Excel
design file — genotype, condition, plate geometry) is ingested, validated, and finally **injected** into
the pipeline at the point it matters. It splits one tangled stage into **three validation moments** and
adds the missing gate that catches "a real embryo with no metadata."

**Companion to:** `front_end_naming_and_frame_inventory_flow.md` (the plate lineage is the *other root* —
ingested early, joined late at `consolidate_features`), `pipeline_file_philosophy.md` (how the loader's
functions must read), `output_tree_doctrine.md` (the `acquisition/` regime + `PIPELINE_STEPS` registry),
and `../features/targets/feature_world.md` (the `snip_id` feature universe L3 judges over).

**Scope of THIS doc:** the plate-metadata ingest design (loader + contract), and the **specification** of
the deferred entity-completeness QC gate. **Phase-1 (loader + contract) is described here; the L3 QC
product is spec-only and built later.**

---

## 🎯 WHY THIS DOC EXISTS

Features need real plate metadata (genotype, temperature, medium, …). Today
(`metadata_ingest/plate/plate_processing.py`, ~230 lines) loading, multi-sheet Excel parsing, column
normalization, and validation are all tangled in one file. That causes the chronic failure modes this doc
removes:

- generic `Missing required columns: [...]` errors with **no fix hint**;
- a **silent `temperature = 28.5` default** that *invents data*;
- extra sheets (e.g. the recent **`sequenced`** page) ravel through only implicitly — undocumented,
  untrusted, easy to break;
- required columns live in `schemas/plate_metadata.py` (legacy-compat), not a co-located contract.

Because **changing the metadata forces a full pipeline rerun**, this stage must be robust and
self-explaining. It is where work gets halted.

---

## 🪨 THE CENTRAL INSIGHT — three validation moments, not one

> **Fail at the mouth of the river. Validate again before the water joins the sea.**

Plate-metadata validation is really **three different questions asked at three different times**, and
collapsing them is the bug. A **blank well** may legitimately have **blank metadata** — there were no
embryos there, so a biologist correctly left it empty. A **real embryo** may *not* be missing its
metadata. So:

| Layer | Question it answers | Where it lives | STRICT about | LAX about |
|---|---|---|---|---|
| **L1 — ingest / loader** | "Can I read this file into a sane well-indexed long table — and fail loud if it isn't sensibly laid out?" | `plate_metadata_loader.py` | file/grid **shape**, `well_index` sanity, ingest ambiguity, output-column collisions | biological values (cells may be blank) |
| **L2 — contract** | "Does the assembled artifact have the required columns, by name?" | `plate_metadata_contract.py` | required **columns exist** + identity consistency + no dup keys (plain schema check) | **null** biological values at plate/well level (allowed) |
| **L3 — entity metadata completeness QC** | "Conditional on a physical embryo being **registered** (`physical_embryo_registry`), does it actually have the required metadata entries?" | deferred `entity_metadata_completeness_qc` | required values **non-null for already-registered real entities** | discovery/registration itself (NOT its job); wells with no embryos |

```
L1 ingest          L2 contract            L3 entity completeness QC
─────────          ──────────             ─────────────────────────
"Is this file      "Does the table have   "Do the embryos that ACTUALLY
 readable + sane?"   the columns we need?"  EXIST have their metadata?"
   shape               schema                 completeness, conditional
                                              on physical_embryo_registry
fail at the mouth ───────────────────────►  before the water joins the sea
```

### The phase model (where metadata gets injected)

The plate metadata is **ingested early but only becomes load-bearing late** — L3 is the injection point.

- **Phase 1** — plate_metadata **ingest** (parse, normalize, validate shape + columns + identity; **allow
  nulls**). ← *this doc builds this.*
- **Phase 2** — object/entity generation (tracking **discovers** which wells yielded embryos;
  `physical_embryo_registry` **registers** them — the authoritative real-entity set, one row per
  `physical_embryo_id`; see `../detect-seg-track/targets/physical_embryo_registry_world.md`).
- **Phase 3** — `entity_metadata_completeness_qc` (join the **registered**-entity universe to
  plate_metadata; flag real entities missing required metadata). ← *spec'd here, built later.*
- **Phase 4** — `snip_qc` consumes `metadata_missing_flag` as one exclusion reason (`missing_metadata`).
- **Phase 5** — `analysis_ready` joins features + metadata + the QC verdict.

> Tiny doctrine: **Ingest validates the map. The contract validates pipeline truth. Entity QC validates
> the territory. Analysis-ready joins only trusted territory.**

---

## 🌊 LAYER 1 — `plate_metadata_loader.py` (the dedicated loader)

> **Renamed** from the historical inline parsing (`_parse_plate_metadata_excel`) to a standalone,
> important-by-design module. It reads Excel workbooks today and is **designed to take standalone CSVs
> later** — so the name is `plate_metadata_loader`, not `plate_excel_loader`: **name the interface, not
> the file extension.**

**The loader's boundaries (what it does NOT do):** it reads pages and assembles a long table. It does
**not** mint identity (`well_id` is minted by the orchestrator), does **not** enforce the required-field
schema (that is L2), and does **not** write side files. Per `pipeline_file_philosophy.md`, write the key
functions with **verbose, self-narrating names — code more, organized, over contract-compact.** A reader
should be able to follow the loader by reading its function names top to bottom.

### The two-ingester seam (the load-bearing design idea)

The canonical intermediate is a **long table**: one row per `well_index`, one column per value. There are
**two named ingest modes** that both converge to that long table:

| Ingest mode | Reads | Status |
|---|---|---|
| **grid** — `ingest_plate_grid_sheet_to_long` | the standard MorphSeq **8×12** sheet (rows A–H × cols 1–12) → long | **BUILT in MVP** |
| **long** — `ingest_long_well_table` | a page/CSV that already carries a derivable `well_index` → long | **DESIGNED-IN, deferred** |

**Provenance lives in `PlatePages.page_methods`, NOT as a row-level column.** Which ingest mode read each
page is recorded per-page in `page_methods` and the accepted/rejected summary — **never** as a single
`ingest_format` column on `plate_metadata.csv`. A row-level format column becomes *wrong* the moment
different pages use different ingest modes (genotype via grid, sequenced via long → one column cannot
describe the row).

> **MVP decision (mdcolon 2026-06-22):** just get plate metadata flowing — build the **grid** ingester
> only. The long ingester adds quality qualifications (it requires that `well_index` is already derivable),
> so it is **specified now and slotted into the same dispatch, built later**. Designing the dispatch + the
> per-page provenance now is what makes that drop-in clean. In MVP, long-shaped pages are **rejected** with
> a clear reason.

### Public surface

```python
@dataclass(frozen=True)
class PlatePages:
    table: pd.DataFrame              # canonical long table, one row per well_index
                                     #   (NO row-level ingest_format column)
    accepted: list[str]              # page names that became columns
    rejected: list[tuple[str, str]]  # (page_name, human reason) — the "didn't meet the format" list
    page_methods: dict[str, str]     # page name -> "grid" | "long"  (ingest provenance lives HERE)


def load_plate_metadata_pages(input_file: Path) -> PlatePages: ...
```

The loader: open the workbook → iterate **all** pages → `classify_plate_page` each → dispatch to the
matching `ingest_*` function → outer-merge accepted pages by `well_index` → return `PlatePages`.

### Detection + dispatch

`classify_plate_page(df) -> ("grid" | "long" | "rejected", reason)` selects the ingest mode per page.

- If a page is readable as **both** grid and long → **fail loud** (ambiguity guard — "shouldn't happen but
  maybe"; never silently pick one).
- A `long`-classified page in MVP → **rejected** with a clear *"long-table ingest not implemented in MVP"*
  reason (recognized, not silently skipped — the seam stays visible).

### Grid ingester (MVP)

`ingest_plate_grid_sheet_to_long(df, page_name) -> long DataFrame` — produces **one** value column named
the **normalized page name**. **Fail loud on a grid that is not sensibly laid out:**

- rows must be **A–H in sequence** (normalizable) — no arbitrary row labels;
- columns must be **1–12 in sequence** — no arbitrary ranges (no silently accepting 1–10 or 2–13);
- the full A–H × 1–12 **span** must be present;
- every cell maps deterministically to exactly one `well_index`; **no duplicate `well_index`**.

### Long ingester (deferred — spec only)

`ingest_long_well_table(source) -> long DataFrame` — `source` may be an xlsx sheet frame OR a CSV path;
passes through **all** non-identity value columns. Requires a derivable, unique, in-range `well_index` and
≥1 value column. Documented now so the dispatch contract is complete and the future build is a drop-in.

### `well_index` comes from `shared/identifiers/`, NOT reinvented in the loader

Both ingesters produce a `well_index` (the grid mints it from each A–H × 1–12 cell; the long reader
normalizes a `well` / `well_name` column into it). That `well_index` is **identity**, so per
`pipeline_file_philosophy.md` hard constraint #1 it must be built/validated by `shared/identifiers/` —
**the loader never hand-rolls a `f"{row}{col:02d}"` or its own zero-pad/range check inline.** The whole
point is to **propagate the one canonical `well_index` grammar** (the same one `build_well_id` consumes)
everywhere, so a `well_index` minted here is byte-identical to one parsed out of a `well_id` downstream.

- The loader **imports** the identifier helpers; `shared/identifiers/` never imports the loader
  (identity flows *into* the pipeline, one-way).
- **Today's gap:** `shared/identifiers/` has `build_well_id` / `validate_well_id` / `split_well_id`, but
  **no canonical `well_index` normalizer/validator** (the `A01` local-label grammar — letter A–H + 1–12,
  zero-padded). The loader needs exactly that, so **add it to `shared/identifiers/`** (e.g.
  `normalize_well_index(row, col)` / `validate_well_index(label)` in `constructors.py` / `validators.py`)
  and have the grid + long ingesters call it. This is a small, reusable addition — `position_well_mapping`
  and any other producer of `well_index` should converge on the same helper rather than each rolling its
  own.
- Fail-loud lives in the helper: a label outside A–H × 1–12 raises with a message naming the fix, so the
  rule "grid cols must be 1–12 in sequence" is enforced *by the identifier grammar*, not duplicated logic.

### Merge policy

Accepted pages are **outer-merged** by `well_index` (allows partial pages — a page need not cover all 96
wells). The grid output column is the **normalized page name**.

### Loader failure semantics — reject (soft) vs fail-loud (hard)

The loader distinguishes a **page it can ignore** from a **condition that must abort the load**. The rule:
a page that simply *isn't plate metadata* is **rejected** (recorded, skipped); anything that signals the
input is *malformed or contradictory* is **fail-loud** (raises, no artifact written).

| Condition | Behavior | Layer |
|---|---|---|
| Unknown / unparseable extra page (not grid, not long, just not plate data) | **rejected with reason** → `PlatePages.rejected` | L1 |
| Page classified **grid but malformed** (rows not A–H in seq, cols not 1–12, missing span, dup `well_index`) | **fail loud** | L1 |
| Page readable as **both grid and long** (ambiguous) | **fail loud** | L1 |
| Long-classified page (MVP) | **rejected with reason** ("long ingest not implemented in MVP") | L1 |
| **Normalized output-column collision** across accepted pages | **fail loud** | L1 |
| **Missing required field** (`genotype` / `start_age_hpf` / `temperature` / `medium`) | **fail loud with fix hint** | **L2 (not L1)** |

> The split matters: **L1 never decides "you're missing genotype."** It assembles whatever valid pages it
> found. **Required-field absence is L2's job** (checked against `REQUIRED_PLATE_METADATA_COLUMNS`, with the
> fix hint). L1 = *can I read these pages and are they internally sane?*; L2 = *does the assembled artifact
> have what the pipeline requires?*

### `series_number_map` — RESOLVED: the loader writes no side files

Today `plate_processing.py:49-53` writes a `series_number_map.csv` next to `plate_metadata.csv`.
**Audit (2026-06-22):** the only reference to that CSV is the writer itself — **no reader exists**. The
real consumers (`metadata_ingest/scope/yx1/generate_xy_reference.py`,
`build/build01B_compile_yx1_images_torch.py`) read the `series_number_map` **sheet directly from the
workbook** (`pd.read_excel(..., sheet_name="series_number_map")`), not the side-write. So:

- the **`series_number_map.csv` side-write is REMOVED** (dead artifact);
- the loader writes **no** side files at all (clean boundary — loaders return data, orchestrators write
  artifacts);
- if a future need to materialize `series_number_map` as a registry artifact appears, it gets its own
  `PIPELINE_STEPS` row and is written by an orchestrator/entrypoint — never as a loader side-effect.

---

## 📐 LAYER 2 — `plate_metadata_contract.py` (plain schema validation)

**Plain schema validation**: does the assembled artifact have the columns the pipeline expects, by name?
Nothing fancy. Mirrors `position_well_mapping_contract.py` in shape, co-located with the data product (per
`schema_layout.md`). The **required semantic fields** may come from grid sheets now or long-format columns
later — hence `FIELDS`, not `PAGES`.

```python
REQUIRED_PLATE_METADATA_FIELDS = ("genotype", "start_age_hpf", "temperature", "medium")
REQUIRED_PLATE_METADATA_COLUMNS = (
    "experiment_id", "well_id", "well_index",
    *REQUIRED_PLATE_METADATA_FIELDS,
)


def validate_plate_metadata(df, *, scope_label: str = "plate_metadata") -> None: ...
```

`validate_plate_metadata` validates **required columns + identity consistency only**. It **allows null
biological values** at the plate/well level and does **no biological type enforcement** beyond what
identity / `well_index` need (type enforcement is a later concern). Checks:

- **missing required field/column → fail loud with a fix hint** naming the field AND both input forms:
  `"[plate_metadata] missing required field 'genotype'. Add an 8×12 sheet named 'genotype' (rows A–H,
  cols 1–12) to the well_metadata workbook, or a long-format sheet/CSV with a 'genotype' column."`
- `well_id` is consistent with `build_well_id(experiment_id, well_index)` (pattern from
  `position_well_mapping_contract.py`);
- no duplicate `(experiment_id, well_id)` rows (folds in today's `validate_plate_metadata.py` uniqueness
  check);
- **required columns must EXIST but MAY contain nulls** at plate level — a blank no-embryo well is
  legitimate. **Null/completeness enforcement is deferred to L3.**

> **No more silent defaults.** The historical `temperature = 28.5` fallback is **removed**. A missing
> `temperature` field is now a loud L2 error with a fix hint — you fix it in the Excel (or, later, supply
> it via config), never have it silently invented.

---

## 🧪 LAYER 3 — `entity_metadata_completeness_qc` (SPEC ONLY — deferred build)

The **missing middle step** between snip processing and feature extraction — the moment plate metadata
finally becomes load-bearing and gets *injected* into the per-entity world. The spec defines the product;
**it is built later, not in Phase 1.**

> **It does NOT discover or register.** L3's job is *not* to figure out which embryos exist — that is
> the tracking/registration pipeline's job (`physical_embryo_registry`). L3 is purely **conditional on
> registration having already happened**: for each entity in `physical_embryo_registry`, confirm the
> required metadata fields actually have entries. A blank well with no registered embryos is **never**
> flagged; a registered embryo missing `temperature` **is**.

- **Domain:** `quality_control/entity_metadata_completeness_qc/` (names the failure mode + the entity
  grain). Follows feature_world's per-product folder shape (`contract.py` / `compute.py` / `entrypoint.py`)
  and the QC table doctrine (every boolean output column ends in `_flag`).
- **Grain:** `snip_id` (the feature universe). Metadata is inherited well → physical_embryo → snip.
- **Inputs:** `physical_embryo_registry` (the authoritative **registered**-entity set — *the existence
  signal*; one row per `physical_embryo_id`) + `snip_inventory` (projects registered entities to the
  `snip_id` grain via its explicit `physical_embryo_id` column) + `plate_metadata`. **No separate
  discovery list:** presence in `physical_embryo_registry` *is* "this animal exists." L3 reads the
  explicit `physical_embryo_id` column — it does **not** parse `snip_id` to rediscover the animal.
- **Outputs:** `metadata_missing_flag` (non-null bool) + `missing_metadata_fields` (e.g.
  `"temperature|medium"`).
- **Consumed by** `snip_qc` / `consolidated_qc` as exclusion reason `"missing_metadata"` →
  `metadata_missing_flag`.

```
snip_id                     metadata_missing_flag  missing_metadata_fields
20250912_B01_e01_BF_t0001   false                  ""
20250912_B02_e01_BF_t0001   true                   "temperature"
```

> Why this layer exists: a biologist correctly leaves an empty well empty. We must be **lax** there (L2
> allows nulls) and **strict** once we *know* an embryo is present (L3 flags it). That is the whole reason
> the silent `temperature = 28.5` default was wrong — it papered over exactly the gap L3 is meant to catch.

> **Where "we know an embryo is present" comes from — the registration doctrine:**
> *Segmentation finds masks. Tracking links masks. Entity registration names organisms. Snip processing
> crops organisms.* The "this animal exists" signal is owned by `physical_embryo_registry`
> (`../detect-seg-track/targets/physical_embryo_registry_world.md`) — **not** buried as a side effect of
> snip processing. L3 consumes that registry as its existence signal; `snip_inventory` is only the
> crop manifest that projects registered entities to `snip_id` grain.

---

## 🛠️ PHASE-1 IMPLEMENTATION (the build that follows this spec)

All under `src/data_pipeline/metadata_ingest/plate/` (sibling to `position_well_mapping/`):

1. **`plate_metadata_loader.py`** — NEW (L1). Grid ingester + the `classify_plate_page` dispatch seam +
   per-page `page_methods` provenance (no row-level format column). Long ingester is recognized but
   **rejected** with a clear "not implemented in MVP" reason (visible seam). Verbose, self-narrating
   names. Ingest-shape validation only; no identity, no schema, **no side-writes**.
2. **`plate_metadata_contract.py`** — NEW (L2). `REQUIRED_PLATE_METADATA_FIELDS` /
   `REQUIRED_PLATE_METADATA_COLUMNS` + `validate_plate_metadata` with fix hints.
3. **`plate_processing.py`** — REFACTOR to a thin orchestrator: `load_plate_metadata_pages` → mint
   `well_id` via `build_well_id` → `validate_plate_metadata` → write CSV → print the accepted/rejected
   page summary. **Delete the silent `temperature = 28.5` default.** Drop the inline grid parsing
   (moved into the loader). **Remove the dead `series_number_map.csv` side-write.**
4. **`validate_plate_metadata.py`** — delegate the body to
   `plate_metadata_contract.validate_plate_metadata`; keep the CLI + `.validated` sentinel write.
5. **`schemas/plate_metadata.py`** — keep as a **legacy re-export** of `REQUIRED_PLATE_METADATA_COLUMNS`
   so existing importers don't break (per AGENT_QUICKSTART: no new semantics in `schemas/`). New code
   imports from `plate_metadata_contract`.
6. **`shared/identifiers/`** — ADD the canonical `well_index` helper(s) the loader needs
   (`normalize_well_index` / `validate_well_index` for the A–H × 1–12 local-label grammar) if not already
   present, and have both ingesters call them. Propagate the same helper to other `well_index` producers
   (e.g. `position_well_mapping`) so the grammar has one home.

**Paths / registry: no change.** `ingest_plate_metadata` already exists in `PIPELINE_STEPS`
(`orchestration/paths.py`: `stage="acquisition"`, `fanout=EXPERIMENT`, `artifacts={"csv":
"plate_metadata.csv"}`), wired via the Snakefile rule + `tasks.py cmd_normalize_plate`. This work is
internal to the stage — no new row, no new rule. The config knob for page allow/deny is **post-MVP** (no
`config.yaml` change now): MVP allows every valid page through automatically; the contract enforces only
the required set; extras (e.g. `sequenced`) ride along.

### Tests (parallel tree: `tests/data_pipeline/metadata_ingest/plate/`)

`test_plate_metadata_loader.py` (MVP = grid ingester):
- grid sheet parses to `well_index`-keyed long rows;
- `page_methods` records `"grid"` for accepted grid pages (no row-level format column on the table);
- grid with columns not 1–12 in sequence (e.g. 2–13) → fail loud;
- grid with rows not A–H in sequence → fail loud;
- unknown/unparseable extra page → in `rejected` with a readable reason;
- a `long`-classified page → **rejected** with the "long ingest not implemented in MVP" reason;
- ambiguity guard: page readable as both grid and long → raises;
- an extra valid grid page (`sequenced`) flows through, listed in `accepted`, NOT in required fields;
- normalized output-column collision across accepted pages → raises.

`test_plate_metadata_contract.py` (mirror `test_frame_detections_contract.py`):
- required columns is a tuple; a valid df passes;
- missing `genotype` → `ValueError` naming `genotype` + the "add a sheet" fix hint;
- **missing `temperature` raises** (guards against re-introducing the silent 28.5 default);
- a required column present but with **null values passes** (nulls allowed at L2; completeness is L3);
- duplicate `(experiment_id, well_id)` → raises; inconsistent `well_id` → raises.

### Verification

1. `PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output python -m pytest
   tests/data_pipeline/metadata_ingest/plate/ -q`
2. End-to-end on a real workbook with a `sequenced` page
   (`metadata/morphseq_maps/*_well_metadata.xlsx`) via the existing task:
   `... tasks ingest-plate-metadata --input-file <xlsx> --experiment <exp> --output-csv <out>` — confirm
   `sequenced` is a column, the accepted/rejected summary prints, and a workbook missing a `temperature`
   sheet now errors with the fix hint (no silent default).
3. Grep importers of `REQUIRED_COLUMNS_PLATE_METADATA`; the legacy re-export keeps them green.

---

## ✅ DECISIONS LOCKED IN THIS DOC (mdcolon 2026-06-22)

1. **Three validation layers, not one** — L1 shape (ingest), L2 schema (contract), L3 completeness (entity
   QC). Fail at the mouth of the river; validate again before the sea.
2. **Loader renamed** to `plate_metadata_loader.py` — name the interface, not the extension. Loader returns
   data; mints no identity, enforces no schema, **writes no side files.**
3. **Two-ingester seam**, canonical = **long table**. **grid** ingester built in MVP; **long** ingester
   designed-in, deferred (rejected in MVP with a clear reason).
4. **Ingest provenance lives in `PlatePages.page_methods`**, NOT a row-level `plate_ingest_format` column.
5. **Grid is strict** — rows A–H in sequence, cols 1–12 in sequence, full span; deterministic
   cell→`well_index`.
5a. **`well_index` is identity → comes from `shared/identifiers/`**, never reinvented inline. Add the
   canonical `well_index` normalizer/validator there (today's gap) and propagate it to every `well_index`
   producer so the grammar has one home.
6. **Failure semantics:** non-plate/long/unparseable pages are **rejected (soft)**; malformed-grid,
   grid/long ambiguity, and output-column collision are **fail-loud (hard)**.
7. **Required semantic fields** = `genotype`, `start_age_hpf`, `temperature`, `medium`
   (`REQUIRED_PLATE_METADATA_FIELDS`). **Missing required field is L2's job**, with a fix hint.
8. **No silent defaults** — the `temperature = 28.5` fallback is removed.
9. **L2 allows null biological values at plate level**; no biological type enforcement beyond identity /
   `well_index` needs.
10. **L3 does NOT discover or register** — it is conditional on `physical_embryo_registry` (the
    authoritative registered-entity set; `snip_inventory` projects it to `snip_id` grain via its explicit
    `physical_embryo_id` column); flags only real entities; emits `metadata_missing_flag` +
    `missing_metadata_fields`; consumed by snip_qc as `missing_metadata`. L3 reads the explicit
    `physical_embryo_id` column — never parses `snip_id` to rediscover the animal.
11. **`series_number_map.csv` side-write is dead → removed** (only the writer references it; real consumers
    read the sheet from the workbook).
12. **No config allow/deny list in MVP** — all valid pages pass through; the contract enforces only the
    required set; the config knob is post-MVP.

## 🪧 OPEN (carried, not decided here)
- **Long-table ingest build** — the `ingest_long_well_table` implementation + its classification rules
  (when is a page "long" vs "rejected"?). Spec'd, deferred.
- **Config knob** for page allow/deny (and possibly per-experiment metadata defaults, e.g. a config-sourced
  `temperature`) — post-MVP.
- **`entity_metadata_completeness_qc` build** — its `PIPELINE_STEPS` row, contract, and exact
  inheritance projection (well → physical_embryo → snip), keyed off `physical_embryo_registry` as the
  existence signal. Spec'd here; built in the QC world later.
- **`physical_embryo_registry` build** — the registry product itself (its contract/validator/builder,
  the `validate_physical_embryo_id` addition to `shared/identifiers/`, and the relocation of the mint
  chain out of `run_snip_processing.py`). Spec'd in
  `../detect-seg-track/targets/physical_embryo_registry_world.md`; built later.
