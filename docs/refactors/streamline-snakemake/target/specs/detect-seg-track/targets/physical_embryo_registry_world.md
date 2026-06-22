# Physical Embryo Registry World

**Status:** target-planning draft, mdcolon 2026-06-22.

**Companion to:** `snip_world.md` (the consumer — crops, `snip_id`), `segmentation_world.md`
(the producer of `frame_masks`, the registry's input), `../../front_end/plate_metadata_ingest_and_entity_qc.md`
(L3 entity-completeness QC, which treats this registry as the authoritative discovered-entity set),
and `../../pipeline_file_philosophy.md` (the product/contract/validator conventions this follows).

---

## 🎯 Role

The physical embryo registry answers one question, and owns it exclusively:

> Given validated frame masks (with track identity), **which physical embryos exist**, what are
> their stable IDs, and where did each one come from?

This is the moment a tracked `track_id` becomes a named biological entity — the
**identity-origination boundary**. It is the *only* place in the pipeline where
`track_id → physical_embryo_id` resolution happens.

```
Segmentation finds masks.
Tracking links masks.
Entity registration names organisms.   ← this world
Snip processing crops organisms.
```

Naming (stages name operations; artifacts name tables):

```
physical_embryo_registry   = stage / action AND artifact / table (same noun — name the fundamental unit, no abbreviation)
physical_embryo_id         = the entity key the registry mints
```

---

## 🪨 WHY THIS WORLD EXISTS

Today (`snip_processing/entrypoints/run_snip_processing.py:114–116`) `physical_embryo_id` is minted
*inside the per-mask crop loop* — interleaved with image reads, RLE decode, rotation, augmentation.
Identity is born as a **byproduct of cropping**, and re-derived redundantly on every frame the animal
appears in. The entity (one row per animal) is smeared across the fact rows (one row per mask).

That is the conflation this world removes. **snip_processing may be where embryo identity becomes
*observable*; it must not be the *semantic owner* of embryo identity.** Registration is cheap — the
mint chain is three lines — but cheapness is not a reason to bury a contract boundary.

> **Doctrine (from `pipeline_file_philosophy.md`):** Stages reflect worlds. Products reflect promises.
> Steps may be cheap if the promise is important. This world is a cheap step whose promise — *these
> animals exist, with these stable IDs* — is load-bearing for every downstream join.

This is **not** quality control. QC asks "is this entity good?" The registry asks "does this entity
exist, and is it now official?" Those are different questions, and the registry must not import a QC
judgment.

---

## 🌊 Target river

```
frame_masks[well]              validated, per_well shard (input — the DETECTED/tracked set)
  → physical_embryo_registry[well]   per_well shard: one row per physical_embryo_id in the well
  → physical_embryo_registry         merged experiment table (the authoritative entity set)
```

Downstream:

```
physical_embryo_registry  →  snip_processing  (joins physical_embryo_id on (well_id, track_id))
physical_embryo_registry  →  entity_metadata_completeness_qc (L3: the discovered-entity set)
```

**Input is `frame_masks`, NOT `valid_masks`.** The registry lives *upstream* of the valid/invalid QC
split, so it never inherits a QC filter. Its job is "which embryos were **discovered**," and discovery
is a tracking fact, not a quality verdict. (This is the concrete reason a count column, if ever added,
must be `n_detected_masks` — never `n_valid_masks`; `valid` is a QC word.)

---

## 🧬 Identity model — the registry is the origination boundary

The full ID stack (all constructors/parsers in `src/data_pipeline/shared/identifiers/`):

```
physical_embryo_id  = {well_id}_e{local_embryo_index:02d}   the animal           ← MINTED HERE
embryo_id           = {physical_embryo_id}_{channel_id}     the animal in a channel
snip_id             = {embryo_id}_t{time_index:04d}         the crop/product
```

The registry mints **only `physical_embryo_id`** — the animal. The mint chain (named functions, no
inline arithmetic), relocated out of the snip crop loop:

```python
raw_track_index    = parse_embryo_local_track_id(track_id)        # "embryo_0" → 0
local_embryo_index = track_index_to_embryo_index(raw_track_index) # 0 → 1 (one-based)
physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
```

This chain runs **once per distinct `(well_id, track_id)`** — not once per mask. `embryo_id` and
`snip_id` are *not* minted here: they are per-frame, channel/time projections that belong to
snip_processing (see "The seam" below). The registry has no `image_id`, no channel, no time — it is
strictly per-animal.

> **Locked doctrine:**
> physical_embryo_id names the animal. snip_id names the crop.
> snip_inventory carries both. The contract proves they agree.

`local_embryo_index` is **well-scoped** (per-well, one-based), so per-well minting is independently
correct with no cross-well coordination — the merge is a pure concat. Because `well_id` is globally
unique and `local_embryo_index` is unique within a well, `physical_embryo_id` is globally unique **by
construction**. The merge step *enforces* that invariant rather than trusting it (see merge policy).

---

## 📋 physical_embryo_registry.csv — required columns (MVP)

Grain: **one row per `physical_embryo_id`** (one row per animal).

```
# Identity (the promise this artifact makes)
physical_embryo_id    primary key — the animal; build_physical_embryo_id(well_id, local_embryo_index)
experiment_id
well_id
local_embryo_index    one-based integer ≥ 1 (well-scoped)

# Provenance (where this animal came from — provenance, NOT judgment)
track_id              the tracking identity this entity was resolved from
track_id_source       e.g. "frame_masks" (MVP) — how track identity was established
```

**MVP carries identity + minimal provenance only.** No counts, no time-spans, no first/last image —
those are *auditability* niceties (and a count, if added, must be `n_detected_masks`, never
`n_valid_masks`). They can be added later without changing the contract's *meaning*, because the
promise — "these animals exist, with these stable IDs" — is unchanged. The bright line for any future
column: it must describe **where the animal came from** (provenance — allowed) and never **whether to
trust the animal** (judgment — that is the QC table's job).

---

## 📐 Contract + validator — two files, copying the `frame_masks` shape

Following `pipeline_file_philosophy.md` and the existing `frame_masks_contract.py` /
`validate_frame_masks.py` split: **the contract names the columns; the validator decides if a frame is
legal.** Two files, not one.

### `physical_embryo_registry_contract.py` (schema constants only — no logic)

```python
PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "physical_embryo_id",
    "experiment_id",
    "well_id",
    "local_embryo_index",
    "track_id",
    "track_id_source",
)
PHYSICAL_EMBRYO_REGISTRY_UNIQUE_KEY: tuple[str, ...] = ("physical_embryo_id",)

def empty_physical_embryo_registry() -> pd.DataFrame: ...
```

### `validate_physical_embryo_registry.py` (the dataframe-level validator)

`validate_physical_embryo_registry(df)` — structural + identity checks:

- all required columns present;
- `physical_embryo_id` present, non-null, **unique** (grain check = one row per animal);
- **`physical_embryo_id` globally unique across the merged table** (the by-construction invariant,
  *enforced* at merge — the difference between the registry being a promise and being a hope);
- `local_embryo_index` integer ≥ 1;
- `physical_embryo_id == build_physical_embryo_id(well_id, local_embryo_index)` (round-trip);
- `track_id` non-null; maps to **exactly one** `physical_embryo_id` within a `well_id` (no track
  resolving to two animals);
- no duplicate `(well_id, local_embryo_index)`; no duplicate `(well_id, track_id)`;
- `track_id_source` non-empty.

The dataframe validator **calls the string-level identity validator row-wise** (next section) — the
same way `validate_frame_masks` calls `parse_track_id` per row.

### The reusable string validator lives in `shared/identifiers/`, NOT here

There are two validators, at two grains, and they live in two places — per the hard constraint
*"Identity comes from `shared/identifiers/`; never reinvented in a product."*

| Validator | Grain | Home | Reused by |
|---|---|---|---|
| `validate_physical_embryo_id(physical_embryo_id, *, well_id=None)` | one **string** | **`shared/identifiers/validators.py`** (next to `validate_well_id`) | registry, snips, features, QC — everyone |
| `validate_physical_embryo_registry(df)` | the **table** | this world | the registry product only |

**Today's gap:** `build_physical_embryo_id` and `parse_physical_embryo_id` exist in
`shared/identifiers/`, but there is **no `validate_physical_embryo_id`**. Add it (small, reusable, next
to `validate_well_id`) and have the registry's dataframe validator call it row-wise. This is the piece
the whole pipeline reuses; the registry's table validator merely leans on it.

---

## 🔗 The seam — what snip_processing still does (and what it no longer does)

snip_processing **stops minting `physical_embryo_id`**. Instead it **joins** it in from the registry on
`(well_id, track_id)`:

```
frame_masks[well] ⋈ physical_embryo_registry[well]  on (well_id, track_id)  →  physical_embryo_id per frame
```

That join *is* the channel-free "this animal is present in this frame" fact (occurrence). In MVP it
stays a **join, not a separate artifact** — it is losslessly derivable, and nothing yet consumes
occurrence independent of crops. (Promotion trigger is spec'd under "Deferred" below.)

Then snip_processing — *only because it is the thing making crops* — calls
`build_embryo_id(physical_embryo_id, image_id)` and `build_snip_id(embryo_id, image_id)` to **name the
crop it is extracting**. This is **crop/product naming, not identity origination**: those constructors
project an *already-registered* animal onto a frame/channel/time; they perform no `track_id → animal`
resolution. So the doctrine holds:

> snip_processing never originates *who an animal is* — it receives `physical_embryo_id` and only names
> the crops it produces.

`snip_inventory` then **carries `physical_embryo_id` explicitly** and **validates that `snip_id`,
`embryo_id`, and `physical_embryo_id` agree** (see `snip_world.md`). Downstream feature/QC products
consume the explicit `physical_embryo_id` column — they **must not parse `snip_id` to rediscover the
animal**. This is not a one-product courtesy; it is a pipeline-wide law (next section).

### The channel-free occurrence is the bridge (derived, not minted)

The `frame_masks ⋈ registry` join *is* the channel-free per-frame occurrence fact — the real
conceptual grain `physical_embryo_id × image_id` ("this animal was observed in this frame"). It is the
**bridge between animal identity and crop identity**: snip_inventory construction uses it to root every
crop row in *both* a registered `physical_embryo_id` and a real frame/mask occurrence. In MVP it is
**derived during snip_inventory construction, not minted as a new `*_id` or materialized as a table**
(promotion trigger under "Deferred"). You don't need a "physical snip id"; you need the pipeline to
*know* each snip came from a real animal observed in a real frame — and the spine contract is what
makes it know.

---

## 🔒 Identity-Carrying Contract — the snip/embryo-grain identity spine

> **The registry minting `physical_embryo_id` once does not, by itself, prevent identity from being
> re-buried downstream.** What prevents it is this law: the parent identity must **travel with every
> derived row** and be **validated against every derived ID at every grain**. The registry world owns
> this law (it owns the meaning of physical embryo identity); every snip/embryo-grain contract
> *enforces* it by calling the shared validator.

**Tiny doctrine:**
```
The registry mints the animal.
The frame mask observes the animal.
The snip row crops the animal.
Every derived row carries the animal's name.
The contract proves the names agree.
```

### The law

> **LOCKED doctrine (mdcolon 2026-06-22) — authored here, cited elsewhere, never restated:**
> 1. Every derived row carries its own identity level **plus all parent identities**.
> 2. `physical_embryo_id` is **required parent identity, not optional provenance**.
> 3. Syntactic validity of `embryo_id` or `snip_id` is **insufficient** if it disagrees with
>    `physical_embryo_id`.
> 4. Downstream tables **do not** rediscover `physical_embryo_id` by parsing IDs ad hoc.
> 5. They consume the **explicit column** and call the **shared spine validator**.

`physical_embryo_id` is **not optional provenance**. It is the **parent identity** that every
channel/time/crop identity must validate against. Concretely:

- **Every artifact at snip or embryo grain must carry its identity spine explicitly** — its own ID
  level **plus all parent levels**, always including `experiment_id` and `well_id`. The spine is
  **grain-aware**:

  | Table grain | Required identity spine (own level + all parents) |
  |---|---|
  | **snip** (one row per `snip_id`) | `experiment_id`, `well_id`, `physical_embryo_id`, `embryo_id`, `snip_id` (+ frame-derived: `image_id`, `time_index`, `channel_id`) |
  | **embryo** (one row per `physical_embryo_id`, aggregated over time/channel) | `experiment_id`, `well_id`, `physical_embryo_id` (**no** `snip_id`/`embryo_id` — those are finer grain) |

- **Downstream products consume `physical_embryo_id` as a column** — they must **never** parse `snip_id`
  or `embryo_id` ad hoc to rediscover the animal. Parsing-to-rediscover is the re-burial failure mode
  this law forbids.

- **A row is invalid even if its `snip_id` is syntactically valid**, if `snip_id`/`embryo_id` disagree
  with `physical_embryo_id`, or if `physical_embryo_id` is missing. *A syntactically valid `snip_id` is
  not enough.* ⚡

### Required invariant chain (against the real `shared/identifiers` API)

> **LOCKED constructor note (do not "fix"):** `build_embryo_id` and `build_snip_id` **currently take
> `image_id`** — `build_embryo_id(physical_embryo_id, image_id)`, `build_snip_id(embryo_id, image_id)`.
> They parse the `image_id` internally to extract/cross-check `channel_id` and `time_index`. **Do NOT
> rewrite these examples as `(..., channel_id)` / `(..., time_index)`** unless `shared/identifiers`
> itself changes — the `image_id` signature is the verified API (`constructors.py`), and the
> `channel_id`/`time_index` rewrite is a recurring burr that has been proposed and rejected.

The minting constructors already **fail loud on mismatch** at build time (`build_embryo_id` /
`build_snip_id` parse the `image_id` and assert well/channel agreement). The spine contract re-checks
the same invariants at the **table** level — defense in depth over a mint-time guarantee, persisted as
queryable columns:

```python
# minting (note: both take image_id — verified against constructors.py, NOT channel_id/time_index)
physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
embryo_id          = build_embryo_id(physical_embryo_id, image_id)   # parses image_id; asserts well match
snip_id            = build_snip_id(embryo_id, image_id)              # parses image_id; asserts well+channel

# therefore, at the table level (parsers return tuples):
parse_snip_id(snip_id)          == (embryo_id, time_index)
parse_embryo_id(embryo_id)      == (physical_embryo_id, channel_id)
parse_physical_embryo_id(physical_embryo_id) == (well_id, local_embryo_index)
# and (consume boundary): physical_embryo_id exists in physical_embryo_registry
```

### The reusable spine validator — owned here, called everywhere

Lives in the registry world because this world owns the biological parent→child identity relationship
(distinct from `shared/identifiers`, which owns the *string grammar*). Three tiers, no overlap:

```
shared/identifiers/validators.py    →  validate_physical_embryo_id(...)   # ONE string is well-formed
physical_embryo_registry/snip_identity_contract.py
                                    →  validate_snip_grain_identity_columns(...)  # IDs in a table AGREE
<product>/contract.py               →  call the spine first, then product-specific columns
```

```python
def validate_snip_grain_identity_columns(
    df: pd.DataFrame,
    *,
    grain: str = "snip",                              # "snip" | "embryo" — selects the required spine
    physical_embryo_registry_df: pd.DataFrame | None = None,
    check_sources: bool = False,                      # consume-boundary: assert membership in the registry
    scope_label: str = "snip_grain_table",
) -> None: ...
```

Checks (build time — declared facts):
- required identity columns for the declared `grain` exist and are **non-null**;
- `physical_embryo_id` is a valid identifier (delegates to `validate_physical_embryo_id`);
- `physical_embryo_id == build_physical_embryo_id(well_id, local_embryo_index)`; `experiment_id` /
  `well_id` agree with the parsed `physical_embryo_id`;
- at snip grain: `parse_embryo_id(embryo_id).physical_embryo_id == physical_embryo_id`;
  `parse_snip_id(snip_id).embryo_id == embryo_id`; and where frame columns are present,
  `channel_id`/`time_index` agree with `parse_image_id(image_id)`;
- `snip_id` (or `physical_embryo_id` at embryo grain) unique — **only when the caller declares
  one-row-per-grain** (grain is the caller's assertion, never guessed).

Checks (`check_sources=True` — consume boundary): every `physical_embryo_id` exists in
`physical_embryo_registry_df`. Same lifecycle-flag doctrine as
`validate_yx1_acquisition_inventory(check_sources=True)` — one validator, two moments, the louder
cross-table check at the moment it matters.

Every downstream product then gets a one-liner, then adds its own columns:

```python
def validate_mask_geometry_features(df, *, registry_df=None):
    validate_snip_grain_identity_columns(df, grain="snip", physical_embryo_registry_df=registry_df,
                                         scope_label="mask_geometry_features")
    # ... then require area_um2, centroid_x_um, ...
```

### One law, many citations (anti-duplication)

The spine law text lives **here**. `snip_world.md`, `feature_world.md`, and the QC docs **link to this
section and state only their per-product additions** — they must **not** restate the spine, or it
drifts (the exact failure the identity kingdom exists to prevent). One law, many citations — the
doc-level mirror of "one validator, many callers."

---

## 🌳 Output tree

```
object_extraction/
  <experiment_id>/
    physical_embryo_registry/
      per_well/
        <well_id>/
          <well_id>_physical_embryo_registry.csv
          <well_id>_physical_embryo_registry.csv.validated
      <experiment_id>_physical_embryo_registry.csv
      <experiment_id>_physical_embryo_registry.csv.validated
```

Per-product convention from `output_tree_doctrine.md`: product folder = `physical_embryo_registry/`,
table name = `physical_embryo_registry`. Sentinels (`.validated`) are *derived* from the artifact path
via `validated_path`, never hardcoded.

---

## 🧩 Home + orchestration

- **Package home:** `src/data_pipeline/segmentation_and_tracking/physical_embryo_registry/` — the stage
  that already owns `frame_masks` (the registry's input) and tracking outputs. Registration is "name
  the tracked entities," so it lives where tracking lives.
- **Fanout:** `PER_WELL_THEN_MERGE` — matches `frame_masks`; per-well shards minted independently,
  merged to the experiment table; merge enforces global `physical_embryo_id` uniqueness.
- **`PIPELINE_STEPS` row** (deferred build): step key `physical_embryo_registry` (noun — reads right
  from `build_*` / `validate_*` / `merge_*` rules), `stage="segmentation_and_tracking"` (or its
  output-tree folder), `fanout=PER_WELL_THEN_MERGE`,
  `artifacts={"csv": {<per_well>: "{well_id}_physical_embryo_registry.csv",
  <merged>: "{experiment_id}_physical_embryo_registry.csv"}}`.

---

## 🆚 Not a `tracks` stage

`snip_world.md` anticipates a deferred discrete `tracks` stage (gap/swap/coherence QC). That is a
**different grain and a different concern**: `tracks` would be about *track coherence/quality*; this
registry is about *entity declaration/identity*. They can coexist later (`frame_masks → tracks →
physical_embryo_registry`), but the registry is the identity-ownership move and is the one needed now.
Promoting `tracks` does not remove the need for the registry; it would feed it.

---

## 🪧 Deferred (spec'd, not built now)

- **Channel-free occurrence artifact.** The occurrence bridge (`frame_masks ⋈ registry`, see above)
  stays a *derived join* in MVP. Promote it to a first-class `physical_embryo_occurrence` table
  (per `physical_embryo_id × image_id`, **channel-free**) **only when** a consumer needs occurrence
  *independent of any crop* — e.g. track-coherence QC, or a multi-channel world where occurrence is
  shared across channels but crops differ per channel. That is the explicit promotion trigger; until
  then, materializing it would be a redundant copy of a join. **Do not** mint a `physical_snip_id` /
  `embryo_occurrence_id` for MVP — the missing guarantee was the identity-spine *contract* (above),
  not another `*_id`.
- **Auditability columns** (`n_detected_masks`, `first_time_index`, `last_time_index`, …) — provenance,
  not judgment; add when a consumer needs them. Never `n_valid_masks`.

---

## 🛠️ Files to create (implementation, not this session)

| File | Role |
|---|---|
| `src/data_pipeline/segmentation_and_tracking/physical_embryo_registry/physical_embryo_registry_contract.py` | Schema constants (`*_REQUIRED_COLUMNS`, `*_UNIQUE_KEY`, `empty_*`) |
| `src/data_pipeline/segmentation_and_tracking/physical_embryo_registry/validate_physical_embryo_registry.py` | Dataframe validator (`validate_physical_embryo_registry`) |
| `src/data_pipeline/segmentation_and_tracking/physical_embryo_registry/build_physical_embryo_registry.py` | Builder — distinct `(well_id, track_id)` from `frame_masks` → mint chain → one row per `physical_embryo_id` |
| `src/data_pipeline/shared/identifiers/validators.py` | ADD `validate_physical_embryo_id` (next to `validate_well_id`) — the reusable **string** validator |
| `src/data_pipeline/segmentation_and_tracking/physical_embryo_registry/snip_identity_contract.py` | `validate_snip_grain_identity_columns(...)` — the reusable **table-level** identity-spine validator (grain-aware, `check_sources` flag); called by every snip/embryo-grain product contract |
| `tests/data_pipeline/segmentation_and_tracking/physical_embryo_registry/test_physical_embryo_registry.py` | Contract + validator + builder tests (parallel test tree) |
| Registry entry in `orchestration/paths.py` | `physical_embryo_registry` step, `PER_WELL_THEN_MERGE` |
| `Snakefile` rules | `build_physical_embryo_registry[well]`, `validate_physical_embryo_registry[well]`, `merge_physical_embryo_registry` |
| Edit `snip_processing/entrypoints/run_snip_processing.py` | DELETE the mint chain (`:114–116`); JOIN `physical_embryo_id` from the registry instead |
