# Feature World - computed feature targets

**Status:** planning spec, 2026-06-22. This doc lists the computed features and feature-derived QC
targets we can start, their dependencies, and the acceptance bar for each.

**Doctrine:** features quantify validated objects. QC consumes feature tables and emits flags or
QC-specific annotations over the same `snip_id` universe. Neither layer mints object identity,
chooses segmentation prompts, or performs analysis-ready joins.

---

## Shared Feature And QC Stage Pattern

Every computed feature and feature-derived QC stage should use the same WellRunner-compatible
per-well stage recipe:

- one noun-like registry row in the pipeline-wide path registry;
- one compute function with explicit inputs and no hidden globals;
- one thin `tasks.py` verb or entrypoint that parses arguments and delegates;
- one templated per-well rule;
- one focused contract/validator for the feature or QC table it owns;
- tests in the parallel `tests/data_pipeline/...` tree.

Feature and QC paths must come from the orchestration registry, not domain-local path registries or
raw strings in rules. Code may import identity constructors/parsers when it needs IDs, but it must
not mint or split IDs inline. The per-well shard paths must be compatible with `well_runner`: DAG-time
helpers declare expected per-well outputs, runtime collectors inspect finished shards.

Each file written for feature work must pass the `pipeline_file_philosophy.md` first-read check:
clear names, explicit signatures, flow-order organization, and fail-loud errors that name the fix.

---

## Code Organization Pattern

Do not add a `stages/` folder under `feature_extraction` or `quality_control`. Those package names
already name the pipeline world; the child folder should name the product directly.

Computed feature products live here:

```text
src/data_pipeline/feature_extraction/
  mask_geometry/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  curvature_metrics/
    contract.py
    compute.py
    skeletonization.py
    entrypoint.py
    __init__.py
  fraction_alive/
    contract.py
    compute.py
    via_masks.py
    entrypoint.py
    __init__.py
  stage_predictions/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  consolidated_features/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  io/
    loaders.py
    writers.py
  shared/
    feature_table_utils.py
```

Feature-derived QC products mirror that shape:

```text
src/data_pipeline/quality_control/
  death_detection/
    contract.py
    compute.py
    persistence.py
    alignment.py
    stage_at_death.py
    entrypoint.py
    __init__.py
  surface_area_qc/
    contract.py
    reference_contract.py
    reference.py
    compute.py
    entrypoint.py
    __init__.py
    references/
      surface_area_reference_v1.csv
      README.md
  mask_quality_qc/
    contract.py
    compute.py
    entrypoint.py
    __init__.py
  snip_qc/
    contract.py
    inputs.py
    build.py
    entrypoint.py
    __init__.py
  io/
    loaders.py
    writers.py
  shared/
    qc_table_utils.py
  # STUBS — in development, not built this pass (z-stack ingest):
  #   focus_qc/   -> focus_flag
  #   blur_qc/    -> blur_flag
```

The four MVP QC product folders above are `death_detection/`, `surface_area_qc/`, `mask_quality_qc/`,
and `snip_qc/`. `focus_qc/` and `blur_qc/` are **stubs** (deferred; see their sections). `motion_qc/`
and `viability_qc/` are **not** target products — they are dropped (see Legacy Domain Retirement) and
must not appear under `quality_control/`.

Per product folder:

- `contract.py` defines the table truth: grain, required columns, nullable columns, and validator;
- `compute.py` owns pure feature/QC logic and returns dataframes or values;
- `build.py` is allowed when the product assembles a verdict rather than measuring a feature;
- `inputs.py` is allowed when a product needs small, explicit in-memory assembly from registered upstream artifacts;
- `config.py` owns product-local defaults when the product has thresholds, model choices, or policies;
- `entrypoint.py` is the thin CLI/task adapter that loads inputs, resolves config, calls compute/build, validates, writes;
- extra `*.py` files are product-specific helpers only.

Domain-level `io/` and `shared/` are for boring mechanics reused by multiple products. `shared/`
is earned: keep helpers product-local until a second product needs them. Existing legacy entrypoints
may remain as temporary wrappers that import the product-local `entrypoint.main`.

Doctrine: domain package names the world; product folder names the table; `contract.py` defines it;
`compute.py` makes it; `entrypoint.py` touches the filesystem. No `stages/` inside stages.

Code package and output stage intentionally differ: source code lives under `feature_extraction/`,
while feature artifacts land under the `features/` output stage. Do not add a new
`data_pipeline.features` package for future feature work.

---

## Legacy Domain Retirement

The old flat layout under `quality_control/` must be retired as each per-product subfolder is
implemented. Retirement is part of the definition of done for each product — it is not a
separate cleanup pass.

**Modules to retire (delete or convert to thin import shims, then delete):**

| Old module | Replaced by |
|---|---|
| `quality_control/core/death_detection.py` | `quality_control/death_detection/compute.py` + `persistence.py` + `alignment.py` |
| `quality_control/core/surface_area_outlier_detection.py` | `quality_control/surface_area_qc/compute.py` + `reference.py` |
| `quality_control/core/motion_qc.py` | **dropped** — motion QC is not used; delete the module, no folder |
| `quality_control/core/focus_qc.py` | **STUB** — a 3-line `focus_flag = False` placeholder, never implemented. Real focus QC is in development (z-stack ingest); do not migrate this stub. Delete it; the future `focus_qc/` lands when that work finishes. |
| `quality_control/morphology_qc/size_validation_qc.py` | **deleted** — duplicate of surface-area logic with a broken absolute import and stale 1.2/0.9 defaults; no home in the target layout |
| `quality_control/core/viability_qc.py` | **dropped for MVP** — delete the module; do not create a folder. `viability_qc` is mask-plausibility QC over `mask_geometry` (area/aspect-ratio); it has **zero** algorithmic overlap with death detection (which consumes `fraction_alive`) and is explicitly **not** folded into it. Revisit as its own QC product post-MVP if the flag proves needed downstream. |
| `quality_control/core/consolidate_qc.py` | `quality_control/snip_qc/build.py` |
| `quality_control/core/_shared.py` | helpers inline into product or into `quality_control/shared/` if genuinely reused |
| `quality_control/core/auxiliary_mask_qc.py` | `quality_control/` product folder TBD (out of MVP scope) |
| `quality_control/core/segmentation_quality_qc.py` and `quality_control/segmentation_qc/segmentation_quality_qc.py` | `quality_control/mask_quality_qc/compute.py` — migrate the edge/discontinuous/overlap flag logic, but **re-point input from raw SAM2 `mask_rle` to canonical `frame_masks`** via the shared decoder |
| `quality_control/entrypoints/compute_segmentation_qc.py` | `quality_control/mask_quality_qc/entrypoint.py` |
| `quality_control/consolidation/consolidate_qc.py` | deleted; `snip_qc/build.py` owns the verdict |
| `quality_control/entrypoints/compute_death_detection.py` | `quality_control/death_detection/entrypoint.py` |
| `quality_control/entrypoints/compute_surface_area_qc.py` | `quality_control/surface_area_qc/entrypoint.py` |
| `quality_control/entrypoints/compute_motion_qc.py` | **deleted** — motion QC dropped |
| `quality_control/entrypoints/compute_focus_qc.py` | `quality_control/focus_qc/entrypoint.py` |
| `quality_control/entrypoints/consolidate_qc.py` | `quality_control/snip_qc/entrypoint.py` |
| `quality_control/io/paths.py` | deleted; paths come from orchestration registry |
| `quality_control/io/loaders.py` | loaders move product-local or into `quality_control/io/loaders.py` scoped to shared mechanics only |
| `schemas/quality_control.py` (`SNIP_EXCLUSION_FLAGS`, `REQUIRED_COLUMNS_QC`) | each product owns its contract; `snip_qc/contract.py` owns `SNIP_QC_EXCLUSION_REASONS`; delete the domain-level schema module once per-product contracts cover all callers |

The `quality_control/core/` and `quality_control/consolidation/` and `quality_control/entrypoints/`
directories must be empty (or deleted) after the MVP products are implemented. Leaving dead modules
alongside new product folders is not acceptable — it creates two competing sources of truth.

Similarly, **`feature_extraction/core/`** and **`feature_extraction/entrypoints/`** are the legacy
layout for features. Each per-product subfolder that is implemented retires its counterpart in those
old directories. The old flat modules at `feature_extraction/` package root (e.g.
`consolidate_features.py`, `mask_geometry_metrics.py`, `stage_inference.py`) must also be deleted
once their product subfolders are live.

---

## Contract Naming Pattern

Inside a single `contract.py`, generic local names are acceptable. Public exports must name the
product so imports stay readable when multiple contracts are in scope.

Examples:

- `MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS` and `validate_mask_geometry_features(df)`
- `FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS` and `validate_fraction_alive_features(df)`
- `DEATH_DETECTION_QC_REQUIRED_COLUMNS` and `validate_death_detection_qc(df)`
- `SURFACE_AREA_QC_REQUIRED_COLUMNS` and `validate_surface_area_qc(df)`
- `MASK_QUALITY_QC_REQUIRED_COLUMNS` and `validate_mask_quality_qc(df)`
- `SNIP_QC_REQUIRED_COLUMNS` and `validate_snip_qc(df)`

Avoid exported generic names like `REQUIRED_COLUMNS` from product contracts.

---

## Stage Table Pattern

**One-row grain:** default grain is one row per `snip_id`. A feature or QC stage may use a different
grain only if its contract names that grain explicitly.

**Feature tables:** features are measured or predicted values.

- required key: `snip_id`;
- feature columns may be continuous, categorical, or numeric predictions;
- feature columns must not be boolean inclusion/exclusion decisions;
- feature columns should not end in `_flag`.

Examples: `area_um2`, `centroid_x_um`, `mean_curvature_per_um`, `speed_um_per_s`,
`fraction_alive`, `predicted_stage_hpf`.

**QC tables:** QC tables are judgments over the feature universe.

- required key: `snip_id`;
- every boolean QC output column must end in `_flag`;
- every `*_flag` column must be non-null boolean dtype;
- annotation columns are allowed, but they are not flags;
- stage-specific QC tables emit one or more `*_flag` columns;
- `snip_qc` computes `use_snip` and pipe-delimited `qc_fail_reasons` from selected exclusion flags.

Examples: `viability_dead_flag`, `persistence_dead_flag`, `sa_outlier_flag`, `edge_flag`,
`discontinuous_mask_flag`, `overlapping_mask_flag`, `focus_flag`.
`death_event_time_index` and `death_event_stage_hpf` are per-`physical_embryo_id` annotations in a
separate `death_event` table, not per-snip columns (they carry a configurable lead-time adjustment).

**Grain + schema validation (universal, every output table):** *every* feature and QC output table —
snip-grain, embryo-grain, or event-grain — runs a grain/identity schema check immediately before it is
written. No exceptions, no "this one is small." This is the systemic guard: a snip-grain checker run on
each output catches a missing spine, a wrong grain, duplicate keys, or null/non-boolean flags at the
write boundary, before a bad table ever reaches a consumer.

The product's own `validate_*` function is the single gate and does **two** things in order:

1. **Spine check** — call the shared identity validator with the table's declared grain. The grain
   token is the **leaf ID of the spine it checks**, spelled with the `_id` suffix so there is no
   ambiguity about which level (`physical_embryo_id` vs `embryo_id` are distinct grains):
   `_VALID_GRAINS = ("physical_embryo_id", "embryo_id", "snip_id")`.
   - snip grain: `validate_snip_grain_identity_columns(df, grain="snip_id", physical_embryo_registry_df=registry_df)`
     → validates `SNIP_ID_SPINE_COLUMNS`;
   - embryo grain: `grain="embryo_id"` → validates `EMBRYO_ID_SPINE_COLUMNS` (adds `embryo_id`);
   - physical-embryo / event grain (e.g. `death_event`): `grain="physical_embryo_id"` → validates
     `PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` (no `embryo_id`).

   > **Minting-site change:** today the validator has only `_VALID_GRAINS = ("snip", "embryo")`, and
   > `"embryo"` maps (ambiguously) to the physical-embryo spine. The rename adds the third level and
   > suffixes every token with `_id`, so a **physical-embryo-grain check is first-class**, not borrowed
   > from the `embryo` token: `("physical_embryo_id", "embryo_id", "snip_id")`.
2. **Schema check** — then validate the **product-specific columns**: required QC columns present,
   every `*_flag` non-null boolean, annotations typed, grain one-row-per-key, no missing/duplicate/extra
   keys versus the declared universe.

Both halves are mandatory. A table that validates its spine but not its QC columns (or vice versa) is
**not done**. The entrypoint calls `validate_*` and only then writes the artifact and its
`validated_path(...)` marker — an unvalidated write is a contract violation.

**Spine columns are imported from their minting site, never re-declared (DRY, one concept one home).**
The spine column *sets* and the validator both live at the identity minting site
(`segmentation/physical_embryo_registry/snip_identity_contract.py`, which owns `SNIP_ID_SPINE_COLUMNS`,
`EMBRYO_ID_SPINE_COLUMNS`, and `validate_snip_grain_identity_columns`). A product contract that wants a
declarative column list **imports** those constants and composes — it does **not** retype the spine
columns inline, because a literal copy drifts from the minting site the moment the spine changes
(per the philosophy's "one concept, built in exactly one place" and "identity comes from its owner;
identity never imports orchestration"). The two spine sets are themselves related by composition, not
copied:

The spine sets follow the parent→child identity chain **additively** — each level is the level above
plus exactly one more ID — so reading the three constants top to bottom *is* reading the identity
hierarchy:

```python
# at the minting site — the additive identity chain (each = previous + one ID)
PHYSICAL_EMBRYO_ID_SPINE_COLUMNS = ("experiment_id", "well_id", "physical_embryo_id")
EMBRYO_ID_SPINE_COLUMNS          = PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + ("embryo_id",)
SNIP_ID_SPINE_COLUMNS            = EMBRYO_ID_SPINE_COLUMNS + ("snip_id",)
```

The chain is `physical_embryo_id → embryo_id → snip_id`: `embryo_id` is the physical embryo **at a
channel** (a child of `physical_embryo_id`), and `snip_id` is the embryo **at a time** (a child of
`embryo_id`). A physical-embryo-grain table (e.g. `death_event` — the *animal* died, channel- and
time-independent) carries `PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` and **must not** carry `embryo_id`
(that would over-specify it to a channel it does not have).

> ⚠️ **Animal-level facts group by `physical_embryo_id`, not `embryo_id`.** Because `embryo_id` is the
> animal *at a channel* (not the animal), grouping an animal-level quantity by `embryo_id` silently
> splits one animal across channels — the exact bug death_detection's persistence grouping caught. Use
> `embryo_id` **only** when the quantity is genuinely channel-specific. This applies everywhere
> downstream (death detection, fraction_alive projection, pose, consolidated features): if the fact is
> about *the animal*, the group key is `physical_embryo_id`.

So a per-snip product declares `list(SNIP_ID_SPINE_COLUMNS + product_columns)` and a physical-embryo-grain
event table declares `list(PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + event_columns)`, both importing the spine
from the contract module — the snip-grain checker that the feature columns already use is the same one,
reused, not re-implemented.

**Required change at the minting site (prerequisite).** Today `snip_identity_contract.py` defines the
spine sets as **private, independent literals** (`_SNIP_SPINE_COLUMNS`, `_EMBRYO_SPINE_COLUMNS`) — they
cannot be imported, they can drift apart, and `_EMBRYO_SPINE_COLUMNS` is **misnamed**: it has no
`embryo_id` (its key is `physical_embryo_id`). Before product contracts import them, the contract
module must:

1. **Publish** the constants (drop the leading underscore) so they are the one public, importable
   source of spine membership;
2. **Rename** the current `_EMBRYO_SPINE_COLUMNS` (which is keyed on `physical_embryo_id`, no
   `embryo_id`) to **`PHYSICAL_EMBRYO_ID_SPINE_COLUMNS`** so the name matches the grain;
3. **Compose additively, don't duplicate**: define the full three-level chain above
   (`EMBRYO_ID_SPINE_COLUMNS = PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + ("embryo_id",)`,
   `SNIP_ID_SPINE_COLUMNS = EMBRYO_ID_SPINE_COLUMNS + ("snip_id",)`) so each spine is literally its parent
   plus one ID — one edit site, no drift. Define all three even if only physical-embryo and snip grains
   have tables today; a partial chain (skipping a level) is not declarative.
4. **Collapse the existing duplicate spine literals into this one source.** The snip spine is currently
   re-declared in at least **three** places — `snip_identity_contract._SNIP_SPINE_COLUMNS`,
   `feature_extraction/shared/feature_table_utils.SNIP_SPINE_COLUMNS`, and a local `_SPINE_COLUMNS` in
   `mask_geometry/contract.py`. That is the exact "one concept, three homes, will drift" smell the
   philosophy bans. All of them must import `SNIP_ID_SPINE_COLUMNS` from the minting site; the
   duplicate literals are deleted, not kept "for convenience."

This keeps identity owned where it is minted and lets every feature/QC contract — and the existing
snip-grain checker used by the feature columns — share exactly one definition.

> Doctrine in one line: every output gets a grain/identity + column schema check before write; the
> spine is imported from its minting site (never re-typed), and the per-snip grain checker is exactly
> what would have caught earlier spine/column gaps, so it runs on every table, every time.

**WellRunner compatibility:** every feature and QC stage should produce per-well shards first, then
merge. The stage contract must identify whether the shard is per-well or merged, and the registry row
must be the single source of path truth for both Snakemake and Python entrypoints.

---

## Feature Universe

The feature universe is the one-row-per-`snip_id` table that feature and QC products align to.
The canonical source is validated `snip_inventory`; feature and QC products consume this universe, they do not
create it.

**The identity spine** (the *required* snip-grain identity columns, = `SNIP_ID_SPINE_COLUMNS` at the
minting site):

- `experiment_id`
- `well_id`
- `physical_embryo_id`
- `embryo_id`
- `snip_id`

**Frame-derived columns** carried by the universe and cross-checked **when present**, but **not**
required spine (an embryo-grain / time-aggregated table legitimately lacks them):

- `image_id`
- `time_index`
- `channel_id`

The spine is what `validate_snip_grain_identity_columns(grain="snip_id")` *requires*; frame-derived
columns are validated only when a table carries them (matching the minting-site comment in
`snip_identity_contract.py`).

> **Spine ≠ everything a product needs.** Frame-derived columns are not identity spine, but many
> products **require them operationally** — `time_index` for any temporal logic (death persistence,
> pose kinematics), `image_id` for any per-frame grouping (mask-quality overlap). So a product contract
> composes its required columns as:
> ```python
> PRODUCT_REQUIRED_COLUMNS = list(SNIP_ID_SPINE_COLUMNS + required_frame_columns + product_output_columns)
> ```
> i.e. **spine (who) + required frame columns (where/when) + product columns (what)**. death_detection
> already does this (`SNIP_ID_SPINE_COLUMNS + time_index + fraction_alive`); mask_quality_qc needs
> `image_id`, pose_kinematics needs `time_index` + frame timing — name these explicitly in each
> contract, don't smuggle them in as "the spine."

These required spine columns are **not optional provenance** — they are
governed by the pipeline-wide Identity-Carrying Contract owned by
`../../detect-seg-track/targets/physical_embryo_registry_world.md` (§ "Identity-Carrying Contract").
This doc **does not restate that law** (one law, many citations). Every feature/QC product contract
**calls the shared spine validator first**, then validates its own columns:

```python
validate_snip_grain_identity_columns(df, grain="snip_id", physical_embryo_registry_df=registry_df)
# ... then product-specific feature/QC columns
```

The spine guarantees `physical_embryo_id` is carried explicitly and that
`physical_embryo_id`↔`embryo_id`↔`snip_id` agree. **No feature or QC product may parse `snip_id` or
`embryo_id` to rediscover the animal** — it consumes the explicit `physical_embryo_id` column. A
product at a non-`snip_id` grain (e.g. one row per `physical_embryo_id`) calls the validator with the
matching grain token — `grain="physical_embryo_id"`, validating `PHYSICAL_EMBRYO_ID_SPINE_COLUMNS`
(see the minting-site grain-token change in Stage Table Pattern: tokens are
`("physical_embryo_id", "embryo_id", "snip_id")`).

Most feature and QC products either compute directly at `snip_id` grain or compute at an upstream
object/mask grain and then explicitly project to `snip_id` grain through validated `snip_inventory`. No
feature or QC product may silently change the universe. Missing, duplicate, or extra `snip_id` rows
must be documented by the product contract and fail loud unless the product explicitly defines a
different behavior.

Tiny doctrine: validated snip inventory defines the universe; feature tables measure it; QC
tables judge it; `snip_qc` summarizes the verdict; `analysis_ready` applies the verdict.

---

## Product Overview

| Product | Domain | Grain | Primary input | Output product |
|---|---|---|---|---|
| `mask_geometry` | `feature_extraction` | `snip_id` | validated `snip_inventory` + `frame_masks` | `mask_geometry_features` |
| `consolidated_features` | `feature_extraction` | `snip_id` | validated feature tables | `consolidated_features` |
| `curvature_metrics` | `feature_extraction` | `snip_id` | `mask_geometry` + canonical masks | `curvature_features` |
| `pose_kinematics` | `feature_extraction` | `snip_id`; track operations internal | `mask_geometry` + `frame_inventory` | `pose_kinematics_features` |
| `stage_predictions` | `feature_extraction` | `snip_id` | morphology/timing features | `stage_prediction_features` |
| `fraction_alive` | `feature_extraction` | `snip_id` | embryo masks + auxiliary/VIA masks | `fraction_alive_features` |
| `surface_area_qc` | `quality_control` | `snip_id` | `mask_geometry` + `stage_predictions` + feature universe + packaged reference | `surface_area_qc` |
| `mask_quality_qc` | `quality_control` | `snip_id` (overlap computed per `image_id`) | validated `snip_inventory` + canonical `frame_masks` | `mask_quality_qc` |
| `death_detection` | `quality_control` | `snip_id` flags; `physical_embryo_id` `death_event` | `fraction_alive` + feature universe + frame timing (+ `stage_predictions` for `death_event`) | `death_detection_qc` + `death_event` |
| `focus_qc` | `quality_control` | `snip_id` (from z-stack) | z-stack focus metric + feature universe | `focus_qc` (**STUB** — z-stack, in dev) |
| `blur_qc` | `quality_control` | `snip_id` (from z-stack) | z-stack blur metric + feature universe | `blur_qc` (**STUB** — z-stack, in dev) |
| `metadata_completeness_qc` | `quality_control` | `snip_id` | validated `snip_inventory` + required metadata contracts | `metadata_completeness_qc` (deferred) |
| `snip_qc` | `quality_control` | `snip_id` | feature universe + selected QC flag inputs | `snip_qc` |

Any product not actually `snip_id` grained must say so in its contract and must also name the
projection step that returns to the feature universe.

### QC Product Roster (single source of truth)

This is the authoritative list of `quality_control/` products. Anything not named here as a **target**
or **stub** does not belong under `quality_control/` and should be deleted, not migrated.

**Target QC products (build these):**

| Product | Output flags / columns | Primary input |
|---|---|---|
| `death_detection` | `viability_dead_flag`, `persistence_dead_flag` (per snip); `death_event_time_index`, `death_event_stage_hpf` (per `physical_embryo_id`, in `death_event`) | `fraction_alive` + frame timing (+ `stage_predictions`) |
| `surface_area_qc` | `sa_outlier_flag` | `mask_geometry` + `stage_predictions` + packaged reference |
| `mask_quality_qc` | `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag` (no persisted composite) | canonical `frame_masks` + `snip_inventory` |
| `snip_qc` | `use_snip`, `qc_fail_reasons` | the QC flags above |

**Stub QC products (in development, not built this pass — z-stack ingest):**

| Product | Future flag | Status |
|---|---|---|
| `focus_qc` | `focus_flag` | stub; reserved `snip_qc` hook |
| `blur_qc` | `blur_flag` | stub; reserved `snip_qc` hook |
| `metadata_completeness_qc` | `metadata_missing_flag` | deferred; reserved `snip_qc` hook |

**Dropped (not target products — delete legacy, do not create folders):**

| Product | Why dropped |
|---|---|
| `motion_qc` | not used |
| `viability_qc` | mask-plausibility flag with no MVP consumer; revisit post-MVP if needed |

The legacy modules behind every dropped/migrated item are enumerated in **Legacy Domain Retirement**.

---

## Feature Table Provenance

Feature tables carry minimal method provenance, not biological metadata. Add provenance columns when
multiple methods, formulas, references, configs, or models could produce the same product.

Allowed or required where relevant:

- `feature_backend` or `method_name`
- `feature_version` or `model_version`
- `reference_version` for reference-backed QC or features
- `config_name` or `config_hash` when config materially changes output values

Do not join genotype, condition, perturbation, or `use_snip` into feature products. Those joins belong
in `analysis_ready`, after feature and QC contracts are validated.

---

## Product Path Acceptance

Every product-specific Done When inherits this path acceptance bar:

- per-well output paths come from `artifact_path(...)` with registry step/artifact keys;
- validated sentinels come from `validated_path(...)`;
- output directories come from `step_dir(...)` or `per_well_step_dir(...)`;
- no raw output path strings appear in rules, `tasks.py`, or entrypoints;
- packaged static references, such as `surface_area_qc/references/surface_area_reference_v1.csv`,
  are source assets and are not `paths.py` artifacts.

---

## Config Pattern

Every feature or QC product must declare its config surface before implementation. Any threshold,
method choice, reference version, smoothing window, missing-data policy, or output policy is a config
knob, not a hidden literal inside `compute.py`.

Target convention:

- run-level overrides live under the product key in pipeline config, for example
  `feature_extraction.mask_geometry` or `quality_control.death_detection`;
- product defaults live beside the product code when the product has real knobs, usually in
  `config.py`;
- `entrypoint.py` resolves defaults plus run overrides, then passes the resolved config or explicit
  keyword values into `compute.py`;
- `compute.py` never imports global run config and never reads thresholds or reference paths by
  itself;
- each contract test should include at least one non-default config case when a product has knobs.

A product with no knobs should still say that explicitly in its mini-spec. Do not leave a blank space
where a future agent has to guess.

---

## Output Location Pattern

All feature steps land under the `features/` stage on disk. Each step gets its own `product_dir`
— the named product family within the stage. This is how `PIPELINE_STEPS` in `orchestration/paths.py`
constructs the path:

```python
"latent_embeddings": {
    "stage": "features",          # top-level folder
    "product_dir": "latent_embeddings",  # product family within it
    "fanout": PER_WELL_THEN_MERGE,
    "artifacts": { ... }
}
```

Which gives you, for free via `artifact_path(...)`:

```
features/<exp>/latent_embeddings/per_well/<well_id>/<well_id>_latents.parquet
features/<exp>/mask_geometry/per_well/<well_id>/<well_id>_mask_geometry.parquet
features/<exp>/consolidated/per_well/<well_id>/<well_id>_consolidated_features.parquet
```

The output tree is declarative: stage names the phase; `product_dir` names what the artifact is.
`product_dir` uses the artifact type, not the method — `latent_embeddings` not `legacy_vae`,
`mask_geometry` not `sam2_geometry`. Method provenance belongs in code and config, not paths.

No caller ever types a raw path string. Every path is resolved through `artifact_path()`,
`validated_path()`, or `step_dir()` with the step name and artifact key from the registry.

---

## Priority Order

1. **Stage table pattern and contracts** - lock `snip_id` grain, feature-vs-QC schema rules,
   `_flag` naming, and WellRunner-compatible shard/merge paths.
2. **Mask geometry** - first computed feature; smallest dependency surface after segmentation
   Session A/B and proves canonical mask consumption.
3. **Consolidated features v1** - merge one feature table by `snip_id`; proves the feature universe
   and collision/duplicate policy early.
4. **Stage predictions** - morphology/timing → `predicted_stage_hpf`. Placed **before** surface area
   QC because surface area QC requires it (stage-binned reference). Do not keep a priority list where a
   QC item depends on a later item.
5. **Surface area QC** - first feature-derived QC pattern; depends on `mask_geometry` (area) **and**
   `stage_predictions` (`predicted_stage_hpf`), both now upstream of it.
6. **Curvature metrics** - follows mask geometry after mask decode and centerline behavior are stable.
7. **Pose and kinematics** - follows stable `track_id`, temporal ordering, and frame timing.
8. **Mask quality QC** - follows canonical `frame_masks` + `snip_inventory`; migrates the
   edge/discontinuous/overlap flags out of the SAM2 path onto canonical masks.
9. **Fraction alive** - follows auxiliary/VIA mask product clarity.
10. **Death detection QC** - follows `fraction_alive`, frame timing, and (for `death_event`)
   `stage_predictions`; first viability-derived QC flag table.
11. **Snip QC** - follows the three MVP exclusion QC tables (death_detection, surface_area_qc,
   mask_quality_qc) and builds the final `use_snip` verdict.
12. **Latent embeddings** - follows validated snip inventory; has its own spec
   (`legacy_embeddings.md`) due to the 3.9 env boundary and batch-execution constraints.
   Feeds `analysis_ready` on `snip_id`.

> **DAG rule:** the priority order is a real dependency order, not a wishlist. No item may depend on a
> later item. (This is why stage_predictions moved ahead of surface_area_qc.)

**Deferred / stub (not scheduled this pass):** `focus_qc` and `blur_qc` (z-stack ingest, in
development; future `focus_flag` / `blur_flag` are reserved `snip_qc` hooks); `metadata_completeness_qc`
(may later feed `snip_qc` as `missing_metadata`). **Dropped:** `motion_qc` (not used), `viability_qc`.

---

## Mask Geometry

**Product folder:** `src/data_pipeline/feature_extraction/mask_geometry/`

> Note: the source code package is `feature_extraction/`; the on-disk output stage is `features/`.
> These are intentionally different — the code package name describes what the code does; the output
> stage name describes what the artifact is. All paths come from the registry, so no caller conflates
> the two.

**Key functions:**

- `contract.py::MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_mask_geometry_features(df)`
- `compute.py::compute_mask_geometry_features(snip_inventory_df, frame_masks_df, frame_inventory_df, *, mask_decoder)`
- `compute.py::compute_mask_geometry_for_mask(mask, pixel_size_um)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical validated `snip_inventory`, `frame_masks`, `frame_inventory`, segmentation
mask decode/geometry helpers, and shared `snip_id` identity.

**Config surface:** `feature_extraction.mask_geometry`

- `pixel_size_source`: column/contract source for micron calibration, default `frame_inventory`;
- `placeholder_policy`: how no-mask placeholders are handled, default `exclude`;
- `mask_decoder`: decoder name or injected decoder policy, default canonical mask RLE decoder;
- `output_units`: fixed to micron-aware outputs where calibration exists;
- `min_valid_area_px`: smallest non-empty mask accepted for feature computation.

**Grain:** one row per `snip_id`.

Mask Geometry uses validated `snip_inventory` as the object-to-snip handoff:

```text
snip_id -> mask_id / track_id / image_id
```

It does not scan all `frame_masks` independently and invent a feature universe. Frame masks are
looked up through the snip inventory row.

**What it computes:** area, perimeter, centroid, width/height-style geometry, and calibration-aware
micron-scale measurements for each snip-selected object mask.

**Depends on:**

- validated `snip_inventory` rows with one row per `snip_id` and a documented mask handoff;
- validated `frame_masks` rows with parseable `mask_id` / `track_id`;
- pixel calibration from `frame_inventory`;
- mask RLE/decode and geometry helpers from segmentation Session A;
- no-mask placeholders filtered out or handled explicitly before feature computation.

**Needs before coding:**

- feature table contract: one row per `snip_id`;
- source of pixel size named in the input contract;
- decision that this feature reads canonical masks, not backend-native SAM2 outputs;
- documented join from `snip_inventory` to `frame_masks` by `mask_id`, `track_id`, and/or `image_id`.

**Done when:**

- synthetic masks produce deterministic geometry metrics;
- invalid or placeholder masks fail loud or are excluded by contract;
- per-well output validates and can be merged without experiment-grain assumptions.

---

## Curvature Metrics

**Product folder:** `src/data_pipeline/feature_extraction/curvature_metrics/`

**Key functions:**

- `contract.py::CURVATURE_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_curvature_features(df)`
- `compute.py::compute_curvature_features(mask_geometry_df, frame_masks_df, *, mask_decoder)`
- `compute.py::compute_curvature_for_mask(mask, pixel_size_um)`
- `skeletonization.py::extract_centerline_points(mask)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, canonical masks, pixel calibration, and deterministic
centerline extraction.

**Config surface:** `feature_extraction.curvature_metrics`

- `skeletonization_method`: centerline extraction method;
- `min_centerline_points`: minimum points required before curvature is computed;
- `smoothing_window_points`: optional centerline smoothing window;
- `resample_spacing_um`: optional centerline resampling interval;
- `low_information_policy`: return documented null metrics or fail loud for tiny/empty masks.

**What it computes:** centerline length, centerline point count, and curvature summaries from valid
embryo masks.

**Depends on:**

- mask geometry inputs and calibration;
- stable binary mask decode/read policy;
- centerline/skeletonization behavior that is deterministic on synthetic masks.

**Needs before coding:**

- explicit behavior for masks with too few centerline points;
- clear units: curvature in inverse microns, lengths in microns;
- tests for straight, curved, tiny, and empty masks.

**Done when:**

- low-information masks return documented null metrics instead of silent bad numbers;
- synthetic fixtures pin centerline and curvature behavior;
- output joins cleanly by `snip_id` with mask geometry.

---

## Pose And Kinematics

**Product folder:** `src/data_pipeline/feature_extraction/pose_kinematics/`

**Key functions:**

- `contract.py::POSE_KINEMATICS_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_pose_kinematics_features(df)`
- `compute.py::compute_pose_kinematics_features(mask_geometry_df, frame_inventory_df)`
- `compute.py::compute_pose_features_for_mask(mask, pixel_size_um)`
- `compute.py::compute_kinematics_for_track(track_df)`
- `entrypoint.py::main()`

**Direct dependencies:** mask geometry feature rows, `track_id`, `snip_id`, frame timing from
`frame_inventory`, and deterministic ordering within each track.

**Config surface:** `feature_extraction.pose_kinematics`

- `orientation_method`: how object orientation is computed;
- `time_column`: frame timing column, default `elapsed_time_s`;
- `first_frame_kinematics_policy`: null behavior for first row in each track;
- `max_allowed_time_gap_s`: optional fail-loud guard for track gaps;
- `coordinate_units`: expected position units after calibration.

**What it computes:** orientation, bounding box dimensions, displacement, speed, coordinate deltas,
and elapsed-time deltas for each tracked object.

**Depends on:**

- valid object masks and mask geometry;
- stable `track_id` from the segmentation contract;
- frame time carried through `frame_inventory`;
- deterministic ordering within each `track_id`.

**Needs before coding:**

- segmentation Session B fake-predictor path proving deterministic `track_id` and mask rows;
- clear first-frame behavior for displacement/speed nulls;
- validation that time deltas are positive within each track.

**Done when:**

- synthetic two-frame and three-frame tracks produce expected displacement/speed;
- missing or non-monotonic time fails loud with the track and frame named;
- first observation per track has documented null kinematics.

---

## Consolidated Features

**Product folder:** `src/data_pipeline/feature_extraction/consolidated_features/`

**Key functions:**

- `contract.py::CONSOLIDATED_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_consolidated_features(df)`
- `compute.py::consolidate_feature_tables(feature_tables, *, key="snip_id")`
- `compute.py::assert_feature_table_compatible(df, *, key="snip_id", feature_name)`
- `entrypoint.py::main()`

**Direct dependencies:** one or more validated feature shards, `snip_id` uniqueness, and the registry
paths for per-well and merged feature products.

**Config surface:** `feature_extraction.consolidated_features`

- `join_key`: default `snip_id`;
- `feature_tables`: ordered list of feature products to merge;
- `column_collision_policy`: fail loud unless collisions are explicitly allowed;
- `required_core_features`: minimum columns expected in the consolidated output;
- `missing_feature_policy`: fail, allow-null, or skip for optional feature products.

**What it computes:** the merged per-object/per-snip feature table used by downstream model/QC and
analysis-ready stages.

**Depends on:**

- one or more validated per-well feature shards;
- shared identity keys across feature tables;
- registry-supported per-well and merged artifact paths.

**Needs before coding:**

- canonical join key, expected to be `snip_id` or the post-segmentation object identity chosen by
  the object contract;
- collision policy for overlapping columns;
- explicit list of required core feature columns for the consolidated contract.

**Done when:**

- shard merge is one-to-one on the chosen key;
- duplicate keys and column collisions fail loud;
- downstream QC can read the consolidated contract without feature-specific path knowledge.

---

## Stage Predictions

**Product folder:** `src/data_pipeline/feature_extraction/stage_predictions/`

**Key functions:**

- `contract.py::STAGE_PREDICTION_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_stage_prediction_features(df)`
- `compute.py::compute_stage_prediction_features(feature_df, *, model_version)`
- `compute.py::predict_stage_hpf(feature_row)`
- `entrypoint.py::main()`

**Direct dependencies:** morphology/size feature inputs, stable feature units, and model/version
provenance if persisted.

**Config surface:** `feature_extraction.stage_predictions`

- `model_name`: default stage model name, for example `kimmel1995_temp_rate`;
- `model_version`: persisted model/formula version;
- `temperature_rate_slope` and `temperature_rate_intercept`: formula coefficients if using the
  temperature-rate model;
- `stage_min_hpf` and `stage_max_hpf`: optional clipping bounds;
- `required_input_columns`: morphology/timing columns required by the model;
- `missing_input_policy`: fail loud unless a documented fallback exists.

**What it computes:** developmental stage prediction from morphology/size features.

**Depends on:**

- consolidated or selected geometry feature inputs;
- stage inference model/rule already present in `feature_extraction`;
- stable feature names and units.

**Needs before coding:**

- feature inputs required by the stage model;
- model/version provenance fields if predictions become a persisted contract;
- null behavior when required morphology features are missing.

**Done when:**

- deterministic fixture rows produce expected stage predictions;
- missing feature inputs fail loud;
- output can be merged with consolidated features without changing upstream feature contracts.

---

## Fraction Alive

**Product folder:** `src/data_pipeline/feature_extraction/fraction_alive/`

**Key functions:**

- `contract.py::FRACTION_ALIVE_FEATURES_REQUIRED_COLUMNS`
- `contract.py::validate_fraction_alive_features(df)`
- `compute.py::compute_fraction_alive_features(frame_masks_df, auxiliary_masks_df, *, mask_decoder)`
- `compute.py::compute_fraction_alive_for_masks(embryo_mask, via_mask)`
- `via_masks.py::build_via_mask_lookup(auxiliary_masks_df)`
- `entrypoint.py::main()`

**Direct dependencies:** canonical embryo masks, auxiliary/VIA mask product, `snip_id`, `image_id`, and
feature-table alignment rules.

**Config surface:** `feature_extraction.fraction_alive`

- `auxiliary_mask_type`: named auxiliary mask product to consume;
- `viability_channel_id`: the **single designated channel** on which viability/`fraction_alive` is
  computed per animal. `fraction_alive` projects to one trace per `physical_embryo_id` / `time_index`
  so that downstream animal-level consumers (death_detection) get exactly one series per animal. This
  is where the channel is chosen — death_detection does not choose or combine channels;
- `join_key`: default `image_id` or documented object key;
- `missing_auxiliary_mask_policy`: fail loud, return null, or configured fallback;
- `empty_embryo_mask_policy`: fail loud or documented null behavior;
- `fraction_clip_min` and `fraction_clip_max`: expected output range, default 0.0 to 1.0;
- `overlap_mode`: pixel-overlap rule for embryo mask versus auxiliary/VIA mask.

**What it computes:** continuous viability fraction from embryo masks and auxiliary/VIA masks.

**Depends on:**

- canonical embryo/object masks;
- auxiliary/VIA mask product contract and paths;
- clear join between object mask rows and auxiliary mask rows by `image_id` or object identity.

**Needs before coding:**

- auxiliary mask world or QC-world decision on VIA mask ownership;
- explicit behavior when VIA masks are missing;
- tests for empty embryo mask, full dead tissue mask, no overlap, and partial overlap.

**Done when:**

- feature computation uses canonical mask products, not legacy path guesses;
- missing auxiliary masks fail loud with the expected source named;
- output joins cleanly into consolidated features.

---

## Death Detection QC

**Product folder:** `src/data_pipeline/quality_control/death_detection/`

> **Input clarification (do not re-conflate):** death detection consumes the `fraction_alive`
> **feature** (continuous %-alive from VIA masks). It does **not** consume `viability_qc`.
> `viability_qc` is a separate mask-plausibility flag over `mask_geometry` and is dropped for MVP
> (see Legacy Domain Retirement). The only viability-derived input here is `fraction_alive`.

**Key functions:**

- `contract.py::DEATH_DETECTION_QC_REQUIRED_COLUMNS`
- `contract.py::validate_death_detection_qc(df)`
- `contract.py::DEATH_EVENT_REQUIRED_COLUMNS`
- `contract.py::validate_death_event(df)`
- `compute.py::compute_death_detection_flags(fraction_alive_df, snip_universe_df, frame_timing_df, *, thresholds)`
  — returns the per-snip table with both `viability_dead_flag` and `persistence_dead_flag`
- `compute.py::compute_viability_dead_flag(fraction_alive_df, *, dead_fraction_threshold)`
  — per-frame, no grouping
- `persistence.py::find_inflection_candidates(physical_embryo_fraction_alive_df, *, thresholds)`
- `persistence.py::validate_death_persistence(physical_embryo_fraction_alive_df, inflection_time_index, *, thresholds)`
- `persistence.py::broadcast_persistence_dead_flag(physical_embryo_fraction_alive_df, called_death_time_index)`
- `stage_at_death.py::compute_death_event(persistence_deaths_df, frame_timing_df, stage_predictions_df, *, lead_time_hr)`
  — returns the per-`physical_embryo_id` `death_event` table (`physical_embryo_id`, `experiment_id`,
  `well_id`, `death_event_time_index`, `death_event_stage_hpf`)
- `alignment.py::align_death_flags_to_snip_universe(death_flags_df, snip_universe_df)`
- `entrypoint.py::main()`

**Direct dependencies:** `fraction_alive` feature rows, feature universe keyed by `snip_id`,
per-embryo temporal ordering, **frame timing** (elapsed hours per `time_index`, from
`frame_inventory`), and QC defaults/thresholds. The per-embryo stage-at-death output additionally
depends on `stage_predictions`.

**Config surface:** `quality_control.death_detection`

- `persistence_threshold`: required post-inflection dead fraction, current default 0.80;
- `lead_time_hr`: lead time in **hours** subtracted from the inflection's elapsed time to define the
  called death time, current default 4.0. This requires frame timing (elapsed hours per
  `time_index`); it is **not** subtracted from a raw frame index. See "Time and lead-time" below;
- `decline_rate_threshold`: minimum decline rate for candidate inflections (persistence mode), current
  default 0.05;
- `dead_fraction_threshold`: per-frame `fraction_alive` cutoff that sets `viability_dead_flag` and
  provides dead-state evidence for persistence, current default 0.90;
- `min_timepoints`: minimum observations per embryo before death detection is attempted;
- `time_column`: `time_index` — the per-frame spine axis. Death detection sorts and groups on
  `time_index`; elapsed-hour conversion for lead time is a separate timing join, not a column swap;
- `smoothing_window`: optional smoothing window for noisy fraction-alive traces;
- `transient_decline_policy`: reject transient dips unless persistence passes;
- `missing_fraction_policy`: fail loud or documented skip behavior;
- `output_alignment_policy`: output must align one-to-one to the feature universe.

**What it decides:** whether a snip is dead, decided in **two independent modes**. A separate
per-embryo output records the lead-time-adjusted predicted death time and stage.

**Two death modes, two flags (kept separate):** "dead" can mean two different facts, and collapsing
them into one flag hides information, so this product emits **two** per-`snip_id` boolean flags:

| Flag | Grain | Mode | Fires when | Decided from |
|---|---|---|---|---|
| `viability_dead_flag` | per-frame | viability | *this frame* is mostly dead tissue | threshold on the frame's own `fraction_alive` |
| `persistence_dead_flag` | per-embryo → snip | persistence | the embryo's `fraction_alive` trace inflects and stays down | inflection + persistence over the embryo timeseries, broadcast to snips |

These can disagree, and the disagreement is informative: a viability hit with no persistence is a
transient dip (the embryo looked dead one frame and recovered); persistence true with a clean frame
means the animal is past its death time but that frame still segments. **Do not pre-merge them.**
Flags are facts; `snip_qc` ORs them into the `dead` exclusion reason and the final `use_snip` verdict.

A pre-death frame that trips `viability_dead_flag` is still unusable even though
`persistence_dead_flag` is false for it — that exclusion simply comes from the viability flag. We do
**not** blanket-condemn every snip of a dead embryo (see broadcast rule below).

**Classification:** QC, not feature extraction. It consumes `fraction_alive` and the feature universe,
then emits flags/annotations. It does not compute a new measured morphology feature.

**No diagnostic widening:** the flag tables carry only the decision plus the timing/stage annotations.
Do **not** add diagnostic columns (decline rate, post-inflection dead fraction, confidence scores,
the `fraction_alive` trace) to satisfy a review plot. Every such diagnostic is recomputable from
`fraction_alive` + these two output tables, so it lives in the review/visualization tooling (the
per-embryo death-review plots), not in the persisted contract. Keep `*_flag` tables narrow so
`snip_qc` and `analysis_ready` stay simple. A future genuinely-needed diagnostic is added only through
an explicit contract migration, not opportunistically.

**Two outputs, two grains:** this product emits **two** tables. The primary table is the per-`snip_id`
death flag table. The secondary table is the per-`physical_embryo_id` stage-at-death table produced by
`stage_at_death.py`. They have different grains and different contracts; they are not merged into one.

**Depends on:**

- `fraction_alive` feature rows carrying the **full snip spine** (`SNIP_ID_SPINE_COLUMNS`),
  `time_index`, and `fraction_alive`;
- a feature universe table with exactly one row per `snip_id`;
- frame timing (elapsed hours per `time_index`) from `frame_inventory`, required for lead-time;
- `stage_predictions` rows, required only for the per-`physical_embryo_id` `death_event` output;
- stable per-`physical_embryo_id` temporal ordering.

**Input contract:**

- `fraction_alive` input columns — the **full snip spine plus the trace**, because `fraction_alive` is
  a feature table and the universal rule is that feature outputs carry identity (so the compute does
  not re-join to discover `physical_embryo_id` / `well_id`):
  - `SNIP_ID_SPINE_COLUMNS` (`experiment_id`, `well_id`, `physical_embryo_id`, `embryo_id`, `snip_id`)
  - `time_index`
  - `fraction_alive`
- frame timing input — keyed for a globally-safe elapsed-time join:
  - `experiment_id`, `well_id`, `time_index`
  - elapsed-hours column (from `frame_inventory`), used only for lead-time conversion
- `stage_predictions` input (`death_event` output only):
  - `snip_id` (or `embryo_id` + `time_index`)
  - `predicted_stage_hpf`
- feature universe input:
  - `SNIP_ID_SPINE_COLUMNS`
  - any extra feature columns needed only to prove the universe and downstream joins

**Single viability channel (fail loud) — channel chosen upstream:** persistence is computed per
`physical_embryo_id`, so each physical embryo must have **exactly one** `fraction_alive` time series.
**death_detection does not choose or combine channels.** The designated viability channel is selected
**upstream by `fraction_alive`/config** (see `feature_extraction.fraction_alive.viability_channel_id`),
which projects to one trace per `physical_embryo_id` / `time_index`. death_detection simply **assumes**
that projection has happened and **fails loud** if a `physical_embryo_id` carries more than one trace —
death is an animal-level fact and this product does not become a channel-selection swamp.
(Multi-channel combination is a documented future extension owned by `fraction_alive`, not MVP, not
death_detection.)

QC should align to the feature universe. Missing, duplicate, or extra `snip_id` rows must fail loud
before flags are written.

**Time and lead-time (hard implementation requirement):** the per-frame axis is `time_index` (the
spine column). Death detection sorts and groups on `time_index`. The configurable `lead_time_hr` is in
**hours**, so the inflection's `time_index` is first converted to elapsed hours via the **frame timing
input**, the lead time is subtracted in hours, and the called-death time maps back to the appropriate
frame. Do not subtract `lead_time_hr` directly from `time_index` — that conflates frames with hours.

Because `lead_time_hr` is exposed, the **frame timing input is mandatory, not optional**. It must map
each `(experiment_id, well_id, time_index)` to an elapsed-time column (`elapsed_time_s` or
`elapsed_time_hr`); `time_index` alone is not globally safe across wells. The timing join is keyed on
`experiment_id`, `well_id`, `time_index` (add `channel_id` only if timing genuinely differs by
channel — ideally timing is well/timepoint-level). A run that exposes `lead_time_hr` without a real
timing table is a contract violation, not a fallback.

**Output contract — per-snip death flag table:**

- `SNIP_ID_SPINE_COLUMNS`
- `viability_dead_flag`
- `persistence_dead_flag`

`viability_dead_flag` and `persistence_dead_flag` are both required non-null booleans (both end in
`_flag` per the Stage Table Pattern). Beyond the snip spine, this table is **product-pure**: its only
product columns are the two per-frame flags. It does **not** carry any death-time or stage column —
those are physical-embryo-grain facts and live in the `death_event` table below. The raw inflection
frame is an internal intermediate inside `persistence.py`, not an output column; the only persisted
death time is the lead-time-adjusted `death_event_time_index` in the `death_event` table.

**Persistence broadcast rule:** `persistence_dead_flag` is decided once per `physical_embryo_id` (the
lead-time-adjusted death frame `D`) and broadcast to that animal's snips: it is **true for snips with
`time_index >= D`** and
**false before `D`**. Pre-death frames stay persistence-false so healthy early timepoints remain
usable; any of them that are independently garbage are caught by `viability_dead_flag`. Embryos with
no detected death have `persistence_dead_flag = false` for all their snips.

**Output contract — per-embryo `death_event` table:**

Identity-bearing event table at **physical-embryo grain** (the *animal* died — channel- and
time-independent, so **no `embryo_id`**). It carries the **physical-embryo spine**
(`PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` — `experiment_id`, `well_id`, `physical_embryo_id`), imported from the
identity minting site (not re-typed) and validated for physical-embryo grain, so reviewers and joins
know *where* a dead embryo lived without decoding IDs or joining the registry, plus the two event
annotations:

```python
from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

DEATH_EVENT_REQUIRED_COLUMNS = list(
    PHYSICAL_EMBRYO_ID_SPINE_COLUMNS + ("death_event_time_index", "death_event_stage_hpf")
)
# validated at physical-embryo grain (no embryo_id); see the validator grain-name note in
# Stage Table Pattern.
```

One row per **persistence-dead** `physical_embryo_id`. Both event annotations are auto-computed by
`stage_at_death.py` and both carry the **lead-time adjustment**:

- `death_event_time_index` — the lead-time-adjusted death frame (`D` in the broadcast rule above).
  This is the **single** persisted death-time output; the raw pre-adjustment inflection frame is an
  internal intermediate and is not surfaced as its own column;
- `death_event_stage_hpf` — the developmental stage at that adjusted death event. **This is "stage at
  the inferred death event," not a raw `stage_predictions` output.** It is `stage_predictions` sampled
  at the lead-time-adjusted death frame; it is named `death_event_*` precisely so it is never read as
  "the stage model predicted death."

They depend on `lead_time_hr`, which is why they are derived helper outputs and not raw passthroughs
from `stage_predictions`. Computed from the internal inflection time, frame timing, the configured
lead time, and `stage_predictions`. This table is keyed off `persistence_dead_flag` only — per-frame
viability hits do not define an embryo death event. Embryos with no persistence death do not appear.

**Grain/spine rule (hard):** snip-grain QC tables carry the **full snip spine**
(`SNIP_ID_SPINE_COLUMNS`); physical-embryo-grain event tables carry the **physical-embryo spine**
(`PHYSICAL_EMBRYO_ID_SPINE_COLUMNS` — `experiment_id`, `well_id`, `physical_embryo_id`; no `embryo_id`).
An event table is not per-snip, but it is still identity-bearing — never make a consumer decode IDs to
learn where an event happened. Every output table — snip-grain and event-grain alike — runs a
grain/identity schema check before writing (the per-snip checker would have caught the earlier missing
spine; see Stage Table Pattern).

**Algorithm shape:**

- **viability flag (per frame):** for each snip, set `viability_dead_flag` from a threshold on that
  frame's own `fraction_alive`. No grouping, no time ordering needed;
- **persistence flag (per animal):** group by **`physical_embryo_id`** (the animal — *not* `embryo_id`,
  which is channel-specific and would split one animal across channels; death is an animal-level fact,
  matching the `death_event` grain), sort by `time_index`, find sustained `fraction_alive` decline
  candidates, validate post-inflection persistence, convert the inflection `time_index` to elapsed
  hours, subtract `lead_time_hr`, map back to the called-death frame `D`, then broadcast
  `persistence_dead_flag = time_index >= D` onto that animal's snips. Each `physical_embryo_id` must
  have exactly one `fraction_alive` trace (single viability channel) — fail loud otherwise;
- align the per-snip flag table back to the feature universe by `snip_id`;
- separately, for each persistence-dead `physical_embryo_id`, compute `death_event_time_index` (the
  adjusted death frame `D`) and `death_event_stage_hpf` at that frame, and emit the
  per-`physical_embryo_id` `death_event` table with its physical-embryo spine.

**Done when:**

- synthetic time series cover alive, clearly dead, transient decline, and too-few-timepoints cases;
- a transient-dip fixture sets `viability_dead_flag=true` on the dip frame while
  `persistence_dead_flag` stays false — proving the two flags are independent and the dip is not
  promoted to embryo death;
- a clear-death fixture sets `persistence_dead_flag=true` only from the adjusted death frame onward,
  leaving healthy pre-death frames persistence-false;
- lead-time conversion is tested against a fixture with a **non-uniform frame interval** joined by
  `(experiment_id, well_id, time_index)` — proving hours, not frames, are subtracted;
- both outputs pass their grain/identity schema checks: the per-snip table the full snip spine, the
  `death_event` table the physical-embryo spine (`experiment_id`, `well_id`, `physical_embryo_id`);
- the `death_event` table only contains persistence-dead embryos and carries both
  `death_event_time_index` and `death_event_stage_hpf`;
- consolidated QC can merge both death flags without rereading masks or segmentation outputs.

---

## Surface Area QC

**Product folder:** `src/data_pipeline/quality_control/surface_area_qc/`

**Packaged reference location:**

```text
src/data_pipeline/quality_control/surface_area_qc/references/
  surface_area_reference_v1.csv
  README.md
```

A curated static default reference lives beside the code because it is part of the QC product and
should be versioned/reviewed with the logic that consumes it. `paths.py` does not need a row for
this packaged reference because it is source data, not a pipeline artifact.

**Reference migration:** the live reference already exists as
`metadata/sa_reference_curves.csv` (columns: `stage_hpf`, `p5`, `p50`, `p95`, `n`; 259 data rows).
That file is the source for `surface_area_reference_v1.csv`. When implementing this product:

1. copy `metadata/sa_reference_curves.csv` into
   `surface_area_qc/references/surface_area_reference_v1.csv`, **keeping all five columns**
   (`stage_hpf`, `p5`, `p50`, `p95`, `n`). Flagging uses `p5`/`p95`; `p50` and `n` are retained for
   review plots, future retuning, post-MVP alternate-percentile selection, and provenance;
2. write `reference_contract.py` validating all five column names (the legacy
   `validate_sa_reference` body in `surface_area_outlier_detection.py` ports almost verbatim);
3. retire the `metadata/sa_reference_curves.csv` path from `quality_control.smk` — the rule must
   no longer receive the reference as an input; `entrypoint.py` loads it via `reference.py` instead;
4. once the packaged copy is the source of truth, the `metadata/sa_reference_curves.csv` copy may be
   deleted in a later cleanup (it is git-tracked; deletion is a separate, explicit step).

**Legacy cleanup (part of done, not a later pass):**

- delete `quality_control/morphology_qc/size_validation_qc.py` — a duplicate of the SA logic with a
  broken absolute import and stale 1.2/0.9 defaults; it has no home in the target layout;
- `quality_control/generate_references/build_sa_reference.py` is **reference-build tooling, not a
  pipeline step**. It carries hardcoded paths into the *other* repo (`proj/morphseq`, not `-docs`).
  Keep it as documented provenance for how `surface_area_reference_v1.csv` was built, but it lives
  outside the product folder (a `tools/`-style location) and gets no `paths.py` row;
- the rule is **broken today**: `entrypoints/compute_surface_area_qc.py` requires `--mask-geometry-csv`
  but `quality_control.smk` never passes it. The migrated entrypoint must wire `mask_geometry` (and now
  `stage_predictions`) as explicit registry-resolved inputs.

**Key functions:**

- `contract.py::SURFACE_AREA_QC_REQUIRED_COLUMNS`
- `contract.py::validate_surface_area_qc(df)`
- `reference_contract.py::SURFACE_AREA_REFERENCE_REQUIRED_COLUMNS`
- `reference_contract.py::validate_surface_area_reference(df)`
- `reference.py::load_packaged_surface_area_reference(version="v1")`
- `reference.py::interpolate_reference_band(stage_hpf, surface_area_reference_df)`
  — returns the `(p5, p95)` band interpolated at the embryo's stage
- `compute.py::compute_surface_area_qc_flags(mask_geometry_df, stage_df, snip_universe_df, surface_area_reference_df, *, thresholds)`
- `compute.py::compute_surface_area_flag(area_um2, stage_hpf, surface_area_reference_df, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** `mask_geometry` feature rows (`area_um2`), `stage_predictions` rows
(`predicted_stage_hpf`), feature universe keyed by `snip_id`, validated surface-area reference rows,
and configured threshold policy.

**Stage input (required, not optional):** surface_area_qc is **stage-binned** QC — it interpolates
the reference `p5`/`p95` band at each snip's developmental stage and flags area outside
`[k_lower·p5, k_upper·p95]`. It therefore **requires `predicted_stage_hpf`**, which comes from the
`stage_predictions` product and is joined on `snip_id`. This is a real DAG edge: `surface_area_qc`
depends on `stage_predictions`, not just `mask_geometry`. There is no stage-free path in MVP — a snip
missing `predicted_stage_hpf` fails loud per `missing_stage_policy` rather than silently using a
global band.

**Config surface:** `quality_control.surface_area_qc`

- `reference_version`: packaged static reference version, default `v1`;
- `area_column`: default `area_um2`;
- `stage_column`: developmental-stage column for reference interpolation, default `predicted_stage_hpf`;
- `k_upper`: upper multiplier (flag when `area_um2 > k_upper·p95`), **canonical default 1.4**;
- `k_lower`: lower multiplier (flag when `area_um2 < k_lower·p5`), **canonical default 0.7**;
- `missing_reference_policy`: fail loud or documented fallback;
- `missing_area_policy`: fail loud or documented flag behavior;
- `missing_stage_policy`: fail loud when `predicted_stage_hpf` is absent (no stage-free fallback in MVP).

> **One tuning dial per side; the band is fixed.** The flag is `area > k_upper·p95` /
> `area < k_lower·p5`. The reference percentile **curves** (`p5`/`p95`) are the fixed baseline — the
> 5th/95th-percentile wildtype area at each stage — and `k_upper`/`k_lower` are the **only** tuning
> dials: how far past that baseline is tolerated before flagging. Percentile and `k` are
> mathematically redundant for the decision (moving either moves the cutoff), so the MVP exposes only
> `k` to avoid two interacting knobs that say the same thing. Exposing the percentile (selecting a
> different reference column, or a numeric percentile against a regenerated reference) is a documented
> **post-MVP** extension; the reference already carries `p50`/`n` so that door stays open without a
> rebuild.
>
> **Config declaration is required (self-documenting at runtime).** When `surface_area_qc` resolves
> its config, it **must print a plain-language statement of the active band** so the meaning is never
> reverse-engineered from code. The statement names both multipliers, both percentile curves, the
> per-stage interpolation, and the too-small / too-large directions. Required form (values filled from
> resolved config):
>
> ```text
> surface_area_qc band: flag area_um2 OUTSIDE [ k_lower(0.70) x p5 , k_upper(1.40) x p95 ]
>   percentiles interpolated per snip at predicted_stage_hpf (wildtype reference, fixed for MVP);
>   k = tolerance multiplier beyond the reference curve.
>   -> "too small" if area < 0.70 x p5;  "too large" if area > 1.40 x p95.
> ```
>
> This declaration is part of done: a test asserts the statement is emitted and reflects the resolved
> `k_lower`/`k_upper`.

> **Threshold canon:** `k_upper=1.4` / `k_lower=0.7` (from `config.py`, what actually runs) is
> canonical. The legacy function-signature defaults and docstring claiming `1.2` / `0.9` are **stale**
> and must be corrected to match during migration — do not carry the 1.2/0.9 values forward.

> **No `reference_group_columns`:** the reference is a single global stage→percentile curve;
> selection is **stage interpolation**, not row-grouping. The earlier `reference_group_columns` /
> `select_surface_area_reference` grouping idea was scope creep with no legacy basis and is dropped.
> Per-group references are a possible post-MVP extension and would arrive as a new reference version.

**What it decides:** whether a snip has a suspicious area measurement for downstream analysis.

**Reference policy:**

- the default curated reference is packaged in `surface_area_qc/references/`;
- packaged reference files are static source assets, not runtime outputs;
- `paths.py` does not know about the packaged reference;
- `compute.py` never secretly reads the reference source;
- `entrypoint.py` loads the packaged default through `reference.py`, validates it, then passes
  `surface_area_reference_df` into `compute.py`;
- if a future generated reference becomes a true pipeline output, that future generated artifact gets
  a path-registry row, but the packaged default still stays beside this code.

**Depends on:**

- `mask_geometry` rows with `snip_id` and `area_um2`;
- `stage_predictions` rows with `snip_id` and `predicted_stage_hpf`;
- a feature universe table with exactly one row per `snip_id`;
- a validated surface-area reference table;
- threshold/reference configuration owned by QC config, not hardcoded in the compute function.

**Input contract:**

- `mask_geometry` input: `snip_id`, `area_um2`;
- `stage_predictions` input: `snip_id`, `predicted_stage_hpf`;
- feature universe input: `snip_id`;
- packaged reference: `stage_hpf`, `p5`, `p50`, `p95`, `n` (flag math uses `p5`/`p95`; `p50` and `n`
  are kept for review plots, retuning, and provenance — see reference policy).

The stage and area joins onto the universe are one-to-one on `snip_id`; missing, duplicate, or extra
rows fail loud before flags are written.

**Output contract:**

- `SNIP_ID_SPINE_COLUMNS`
- `sa_outlier_flag`

`sa_outlier_flag` is a required non-null boolean. Any threshold annotations must be named as
annotations, not flags.

**Future (possible) annotations — not MVP, do not sneak in:** because the band is a *directly applied
threshold* (unlike death's derived review statistics), persisting the interpolated band can save exact
re-interpolation during review: `surface_area_reference_p5`, `surface_area_reference_p95`,
`surface_area_lower_threshold`, `surface_area_upper_threshold`. These are **optional annotation
columns for a future version**, added deliberately if review plots need them often — never required in
MVP and never flags.

**Done when:**

- packaged `surface_area_reference_v1.csv` validates with `reference_contract.py` (all five columns);
- fixture rows cover low, normal, high, missing, and duplicate `snip_id` cases;
- a fixture covers stage-binned behavior: the same `area_um2` flags at one stage and passes at
  another, proving interpolation is stage-driven;
- the config-declaration statement is emitted on config resolution and a test asserts it reflects the
  resolved `k_lower`/`k_upper` and the `p5`/`p95` band;
- missing area, missing stage, or missing reference values fail loud or map to documented QC behavior;
- the output aligns one-to-one with the feature universe.

---

## Mask Quality QC

**Product folder:** `src/data_pipeline/quality_control/mask_quality_qc/`

> **Migrated out of the SAM2 path.** This logic currently lives in
> `quality_control/segmentation_qc/segmentation_quality_qc.py` and reaches into the **raw SAM2
> tracking CSV**, decoding `mask_rle` inline. That coupling is wrong: QC consumes **canonical
> `frame_masks`** through the shared mask decoder, exactly like `mask_geometry`. The migration keeps
> the flag logic but **re-points the input** from backend-native SAM2 output to canonical masks. This
> is the integration cost — the checks are simple, the input swap is the work.

**What it decides:** whether a snip's mask is structurally untrustworthy — cut off at the frame edge,
broken into disconnected pieces, or overlapping another embryo. These are **segmentation-quality**
judgments, distinct from morphology features and from viability/death.

**Three per-snip flags (no persisted composite):**

| Flag | Fires when | Detects |
|---|---|---|
| `edge_flag` | mask touches the image boundary within `margin_pixels` | incomplete embryo cut off at frame edge |
| `discontinuous_mask_flag` | more than one significant connected component (> `min_component_fraction` of the largest) | tracking/segmentation split the mask |
| `overlapping_mask_flag` | IoU with another embryo's mask in the same image exceeds `iou_threshold` | embryo ID confusion |

**No `mask_quality_flag` composite in MVP.** A persisted composite is contract bloat and a double-count
foot-gun: the verdict builder already ORs reasons, so a stored composite alongside its components risks
someone later mapping both into `SNIP_QC_EXCLUSION_REASONS` and producing duplicate verdict semantics.
`snip_qc` consumes the **three component flags** and does the OR itself. If a future dashboard needs a
single convenience column it can derive it on read — it is not part of the persisted contract, and
`snip_qc` must never consume a composite alongside the components.

**Key functions:**

- `contract.py::MASK_QUALITY_QC_REQUIRED_COLUMNS`
- `contract.py::validate_mask_quality_qc(df)`
- `compute.py::compute_mask_quality_qc_flags(snip_inventory_df, frame_masks_df, *, mask_decoder, thresholds)`
- `compute.py::compute_edge_flag(mask, *, margin_pixels)`
- `compute.py::compute_discontinuous_flag(mask, *, min_component_fraction)`
- `compute.py::compute_overlap_flags_for_image(image_masks_by_physical_embryo, *, iou_threshold)`
  — pairwise IoU only between distinct `physical_embryo_id`s in one image; flags both snips of an
  over-threshold pair
- `entrypoint.py::main()`

**Direct dependencies:** validated `snip_inventory` (the snip→mask handoff, carrying the
`physical_embryo_id`, `well_id`, and `image_id` spine columns), canonical `frame_masks`, the shared
mask decoder, and configured thresholds. The overlap check uses the explicit `physical_embryo_id`
column — it does not parse `snip_id`. It does **not** read `segmentation_tracking.csv` or decode RLE
inline.

**Config surface:** `quality_control.mask_quality_qc`

- `margin_pixels`: edge-contact margin, current default 2;
- `min_component_fraction`: minimum component size relative to the largest to count as significant,
  current default 0.05;
- `iou_threshold`: pairwise IoU cutoff for overlap, current default 0.10;
- `missing_mask_policy`: fail loud or documented flag behavior for snips with no decodable mask.

**Grain — overlap is the special case:** `edge_flag` and `discontinuous_mask_flag` are pure per-snip
(each depends only on its own mask). `overlapping_mask_flag` is the ID-confusion check, and it has a
specific, narrow scope:

- it is computed **per image plane** — same well, same `time_index`, same channel — so only masks
  present in the *same frame* are compared. Different timepoints are never overlap-tested. The image
  plane is keyed by `image_id`, which is a **contract-guaranteed-unique** minted identity (it encodes
  experiment/well/time/channel); group by `image_id` directly. Do not group by an under-specified raw
  field, and do not assume `image_id` is non-unique — if that guarantee ever weakens, group by the
  explicit `(experiment_id, well_id, time_index, channel_id)` tuple instead;
- everything resolves **within `well_id`**: a well's embryos are the only masks that can spatially
  collide, and the per-well shard already bounds the comparison. The overlap check never reaches
  across wells;
- the pairwise IoU is computed only between **distinct `physical_embryo_id`s**. Two masks belonging to
  the *same* physical embryo overlapping is not ID confusion and is not flagged. Identity comes from
  the explicit `physical_embryo_id` column on every row (the identity spine) — the compute does not
  parse `snip_id` to recover the animal;
- when two distinct physical embryos' masks exceed `iou_threshold`, **both** snips in the pair get
  `overlapping_mask_flag = true` (you cannot tell which mask is wrong, so both are suspect). No partner
  annotation is recorded — the contract stays narrow.

The output is still one row per `snip_id`; the per-image, within-well, distinct-embryo grouping is an
internal compute step, named in the contract like the death-persistence broadcast.

**Output contract:**

- `SNIP_ID_SPINE_COLUMNS`
- `edge_flag`
- `discontinuous_mask_flag`
- `overlapping_mask_flag`

All three flags are required non-null booleans (all end in `_flag` per the Stage Table Pattern). No
composite column is persisted.

**Done when:**

- the per-snip grain/identity schema check passes (full snip spine carried);
- synthetic masks pin each flag: an edge-touching mask, a two-component mask, and an overlapping pair
  of **distinct** physical embryos in one image, plus a clean mask that trips none;
- the overlap check flags **both** snips of an over-IoU distinct-embryo pair and leaves non-overlapping
  siblings clean;
- a same-physical-embryo overlap fixture is **not** flagged (proves identity-keyed comparison, not
  blind snip pairing);
- input is canonical `frame_masks` via the shared decoder — no `mask_rle` / SAM2 CSV read remains;
- output aligns one-to-one with the feature universe without adding or dropping `snip_id` rows.

---

## Focus QC (STUB — in development, not MVP)

> **STUB.** Focus QC is **not** part of MVP and is not migrated this pass. It is in active development
> in `results/mcolon` (the `20260423_focus_artifact_detection` bundle) and is a special case because
> it **ingests z-stacks** — it needs wiring specifically for that input, unlike the snip-table QC
> products. The design below is a placeholder for when it lands; do not implement against it yet. Its
> future flag is `focus_flag`, and `snip_qc` reserves a future hook for it (see Snip QC).

**Product folder:** `src/data_pipeline/quality_control/focus_qc/`

**Key functions:**

- `contract.py::FOCUS_QC_REQUIRED_COLUMNS`
- `contract.py::validate_focus_qc(df)`
- `compute.py::compute_focus_qc_flags(focus_feature_df, snip_universe_df, *, thresholds)`
- `compute.py::compute_focus_flag(focus_metric, *, thresholds)`
- `entrypoint.py::main()`

**Direct dependencies:** a named focus/input-quality feature table or image-quality contract,
feature universe keyed by `snip_id`, and configured thresholds.

**Config surface:** `quality_control.focus_qc`

- `focus_metric_product`: named upstream product that owns the focus metric;
- `focus_metric_column`: metric column to threshold;
- `focus_min_threshold` or `focus_max_threshold`: configured cutoff, depending on metric direction;
- `inheritance_grain`: whether focus is per snip, per image, or inherited from frame-level QC;
- `missing_focus_policy`: fail loud or documented neutral flag behavior.

**What it decides:** whether a snip should be flagged for focus or image-quality failure.

**Depends on:**

- a prior contract that defines the focus metric input;
- one row per `snip_id` in the feature universe;
- QC thresholds from config.

**Needs before coding:**

- name the input focus metric product and its schema;
- decide whether focus is computed per image, per object/snip, or inherited from a frame-level
  quality product;
- avoid reading raw images from this QC stage unless the focus metric contract explicitly says so.

**Output contract:**

- `SNIP_ID_SPINE_COLUMNS`
- `focus_flag`

`focus_flag` is a required non-null boolean.

**Done when:**

- the input focus metric contract exists;
- fixture rows cover good focus, bad focus, missing focus, and duplicate keys;
- output aligns to the feature universe.

---

## Blur QC (STUB — in development, not MVP)

> **STUB.** Blur QC is **not** part of MVP and is not migrated this pass — it has no legacy module to
> migrate; it is net-new and in active development. Like Focus QC, it **ingests z-stacks** and needs
> input wiring specific to that, so it is not a simple snip-table QC product. This section reserves
> the slot so the deck and `snip_qc` know it is coming.

**Product folder:** `src/data_pipeline/quality_control/blur_qc/` (future)

**What it decides:** whether a snip should be flagged for blur / out-of-focus image degradation,
distinct from Focus QC's metric. (The exact metric and the focus-vs-blur boundary are part of the
in-development design and are not pinned here.)

**Output contract (future):**

- `SNIP_ID_SPINE_COLUMNS`
- `blur_flag`

`blur_flag` will be a required non-null boolean when the product lands. `snip_qc` reserves a future
hook for it (see Snip QC).

**Needs before coding:**

- the z-stack input contract and how blur is measured across the stack;
- the per-snip projection from the z-stack-grained metric;
- the focus-vs-blur boundary so the two flags are not redundant.

---

## Snip QC

**Product folder:** `src/data_pipeline/quality_control/snip_qc/`

**Files:**

```text
src/data_pipeline/quality_control/snip_qc/
  contract.py
  inputs.py
  build.py
  entrypoint.py
  __init__.py
```

**Purpose:** build the final per-snip QC verdict from already-computed stage-specific QC flags.

Doctrine: QC products find problems; `snip_qc` builds the verdict; `analysis_ready` applies the
verdict. Flags are facts. Reasons are verdict prose. `use_snip` is the switch.

**Grain:** one row per `snip_id`.

**Output artifact:**

- step/product: `snip_qc`
- artifact key: `verdict`
- per-well file: `<well_id>_snip_qc.parquet`
- merged file: `<experiment_id>_snip_qc.parquet`

**Output columns:** the full snip spine (`SNIP_ID_SPINE_COLUMNS`, by reference — not re-listed) plus the
verdict columns `use_snip` and `qc_fail_reasons`.

`snip_qc` is the final operational QC table and must **not** be weaker than its inputs: by the
identity-spine doctrine, a per-snip table carries the full spine. It is **not** a minimal-key
exception. The spine is owned by `SNIP_ID_SPINE_COLUMNS` and enforced by
`validate_snip_grain_identity_columns` — the contract does not re-type the spine columns. This lets
`analysis_ready` consume the verdict without joining the registry just to learn basic identity.

`qc_fail_reasons` is a non-null pipe-delimited string. Empty string means pass. Examples: `""`,
`"dead_viability"`, `"dead_persistence"`, `"surface_area_outlier"`, `"edge"`, `"overlapping_mask"`,
`"dead_persistence|surface_area_outlier"`, `"edge|discontinuous_mask"`.

**Key functions and constants:**

- `contract.py::SNIP_QC_REQUIRED_COLUMNS`
- `contract.py::SNIP_QC_EXCLUSION_REASONS`
- `contract.py::validate_snip_qc(df, *, source="")`
- `inputs.py::load_snip_qc_flag_inputs(...)`
- `build.py::build_snip_qc_verdict(snip_universe_df, qc_flags_df, *, exclusion_reasons)`
- `entrypoint.py::main()`

**Contract policy:**

```python
# Spine imported from its minting site (NOT re-typed), composed with the verdict columns.
# This is the TARGET pattern every product contract converges on, e.g.
#   MASK_GEOMETRY_FEATURES_REQUIRED_COLUMNS = list(SNIP_ID_SPINE_COLUMNS + _FEATURE_COLUMNS)
# (today mask_geometry uses a local _SPINE_COLUMNS literal — that is one of the duplicates the
#  minting-site prerequisite collapses; see Stage Table Pattern.)
from data_pipeline.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    validate_snip_grain_identity_columns,
)

_SNIP_QC_VERDICT_COLUMNS = ("use_snip", "qc_fail_reasons")
SNIP_QC_REQUIRED_COLUMNS = list(SNIP_ID_SPINE_COLUMNS + _SNIP_QC_VERDICT_COLUMNS)

SNIP_QC_EXCLUSION_REASONS = {
    "dead_viability": "viability_dead_flag",
    "dead_persistence": "persistence_dead_flag",
    "surface_area_outlier": "sa_outlier_flag",
    "edge": "edge_flag",
    "discontinuous_mask": "discontinuous_mask_flag",
    "overlapping_mask": "overlapping_mask_flag",
}

# Future hooks — NOT in the MVP map. Add only when the source product lands and its
# flag column actually exists, else snip_qc fails loud on a missing column:
#   "missing_metadata" -> "metadata_missing_flag"   (after metadata_completeness_qc)
#   "focus"            -> "focus_flag"               (after focus_qc; z-stack, in dev)
#   "blur"             -> "blur_flag"                (after blur_qc; z-stack, in dev)
```

The `snip_qc` contract owns the verdict policy: reason name maps to source flag column. The MVP map
covers the three migrated/specced exclusion families: death (two modes), surface area, and mask
quality (three flags). The two death modes surface as **two distinct reasons** (`dead_viability`,
`dead_persistence`) rather than a single `dead`, and the three mask-quality checks surface as `edge`,
`discontinuous_mask`, and `overlapping_mask` — so the verdict prose records exactly which fact
excluded the snip, consistent with keeping the flags separate upstream. A snip can carry several. For
MVP, do not add a source-product registry to the contract, and do **not** add the future-hook reasons
until their products exist and emit the named flag column.

**Input assembly:**

`inputs.py` owns the small, explicit in-memory assembly of QC flag columns needed to build the final
verdict. It imports public path-registry helpers from `orchestration.paths`; it does not
define a second QC source registry.

```python
from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
    known_artifacts,
    validated_path,
)
```

Use registry helpers. Do not inspect registry internals unless no helper exists. Once the back-half
QC rows are added to `paths.py`, `inputs.py` resolves paths with `artifact_path(...)`, optionally
checks sentinels with `validated_path(...)`, and reads only the requested flag columns. It must not
construct raw paths, discover arbitrary QC artifacts, or duplicate source step/artifact filename
mappings locally.

`load_snip_qc_flag_inputs(...)` takes explicit registered source step/artifact pairs from
`entrypoint.py`, for example the MVP sources `death_detection_qc` and `surface_area_qc`. The
entrypoint should pass artifact keys explicitly. `inputs.py` may infer an artifact key with
`known_artifacts(step)` only when a source step has exactly one registered artifact, and must fail
loud if a source step has multiple artifacts and no key was passed. The function returns one row per
`snip_id` with only `snip_id` plus the requested flag columns. Missing registry rows, missing source
artifacts, missing flag columns, duplicate `snip_id`, null flags, and non-boolean flags fail loud.
MVP must not treat missing flags as pass.

**Build behavior:**

`build.py` is pure verdict logic. It does not import `paths.py`, know product artifact names, or load
files. It consumes `snip_universe_df` and an already-assembled `qc_flags_df`.

- start from `snip_universe_df[list(SNIP_ID_SPINE_COLUMNS)]` — the **full snip spine**, not just
  `snip_id`. The verdict table carries the spine, so the builder must keep it from the first line (if
  it started from `[["snip_id"]]` the final table could never satisfy `SNIP_QC_REQUIRED_COLUMNS`);
- require `snip_universe_df` has one row per `snip_id`;
- require `qc_flags_df` has one row per `snip_id` for the relevant universe;
- require every flag column named by `SNIP_QC_EXCLUSION_REASONS` is present in `qc_flags_df`;
- require those flag columns are non-null boolean;
- for each snip, build `qc_fail_reasons` from reasons whose flag column is true;
- set `use_snip = qc_fail_reasons == ""`;
- return exactly `SNIP_QC_REQUIRED_COLUMNS` (spine + `use_snip` + `qc_fail_reasons`).

**Validation:**

- `snip_id` is present, non-null, and unique;
- `use_snip` is present and non-null boolean;
- `qc_fail_reasons` is present, non-null string;
- empty `qc_fail_reasons` means the snip passed QC;
- non-empty `qc_fail_reasons` is a pipe-delimited list of known reason keys;
- all reasons are known keys in `SNIP_QC_EXCLUSION_REASONS`;
- `use_snip` is true iff `qc_fail_reasons == ""`;
- `use_snip` is false iff `qc_fail_reasons != ""`.

**Entrypoint behavior:**

- load the feature universe from validated `snip_inventory`;
- call `load_snip_qc_flag_inputs(...)` for current MVP exclusion flags;
- call `build_snip_qc_verdict(...)`;
- call `validate_snip_qc(...)`;
- write the `verdict` artifact via `artifact_path(...)`;
- write the validation marker via `validated_path(...)`.

**Explicit non-goals:**

- no `flag_contract.py`;
- no `exclusion_reasons.py`;
- no source-product registry in `contract.py`;
- no local duplicate of `paths.py` registry data in `inputs.py`;
- no direct `PIPELINE_STEPS` import or inspection unless a registry helper is genuinely missing;
- no `nullifies_columns`;
- no `snip_qc_flags` wide artifact in MVP;
- no feature nullification;
- no biological metadata joins;
- no genotype/condition/`use_snip` mutation in feature products.

**Boundary:**

Stage-specific QC tables remain the source of detailed QC facts. `snip_qc` is the final operational
verdict table. `analysis_ready` decides whether to filter rows, expose rows with `use_snip=false`,
or null selected outputs.

Tiny doctrine: paths locate artifacts; inputs assemble columns; build resolves verdicts; contract
guards meaning.

---

## Not This World

- Detection, segmentation, tracking, and prompt adaptation live in detect/seg/track specs.
- Analysis-ready joins live after features and `snip_qc`.
- GPU SAM2 validation lives in segmentation Session C, not feature computation.

> Candidate legacy review/visualization scripts to mine when building the QC/feature *debug plots*
> (not part of the contract) were moved out of this target spec into
> `_feature_qc_visualization_migration_notes.md` — they were field notes, not doctrine.
