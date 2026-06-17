# Domain-Level Schema Layout - contracts live with data products

**Status:** target layout, mdcolon 2026-06-17. Companion to
`front_half_reorg_roadmap.md`.

This file records the schema ownership rule for the front-half refactor:

```text
Domain packages own data-product meaning.
Shared code owns validation mechanics.
The old data_pipeline/schemas package is legacy compatibility during migration.
```

The goal is to avoid a central schema drawer where column lists drift away from the pipeline
boundary that gives them meaning.

Naming rule:

```text
*_contract.py      owns a data-product contract
validate_*.py      validates a stage/artifact
*_validators.py    shared validation mechanics
```

Avoid vague names like `contracts.py` for new target files. Avoid generic helper names that look
like first-class pipeline artifacts.

---

## Ownership Rule

Each contract module answers one domain question:

| Domain product | Contract owner | Meaning |
|---|---|---|
| `discovered_wells.txt` | `metadata_ingest/well_discovery/discovered_wells_contract.py` | physical well identities from canonical metadata |
| `acquisition_inventory__yx1.csv` | `metadata_ingest/scope/yx1/acquisition_inventory.py` | YX1 raw tensor-coordinate inventory |
| `acquisition_inventory__keyence.csv` | `metadata_ingest/scope/keyence/acquisition_inventory.py` | Keyence raw acquisition/tile/plane inventory |
| `well_acquisition_summary__{scope}.csv` | `metadata_ingest/contracts/well_acquisition_summary.py` | per-well stitch eligibility |
| `frame_inventory.csv` | `image_materialization/stitched/contracts/frame_inventory_contract.py` | downstream stitched-frame handoff |

Shared validation helpers are intentionally generic. They do not know what a frame, well,
acquisition, or stitcher means.

---

## Target Tree

```text
data_pipeline/
  shared/
    identifiers/
      constructors.py
      parsers.py
      validators.py

    table_validators.py
      assert_columns_present
      assert_unique_on_key
      assert_positive_numeric
      assert_allowed_values

  metadata_ingest/
    contracts/
      well_acquisition_summary.py

    well_discovery/
      discovered_wells_contract.py
      discover_wells_from_scope_metadata.py
      discover_wells_from_frame_inventory.py   # future/drop-in
      discover_wells.py                        # future dispatcher only when 2 sources exist

    scope/
      yx1/
        acquisition_inventory.py
      keyence/
        acquisition_inventory.py

  image_materialization/
    stitched/
      contracts/
        frame_inventory_contract.py
      layout.py
      frame_inventory.py
      materialize_stitched_images.py
      scope/
        yx1/
          materialize_yx1_stitched_images.py
        keyence/
          materialize_keyence_stitched_images.py

  schemas/
    frame_contract.py      # legacy compatibility only during migration
    ...
```

---

## Frame Inventory Contract

The stitched handoff contract is owned by:

```text
image_materialization/stitched/contracts/frame_inventory_contract.py
```

It should define the atom-based identity and derived ids:

```python
FRAME_INVENTORY_IDENTITY_ATOMS = (
    "experiment_id",
    "well_index",
    "channel_id",
    "time_index",
)

DERIVED_COLUMNS_FRAME_INVENTORY = (
    "well_id",
    "image_id",
)

REQUIRED_COLUMNS_FRAME_INVENTORY = (
    *FRAME_INVENTORY_IDENTITY_ATOMS,
    "source_image_path",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
)

UNIQUE_KEY_FRAME_INVENTORY = FRAME_INVENTORY_IDENTITY_ATOMS
```

Rules:

- `well_id` is derived from `experiment_id + well_index`.
- `image_id` is derived from `well_id + channel_id + time_index`.
- If supplied, derived ids are recomputed and checked.
- Native materializers get `source_image_path` from `layout.py`.
- External/drop-in manifests may use arbitrary paths only when explicit, readable, unique, and
  complete.
- External canonical-layout mode is a helper that generates this manifest; it is not a second
  contract.

Compatibility during migration:

| Legacy | Target |
|---|---|
| `frame_contract` | `frame_inventory` |
| `time_int` | `time_index` |
| `stitched_image_path` | `source_image_path` |
| `micrometers_per_pixel` | `source_micrometers_per_pixel` |

Compatibility aliases may exist at adapters, but target contracts should name target columns.

---

## Shared Mechanics

`shared/table_validators.py` should contain generic mechanics only:

```python
def assert_columns_present(df, required_columns, *, contract_name): ...
def assert_unique_on_key(df, key, *, contract_name): ...
def assert_positive_numeric(df, column, *, contract_name): ...
def assert_allowed_values(df, column, allowed_values, *, contract_name): ...
```

It must not import microscope packages, frame inventory, well discovery, segmentation, or Snakemake.
Domain packages import these helpers, not the other way around.

Do not add a `TableContract` class unless an actual caller needs a structured object. Named constants
plus explicit validator functions are clearer for the front-half work.

### Path values are not artifact paths

`pipeline_orchestrator/orchestration/paths.py` owns artifact paths:

```text
output_root / stage / experiment / per_well / well_id / artifact
```

The old `shared/path_contracts.py` is now a deprecation tripwire, not an active target helper. It
used to resolve path **values inside tables** with a hidden `data_pipeline_output` default. Do not
move that behavior. Fix call sites to receive concrete paths or explicit configured roots from the
Snakefile/tasks layer. If future work needs a helper, the clearer name would be
`shared/path_value_validators.py`, and it must take roots as parameters. Do not add artifact-path
semantics there.

---

## Import Rules

- New front-half code imports domain contracts from the owning domain package.
- New code must not add target semantics to `data_pipeline/schemas/frame_contract.py`.
- Legacy importers may continue using `data_pipeline/schemas/*` until migrated.
- When moving an importer, move it all the way to the domain contract rather than adding another
  central alias.
- Contract modules may import `shared/identifiers` when they need to recompute or validate ids.
- `pipeline_orchestrator/paths.py` owns artifact paths, not table schemas.
- `image_materialization/stitched/layout.py` owns native pixel paths, not table schemas.

---

## Migration Order

Do not migrate every schema at once. Start with the front-half handoff because it is the active
refactor boundary.

1. Add `metadata_ingest/well_discovery/discovered_wells_contract.py`.
2. Add `shared/table_validators.py` only if local contract modules would otherwise duplicate checks.
3. Add `metadata_ingest/contracts/well_acquisition_summary.py`.
4. Add `image_materialization/stitched/contracts/frame_inventory_contract.py`.
5. Move frame-inventory builders/validators to import the domain frame contract.
6. Leave `schemas/frame_contract.py` as legacy compatibility until no active importer needs it.
7. Later, apply the same domain-owned pattern to segmentation, snips, QC, and analysis-ready.

The principle is the same throughout:

```text
schemas next to owners,
validators as reusable mechanics,
legacy central schemas retired gradually.
```
