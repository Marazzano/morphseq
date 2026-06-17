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

---

## Ownership Rule

Each contract module answers one domain question:

| Domain product | Contract owner | Meaning |
|---|---|---|
| `discovered_wells.txt` | `metadata_ingest/well_discovery/contracts.py` | physical well identities from canonical metadata |
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

    table_contracts.py
      TableContract
      assert_columns_present
      assert_unique_on_key
      assert_positive_numeric
      assert_allowed_values

  metadata_ingest/
    contracts/
      well_acquisition_summary.py

    well_discovery/
      contracts.py
      from_scope_metadata.py
      from_frame_inventory.py
      discover_wells.py

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

`shared/table_contracts.py` should contain generic mechanics only:

```python
@dataclass(frozen=True)
class TableContract:
    name: str
    required_columns: tuple[str, ...]
    unique_key: tuple[str, ...] = ()

def assert_columns_present(df, required_columns, *, contract_name): ...
def assert_unique_on_key(df, key, *, contract_name): ...
def assert_positive_numeric(df, column, *, contract_name): ...
def assert_allowed_values(df, column, allowed_values, *, contract_name): ...
```

It must not import microscope packages, frame inventory, well discovery, segmentation, or Snakemake.
Domain packages import these helpers, not the other way around.

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

1. Add `metadata_ingest/well_discovery/contracts.py`.
2. Add `shared/table_contracts.py`.
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
