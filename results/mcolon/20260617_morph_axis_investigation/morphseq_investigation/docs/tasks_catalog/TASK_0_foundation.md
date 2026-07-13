# TASK 0 — Foundation: typed-column Distribution + derived SampleSet (BLOCKING, SERIAL, DO FIRST)

**Nobody else starts until this is committed on `main`.** You are freezing the
`Distribution` / `LabelColumn` / `SampleSet` / `DistributionLabelGroup` contract
and the id scheme that A/B/C all key off. A change here after the fact ripples
into three branches.

## Source of truth
`docs/DISTRIBUTION_CATALOG_API.md` — "The ontology", "Objects", "Typed facet
keys", and "What must stay in sync". The existing `engine/objects.py` is your
starting point (reshape it; keep what the spec keeps).

## Scope (what you build)

### `engine/objects.py` — reshape the core

`Distribution` becomes a typed-column sample table. Keep it RECOGNIZABLE (NOT a
generic `columns: Mapping[str, Column]` framework — that was explicitly rejected
as recreating AnnData). Concretely:

```python
@dataclass(frozen=True)
class Distribution:
    distribution_id: str                 # DERIVED from coordinates (TASK_0 identifiers)
    sample_ids: tuple[str, ...]          # the unique join key
    feature_names: tuple[str, ...]
    feature_values: np.ndarray           # (n_samples, n_features); col j ↔ feature_names[j]
    labels: Mapping[str, LabelColumn] = field(default_factory=dict)      # sample-aligned partitions
    coordinates: Mapping[str, Hashable] = field(default_factory=dict)    # distribution-level constants
```

Methods on `Distribution` (pure, return NEW objects):
- `with_label(name, assignments: Mapping[str,Hashable], *, provenance=None) -> Distribution`
  — attach a label column; unassigned where a sample_id is missing.
- `sample_sets(label_name) -> tuple[SampleSet, ...]` — DERIVE sets by slicing the
  label column (one SampleSet per category). NEVER store SampleSets durably.
- `label_group(label_name, *, display_name=None) -> DistributionLabelGroup`.
- `coordinate(name) -> Hashable` — read one coordinate (used by faceting).
- (`from_dataframe` / `discover_modes` are declared here but IMPLEMENTED in
  A / B respectively — TASK_0 may leave `discover_modes` as a documented stub that
  raises `NotImplementedError` so B fills it. `from_dataframe` likewise stubbed
  for A, OR put it on the catalog only — confirm placement against the spec.)

```python
@dataclass(frozen=True)
class LabelColumn:
    name: str
    values: Mapping[str, Hashable]        # sample_id -> call (UNASSIGNED_LABEL where absent)
    provenance: LabelProvenance | None = None   # discover_modes fills method/features/spec + geometry
```

`SampleSet` — KEEP the dataclass but it is now a DERIVED VIEW (name + sample_ids
+ optional geometry read back from the label column's provenance). It is produced
by `Distribution.sample_sets(...)`, not hand-constructed by callers. Keep the
existing geometry/HDR fields so peak rings survive.

`DistributionLabelGroup` — the plotting bridge:
```python
@dataclass(frozen=True)
class DistributionLabelGroup:
    distribution: Distribution
    label_name: str
    display_name: str
    def sample_sets(self) -> tuple[SampleSet, ...]:
        return self.distribution.sample_sets(self.label_name)
    def coordinate(self, key: "FacetKey") -> Hashable: ...   # LabelGroupFacet -> display_name; else distribution.coordinate
```

`UNASSIGNED_LABEL` — a module constant; the canonical "no call" value used by
`with_label` and pooling.

### `engine/identifiers.py` — ids from coordinates
- `make_distribution_id(coordinates: Mapping[str,Hashable]) -> str` — DERIVED from
  the coordinate map (canonical, deterministic; prefix `dist_`). Same coordinates
  → same id. Follow the compose-from-typed-parts, never-parse-back doctrine.
- `make_sample_set_id(distribution_id, name) -> str` — double-underscore separator
  so `peak_0`s from different distributions never collide.
- Keep/port `make_grid_id` from the old engine UNCHANGED (grid_id raster-
  comparability is a preserved invariant; grid.py still uses it).
- **Do NOT build `comparison_id` or pooled-distribution content hashes** — those
  are DEFERRED in the spec. Pooled distribution ids just re-derive from their
  (post-collapse) coordinate map like any other.

### `engine/invariants.py` — keep the guards
Port the existing coverage + assignment-consistency + ordered-feature guards.
Add: `sample_sets(label)` output must cover exactly the label column's assigned
samples (derived-view consistency). Expose one `validate_*` entry the labelers
call.

### Typed FacetKey — declare here (used by C)
Put the `FacetKey` types where both objects.py and plotting.py can import them
(a small `engine/facets.py`, or top of objects.py — confirm against layout):
```python
@dataclass(frozen=True)
class CoordinateFacet: name: str
@dataclass(frozen=True)
class LabelGroupFacet: pass
FacetKey = CoordinateFacet | LabelGroupFacet
# ComparisonMemberFacet is DEFERRED — do not add it.
```

## Out of scope (later tasks — do NOT build)
- `DistributionCatalog`, `compare`, `pool_by`, `from_dataframe` impl → TASK_A.
- Column-writer labelers, `discover_modes` impl → TASK_B.
- Any plotting, `build_1d_*`, `CurveKey` → TASK_C.
- `DistributionGrouping` / `MaterializedDistributionGrouping` — DELETE these from
  objects.py (they leave the public API). Grep callers; TASK_C/E replace them.

## Tests (green before every commit) — `tests/engine/test_objects.py`, `test_identifiers.py`, `test_invariants.py`
- Build a `Distribution` with features + coordinates + one label column.
- Frozen-ness (assignment raises); `feature_values[:,j] ↔ feature_names[j]`.
- `with_label` attaches; missing sample_id → `UNASSIGNED_LABEL`; returns a NEW
  object (original unchanged).
- `sample_sets(label)` derives one set per category, covers exactly the assigned
  samples, and is NOT stored on the object.
- `make_distribution_id`: same coordinates → same id; different coordinates →
  different id; order-independent over the coordinate map.
- `make_grid_id` ⟺ property still holds (port the old test).
- `DistributionLabelGroup.coordinate(LabelGroupFacet()) == display_name`.

## Commit checkpoints
- `⟢ COMMIT 1` — `engine/objects.py` reshaped (Distribution table + LabelColumn +
  derived SampleSet + DistributionLabelGroup + UNASSIGNED_LABEL) + test green.
  `catalog(task-0): typed-column Distribution + LabelColumn + derived SampleSet`
- `⟢ COMMIT 2` — `engine/identifiers.py` (coordinate-derived dist id, kept grid_id)
  + FacetKey types + test green.
  `catalog(task-0): coordinate-derived distribution id + typed FacetKey`
- `⟢ COMMIT 3` — `engine/invariants.py` guards (incl. derived-view consistency) +
  test green; DistributionGrouping/MaterializedDistributionGrouping deleted;
  `tasks_catalog/README.md` layout confirmed/adjusted.
  `catalog(task-0): guards + retire DistributionGrouping glue`
- Open PR. **Announce TASK_0 merged** — green light for A/B/C.

## Definition of done
Frozen table-shaped `Distribution` imports cleanly; `with_label` / `sample_sets`
/ `label_group` work and return new objects; ids derive from coordinates;
DistributionGrouping glue is gone; grid_id ⟺ property still proven; three
checkpoint commits; PR open. No catalog, no labelers, no plotting.
