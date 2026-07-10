# TASK A — DistributionCatalog: from_dataframe / split / pool_by / compare / id-helpers

Parallel after TASK_0. You build the control tower — the object that answers
"what populations exist" and turns them into matched comparisons.

## Source of truth
`docs/DISTRIBUTION_CATALOG_API.md` — "Construction and labeling are separate
steps", "PATH B — cross-population", "pool_by", "ID helpers", "What must stay in
sync". Import the frozen TASK_0 objects; do not redefine them.

## Scope — `engine/catalog.py`

### Construction
```python
DistributionCatalog.from_dataframe(
    df, *, sample_id_column, feature_columns,
    label_columns=(),          # carried through best-effort (unassigned where missing)
    split_columns=(),          # one Distribution per unique coordinate combo
) -> DistributionCatalog
```
- Group `df` by `split_columns`; each group → one `Distribution` whose
  `coordinates` = that group's split values, `distribution_id` derived from them
  (TASK_0 `make_distribution_id`).
- `label_columns` become `LabelColumn`s on each distribution (via `with_label`),
  best-effort, `UNASSIGNED_LABEL` where a sample lacks a value.
- Binning/cleaning is UPSTREAM (the caller's df is already tidy) — the catalog
  never bins.

### Index / discovery (secondary surface — below compare)
- `to_index_dataframe() -> pd.DataFrame` — one row per distribution, one column
  per coordinate. The key debugging affordance; must be filterable/queryable.
- `find_ids(**coords) -> tuple[str, ...]` — searches COORDINATES only. Raise
  (helpful message) if a kwarg names a LABEL, not a coordinate.
- `resolve_id(**coords) -> str` — exactly one match or raise (ambiguous/none).

### pool_by (BUILD NOW — second half of the construction idiom)
```python
pool_by(coordinate: str) -> DistributionCatalog
```
- For distributions differing ONLY in `coordinate`: concatenate their
  sample-aligned rows into one `Distribution`; collapse `coordinate` out of the
  coordinate map (new `distribution_id` re-derives from the reduced coordinates).
- Density is NOT stored, so nothing to re-fit here — pooled samples are just the
  union; downstream materialization re-KDEs per cell as always.
- **Reject duplicate sample_ids across pooled sources** (raise; NO auto-dedup).
- Labels ride along per-sample (concatenate the label columns; `UNASSIGNED_LABEL`
  where a source lacks that label group). Do NOT merge by category name.
- Provenance is SOFTENED (it's in the samples): keep only a `pooled_coordinates`
  note; do NOT store source_distribution_ids / source_coordinate_values.
- Density mixing is NOT this verb (no `mix_densities` — deferred).

### compare (the real surface)
```python
compare(across: str, *, values=None, match_on=None) -> DistributionComparisons
```
Algorithm (per spec):
1. `across` must be a COORDINATE (raise with the "it's a label group — split on
   it, or use build_1d_density_grid" message otherwise).
2. `match_on` default = ALL catalog coordinates except `across`.
3. Group distributions by `match_on` values. Group-by-zero-columns (a single
   split coordinate) → ONE global comparison keyed by `{}` — this is VALID, do
   NOT reject it.
4. Index each group's distributions by `across` value; require exactly one per
   requested `value` (raise on missing/duplicate).
5. Preserve requested `values` order (do NOT sort).
6. Omitting a coordinate from an explicit `match_on` = ASSERT it's constant in the
   selection; if it varies → raise (that would make duplicate members). NOT
   pooling.
7. Log the inference via the standard logger (NOT a `verbose=` flag): "matching on
   (…); resolving across …".

Return the minimal objects (from TASK_0's neighborhood or defined here — confirm):
```python
@dataclass(frozen=True)
class DistributionComparison:
    coordinates: Mapping[str, Hashable]
    members: Mapping[Hashable, Distribution]   # indexed by across value, ORDERED

@dataclass(frozen=True)
class DistributionComparisons:
    comparisons: tuple[DistributionComparison, ...]
    across: str
    values: tuple[Hashable, ...]
    match_on: tuple[str, ...]
```
Invariant: `tuple(c.members) == values` for every `c`. Do NOT add `comparison_id`
(deferred).

### Catalog-wide labeling conveniences (NAMED — no lambdas for common ops)
Because `with_label` and `detect_peaks` (TASK_B) are clean `Distribution ->
Distribution` functions, the catalog exposes NAMED conveniences that apply them to
every distribution with NATIVE arguments — the caller never writes a lambda:
```python
with_labels(df, label_columns) -> DistributionCatalog     # attach provided label cols catalog-wide
detect_peaks(*, features, output_label="resolved_peak", spec=DEFAULT_ANALYSIS_SPEC) -> DistributionCatalog
map_distributions(fn: Callable[[Distribution], Distribution]) -> DistributionCatalog  # generic ESCAPE HATCH
```
- `detect_peaks` / `with_labels` are the API callers use. `map_distributions(fn)`
  is the generic fallback for arbitrary one-off transforms (the only place a
  lambda is appropriate). The named conveniences are thin wrappers over it.

### label_groups (Path A convenience on the catalog)
```python
label_groups(label_name, *, display_name=None) -> tuple[DistributionLabelGroup, ...]
```
— one per distribution in the catalog (each wraps that distribution + the label).

## Stub you may lean on while B lands
`discover_modes` (TASK_B) — you don't need it for catalog/compare tests; build
against `with_label`-created label columns (e.g. a fake `resolved_peak`).

## Tests — `tests/engine/test_catalog.py`
- `from_dataframe` with `split_columns=("time_bin","genotype")` → correct number
  of distributions, right coordinates, labels attached, unassigned where missing.
- `to_index_dataframe` shape; `find_ids`/`resolve_id` hit/ambiguous/none/label-not-
  coordinate-raises.
- `pool_by("experiment")` collapses the coordinate, unions samples, drops the
  coordinate; duplicate sample_id across sources RAISES; labels ride along.
- `compare(across="genotype")`: default match_on inference; single-split-coordinate
  → one global comparison; ordered values preserved; missing/duplicate member
  raises; `across`-is-a-label raises; omitted-but-varying coordinate raises.

## Commit checkpoints
- `⟢ COMMIT 1` — `from_dataframe` + index/find/resolve + test green.
  `catalog(task-a): DistributionCatalog.from_dataframe + coordinate index/lookup`
- `⟢ COMMIT 2` — `pool_by` (with duplicate-id guard + softened provenance) + test.
  `catalog(task-a): pool_by sample-pooling with duplicate-id guard`
- `⟢ COMMIT 3` — `compare` + `DistributionComparison(s)` + `label_groups` + test.
  `catalog(task-a): compare() matched-comparison engine + label_groups`
- Open PR.

## Definition of done
`from_dataframe → (pool_by) → compare` works end to end on a synthetic df;
guards raise on the spec's error cases; `label_groups` feeds Path A; no
`comparison_id`; three checkpoint commits; PR open.
