# Distribution catalog API (DECIDED)

The ontology is simpler than the API vocabulary we circled. Locked model below.
Two analysis shapes — **within-population** and **cross-population** — chosen by
whether you split on the comparison axis. The cross-population structure is an
analysis object reused beyond plotting (densities, peak tables, peak deltas).

## The ontology

A `Distribution` is ONE population:

```
samples        (sample_ids — the join key)
features       (feature_names + feature_values)
label groups   (sample-aligned partitions: genotype, phenotype, resolved_peak)
coordinates    (distribution-level constants: time_bin, scope_id, ...)
```

The two metadata types have different jobs:

```
coordinates  →  which distribution you have   →  selection & faceting & compare()
label groups →  how samples inside it split   →  within-cell grouping/overlay
```

### The coordinate-vs-label test (the rule)

Not "is it constant?" but:

> Should different values coexist in the same analytical population?

- **14 hpf vs 48 hpf** → no → **coordinate** (`time_bin`, `scope_id`, `experiment_id`).
- **b9d2 vs WT / CE vs HTA at one time** → yes → **label group** (`genotype`, `phenotype`, `resolved_peak`).

### role is a RELATIONSHIP, not a distribution label

Reference/target is NOT intrinsic to a distribution — the same WT population is
the reference in a b9d2 comparison, a target in a batch-effect comparison, a peer
elsewhere. So `role` is assigned by a COMPARISON when distributions are paired
(see `compare()`), never stored as a label group on the Distribution. (Earlier
drafts wrongly called role a label group — corrected.)

## Objects

```
DistributionCatalog       coordinate-indexed collection of populations
Distribution              samples + features + label groups + coordinates
DistributionLabelGroup    "use THIS label group from THIS distribution" (within-population)
DistributionComparison    ONE matched group: {coordinates, members: {across_value → Distribution}}
DistributionComparisons   the family compare() returns (+ across + values + match_on)
SampleSet                 derived categories inside a label group (a view, not stored glue)
```

Minimal shapes (NO role field — member identity IS the across value; reference/
target is optional presentation, see ComparisonMemberFacet):

```python
@dataclass(frozen=True)
class DistributionComparison:
    coordinates: Mapping[str, Hashable]        # held-constant, e.g. {"time_bin": 30}
    members: Mapping[Hashable, Distribution]   # indexed by across value: {"wildtype":…, "b9d2":…}

@dataclass(frozen=True)
class DistributionComparisons:
    comparisons: tuple[DistributionComparison, ...]
    across: str
    values: tuple[Hashable, ...]
    match_on: tuple[str, ...]
    # invariant: tuple(c.members) == values for every c (unless partials allowed later)
```

`DistributionComparison(s)` are ANALYSIS structures, not plotting objects — the
same matched family drives density plots, peak-count tables, peak-position
deltas, HDR widths. That reuse is what justifies the object.

## Typed facet keys (NOT an enum)

Coordinates are open-ended (`time_bin`, `batch`, `plate_id`, `stage`, …), so an
enum rots. Use a typed key that says WHERE the value comes from:

```python
@dataclass(frozen=True)
class CoordinateFacet:    name: str     # a distribution coordinate
@dataclass(frozen=True)
class LabelGroupFacet:    pass          # the label-group display axis
@dataclass(frozen=True)
class ComparisonMemberFacet: pass       # facet by member IDENTITIES (across values)
                                        # within each comparison. NOT "role" — across
                                        # values are member identities (wildtype/b9d2,
                                        # dose 0/1/5, experiment names); reference/target
                                        # is optional presentation, not ontology.

FacetKey = CoordinateFacet | LabelGroupFacet | ComparisonMemberFacet
```

## Construction and labeling are SEPARATE steps

```python
catalog = DistributionCatalog.from_dataframe(
    df, sample_id_column="embryo_id", feature_columns=FEATURE_NAMES,
    label_columns=("genotype", "phenotype"),   # carried through best-effort (unassigned where missing)
    split_columns=("time_bin",),               # one distribution per coordinate combo
)
catalog = catalog.map_distributions(
    lambda d: d.discover_modes(
        features=FEATURE_NAMES, output_label="resolved_peak", spec=peak_spec,
    ).distribution
)
```

`discover_modes` writes the `resolved_peak` label column AND (eager) its peak
geometry; the grid is built from each distribution's OWN points.

---

## PATH A — within-population (label group + build_1d_density_grid)

When the comparison axis is a LABEL (you did NOT split on it). One distribution
per cell; curves = SampleSets of one label group on that distribution.

```python
groups = (
    *catalog.label_groups("resolved_peak", display_name="Resolved peaks"),
    *catalog.label_groups("phenotype",     display_name="Phenotype"),
    *catalog.label_groups("genotype",      display_name="Genotype"),
)
grid = build_1d_density_grid(
    groups, feature="total_length_um",
    facet_row=LabelGroupFacet(), facet_col=CoordinateFacet("time_bin"),
)
plot_1d_density_grid(grid, output_path="outputs/b9d2_grid_total_length_um.png")
```

The label group is chosen by the INPUT object — the builder gets NO second
grouping parameter (no `group_by`, no `overlay_across`).

### One-distribution-per-cell invariant (structural)

```python
if len({g.distribution.distribution_id for g in cell_groups}) != 1:
    raise IncomparableDistributionsError(...)
```

Forbids nonsense like overlaying `time=14` against `time=48` in one cell. Cell
grid = union bounds of the CURVES SELECTED for that cell (NOT automatically all
of the distribution's samples — if `unassigned` is omitted, those samples do not
stretch the grid). "One distribution" ≠ "all its samples."

---

## PATH B — cross-population (compare + build_1d_distribution_comparison)

When the comparison axis is a COORDINATE (you DID split on it) → separate
populations, matched into comparisons. This is the catalog's superpower: it turns
into a matched-comparison engine.

### compare(): group by everything-but-`across`, vary `across`

```python
comparisons = catalog.compare(
    across="genotype",            # the coordinate that varies
    values=("wildtype", "b9d2"),  # (optional) restrict/order the across values
    # match_on defaults to ALL OTHER coordinates held constant.
)
```

Semantics = "hold everything else constant, vary `across`":

```
time=14: {wildtype: distA, b9d2: distB}
time=18: {wildtype: distC, b9d2: distD}
...                                        (one DistributionComparison per time bin)
```

Algorithm:
1. Confirm `across` is a catalog COORDINATE (a split column). If it's a label
   group, raise ("genotype is a label group, not a coordinate — split on it, or
   use build_1d_density_grid with label_group('genotype')").
2. Resolve `match_on` = ALL catalog coordinates except `across` (default).
3. Group distributions by `match_on` values (group-by-zero-columns → ONE global
   comparison keyed by `{}` is valid; do NOT reject a single-split-coordinate
   catalog).
4. Within each group, index distributions by `across` value.
5. Require exactly one distribution per requested `across` value (raise on
   missing or duplicate).
6. Preserve the requested `values` order.

Guards / semantics:
- **Omitting a coordinate from an explicit `match_on` is NOT pooling** — it is an
  assertion that the omitted coordinate is CONSTANT in the selection. If it
  varies, that would create duplicate members → raise ("Cannot omit coordinate
  'experiment_id': multiple values would produce duplicate genotype members.
  Filter the catalog first or include experiment_id in match_on."). True pooling
  (aggregating several distributions into one member) is a separate FUTURE op
  (`catalog.pool_by(...)` / `combine_distributions(...)`), NOT this.
- **Log the inference** (standard logger, NOT a verbose= API flag): "matching on
  (time_bin); resolving across genotype." Makes a forgotten/constant coordinate
  visible without leaking UI behavior into the scientific API.

### Consume the SAME comparisons many ways

Plot densities (one shared grid per comparison cell, across ALL members):

```python
grid = build_1d_distribution_comparison(
    comparisons, feature="total_length_um",
    label_group="resolved_peak",          # ONE shared label group on every member (like-with-like)
    facet_col=CoordinateFacet("time_bin"),
)
plot_1d_density_grid(grid)
# each time cell: wildtype·peak_0, wildtype·peak_1, b9d2·peak_0, b9d2·peak_1, ...
```

Shared grid = union bounds of EVERY selected curve across ALL comparison members
in that cell (wildtype's peaks + b9d2's peaks). Curve identity carries BOTH axes
as a structured key (never a concatenated `"wildtype_peak_0"` string):

```python
@dataclass(frozen=True)
class CurveKey:
    comparison_member: Hashable   # the across value, e.g. "wildtype"
    sample_set: str               # e.g. "peak_0"
# display renders "wildtype · peak 0"; the model keeps the axes separate so you
# can facet/style on either.
```

Tabulate / delta peak properties (not plotting — same structure):

```python
peak_table = comparisons.to_peak_dataframe(label_group="resolved_peak")
# columns: time_bin, genotype (across value), LOCAL_peak_id, peak_position, peak_mass, ...
```

Deferred escape hatch (NOT built now): per-member asymmetric label groups
`label_groups={"wildtype":"reference_peak","b9d2":"resolved_peak"}`. Reopens
"comparing incomparable partitions"; single shared `label_group=` only for now.

### Distribution matching ≠ peak matching (critical scientific boundary)

`compare()` matches POPULATIONS. It does NOT make `wildtype·peak_0` and
`b9d2·peak_0` the same biological mode — `peak_0` is a LOCAL label from each
distribution's own resolver. `to_peak_dataframe()` therefore emits `local_peak_id`
(honestly local), never a globally-comparable peak id.

Cross-distribution peak correspondence is a SEPARATE future operation
(`match_peaks`, uses the eager peak geometry `discover_modes` computes). Strategy
names are NOT locked here — naming them now reads as promised semantics. See the
Deferred section. Do NOT let the density/table path silently imply peak_0↔peak_0
correspondence.

---

## ID helpers (debugging / explicit one-offs — below compare())

IDs are the debugging surface; `compare()` is the real surface. `find_ids` /
`resolve_id` search COORDINATES only (raise if you query a label value).

```python
catalog.to_index_dataframe()      # one row per distribution, coordinate columns — inspectable/filterable
catalog.find_ids(time_bin=30)     # -> tuple[str, ...]
catalog.resolve_id(time_bin=30, genotype="wildtype")  # exactly one, or raise (ambiguous/none)
```

`distribution_id` is implementation identity (cache / provenance / equality) —
never a plotting concept. Plotting reads `coordinate("time_bin")`.

## Which path? (decided by the split)

```
split on the comparison axis?  → PATH B: compare() + build_1d_distribution_comparison
keep it in one population?      → PATH A: label_group() + build_1d_density_grid
```

Same coordinate-vs-label test, applied to "which API." `find_ids`/`compare`
search coordinates; `label_group` selects inside one distribution.

## pool_by — the second half of the construction idiom (BUILD NOW)

Pooling is NOT an exotic future op — it is how the standard split is made safe.
The honest default split is `("time_bin", "genotype", "experiment")`: keep
experiment SEPARATE at construction so it is never silently confounded, then
`pool_by("experiment")` to collapse replicates for the actual comparison. Without
`pool_by` you are forced to NOT split on experiment, reintroducing the exact
confounding `compare()`'s guards prevent. So the strict split and `pool_by` are
ONE feature.

```python
catalog = DistributionCatalog.from_dataframe(
    df, ..., split_columns=("time_bin", "genotype", "experiment"),
)
catalog = catalog.pool_by("experiment")   # collapse the nuisance axis
```

`pool_by("experiment")` — SAMPLE pooling only: for distributions differing only
in `experiment`, concatenate their sample-aligned ROWS into one Distribution and
collapse `experiment` out of the coordinate map. Density RE-FITS from the pooled
samples. (Density mixing/averaging is a separate future verb, never `pool_by`.)

### Provenance is in the samples (soften it)

When rows are concatenated, EACH pooled sample keeps its own `sample_id` and its
own label calls — including which experiment it came from (as a label). So the
pooled distribution ALREADY knows its lineage at the SAMPLE grain; do NOT
re-store it at the distribution grain. Keep only what is NOT recoverable from
samples:

- **`pooled_coordinates: tuple[str, ...]`** — a tiny note of which coordinate(s)
  were collapsed (so `compare()` won't still treat them as coordinates). That's
  it. No `source_distribution_ids`, no `source_coordinate_values` — those are in
  the samples.

The ONE guard that must stay (it's join-key integrity, not ceremony):
- **Reject duplicate sample_ids across pooled sources** (raise, no auto-dedup).
  If two experiments both have `embryo_001`, the sample-grain lineage is
  ambiguous and the whole "provenance is in the samples" scheme collapses.

Labels ride along per-sample (unassigned where a label group is absent for some
source); do NOT merge by category name.

## Deferred — TODO notes, NOT committed architecture

Let these earn their place by becoming necessary, not by being possible:
- **`comparison_id` hashing** — for now, table rows carry explicit keys
  (`time_bin`, `across`, `across_value`, `distribution_id`, `local_peak_id`) —
  readable and sufficient. Add a deterministic id when caching / cross-product
  table joins actually need one.
- **`match_peaks`** — keep ONLY the warning: local `peak_0` is not globally
  comparable. "Future: explicit peak-matching operation." Do NOT name strategies
  in the locked API (naming them reads as promised semantics).
- **`ComparisonMemberFacet`** — ship `CoordinateFacet` + `LabelGroupFacet` only;
  member identity still lives in `CurveKey`. Add when a plot facets WT-vs-b9d2
  across cells (rather than styling them within a cell).
- **`LineageRef`, `PoolSpec`, `mix_densities`, cache/dedup infra** — not yet.

## What must stay in sync (the real invariant surface)

- `Distribution.coordinates` ↔ `distribution_id` (id derived from coordinates;
  changing coordinates → new id).
- catalog coordinate index ↔ contained distributions (`compare(across=X)` only
  works if X is a catalog coordinate; the catalog must know its coordinate names).
- `Distribution.labels` ↔ `SampleSet` — SampleSets are DERIVED from labels, never
  stored durably. ("Never store SampleSets as durable state unless absolutely
  necessary.")
- `DistributionComparisons.values` ↔ each comparison's members:
  `tuple(comparison.members) == values` (order-preserving). Out of sync → plots
  and tables lie.
- `local_peak_id` ↔ provenance — peak ids are LOCAL to a distribution; don't
  pretend otherwise (a conceptual sync burden more than a code one).

## Removed vocabulary

```
view(...)          → label_group(...)
group_by=...       → gone (the label group IS the grouping)
overlay_across=... → gone (compare() is the real cross-population engine)
FacetCoordinate enum → typed FacetKey (CoordinateFacet / LabelGroupFacet / ComparisonMemberFacet)
DistributionGrouping / MaterializedDistributionGrouping → gone from public API
```
