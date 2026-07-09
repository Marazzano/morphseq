# Primitive Ontology — LOCKED

The core primitive for the distribution comparison + plotting engine. Replaces
the earlier `LabeledDistribution` / `LabeledDistributionPartition` /
`PeakMembership` framing (dropped — heavyweight, overloaded "partition").

**Design rule above all else: keep the objects dumb, don't leak between layers.**
Validity/grain/time/binning are the *caller's* job, resolved upstream; the
objects only hold what they are.

---

## The four nouns

```
Distribution        a dumb bag: named samples + named features (caller-composed)
   │
   │  labeler(distribution, method, params)     [peers: genotype | peak | dtw]
   ▼
LabelGroup          one labeler RUN over a Distribution (assignments + evidence)
   │  materializes
   ▼
SampleSet(s)        the durable atom: one named subset + its measured shape
   ▲
   │  indexed by
label_groups        a plain dict {label_group_name: (sample_set_ids,)}  — a VIEW
```

`Distribution`, `Grid`, `LabelGroup`, and `SampleSet` are all **domain objects**
(durable scientific records). `label_groups` is the only non-object — a
lightweight **derived index**. ("Dumb" describes how little `Distribution`
*does*, not whether it's real.)

---

## 1. `Distribution` — the dumb substrate

```python
@dataclass(frozen=True)
class Distribution:
    # --- structured identity (composed from typed parts; NEVER parsed back) ---
    distribution_id: str                 # = make_distribution_id(scope_id, time_bin, role)
    scope_id: str                        # "b9d2"  (gene / experiment)
    time_bin: Any                        # 30  (hpf bin center) — identity, not a live column
    role: str                            # "reference" | "target"
    # --- payload (NATIVE feature values; no owned coordinate frame) ---
    sample_ids: tuple[str, ...]          # sample identities — the CALLER's grain
    feature_names: tuple[str, ...]       # ORDERED axis identity  ← FEATURES ARE PRIMITIVE
    feature_values: np.ndarray           # (n_samples, n_features); col j ↔ feature_names[j]
    scope_tag: str | None = None         # human doc string — never enforced/parsed
```

**No `coordinate_frame`.** (An earlier draft put one here — reverted.) Features
ARE the primitive coordinates; a Distribution stores native feature values in
feature units. There is no separate "canonical" basis to invert. Grids (built
FROM features) are the derived evaluation layouts — see §Grid. Comparability is
by `feature_names` equality; scientific validity (build a shared representation
BEFORE splitting into target/reference) is the caller's responsibility, not the
object's to infer.

Locked responsibilities inversion — the object knows **almost nothing about
process**, but it IS the self-contained unit of comparison:

- **No time-*axis*, no binning, no grain logic.** The caller subsets, bins, picks
  the grain, and hands in a clean bag. `time_bin`/`role` are *identity parts*
  (which bin/role this frozen bag IS), not a live time column to slice.
- **Structured identity, no string archaeology** (mirrors repo `shared/
  identifiers` doctrine). `distribution_id` is *rendered* from `(scope_id,
  time_bin, role)` via `make_distribution_id`; the parts are stored so nothing
  ever parses the string back. `scope_tag` is a free human label only.
- **`sample_ids` is the caller's grain**, opaque to the object. A labeler needing
  biological grain reads a passed-in column; the Distribution never parses ids.
- **Features are ORDERED** — `feature_names` + `feature_values` with the hard
  invariant `feature_values[:, j] ↔ feature_names[j]`. A friendly helper accepts
  a dict/DataFrame at the boundary and normalizes to this ordered pair; the
  interior stays disciplined (grids, centers, covariance, distance need stable
  order). NEVER an anonymous `N×D`.
- Because the object is otherwise dumb, **simple helpers** (`build_distribution`,
  a batch spec form) do the composing — weight in the helper, not the object.

## 1b. `Feature → Grid → DensityGrid` (features are primitive)

The coordinate story, resolved. **Features are the primitive** — they are what
you build grids FROM. A **Grid is a derived, first-class object**: a chosen
*discretization* over some features. **Its axes are in FEATURE units** (PC1 stays
PC1) — there is NO separate "canonical" basis and nothing to invert. Whitening /
quantile is just how the grid picks *bounds and cell spacing*, never a change of
what the numbers mean. (The word "canonical" is retired.)

```python
@dataclass(frozen=True)
class Grid:
    grid_id: str                        # see make_grid_id rule below
    feature_names: tuple[str, ...]      # which features this grid discretizes (ORDERED)
    axis_values: tuple[np.ndarray, ...] # per-axis cell coordinates, IN FEATURE UNITS
    construction_method: str            # "pooled_min_max" | "pooled_quantile" | "pooled_mad_scaled" | ...
    construction_params: Mapping[str, Any] = field(default_factory=dict)  # resolution, bounds rule
    fit_sample_ids: tuple[str, ...] = () # which samples defined its bounds (e.g. pooled target+ref)

@dataclass(frozen=True)
class DensityGrid:
    grid_id: str                        # the Grid it was evaluated on
    feature_names: tuple[str, ...]
    density: np.ndarray                 # shape == tuple(len(a) for a in grid.axis_values)
```

- **Recompute-to-raw is a non-issue.** Because axes are in feature units, a peak
  center / radius read off a grid is *already* in feature (PC1) units. Nothing
  ever left feature space, so there is no inverse to store and no `CoordinateFrame`
  to keep alive. (`pooled_mad_scaled` chooses bounds/spacing via MAD but the
  stored `axis_values` remain feature-unit — decision (b).)
- **Comparability, two levels:**
  - same `feature_names` → **feature-space / scalar** comparable (centers,
    distances) — the axes claim the same features.
  - same `grid_id` → **raster** comparable (density overlap, HDR overlap, basin
    rasters) — evaluated at the same cell locations.
- **Density does not auto-transform between grids.** To compare rasters on
  different grids, re-evaluate the *samples* on one shared grid (exact) — never
  interpolate a density across grids.
- **`grid_id` must hash the ACTUAL axis coordinates** (#6/#7). Construction
  depends on feature *values*, not just IDs — same `fit_sample_ids` under changed
  feature values yields different bounds. So `grid_id` = stable hash of
  `(ordered feature_names, construction_method, normalized params, hash of
  produced axis_values)`. Then **`same grid_id` ⟺ same evaluation coordinates**,
  which is exactly what raster comparability needs. `fit_sample_ids` hashing must
  be **order-independent** — `hash(tuple(sorted(fit_sample_ids)))` — so the same
  pool in a different row order gives the same id. No stateful registry.
- **Shared comparison grid** = build ONE grid from *pooled* target+reference
  feature values; both peak runs reference that `grid_id`. That IS "shared grid
  for the comparison" — no separate frame object.

### KDE strips fall out for free (the whole payoff)

A **KDE strip = a 1-D Grid over ONE feature + a DensityGrid on it.** Same
machinery as the 2-D peak grid, dimension = 1:

- strip curve = `DensityGrid` on `Grid(feature_names=("PC1",), axis_values=(<axis>,))`
- **overlay** N sample sets = evaluate each on the *same* 1-D `grid_id` → N curves,
  one feature-unit x-axis (directly comparable — same grid).
- **N features** = N one-feature grids; **M comparisons** = M sample sets per strip;
  the N×M panel = a faceting-engine grid of these curves.
- No separate strip code path — strips reuse `Grid`/`DensityGrid`. This is exactly
  what "feature is primitive" buys: build a grid over *any* single feature on
  demand, independent of the 2-D grid peaks were found on.

## 2. `SampleSet` — the durable atom

```python
@dataclass(frozen=True)
class SampleSet:
    sample_set_id: str                 # DURABLE, unambiguous:
                                       #   make_sample_set_id(distribution_id, name)
                                       #   → "b9d2_30hpf_reference__peak_0"
    sample_set_name: str               # READABLE, LabelGroup-local: "peak_0", "CE"
    distribution_id: str               # foreign key to its substrate
    sample_ids: tuple[str, ...]        # its members
    feature_profile: FeatureProfile | None = None    # per-feature stats, FEATURE units
    hdr: HDR | None = None                           # density shape (carries grid_id)
    geometry: SampleSetGeometry | None = None        # INTRINSIC geometry (carries grid_id)
    provenance: Mapping[str, Any] = field(default_factory=dict)   # how it was made

@dataclass(frozen=True)
class SampleSetGeometry:
    grid_id: str                       # which grid these coords came from  (#5)
    feature_names: tuple[str, ...]     # so a center is self-describing      (#5)
    center: np.ndarray                 # INTRINSIC only — feature units
    radius: float
    r80: float
    cv_radius_from_center: float
    # NOTE: support_fraction / prominence_rank / height_relative_to_max /
    # is_dominant are RUN-RELATIVE → LabelGroup.per_sample_set_metrics, NOT here (#4)

@dataclass(frozen=True)
class HDR:
    grid_id: str                       # explicit — comparability without tracing  (#5)
    feature_names: tuple[str, ...]
    level: float
    mask: np.ndarray
```

- **Geometry is a measured property, NOT provenance.** center / radius /
  cv_radius / r80 / support_fraction live in a typed `geometry` field — like
  `hdr`/`feature_profile`, they are *what the set is*. A genotype set has
  `geometry=None` (no center); a peak set fills it. Consumers check
  `geometry is not None`, never spelunk a dict for a center.
- **A genotype subset, a DTW cluster, and a peak are the SAME type.** They differ
  only in which optional slots are filled. Downstream never asks "which labeler."
- `provenance` is structured (avoid the junk drawer):
  ```python
  provenance = {
      "labeler":  {"method": ..., "params": {...}, "feature_names": (...)},  # shared run-level
      "evidence": {...this set's detector_detail / basin_validation...},     # per-set
  }
  ```
  `feature_names` = **the features the labeler USED to form the groups** (peaks:
  `("umap_1","umap_2")`; genotype: `()`). This is run-level provenance, the hook
  for later comparability. It is NOT the same as the features you can *profile*
  (that's the Distribution's full feature menu — any of them work on any set).
- **`sample_set_id` (durable) vs `sample_set_name` (local).** The id is composed
  and unambiguous (`b9d2_30hpf_reference__peak_0`) so target/reference `peak_0`s
  never collide in exports/joins; the name is the short readable label. Same for
  `CE`/`HTA`/`unlabeled` across distributions.
- **`geometry`/`feature_profile`/`hdr` are in FEATURE units and carry `grid_id`
  + `feature_names` explicitly** (self-describing — comparability never has to
  trace back through provenance). A center is already in feature (PC1) units.
  Two centers compare directly when they share `feature_names` (scalar) or
  `grid_id` (raster). **Geometry holds only INTRINSIC coords** (center/radius/
  r80/cv); run-relative scalars (support_fraction/prominence_rank/...) live in
  `LabelGroup.per_sample_set_metrics`, never on geometry (one grain each).
- **`unlabeled` ≠ NA ≠ abstention** (three distinct beasts, kept apart in
  provenance so no later normalization fuses them):
  ```python
  evidence = {"source_value": "unlabeled", "is_missing_value": False}  # literal category
  # NA source value → is_missing_value=True
  # labeler abstention → the sample is in LabelGroup.unassigned_sample_ids, not any SampleSet
  ```

## 3. `LabelGroup` — one labeler run (thin run-result, NOT a dict)

A naked dict can't hold the shared field artifacts and run evidence, so the
run is a thin object. (This is **not** a return to `LabeledDistribution` — it
references the Distribution by id and owns the *run's* evidence, not the
samples.)

```python
@dataclass(frozen=True)
class LabelGroup:
    label_group_name: str                          # "genotype", "peak_bwA"
    distribution_id: str                           # foreign key
    sample_set_ids: tuple[str, ...]                # real groups only (NOT unassigned)
    sample_id_to_sample_set_id: Mapping[str, str]  # the assignment
    unassigned_sample_ids: tuple[str, ...]         # residual — a FIELD, not a SampleSet
    provenance: Mapping[str, Any] = field(default_factory=dict)   # vote/is_reliable/params
    artifacts: LabelGroupArtifacts | None = None    # shared, heavy field products
    per_sample_set_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # sibling-relative
    across_sample_set_metrics: Mapping[str, Any] | None = None       # pairwise + summary
```

> **Never `{}`/`[]` as a dataclass default** — always `field(default_factory=...)`.
> This applies to every Mapping/params/provenance field in this doc (#3).

- **Coverage invariant (on the LabelGroup, not via a SampleSet):**
  `⋃ SampleSet.sample_ids ∪ unassigned_sample_ids == Distribution samples`, disjoint.
  Every sample is accounted for; unplaceable ones go to `unassigned_sample_ids`.
- **`unassigned` is NOT a SampleSet.** Residual support is not a coherent group
  (no center, no HDR, may be unrelated rejects) — counting it as a SampleSet
  would inflate the group count and read as a biological mode. It's a first-class
  *field*; plotting renders it as a reserved visual category (`label="unassigned"`).
- **The two extra grains only a run can hold** (grain names ARE the ontology — no
  zoo of "relative"/"pairwise"/"summary" classes):
  - `per_sample_set_metrics` — one value per `sample_set_id`, defined *relative to
    siblings*: `support_fraction`, `prominence_rank`, `height_relative_to_max`,
    `is_dominant`. (Intrinsic geometry stays on the SampleSet; this is only
    sibling-relative stuff.)
  - `across_sample_set_metrics` — requires ≥2 sets: `{"pairwise": [...],
    "summary": {...}}` → inter-set distance / valley depth / bridge mass; and
    group-level balance / entropy / organization. This is the Valley +
    Organization metric families. **Within-one-run only** — cross-run/cross-
    distribution relations (peak in run A ↔ peak in run B) are the *comparison*
    layer, not this slot.

```python
@dataclass(frozen=True)
class LabelGroupArtifacts:
    grid_id: str | None = None                # = grid.grid_id (hashes actual axis_values, §1b)
    grid: Grid | None = None                  # the evaluation grid (§1b), feature-unit axes
    density_grid: DensityGrid | None = None   # the KDE field on that grid
    basin_labels: np.ndarray | None = None    # B(x,y) raster on that grid
    detection_result: PeakDetectionResult | None = None
```

- **Grid is a run artifact, not provenance.** provenance records the *spec*
  (bandwidth, resolution); the grid itself is a concrete data product (§1b).
- **Shared-grid identity via a deterministic `grid_id`, not a registry.**
  Identical construction inputs → identical `grid_id` *by construction*. Two runs
  built from one pooled grid get the same id, and `assert target.artifacts.grid_id
  == reference.artifacts.grid_id` proves it — **no stateful registry**. Each
  artifact still holds its grid (self-contained); dedup is a later optimization.
- **Axes are in feature units** (§1b) — a center off this grid is already in
  feature (PC1) units, no inverse. `grid_id` equality → raster comparability;
  `feature_names` equality → scalar comparability. Different grids, same features:
  reconcile by re-evaluating samples on a shared grid (never interpolate density).

## 4. `label_groups` — a lightweight index (a VIEW, not an object)

```python
label_groups: Mapping[str, tuple[str, ...]]   # {name: (sample_set_ids,)}
```

- **references** its SampleSets, does not own them. Derivable from the LabelGroups
  (`{g.label_group_name: g.sample_set_ids}`) for quick serialization.
- Multiple views coexist for free (two peak bandwidths = two keys).
- **Alias resolution is strict (#8):** exact `label_group_name`s are canonical.
  An alias (`"peak"`) may resolve **only if it identifies exactly one** LabelGroup;
  an ambiguous alias (`peak_bwA` + `peak_bwB` both present) **raises**. No silent
  "first peak-ish thing wins."

---

## Labelers are peers — provided & unsupervised map symmetrically

A labeler is any `label(distribution, method, params) → LabelGroup + [SampleSet]`.
`genotype`, `peak_finding`, `dtw` are peers. Reorganization (KDE field, DTW warp)
is a labeler's *private business*, recorded in provenance/artifacts — never a
structural layer. There is **no `provided` vs `distribution_derived` type** — it's
just provenance (`method`).

The proof the ontology holds: both label kinds run the **same skeleton**,
differing only in which optional slots fill. No `if genotype … else if peak …`
anywhere.

| Slot | genotype (provided) | peak (unsupervised) |
|---|---|---|
| `SampleSet.sample_ids` | ✓ | ✓ |
| `SampleSet.feature_profile` / `hdr` | ✓ | ✓ |
| `SampleSet.geometry` | `None` | ✓ center/radius |
| `provenance.labeler.feature_names` | `()` | `("umap_1","umap_2")` |
| `provenance.evidence` | empty | detector_detail, basin_validation |
| `LabelGroup.artifacts` | `None` | ✓ KDE + basins + grid |
| `LabelGroup.provenance` (vote/is_reliable) | trivial | ✓ bootstrap vote |
| `unassigned_sample_ids` | usually `()` | ✓ residual |
| `across_sample_set_metrics` | `None` (or centroid dists later) | ✓ pairwise/summary |

Two clarifications the trace surfaced:

- **Unknown genotype = a real SampleSet** named `"unknown"` (coherent, plottable,
  has its own HDR). **`unassigned` = the labeler declined/failed to place** a
  sample. Semantically different; kept different.
- Making `unassigned` a LabelGroup *field* (not a SampleSet) is what makes the
  coverage invariant uniform across labelers (genotype `unassigned=()`, peak
  `unassigned≠()`, same shape).

---

## Peak-finding as a labeler (how the existing machinery folds in)

`compute_resolved_peaks` / `ResolvedPeakDistribution` / the bootstrap vote
**become the `peak_finding` labeler**. `ResolvedPeakDistribution` already *is*
"a label_group + evidence for one peak run"; we re-express it, we don't compete.

- Returns **one `SampleSet` per accepted, robust mode** + `unassigned_sample_ids`.
  The vote decides count *before* carving, so two WT-clusters'-worth of points
  vote to **one** mode → one SampleSet (rejected candidates do not become phantom
  SampleSets; the collapse story stays in run provenance).

Field-by-field destination:

| `ResolvedPeakDistribution` field | New home |
|---|---|
| `distribution_id` | `Distribution` (FK on LabelGroup + SampleSet) |
| accepted `ResolvedPeak` (each) | one **`SampleSet`** |
| `PeakGeometry` (center/radius/cv/r80/support) | **`SampleSet.geometry`** (measured, typed) |
| `sample_peak_ids` (positional) | joined to real ids → `SampleSet.sample_ids` + `LabelGroup.sample_id_to_sample_set_id` |
| `PeakCandidateDetail` (per-candidate) | **`SampleSet.provenance.evidence`** |
| per-peak basin validation | **`SampleSet.provenance.evidence`** |
| vote / `count_stability` / `is_reliable` | **`LabelGroup.provenance`** |
| `density_grid`, `basin_labels`, `detection_result` | **`LabelGroup.artifacts`** |
| `source_type: truth/empirical` | excluded from the public real-data ontology; a provenance flag at most (#9) |
| `grid_peak_ids` (truth path) | excluded from real-data ontology (#9) |
| `resolved_peak_count` | **derived** = `len(sample_set_ids)` |

Removed/demoted from the REAL-DATA primitive: positional `sample_peak_ids` and
`resolved_peak_count` as public surface; `ResolvedPeakDistribution` as a
*returned primitive* (it decomposes into `LabelGroup + [SampleSet]`; may survive
as an internal builder). **Truth-only fields (`truth`/`empirical` branch,
`grid_peak_ids`) are excluded from the public real-data ontology but the
synthetic benchmark builders may retain them internally** — the benchmark path is
deferred, not deprecated (#9).

Kept (load-bearing): the bootstrap vote / count-stability (the robustness story),
per-peak geometry, the density field + basin raster (so HDR/basins plot exactly
as resolved), and the assignment (via the identity spine, Sub-spec B).

---

## Invariants (guardrails)

1. **Objects stay dumb; the caller guarantees validity.** Distribution enforces
   no binning/grain logic. `time_bin`/`role` are frozen *identity parts* (this IS
   bin X, role Y), not a live time column to slice. `sample_ids` opaque;
   `scope_tag` documentation.
2. **Identity is structured, never parsed back.** `distribution_id`, `sample_set_id`,
   `grid_id` are *composed* from typed parts and stored; the string is a rendering
   (repo `shared/identifiers` doctrine). No string archaeology.
3. **Features are ORDERED.** `feature_values[:, j] ↔ feature_names[j]`; never an
   anonymous `N×D`.
4. **Features are the primitive; grids are built from them; axes stay in feature
   units.** No "canonical" basis, no coordinate_frame, no inverse to store —
   a metric read off a grid is already in feature units. Recompute-to-raw is a
   non-issue because nothing leaves feature space. Comparability: same
   `feature_names` → scalar-comparable; same `grid_id` → raster-comparable.
   Caller owns scientific validity (shared representation before splitting).
5. **`SampleSet` is the durable unit of membership, profiling, plotting, and
   comparison** (Distribution/Grid/LabelGroup are also durable records).
   `label_groups` is a derived dict; `LabelGroup` is a thin run-result. Do not
   promote the dict; do not special-case peaks. `sample_set_id` durable+composed,
   `sample_set_name` local+readable.
6. **Geometry is measured (on SampleSet), not provenance.** Provenance is *how
   made*; geometry/hdr/profile are *what it is*. Intrinsic-only on geometry
   (center/radius/r80/cv); run-relative scalars → `per_sample_set_metrics`.
   `unlabeled` ≠ NA ≠ abstention.
6b. **Assignment consistency (enforce centrally).** Membership exists twice —
   `LabelGroup.sample_id_to_sample_set_id` AND each `SampleSet.sample_ids` — so
   guard against drift: every assignment *value* ∈ `sample_set_ids`; every
   assignment *key* ∈ `Distribution.sample_ids`; assigned and
   `unassigned_sample_ids` are disjoint and together cover the Distribution;
   `SampleSet.sample_ids` agrees with the assignment map. (#10)
7. **A LabelGroup covers its Distribution.** Every sample assigned or in
   `unassigned_sample_ids`. `unassigned` is a field, never a SampleSet.
8. **Grain names are the ontology.** `per_sample_set` (sibling-relative) vs
   `across_sample_set` (≥2 sets, WITHIN one run). No relational-metric class zoo.
   Cross-distribution relations live at the comparison layer.
9. **Grid is a run artifact, built from features, feature-unit axes** (no
   canonical basis, no inverse). Shared-grid identity via deterministic `grid_id`
   (no registry). Comparability: `feature_names` equality (scalar) + `grid_id`
   equality (raster). Different grids/same features → re-evaluate on a shared grid.
10. **Compare LabelGroups, not naked SampleSet tuples.** Comparison keeps
    artifacts/provenance/across-metrics/unassigned; correspondence policy
    (`largest_reference`/`closest_center`/`matched_by_overlap`/`all_pairs`) picks
    the pairs. Sub-spec A peak-matching = a correspondence policy.

---

## Construction walk — grounded in real code (`valley_visualization.py`)

Verified against the actual flow in
`results/mcolon/20260617_morph_axis_investigation/valley_visualization.py`
(`load_bins` + `render_gene`). Every step maps to an object with no forcing.

**Grain (from the code):** `distribution_id = (scope_id, time_bin, role)`. A
Distribution is **single-(bin, role)** — the time bin is baked into identity, not
a live column. "Over time" is an outer loop that mints one Distribution per bin.

```
Step 0  Caller loads + bins → one row per (physical_embryo_id, time_bin).
        [load_bins: embryo-grain agg. Binning is UPSTREAM, not in the object.]

Step 1  CARVE by label column → one Distribution per (scope × time_bin × role).
        reference = zygosity == "wildtype"
        target    = phenotype_clean ∈ {CE, HTA}
        [The carve is the distribution BOUNDARY. Two distribution_ids coexist per
         bin. Role decided upstream. NATIVE feature values stored (feature units,
         no frame). Caller must have built a SHARED representation before this
         split (same feature_names on target & reference).]

Step 2  PROVIDED labelers → within-distribution LabelGroups.
        label(target, method="column", params={"column":"phenotype_clean"})
        → LabelGroup "phenotype" : SampleSets CE, HTA, unlabeled   (artifacts=None)
        [Same phenotype_clean column, used a SECOND time — now as a within-target
         label_group. This is the two-level use to keep distinct: carve ≠ label.]

Step 3  DERIVED labelers → within-distribution LabelGroups (run independently
        on target AND reference, matching the two _build_distribution_record calls).
        label(target,    method="peak_finding", params={spec, vote, seed})
        label(reference, method="peak_finding", params={spec, vote, seed})
        → LabelGroup "peak" : peak SampleSets (geometry filled) + unassigned
          provenance = vote/is_reliable;  artifacts = grid+density+basins (grid_id)
        [Both evaluated on ONE grid built from POOLED target+reference features
         (§1b) → same grid_id → directly raster-comparable. Two clusters "emerge"
         = "peak".sample_set_ids grows 1→2 across bins.]

Step 4  COMPARE label groups (not naked tuples — keeps provenance/artifacts/
        across-metrics/unassigned alive):
        compare_label_groups(
            reference_label_group = reference_peak_lg,
            target_label_group    = target_peak_lg,
            correspondence_spec   = "largest_reference" | "closest_center" |
                                    "matched_by_overlap" | "all_pairs",
        )
        [= the "REF READOUT vs WT" row. Sub-spec A peak-matching IS the
         correspondence policy. Guardrails: same feature_names (scalar) and same
         grid_id (raster). Values already in feature units — no frame needed.]

Step 5  LOOP over time_bins (render_gene's `for hpf in hpfs`), collect outputs.
        No DistributionSequenceView object built — the loop + (scope,role) naming
        IS the series. Reserve the name only (see below).
```

Two truths this walk locks:

- **The same column serves two levels** — `phenotype_clean` *carves* target vs
  reference (Step 1, distribution boundary) AND *labels* CE/HTA within target
  (Step 2, label_group). Kept distinct by which step consumes it.
- **Distribution is single-(bin, role)**; the shared per-bin grid (built from
  pooled features, §1b) lives in both peak runs' artifacts under one `grid_id`;
  values are in feature units, so Step 4 reports differences directly — no frame.

### Reserved (named, NOT built): `DistributionSequenceView`

"Over time" today = a loop. If the series ever needs to be addressable, it is a
lightweight **view** = *distribution_ids sharing `(scope_id, role)`, ordered by
`time_bin`* — never a new container, and never filename-sort as the temporal
model. Named now so future code doesn't assume otherwise.

## Resolved this pass (was TBD)
- **Feature-space / coordinate-frame** — RESOLVED. Features are the primitive;
  grids are built from them with feature-unit axes (§1b). No `FeatureSpace`
  object, no `coordinate_frame`, no "canonical" basis. Comparability = ordered
  `feature_names` equality (scalar) + `grid_id` equality (raster). Caller owns
  scientific validity (shared representation before splitting). No auto lineage
  check — matching names are taken as matching meaning, by design.

## TBD — not yet locked
- **Comparison** — `compare_label_groups(reference, target, correspondence_spec)`,
  roles at call time, correspondence policy picks the pairs (Sub-spec A peak
  matching = one policy). Values already in feature units. Genotype-vs-peak
  agreement (two total partitions — do they carve alike?) lives here.
- **`make_grid_id` / grid construction methods** — `pooled_min_max` /
  `pooled_quantile` / `pooled_mad_scaled` / `fixed_bounds`; all keep axes in
  feature units (decision (b)). Deterministic id from (names, method, params,
  fit_ids). Spec the exact hash inputs when building.
- **Plotting** — per-SampleSet HDR overlays, 1-D KDE strips, N-feature × M grid
  via the faceting engine (Sub-spec D). One row per `label_group` view.
- **Simple helpers** — `label_dataframe(df, ...)` front door + a dict/spec batch
  form. Built on the dumb objects.
