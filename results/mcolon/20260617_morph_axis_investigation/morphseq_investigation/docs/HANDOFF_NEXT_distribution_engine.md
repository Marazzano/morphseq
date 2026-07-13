# HANDOFF — Distribution Comparison & Plotting Engine

Read this first. It orients the next agent, points at the ONE source of truth,
warns which docs are stale, and lists the four next steps + what to verify.

## Source of truth (read in this order)

1. **`PRIMITIVE_ONTOLOGY.md`** — THE LOCKED spec. Everything below defers to it.
2. `DISTRIBUTION_COMPARISON_MOTIVATION.md` — the *why* (the biological/statistical
   goal: describe modal organization, not a discrete/continuous binary).
3. `DISTRIBUTION_ENGINE_UPGRADE_INDEX.md` — the map/index.

## Sub-specs were DELETED (on purpose)

An earlier pass wrote `SUBSPEC_A/B/C/D_*.md` against framings that were
**superseded** during the ontology work (dead vocab: `PeakMembership`,
`coordinate_frame`, "canonical" basis, `diff_vs_null`, `DistributionSeries`,
L0–L6 stack). They were **removed** rather than left as landmines — everything
load-bearing in them (A's peak-matching correspondence policies, D's
faceting-engine plotting plan, grid construction, helper front-doors) is already
folded into `PRIMITIVE_ONTOLOGY.md` in the correct vocabulary.

**You (next agent) decide the spec ordering** from `PRIMITIVE_ONTOLOGY.md` +
`DISTRIBUTION_COMPARISON_MOTIVATION.md`. The "four next steps" below are the
natural order, but confirm nothing is lost vs the ontology before you commit to
a sequence — re-derive, don't trust a stale ordering.

## The locked ontology in 6 lines

- **Distribution** — dumb native bag: `(scope_id, time_bin, role)` identity +
  ordered `feature_names`/`feature_values` (FEATURE units). No coordinate frame.
- **Feature → Grid → DensityGrid** — features are PRIMITIVE; a Grid is built FROM
  features with **feature-unit axes** (no "canonical", nothing to invert). A KDE
  strip is the 1-D case of a Grid.
- **SampleSet** — the durable atom: members + `geometry`/`hdr`/`feature_profile`
  (feature units, `grid_id`-tagged) + structured `provenance`.
- **LabelGroup** — one labeler RUN: assignments + `unassigned_sample_ids` (a
  field, NOT a SampleSet) + `provenance` + `artifacts`(grid/density/basins) +
  `per_sample_set_metrics` + `across_sample_set_metrics`.
- **label_groups** — a lightweight dict view `{name: (sample_set_ids,)}`.
- **Labelers are peers**: genotype/phenotype (column), peak_finding, dtw all map
  `label(distribution, method, params) → LabelGroup + [SampleSet]`.

Comparability: same `feature_names` → scalar-comparable; same `grid_id` →
raster-comparable. Caller owns scientific validity (shared representation before
splitting target/reference). Grounding: `valley_visualization.py` construction
walk is in the ontology doc.

## The b9d2 worked example (the thing we're building toward)

End goal, concrete: **"generate the b9d2 phenotype distributions over time vs
their controls, and watch the two clusters emerge."** This is the acceptance
target for the whole engine. Mapped to the ontology:

```
for each time_bin (e.g. 24/30/36/48 hpf):
  # Step 1 — carve two distributions by label column (upstream)
  reference = Distribution(scope_id="b9d2", time_bin=t, role="reference",
                           samples = zygosity == "wildtype")
  target    = Distribution(scope_id="b9d2", time_bin=t, role="target",
                           samples = phenotype_clean ∈ {CE, HTA})
    # caller must have built a SHARED feature representation BEFORE this split
    # (same feature_names on both)

  # Step 2 — provided labeler on target: phenotype (CE / HTA / unlabeled)
  # Step 3 — peak_finding labeler on target AND reference, on ONE pooled grid
  #          → target "peak" label_group; 1 mode early → 2 modes late = "emergence"
  # Step 4 — compare_label_groups(reference_peak_lg, target_peak_lg, policy)

collect across bins → feature-over-time plot of the peak SampleSets' profiles
  (peak count / centers drifting; CE vs HTA strips). NO DistributionSequenceView.
```

The live reference implementation of the OLD (pre-ontology) version is
`results/mcolon/20260617_morph_axis_investigation/valley_visualization.py`
(`load_bins` + `render_gene`) — grounding, not a template to copy.

### ⚠️ Data source: use UPDATED Snakemake, NOT the hand-cleaned CSV

`valley_visualization.py` today loads a **hand-cleaned CSV**
(`.../gene14/tables/reference_b9d2_clean.csv`). When you build the real b9d2
example, **source it from the updated Snakemake pipeline output**, not that
ad-hoc CSV. Rationale + status:

- The gene14 QC / Snakemake work is active — see MEMORY.md ("Gene14 QC Pipeline",
  "Snakemake Refactor / streamline-snakemake") and the `data_pipeline` tree.
  Confirm the current analysis-ready output path with the user before wiring it.
- The point: the b9d2 example should ride the maintained pipeline (correct
  binning, QC gates, latents, `physical_embryo_id`), so it stays reproducible and
  doesn't rot against a frozen CSV.
- Keep binning UPSTREAM regardless (reuse `bin_embryos_by_time`); the ontology
  objects never bin.
- **Status (per repo owner):** the Snakemake source is the *target*; for the
  immediate build it's fine to do an **ad-hoc column manipulation** to make the
  current columns work. Leave the worked example above unchanged; treat "wire to
  updated Snakemake" as a follow-up, not a blocker.

## The FOUR next steps (order is a suggestion — re-derive, see above)

### 1. Comparison layer — `compare_label_groups`
- `compare_label_groups(reference_lg, target_lg, correspondence_spec)`. Compare
  **LabelGroups, not naked SampleSet tuples** (keep artifacts/provenance/across-
  metrics/unassigned). Roles assigned at call time.
- `correspondence_spec ∈ {largest_reference, closest_center, matched_by_overlap,
  all_pairs}` — this IS Sub-spec A's peak-matching, re-expressed.
- Guardrails: assert same `feature_names` (scalar) / same `grid_id` (raster).
- Also here: genotype-vs-peak **agreement** (two total partitions of the same
  samples — do they carve alike? cross-tab).

### 2. `make_grid_id` + grid construction
- `build_grid(feature_names, pooled_values, method, params) → Grid`.
- Methods: `pooled_min_max` / `pooled_quantile` / `pooled_mad_scaled` /
  `fixed_bounds`. **Decision (b): axes stay in feature units** for all methods
  (whitening chooses bounds/spacing only, never a basis change).
- `grid_id` deterministic from `(feature_names, method, params, fit_sample_ids)`.
  Spec the exact hash inputs. Shared comparison grid = built from POOLED
  target+reference values; both peak runs reference that one `grid_id`.

### 3. Plotting (Sub-spec D, re-expressed) — via the faceting engine
- Emit IR for `src/analyze/viz/plotting/faceting_engine/` (`render()`,
  mpl+plotly). Do NOT fork it. There is NO `plot_distribution_feature_over_time`.
- **KDE strip = 1-D Grid over one feature + DensityGrid → a `TraceData` curve**;
  overlay = several SampleSets on the SAME 1-D `grid_id`; N features × M
  comparison = a `FigureData` grid. Strips get a wider aspect ratio.
- Gap: the faceting IR has no 2-D density-field panel → 2-D dual-distribution
  overlay is a fast-follow (`DensityFieldData` IR extension; honor the repo viz
  coordinate contract — see MEMORY.md "Matplotlib Visualization Contract").

### 4. Simple helpers (front door) — build LAST, on the dumb objects
- `label_dataframe(df, distribution_id, sample_id_column, feature_columns,
  label_group_name, method, params) → result`.
- Batch/spec form `label_dataframe_from_spec(df, spec)` producing the SAME
  objects (for Snakemake/config/high-throughput).
- Ordered `feature_columns` → `Distribution.feature_names`.

## Peak-fitting (already DESIGNED, needs implementing)

The existing `core/distribution_records.py::compute_resolved_peaks` +
`ResolvedPeakDistribution` + bootstrap vote **become the `peak_finding`
labeler**. Field-by-field destination table is in `PRIMITIVE_ONTOLOGY.md`
("Peak-finding as a labeler"). Key points: one SampleSet per accepted mode +
`unassigned` field; vote/is_reliable → LabelGroup.provenance; grid/density/basins
→ artifacts; PeakGeometry → SampleSet.geometry; drop the truth/empirical branch
(real data is always empirical).

## What to VERIFY before building

1. Re-read `PRIMITIVE_ONTOLOGY.md` end-to-end — it supersedes A/B/C/D.
2. Confirm the construction walk against **live** `valley_visualization.py`
   (`load_bins` + `render_gene`) — that grounding must stay true.
3. Check the real ID grammar in `src/data_pipeline/shared/identifiers/README.md`
   — `make_distribution_id`/`make_sample_set_id`/`make_grid_id` follow that
   doctrine (compose from typed parts, never parse strings back).
4. Confirm the faceting engine surface: `src/analyze/viz/plotting/
   faceting_engine/ir.py` (`FigureData`/`SubplotData`/`TraceData`/`FacetSpec`) +
   `src/analyze/viz/README.md`.
5. Binning stays UPSTREAM (caller/loader) — reuse `src/analyze/utils/binning.py::
   bin_embryos_by_time`; the objects never bin.

## Open design questions (not yet locked)

- Exact hash inputs for `make_grid_id` (must be stable + collision-safe).
- `across_sample_set_metrics` payload shape (`{pairwise:[...], summary:{...}}`) —
  which valley/organization metrics land in v0 vs later.
- Whether `label_group` alias resolution (`"peak"` → the peak-ish runs) is worth
  building now or deferred.
