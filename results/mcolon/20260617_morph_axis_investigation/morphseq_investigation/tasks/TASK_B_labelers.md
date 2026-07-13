# TASK B — Labelers (peak_finding + genotype)

**Prereq: TASK_0 merged. Soft-dep on TASK_A** (you need a real `grid_id`/Grid for
peak artifacts — stub against TASK_A's `build_grid`/`evaluate_density` signatures
and integrate once A lands). **This is the biggest task (~40%)** — it is mostly
*re-expression* of live peak code, not new science. Give it your full attention.

## Source of truth
`docs/PRIMITIVE_ONTOLOGY.md` — "Labelers are peers" table (line 312), the
"Peak-finding as a labeler" section + the **field-by-field destination table**
(lines 348–360), and §2 (SampleSet). Handoff step 3 + "Peak-fitting" section.

## The peer contract (the whole point — no `if genotype … else if peak …`)
A labeler is `label(distribution, method, params) -> (LabelGroup, [SampleSet])`.
Both labelers below run the SAME skeleton, differing only in which optional slots
fill. Prove it: the two share one return path and one `validate_label_group` call.

## Scope
`engine/labelers.py`:

### 1. `genotype` (provided / column) labeler — build this FIRST (it's tiny, proves the skeleton)
- `label(distribution, method="column", params={"column": ...})`.
- Reads the passed-in label column (the Distribution never parses ids — the column
  is handed in alongside, per §1). One `SampleSet` per category value.
- `geometry=None`, `provenance.labeler.feature_names=()`, `artifacts=None`,
  `unassigned_sample_ids=()` typically.
- **Three distinct beasts kept apart** (§2): `unlabeled` = a real SampleSet (literal
  category, `is_missing_value=False`); NA source → `is_missing_value=True`;
  labeler abstention → `unassigned_sample_ids`. Unknown genotype = a REAL SampleSet
  named `"unknown"` with its own HDR, NOT unassigned.

### 2. `peak_finding` (unsupervised) labeler — fold in the LIVE machinery
Re-express `core/distribution_records.py::compute_resolved_peaks` +
`ResolvedPeakDistribution` + the bootstrap vote. **Do not compete with it — map it.**
Use the destination table verbatim:

| `ResolvedPeakDistribution` field | New home |
|---|---|
| accepted `ResolvedPeak` (each) | one `SampleSet` |
| `PeakGeometry` (center/radius/cv/r80/support) | `SampleSet.geometry` (INTRINSIC only) |
| `sample_peak_ids` (positional) | join to real ids → `SampleSet.sample_ids` + `LabelGroup.sample_id_to_sample_set_id` |
| `PeakCandidateDetail` | `SampleSet.provenance.evidence` |
| per-peak basin validation | `SampleSet.provenance.evidence` |
| vote / `count_stability` / `is_reliable` | `LabelGroup.provenance` |
| `density_grid`, `basin_labels`, `detection_result` | `LabelGroup.artifacts` |
| `resolved_peak_count` | derived = `len(sample_set_ids)` |

Rules:
- **The vote decides count BEFORE carving** — two WT-clusters'-worth of points that
  vote to one mode → ONE SampleSet. Rejected candidates do NOT become phantom
  SampleSets; the collapse story stays in run provenance.
- Peak run evaluates on ONE grid built from POOLED target+reference features
  (§1b) → both target & reference peak runs reference the same `grid_id` → directly
  raster-comparable. (The pooling is the caller's; you accept a Grid or the pooled
  values + a Grid.)
- **support_fraction / prominence_rank / height_relative_to_max / is_dominant are
  RUN-RELATIVE** → `LabelGroup.per_sample_set_metrics`, NOT on geometry (§2, #4).
- **Drop the truth/empirical branch** — real data is always empirical
  (`source_type`, `grid_peak_ids` excluded from the public real-data ontology, #9).
  The synthetic benchmark path may keep them internally; that path is deferred.
- `ResolvedPeakDistribution` may survive as an internal builder, but the *returned*
  primitive is `LabelGroup + [SampleSet]`.

Both labelers call `engine.invariants.validate_label_group(...)` before returning.

## Out of scope
Grid construction internals (TASK_A owns `build_grid`/`evaluate_density`). Comparison
across runs (TASK_C). Plotting (TASK_D). `label_dataframe` front door (helpers, later).

## Tests (green before each commit)
`tests/engine/test_labelers.py`:
- `genotype`: category counts match; `unknown` is a SampleSet; NA →
  `is_missing_value=True`; coverage invariant holds.
- `peak_finding` on a clearly bimodal fixture → 2 SampleSets each with filled
  `geometry`; on unimodal → 1. `per_sample_set_metrics` populated, geometry has NO
  run-relative scalars. Artifacts carry grid/density/basins with a `grid_id`.
- Vote-collapse: a fixture where naive counting sees 3 but the vote resolves 2 →
  2 SampleSets, no phantoms.
- Peer-symmetry: assert both labelers return `(LabelGroup, list[SampleSet])` and
  pass the same `validate_label_group`.

## Commit checkpoints
- `⟢ COMMIT 1` — `genotype` column labeler + test green (skeleton proven).
  `dist-engine(task-b): genotype column labeler on the peer skeleton`
- `⟢ COMMIT 2` — `peak_finding` labeler mapping the live machinery + test green.
  `dist-engine(task-b): fold compute_resolved_peaks into peak_finding labeler`
- `⟢ COMMIT 3` — integrate against real TASK_A grids (drop stubs) + test green.
  `dist-engine(task-b): wire peak_finding to shared build_grid/evaluate_density`
- Open PR.

## Definition of done
Both labelers return the same shape through one skeleton; peak fields land per the
destination table; vote-before-carve honored; truth branch dropped; run-relative
metrics off geometry; three checkpoint commits; PR open.
