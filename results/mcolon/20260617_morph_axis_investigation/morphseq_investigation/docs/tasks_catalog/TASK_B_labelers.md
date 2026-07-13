# TASK B — Labelers as column writers + detect_peaks (a richer label group)

Parallel after TASK_0. The biggest task — you re-express the existing labelers so
they WRITE LABEL GROUPS onto a `Distribution` instead of returning free
`(LabelGroup, [SampleSet])` tuples. The 2-D peak machinery is reused, not rebuilt.

## THE ABSTRACTION (do not lose this — it was hard-won)
Detecting peaks does NOT create a new distribution and does NOT create a special
new object. **It creates a LABEL GROUP** — the same noun as `genotype` or
`phenotype` — whose SampleSets simply have MORE of their optional slots filled.
Per the existing `SampleSet` docstring: *"A genotype subset, a DTW cluster, and a
peak are the SAME type — they differ only in which optional slots are filled."*
So a detected `peak_0` is a `SampleSet` with `geometry`/`hdr` filled; a provided
`wildtype` is a `SampleSet` with those `None`. ONE type all the way down.

Where the "more information" lives, by grain (BOTH slots already exist in the spec):
- **per-category** (this peak's center / radius / r80 / HDR) → the `SampleSet`'s
  optional `geometry` / `hdr` slots.
- **per-run robustness** (bootstrap vote frequencies, count stability,
  `is_reliable`, the density field, basin labels) → the label group's
  `LabelGroupArtifacts` / provenance (§3, already in the spec).

`detect_peaks` is therefore the SIBLING of `with_label`: both return the
distribution carrying a new label group; they differ only in whether the derived
SampleSets have their optional slots filled. Detection = labeling with more
evidence.

## Source of truth
`docs/DISTRIBUTION_CATALOG_API.md`; the existing `SampleSet` /
`LabelGroupArtifacts` shapes in `engine/objects.py`; and the EXISTING
`engine/labelers.py` (`label_peak_finding`) which already bridges the live
`core.distribution_records` peak code. You are changing OUTPUT SHAPE, not the
peak algorithm.

## Scope — `engine/labelers.py` + `Distribution.detect_peaks`

### `detect_peaks` (implement the TASK_0 stub)
```python
Distribution.detect_peaks(
    self, *, features: Sequence[str], output_label: str = "resolved_peak",
    spec=DEFAULT_ANALYSIS_SPEC,          # DEFAULTED — users need not pass one
) -> Distribution                         # SAME population, now carrying the new label group
```
- Run the live 2-D peak machinery on `self`'s OWN points for `features` (grid
  built from this distribution's points — no pooled target/reference grid; the
  1-D plot re-grids per cell downstream — see spec).
- Attach a label group named `output_label`: per-sample assignment (peak id →
  "peak_k"; residual → `UNASSIGNED_LABEL`).
- The derived SampleSets carry **eager per-category geometry** (centers / radius /
  r80 / HDR) in their `geometry`/`hdr` slots, so `dist2.sample_sets("resolved_peak")`
  yields geometry-bearing sets — peak rings survive.
- **Per-run robustness** (vote frequencies, count stability, `is_reliable`,
  density field, basin labels) → the label group's `LabelGroupArtifacts`.
- Returns the distribution with the label group attached (new frozen object;
  original unchanged). NO `ModeDiscoveryResult` wrapper, NO new distribution.
  Because it is a clean `Distribution -> Distribution` function, TASK_A can expose
  a NAMED catalog convenience `catalog.detect_peaks(features=..., output_label=...)`
  that applies it to every distribution WITHOUT the caller writing a lambda. (The
  generic `map_distributions(fn)` escape hatch still exists for arbitrary one-off
  transforms; the named convenience is what callers use.) Keep this signature
  map-friendly: no reliance on catalog/coordinate state.
- `DEFAULT_ANALYSIS_SPEC` already exists in the live code (the old
  `label_peak_finding` uses it) — reuse it as the default.
- Peak ids are LOCAL to this distribution (spec: matching ≠ discovery). Do NOT
  imply cross-distribution peak_0↔peak_0 correspondence anywhere.

### Provided-column labeler → `Distribution.with_label` already covers it
The old `label_genotype(method="column")` becomes ordinary `with_label` (TASK_0).
If a thin `label_column_from_series` convenience is wanted, add it, but the column
path is just `with_label`. Confirm no caller still needs the old tuple return.

### Retire the old return shape
Delete/rewrite the parts of `labelers.py` that returned `(LabelGroup, [SampleSet])`
and the peer `_finalize` plumbing that only existed to build those free objects.
Keep the peak-bridge internals (`_grid_to_canonical`, HDR carving, the live
`compute_resolved_peaks` call) — reuse them.

## Preserved invariants (do NOT regress)
- Grid built from the distribution's own points; grid_id discipline intact.
- KDE for the ARTIFACT density is fine here; but the PLOT densities are still
  materialized downstream by TASK_C — do not try to precompute plot marginals.
- Geometry eager; everything frozen; `discover_modes` returns NEW objects.

## Stub you may lean on
TASK_0's `Distribution` + `with_label` + `sample_sets`. You do NOT need the
catalog — test `discover_modes` on a single hand-built `Distribution`.

## Tests — `tests/engine/test_labelers.py`
- `discover_modes` on a synthetic bimodal 2-D distribution: writes a
  `resolved_peak` column with the expected number of categories; residual →
  UNASSIGNED_LABEL.
- `distribution.sample_sets("resolved_peak")` returns sets WITH geometry
  (centers/HDR present) — eager-geometry proof.
- Provenance records method + features + spec.
- Returns a NEW Distribution (original has no `resolved_peak` column).
- Local-peak-id honesty: two independently-discovered distributions may both have
  `peak_0` and the API does not claim they correspond (assert no shared-id
  machinery exists).
- Regression: on the b9d2-like fixture, the peak COUNT per bin matches the old
  engine's counts (port the numbers from the existing worked example: 14hpf→1,
  later→2) — proves the algorithm is unchanged, only the output shape moved.

## Commit checkpoints
- `⟢ COMMIT 1` — `discover_modes` writing the `resolved_peak` column (calls only,
  geometry wired) + test green.
  `catalog(task-b): discover_modes writes a resolved_peak label column`
- `⟢ COMMIT 2` — eager geometry attached; `sample_sets` returns geometry; old
  tuple-return plumbing retired + test green.
  `catalog(task-b): eager peak geometry on the label column; retire tuple returns`
- `⟢ COMMIT 3` — regression test vs old peak counts green.
  `catalog(task-b): peak-count regression parity with the pre-migration engine`
- Open PR.

## Definition of done
`Distribution.discover_modes(...)` returns a `ModeDiscoveryResult` whose
`.distribution` carries a `resolved_peak` label column with eager geometry;
`sample_sets` yields geometry-bearing sets; peak counts match the old engine; no
`(LabelGroup,[SampleSet])` return shape remains; three checkpoint commits; PR open.
