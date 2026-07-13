# TASK C — compare_label_groups + correspondence policies

**Prereq: TASK_0 merged.** You import the objects. You can develop against
**hand-built fixture LabelGroups** (don't wait for TASK_B); integrate with real
peak LabelGroups once B lands.

## Source of truth
`docs/PRIMITIVE_ONTOLOGY.md` Invariant #10 + "TBD — Comparison". Handoff step 1
(lines 100–108). This IS the re-expressed Sub-spec A peak-matching.

## Scope
`engine/compare.py`:

### `compare_label_groups(reference_lg, target_lg, correspondence_spec) -> ComparisonResult`
- **Compare LabelGroups, NOT naked SampleSet tuples** (#10) — keep artifacts,
  provenance, across-metrics, and `unassigned` alive through the comparison.
- Roles are assigned AT CALL TIME (a LabelGroup is not intrinsically ref/target).
- `correspondence_spec ∈ {largest_reference, closest_center, matched_by_overlap,
  all_pairs}`:
  - `largest_reference` — pair each target set to the biggest reference set.
  - `closest_center` — pair by nearest `geometry.center` (feature units — no frame).
  - `matched_by_overlap` — pair by HDR/basin raster overlap (needs same `grid_id`).
  - `all_pairs` — full cross product (no reduction).
- **Guardrails (assert, fail loudly):**
  - scalar comparisons (centers/distances) require identical `feature_names`.
  - raster comparisons (overlap) require identical `grid_id`.
  - values are already in feature units — NO frame conversion anywhere.
- Emit per-correspondence metrics (center distance, overlap, valley depth between
  paired sets as available) and keep unmatched sets / `unassigned` visible in the
  result (don't silently drop them).

### Genotype-vs-peak agreement (also lives here)
- Two TOTAL partitions of the SAME samples (genotype LabelGroup vs peak LabelGroup)
  — do they carve alike? Produce a cross-tab (contingency of
  `sample_id → genotype set` vs `sample_id → peak set`) + a summary agreement
  scalar (e.g. adjusted Rand / normalized MI — pick one, document it).
- This is partition-vs-partition on one Distribution, distinct from the
  ref-vs-target correspondence above.

### `ComparisonResult` (define it here, small + typed)
Holds: the correspondence pairs, per-pair metrics, unmatched sets, the two source
`distribution_id`s, and the `correspondence_spec` used. Frozen dataclass, same
discipline as the ontology objects (`field(default_factory=...)`, no naked dict).

## Boundary note
`across_sample_set_metrics` on a LabelGroup is WITHIN one run (#8). This task is the
CROSS-run / cross-distribution layer. Do not stuff cross-run relations back into a
single LabelGroup.

## Out of scope
Grid construction (A), labelers (B), plotting (D). You consume finished LabelGroups.

## Tests (green before each commit)
`tests/engine/test_compare.py`:
- Each of the four correspondence specs on fixture LabelGroups with known geometry
  → expected pairing.
- Guardrail: mismatched `feature_names` → raises on a scalar comparison; mismatched
  `grid_id` → raises on `matched_by_overlap`.
- `unassigned` and unmatched sets survive into `ComparisonResult`.
- Agreement cross-tab on two partitions of the same samples: identical partitions →
  agreement scalar at its max; orthogonal partitions → near chance.

## Commit checkpoints
- `⟢ COMMIT 1` — `compare_label_groups` + four policies + guardrails + test green.
  `dist-engine(task-c): compare_label_groups with 4 correspondence policies`
- `⟢ COMMIT 2` — genotype-vs-peak agreement cross-tab + test green.
  `dist-engine(task-c): genotype-vs-peak partition agreement cross-tab`
- Open PR.

## Definition of done
LabelGroup-level comparison (not tuples); four policies; feature_names/grid_id
guardrails enforced; agreement cross-tab; `unassigned` preserved; two checkpoint
commits; PR open.
