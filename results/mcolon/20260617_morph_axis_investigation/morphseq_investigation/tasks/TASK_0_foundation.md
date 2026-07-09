# TASK 0 — Foundation objects (BLOCKING, SERIAL, DO FIRST)

**Nobody else starts until this is committed on `main`.** You are freezing the
object contract *and* the `grid_id` hash that Tasks A/B/C/D all key off. Get it
right; a change here after the fact ripples into four branches.

## Source of truth
`docs/PRIMITIVE_ONTOLOGY.md` §1, §1b, §2, §3, §4, and the Invariants list (#1–#10).
Copy the dataclass field lists from there **verbatim** — they are locked.

## Scope (what you build)

Create the package `morphseq_investigation/engine/`:

### `engine/objects.py` — the four nouns + the view
Frozen dataclasses, exactly as specced:
- `Distribution` (§1) — `distribution_id, scope_id, time_bin, role, sample_ids,
  feature_names, feature_values, scope_tag`. **No `coordinate_frame`.** Hard
  invariant `feature_values[:, j] ↔ feature_names[j]`; features ORDERED, never an
  anonymous N×D.
- `Grid`, `DensityGrid` (§1b) — axes IN FEATURE UNITS. `Grid` carries
  `grid_id, feature_names, axis_values, construction_method, construction_params,
  fit_sample_ids`.
- `SampleSet` + `SampleSetGeometry` + `HDR` + `FeatureProfile` (§2). Geometry holds
  **INTRINSIC only** (center/radius/r80/cv). Run-relative scalars do NOT go here.
  `sample_set_id` durable+composed, `sample_set_name` local+readable.
- `LabelGroup` + `LabelGroupArtifacts` (§3) — `sample_set_ids` are real groups
  only; `unassigned_sample_ids` is a FIELD, never a SampleSet.
- `label_groups` — just a type alias / a `derive_label_groups(label_groups_list)`
  helper returning `{name: (sample_set_ids,)}`. NOT a class. Strict alias
  resolution (#8): an alias resolves only if it identifies exactly ONE LabelGroup;
  ambiguous → raise.

Rules baked in:
- `@dataclass(frozen=True)` everywhere.
- **Never `{}`/`[]` as a default** — always `field(default_factory=...)` (#3 in doc).
- Make array fields read-only where the module already has a `_readonly_array`
  pattern (see `core/distribution_records.py:72`); reuse that idiom, don't reinvent.

### `engine/identifiers.py` — structured IDs (NEVER parsed back)
Follow `src/data_pipeline/shared/identifiers/README.md` doctrine: **compose from
typed parts, store the parts, never string-archaeology the result.**
- `make_distribution_id(scope_id, time_bin, role) -> str`
  e.g. `"b9d2_30hpf_reference"`.
- `make_sample_set_id(distribution_id, name) -> str`
  e.g. `"b9d2_30hpf_reference__peak_0"` (double-underscore separator so
  target/reference `peak_0`s never collide — see §2).
- **`make_grid_id(...)` — LOCK THE HASH HERE** (this is the design question the
  handoff left open, lines 162 / ontology 510; repo owner said lock it in Task 0).
  Hash inputs, per §1b:
  ```
  grid_id = stable_hash(
      ordered feature_names,
      construction_method,
      normalized construction_params,     # canonicalize: sort keys, round floats to
                                          # a fixed precision, stringify deterministically
      hash of the PRODUCED axis_values,   # the actual cell coordinates — NOT fit ids alone
  )
  # fit_sample_ids hashing MUST be order-independent: hash(tuple(sorted(fit_sample_ids)))
  # (fit ids feed into params/provenance, but axis_values is what makes id ⟺ coords)
  ```
  Property that MUST hold and MUST be tested: **`same grid_id ⟺ same evaluation
  coordinates`** (raster comparability). Same pool in a different row order → same
  id. Changed feature values under the same fit ids → different id. Use a stable
  hash (`hashlib.blake2b`/`sha256` over a canonical byte serialization; do NOT use
  Python's salted `hash()` for the id string). No stateful registry.

### `engine/invariants.py` — central guards (called by labelers, not by users)
- Coverage (#7): `⋃ SampleSet.sample_ids ∪ unassigned_sample_ids ==
  Distribution.sample_ids`, disjoint.
- Assignment consistency (#6b/#10): every assignment value ∈ `sample_set_ids`;
  every key ∈ `Distribution.sample_ids`; assigned ∩ unassigned = ∅;
  `SampleSet.sample_ids` agrees with the assignment map.
- Ordered-features (#3): `feature_values.shape[1] == len(feature_names)`.
- Grid/DensityGrid shape: `density.shape == tuple(len(a) for a in axis_values)`.
Expose one `validate_label_group(distribution, label_group, sample_sets)` that runs
all applicable checks and raises with a precise message. Labelers call it before
returning.

## Out of scope (belongs to later tasks — do NOT build)
- Any grid *construction* method (`build_grid`) → TASK_A.
- Any labeler → TASK_B.
- `compare_label_groups` → TASK_C.
- Any plotting → TASK_D.
Build the *shapes* and the *id/invariant machinery* only.

## Tests (must be green before every commit)
`tests/engine/test_objects.py`, `test_identifiers.py`, `test_invariants.py`:
- Construction of each object with minimal valid inputs.
- Frozen-ness (assignment raises).
- `feature_values[:, j] ↔ feature_names[j]` invariant enforced.
- `make_grid_id`: order-independence of `fit_sample_ids`; **different axis_values →
  different id**; identical inputs → identical id (the ⟺ property).
- Coverage + assignment-consistency guards: a hand-built valid group passes; a
  drifted one (assignment value not in sample_set_ids; overlapping assigned/
  unassigned) raises.
- Strict alias resolution: unique alias resolves; ambiguous raises.

## Commit checkpoints
- `⟢ COMMIT 1` — `engine/objects.py` + its test green.
  `dist-engine(task-0): freeze the four ontology objects + label_groups view`
- `⟢ COMMIT 2` — `engine/identifiers.py` (incl. locked `make_grid_id` hash) + test green.
  `dist-engine(task-0): lock make_*_id incl. deterministic grid_id hash`
- `⟢ COMMIT 3` — `engine/invariants.py` + test green, and the `tasks/README.md`
  layout confirmed/adjusted.
  `dist-engine(task-0): central coverage + assignment-consistency guards`
- Open PR. **Announce in the shared channel that TASK_0 is merged** — that is the
  green light for A/B/C/D.

## Definition of done
Frozen objects import cleanly; `make_grid_id` ⟺ property proven by test; invariant
guards catch drift; three checkpoint commits on the branch; PR open. No grid
construction, no labelers, no comparison, no plotting.
