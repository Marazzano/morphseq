# Per-Well Through-Line — Findings & North Star

**Status:** active thinking doc. Keep the goal pinned at the top; append findings below.
**Owner:** mdcolon
**Companion to:** `well_id_throughline_refactor_plan.md` (the formal scopes/plan).
**Verified against disk + Snakefile:** 2026-06-02.

---

## 🎯 THE GOAL (pin this — everything serves it)

> **Cross the experiment-grain bootstrap once; after the well list exists, run each well independently end-to-end.**
> The whole point of this refactor is to make a single well flow through the entire
> pipeline as an independent unit. Once that works, we map the file structure (`src/`
> and data output) *onto that shape* — the grain drives the layout, not the reverse.

Corollary goals that fall out of it:
- After the well list is known, **everything should be embarrassingly parallel** — one
  well per process, no cross-well barrier.
- The registry / layout should make it trivial to see, for any stage, **what artifacts
  it produces and at what grain.**

---

## 🔑 THE CENTRAL INSIGHT (the thing that reframes everything)

**Zone A cannot be per-well. That is physics, not a design flaw.**

You cannot "run well B01 end-to-end" starting from raw microscope data, because *at
that moment B01 does not yet exist as a known entity.* The raw ND2 / Keyence file is a
single blob of microscope **series**. The entire job of the front of the pipeline
(scope-metadata extraction → series→well mapping) **is the act of discovering which
wells are in the file.**

> **Well identity is an OUTPUT of the front stages, not an input.**

So the pipeline has a mandatory, irreducible **experiment-grain bootstrap prefix**
whose only job is to *manufacture the well list*. Per-well parallelism is only possible
*after* that prefix. The honest shape is:

```
  EXPERIMENT-GRAIN BOOTSTRAP            ──►   FAN POINT   ──►   PER-WELL, PARALLEL
  (process the whole experiment file          (the well        (light; one well = one
   to discover what wells exist)               list exists)      independent row, E2E)
```

This resolves the conflict that kept surfacing ("how do we make everything per-well
when some things clearly aren't?"). Answer: **don't.** Name the prefix for what it is
(a bootstrap that discovers wells), make the fan point explicit, and let `per_well` be
the contract for *everything after* it.

---

## 🌾 THE GRAIN TRANSITION (verified from the active Snakefile, 2026-06-02)

**Four zones** (stitching pulled out of bootstrap into its own image-materialization
zone). Transition points are sharp. Each stage is classified by two *independent* fields
— `fanout` (where files land) and `execution` (how many jobs) — defined in the
DESIGN PRINCIPLES section below.

### Zone A — EXPERIMENT-GRAIN BOOTSTRAP  (needs the raw blob before wells exist)
Family: `experiment_metadata/` · all stages `fanout=experiment`
| Stage | Output (under `{output_root}`) | Why it can't be per-well |
|---|---|---|
| `normalize_plate_metadata` | `experiment_metadata/{exp}/plate_metadata.csv` | the plate *is* the experiment |
| `extract_scope_metadata_{scope}` | `experiment_metadata/{exp}/scope_metadata__{scope}.csv` | reads the whole raw file |
| `map_series_to_wells` | `experiment_metadata/{exp}/series_well_mapping.csv` (+`.provenance.json`) | **this is where wells are discovered** |
| `apply_series_mapping` | `experiment_metadata/{exp}/scope_metadata_mapped.csv` (+`.validated`) | joins mapping onto scope rows |
| `build_frame_contract` | `experiment_metadata/{exp}/frame_contract.csv` | the canonical frame-level contract; feeds the checkpoint |
| `discover_wells` *(checkpoint)* | `experiment_metadata/{exp}/wells.txt` | filters contract wells → the canonical well list (global `well_id`) |

### ⟱ FAN POINT — `discover_wells` checkpoint (verified: Snakefile:427) ⟱
The well list becomes a **first-class artifact** here: `wells.txt`, keyed on global
`well_id`, emitted by the `discover_wells` Snakemake **checkpoint** (Snakefile:427),
which `wells_for_experiment()` expands the per-well DAG from. This — not
`series_well_mapping.csv` — is the true fan point. (Q1/Q3 RESOLVED.)
> ⚠️ **Two redundant well-discovery paths exist today** — flag for consolidation
> (plan Win 4/5): `_wells_from_mapping` (Snakefile:270, keyed on local `well_index`,
> used only by stitching) vs. the `discover_wells` checkpoint (keyed on global
> `well_id`). Collapse to one source of truth = the checkpoint.

### Zone B0 — IMAGE MATERIALIZATION  (per-well output, NO merge — see fanout-value note)
Family: `built_image_data/`
| Stage | Output | fanout / execution |
|---|---|---|
| `build_stitched_images` | `built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` | **per_well** output, **single** looping job |

**Why its own zone:** stitching is *not* discovery — it *consumes* the discovered
mapping to **materialize physical images** for downstream computation. That is a real
boundary (bootstrap answers "what exists?"; stitching answers "materialize it"). But it
is **well-addressable, not necessarily one-job-per-well**: the raw reader may prefer to
open the experiment blob once (conceptually one looping job — but that's an `execution`
concern, deferred, not a path concern).

> **⚠️ fanout-value wrinkle (stitching forces a decision).** Stitching writes per-well
> output **with NO merged table** (`built_image_data/{exp}/stitched_ff_images/{well}/`).
> The MVP decided two `fanout` values (`experiment` | `per_well_then_merge`) on the
> grounds that "every per-well stage merges" — but **stitching is the counterexample.**
> So either: (a) add the bare `per_well` value back (stitching is its real consumer →
> three values after all), or (b) treat stitched images as **not a registry artifact**
> (it's an image *tree*, not a CSV/parquet table — the registry may only govern tabular
> contract artifacts, and image dirs live outside it). **OPEN — decide when writing
> `paths.py`.** Leaning (b): the registry governs *contract tables*; image trees are a
> separate concern.

### Zone B — PER-WELL COMPUTATION  (per-well shards → merged table; registry `fanout=per_well_then_merge`)
Families today: `segmentation_and_tracking/`, `processed_snips/` (the only `per_well/`
that exists). Target adds: `computed_features/`, `quality_control/`, `analysis_ready/`.
| Stage | Per-well shard | Merged artifact (Zone C) |
|---|---|---|
| `segment_and_track_per_well` → `merge_segmentation_tracking` | `segmentation_and_tracking/{exp}/per_well/{well_id}/contracts/segmentation_tracking.csv` | `segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv` |
| `run_snip_processing_per_well` → `merge_snip_manifests` | `processed_snips/{exp}/per_well/{well_id}/contracts/snip_manifest.parquet` | `processed_snips/{exp}/contracts/snip_manifest.parquet` |
| *(Scope-5 target)* features / local QC / analysis_ready | `…/{exp}/per_well/{well_id}/…` | thin concat → Zone C |

Segmentation + snips are **already** the plan's Scope-5 pattern (per-well shard +
merge), built and working. Scope 5 extends this rhythm (`run_X_per_well → merge_X →
validate_X`) to features/QC/analysis_ready.

**Zone B target is CONFIRMED safe (mdcolon known fact + verified 2026-06-02):** every
features/QC computation is per-snip / per-embryo — **zero cross-well computation.** The
only thing making features/QC experiment-grain today is that they read the *merged*
Zone-B contract; **the merge is the sole barrier**, not any cohort statistic. (Empirical
check: the only percentile/quantile code is `build_sa_reference.py` — an **offline**
reference-curve builder, *not* a Snakefile stage — and `embryo_qc` percentiles are
per-embryo across Z-pairs. So there is **no cohort QC stage**.)

### Zone C — MERGE / PUBLICATION  (thin concat at the END of the DAG; NOT "cohort")
Families: `segmentation_and_tracking/`, `processed_snips/`, `computed_features/`,
`quality_control/`, `analysis_ready/` (each `{exp}/contracts|consolidated/…`).
- Pure **concatenation** of per-well shards into experiment-level publication tables.
  Layout: `family/{exp}/{contracts|consolidated}/{file}` (+ `.validated` sentinel).
- **It is *not* a cohort-computation zone** — verified above, nothing cross-well is
  computed. "Merge/publication," not "cohort." Moves to the **end** of the DAG so it
  stops acting as a mid-pipeline barrier.

---

## 🧭 DESIGN PRINCIPLES WE'VE LOCKED (from this discussion)

Orthogonal axes — keep them separate or the model gets muddy:

| Axis | Question | Lives in | MVP? |
|---|---|---|---|
| **identity** | "What is this object's canonical name?" | the **data** | `well_id = {exp}_{well}`, the through-line in *rows* AND the canonical post-fan path. |
| **layout** | "Where does this artifact live under the root?" | the **registry (code)** | `family / {exp} / [per_well/{well_id}/] [subfolder] / file`. A fixed contract. |
| **fanout** | "*Where do this stage's files land?*" | the **registry (per stage)** | ✅ MVP. `experiment` vs `per_well_then_merge` (two values — see contract below). |
| **execution** | "*How many jobs run this stage?*" | (future Snakefile) | ⛔ **DEFERRED** — documented concept, NOT an MVP field. `paths.py` never reads it. See below. |
| **grain** *(informational)* | "What is one row?" | (future) | ⛔ **DEFERRED** — `frame`·`snip`·`embryo`·`well`. Nothing reads it yet. |

### ⭐ `fanout` vs `execution` — the distinction (CONCEPT, not an MVP field)

The names both sound like "per well," but they answer **different questions** and are
**independent**. Keep the *insight* (it stops a future re-merge of the two); but
`execution` is **not a registry field yet** — nothing reads it, so it stays a doc note
(see Bloat Audit). This is the subtle hinge of the design, recorded so it isn't lost.

| | **`fanout`** | **`execution`** |
|---|---|---|
| Question | *Where do the output files land?* | *How many processes run the stage?* |
| About | the **artifact path** (DAG output shape) | the **job count** (scheduler work plan) |
| Decided by | the **data model** (is this a per-well thing?) | **practical compute/IO** (worth N jobs vs. 1 loop?) |
| Changes when | the **layout contract** changes | the **performance strategy** changes |
| Read by | `paths.py` (build the path) + the merge step | the Snakefile (`expand()` over wells vs. one rule) |

**Mnemonic:** `fanout` = *where the files go.* `execution` = *how many workers go.*

**Proof they're independent — all four combinations are real or reachable:**

| `fanout` | `execution` | Example | Meaning |
|---|---|---|---|
| `experiment` | `single` | scope_metadata; an experiment-level **plot** | one file, one job — plain experiment stage |
| `per_well` | `single` | **stitched images today** | files land per well, but **one job loops** all wells (open the raw blob once) |
| `per_well` | `per_well` | segmentation; features (post-Scope-5) | files land per well **and** one job per well (parallel, incremental rerun) |
| `experiment` | `per_well` | *(rare / none today)* | reachable but unusual — N workers → one experiment-grain output |

The `per_well` + `single` row (stitching) is the clincher: if the two were one concept,
that combination couldn't exist — yet it is exactly what stitching does. **A single
overloaded `fanout` enum would re-conflate them**, so when stitching later goes per-job
you'd change `fanout` and accidentally imply its *paths* moved. Two fields, two reasons
to change.

**Why this earns its keep (mdcolon):** adding, say, an experiment-level plot becomes
trivial and unambiguous — declare `fanout=experiment` and `paths.py` lands it at the
experiment level, with execution strategy a completely separate, later concern.

Pinned mantra:
> **Bootstrap is experiment-grain. After well discovery, computation is per-well.
> `fanout` says where files go; `execution` says how many workers go — keep them apart.
> Merged artifacts are publication products. Scoped runs are optional scratch, not
> architecture.**

Decisions adopted from the discussion:
- **Per-well-canonical is the target (DECISION, 2026-06-02).** The real output tree
  carries `{family}/{exp}/per_well/{well_id}/...` for *every* post-fan stage (Zones B
  and C alike). "Run one well" is then **not a special mode** — it is just targeting
  that one well's canonical outputs and letting Snakemake's per-file staleness do the
  rest. This **supersedes** the earlier "root-rebasing first" stance.
- **`scoped_runs/` is demoted to optional scratch/dev — NOT architecture.** Root
  rebasing (`config.yaml` `target_wells` → `{data_root}/scoped_runs/wells-.../`) was an
  *approximation* of per-well targeting, needed only while the main tree was
  experiment-grain. Once canonical `per_well/` exists, it is redundant. Keep it as a
  scratch lever; do **not** build the registry around it; revisit removal after the
  post-fan per-well conversion is complete.
- **Every post-fan stage follows the same rhythm:** `run_X_per_well → merge_X →
  validate_X`. Merged artifacts become thin **publication products** at the end of the
  DAG, not a mid-pipeline barrier. Snakemake then gives incremental single-well
  recompute for free.
- **Registry is two levels only:** `stage → artifact`. No deeper nesting. See the Lean
  MVP Contract below for the exact (small) field set.
- **One pipeline-wide registry** in `pipeline_orchestrator/lib/paths.py` (orchestration
  kingdom), imported by both the Snakefile and the Python entrypoints. Identity stays in
  `identifiers/` (separate kingdom).

---

## ✂️ BLOAT AUDIT — "a field must have a reader" (mdcolon, 2026-06-02)

The registry is a **path resolver — a tiny machine with sharp teeth — not the Grand
Unified Philosophy of MorphSeq.** Test every field by one question: *does `paths.py`
need it to resolve a path?* If not, it's a comment pretending to be a schema → demote to
a doc note, add the field the day a real consumer exists.

| Field | Reader for path resolution? | Verdict |
|---|---|---|
| `family` | ✅ first path segment | **KEEP** |
| `fanout` | ✅ decides `per_well/{well_id}/` branch | **KEEP** |
| `subfolder` | ✅ Zone C needs `mask_geometry/`, `consolidated/` | **KEEP** (optional per stage) |
| `artifacts` → filename | ✅ the filename | **KEEP** (templated; see microscope note) |
| `execution` | ❌ Snakefile-only, never a path | **CUT** → doc concept only |
| `grain` | ❌ informational, nothing reads it | **CUT** → doc note |
| `kind` (primary/sidecar/sentinel/report) | ❌ nothing branches on it | **CUT** → sentinels become *derived helpers* |
| `schema` | ❌ not for *pathing* (validators want it) | **CUT** for MVP → add when validators wired |

**Deferred fields, in priority order (each named with its eventual reader):**
1. **`src_family`** — *where the generating code lives* (e.g. output `computed_features`
   ← code `feature_extraction`; output `experiment_metadata` ← code `metadata_ingest`).
   Reader: humans + future `tasks.py` dispatch. **The "registry-as-index" field; least
   speculative.** NOTE: it is a *second* field, NOT a rename of `family` — output family
   and code family don't match 1:1, and `family` must stay the real output folder.
2. **`schema`** — reader: `validate_*` (Scope 2 / Win 3).
3. **`grain`** (frame/snip/embryo/well) — reader: none yet; informational.
4. **`execution`** (single/per_well) — reader: a future Snakefile job-count decision.

---

## 🧰 LEAN MVP CONTRACT for `lib/paths.py` (the spec to build)

**Per stage:** `family`, `fanout` ∈ {`experiment`, `per_well_then_merge`} (**two values
only** — every per-well stage on disk merges; ship a bare `per_well` value the day a
no-merge stage exists), optional `subfolder`.
**Per artifact:** a filename template (supports `{token}` for microscope variants).
**No `kind`/`schema`/`grain`/`execution` fields.** Sentinels/sidecars are *derived
helpers*, not rows.

**API (boring on purpose — explicit `artifact_scope`, no magical `well_id=None`):**
```python
ArtifactScope = Literal["experiment", "per_well", "merged"]
Fanout        = Literal["experiment", "per_well_then_merge"]

artifact_path(root, stage, artifact, experiment_id, *,
              artifact_scope="experiment",      # experiment | per_well | merged
              well_id=None,                     # required iff artifact_scope == per_well
              format_vars=None)                 # filename tokens, e.g. {"scope": "yx1"}

validated_path(*args, **kwargs)   # derived: artifact_path(...) + ".validated"
provenance_path(*args, **kwargs)  # derived: artifact_path(...) + ".provenance.json"
```

**Resolution rules (the whole engine):**
```
experiment                       → {root}/{family}/{exp}/{subfolder?}/{file}
per_well_then_merge + per_well   → {root}/{family}/{exp}/per_well/{well_id}/{subfolder?}/{file}
per_well_then_merge + merged     → {root}/{family}/{exp}/{subfolder?}/{file}
```
Validate loudly: `experiment` rejects `well_id`; `per_well` requires it; `merged`
rejects it. (Boring APIs age well.)

### 🔬 Microscope (`yx1`/`keyence`) — a Zone-A-only variant (RESOLVED, verified 2026-06-02)

The microscope is load-bearing for **exactly two front stages** and then **dissolves**:
- It appears as a **filename token in ONE artifact**: `scope_metadata__{scope}.csv`
  (YX1 vs Keyence). Handled by `format_vars={"scope": "yx1"}` — **do not** explode into
  fake per-microscope artifacts, and **do not** overload the word `scope` (that's the
  `artifact_scope` path-branch). Use `format_vars` (filename tokens) vs. `artifact_scope`
  (path branch) — two different ideas, two different argument names.
- It is a **code-dispatch** choice for extract + stitch (`build_stitched_images_yx1` vs.
  a Keyence rule) — a `src_family`/deferred-index concern, **NOT a path concern.**
- **Verified:** stitched-image *output* paths are microscope-agnostic
  (`built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` — no `yx1`/`keyence`),
  and `scope_metadata_mapped.csv` has already dropped the token. So the microscope
  **converges into canonical metadata at `apply_series_mapping` and never reappears as a
  path/filename after Zone A.** The registry needs **no** microscope-aware `fanout`.

---

## 🎯 THE TARGET MODEL (committed 2026-06-02 — Zone C confirmed per-well-able)

```
Zone A — experiment bootstrap        (fanout=experiment; discovers wells)
  experiment_metadata/{exp}/...      plate, scope, series_mapping, frame_contract, wells.txt

  ── FAN POINT: discover_wells checkpoint → wells.txt (global well_id) ──

Zone B0 — image materialization      (per-well image tree, NO merge; likely outside the registry)
  built_image_data/{exp}/stitched_ff_images/{well}/{channel}/   (one job loops wells)

Zone B — per-well canonical computation   (fanout=per_well_then_merge; one well = one unit, E2E)
  segmentation_and_tracking/{exp}/per_well/{well_id}/...
  processed_snips/{exp}/per_well/{well_id}/...
  computed_features/{exp}/per_well/{well_id}/...        ← NEW (was experiment-grain)
  quality_control/{exp}/per_well/{well_id}/...          ← NEW
  analysis_ready/{exp}/per_well/{well_id}/...           ← NEW

Zone C — merged / publication products    (thin concat at the END of the DAG; NOT cohort)
  segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv
  processed_snips/{exp}/contracts/snip_manifest.parquet
  computed_features/{exp}/consolidated/...
  quality_control/{exp}/consolidated/...
  analysis_ready/{exp}/analysis_ready.csv
```

- **Full run** = build all per-well outputs for all discovered wells, then build merged.
- **One-well run** = build per-well outputs for one `well_id` (merged optional/skipped).
  No special mode — just a different target.

**Path function — `well_id` lives in the canonical path, not a separate root:**
```python
artifact_path(output_root, stage="computed_features", artifact="mask_geometry",
              experiment_id=exp, well_id=well_id, scope="per_well")
#   → computed_features/{exp}/per_well/{well_id}/mask_geometry/mask_geometry_metrics.csv

artifact_path(output_root, stage="computed_features", artifact="mask_geometry",
              experiment_id=exp, scope="merged")
#   → computed_features/{exp}/mask_geometry/mask_geometry_metrics.csv
```
`scope ∈ {per_well, merged}` selects the branch; `fanout` in the registry declares which
stages legally support `per_well`.

---

## ❓ OPEN QUESTIONS (next session — don't lose these)

1. ~~**Where exactly is the fan point?**~~ **RESOLVED: the `discover_wells` checkpoint
   (Snakefile:427) → `wells.txt` (global `well_id`).** Stitching's earlier
   `_wells_from_mapping` iteration is a *separate, redundant* pre-fan path (local
   `well_index`) → consolidate to the checkpoint (Win 4/5).
2. ~~**Is Zone C truly per-well-able?**~~ **RESOLVED (mdcolon known fact + verified):
   100% per-snip/per-embryo, zero cross-well. No cohort QC stage exists
   (`build_sa_reference.py` is offline, not a Snakefile rule). Green light.**
3. ~~**Well list as first-class artifact?**~~ **RESOLVED: `wells.txt` from the checkpoint
   already is one (global `well_id`).** Cleanup = make it the *single* source;
   `selected_wells.txt` becomes a pure input filter (Win 5: checkpoint is truth, config
   only filters); `series_well_mapping.csv` stays raw discovery.
4. **Map `src/` onto the grain shape — light, and via the registry, later.** `src/` is
   already stage-organized; the "map" is the **registry itself** (`stage → family →
   fanout → execution → entrypoint`), NOT a directory move. **Principle: code location ≠
   pipeline stage** (e.g. YX1/Keyence stitching lives in microscope-specific folders but
   is one conceptual stage). Only light touch worth doing: standardize each stage's
   entrypoint behind a `tasks.py` verb (Win 2) so the entrypoint name matches its
   registry key. Do this *after* the registry exists.

### Remaining real unknowns (genuinely open)
- **Stitching `execution`:** stays `single` for now; revisit `per_well` only if it buys a
  real compute/staleness/debug win (depends on raw-reader IO behavior).
- **`scope_metadata__{scope}` variant dimension:** the microscope token (`__yx1`,
  `__keyence`) — **RESOLVED:** filename token via `format_vars`, converges out at
  `scope_metadata_mapped.csv` (see Microscope section above).
- **Stitching fanout value:** per-well-no-merge — add bare `per_well`, or keep image
  trees out of the registry entirely (leaning the latter). Decide when writing `paths.py`.
- **Scope-2 migration touch:** `discover_wells` reads `frame_contract.csv`'s `well_index`
  column (Snakefile:448), which Scope 2 deletes → this checkpoint is a migration site.

---

## ⏸️ WHERE WE PAUSED (2026-06-02)

The grain model is **settled and committed**: four zones — A (experiment bootstrap,
discovers wells) → **fan = `discover_wells` checkpoint** → B0 (image materialization,
per-well image tree, likely outside the registry) → B (per-well canonical,
`per_well_then_merge`) → C (thin merged publication, not cohort). The registry is trimmed
to a **lean path resolver** (Bloat Audit): per-stage `family`/`fanout`/optional
`subfolder`, per-artifact filename template; sentinels are derived helpers;
`execution`/`grain`/`schema`/`kind`/`src_family` are deferred (each named with its future
reader). The `fanout` vs `execution` distinction stays as a *documented concept*, not a
field. Zone-C per-well-safety and
the absence of any cohort stage are verified facts, not assumptions; scoped_runs is
demoted to scratch. **All four original open questions are resolved.** Next: write the
concrete `lib/paths.py` registry against this model (stage→artifact, with
per-stage `family`/`fanout`/optional `subfolder` and per-artifact filename templates
(see Lean MVP Contract).
