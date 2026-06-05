# Per-Well Through-Line — Findings & North Star

**Status:** active thinking doc. Keep the goal pinned at the top; append findings below.
**Owner:** mdcolon
**Companion to:** `well_id_throughline_refactor_plan.md` (the formal scopes/plan).
**Verified against disk + Snakefile:** 2026-06-02.

> ## ⚠️ READ THIS FIRST — CURRENT vs TARGET
> This doc describes **two architectures** and they must never be confused:
> - **CURRENT** = the verified shape of the active Snakefile *today* (tagged 🔵 CURRENT).
> - **TARGET** = the architecture we're refactoring *toward* (tagged 🟢 TARGET).
>
> Where a fact is "verified," it describes **CURRENT** unless tagged 🟢. The refactor's
> whole job is to move CURRENT → TARGET. A statement that blends them is a bug in this doc.
>
> **Registry scope (decided, applies to all TARGET sections):** `paths.py` registers
> **tabular contract artifacts only** (`.csv`/`.parquet` + their derived sentinels).
> **Image directory trees** (`stitched_ff_images/`) are **outside the registry** — they
> have their own path logic. This is why `fanout` has only two values (see Bloat Audit).
>
> **Document map:** Goal → Central Insight → Grain Transition (🔵 current zones, in
> execution order: A-metadata → B0 stitching → frame_contract → fan → B → C) →
> Design Principles → Bloat Audit → Lean MVP Contract + Worked Example (🟢 paths.py spec)
> → Well-Runner (🟢) → DAG Mechanics → Zone-A Narrowing (🟢 the frame-contract split) →
> Target Model → Open Questions → Pause/Next-steps.
>
> **Companion target docs:** `target/front_end_naming_and_flow.md` (front-end ingest +
> fan detail) and `target/stitched_handoff_contract.md` (the stitched **drop-in** input
> contract — tree layout + frame-contract columns + strict entry gate for outside datasets;
> standardizes the per-frame axis as `time_index`, the T dimension).

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

## 🌾 THE GRAIN TRANSITION — 🔵 CURRENT (verified from the active Snakefile, 2026-06-02)

**This section describes the VERIFIED CURRENT Snakefile.** The frame-contract split and
the earlier fan point are 🟢 TARGET — see "Zone-A Narrowing" below. Here, four zones.

> **🔵 CURRENT ordering (critical — the fan sits LATE):** in the active Snakefile,
> stitching (Zone B0) runs **before** the fan, because `build_frame_contract` consumes the
> stitched inventory and the `discover_wells` checkpoint reads `frame_contract.csv`. So the
> CURRENT order is: **A-metadata → B0 stitching → frame_contract → ⟱ FAN ⟱ → B → C.**
> (🟢 TARGET inverts this: the fan moves up to right after raw discovery, and stitching
> moves *below* the fan — see Zone-A Narrowing.) The zones below are presented in CURRENT
> execution order.

### Zone A — EXPERIMENT-GRAIN BOOTSTRAP  (needs the raw blob before wells exist) 🔵
Family: `experiment_metadata/` · all stages `fanout=experiment`
| Stage | Output (under `{output_root}`) | Why it can't be per-well |
|---|---|---|
| `normalize_plate_metadata` | `experiment_metadata/{exp}/plate_metadata.csv` | the plate *is* the experiment |
| `extract_scope_metadata_{scope}` | `experiment_metadata/{exp}/scope_metadata__{scope}.csv` | reads the whole raw file |
| `map_series_to_wells` | `experiment_metadata/{exp}/series_well_mapping.csv` (+`.provenance.json`) | **raw discovery — wells' identities first appear here** |
| `apply_series_mapping` | `experiment_metadata/{exp}/scope_metadata_mapped.csv` (+`.validated`) | joins mapping onto scope rows |

### Zone B0 — IMAGE MATERIALIZATION  (🔵 CURRENT: runs BEFORE the fan — per-well output, NO merge)
Family: `built_image_data/`
| Stage | Output | fanout / execution |
|---|---|---|
| `build_stitched_images` | `built_image_data/{exp}/stitched_ff_images/{well_index}/{channel}/` | **per_well** output, **single** looping job (🔵 CURRENT keys on local `well_index`; 🟢 TARGET → `well_id`, see below) |

> **Conceptual shape ≠ registry fanout.** Stitching's *conceptual* output shape is
> per-well-addressable (one dir per well). But this is **not** a registry `fanout` value —
> the registry governs tabular contract artifacts only, and (leaning (b) below) image trees
> are off-registry, so they carry no `fanout` at all. "per_well output" here describes the
> directory layout, not a registry field.

**Why its own zone:** stitching is *not* discovery — it *consumes* the discovered
mapping to **materialize physical images** for downstream computation. That is a real
boundary (bootstrap answers "what exists?"; stitching answers "materialize it"). But it
is **well-addressable, not necessarily one-job-per-well**: the raw reader may prefer to
open the experiment blob once (conceptually one looping job — but that's an `execution`
concern, deferred, not a path concern).

> **🔵 CURRENT placement:** stitching sits **upstream of the fan** today because
> `build_frame_contract` reads the stitched inventory. 🟢 TARGET moves it *below* the fan
> (`stitch_well`, per discovered well) — see Zone-A Narrowing. Its per-well-no-merge output
> shape is unchanged; only its DAG position relative to the fan moves.

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

### Zone A (cont.) — FRAME CONTRACT + FAN  (🔵 CURRENT: after stitching)
| Stage | Output (under `{output_root}`) | Role |
|---|---|---|
| `build_frame_contract` | `experiment_metadata/{exp}/frame_contract.csv` | 🔵 **CURRENT** canonical frame-level contract; consumes stitched inventory; feeds the checkpoint. 🟢 TARGET splits this per-well (Zone-A Narrowing). |
| `discover_wells` *(checkpoint)* | `experiment_metadata/{exp}/wells.txt` | filters contract wells → the canonical well list (global `well_id`) |

### ⟱ FAN POINT ⟱
> **Two distinct events — don't merge them:** (a) **raw discovery** = the well *identities*
> first appear as rows (`map_series_to_wells` writes them into `series_well_mapping.csv`);
> (b) **checkpoint materialization** = the `discover_wells` checkpoint emits `wells.txt`,
> which is what Snakemake re-plans the DAG on. The *fan* is (b). Moving the fan "earlier"
> means feeding the checkpoint a discovery-stage input, not adding a new discovery.

- **🔵 CURRENT:** `discover_wells` checkpoint reads **`frame_contract.csv`** (Snakefile:427)
  → materializes `wells.txt`. So the fan currently sits *after* stitching + frame-contract
  build, even though the identities were discovered upstream at `map_series_to_wells`.
- **🟢 TARGET:** `discover_wells_from_metadata` reads **`scope_metadata_mapped.csv`** (the
  canonical joined metadata; #13) → materializes `wells.txt` earlier (before stitching). Same
  checkpoint mechanism, earlier input — the fan moves up to sit right after the metadata is
  mapped (still well before stitching).
  (See Zone-A Narrowing.)

The well list is a **first-class artifact** either way: `wells.txt`, keyed on global
`well_id`, emitted by the `discover_wells` Snakemake **checkpoint** (Snakefile:427),
which `wells_for_experiment()` expands the per-well DAG from. **The fan point is the
checkpoint output that *materializes* `wells.txt`** — the act that lets Snakemake re-plan
the DAG over the discovered wells. This is distinct from where well *identities are first
discovered* in the raw mapping: `series_well_mapping.csv` already names the wells (raw
discovery), but the **checkpoint materialization** is what the DAG fans on. Raw mapping =
discovery; checkpoint = the materialized fan. (Q1/Q3 RESOLVED.)
> ⚠️ **Two redundant well-discovery paths exist today** — flag for consolidation
> (plan Win 4/5): `_wells_from_mapping` (Snakefile:270, keyed on local `well_index`,
> used only by stitching) vs. the `discover_wells` checkpoint (keyed on global
> `well_id`). Collapse to one source of truth = the checkpoint.

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

**Zone B target — two separate claims, don't conflate them:**
- **MATH: per-well (owner-confirmed + spot-checked).** mdcolon states as known fact that
  every features/QC computation is per-snip / per-embryo — no cohort statistic. Spot-check
  agrees: the only percentile/quantile code is `build_sa_reference.py` (an **offline**
  reference-curve builder, *not* a Snakefile stage) and `embryo_qc` percentiles are
  per-embryo across Z-pairs → **no cohort QC stage exists.** *(Not a full grep-audit; an
  appendix table of module/searched-for/result would upgrade this from "confirmed by owner
  + spot-check" to "exhaustively verified.")*
- **WIRING: not yet per-well.** Several rules currently read the *merged* Zone-B contract
  (e.g. `surface_area_qc` reads `consolidated_features`) — accidental merge walls. The math
  doesn't need cross-well data, but the **wiring** must be converted to per-well shard
  inputs (Scope-5 work). "Zero cross-well math" ≠ "zero rules read merged inputs."

So: the merge is the **only** barrier (no cohort math blocks the split), and the remaining
work is **rewiring**, not rewriting algorithms. (Scope note: "zero cross-well math"
describes the pipeline **as it exists today** — there is no cohort stage to design around,
so this model doesn't carry one.)

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
| **identity** | "What is this object's canonical name?" | the **data** | `well_id = {exp}_{well}`, the through-line in *rows* AND the canonical post-fan path of **registered** artifacts. |
| **layout** | "Where does this *registered* artifact live?" | the **registry (code)** | `family / {exp} / [per_well/{well_id}/] [subfolder] / file`. A fixed contract. |
| **fanout** | "*Where do this stage's REGISTERED (tabular) artifacts land?*" | the **registry (per stage)** | ✅ MVP. `experiment` vs `per_well_then_merge` (two values — image trees are off-registry, see contract). |
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
  carries `{family}/{exp}/per_well/{well_id}/...` for *every registered (tabular) post-fan
  stage* (Zones B and C alike). **(Stitching's image tree is off-registry** — separate path
  helper, no `STAGES` row — but it now **also keys on `well_id`**: `stitched_ff_images/{well_id}/`,
  not local `well`. REVISED 2026-06-03; see Zone B0 and `target/front_end_naming_and_flow.md`.)
  "Run one well" is then **not a special mode** — it is just targeting
  that one well's canonical outputs and letting Snakemake's per-file staleness do the
  rest. This **supersedes** the earlier "root-rebasing first" stance.
- **`scoped_runs/` is demoted to optional scratch/dev — NOT architecture.** Root
  rebasing (`config.yaml` `target_wells` → `{data_root}/scoped_runs/wells-.../`) was an
  *approximation* of per-well targeting, needed only while the main tree was
  experiment-grain. Once canonical `per_well/` exists, it is redundant. Keep it as a
  scratch lever; do **not** build the registry around it; revisit removal after the
  post-fan per-well conversion is complete.
- **Every registered post-fan tabular stage follows the same rhythm:** `run_X_per_well →
  merge_X → validate_merged_X`. Merged artifacts become thin **publication products** at the
  end of the DAG, not a mid-pipeline barrier. Snakemake then gives incremental single-well
  recompute for free. (The rhythm is for *registered tabular* stages; off-registry image
  trees like stitching don't merge and don't follow it.)

  > **"validate" means three different things (#11) — keep them distinct:**
  > 1. **per-well validation gate** — a per-well check (`validate_frame_contract_well`:
  >    does this well's metadata align with its images? QC flag computation). On the spine.
  > 2. **merged-file validation** (`validate_merged_X`) — structural check that the
  >    concatenated file is well-formed; writes the `.validated` sentinel. Off-spine.
  > 3. **algorithmic QC** — produces QC *flags* (data), not just a sentinel.
  >
  > "Merge = concat + validate" specifically means **concat + (2) merged-file validation.**
  > The frame/image *alignment* check is **(1)**, and it is per-well — not part of the merge.
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

## 🧰 LEAN MVP CONTRACT for `lib/paths.py` (the spec to build) — 🟢 TARGET

> **SCOPE BOUNDARY (decided):** `paths.py` registers **tabular contract artifacts only**
> (`.csv`/`.parquet` + their derived `.validated`/`.provenance.json` sentinels). It does
> **NOT** register **image directory trees** (`stitched_ff_images/`) — those have their
> own path logic. This is *why* `fanout` needs only two values: the one per-well-no-merge
> output (stitched images) is off-registry.

> **Stage keys = executable verbs, not families (decided, #13):** a registry stage key
> maps to a rule/`tasks.py` verb (`mask_geometry`, `pose_kinematics`, `consolidated_features`),
> **not** to a top-level family (`computed_features`). `family` is a *field*; many stages
> share one family. This keeps stage keys aligned with `tasks.py` verbs (Win 2).

> **`well_id` everywhere in paths (REVISED 2026-06-03, supersedes the earlier split).**
> `well_id` (`{exp}_{well}`, global) is the canonical key for **all** per-well paths — both
> **registered** artifacts (`per_well/20250912_B01/…`) **and** off-registry **image trees**
> (`stitched_ff_images/20250912_B01/…`). `well_index` (local, `B01`) survives only as a
> **column** in the scope-metadata tables; it is promoted to `well_id` once, for free, at
> `discover_wells` (the experiment is already known: `well_id = f"{exp}_{well_index}"`).
> **This reverses the earlier "image-tree = local" decision** (which had image trees mirror
> the microscope/plate layout): the `{exp}/` parent already disambiguates, and keying the
> addressable unit on a globally-unique id is the whole spirit of the per-well spine. The
> image-building change is a small opaque-string substitution (verified).
> **See `target/front_end_naming_and_flow.md` → "The canonical well key" for the full
> rationale, pros/cons, and blast-radius verification.**

**Per stage:** `family`, `fanout` ∈ {`experiment`, `per_well_then_merge`} (**two values
only** — every *registered* per-well stage merges; image trees are off-registry; ship a
bare `per_well` value the day a registered no-merge stage exists), optional `subfolder`.
**Per artifact:** a filename template (supports `{token}` for microscope variants).
**No `kind`/`schema`/`grain`/`execution` fields.** Sentinels/sidecars are *derived
helpers*, not rows.

**API (boring on purpose — explicit `path_mode`, no magical `well_id=None`):**
```python
PathMode = Literal["experiment", "per_well", "merged"]
Fanout   = Literal["experiment", "per_well_then_merge"]

artifact_path(root, stage, artifact, experiment_id, *,
              path_mode="experiment",      # experiment | per_well | merged (the path branch)
              well_id=None,                # required iff path_mode == per_well
              format_vars=None)            # filename tokens, e.g. {"scope": "yx1"}

validated_path(*args, **kwargs)   # derived: artifact_path(...) + ".validated"
provenance_path(*args, **kwargs)  # derived: artifact_path(...) + ".provenance.json"
```
> **`path_mode`, not `artifact_scope`** — avoids colliding with microscope-`scope`
> (`format_vars={"scope":"yx1"}`). `path_mode` selects the *path branch*; `format_vars`
> fills *filename tokens*. Two different ideas, two different arg names.

**Resolution rules (the whole engine):**
```
experiment                       → {root}/{family}/{exp}/{subfolder?}/{file}
per_well_then_merge + per_well   → {root}/{family}/{exp}/per_well/{well_id}/{subfolder?}/{file}
per_well_then_merge + merged     → {root}/{family}/{exp}/{subfolder?}/{file}
```
**The resolver MUST enforce `fanout` — this is what makes the registry a contract, not
"an f-string generator with delusions of authority."** A field with no reader shouldn't
exist (bloat audit); enforcing `fanout` is what *earns it its place*:
- `fanout=experiment` → only `path_mode="experiment"`; rejects `well_id`.
- `fanout=per_well_then_merge` → `path_mode` ∈ {`per_well` (requires `well_id`), `merged`
  (rejects `well_id`)}.
- Without this check, `artifact_path("plate_metadata", path_mode="per_well", ...)` would
  hallucinate a `per_well/` path the registry said can't exist. (See Worked Example.)

**Error & `format_vars` behavior (specified, #15) — the resolver fails loud, never guesses:**
- **Unknown `stage`** → raise `KeyError`-with-message (`f"unknown stage {stage!r}; known: {sorted(STAGES)}"`),
  **not** a bare `KeyError`. The registry is the closed set of legal stages.
- **Unknown `artifact`** for a known stage → raise with the stage's available artifact keys
  in the message (`f"{stage!r} has no artifact {artifact!r}; known: {sorted(spec['artifacts'])}"`).
- **`format_vars`:**
  - A template with `{token}`s **requires** those tokens in `format_vars` — a missing token
    raises (Python `str.format` raises `KeyError`; wrap it to name the artifact + missing
    token, don't leak a bare `KeyError`).
  - **Extra/unused** keys in `format_vars` are **rejected** (raise) — silently ignoring them
    hides typos (`{"scoep": "yx1"}` must fail, not no-op).
  - A template with **no** `{token}`s called **with** `format_vars` → raise (the artifact
    takes no tokens; passing them is a caller error).
  - `format_vars` fills **filename tokens only** — it never affects the path branch (that's
    `path_mode`). The two are orthogonal by construction.
- **MVP non-goal:** the resolver does **not** validate that `well_id` is well-formed or that
  it exists in `discovered_wells.txt` — see the MVP-validation note below. It validates *fanout/path_mode
  consistency* and *registry membership*, nothing about identity content.

> **`well_id` validation in `paths.py` (decided, #16) — MVP does NOT validate it.** `paths.py`
> treats `well_id` as an **opaque string** to slot into the path; it does *not* check the
> `{exp}_{well}` shape and does *not* check membership in `discovered_wells.txt`. **Why deliberate, not
> lazy:** (1) **kingdom boundary** — well-formedness is the `identifiers/` kingdom's job (its
> validators), and `paths.py` (orchestration) must not duplicate identity logic; (2)
> **existence** is the **checkpoint/well-runner's** job — `run_wells` only ever feeds
> resolved `well_id`s that came *from* `discovered_wells.txt`, so by construction the resolver never sees
> a non-existent well on the sanctioned path. So the *callers* guarantee a valid `well_id`;
> the resolver just places it. **If** we later want belt-and-suspenders, the hook is explicit:
> `paths.py` would *import* `identifiers.validators.is_well_id()` (never re-implement it) and
> call it at the top of the `per_well` branch. Not in the MVP — recorded so the boundary
> stays clean.

### 🔬 Microscope (`yx1`/`keyence`) — a Zone-A-only variant (RESOLVED, verified 2026-06-02)

The microscope is load-bearing for **exactly two front stages** and then **dissolves**:
- It appears as a **filename token in ONE artifact**: `scope_metadata__{scope}.csv`
  (YX1 vs Keyence). Handled by `format_vars={"scope": "yx1"}` — **do not** explode into
  fake per-microscope artifacts, and **do not** overload the word `scope` (that's why the
  path-branch arg is `path_mode`, not `scope`). Use `format_vars` (filename tokens) vs.
  `path_mode` (path branch) — two different ideas, two different argument names.
- It is a **code-dispatch** choice for extract + stitch (`build_stitched_images_yx1` vs.
  a Keyence rule) — a `src_family`/deferred-index concern, **NOT a path concern.**
- **Verified:** stitched-image *output* paths are microscope-agnostic
  (`built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` — no `yx1`/`keyence`; the
  `{well}` slot is local `well_index` today, → `well_id` in TARGET, but either way no scope
  token), and `scope_metadata_mapped.csv` has already dropped the token. So the microscope
  **converges into canonical metadata at `apply_series_mapping` and never reappears as a
  path/filename after Zone A.** The registry needs **no** microscope-aware `fanout`.

---

## 📐 WORKED EXAMPLE — `paths.py` end-to-end (the spec to build against)

This is the concrete walkthrough of how the pipeline interacts with the registry, on
**real** stages. It shows all three sides: **registry row → Snakefile `output:` →
entrypoint call → resolved path.** Hardened per the fanout-enforcement critique.

### The registry (written once, in `lib/paths.py`)
```python
STAGES = {
    "plate_metadata": {                          # experiment-grain (the real first rule)
        "family": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {"csv": "plate_metadata.csv"},
    },
    "scope_metadata": {                          # experiment-grain + microscope token
        "family": "experiment_metadata",
        "fanout": "experiment",
        "artifacts": {
            "raw":    "scope_metadata__{scope}.csv",   # {scope} → format_vars
            "mapped": "scope_metadata_mapped.csv",
        },
    },
    "mask_geometry": {                           # per-well (Zone B)
        "family": "computed_features",
        "fanout": "per_well_then_merge",
        "subfolder": "mask_geometry",
        "artifacts": {"metrics": "mask_geometry_metrics.csv"},
    },
}
```

### The engine (enforces `fanout` + loud registry/format errors, #15)
```python
def _fill(template, artifact, format_vars):
    """Fill filename tokens; reject missing AND extra keys (no silent guessing, #15)."""
    needed = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    given  = set(format_vars or {})
    if missing := needed - given:
        raise ValueError(f"{artifact}: missing format_vars {sorted(missing)}")
    if extra := given - needed:
        raise ValueError(f"{artifact}: unexpected format_vars {sorted(extra)}")
    return template.format(**(format_vars or {}))

def artifact_path(root, stage, artifact, experiment_id, *,
                  path_mode="experiment", well_id=None, format_vars=None):
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    spec = STAGES[stage]
    if artifact not in spec["artifacts"]:
        raise KeyError(f"{stage!r} has no artifact {artifact!r}; "
                       f"known: {sorted(spec['artifacts'])}")
    fanout   = spec["fanout"]
    filename = _fill(spec["artifacts"][artifact], f"{stage}.{artifact}", format_vars)
    base     = Path(root) / spec["family"] / experiment_id

    if fanout == "experiment":
        if path_mode != "experiment":
            raise ValueError(f"{stage}.{artifact} is experiment-grain")
        if well_id is not None:
            raise ValueError(f"{stage}.{artifact} does not accept well_id")
    elif fanout == "per_well_then_merge":
        if path_mode == "per_well":
            if well_id is None:
                raise ValueError(f"{stage}.{artifact} requires well_id")
            base = base / "per_well" / well_id
        elif path_mode == "merged":
            if well_id is not None:
                raise ValueError(f"{stage}.{artifact} merged output rejects well_id")
        else:
            raise ValueError(f"{stage}.{artifact} supports path_mode per_well|merged")
    else:
        raise ValueError(f"Unknown fanout: {fanout}")

    if "subfolder" in spec:
        base = base / spec["subfolder"]
    return base / filename
```

### Case 1 — experiment-grain (`plate_metadata`)
```python
artifact_path(ROOT, "plate_metadata", "csv", "20250912")
#   → {ROOT}/experiment_metadata/20250912/plate_metadata.csv   (byte-identical to today)
```
**Snakefile** — `output:` becomes a call, not a typed string:
```python
from data_pipeline.pipeline_orchestrator.lib.paths import artifact_path
rule normalize_plate_metadata:
    output:
        csv = lambda wc: artifact_path(DATA_ROOT, "plate_metadata", "csv", wc.experiment)
```

### Case 2 — microscope token (`format_vars`, NOT exploded artifacts)
```python
artifact_path(ROOT, "scope_metadata", "raw", "20250912", format_vars={"scope": "yx1"})
#   → {ROOT}/experiment_metadata/20250912/scope_metadata__yx1.csv
artifact_path(ROOT, "scope_metadata", "mapped", "20250912")
#   → {ROOT}/experiment_metadata/20250912/scope_metadata_mapped.csv
```

### Case 3 — per-well stage, the `path_mode` branch (`well_id` lives in the path)
```python
artifact_path(ROOT, "mask_geometry", "metrics", "20250912",
              path_mode="per_well", well_id="20250912_B01")
#   → {ROOT}/computed_features/20250912/per_well/20250912_B01/mask_geometry/mask_geometry_metrics.csv
artifact_path(ROOT, "mask_geometry", "metrics", "20250912", path_mode="merged")
#   → {ROOT}/computed_features/20250912/mask_geometry/mask_geometry_metrics.csv
validated_path(ROOT, "mask_geometry", "metrics", "20250912",
               path_mode="per_well", well_id="20250912_B01")
#   → ...mask_geometry_metrics.csv.validated   (derived suffix; convention pending audit — see AUDIT TODO)
```

### Case 4 — the hallucination the resolver now BLOCKS
```python
artifact_path(ROOT, "plate_metadata", "csv", "20250912",
              path_mode="per_well", well_id="20250912_B01")
#   ❌ ValueError: plate_metadata.csv is experiment-grain
#   (without fanout-enforcement this would mint a fake per_well/ path — the whole point)
```

### Case 5 — the merge consumes the SAME registry (ties to the well-runner)
```python
rule merge_mask_geometry:
    input:
        shards = lambda wc: merge_trigger_inputs(
            checkpoints=checkpoints, wc=wc, config=config, root=DATA_ROOT,
            stage="mask_geometry", artifact="metrics")
        #   → one DECLARED per-well path per RUN well (the trigger; merge body scans for full content)
        #     all args explicit kwargs — never reads root/config as module globals (#8)
    output:
        merged = lambda wc: artifact_path(DATA_ROOT, "mask_geometry", "metrics",
                                          wc.experiment, path_mode="merged")
```
The same row produces both the per-well shard paths AND the merged path → the merge can
never disagree with what the per-well stage wrote.

### ⚠️ Orchestration / core boundary (do NOT infect core functions)
Path resolution lives at the **task/rule boundary**. Pure processing functions still
receive **concrete paths** — they must not know about `output_root` or the registry:
```python
# tasks.py / rule boundary — KNOWS the registry
out = artifact_path(output_root, "plate_metadata", "csv", exp)
process_plate_layout(input_file=..., experiment_id=exp, output_csv=out)

# core function — registry-ignorant, testable in isolation
def process_plate_layout(input_file: Path, experiment_id: str, output_csv: Path): ...
```
This keeps the two kingdoms clean: resolution is orchestration; compute stays pure.

---

## 🤖 THE WELL-RUNNER — `lib/well_runner.py` (orchestration kingdom)

**Core tenet: the well-runner provides MECHANISM, not staleness.** Snakemake owns
staleness (mtimes + DAG). The well-runner's whole job is to make it *easy to wire the DAG
so that per-well staleness is preserved* — i.e. to stop every rule from re-deriving the
per-well path/shard logic (slightly wrong) by hand.

> **Two kingdoms (hard boundary):** the well-runner **imports** ID constructors from
> `identifiers/`; it **never mints IDs**. Identity flows *into* orchestration, never the
> other way. `lib/paths.py` (placement) + `lib/well_runner.py` (well selection + shard
> collection) are the orchestration kingdom; `identifiers/` is the identity kingdom.

**The mechanism it exposes (small, sharp):**
```python
# paths (in lib/paths.py — the registry; see Lean MVP Contract above)
artifact_path(root, stage, artifact, exp, path_mode=..., well_id=..., format_vars=...)
validated_path(...) / provenance_path(...)            # derived sentinel helpers

# well selection + the merge trigger (lib/well_runner.py)
def run_wells(*, checkpoints, wc, config) -> list[str]:
    """The ONE definition of 'what computes now.' discovered ∩ target; empty target → all.
    Fail loud on a requested-but-missing well."""
    discovered = read_wells(checkpoints.discover_wells.get(experiment=wc.experiment).output.discovered_wells_txt)
    return select_run_wells(discovered=discovered, target=config.get("target_wells") or [], exp=wc.experiment)

def merge_trigger_inputs(*, checkpoints, wc, config, root, stage, artifact) -> list[str]:
    """The merge's DECLARED inputs = the RUN wells' shards (the trigger edges). Built via
    WellRun.shard() so generate-side and merge-side agree. NOT the full merge content —
    the merge BODY scans all present `.validated` shards at execution time (tasks.py)."""
    return [str(WellRun(wc.experiment, w, root).shard(stage, artifact))
            for w in run_wells(checkpoints=checkpoints, wc=wc, config=config)]
```

> ## ⭐ GUIDING PRINCIPLE — the merge triggers narrowly, composes broadly, NEVER shrinks (2026-06-04)
> **Triggered** by what ran (the run wells' shards are the merge's *declared* inputs = the DAG
> edges). **Composed** over every valid shard *seeable on disk* (an execution-time scan in the
> merge body). `target_wells` selects what **computes**, never what the merge **includes**.
> So: run B01 → B01's edge fires the merge → the merge re-cats **all** present valid shards
> (A01 + A02 + B01) → the merged file stays whole. It grows as wells gain shards; it cannot shrink.

**Three well-lists (keep distinct):**
| List | Meaning | Role in the DAG |
|---|---|---|
| `discovered_wells` | metadata says these exist (`discovered_wells.txt`) | discovery output (the checkpoint) |
| `run_wells` | `discovered ∩ target` (empty target → all) | **the compute-control list** — the per-well fan-out **and** the merge's *trigger* edges |
| *present valid shards* | wells with a `.validated` shard on disk **right now** | the merge **content** (scanned at execution time, not a DAG dependency) |

> ☠️ **The skull rule, refined (2026-06-04 — supersedes "merge depends on `active_wells`").**
> A **run well** failing validation is a **hard error** (its shard is a required, declared input
> that didn't materialize) — *that* invariant stands. But a **non-run well** that simply has no
> shard is **fine** — it is just absent from the merge content, not an error. This is **not**
> "validation gating the merge": the merge content comes from *which shards exist on disk*, not
> from live validation results driving the DAG. (Old text said merge inputs = `active_wells`; that
> would shrink the merged file to the run subset — the bug this revision fixes.)

**This is plan Win 4 + Win 5 made concrete** (see `well_id_throughline_refactor_plan.md` §Scope 4):
one `select_run_wells()` = one well-selection source (Win 4, delete the `targets.py` copy);
checkpoint is truth, config only filters (Win 5). Also folds in the cleanup of the **two redundant
discovery paths** today (`_wells_from_mapping` on local `well_index` vs. the `discover_wells`
checkpoint on global `well_id` → collapse to the checkpoint).

> **`target_wells` is a COMPUTE-selection knob, NOT a merge-membership filter (loud note).**
> `target_wells: [B01]` means "recompute B01 now" — it does **not** mean "publish only B01."
> The merge always composes over all seeable valid shards. (A future `publish_wells` could filter
> the *merged product* if ever needed; out of MVP scope.)

### What's actually IN `lib/well_runner.py` (the full contents)

Six things, no more. It is glue, not logic. (Formal home: `well_id_throughline_refactor_plan.md`
§Scope 4.)

| Symbol | Job | Reads | Returns |
|---|---|---|---|
| `select_run_wells(*, discovered, target, exp)` | **PURE** list arithmetic (unit-testable, no Snakemake). `discovered ∩ target`; empty target → all; fail-loud on missing. Lifts the inline subset-normalize logic out of the `discover_wells` checkpoint (Snakefile:453–467). | args only | `list[well_id]` |
| `read_wells(path)` | parse `discovered_wells.txt` → `list[well_id]` (lifts `wells_for_experiment`, Snakefile:478–480) | the file | `list[str]` |
| `run_wells(*, checkpoints, wc, config)` | Snakemake glue: force checkpoint → `read_wells` → `select_run_wells`. The ONE "what computes now." Replaces `wells_for_experiment` (Snakefile:476). | checkpoint + config | `list[well_id]` |
| `WellRun` + `well_run(wc, root)` | **per-well path binder** — bind `(experiment_id, well_id, root)` once; `.shard(stage, artifact)` / `.sentinel(...)` delegate to `paths.py` (`artifact_path(path_mode="per_well")`). Composes NO path strings itself. | — | dataclass |
| `merge_trigger_inputs(*, checkpoints, wc, config, root, stage, artifact)` | the merge's **declared inputs = the run wells' shards** (the trigger edges). Built via `WellRun.shard()` so generate-side and merge-side can't disagree. **Not** the full content. | the above | `list[path]` |

> **`artifact_path` is the `paths.py` registry call.** `WellRun` imports it; the **layout**
> (`{family}/{exp}/per_well/{well_id}/...`) lives in `paths.py`, never in the well-runner. The
> stage→family through-line is resolved there: `stage="mask_geometry"` → its registry row has
> `family="computed_features"` → path `computed_features/{exp}/per_well/{well_id}/...`.

> **Parse-time vs checkpoint-time (the Snakemake gotcha).** `artifact_path` is imported at
> Snakefile **parse time** (plain Python). But per-well paths are wrapped in `lambda wc: ...` and
> evaluated **lazily, after the `discover_wells` checkpoint resolves** — because the well list is
> runtime data. That's *why* every per-well `input:`/`output:` and the merge inputs are lambdas.

> **Pure-vs-glue split (#8/#9):** `select_run_wells` is **pure** — no `checkpoints`, no globals —
> unit-testable "without summoning Snakemake from the basement." `merge_trigger_inputs` takes
> `root`/`config` as **explicit params**, never module globals (specs that lean on unmentioned
> globals become haunted).

> **The merge CONTENT does NOT live here.** `merge_trigger_inputs` returns only the *trigger*
> edges. The execution-time "scan all present `.validated` shards and cat them" is the **merge
> body's** job (`tasks.py` / the merge core fn), which asks `paths.py` for the per-well glob
> grammar. **Tenet-8 carve-out (2026-06-04):** a present-shard scan is permitted **inside the
> merge body at execution time** (the run-wells' declared edges already triggered the rule; the
> scan only assembles content). It is **still banned** in any `input:` list (glob-as-dependency /
> glob-as-well-list destroys staleness). Keying the scan on the `.validated` sentinel (atomic,
> post-completion — Tenet 9) avoids mid-write reads.

It does **not** contain: ID minting (→ `identifiers/`), path layout (→ `paths.py`), execution or
the present-shard scan (→ `tasks.py`), or any staleness logic (→ Snakemake). If a function here
starts composing a path string, running a stage, or scanning the filesystem, it has crossed into
the wrong kingdom.

### Worked run — `target_wells: [B01]`, discovered `{A01, A02, B01}`, A01/A02 already built

```
rule compute_mask_geometry_well:                              # GENERATE (fans over run_wells)
    input:  lambda wc: well_run(wc, ROOT).shard("segmentation_tracking")
    output: lambda wc: well_run(wc, ROOT).shard("mask_geometry"),     # → computed_features/{exp}/per_well/{well_id}/...
            lambda wc: well_run(wc, ROOT).sentinel("mask_geometry")
    shell:  "tasks compute-mask-geometry --well-id {wildcards.well_id} --in {input} --out {output[0]}"

rule merge_mask_geometry:                                     # MERGE
    input:  lambda wc: merge_trigger_inputs(checkpoints=checkpoints, wc=wc, config=config,
                            root=ROOT, stage="mask_geometry", artifact="metrics")   # = run wells' shards (TRIGGER)
    output: artifact_path(ROOT, "mask_geometry", "metrics", "{experiment}", path_mode="merged")
    shell:  "tasks merge-mask-geometry --stage mask_geometry --output-root {ROOT} --out {output}"
            #        └─ merge body scans per_well/*/mask_geometry.validated → cats ALL present (CONTENT)
```
1. `discover_wells` → `discovered_wells.txt = {A01, A02, B01}`.
2. `select_run_wells(discovered={A01,A02,B01}, target=[B01])` → `run_wells = [B01]`.
3. Generate fans over `[B01]` → builds **only** `per_well/B01/...` (A01/A02 not recomputed — isolated).
4. `merge_trigger_inputs` = `[WellRun(20250912, B01, ROOT).shard("mask_geometry")]` → the merge's
   declared input is **B01's shard**.
5. **Trigger:** B01's shard is newer than the merged output → `merge_mask_geometry` fires. **Once.**
6. **Content:** the merge body scans `per_well/*/mask_geometry.validated` → finds **A01, A02, B01**
   → cats all three → merged file = `A01 + A02 + B01`. **Does not shrink.**
7. Re-invoke with nothing changed → B01's shard mtime unchanged → merge **skipped** (idempotent).
   Full plate (`target_wells: []`) → `run_wells = {A01,A02,B01}` → same DAG, wider selection.

> **The one caveat (kept honest):** if a non-run shard changed *without* any run well changing in
> the same invocation, the merge wouldn't auto-fire (its declared edges didn't move). But you only
> change a shard by **running its well** — which makes it a run well, a declared edge, a trigger.
> So in normal operation the merged file always reflects the full current state.

### How it interfaces with `config.yaml` (the science knobs)

The well-runner reads exactly **one** key from config — the selection filter:
```yaml
# config.yaml (committed, science)
experiments:  [20250912]      # which experiments (read by the Snakefile, not the runner)
target_wells: []              # [] = all discovered; ["B01"] or ["20250912_B01"] = subset
                              #   ↑ the ONLY config key the selection logic reads
```
**Config only *filters*; the checkpoint *decides* what exists** (Win 5). This is the
one-line escape hatch that makes one-well / subset / full runs the same DAG (Tenets 10, 11).

**`target_wells` semantics (specified, #7):**
- **Local (`B01`)** → interpreted **per experiment** (matches `B01` in *every* experiment
  in the run). Normalized to global `well_id` against the *current* experiment's
  `discovered` set.
- **Global (`20250912_B01`)** → applies **only** to that experiment.
- **Mixed local/global** in one list: allowed.
- **Validation (all hard errors, never silent):** requested well not in `discovered` →
  raise; duplicate after normalization → raise; empty `active` set → raise. (`[]` means
  "all discovered," which is the only way to get the full set — not the same as empty active.)
- Normalization to global `well_id` uses `identifiers/` (the runner imports, never mints).
- **Multi-experiment empty-policy (#7):** `target_wells` is evaluated **per experiment**
  independently. With **`[]`** (all discovered), every experiment in the run gets its full
  discovered set — never empty (a discovered-empty experiment is itself a hard error, raised
  at discovery). With a **non-empty filter**, each experiment intersects the filter against
  *its own* `discovered` set:
  - A **local** filter (`B01`) that matches in some experiments but not others → **raise**
    for the experiments where it's missing (requested-but-absent is a hard error, applied
    per experiment — not a silent skip).
  - A **global** filter (`20250912_B01`) names exactly one experiment; experiments it
    doesn't name contribute **nothing** to `active` from that entry. If the *union* of
    filter entries leaves any experiment in the run with an **empty `active` set → raise**
    (every experiment in the run must resolve to ≥1 active well; to exclude an experiment,
    drop it from `experiments:`, don't leave it filter-empty).

> Machine knobs (`output_root`, `python`, `device`) live in **`env.yaml`** (Scope 3), not
> here. The well-runner takes `output_root` as a **parameter**, never derives it — that's
> why Scope 3 precedes the runner.

### How it interfaces with Snakemake tasks (the wiring)

Two touch points, both at the **rule boundary** (never inside core functions):

**(1) Per-well rules — expand the DAG over `run_wells` (paths via `WellRun`):**
```python
rule compute_mask_geometry_well:
    input:   seg = lambda wc: well_run(wc, DATA_ROOT).shard("segmentation_tracking")
    output:  out = lambda wc: well_run(wc, DATA_ROOT).shard("mask_geometry"),
             done = lambda wc: well_run(wc, DATA_ROOT).sentinel("mask_geometry")
```
**(2) Merge rules — declared inputs = run wells' shards (the trigger); body scans for content:**
```python
rule merge_mask_geometry:
    input:   shards = lambda wc: merge_trigger_inputs(           # TRIGGER edges = run wells' shards
                 checkpoints=checkpoints, wc=wc, config=config, root=DATA_ROOT,
                 stage="mask_geometry", artifact="metrics")      # all explicit kwargs (#8)
    output:  merged = lambda wc: artifact_path(DATA_ROOT, "mask_geometry", "metrics",
                                wc.experiment, path_mode="merged")
    # tasks merge-mask-geometry body scans per_well/*/mask_geometry.validated → cats ALL present (CONTENT)
```

**The dispatch layer (`tasks.py`) — STATUS: does not exist yet (Win 2).** Today every
rule shells out via deep `-m data_pipeline.feature_extraction.entrypoints.compute_mask_geometry`
(verified: Snakefile uses `-m module` for *all* stages; there is no `tasks.py`). Win 2
routes these through verbs: `... tasks compute-mask-geometry --well-id ...`, so the
Snakefile knows **verbs, not module paths**. The well-runner and `tasks.py` are siblings:
```
Snakefile rule  ──shell──►  tasks.py <verb> --well-id B01 --output-root ...
                                │  (resolves paths via paths.py, picks core fn)
                                ▼
                            core compute fn(input_path, output_path)   ← registry-ignorant
```
So the full chain per stage: **rule (expands over run_wells) → tasks verb (resolves
paths) → core fn (concrete paths).** Three layers, each in its own kingdom. Building
`tasks.py` is Win 2 (independent, eases Scope 5); until then rules call `-m module`
directly and the well-runner helpers are imported into the Snakefile at parse time.

---

## 🎯 THE TARGET MODEL (committed 2026-06-02 — Zone C per-well-able: owner-confirmed + spot-checked, exhaustive grep pending)

```
Zone A — experiment bootstrap        (fanout=experiment; discovers wells)
  experiment_metadata/{exp}/...      plate, scope, series_mapping, frame_inventory, discovered_wells.txt

  ── FAN POINT: discover_wells checkpoint → discovered_wells.txt (global well_id) ──

Zone B0 — image materialization      (per-well image tree, NO merge; outside the registry)
  built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/   (well_id; one job loops wells)

Zone B — per-well canonical computation   (fanout=per_well_then_merge; one well = one unit, E2E)
  segmentation_and_tracking/{exp}/per_well/{well_id}/...
  processed_snips/{exp}/per_well/{well_id}/...
  computed_features/{exp}/per_well/{well_id}/...        ← NEW (was experiment-grain)
  quality_control/{exp}/per_well/{well_id}/...          ← NEW
  analysis_ready/{exp}/per_well/{well_id}/...           ← NEW

Zone B (cont.) — MODEL / EMBEDDINGS    (legacy build_06; encodes snips → latents; FEEDS analysis_ready)
  embeddings/{exp}/per_well/{well_id}/latents.parquet   (snip_id → z_mu_*; per-well shard)
  embeddings/{exp}/bf_embryo_snips/{exp}/...             (symlink VIEW of processed_snip_path)

Zone C — merged / publication products    (thin concat at the END of the DAG; NOT cohort)
  segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv
  processed_snips/{exp}/contracts/snip_manifest.parquet
  computed_features/{exp}/consolidated/...
  quality_control/{exp}/consolidated/...
  embeddings/{exp}/contracts/latents.parquet
  analysis_ready/{exp}/analysis_ready.csv               (joins latents → embedding_calculated)
```

> **Embeddings is a LATE SPINE STAGE that feeds analysis_ready — NOT a terminal "Zone D".**
> Legacy is explicit: `build06: df02 + latents → df03`, and `analysis_ready` already
> reserves `embedding_calculated` (today hardcoded False). So the spine is
> `… features → qc → embeddings → analysis_ready`; latents join on `snip_id` and flip
> `embedding_calculated`. The seam is specified in `target/model_input_handoff_contract.md`
> (inference-first; reuses `gen_embeddings` + the `mseq_pipeline_py3.9` sub-env; no
> `pert_id`/splits — training deferred). **Not yet built** (`src/data_pipeline/embeddings/`
> is empty).
>
> **⭐ Embeddings is the canonical `fanout=per_well` + `execution=single` (batched) stage**
> (decided 2026-06-05). Latents land **per-well** (spine + incremental staleness preserved),
> but **ONE job loads the model once** and encodes all *run* wells, writing each well's
> shard — because loading the legacy model through the Py-3.9 `conda run` sub-env **per well**
> would dominate the actual encode (process spawn + checkpoint deserialize ≫ encoding a few
> hundred snips). This is the *clincher* row of the `fanout` vs `execution` table made real:
> per-well files, single looping job — same shape as stitching. (See the merge body / present-
> shard scan; the batched encoder writes shards, the merge composes them.)

**Merged-or-not is a TARGET distinction, not a mode (#12):** the rhythm
(`run_X_per_well → merge_X → validate_merged_X`) doesn't conflict with one-well runs —
*you just request a different target:*
```
snakemake .../analysis_ready/{exp}/per_well/{well_id}/...   # one well; NO merge built
snakemake .../analysis_ready/{exp}/analysis_ready.csv       # active wells + merged product
```
- **Per-well target** → builds that well's chain only; merge not requested → not built.
- **Experiment/publication target** → builds active wells, *then* the merged product.

No special mode — same DAG, the requested target decides whether the merge node runs.

**Path function — `well_id` lives in the canonical path, not a separate root.** See the
**Worked Example** above for the authoritative API (`path_mode`, `stage="mask_geometry"`
as a *verb* not a family, fanout-enforced). Sketch:
```
artifact_path(root, "mask_geometry", "metrics", exp, path_mode="per_well", well_id=well_id)
#   → computed_features/{exp}/per_well/{well_id}/mask_geometry/mask_geometry_metrics.csv
artifact_path(root, "mask_geometry", "metrics", exp, path_mode="merged")
#   → computed_features/{exp}/mask_geometry/mask_geometry_metrics.csv
```

---

## ⚙️ PER-WELL DAG MECHANICS (staleness · spine · merge-DAG · concurrency)

### Staleness model
Snakemake staleness = **file mtimes + DAG edges**, nothing more. There is no per-well
intelligence — output is stale iff a declared `input:` is newer. So **per-well staleness
exists only if the spine is wired per-well end-to-end.** Today the **merge wall** breaks
it: Zone C reads the *merged* contract, so touching one well re-runs all wells'
features/QC. (Tenet 7)

### The spine rule ☠️
The spine is the per-well shard chain — each well-local stage reads the **previous
stage's per-well shard**:
```
seg/{well} → snips/{well} → features/{well} → qc/{well} → analysis_ready/{well}
```
Merged files are **off-spine materialized views** (notebooks, inspection, publication).
☠️ **A merged file must never be read as input by a well-local stage** — that recreates
the merge wall and destroys per-well staleness.

> **Precise rule (bootstrap-OK vs merged-bad):** reading an *experiment-grain bootstrap*
> artifact (`discovered_wells.txt`) as a well-local input is **fine** — it's upstream of the fan.
> Reading a *merged view* (post-fan, assembled-from-shards) is **forbidden**. The test:
> was this file made by collapsing per-well shards? If yes, don't depend on it.

**Scope-5 fix item:** `surface_area_qc` currently reads `consolidated_features` (merged)
— an accidental merge wall. It does per-snip work vs. an external curve, so it converts
cleanly to reading per-well shards. (Tenets 2, 7)

### Merge-DAG construction (how the DAG auto-detects the right wells)
At parse time Snakemake doesn't know how many wells exist (it's runtime data). The
**checkpoint** resolves this: `checkpoints.discover_wells.get(...)` forces a DAG re-plan
with the discovered list. Merge inputs must therefore be a **checkpoint-derived,
DECLARED `input:` list** (via `merge_trigger_inputs`) — the only pattern that gives
**both** correct well-detection **and** correct staleness. (Tenet 8)

**DO-NOT (each silently breaks the dependency edge):**
| Anti-pattern | Why it breaks |
|---|---|
| `glob("per_well/*/...")` | parse-time snapshot; misses unbuilt wells; can read mid-write |
| inputs from `config["wells"]` | bypasses checkpoint; can name nonexistent wells (Win 5) |
| inputs from `series_well_mapping` directly | the *other* (pre-fan, local `well_index`) discovery path |
| shell iterates the per-well dir | no `input:` edges at all → fires early, no re-trigger |

### Concurrency / write safety — a non-issue by construction
| Scenario | Concern? | Why |
|---|---|---|
| Parallel *different* wells → own shards | ❌ | `well_id` in path = disjoint files; can't collide |
| Merge reads a mid-write shard | ❌ | DAG dependency edge = the lock (inputs declared, not globbed) |
| Same well, two concurrent invocations | ⚠️ rare | operator error; Snakemake's working-dir lock catches the common form |

No locking machinery needed. Two invariants keep it safe (both already decided):
**(1) well_id in the path**, **(2) merge inputs declared, not globbed.** Optional cheap
hardening: per-well entrypoints write to `*.tmp` then atomic `os.rename()`. (Tenet 9)

### Merge cadence — deferred (Step 6 / tuning, NOT architecture)
*Per-stage* merges (fire on every well touch) vs *stage-group* merges (one big merge at
a zone boundary, e.g. `consolidated_features` after all features). This only changes
*how often* off-spine merges fire — the spine is per-well shards either way. **Decide
later; it is performance tuning, not a design decision.**

---

## 🪟 ZONE-A NARROWING — the frame contract joins the spine (2026-06-02)

> **The whole refactor in one sentence:** the frame contract moves from an
> experiment-grain fan-point *barrier* to a per-well *spine artifact*. Metadata discovers
> wells; per-well frame contracts validate that each active well's metadata aligns with
> its stitched images. The merged experiment frame contract is a derived view and must
> not feed downstream well-local rules.

**Accepted revision (mdcolon):** Zone A *narrows* to **discovery only**. Reconciliation
(metadata ∩ images) and the existence check join the **per-well spine**. Discovery stays
experiment-grain (you can't discover B01 from B01 alone) — so this narrows, not
contradicts, the "Zone A is experiment-grain" founding assumption.

**Today, `build_frame_contract` fuses THREE jobs at experiment grain** (discover +
reconcile + existence-check across all wells) → an experiment-grain barrier. **Split it:**

| Step | Stage (renamed) | Inputs | Output | Fanout |
|---|---|---|---|---|
| 1 | `discover_wells_from_metadata` *(checkpoint)* | `scope_metadata_mapped.csv` (**canonical metadata, no images** — extracts the well set; #13) | `discovered_wells.txt` (🟢 TARGET renames `wells.txt`; see `target/front_end_naming_and_flow.md`) | experiment (fan point, moved **earlier**) |
| 2 | `stitch_well` | this well's raw images + its `scope_metadata_mapped` rows | `built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/` (well_id; REVISED 2026-06-03) | per-well image tree (**off-registry**) |
| 3 | `validate_frame_contract_well` | this well's **images (Step 2)** + this well's **metadata rows** | `<frame-contract-family>/{exp}/per_well/{well_id}/frame_contract.csv` | `per_well_then_merge` |

> **⚠️ TODO (#5) — decide the target frame-contract family. Why this is a real open
> question (not a naming nitpick):** the frame contract is the **one artifact that changes
> grain across the refactor.** In CURRENT it's a *bootstrap* artifact (experiment-grain,
> `fanout=experiment`) and so it naturally lives in `experiment_metadata/` alongside
> plate/scope/mapping — a family where **every** member is experiment-grain. In TARGET it
> becomes a *per-well spine* artifact (`fanout=per_well_then_merge`, one shard per well +
> a merged view). That is a **different grain than everything else in `experiment_metadata/`**.
> The registry's whole job is to make grain legible from the path, so we can't just let it
> inherit the CURRENT family by default — that would put a `per_well/{well_id}/` subtree
> inside a family the reader expects to be uniformly experiment-grain. Hence the
> `<frame-contract-family>` **placeholder**: it marks the spot where the registry author
> must *consciously* choose, rather than silently carrying CURRENT forward.
>
> **Two options:**
> - **(a) keep it in `experiment_metadata/`** — minimal churn and migration, but mixes a
>   per-well spine artifact into the experiment-grain bootstrap family. The family stops
>   being "all experiment-grain," which muddies the grain-from-path story.
> - **(b) dedicated `frame_contracts/` family** — clean grain story: the family is uniformly
>   `fanout=per_well_then_merge`, matching the other spine families (`segmentation_and_tracking/`,
>   `processed_snips/`, …). Cost: a new family + a migration of the on-disk location.
>
> **DECIDED 2026-06-05 → (b) dedicated `frame_contracts/` family.** The frame contract
> becomes a per-well spine artifact (`fanout=per_well_then_merge`), so it goes in its own
> uniformly-per-well family alongside `segmentation_and_tracking/`, `processed_snips/`, etc.,
> rather than muddying the experiment-grain `experiment_metadata/` bootstrap family. The
> `<frame-contract-family>` placeholder in the Step-3 row resolves to **`frame_contracts/`**.
> Cost accepted: a new family + on-disk migration of the frame-contract location.
>
> **Multi-channel (DECIDED 2026-06-05): one per-well contract, channel as ROWS.** The frame
> contract already keys on `(experiment_id, well_id, channel_id, time_int)` and built images
> already nest `stitched_ff_images/{well_id}/{channel}/` — so multi-channel is **not** a
> structural change. Keep **one** `frame_contract.csv` shard per well
> (`frame_contracts/{exp}/per_well/{well_id}/frame_contract.csv`); additional channels are
> just more rows. Do **not** split to per-well-per-channel shards (premature while BF-only;
> deeper spine for no current win). Today the builder selects BF (`_determine_bf_channel`),
> collapsing to one channel; the row-grain already supports more without redesign.

**Consequences (each preserves a tenet):**
- **Segmentation reads its per-well slice**, not the whole contract. *(Verified: today it
  reads the full contract and immediately filters to one `well_id`, using only per-well
  fields like `source_micrometers_per_pixel`.)* **This is what makes one-well == whole
  experiment for the front half** — nothing in segmentation's path is experiment-grain.
  (Tenet 10)
- **Step 3 IS the per-well validation gate** (validate type 1: metadata aligns with images
  on disk) — this is a *spine* check, distinct from the merged-file validation that
  `merge_frame_contract` does. (See the three-meanings-of-validate note.)
- `merge_frame_contract` = concat of per-well slices → off-spine experiment **view**.
  ☠️ nothing downstream reads it. (Tenets 2, 3)
- **`stitched_inventory.csv` drops out of the spine** → optional off-spine report (today
  it's load-bearing because the frame contract needs it; here it isn't).
- **Canonical discovery input = `scope_metadata_mapped.csv` (decided by mdcolon, #13).**
  Both `series_well_mapping.csv` and `scope_metadata_mapped.csv` carry the well set.
  `series_well_mapping.csv` is the *minimal* form (well identities only); `scope_metadata_mapped.csv`
  is the **fully joined** form that also carries the scope metadata per row. **mdcolon prefers
  `scope_metadata_mapped.csv`** because we want all the scope metadata available at discovery
  time, not just the bare well list — having the canonical metadata table as the discovery
  input means the well list and its metadata stay co-located and the downstream per-well
  stages read from the same canonical source. So `discover_wells_from_metadata` reads
  `scope_metadata_mapped.csv` (it extracts the well set from it). Both are **already
  microscope-converged** (Tenet 5), so early discovery is microscope-agnostic either way.
  *(Trade-off accepted: this pulls the `apply_series_mapping` edge into the checkpoint —
  intentional, since that table is wanted anyway.)*

**Naming adopted:** `discover_wells_from_metadata` (clarifies: metadata-only, doesn't
check images) and `validate_frame_contract_well` (clarifies: it's a *check*, not just
row formatting).

---

## ❓ OPEN QUESTIONS (next session — don't lose these)

1. ~~**Where exactly is the fan point?**~~ **RESOLVED (with current/target split):**
   🔵 CURRENT fan = `discover_wells` checkpoint reading `frame_contract.csv` (Snakefile:427).
   🟢 TARGET fan = `discover_wells_from_metadata` reading `scope_metadata_mapped.csv` (#13;
   moved earlier, before stitching). Stitching's `_wells_from_mapping` (local `well_index`) is a
   *separate, redundant* path → consolidate to the checkpoint (Win 4/5).
2. ~~**Is Zone C truly per-well-able?**~~ **RESOLVED for planning (owner-confirmed +
   spot-checked, exhaustive grep pending):** the MATH is per-snip/per-embryo with no
   cross-well cohort statistic — owner states it as known fact, and a spot-check agrees
   (the only percentile code is the offline `build_sa_reference.py`, not a Snakefile rule;
   `embryo_qc` percentiles are per-embryo across Z-pairs). **Caveat:** this is a spot-check,
   not a full grep-audit — the exhaustive module/searched-for/result table is still pending
   (see AUDIT TODO). Green light **to plan**; the WIRING still has merge-wall reads to
   convert (separate claim, see Zone B).
3. ~~**Well list as first-class artifact?**~~ **RESOLVED: `wells.txt` from the checkpoint
   already is one (global `well_id`).** Cleanup = make it the *single* source;
   the **selection filter** becomes a pure input filter (Win 5: checkpoint is truth, config
   only filters); `series_well_mapping.csv` stays raw discovery.
   > **Terminology (#6) — `target_wells` is canonical, `selected_wells.txt` is legacy.**
   > These name the **same idea** (the user's "which wells do I want" filter) in two forms:
   > `target_wells` is the **config key** (the form the TARGET well-runner reads — see
   > "How it interfaces with `config.yaml`"); `selected_wells.txt` is the older **file**
   > form. **Decision: standardize on `target_wells` (config key)** as the canonical filter
   > input; treat any `selected_wells.txt` as a legacy alias to migrate off. The filter is
   > never an artifact and never gates the DAG — the checkpoint decides what exists, the
   > filter only narrows it.
4. **Map `src/` onto the grain shape — light, and via the registry, later.** `src/` is
   already stage-organized; the "map" is the **registry itself** (`stage → family →
   fanout → artifacts`, + deferred `src_family`), NOT a directory move. **Principle: code location ≠
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
- **Stitching fanout value:** ~~per-well-no-merge~~ **RESOLVED: image trees live OUTSIDE
  the registry.** The registry governs *tabular contract artifacts* (`.csv`/`.parquet`);
  stitched-image *directories* are a different kind of output with their own path logic.
  Keeps `fanout` at two values. (Confirm the exact image-path helper when writing `paths.py`.)
- **Merge cadence:** per-stage vs stage-group merges — **deferred to Step 6 / tuning**
  (see DAG Mechanics). Not architecture.
- **Two redundant discovery paths:** `_wells_from_mapping` (local `well_index`) vs.
  `discover_wells` checkpoint (global `well_id`) → collapse to the checkpoint via
  `select_run_wells()` (Win 4/5). *(mdcolon to elaborate on specifics.)*
- **Scope-2 migration touch:** `discover_wells` reads `frame_contract.csv`'s `well_index`
  column (Snakefile:448), which Scope 2 deletes → this checkpoint is a migration site.

---

## ⏸️ WHERE WE PAUSED (2026-06-02)

The grain model is **settled**. 🔵 CURRENT order: A-metadata → B0 (stitching) →
frame_contract → fan (`discover_wells` checkpoint, reads frame_contract) → B
(segmentation/snips per-well) → C (merged). **Note the fan sits LATE — after stitching.**
🟢 TARGET: pushes B/C fully per-well, splits
the frame contract per-well (Zone-A narrowing), moves the fan earlier
(`discover_wells_from_metadata`). The registry is trimmed
to a **lean path resolver** (Bloat Audit): per-stage `family`/`fanout`/optional
`subfolder`, per-artifact filename template; sentinels are derived helpers;
`execution`/`grain`/`schema`/`kind`/`src_family` are deferred (each named with its future
reader). The `fanout` vs `execution` distinction stays as a *documented concept*, not a
field. Zone-C per-well-safety: **math** owner-confirmed + spot-checked; **wiring** still
has merge-wall reads to convert (not the same claim). scoped_runs demoted to scratch. The
**well-runner** provides *mechanism, not staleness* (`select_run_wells` pure +
checkpoint glue + `merge_trigger_inputs` + `WellRun`); imports IDs from `identifiers/`, never
mints. Merge **triggers** on run-wells' declared edges, **composes** over all present valid shards
(execution-time scan in the merge body) — never shrinks (revised 2026-06-04).
The **frame contract is split** (🟢 TARGET) into early metadata-only discovery + a per-well
validate stage, so segmentation reads a per-well slice and one-well == whole-experiment for
the front half too.

**The actual blocker:** none of the back/front-half wiring is safe to build until
`well_id` means exactly one thing — **Scope 1** (create `identifiers/`) + **Scope 2**
(flip semantics). The spine keys on `well_id`. That unglamorous front end is the real
critical path.

(The `paths.py` worked example is **done** — see the WORKED EXAMPLE section. ✅)

**Next concrete steps, in order (revised per audit #15 — identifiers BEFORE paths.py):**
1. **Scope 1** — create `identifiers/` (`constructors`/`parsers`/`validators`). Zero-risk,
   additive, unblocks everything that keys on `well_id`.
   - **Tests:** unit tests for round-trip construct↔parse of `well_id = {exp}_{well}`, and
     reject-malformed cases (validators raise, not coerce).
2. **Implement `lib/paths.py`** from the Lean MVP Contract + Worked Example (fanout-
   enforced). Safe to build now; per-well paths use `well_id` but aren't *wired* into the
   DAG until Scope 2.
   - **Tests:** pin the resolution rules against the Worked Example (Cases 1–4) as golden
     paths; assert the **fanout-enforcement** raises — `path_mode="per_well"` on an
     `experiment` stage, `well_id` passed where rejected, missing `well_id` where required
     (Case 4 is the canonical guard). Add `format_vars` token-fill and unknown-stage/unknown-
     artifact error cases (see #15).
3. **Scope 2** — migrate `well`/`well_id` semantics in schemas + call sites; regenerate.
   - **Tests:** regression on regenerated contract files (schema/columns unchanged except the
     intended `well`/`well_id` flip); spot-check a known well (`20250912_B01`).
4. **Implement `lib/well_runner.py`** on normalized IDs (`select_run_wells` pure +
   checkpoint glue + `merge_trigger_inputs` + `WellRun`).
   - **Tests:** **pure** unit tests on `select_run_wells` (no Snakemake) — `[]`→all,
     local/global/mixed filters, multi-experiment empty-policy (#7), and all hard-error cases
     (missing requested, duplicate-after-normalize, empty run set). The glue wrappers get a
     thin smoke test with a faked checkpoint. Plus the merge-content test: with A01/A02 already
     built and `target_wells=[B01]`, the merged file = A01+A02+B01 (does not shrink).
5. **Wire one stage** end-to-end through the registry as the proof, then replicate.
   *(Win 2 `tasks.py` verbs can land independently anytime to ease this.)*
   - **Tests:** a one-well DAG run (request a per-well target → only that well builds);
     assert the merge's declared inputs are `merge_trigger_inputs` (run wells), never a glob,
     and the merge body composes over all present `.validated` shards.

---

## 🗂️ AUDIT TODO (raised 2026-06-02, not yet actioned)
- **Verify `.validated` sentinel convention is uniform** (`{filename}.validated` for *all*
  sentinels). If any rule uses `.{stem}.validated` or `{stem}.validated`, either normalize
  or give `validated_path` options. *(Assumption in the MVP contract, not yet verified.)*
- **Optional exhaustive Zone-C grep-audit** (module / searched-for / result table) to
  upgrade "owner-confirmed + spot-check" → "exhaustively verified."
- **Consider splitting this doc** if it keeps growing: Conceptual (north star) vs
  Implementation Spec. For now the CURRENT/TARGET tags + document map at top carry the load.
