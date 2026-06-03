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
> **Document map:** Goal → Central Insight → Grain Transition (🔵 current zones) →
> Design Principles → Bloat Audit → Lean MVP Contract + Worked Example (🟢 paths.py spec)
> → Well-Runner (🟢) → DAG Mechanics → Zone-A Narrowing (🟢 the frame-contract split) →
> Target Model → Open Questions → Pause/Next-steps.

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
the earlier fan point are 🟢 TARGET — see "Zone-A Narrowing" below. Here, four zones
(stitching pulled out of bootstrap into its own image-materialization zone).

### Zone A — EXPERIMENT-GRAIN BOOTSTRAP  (needs the raw blob before wells exist) 🔵
Family: `experiment_metadata/` · all stages `fanout=experiment`
| Stage | Output (under `{output_root}`) | Why it can't be per-well |
|---|---|---|
| `normalize_plate_metadata` | `experiment_metadata/{exp}/plate_metadata.csv` | the plate *is* the experiment |
| `extract_scope_metadata_{scope}` | `experiment_metadata/{exp}/scope_metadata__{scope}.csv` | reads the whole raw file |
| `map_series_to_wells` | `experiment_metadata/{exp}/series_well_mapping.csv` (+`.provenance.json`) | **this is where wells are discovered** |
| `apply_series_mapping` | `experiment_metadata/{exp}/scope_metadata_mapped.csv` (+`.validated`) | joins mapping onto scope rows |
| `build_frame_contract` | `experiment_metadata/{exp}/frame_contract.csv` | 🔵 **CURRENT** canonical frame-level contract; feeds the checkpoint. 🟢 TARGET splits this per-well (Zone-A Narrowing). |
| `discover_wells` *(checkpoint)* | `experiment_metadata/{exp}/wells.txt` | filters contract wells → the canonical well list (global `well_id`) |

### ⟱ FAN POINT ⟱
- **🔵 CURRENT:** `discover_wells` checkpoint reads **`frame_contract.csv`** (Snakefile:427)
  → `wells.txt`. So the fan currently sits *after* stitching + frame-contract build.
- **🟢 TARGET:** `discover_wells_from_metadata` reads **`series_well_mapping.csv`** → moves
  the fan *earlier* (before stitching). Same checkpoint mechanism, earlier input. (See
  Zone-A Narrowing.)

The well list is a **first-class artifact** either way: `wells.txt`, keyed on global
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
work is **rewiring**, not rewriting algorithms.

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
  validate_merged_X`. Merged artifacts become thin **publication products** at the end of
  the DAG, not a mid-pipeline barrier. Snakemake then gives incremental single-well
  recompute for free.

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

> **`well` vs `well_id` in paths (decided, #6):** **registered** per-well artifacts use
> the global `well_id` (`per_well/20250912_B01/…`). **Image trees** (off-registry) use the
> local `well` (`stitched_ff_images/B01/…`) because they mirror microscope/plate folder
> layout. The distinction is deliberate, not legacy — registered = global, image-tree = local.

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
  (`built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` — no `yx1`/`keyence`),
  and `scope_metadata_mapped.csv` has already dropped the token. So the microscope
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

### The engine (≈20 lines, enforces `fanout`)
```python
def artifact_path(root, stage, artifact, experiment_id, *,
                  path_mode="experiment", well_id=None, format_vars=None):
    spec     = STAGES[stage]
    fanout   = spec["fanout"]
    filename = spec["artifacts"][artifact].format(**(format_vars or {}))
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
#   → ...mask_geometry_metrics.csv.validated   (derived suffix; matches existing on-disk convention)
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
        shards = lambda wc: checkpoint_well_shards(checkpoints, "mask_geometry", "metrics", wc)
        #   → one DECLARED per-well path per active_well (built via artifact_path path_mode="per_well")
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

# well selection + shard collection (lib/well_runner.py)
def active_wells(checkpoints, experiment, config, wildcards) -> list[str]:
    """The ONE definition of 'which wells run.' discovered ∩ target. Fail loud on a
    requested-but-missing well. (excluded = deferred future hook, not built.)"""
    discovered = read_wells(checkpoints.discover_wells.get(experiment=experiment).output.wells_txt)
    target = config.get("target_wells") or discovered
    active = [w for w in discovered if w in target]
    _validate_requested_exist(active, target, discovered)   # missing requested → raise
    return active

def checkpoint_well_shards(checkpoints, stage, artifact, wildcards) -> list[str]:
    """The ONE sanctioned way to build a merge's input list: forces the discover_wells
    checkpoint, returns DECLARED per-well shard paths for active_wells. Never a glob."""
    return [artifact_path(ROOT, stage, artifact, wildcards.experiment,
                          path_mode="per_well", well_id=w)
            for w in active_wells(checkpoints, wildcards.experiment, config, wildcards)]
```

**Three well-lists (keep distinct):**
| List | Meaning | Role in the DAG |
|---|---|---|
| `discovered_wells` | metadata says these exist (`wells.txt`) | discovery output |
| `active_wells` | `discovered ∩ target` | **the control list** — declared inputs, fail-loud |
| `validated_wells` | active wells whose per-well validation passed | **OUTCOME, never a dependency** (off-spine, like a merged view) |

> ☠️ **`validated_wells` must never gate the merge.** Merge depends on `active_wells`;
> an active well failing validation = **hard error**, not a silent shrink of the set.
> (Same skull rule as merged files: observed results never drive the DAG.)

**This is plan Win 4 + Win 5 made concrete:** one `select_active_wells()` = one well-selection
source (Win 4, delete the `targets.py` copy); checkpoint is truth, config only filters
(Win 5). Also folds in the cleanup of the **two redundant discovery paths** today
(`_wells_from_mapping` on local `well_index` vs. the checkpoint on global `well_id` →
collapse to the checkpoint). *(mdcolon to elaborate on the discovery-path specifics.)*

### What's actually IN `lib/well_runner.py` (the full contents)

Five things, no more. It is glue, not logic:

| Function | Job | Reads | Returns |
|---|---|---|---|
| `select_active_wells(discovered, target, exp)` | **PURE** list arithmetic (unit-testable, no Snakemake) | args only | `list[well_id]` |
| `read_wells(path)` | parse `wells.txt` → `list[well_id]` | the file | `list[str]` |
| `active_wells_from_checkpoint(checkpoints, wc, config)` | Snakemake glue: get checkpoint → `read_wells` → `select_active_wells` | checkpoint + config | `list[well_id]` |
| `checkpoint_well_shards(*, checkpoints, wc, config, root, stage, artifact, ...)` | merge input list (declared, never globbed) | the above + `artifact_path` | `list[path]` |
| `WellRun` *(optional value object)* | bind `(exp, well, well_id, root)`; hand ONE object to a per-well entrypoint so it never re-derives shape | — | dataclass |

> **Pure-vs-glue split (#8/#9, decided):** the selection *logic* (`select_active_wells`)
> is **pure** — no `checkpoints`, no globals — so it's unit-testable "without summoning
> Snakemake from the basement." The Snakemake-aware wrapper is a thin shell around it.
> Likewise `checkpoint_well_shards` takes `root`/`config` as **explicit params**, never
> reads them as module globals (specs that depend on unmentioned globals become haunted).

It does **not** contain: ID minting (→ `identifiers/`), path layout (→ `paths.py`), or
any staleness logic (→ Snakemake). If a function here starts computing *what a file is
named* or *what a row means*, it has crossed into the wrong kingdom.

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

> Machine knobs (`output_root`, `python`, `device`) live in **`env.yaml`** (Scope 3), not
> here. The well-runner takes `output_root` as a **parameter**, never derives it — that's
> why Scope 3 precedes the runner.

### How it interfaces with Snakemake tasks (the wiring)

Two touch points, both at the **rule boundary** (never inside core functions):

**(1) Per-well rules — expand the DAG over `active_wells`:**
```python
rule compute_mask_geometry_well:
    input:   seg = lambda wc: artifact_path(DATA_ROOT, "segmentation_tracking", "csv",
                                wc.experiment, path_mode="per_well", well_id=wc.well_id)
    output:  out = lambda wc: artifact_path(DATA_ROOT, "mask_geometry", "metrics",
                                wc.experiment, path_mode="per_well", well_id=wc.well_id)
```
**(2) Merge rules — collect shards via the runner (declared inputs):**
```python
rule merge_mask_geometry:
    input:   shards = lambda wc: checkpoint_well_shards(checkpoints, "mask_geometry", "metrics", wc)
    output:  merged = lambda wc: artifact_path(DATA_ROOT, "mask_geometry", "metrics",
                                wc.experiment, path_mode="merged")
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
So the full chain per stage: **rule (expands over active_wells) → tasks verb (resolves
paths) → core fn (concrete paths).** Three layers, each in its own kingdom. Building
`tasks.py` is Win 2 (independent, eases Scope 5); until then rules call `-m module`
directly and the well-runner helpers are imported into the Snakefile at parse time.

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
> artifact (`wells.txt`) as a well-local input is **fine** — it's upstream of the fan.
> Reading a *merged view* (post-fan, assembled-from-shards) is **forbidden**. The test:
> was this file made by collapsing per-well shards? If yes, don't depend on it.

**Scope-5 fix item:** `surface_area_qc` currently reads `consolidated_features` (merged)
— an accidental merge wall. It does per-snip work vs. an external curve, so it converts
cleanly to reading per-well shards. (Tenets 2, 7)

### Merge-DAG construction (how the DAG auto-detects the right wells)
At parse time Snakemake doesn't know how many wells exist (it's runtime data). The
**checkpoint** resolves this: `checkpoints.discover_wells.get(...)` forces a DAG re-plan
with the discovered list. Merge inputs must therefore be a **checkpoint-derived,
DECLARED `input:` list** (via `checkpoint_well_shards`) — the only pattern that gives
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
| 1 | `discover_wells_from_metadata` *(checkpoint)* | `series_well_mapping.csv` (**metadata only — no images**) | `wells.txt` | experiment (fan point, moved **earlier**) |
| 2 | `stitch_well` | this well's raw images + its `scope_metadata_mapped` rows | `built_image_data/{exp}/stitched_ff_images/{well}/{channel}/` | per-well image tree (**off-registry**) |
| 3 | `validate_frame_contract_well` | this well's **images (Step 2)** + this well's **metadata rows** | `frame_contract/{exp}/per_well/{well_id}/frame_contract.csv` | `per_well_then_merge` |

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
- Discovery reads `series_well_mapping` / `scope_metadata_mapped` — **already
  microscope-converged** (Tenet 5), so early discovery is microscope-agnostic.

**Naming adopted:** `discover_wells_from_metadata` (clarifies: metadata-only, doesn't
check images) and `validate_frame_contract_well` (clarifies: it's a *check*, not just
row formatting).

---

## ❓ OPEN QUESTIONS (next session — don't lose these)

1. ~~**Where exactly is the fan point?**~~ **RESOLVED (with current/target split):**
   🔵 CURRENT fan = `discover_wells` checkpoint reading `frame_contract.csv` (Snakefile:427).
   🟢 TARGET fan = `discover_wells_from_metadata` reading `series_well_mapping.csv` (moved
   earlier, before stitching). Stitching's `_wells_from_mapping` (local `well_index`) is a
   *separate, redundant* path → consolidate to the checkpoint (Win 4/5).
2. ~~**Is Zone C truly per-well-able?**~~ **RESOLVED (mdcolon known fact + verified):
   100% per-snip/per-embryo, zero cross-well. No cohort QC stage exists
   (`build_sa_reference.py` is offline, not a Snakefile rule). Green light.**
3. ~~**Well list as first-class artifact?**~~ **RESOLVED: `wells.txt` from the checkpoint
   already is one (global `well_id`).** Cleanup = make it the *single* source;
   `selected_wells.txt` becomes a pure input filter (Win 5: checkpoint is truth, config
   only filters); `series_well_mapping.csv` stays raw discovery.
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
  `select_active_wells()` (Win 4/5). *(mdcolon to elaborate on specifics.)*
- **Scope-2 migration touch:** `discover_wells` reads `frame_contract.csv`'s `well_index`
  column (Snakefile:448), which Scope 2 deletes → this checkpoint is a migration site.

---

## ⏸️ WHERE WE PAUSED (2026-06-02)

The grain model is **settled**. 🔵 CURRENT: four zones — A (bootstrap) → fan
(`discover_wells` checkpoint, reads frame_contract) → B0 (stitching) → B
(segmentation/snips per-well) → C (merged). 🟢 TARGET: pushes B/C fully per-well, splits
the frame contract per-well (Zone-A narrowing), moves the fan earlier
(`discover_wells_from_metadata`). The registry is trimmed
to a **lean path resolver** (Bloat Audit): per-stage `family`/`fanout`/optional
`subfolder`, per-artifact filename template; sentinels are derived helpers;
`execution`/`grain`/`schema`/`kind`/`src_family` are deferred (each named with its future
reader). The `fanout` vs `execution` distinction stays as a *documented concept*, not a
field. Zone-C per-well-safety: **math** owner-confirmed + spot-checked; **wiring** still
has merge-wall reads to convert (not the same claim). scoped_runs demoted to scratch. The
**well-runner** provides *mechanism, not staleness* (`select_active_wells` pure +
checkpoint glue + `checkpoint_well_shards`); imports IDs from `identifiers/`, never mints.
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
2. **Implement `lib/paths.py`** from the Lean MVP Contract + Worked Example (fanout-
   enforced). Safe to build now; per-well paths use `well_id` but aren't *wired* into the
   DAG until Scope 2.
3. **Scope 2** — migrate `well`/`well_id` semantics in schemas + call sites; regenerate.
4. **Implement `lib/well_runner.py`** on normalized IDs (`select_active_wells` pure +
   checkpoint glue + `checkpoint_well_shards`).
5. **Wire one stage** end-to-end through the registry as the proof, then replicate.
   *(Win 2 `tasks.py` verbs can land independently anytime to ease this.)*

---

## 🗂️ AUDIT TODO (raised 2026-06-02, not yet actioned)
- **Verify `.validated` sentinel convention is uniform** (`{filename}.validated` for *all*
  sentinels). If any rule uses `.{stem}.validated` or `{stem}.validated`, either normalize
  or give `validated_path` options. *(Assumption in the MVP contract, not yet verified.)*
- **Optional exhaustive Zone-C grep-audit** (module / searched-for / result table) to
  upgrade "owner-confirmed + spot-check" → "exhaustively verified."
- **Consider splitting this doc** if it keeps growing: Conceptual (north star) vs
  Implementation Spec. For now the CURRENT/TARGET tags + document map at top carry the load.
