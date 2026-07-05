# Pipeline Overview

> **Draft outline.** Section headers + one-line stubs only. We refine each section in place.
> This is an *overview / glossary* for someone new to the pipeline — short and visual. Real detail
> lives in the referenced markdown files, not here.

---

## Part A — The big picture

### A1. The scope/well overlap regime  *(lead diagram)*
The fundamental running unit is **per well** — but which wells exist can't be known until metadata
is run, and stitching mechanics differ per scope. So the *scope world* and the *well world* overlap.
Bookends: metadata ingest opens the scope world; **materialization closes it**. Everything after is
per-well and scope-free.

_TODO: ASCII diagram — scopes in → [scope world] → discover_wells → materialization → [well world]._

### A2. Where each stage lies  *(river diagram)*
Two words the rest of this doc leans on, and one contains the other:

```
stage                 a major phase of the pipeline — a top-level output directory
├── step_A              a unit of work inside it — one registry entry, one Snakemake rule
├── step_B              a unit of work inside it — one registry entry, one Snakemake rule
└── …
```

A **stage** is the larger container; it is **composed of steps**. Stages organize the pipeline
conceptually; the steps inside them perform the actual computation.

The five stages in order, one line each:

- **acquisition** — has **two parts**:
  1. **Metadata ingestion** — plate + scope metadata ingest, leading to **well discovery**. This is
     where "which wells exist" is answered.
  2. **Materialization** — after wells are discovered, materialize each well's images. Still
     **per-scope** here (stitching mechanics differ), with its own nuances.
- **object_extraction** — detect, seg-track, snip creation, physical_embryo_id + snip_id minting, auxiliary masks.
- **feature_extraction** — predicted stage (hpf), embeddings, curvature, mask measurements, viability. _snip level._
- **quality_control** — motion-blur, focus, death detection, surface-area, snip_qc gate. _snip level._
- **analysis_ready** — final report; merge plate metadata with embryos.

_TODO: ASCII diagram — the five-stage river (with acquisition's ingest→discovery→materialize split)._

---

## Part B — The machinery (glossary)
Brief — one or two sentences + a link each. Not full sections.

### B1. Identifiers
The **single place** that mints and parses object names — nothing else builds or splits an id.
Id strings are **opaque** outside `shared/identifiers/`: never mint with an f-string or split with
`.split("_")`; use the constructors (mint) and parsers (decompose).

**Given by acquisition** — the raw coordinates of a frame:
`experiment_id`, `well_index`, `channel_id`, `time_index`, `z_index`.

**Minted by the pipeline:**
- `well_id` (= `experiment_id` + `well_index`) — **the unit the pipeline runs on.** Wells are
  discovered, then every stage fans out per `well_id`.
- `image_id` (= `well_id` + `channel_id` + `time_index`) — the canonical name of one frame.
- `physical_embryo_id` → `embryo_id` → `snip_id` — minted in object extraction, once we've found the
  animals. The nesting: *animal → animal-in-a-channel → animal-at-a-time.*
  - `physical_embryo_id` — from **detection** (the animal we found).
  - `embryo_id` = `physical_embryo_id` + `channel_id`.
  - `snip_id` = `embryo_id` + `time_index`.

> **z-stacks:** for a projection there's one frame per `(channel, time)`; for a z-stack we add a
> `z_index`, so `image_id` becomes `{well_id}_{channel_id}_z{z_index}_t{time_index}` — one id per
> plane.

→ Full grammar (mask_id, track_id, parsers, track→embryo index conversion):
`../shared/identifiers/README.md`.

### B2. Config
The **run recipe** — the knobs for one run (`pipeline_orchestrator/config.yaml`):
- **which experiments** to process (`experiments:`)
- **which wells** — optional per-experiment subset for smoke runs / targeted reruns (defaults to
  *all discovered*)
- **which microscope** (`microscope: YX1 | keyence`)
- **front-end mode** (`native` — ingest from a microscope | `dropin` — an external dataset's own
  frame_inventory manifest + plate metadata, no scope)
- per-stage knobs (device `auto`/`cpu`/`cuda`, overwrite, reference-file paths)

**The principle:** config is a **science choice — *which* wells/features — never a code contract.**
It does *not* decide where files land (→ B3, `paths.py`) or how ids are formed (→ B1, identifiers).
It can **filter** the run set but never invent a well: naming a well that wasn't discovered fails
loud. (Config reaches rules as a `params:` value, not a file `input:`, so touching it doesn't
cascade a rebuild — see `pipeline_file_philosophy.md`.)

### B3. paths.py — stages & steps
The **single registry** (`PIPELINE_STEPS`) answering *where does every file the pipeline produces
land?* Read by both the Snakefile (at plan time) and the Python entrypoints (at run time) — so rule
outputs and code outputs are identical by construction. If two places disagree about a path, one
bypassed the registry.

The layout it encodes — this is where **stages and steps become folders**:

```
output_root / {stage} / {experiment} / [{product_dir}/] [per_well/{well_id}/] {artifact}
```

**Reading a step: input → execution → output.** Every step is a transform. The registry pins the
step on three facts (its input is wired in the rule):

- **stage** — the folder / pipeline phase the step's output lands under.
- **execution** — *how the compute is dispatched:*
  - `per_well`: one job per well — load a well, compute it, write its result. N wells → N jobs.
  - `run_batch`: **one** job for the whole run set — load the model once, loop over *all* wells
    inside that single job, write everything before exiting. Used when model / GPU load dominates
    per-well cost (SAM2, the VAE), so you pay it once, not N times.
- **output (fanout)** — *what lands on disk:*
  - `EXPERIMENT`: one artifact for the whole experiment.
  - `PER_WELL_THEN_MERGE`: a per-well shard, then a merge step concatenates the shards into the
    experiment-level table. This is how "everything runs per well" shows up on disk.
- **artifacts** — the filenames. `.validated` / `.provenance.json` sidecars are *derived*, never
  listed.

Execution and output are **independent** axes: execution is *how many jobs and what they load*,
output is *how many files land*. In particular `run_batch` (one job) still writes **per-well
output** — the same shards a `per_well` step would, merged identically downstream. Batching is a
compute optimization, invisible to the output contract.

**The boundary:** paths.py only *substitutes* id tokens into filename templates — it never mints or
splits an id (→ B1). Identity flows *in*; paths never reach back.

### B4. well_runner
The code that performs the per-well lifecycle B3 declared: **where each well's shard lands, and how
the present shards merge** into one experiment-level table.

- **Which wells run:** discovered ∩ config filter. Discovery (the fan point) decides which wells
  *exist*; config only narrows. (Filter, never invent — → B2.)
- **The merge is presence-based.** It gathers whatever per-well shards exist *right now* and stacks
  them — so even running one well at a time yields a valid merged table of what's done so far, and it
  grows as wells complete. (It only picks up shards marked `.validated`, so a half-written well is
  skipped, not merged.)
- **The merged table is a *view*, for inspection.** It is not a new unit of work — nothing
  downstream computes on "the whole experiment" as its grain. **The fundamental unit stays
  per-well;** the merge is just a convenient roll-up over the shards.

(Config accepts a local `B01` or a global `20250912_B01`; it's normalized to a global `well_id` once,
at the seam — no bare slug travels inside.)

### B5. Contracts → spines & payloads
Every product table has a `contract.py` **next to the code that mints it** — the single source of
truth for that table's shape (its columns and what makes it valid). Downstream steps import it; the
validator (→ B6) enforces it. Columns come in **named families**, uniform across every product:

- **spine** — *who is this row about?* The identity columns, **imported from where identity was
  minted, never re-typed.** A snip-grain table carries its full lineage:
  `experiment_id → well_id → physical_embryo_id → embryo_id → snip_id` — its own level *plus all
  parents*.
- **provenance** — *where did the row come from?* Source/context columns needed to locate or explain
  the inputs (e.g. the `image_id` / `time_index` / `channel_id` of the frame a snip was cropped from).
- **payload** — *what does this product add?* The columns this step, and only this step, owns — its
  actual contribution.
- **table** — the full emitted table = spine + provenance? + payload; the ordered column list the
  validator checks.

**Spine is imported, not re-declared** (the identity-spine law). Identity travels forward as explicit
columns; downstream tables read the `physical_embryo_id` column, never re-parse it out of an id
string. One minting site (`snip_identity_contract.py`), imported everywhere.

*(A payload column must earn its place: "why is this a column if no consumer uses it?" —
debug/calibration columns stay out of the canonical contract.)*

### B6. Validation + tests
**One authoritative validator per contract, living in the same `contract.py` as the schema** — the
column families and the check that enforces them sit side by side, owned by the product that mints
the table. It checks the **spine first, then the payload** — identity must be right before the
measurements mean anything.

- **One validator, two moments** — a `check_sources=` flag, not two validators that drift.
  - **When writing the table** (`check_sources=False`): check only *this* table — its columns are
    present, typed, non-null, spine intact.
  - **When a downstream step reads it** (`check_sources=True`): do all of the above **plus** re-check
    that its upstream inputs still agree — e.g. every `physical_embryo_id` in the table really exists
    in the registry it came from. The stricter check fires at the moment it matters: right before
    something consumes the table.
- **Fail loud; the message names the fix.** An error says what's wrong *and what to do* — which
  column, which join to fix — not a bare stack trace.

**Tests mirror the src tree** — `src/data_pipeline/X/` is tested by `tests/data_pipeline/X/` — and
**pin contracts, not implementation**: resolved paths, failure modes, the important words in an error
message, so wording can improve without breaking the suite.

### B7. Reports
Reports follow from one fact, in order:

**1. Reports are terminal.** Nothing consumes a report; a report never becomes an input (only the
`reports` target asks for them). So reports are **not contracts** — everything else follows. This is
why their checking can be looser than a product table's.

**2. A report only has access to the scope it's given.** There are two types:
- **per-step report** — sees only *that step's* inputs and output; it summarizes one step and
  emits **PNG** leaves.
- **per-stage report** — sees only *the per-step report artifacts of its stage*. It is **not** a
  stage-wide-data summary; it's a **roll-up** — a document assembler that gathers the stage's PNGs
  into one HTML + PDF. It embeds the leaves; it does not re-read the pipeline.

```
step → PNG
step → PNG        ──▶  stage rollup (HTML + PDF)  — assembles the PNGs, nothing more
step → PNG
```

**3. `analysis_ready` is the rule working, not an exception.** Because the `analysis_ready` step
consumes the *fully joined* dataset, its report naturally has access to every integrated annotation
(genotype, developmental stage, embeddings). Visualizations combining these belong here because no
earlier step possesses that complete view — same rule (a report sees what its step sees), broad
inputs.

---

## Part C — Detail per stage

### C1. Acquisition
Incl. `materialized_image_paths.py`; **produce_codes / stack projection** (materialized-image write
policy, z-stack vs projection); the snapshot-vs-timepoint required-columns nuance. _stub._

### C2. Object extraction — the snip world
Still run per well, but now computed per `snip_id` (spine column vs payload column). _stub._

### C3. Feature extraction
(1) overview, (2) each step. _stub._

### C4. Quality control
(1) overview, (2) what matters for QC generally, (3) each step. _stub._

### C5. Analysis ready
Final merge + report. _stub._
