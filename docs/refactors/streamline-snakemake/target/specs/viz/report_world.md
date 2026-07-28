# Report World — per-step visualization artifacts

**Status:** LIVE, updated 2026-07-03 (was planning spec 2026-07-02). The per-step report tier is
wired end-to-end, and a coarser **stage-rollup** tier sits on top of it. This doc defines what a
*report* is, where its code lives, how it wires into the DAG, and the acceptance bar for each one.
The histogram + gallery toolkit from commit `8945395a` (`viz/reporting.py` +
`quality_control/reporting/`) is now its rendering layer; the ad-hoc driver
(`results/mcolon/20260702_qc_reporting_v1/build_review_index.py`) is superseded for DAG use by the
stage rollups. **See the "Current State" section below for exactly what is wired today.**

**Doctrine:** a report is a per-step, cohort-level visualization that shows stats about *that step's
output* and is **consumed by nothing downstream**. Reports never mint identity and never feed another
stage. They are **terminal leaves** — in the DAG and in the import graph: nothing lists a report as an
input and nothing imports a `report.py`.

> **Clarification (2026-07-03): "terminal" is about graph shape, not about who requests it.** A report
> being a leaf (nothing consumes it, nothing imports it) is independent of whether a target *asks* for
> it. Reports may be requested by `rule all` — doing so puts report generation on a default run without
> making the report any less terminal. A failing report cannot corrupt or roll back upstream data,
> because the data products it reads are already built *before* the report runs; the report is the end
> of the line. Run with `--keep-going` to let unrelated targets finish if one report errors. (This
> supersedes the original spec's "reports are requested by their own target, never `rule all`" — that
> was a *convention* about when reports build, not part of the terminal *invariant*.)

---

## What A Report Is (three hard rules)

1. **Terminal — nothing consumes it.** A report is a DAG leaf. No other rule lists a report artifact as
   input; no module imports a step's `report.py`. This is the defining property. A report exists to let
   a human *see* a step's output, nothing more. If anything ever consumes a report, it is no longer a
   report and the design has drifted. **Terminal means the report *gates nothing* — it does not mean the
   report may *read anything* (see rule 2).**
2. **Scoped to its own step — a report reads only what its step reads or produces.** A report hangs off
   the merge node of its own step and reads that step's **merged** output (plus any input that step
   *already* consumes), through `artifact_path(step=..., path_mode="merged")`. **A report does not cross
   stage boundaries.** It cannot reach an artifact that is not part of its step's input surface — being
   terminal buys it *no* extra reach. `motion_blur_qc` plots over `time_index` not because stage "isn't
   worth it," but because `predicted_stage_hpf` is simply **not one of motion_blur_qc's inputs**, so its
   report cannot see it. The step *is* the scope.
3. **The temporal axis is whatever the step already has; default `time_index`.** `time_index` is always
   present and spine-adjacent, so it is the baseline axis. A report may use a richer axis
   (`predicted_stage_hpf`, `elapsed_time_hr`) **only when that column is already in its step's inputs** —
   e.g. `surface_area_qc` genuinely depends on `stage_predictions`, so its report gets stage on the
   x-axis for free. This is rule 2 applied to the axis, not a separate rule.

> **The one exception is `analysis_ready`, and it isn't an exception to the scope rule — it's the scope
> rule working as intended.** `analysis_ready` is the single step whose input surface *is* the whole
> joined DAG (embeddings + `plate_metadata` + `predicted_stage_hpf` + genotype). So a genotype/stage-
> colored 2D PCA belongs to **`analysis_ready`'s report**, not the embedding step's — the embedding step
> has no metadata, and its report may not reach across the boundary to get it. `analysis_ready`'s report
> reading all of that is not a boundary crossing; that is its step's input surface.

> One line: **inputs are registry-strict** (resolved through `artifact_path`, so they don't rot when a
> path changes); **outputs are registry-light** (a report is a view, not a pipeline artifact, so it is
> a terminal leaf that never gates `analysis_ready` — regardless of whether `reports` or `rule all`
> requests it).

---

## Current State (2026-07-03)

**Wired per-step reports** (registry row + `report.py` + `cmd_*` + `.smk` rule):

| Stage | Report step | `report.py` |
|---|---|---|
| object_extraction | `physical_embryo_registry_report` | `object_extraction/segmentation/physical_embryo_registry/report.py` |
| feature_extraction | `mask_geometry_report` | `feature_extraction/mask_geometry/report.py` |
| quality_control | `surface_area_qc_report` | `quality_control/surface_area_qc/report.py` |
| quality_control | `death_detection_report` | `quality_control/death_detection/report.py` |
| quality_control | `snip_qc_report` | `quality_control/snip_qc/report.py` |

**Wired stage rollups** (`viz/stage_report.py` + `rules/stage_reports.smk`): one each for
`object_extraction`, `feature_extraction`, `quality_control` → `<stage>/<exp>/report/<exp>_<stage>_report.{html,pdf}`.
The `quality_control` rollup currently embeds all three QC per-step reports.

**Targets:** `snakemake reports` (all reports + rollups); `snakemake` / `rule all` (data products +
the 3 rollups, which pull their per-step reports transitively).

**Tests:** `tests/data_pipeline/viz/test_reports_are_terminal.py` (import-graph + DAG terminal-leaf
guard), `tests/data_pipeline/viz/test_reporting.py` (renderer toolkit). *No dedicated test yet for the
`stage_report.py` rollup builder or `html_report.py` assembler — a gap worth closing.*

**Rendering/assembly libraries:** `viz/reporting.py` (the six renderers), `viz/html_report.py`
(`HtmlReport` HTML+PDF assembler, docs in `viz/html_report_usage.md`), `viz/stage_report.py` (rollup
builder). The ad-hoc `results/mcolon/20260702_qc_reporting_v1/build_review_index.py` still works for a
hand-run all-stages page but is superseded for DAG use by the stage rollups.

**Not yet wired:** per-step reports for the many candidate steps in the roster below
(`frame_detections`, `frame_inventory`, `pose_kinematics`, `stage_predictions`, `fraction_alive`,
`mask_quality_qc`, `focus_qc`, `motion_blur_qc`, `death_event`, `latent_embeddings`, `analysis_ready`).

---

## Two Layers: dumb renderers vs. per-step reports

Reports split the same way features and QC do (`compute.py` vs `contract.py` vs `entrypoint.py`):

| Layer | Lives in | Owns | Knows about |
|---|---|---|---|
| **Renderers** | `data_pipeline/viz/reporting.py` | all matplotlib/PIL drawing — the **six shared renderers** below | a DataFrame + column(s) + a grain/cutoff. **No** step, product, path, or registry opinion. |
| **Report specs** | each product's `contract.py` / a `reporting/` spec module | which metric, which cutoff, which fail-direction, which grain/axis a report uses | column *names* (from its own contract), not paths |
| **Per-step report** | each product's `report.py` | pull the step's merged output, derive any report-only values, hand columns + grain to a shared renderer, save the bundle | `artifact_path` (input), `contract.py` (columns), `viz/reporting.py` (drawing) |

The boundary that keeps core clean is **rendering vs. everything else**, not directory distance. All
matplotlib lives in `viz/reporting.py`; `report.py` imports the renderers and never calls matplotlib
directly. So `compute.py` and `contract.py` never grow a matplotlib import, no matter where `report.py`
sits.

### The six shared renderers

Walking the whole-DAG candidate roster (below) collapsed most "custom" reports into **a few recurring
shapes**. The insight: **so many step reports are just "a per-`well_id` or per-`physical_embryo_id`
quantity over `time_index`" or "the distribution of a per-group count"** that those shapes belong in the
shared toolkit, parameterized by *(column, grain)* — not re-implemented per step. A step's `report.py`
then becomes ~3 lines: pick the renderer, pass your column and grain. The toolkit is:

| # | Renderer | Shape | Signature (sketch) | Instantiated by |
|---|---|---|---|---|
| **A** | **grouped-trace-over-time** (the "curtain") | one faint low-alpha line per group vs `time_index`, overplotted; optional event-marker overlay | `plot_grouped_traces(df, value_col, *, group_col, time_col="time_index", marks=None)` | mortality curtain, viability curtain, pose speed-over-time, stage trajectory, mask-validity-over-time, detections-per-well-over-time |
| **B** | **count-per-group distribution** | histogram of a per-group **row count** (how many X per group) — bins *counts of rows*, not a measured value | `plot_count_per_group(df, *, group_col)` | embryos-per-well, snips-per-embryo, detections-per-frame, track-length |
| **C** | **metric histogram** (v1) | raw-metric histogram, bars colored pass/fail by a cutoff | `plot_metric_histogram(...)` *(exists)* | confidence, area, fraction-alive dist |
| **D** | **histogram grid** | one gridded multi-panel PNG, one C-panel per column | `plot_histogram_grid(df, columns, *, cutoffs=None)` | mask_geometry / curvature feature grids |
| **E** | **cutoff-relative gallery** (v1) | 4-band image gallery by distance to cutoff | `render_quartile_gallery(...)` *(exists)* | low-confidence detections, invalid masks, focus/motion |
| **F** | **metric-vs-reference-band scatter** (v1) | metric vs covariate against a per-row band | `plot_metric_vs_reference(...)` *(exists)* | surface_area_qc |

C/E/F already exist from commit `8945395a`. **A, B, and D are the toolkit additions this spec commits
to.** Renderer **B (count-per-group) is deliberately distinct from C (metric histogram)** — they look
similar but bin different things (rows-per-group vs. a measured value), and conflating them was a recurring
error in the candidate review. Keep them separate renderers.

`report.py` **may compute report-only values** — it is not restricted to passing columns through
untouched. Anything recomputable from its step's output and shown only in the report is fair game: e.g.
death_detection's report derives *the time each embryo died* from `persistence_dead_flag` + the trace and
shows it as a histogram. That derivation lives in `report.py` precisely because it is recomputable, shown
only in the report, and consumed by nothing — the same reason the death spec keeps it *out* of the flag
table.

### Co-location: `report.py` lives next to the step

Report code lives **in the product folder**, beside `entrypoint.py`:

```text
quality_control/death_detection/
  contract.py
  compute.py
  persistence.py
  death_event.py
  grain_reconciliation.py
  entrypoint.py
  report.py          ← death-persistence report (may derive report-only values)
  __init__.py
```

Rationale: a report reads exactly one thing — its step's output — and breaks for the *same reason* that
step changes (a renamed column, a split flag). Code that changes together lives together. Unlike `tests/`
(a mirror tree because tests are bulk, runner-discovered, and run as a unit), a report is one small file
that only matters in its step's context; a parallel tree would buy nothing and add a second place to
navigate.

**Any step *may* have a `report.py` — it is the always-available slot for that step's visualization**,
not a privilege earned only by "custom" steps. What varies is how much it contains, not whether it is
allowed:

- most of the time `report.py` is thin — it just imports helpers from `viz/reporting.py` and points them
  at its columns (the stock histogram + gallery). A step whose report is fully covered by a spec entry
  (`QC_REPORT_SPECS`) plus the shared renderer may not need its own file at all;
- when a step wants a custom view or a report-only derived value, that code goes in *its own* `report.py`
  — the slot is already there.

**One report *build*, but it may emit more than one artifact.** `report.py` is not restricted to a
single view: a report step, like any other step, may declare **multiple artifacts under one registry
row** (the same pattern as `death_detection` emitting `death_detection_qc` + `death_event`, or
`ingest_scope_metadata` emitting `raw` + `acquisition_inventory` from one build). Emit separate PNGs when
a view is **independently useful** — e.g. the death-time histogram is a slide-ready summary you'd open
without the curtain. When two views are always read together, make them subplots of one figure instead.
Keep it simple: don't proliferate near-identical PNGs, and don't add a config flag to toggle individual
views on and off — that is over-engineering a cheap terminal build.

**Import rule (bright line):** `report.py` may import from its own `contract.py` and from
`viz/reporting.py`. **Nothing may import from any `report.py`.** A report is a terminal leaf in code
exactly as it is in the DAG; an import of a `report.py` is a doctrine violation and should be lintable.

`report.py` is the per-step analog of `entrypoint.py`:

| `entrypoint.py` | `report.py` |
|---|---|
| read inputs → compute → validate → **write table** | read merged output → derive/render → **write PNG bundle** |

---

## Stock vs. custom reports

- **Stock report** — a raw-metric histogram (bars colored pass/fail by the true cutoff) plus a
  cutoff-relative image gallery, driven entirely by a spec entry. No per-step code. This is the
  `QC_REPORT_SPECS` / `QCReportSpec` shape from commit `8945395a`: name the raw metric column, the scalar
  cutoff, and the fail direction; the shared renderer does the rest. Products with **no persisted
  continuous metric** (`mask_quality_qc`, `snip_qc`) are deliberately spec-omitted — there is no scalar
  to histogram.
- **Custom report** — a `report.py` when the stock shape doesn't fit:
  - `surface_area_qc`: its "cutoff" is a **stage-interpolated reference band**, not a scalar, so a plain
    histogram against one fixed cutoff is misleading. It gets an area-vs-stage scatter against its
    reference band (`SurfaceAreaQCReportSpec` → `plot_metric_vs_reference` /
    `render_quartile_gallery_vs_band`).
  - `death_detection`: the mortality curtain (below).

---

## DAG Wiring

Every step's report is one Snakemake rule of a single shape:

```
merged step output  ──►  report rule  ──►  report bundle (terminal leaf)
```

The pipeline is already regular enough that a report reuses every existing mechanism — a registry row,
a `.smk` rule, a `tasks.py` subcommand, and an aggregate target — with **one** twist: the report is a
DAG leaf, so no *data* rule lists its output as input. It is requested by the `reports` target, and (via
the stage rollups) by `rule all` — see § 3 for how both work without compromising the leaf property.

### 1. Report outputs ARE registry citizens (a report *step*)

**Decided: a report is a full registry citizen — there is no `_reports/` scratch alternative.** A report
gets a real `PIPELINE_STEPS` row so its path is declarative like everything else, resolved through the
same `artifact_path(...)` / `step_dir(...)` helpers as any product (no separate `report_dir` helper is
needed — the registry row *is* the path source). The only difference from a normal step is that *no other
step's rule inputs name it*. This keeps the "no raw path strings anywhere" law intact for outputs too, and
it lands reports in the output tree beside the step they describe:

```python
# in PIPELINE_STEPS, alongside the step it reports on.
# ONE report step may declare MULTIPLE artifacts (like death_detection itself emits two tables).
"death_detection_report": {
    "stage": "quality_control",
    "product_dir": "death_detection/report",   # sits beside the death_detection tables
    "fanout": EXPERIMENT,                       # reports are cohort-level, always merged-grain
    "artifacts": {
        "curtain_png": "{experiment_id}_mortality_curtain.png",
        "death_time_png": "{experiment_id}_death_time_histogram.png",
    },
},
```

Which yields, for free via `artifact_path(...)`:

```
quality_control/<exp>/death_detection/report/<exp>_mortality_curtain.png
quality_control/<exp>/death_detection/report/<exp>_death_time_histogram.png
```

`fanout: EXPERIMENT` always — a report is a cohort view over the merged table, never a per-well shard.
There is no `.validated` sidecar: a PNG has no contract to validate (this is the one place the universal
validate-before-write rule does not apply, precisely because a report is not a consumed artifact).

> Convention: a report step is named `<step>_report` and nests under `<product_dir>/report/`. A single
> naming helper (`report_step_name(step)` / a `report/` product-dir suffix) can generate both so no one
> types them by hand.

### 2. The `.smk` rule attaches to the merged output

The report rule is the terminal twin of a merge rule. Its **input is the step's merged artifact** (via
the existing `rule_artifact(..., path_mode=PATH_MODE_MERGED)` helper already used everywhere in these
`.smk` files), and its **output is the report step's artifact**:

```python
rule death_detection_report:
    """Terminal report: mortality curtain + death-time histogram (two artifacts). Consumed by nothing."""
    input:
        qc=str(rule_artifact("death_detection_qc", "death_detection_qc", "{experiment}", path_mode=PATH_MODE_MERGED)),
        fraction_alive=str(rule_artifact("fraction_alive", "fraction_alive", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        curtain=str(rule_artifact("death_detection_report", "curtain_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        death_time=str(rule_artifact("death_detection_report", "death_time_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks death-detection-report \
          --death-detection-qc-csv "{input.qc}" \
          --fraction-alive-csv "{input.fraction_alive}" \
          --output-curtain-png "{output.curtain}" \
          --output-death-time-png "{output.death_time}"
        """
```

The report rule may take **any merged input its step already has** — `death_detection_report` reads the
merged `death_detection_qc` and merged `fraction_alive` because death_detection already consumes both.
It may **not** add an input outside that surface (no stage here). `analysis_ready`'s report rule, by the
same rule, legitimately takes the whole joined analysis-ready table because that *is* its step's input.

`report.py` lives in the product folder (`quality_control/death_detection/report.py`); the `tasks.py`
subcommand (`cmd_death_detection_report`) is the thin CLI adapter that resolves the merged input paths
handed in by the rule, calls `report.py`, and writes the PNG — exactly the `entrypoint.py` ↔ `tasks.py`
split every other step already uses.

### 3. Two ways to request reports: the `reports` target, and `rule all`

A report is a leaf, so it is only ever *requested*, never *depended on* by a data rule. Two targets
request them (both live in the root `Snakefile`):

```python
def _reports_targets(wc=None):
    # EVERY *_report step's artifacts (per-step AND rollups), for every experiment — the
    # `reports` target. Discovery is by suffix: step.endswith("_report").
    ...

def _stage_rollup_report_targets(wc=None):
    # ONLY the *_rollup_report steps. Requesting a rollup transitively pulls its stage's
    # per-step reports (the rollup rule lists them as inputs), so this one list drags in the
    # whole report tier without enumerating every per-step artifact. Used by `rule all`.
    ...

rule reports:
    input: _reports_targets            # snakemake reports -> every report + every rollup

rule all:
    default_target: True
    input:
        expand(... merged frame_masks ...),
        _stage_rollup_report_targets,  # a default run ALSO refreshes the rollup review pages
```

So: `snakemake reports` builds every report and rollup on demand; `snakemake` (the default `all`)
builds the data products **and** the 3 stage rollups (which drag their per-step reports along). A
single report is still buildable by its concrete output path, e.g.
`snakemake .../20250912_mortality_curtain.png`. This is the "outputs stay non-gating" guarantee
enforced structurally — no *non-report* rule names a report as input, so a report cannot gate a data
product regardless of who requests it.

**Discovery is by name suffix, not a hardcoded list.** `_report` = "a report" (per-step or rollup);
`_rollup_report` = "a rollup". The three collectors (`_reports_targets`, `_stage_rollup_report_targets`,
and `report_steps_for_stage` in `viz/stage_report.py`) all filter `PIPELINE_STEPS` on these suffixes.
**Consequence: adding a `<product>_report` registry row + its per-step rule is enough — the `reports`
target, the stage rollup for that product's stage, and `rule all` all absorb the new report with zero
edits to any of them.** (This is how `snip_qc_report` was picked up automatically.)

### Wiring checklist (per report)

1. add a `<step>_report` row to `PIPELINE_STEPS` (`fanout: EXPERIMENT`, `product_dir: <product>/report`);
2. add `report.py` in the product folder (imports `viz/reporting.py` + own `contract.py` only);
3. add a `cmd_<step>_report` subcommand in `tasks.py`;
4. add `rule <step>_report` in the product's `.smk` — input = merged step output(s), output = report artifact;
5. **nothing else** — the `_report` suffix means the `reports` target, the stage rollup, and `rule all`
   pick it up automatically (see "Discovery is by name suffix" above);
6. **do not** add the report artifact to any *non-report* rule's `input:` and **do not** import
   `report.py` anywhere — both are enforced by the terminal-leaf test
   (`tests/data_pipeline/viz/test_reports_are_terminal.py`).

---

## Stage Rollups — one review page per stage (a coarser terminal tier)

On top of the per-step reports sits a **stage rollup**: one `HtmlReport` page (HTML **and** a sibling
multi-page PDF) that gathers every per-step report PNG produced under a stage onto a single
reviewable/printable page. There is one rollup per stage that has reports:
`object_extraction`, `feature_extraction`, `quality_control`. (There is **no** cross-stage
experiment-level page — the tree is `stage/<exp>/…` with no shared experiment root, and the three
stage pages already cover everything. `build_review_index.py` remains for ad-hoc all-stages review.)

A rollup obeys the same doctrine as a per-step report — **terminal, renders nothing itself, embeds the
PNGs the per-step reports already wrote.** It is a leaf whose *inputs* happen to be other leaves.

**Two layers, one primitive.** The rollup is pure layout glue over two `viz/` modules:

| Layer | Lives in | Owns |
|---|---|---|
| **HTML/PDF assembler** | `viz/html_report.py` (`HtmlReport`) | a flat, path-style builder — `add_image/add_note(section=…, subsection=…)`, find-or-create buckets in insertion order; `write()` emits HTML + sibling PDF (PDF = one page per image, via Pillow, lazily imported). Stdlib-only otherwise. See `viz/html_report_usage.md`. |
| **Stage rollup builder** | `viz/stage_report.py` | `report_steps_for_stage(stage)` (registry discovery, excludes `*_rollup_report` so a rollup never embeds itself) + `stage_report_pngs(...)` + `build_stage_rollup_report(...)` — one section (the stage), one subsection per per-step report (its product), embedding that report's PNGs; a report with no PNG renders a "no report artifacts" stub. |

**DAG wiring (per rollup).** A `<stage>_rollup_report` registry row (`fanout: EXPERIMENT`,
`product_dir: report`, artifacts `index_html` + `index_pdf`), a rule in `rules/stage_reports.smk`, and
the shared `cmd_stage_rollup_report` / `stage-rollup-report` subcommand:

```python
rule quality_control_rollup_report:
    input:  lambda wc: _stage_rollup_pngs("quality_control", wc.experiment)  # this stage's per-step PNGs
    output: html=..._quality_control_report.html, pdf=..._quality_control_report.pdf
    shell:  "{RUN} ... stage-rollup-report --stage quality_control --data-root {DATA_ROOT} \
               --experiment {wildcards.experiment} --output-html {output.html}"
```

The rule's `input:` (`_stage_rollup_pngs`) and the builder (`stage_report_pngs`) **derive the step→PNG
list the same way** — both from `report_steps_for_stage` + the registry — so Snakemake's declared
prerequisites can't drift from what the builder actually embeds. Because the input list is discovered by
suffix, a new per-step report in a stage is embedded by that stage's rollup with **no rollup edits**.

The rollup CLI is **self-resolving**: the rule passes only `--stage`, `--data-root`, `--experiment`, and
the output HTML path; the command re-derives the PNG paths from the registry. The PDF lands beside the
HTML (`.write()` returns both); the second `output:` just makes Snakemake track it.

---

## Worked Example: death-persistence mortality curtain

**Where:** `quality_control/death_detection/report.py`
**Consumes (merged):** `death_detection_qc` (`persistence_dead_flag`, `viability_dead_flag`) + `fraction_alive`
(the trace). Both recomputable-from facts the death spec already keeps *out* of the flag table — the
report is exactly where that recomputation lives ("no diagnostic widening… it lives in the review
tooling").
**Axis:** `time_index` (death_detection's inputs do not include stage; scope is the step).
**Grain:** **per `physical_embryo_id`** — one animal, one strand. *Not* per `embryo_id` (that would
double-draw a two-channel animal and split one animal across channels — the exact bug the death spec
warns about).
**Two artifacts, one step** (`death_detection_report` with `curtain_png` + `death_time_png`) — they are
independently useful (the curtain is the diagnostic; the histogram is a slide-ready summary):

**Artifact A — the mortality curtain.** X = `time_index`, Y = `fraction_alive`; one faint line per
`physical_embryo_id` (low alpha ≈ 0.1), all overplotted. What emerges:

- healthy cohort → a dense band riding high near 1.0 across the whole axis;
- mortality → strands peeling downward; *when* they peel *is* the death-time distribution, at a glance;
- a plate with a mortality problem → the band visibly thins / sags over time instead of holding flat.

Overlay the called deaths: for each embryo whose `persistence_dead_flag` fired, mark its called death
`time_index` (a dot where its trace crosses, or a bottom rug). This lets the eye check whether the
persistence caller fires *where the trace actually collapses* — QC of the QC.

**Artifact B — death-time histogram (a report-only derived value).** X = called-death `time_index`, one
count per embryo that died. `report.py` derives "the time each embryo died" from `persistence_dead_flag`
+ the trace — a value that lives only in the report, never in the flag table. This is the marginal of the
curtain: "most mortality happens in this window" — the "distribution of embryos" instinct, aimed at
*when they die*.

**Optional refinement — disagreement coloring.** Color strands/marks by `viability_dead_flag` vs
`persistence_dead_flag` disagreement, since that disagreement is informative (transient dip vs. true
death, per the death spec). Refinement, not v1.

---

## Candidate Report Roster (all steps — NOT yet committed)

**Status:** a *candidate* backlog from a per-step review (7 agents, 2026-07-02), thinned to **one signature
view per step** (a report is a glance, not a dashboard — every row passed the "would I actually open
this?" test). Every row is scoped to its own step's input surface and respects the hard rules above.
Nothing is wired until promoted (a `<step>_report` registry row, a `report.py` or spec entry, an `.smk`
rule, a slot in the `reports` target). The **Renderer** column names the shared renderer (A–F above) each
report calls — most reports are *just a renderer + (column, grain)*, not bespoke code.

**Acquisition — ingest + materialization/scaffolding steps have no useful report** (small metadata /
identity / sentinel tables). `frame_inventory` is the one exception, and even there one coverage view is
enough for tier-1.

**⭐ = tier-1 (build/promote first).**

| Step | Candidate report | Renderer | Grain × axis | Notes |
|---|---|---|---|---|
| `frame_inventory` | timepoint coverage grid | A/heatmap | `well_id` × `time_index`,`channel_id` | spot missing frames / timing gaps |
| `frame_detections` ⭐ | **detections-per-well over time** | **A** | `well_id` × `time_index` | avg embryos/frame *by well over time* — *named ask*; grain matters (not a flat per-frame hist) |
| `frame_masks` | mask-validity over time | A | `well_id` × `time_index` | valid/invalid fraction per well over time |
| `physical_embryo_registry` ⭐ | **embryos-per-well distribution** | **B** | group=`well_id` | *named ask*; count-per-group |
| `snip_inventory` | snips-per-embryo distribution | B | group=`physical_embryo_id` | timepoints per animal |
| `snip_auxiliary_masks` | aux-mask validity by type + gallery | C+E | — | which UNet masks fail, with exemplars |
| `latent_embeddings` | latent 2D projection (color by `time_index`) | custom | color=`time_index` | genotype coloring **out of scope** → `analysis_ready` |
| `mask_geometry` ⭐ | **feature histogram grid** | **D** | payload cols | *named ask*; one gridded PNG |
| `curvature_metrics` | high-curvature snip gallery | E+panel | — | reuse **panel/snip-image import**; overlay the curvature centerline on the snip |
| `pose_kinematics` | speed over time | A | `physical_embryo_id` × `time_index` | motion arrest / hyperactivity |
| `stage_predictions` | stage trajectory over time | A | `physical_embryo_id` × `time_index` | stalls/reversals — tracking/seg QC signal |
| `fraction_alive` | viability cohort curtain | A | `physical_embryo_id` × `time_index` | raw-feature precursor of the death curtain (see note) |
| `surface_area_qc` ⭐ | **area-vs-stage scatter + band gallery** | **F**+E | `predicted_stage_hpf` | **migrate existing results code** (already built); stage is a step input |
| `mask_quality_qc` | per-flag counts + co-occurrence | custom | — | booleans only; no scalar to histogram |
| `focus_qc` (stub) | interior-edge-fraction hist + gallery | C+E | `time_index` | lands with z-stack ingest |
| `motion_blur_qc` (stub) | bad-z-pair-fraction hist + gallery | C+E | `time_index` | stage **not** a motion_blur_qc input |
| `death_detection_qc` ⭐ | **mortality curtain + death-time histogram** | **A**+C | `physical_embryo_id` × `time_index` | worked example above; A proven here |
| `death_event` | death-time + death-stage distributions | C | — | cohort marginals (2 subplots) |
| `snip_qc` ⭐ | **exclusion reasons over time — dead vs not-dead** | **A/B** | `time_index`; split by dead | see snip_qc note below — the key tier-1 QC view |
| `analysis_ready` | 2D PCA colored by genotype / `predicted_stage_hpf` | custom | — | the *only* step whose input surface is the whole joined DAG |

### Tier-1 set (promote now)

1. **`frame_detections` — detections-per-well over time** (renderer A) — *named ask*, exercises grain+A.
2. **`physical_embryo_registry` — embryos-per-well distribution** (renderer B) — *named ask*, exercises B.
3. **`mask_geometry` — feature histogram grid** (renderer D) — *named ask*, exercises D.
4. **`death_detection_qc` — mortality curtain + death-time histogram** (A+C) — inputs already exist; the
   worked example; proves renderer A end-to-end.
5. **`surface_area_qc` — area-vs-stage scatter + gallery** (F+E) — **migrate the code already written in
   `results/mcolon/20251010_sa_outlier_analysis/`** rather than rebuild; proves F end-to-end.
6. **`snip_qc` — exclusion reasons over time, dead vs not-dead** (A/B) — the whole-experiment health view.

Tier-1 therefore forces exactly the three new shared renderers (A, B, D) into existence plus wires the two
already-built v1 shapes (C-histogram, E-gallery, F-band via the surface-area migration). Everything below
tier-1 then reduces to "call an existing renderer with a different (column, grain)."

**Reviewer / grain notes:**

- **Grain is the point.** Several candidates the agents wrote as flat histograms are only useful *with*
  grain and time: detections is **per-well over `time_index`** (not one global count), mask-validity and
  fraction-alive are **per-group over `time_index`**. That is exactly why renderer A is *(value, group,
  time)* — the grain columns (`well_id`, `physical_embryo_id`) carry the information worth seeing.
- **Two curtains, one shape.** `fraction_alive`'s viability curtain and `death_detection`'s mortality
  curtain are renderer A at the raw-feature vs. flagged level — build the death one for tier-1; the
  fraction_alive one is the same renderer, promote later only if wanted.
- **`snip_qc` needs a dead / not-dead split (important).** Death dominates exclusions, so a single
  reason-breakdown is misleading — most of the plate is filtered *because it died*. Emit **two views**:
  (1) all exclusions, and (2) exclusions **among the not-dead** (drop rows flagged dead, then break down
  the remaining reasons) — this is what surfaces embryos filtered for *non-death* reasons (edge, focus,
  SA outlier). Show exclusion **fraction over `time_index`** (attrition is temporal), not just a flat bar.
  Built from the flag inputs snip_qc **already consumes** (or `qc_fail_reasons`), never a new upstream join.
- **`curvature_metrics` gallery** reuses the shared **panel/snip-image import** and overlays the extracted
  curvature centerline on each snip — it is renderer E plus a centerline overlay, not a fresh gallery.

> The `mask_geometry` / `curvature_metrics` grid rows are the general "histogram of every feature"
> primitive (renderer D): each `*_PAYLOAD_COLUMN` gets a panel, split by any in-step flag that judges it.
> One gridded PNG per step. Cheapest, most universal report and the natural first stock target beyond QC.

---

## Done When (per report)

- report code that draws lives only in `viz/reporting.py`; `report.py` is wiring only and imports no
  matplotlib directly;
- input paths resolve through `artifact_path(..., path_mode="merged")`, no raw path strings;
- the report artifact is a DAG leaf — no rule lists it as input, nothing imports the `report.py`;
- **the terminal-leaf test passes** — a single repo-wide test enforces the two bright lines the doctrine
  depends on, so they are not honor-system:
  1. **no import of any `report.py`** — walk the source tree (AST or grep) and assert no module imports a
     `*/report.py`; a report is a terminal leaf in the import graph;
  2. **no rule consumes a report artifact** — assert no non-report `PIPELINE_STEPS` rule lists a
     `*_report` artifact in its `input:`; a report is a terminal leaf in the DAG.
  This test lives once (e.g. `tests/data_pipeline/viz/test_reports_are_terminal.py`), not per product;
- the report runs on a real merged experiment output (validated against `20250912` for v1) and degrades
  gracefully when an optional axis input (`predicted_stage_hpf`) is absent, falling back to `time_index`;
- a stock report is a spec entry with no per-step file; a custom report is one `report.py` beside its
  `entrypoint.py`.

---

## Not This World

- Reports do not compute or persist any feature/QC table column — every value they show is recomputed
  from a step's merged output at render time.
- Reports do not join genotype/condition/perturbation — that is `analysis_ready`'s job; a report shows a
  *step's* output, not analysis-ready biology.
- Reports are not part of the legacy drift gate — that comparison has its own `_legacy_drift/reports/`
  home and human sign-off (see `features/targets/feature_world.md` § Legacy Drift Comparison).
