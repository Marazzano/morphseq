# Pipeline File Philosophy — how every rule, module, and name should read (🟢 TARGET)

**Status:** the conventions doc, mdcolon 2026-06-07. Distilled from the two reference
implementations that already embody it — `orchestration/paths.py` and `orchestration/well_runner.py`
— and the locked identity/orchestration constraints. **Every new rule, task, and module in the
refactor conforms to this.**
**Companion to:** `front_end_naming_and_flow.md` (what the front-end stages are),
`well_id_throughline_refactor_plan.md` (the Scopes), `stitched_handoff_contract.md` (the seam).

---

## 🎯 THE ONE GOAL — the code tells its own story

> **A reader should be able to follow the pipeline by reading the names, in order, without a
> diagram.** Every identifier — a function, a step key, a path mode, a variable — says *what it is*
> and *where it sits in the flow*. The narrative lives in the code, not in a separate doc you have
> to hold in your head. Comments explain **why**; names explain **what**.

The litmus test: paste a single function name or step key into a chat with someone who's never seen
the repo. If they can tell you roughly what it does and where in the pipeline it runs, the name is
right. `run_well_ids_for_experiment`, `collect_well_shard_paths`, `scope_metadata_mapped.csv`,
`join_series_mapping_to_scope_metadata` all pass. `process_data`, `helper2`, `do_well`, `df2`
all fail.

---

## 🪨 THE TWO HARD CONSTRAINTS (non-negotiable — everything else is style)

These are the load-bearing rules. A rule or entrypoint that breaks one is wrong, not just unidiomatic.

### 1. Paths come from `paths.py`. Identity comes from `shared/identifiers/`. Never hardcode either.
- **No artifact path is ever typed as a raw string** in a rule, a task, or a module. It is resolved
  through `orchestration.paths` (`artifact_path` / `validated_path` / `provenance_path` /
  `step_dir` / `per_well_step_dir`). One registry (`PIPELINE_STEPS`), read by both the Snakefile (at
  parse time) and the Python entrypoints (at run time), so rule-output and code-output paths are
  *guaranteed identical*. If two places disagree about where a file lives, exactly one of them
  bypassed the registry — that is the bug.
- **No id is ever minted or split with an inline f-string or `.split("_")`.** `well_id`,
  `image_id`, `embryo_id`, `snip_id` are built by `shared/identifiers/` constructors and decomposed
  by its parsers. Orchestration *imports* identity; identity never imports orchestration.

### 2. The two kingdoms stay separate.
| Kingdom | Question | Lives in | Imports the other? |
|---|---|---|---|
| **Identity** | "What is this object's canonical name?" | `shared/identifiers/` | never |
| **Orchestration** | "Where does its file land / which wells run / how do shards merge?" | `pipeline_orchestrator/orchestration/` | imports Identity, one-way |

> Identity flows *into* orchestration, never the other way. Keep the crown out of the lab coat.

**Used together via config:** a stage is fully specified by a `PIPELINE_STEPS` row (its `stage`
folder + `fanout` + `artifacts`) plus the global `well_id` grammar. Get those two right and the rule
body is a template — the path is computed, the id is composed, nothing is retyped.

---

## ✍️ NAMING CONVENTIONS (the grammar, with real examples from the reference files)

### Steps are NOUNS (the output slot); rules are VERBS (the action on it)
A **step** is a logical product — a noun naming an output slot in the registry. A **rule** is a
Snakemake action — a verb. One step can be touched by several rules.
- step: `frame_inventory` (the noun/product) ← in `PIPELINE_STEPS`
- rules: `build_frame_inventory_for_well`, `validate_frame_inventory_for_well`, `merge_frame_inventory` (verbs)

A step key reads right from *every* rule that touches it; a verb step key (`build_*`) would read
wrong from the validate rule. (Front-end keys that are verb-shaped — `ingest_*`, `discover_wells` —
are grandfathered because `front_end_naming_and_flow.md` locked them.)

### Names carry their position in the flow
- `join_series_mapping_to_scope_metadata` *names the operation* (join mapping onto scope metadata) —
  not `apply_series_mapping_yx1` (names a mechanism + a microscope that shouldn't be in the name).
- `scope_metadata_mapped.csv` — the artifact says it is scope metadata, after mapping.
- `discovered_wells.txt` — not `wells.txt`; says *which* well list (discovered, pre-filter), so it
  never gets confused with `active_wells` / `validated_wells`.

### Distinguish look-alikes by their *moment*, not a vague suffix
`well_runner.py`'s sharpest move — two collectors that look similar but run at different times, and
the names + docstrings make the difference impossible to miss:
- `run_well_shard_paths` — DAG-planning time, **pure path resolution, no disk read** (returning a
  path IS the dependency declaration).
- `collect_well_shard_paths` — rule-execution time, **scans disk, checks `.validated` sentinels**.

Same shape, opposite moment. The names (`run_*` = the run set / planning; `collect_*` = gather what
exists) + a docstring that states the moment up front make it self-teaching.

### Controlled vocabulary, defined once as constants — never magic strings
`paths.py` never sprinkles `"per_well"` or `"experiment"` through the code; they are named constants
(`PER_WELL_DIRNAME`, `PATH_MODE_PER_WELL`, `EXPERIMENT`, `PER_WELL_THEN_MERGE`) defined once at the
top with a comment on what each permits. A typo becomes an import error, not a silent wrong path.

### One concept, built in exactly one place (no drift)
The per-well directory `{stage}/{exp}/per_well/{well_id}` is composed *downward* from bricks
(`_experiment_step_dir` → `_per_well_step_dir` → `step_dir`), so no caller ever strips a level with
`.parent`. If a path concept can be derived two ways, it will eventually be derived two *different*
ways — so it gets one home.

---

## 🧱 STRUCTURAL CONVENTIONS

### Explicit signatures, no haunted globals
Functions take what they need as parameters (`output_root` is *always* passed, never read from
`PROJECT_ROOT`). Keyword-only (`*`) for anything that could be confused positionally
(`well_id=`, `path_mode=`). No module-level mutable state a function secretly reads.

### Fail loud at contract boundaries; the message names the fix
Every guard raises with a message that says *what was wrong AND what to do*:
- unknown step → lists the known steps;
- a `per_well_then_merge` step called with no `path_mode` → spells out "pass `per_well` or `merged`";
- a bare local label where a global `well_id` was required → says "not a bare local label."

A validation error is a teaching moment, not a stack trace. (Generalizes `validate_well_id`'s
fail-loud-on-`A01` rule to every boundary.)

### `fanout` is executable, not decorative
A step's `fanout` (`EXPERIMENT` vs `PER_WELL_THEN_MERGE`) actually *constrains* which `path_mode`s
are legal (`_normalize_path_mode`). Metadata that the code enforces, not metadata you have to
remember to honor.

### Docstrings teach the WHY and the moment; code shows the WHAT
The module docstring of `well_runner.py` is the template: it names the jobs, draws the 2a/2b
planning-vs-execution split, and states the two hard rules — *before* any code. A reader is oriented
before line 1 of logic. Match that altitude: orient, then implement.

### Sidecars are derived, never first-class
`.validated` and `.provenance.json` are computed *from* an artifact path by a helper
(`validated_path`, `provenance_path`) — they are never their own registry rows or hardcoded strings.
One artifact, its sentinels derived.

---

## 📐 THE "ADD A STAGE" RECIPE (what conformance buys you)

Because of the two constraints, adding a pipeline stage is a fixed recipe, not a design exercise:
1. **One `PIPELINE_STEPS` row** — `stage` (folder) + `fanout` + `artifacts` (noun step key).
2. **One compute function** that takes `output_root` + ids explicitly, imports identity for any id.
3. **One `tasks.py` verb** (route everything through `tasks.py`; the Snakefile knows verbs, not deep
   module paths — Win 2).
4. **One rule from the template** — resolves its paths via `paths.py`, composes ids via identifiers,
   runs under the single `{RUN}` prefix. Change only the step name, the verb, and the inputs.

"Add a feature = one row + one function + one verb + one templated rule." If adding a stage requires
inventing a new path string or a new id format, a constraint was broken.

---

## ✅ CONFORMANCE CHECKLIST (paste into a review)
- [ ] No raw artifact-path string anywhere — all via `paths.py` helpers.
- [ ] No inline id mint/split — all via `shared/identifiers/`.
- [ ] Step keys are nouns; rule names are verbs; the noun reads right from every rule touching it.
- [ ] Names state position-in-flow; look-alikes disambiguated by their moment.
- [ ] Controlled tokens are named constants, defined once.
- [ ] `output_root` (and every dependency) passed explicitly; no haunted globals.
- [ ] Every guard fails loud with a message that names the fix.
- [ ] Module docstring orients (jobs + why + boundaries) before the code.
- [ ] Sidecars derived via helpers, never hardcoded or registry rows.

---

## 🔗 The reference implementations (read these as the worked example)
- `pipeline_orchestrator/orchestration/paths.py` — the registry + path grammar.
- `pipeline_orchestrator/orchestration/well_runner.py` — well selection + the 2a/2b shard collectors.
- `pipeline_orchestrator/orchestration/__init__.py` — the curated public API (what's exported is the
  vocabulary the rest of the pipeline speaks).
