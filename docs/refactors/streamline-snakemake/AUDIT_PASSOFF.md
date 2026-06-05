# Audit Pass-Off — streamline-snakemake refactor

**For:** the model/person auditing whether this refactor plan is coherent,
complete, and buildable.
**From:** prior planning sessions (mdcolon + assistant).
**Date:** 2026-06-05.

> **Your deliverable:** after auditing, **write a new doc `target/OVERALL_PLAN.md`**
> that states the overall plan end-to-end (see §5 for exactly what it must contain).
> Everything before §5 tells you what to read and what to scrutinize so that plan
> is grounded, not invented.

---

## 0. The 30-second frame

The refactor turns an experiment-grain Snakemake pipeline into a **per-well
end-to-end** one: cross the experiment bootstrap once to discover wells, then run
each well independently through the whole spine. The design is spread across
`target/` (authoritative). Your job is **not** to redesign — it's to verify the
plan is internally consistent, that every "decided" item really is decided, and
that the path from "empty `identifiers/`" to "one well runs end-to-end including
embeddings → analysis_ready" is fully specified with no silent gaps.

---

## 1. Where to start (in order)

1. **`ORIENTATION_for_final_review.md`** (this dir) — the map: legacy vs target,
   files that matter, doc trust order. Read first.
2. **`target/current_state_and_next_steps.md`** — verified on-disk state + the
   recommended build order (Scope 1 → paths.py → Scope 2 → well_runner → wire one
   stage). This is the "where are we / what's next" anchor.
3. **`target/per_well_throughline_findings.md`** — the north star. Long but load-
   bearing: grain model (Zones A→B→C), the `fanout` vs `execution` distinction,
   the lean `paths.py` registry + worked example, the well-runner, DAG/staleness
   mechanics, the frame-contract split, open questions, and the numbered
   next-steps with tests (bottom).
4. **`target/well_id_throughline_refactor_plan.md`** — the formal Scopes 1–5.
5. **`target/front_end_naming_and_flow.md`** — ingest lineages + the fan.
6. **`target/stitched_handoff_contract.md`** — the microscope-agnostic input seam.
7. **`target/model_input_handoff_contract.md`** — the model/embedding seam
   (legacy build_06), which feeds analysis_ready.

> **Ground every claim against code**, not just the docs. The Snakefile is
> `src/data_pipeline/pipeline_orchestrator/Snakefile`. The findings doc tags
> facts 🔵 CURRENT (verified Snakefile today) vs 🟢 TARGET (what we build toward) —
> confirm the 🔵 tags still match the Snakefile; flag any drift.

---

## 2. What to audit — the spine claims that everything rests on

These are the load-bearing assertions. If any is wrong, the plan changes.

1. **The fan is a checkpoint, and per-well staleness needs a per-well spine.**
   Verify: `discover_wells` checkpoint (Snakefile) → `wells.txt`; per-well rules
   expand off it. Confirm the TARGET move (fan reads `scope_metadata_mapped.csv`,
   earlier than today's `frame_contract.csv`) is consistent everywhere.
2. **Zone C is per-well-able (no cross-well/cohort math).** The doc calls this
   "owner-confirmed + spot-checked, **exhaustive grep pending**." **This is the
   single biggest audit item** — either run the exhaustive grep-audit
   (module / searched-for / result table) or explicitly note it remains pending.
   If any stage computes a cohort statistic, the per-well split breaks.
3. **The merge wall.** Verify which rules read the *merged* contract instead of
   per-well shards (the doc names `surface_area_qc` reading `consolidated_features`).
   Confirm it's purely a **rewiring** problem, not an algorithm one.
4. **`fanout` vs `execution` holds for embeddings.** Embeddings is the canonical
   `fanout=per_well` + `execution=single` (batched, model loaded once). Verify the
   reasoning (per-well Py-3.9 `conda run` reload would dominate) and that the
   contract + findings doc agree on it.
5. **Embeddings FEEDS analysis_ready** (not a tail). Verify the spine order
   `… qc → embeddings → analysis_ready` and that the `use_snip` gate is NOT taken
   from analysis_ready (circular) — see contract §6.0.

---

## 3. Decisions to confirm are *actually* settled (and consistent across docs)

Spot-check that these read as DECIDED, not "leaning," everywhere they appear:
- **Frame-contract family → new `frame_contracts/`** (per_well_then_merge), not
  `experiment_metadata/`. (findings #5, stitched-contract §193 still say "open" in
  places — **check for stale 'leaning'/'open' text** and confirm it's reconciled.)
- **Multi-channel → channel as ROWS** in one per-well frame contract (schema
  already keys on `channel_id`; built images nest `/well_id/channel/`).
- **`well_id` is the canonical per-well key everywhere** (incl. image trees).
- **`target_wells` = compute filter, never a merge-membership filter** (merge
  composes over all present valid shards; never shrinks).
- **Registry governs tabular artifacts only**; image trees are off-registry.
- **Env: model interpreter → `env.yaml` (Scope 3), not a `config.yaml` block.**

---

## 4. Known-open items (verify these are tracked, decide if they block)

From the docs' own open-lists (findings "Remaining real unknowns" + "AUDIT TODO",
front_end §498, contract §8):
- **Exhaustive Zone-C grep-audit** — pending (see §2.2 above; highest priority).
- **`.validated` sentinel convention uniform?** — assumed `{filename}.validated`,
  not yet verified against all rules.
- **`models_root` location** — leaning Scope-3 `env.yaml` reusing legacy layout;
  confirm.
- **`mseq_pipeline_py3.9` env name** — confirm it still exists on this machine.
- **Stitching `execution`** (single vs per_well) — deferred; confirm deferral OK.
- **Merge cadence** (per-stage vs stage-group) — deferred tuning; confirm.
- **Scope-2 migration sites** — `discover_wells` reads `frame_contract.well_index`
  which Scope 2 changes; confirm the migration order is safe.

For each: mark **blocks-build** vs **safe-to-defer**, with a one-line reason.

---

## 5. YOUR DELIVERABLE — write `target/OVERALL_PLAN.md`

After auditing, create **`target/OVERALL_PLAN.md`**: a single, top-level statement
of the whole refactor plan that someone could execute against. It must contain:

1. **Goal** (one paragraph) — per-well end-to-end; the bootstrap→fan→per-well→merge
   shape; embeddings feeds analysis_ready.
2. **The end-to-end spine** as an ordered stage list (Zone A bootstrap → fan →
   Zone B per-well incl. embeddings → analysis_ready → Zone C merges), each stage
   with: family, fanout, execution, and current-vs-target status (built / merge-
   wall / not-built).
3. **Build order** with dependencies — the numbered sequence (Scope 1 identifiers
   → paths.py → Scope 2 → well_runner → wire one stage → replicate → embeddings),
   noting what unblocks what. Pull from `current_state_and_next_steps.md` §
   "Next concrete steps" and the findings doc's bottom list; don't invent a new one.
4. **Per-stage status table** — every stage × {built, needs-rewire, not-built} so
   the remaining work is countable.
5. **Open items** (from §4) with blocks-build vs defer dispositions.
6. **Audit findings** — anything from §2/§3 that did NOT hold up, with what must
   change.

**Constraints:** it is a *plan/index*, not a re-derivation — link into the
existing `target/` docs for detail (`[[per_well_throughline_findings]]` etc.),
don't duplicate them. Keep CURRENT vs TARGET tags. If you find a contradiction
between docs, the plan names it as a finding rather than silently picking one.

> Place it in `target/` (the morphseq-docs repo via the symlink) and add it to the
> doc-coverage table in `current_state_and_next_steps.md`.
