# Data-Flow Test Plan — proving real data moves through the plumbing (🟢 TARGET)

**Status:** PLAN, mdcolon + Opus 2026-06-26. The plumbing is built and unit-green; this doc is the
tiered runbook for proving **real bytes** traverse every seam, on two real microscopes.

**Companion to:** `pipeline_file_philosophy.md` (the conventions any change MUST honor),
`output_tree_doctrine.md` (where artifacts land), `current_state_and_next_steps.md` (live state),
`../OVERALL_PLAN.md` (the spine + build order).

---

## 0. The thesis — depth vs. width are different failure modes

We built plumbing. Unit tests prove each fitting in isolation; `snakemake -n` proves the pipes
*connect*. Neither proves **water flows end-to-end without leaking at a seam**. Integration bugs
live exactly where unit tests can't reach: schema drift across a handoff, a `.validated` sentinel
written in the wrong convention, an env boundary (`conda run -n …`), a column the upstream stopped
emitting, a path two stages resolve differently.

Those failures split into two axes that fail differently and should be tested separately:

- **DEPTH** — does *one* well reach the end? Exercises every inter-stage **seam** once. Cheapest
  place to find a broken handoff. A depth bug is a contract bug.
- **WIDTH** — does a *whole experiment* fan out and merge? Exercises the **checkpoint fan**, per-well
  parallelism, the merge seams ("merge never shrinks"), and resource contention (GPU scheduling). A
  width bug is an orchestration bug.

Debug depth at 1 well / 1 timepoint before ever reaching for width at 95 wells. **The terminal for
the through-line proof is `snip_qc`** (the per-snip `use_snip` / `qc_fail_reasons` verdict).
Everything past `snip_qc` (embeddings, analysis_ready) is gravy and is explicitly out of scope here.

---

## 1. Named fixtures (concrete, not "an experiment")

| Role | Experiment | Scope | Why this one |
|---|---|---|---|
| Primary / depth+width | `20250912` | YX1 | Front half already smoke-proven (95 wells discovered, global `well_id`, z_stack+projection on B01/C01). Known-good front → failures are genuinely downstream. |
| Agnostic-seam proof | `20250612_24hpf` (`…_ctrl_atf6`) | Keyence | Front half smoke-proven 13/13 (A01+A02, mosaic stitch, canvas focus_index_map). The other microscope. |

These are deliberately the two already-proven-at-small-scale front halves, so every Tier failure is
a real downstream-seam bug, not a front-half regression.

---

## 2. Baseline ledger (Tier 0) — "what does green mean today?"

Before any new run, produce ONE consolidated table: for every rule in the spine
(`ingest_* → … → snip_qc`), record the **last real-data proof** and **at what scale**
(experiment, well count, timepoint count, scope, CPU/GPU, commit). Most of this is already scattered
across the dated snapshots in `current_state_and_next_steps.md` — consolidate, do not re-derive.

Columns: `step_key | rule(s) | last real-data proof (commit/date) | scale (exp/wells/timepoints) |
scope(s) proven | CPU/GPU | gap`.

The ledger's job is to make the **true gaps** explicit and stop us re-proving things already proven.
Known gaps as of 2026-06-26 (the ledger must confirm/refute each):
- Everything **past merged `frame_masks`** (registry → snip → features → QC → `snip_qc`) has passing
  unit tests + `snakemake -n` plans but **no continuous real-data run**. `rule all` stops at merged
  `frame_masks`.
- Full **YX1 at scale** (all 95 wells, all timepoints) never run — only 1–2 wells / 1 timepoint.
- Keyence proven only on the **front half** (A01+A02); never through the agnostic back half.

---

## 3. The tiers

Each tier names: **goal**, **fixture + scale**, **what it uniquely proves** (the failure mode it
catches that the prior tier could not), **how to run**, and **success criteria** (the concrete
artifacts + checks). A tier is DONE only when its success criteria are observed on disk, not when a
dry-run plans.

### Tier 1 — DEPTH: one well, raw → `snip_qc` (the through-line proof)

- **Goal:** one YX1 embryo's data crosses *every* seam from raw ND2 to the `snip_qc` verdict.
- **Fixture + scale:** `20250912`, **one well** (start `B01` — it has both products proven), **1
  timepoint**. CPU where possible; GPU node for detection/segmentation/projection legs.
- **Uniquely proves:** every inter-stage **contract handoff** end-to-end, once:
  `materialize_well → frame_inventory(assembled) → frame_detections → frame_masks →
  physical_embryo_registry → snip_processing → snip_auxiliary_masks → {mask_geometry,
  pose_kinematics, fraction_alive, stage_predictions} → consolidated_features →
  {surface_area_qc, mask_quality_qc, death_detection} → snip_qc`. This is the first time bytes flow
  the whole length; it surfaces schema/sentinel/env-boundary breaks unit tests can't.
- **How to run:** a dedicated single-well overlay (`config_smoke_through_line_20250912_B01.yaml`)
  pinning one well + 1 timepoint, with a Snakemake target that names the `snip_qc` per-well shard as
  the goal. **The `snip_qc` rules are already fully wired** (`write_snip_qc_resolved_sources_for_well`
  → `build_snip_qc_for_well` → `validate_snip_qc_for_well` → `merge_snip_qc`, all included in the
  Snakefile). What is missing is only a **named aggregate target** that *requests* `snip_qc` —
  `rule all` stops at merged `frame_masks`, so nothing currently asks for it. Add a separate named
  target (`through_line`, mirroring `front_half`/`all`) whose input is the snip_qc artifact; do NOT
  extend `rule all`. This is a one-rule "add a target" change, done per §5.
- **Timepoint-limiter precondition:** Tier 1 assumes one timepoint can be restricted **through
  config** (the `smoke_max_time_indices`-style overlay already used by `config_smoke_zstack_*`). If
  that limiter does NOT cleanly cover the whole chain, the FIRST Tier-1 deliverable is to add a
  conformant timepoint/window limiter **through config/rule wiring** — never an ad-hoc
  `if time_index == 0` inside a stage function or script. A sneaky filter buried in stage code is a
  doctrine violation, not a shortcut.
- **Success criteria:**
  - the per-well `snip_qc` verdict artifact exists + `.validated`, with `use_snip` populated and
    `qc_fail_reasons` sane.
  - **No stage silently produced 0 rows.** Concretely, for every per-well table expected to carry
    observations (`frame_inventory`, `frame_detections`, `frame_masks`, `physical_embryo_registry`,
    `snip_inventory`, `snip_qc`): `row_count > 0`; primary-key columns present; no duplicate primary
    keys; the contract validator accepts. For this fixture, an empty `frame_detections`,
    `frame_masks`, `snip_processing`, or `snip_qc` is a **failure**, not an edge case.
  - every intermediate per-well `.validated` sentinel along the chain exists.
  - **Resolver doctrine proof:** the per-well snip_qc `resolved_sources` JSON exists, and its
    `resolved_sources` source paths + `exclusion_reasons` (a) point at files that exist and (b) name
    the **same** source QC CSVs/sentinels Snakemake actually used as inputs to `build_snip_qc_for_well`.
    Mismatch between the resolver's declared sources and the DAG's real inputs is a failure — this is
    the real-data proof of the flag-resolver doctrine.

### Tier 2 — WIDTH: one experiment, full fan, raw → `snip_qc`

- **Goal:** the *whole* YX1 experiment fans out per-well and merges, end-to-end to `snip_qc`.
- **Fixture + scale:** `20250912`, **all discovered wells**, **1 timepoint** (hold timepoints at 1
  to isolate width from temporal volume).
- **Uniquely proves:** the parts depth structurally cannot — the `discover_wells` **checkpoint fan**,
  per-well parallel execution, the **merge seams** (every `merge_*` rule; verify a subset run does
  not shrink a previously-broader merged table — the "merge never shrinks" 🟢 behavior, F6 in
  OVERALL_PLAN), and **GPU contention** across wells.
- **How to run:** the existing all-wells path (`front_half` already does this for the front; point
  the `through_line` target from Tier 1 at all discovered wells).
- **Success criteria:**
  - merged experiment-grain artifacts exist + `.validated` at each merge point through `snip_qc`;
    merged row counts == Σ per-well rows; the run is restartable (re-run is a no-op, proving sentinels
    gate correctly).
  - **Explicit well-accounting** (this is where width bugs hide — a fanout can look green while one
    well slips into a sewer grate). Produce a table reconciling, per merged product:
    `discovered_wells | expected_per_well_shards | actual_per_well_shards | merged_wells_present |
    missing_wells | extra_wells`. The pass condition is `missing_wells == ∅` AND `extra_wells == ∅`
    AND `merged_wells_present == discovered_wells` at every merge point. Any non-empty
    `missing`/`extra` is a failure to investigate, not a rounding error.

### Tier 3 — AGNOSTIC SEAM: Keyence through the same back half

- **Goal:** prove `frame_inventory` is a genuinely microscope-agnostic drop-in — a Keyence-originated
  inventory traverses the *same* back half a YX1 inventory did.
- **Fixture + scale:** `20250612_24hpf`, the front-half-proven wells (A01+A02), 1 timepoint, raw →
  `snip_qc`.
- **Uniquely proves:** that nothing downstream of `frame_inventory` is secretly YX1-shaped. The whole
  architectural bet (the Microscope Zone exits at `frame_inventory`; everything after is agnostic) is
  only *proven* when two different-microscope inventories run the identical downstream rules to the
  same terminal. Per-microscope front halves working in isolation does NOT prove this.
- **How to run:** point the Tier-1/2 back-half target at the Keyence experiment (same rules, Keyence
  config overlay).
- **Success criteria:** Keyence `snip_qc` verdict exists + `.validated`; the back-half rule set
  invoked is **identical** to YX1's (diff the executed rule list). Any microscope-conditional rule or
  branch **past assembled `frame_inventory`** is a doctrine violation to be reported per §5 — *unless*
  it is explicitly documented as microscope-agnostic compatibility handling (e.g. harmless metadata
  normalization that carries no microscope logic downstream). The boundary is protected; benign
  normalization that is named and documented as such is allowed.

### Tier 4 — SCALE + EMBEDDINGS (production dress rehearsal) — OUT OF PRIMARY SCOPE

- **Goal:** full experiment, all timepoints, both products, plus embeddings (E1/E2) and
  `analysis_ready`.
- **Status:** deferred. E1/E2 are **not-built** and legacy VAE weights are **not staged**
  (OVERALL_PLAN §2, latent_embeddings snapshot). `analysis_ready` carries known stale vocabulary
  (`z0/z1`, `time_int`). Listed here so the ladder is complete; not part of the `snip_qc`
  through-line goal. Pick up after Tier 3 is green.

---

## 4. Tier dependency ladder

```
Tier 0 (ledger, no runs)
   └─► Tier 1 (YX1, 1 well, 1 tp → snip_qc)       ← find seam bugs cheap
          └─► Tier 2 (YX1, all wells, 1 tp → snip_qc)   ← find fan/merge/GPU bugs
                 └─► Tier 3 (Keyence, A01+A02, 1 tp → snip_qc)  ← prove agnostic seam
                        └─► Tier 4 (scale + embeddings)  [deferred]
```

A whole experiment first moves end-to-end at **Tier 2**. The Keyence + YX1 pairing is **Tier 3** —
on purpose *after* depth, because a broken seam costs minutes to debug at 1 well, hours at 95.

---

## 5. The philosophy guardrail (READ THIS — it is the point)

The goal is **not** "make data reach `snip_qc`." It is "prove data reaches `snip_qc` **while the
pipeline stays polished and well-named** per `pipeline_file_philosophy.md`." Completion that degrades
the codebase is a regression, not a win.

If pushing data through reveals a real break, the fix MUST conform:

- **No raw path strings** — all artifact paths via `paths.py` helpers. **No inline id mint/split** —
  all via `shared/identifiers/`. (The two hard constraints.)
- **Step keys are nouns; rules are verbs;** controlled tokens are named constants; `output_root` and
  every dependency passed explicitly (no haunted globals).
- **`paths.py` stays pure** (no `.exists()`/disk reads); planning-time input functions declare
  expected paths only; runtime disk-scan collectors say so in their name/docstring.
- **One authoritative validator per contract**, owned where the product lives; lifecycle differences
  are a `check_sources=`-style **mode flag**, never a forked second validator.
- **Banners organize co-living concerns; they do NOT paste over a mix that should split** (two
  microscopes / two kingdoms in one function → split, per the agnostic-seam invariant Tier 3 guards).
- **The "add a stage" recipe** (one `PIPELINE_STEPS` row + one compute fn + one thin `tasks.py` verb
  + one templated rule) applies to any new wiring (e.g. a `snip_qc` aggregate target).
- Sentinel convention is the **dot-prefixed hidden** `.{filename}.validated` (OVERALL_PLAN F2).

**Reporting contract for the Agent:** for every change made to get data flowing, record in a
findings section: (1) the seam that broke + the symptom, (2) the root cause, (3) the fix and which
conformance bullet(s) it had to satisfy, (4) any place where the *honest* fix is bigger than this
pass (flag it, do NOT paper over it with a shim — a shim that hides a kingdom violation is worse than
a documented gap). When a fix is ambiguous between "quick" and "conformant," choose conformant or
stop and surface the decision. Prefer a documented gap over an unprincipled patch.

### Change reporting rule

If **any** Tier requires a change — to code, config, the path registry, a schema, a rule, a
validator, or a target — to make data flow, the Agent MUST state it explicitly in the findings
section **before** marking the Tier green. "Small wiring change, no need to mention" is exactly where
the goblins nest. For every change, record:

1. **What changed** — file(s), rule(s), config(s), schema(s), or artifact/step key(s).
2. **Why it changed** — the observed failure or missing wiring that forced it.
3. **What doctrine it touches** — paths, identifiers, sentinels, validators, rule/step naming,
   config, or the microscope boundary.
4. **Whether it is permanent or provisional** — canonical fix, temporary diagnostic hook, or
   documented gap.
5. **How it was verified** — test, dry-run, real-data artifact, row count, sentinel, or
   restartability check.

**A Tier is not green if it required an undocumented change.** Silent fixes are failures: if the
pipeline changed during a Tier, the report must say so. *A run result without a change log is not
evidence — it is folklore with a timestamp.*

The Agent may update `current_state_and_next_steps.md` with the run outcome + change log (a new dated
snapshot), since that doc is the live "where are we now" anchor.

---

## 6. Open decisions (surface, don't guess)

- **Tier-1 starting well:** `B01` (both products proven) vs. a plain projection-only well.
  **DECIDED: `B01`.**
- **Aggregate target name + reach:** **DECIDED — add a separate named target `through_line`**
  (mirroring `front_half`/`all`), do NOT extend `rule all` (keeps the existing default behavior
  untouched). `through_line` points at the current through-line terminal, which today is `snip_qc`.
  The `snip_qc` rules are already wired — this is only the missing *request*. Shape: per-well snip_qc
  for Tier 1; point the same target at all wells (merged) for Tier 2 — matches the depth-then-width
  ladder.
- **Numeric-regression gate (Keyence Stage D):** out of scope for this plan (byte/numeric diff vs.
  legacy stitch is a separate effort, per `current_state_and_next_steps.md`).
