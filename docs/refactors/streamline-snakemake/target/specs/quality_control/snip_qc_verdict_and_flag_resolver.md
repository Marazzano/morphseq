# snip_qc Verdict + Flag Resolver (🟢 TARGET)

**Status:** architecture spec, mdcolon 2026-06-26. Records the doctrine behind the `snip_qc`
wiring refactor — why it is shaped this way, what the resolver is, and how future QC sources
must be added. **No code in this doc; code follows doctrine.**

**Companion to:** `pipeline_file_philosophy.md` (the two kingdoms), `output_tree_doctrine.md`
(artifact registry), source QC product specs.

---

## 🪨 The one-sentence answer

`snip_qc` does not resolve its own source file paths — it receives a **pre-resolved plan** (a
tracked per-well JSON artifact) that was built by joining two existing registries: `paths.py`
(step → artifact path) and source QC contracts (`*_PAYLOAD_COLUMNS`, step → flag columns emitted).

---

## 🎯 The problem this solves

Before this refactor, `snip_qc/inputs.py` imported `artifact_path` from
`pipeline_orchestrator.orchestration.paths` to locate its source CSVs at runtime. That is a
kingdom violation: a stage module secretly doing orchestration-level path resolution. It also
made the DAG blind to the source files — Snakemake could not see which upstream products
`snip_qc` depended on.

The fix is not just moving an import. It establishes the correct pattern for any QC aggregator
stage that reads multiple upstream products and ORs their flags into a verdict.

---

## 🧭 Load-bearing doctrine

### Rule 1 — The bridge is not a third registry

Two registries already exist:

| Registry | Lives in | Answers |
|---|---|---|
| `PIPELINE_STEPS` | `orchestration/paths.py` | step/artifact → filesystem path |
| `*_PAYLOAD_COLUMNS` | source QC contracts | step → flag columns emitted |

`snip_qc` needs to know both: *which files to load* and *which columns to expect from each*.
The bridge that joins them is **a resolver, not a new registry**. It does not re-declare
column names or path patterns — it imports the existing declarations and joins them.

### Rule 2 — Policy, source allowlist, and payload contracts are three separate concerns

| Concern | Lives in | What it says |
|---|---|---|
| Which flags matter | `SNIP_QC_EXCLUSION_FLAGS` in `contract.py` (config may override) | flat list of flag columns |
| Which steps may supply flags | `_SOURCE_PAYLOADS` in `flag_input_resolver.py` | step → `*_PAYLOAD_COLUMNS` import |
| Which columns each step emits | `*_PAYLOAD_COLUMNS` in source contracts | authoritative, never duplicated |

The resolver joins all three. Nothing else does.

There is no rename layer: a flag column's name in `SNIP_QC_EXCLUSION_FLAGS` IS the token recorded in
`qc_fail_reasons`. No reason→flag_column map, no aliasing.

### Rule 3 — The contract is self-reinforcing

Adding a new QC exclusion flag requires **all three** of the following, or the resolver fails
loud at startup:

1. Add the `"new_flag_column"` entry to `SNIP_QC_EXCLUSION_FLAGS`
   (or a config override)
2. Import the source step's `*_PAYLOAD_COLUMNS` constant in `flag_input_resolver.py`
3. Add the step to `_SOURCE_PAYLOADS` in `flag_input_resolver.py`

If you add the flag to policy but forget the source, the resolver raises:
```
snip_qc resolver: could not resolve all requested flag columns:
  'new_flag_column': not found in any eligible source payload
    (eligible steps: ['death_detection_qc', 'mask_quality_qc', 'surface_area_qc'])
Fix _SOURCE_PAYLOADS or the exclusion_flags config.
```
The contract teaches you what to do.

### Rule 4 — Input functions declare; rules create; runtime consumes

The resolver output (`ResolvedFlagSource` objects with concrete paths) is serialized as a
**tracked per-well JSON artifact** (`{well_id}_snip_qc_resolved_sources.json`) produced by a
real Snakemake rule (`write_snip_qc_resolved_sources_for_well`). It is never written as a
side effect inside an input function.

This writer rule runs **during DAG execution** (not during DAG planning / Snakemake parse time).
The resolver *is* called at parse time (inside `_snipqc_source_shards`, a Python helper that
declares the rule's DAG gating inputs), but it returns paths without writing anything — it is
pure. The actual JSON file is only materialized when the rule's shell command executes.

The source QC CSVs and their `.validated` sentinels are inputs to the writer rule **not because
the resolver reads them** — it does not; the resolver is pure — but because they act as a DAG
gate: Snakemake will not run the writer (and therefore will not run `build_snip_qc_for_well`)
until all upstream QC products are validated. This must be documented in the rule so future
readers are not confused.

### Rule 5 — Planning and runtime use the same policy; never split-brain

Whatever `exclusion_flags` the `.smk` uses to declare DAG inputs is the same policy the
runtime task uses to build the verdict. This is guaranteed by:

- The `.smk` resolves `_SNIP_QC_EXCLUSION_FLAGS` once at parse time (config override or
  `SNIP_QC_EXCLUSION_FLAGS`)
- The writer rule serializes **both** `exclusion_flags` and `resolved_sources` into the JSON
- `cmd_snip_qc` reads `exclusion_flags` from that JSON — never from `SNIP_QC_EXCLUSION_FLAGS`

If planning said "load three source files using this policy", runtime must use that exact policy.

---

## 📐 Wire-through map (DAG to verdict)

```
config.yaml (optional)                  SNIP_QC_EXCLUSION_FLAGS
      │                                          │
      └─────────── .smk parse time ─────────────┘
                          │
                 _SNIP_QC_EXCLUSION_FLAGS (resolved policy)
                          │
              ┌───────────┴──────────────────────────────────┐
              │  flag_input_resolver.py                       │
              │  _SOURCE_PAYLOADS × PIPELINE_STEPS            │
              │  → ResolvedFlagSource(step, artifact, flags, path) per needed step
              └───────────────────────────────────────────────┘
                          │
         ┌────────────────┴─────────────────────────┐
         │                                           │
   DAG gating inputs                  write_snip_qc_resolved_sources_for_well
   (source CSVs + sentinels               ← runs during DAG EXECUTION →
    declared at parse time;                 produces tracked JSON artifact)
    gate execution order)                             │
                                     build_snip_qc_for_well
                                       │   reads JSON → ResolvedFlagSource list
                                       │   reads exclusion_flags from JSON
                                       │
                                   inputs.py
                                   open each source CSV
                                   verify snip_id + flag columns + boolean dtype
                                   merge one-to-one on snip_id → qc_flags_df
                                       │
                                   build.py
                                   build_snip_qc_verdict(snip_universe, qc_flags_df,
                                                         exclusion_flags)
                                       │
                                   contract.py
                                   validate_snip_qc(verdict, registry)
                                       │
                                   {well_id}_snip_qc.parquet
```

---

## 🗺️ Kingdom map

| Module | May import paths.py? | Role |
|---|---|---|
| `flag_input_resolver.py` | **Yes — explicit exception** | Adapter/wiring: bridges payload contracts to paths.py |
| `inputs.py` | No | Pure load + verify: receives resolved paths, checks CSV reality |
| `entrypoint.py` | No | Thin adapter: receives resolved sources + exclusion_flags from JSON |
| `build.py` | No | Pure computation: snip_universe + qc_flags_df → verdict |
| `contract.py` | No | Default policy + schema + validator |

A test enforces this boundary: no module under `snip_qc/` except `flag_input_resolver.py` may
import from `pipeline_orchestrator`.

---

## 🚫 What this is not

**Not a third registry.** `_SOURCE_PAYLOADS` in `flag_input_resolver.py` is not a new
declaration of column names or step metadata. It is an explicit allowlist that maps each
eligible source step to its already-declared `*_PAYLOAD_COLUMNS` constant (imported from the
source contract). The columns live in exactly one place.

**Not a ghost file.** The `resolved_sources` JSON is a real DAG artifact registered in
`PIPELINE_STEPS["snip_qc"]` with a per-well path pattern. It is produced by a named rule and
consumed by the build rule. It is **not merged across wells** — it is a planning artifact for
that well's build, not a product.

**Not runtime re-resolution.** `entrypoint.py` does not call `resolve_snip_qc_flag_sources`.
It receives the already-resolved plan from the JSON artifact. The resolver runs exactly once
per well: at DAG planning time, in the writer rule.

**Not fully automatic.** The resolver does not scan all QC contracts and auto-include any
column ending in `_flag`. That would be spooky auto-wiring where "looks like a flag" ≠ "is
an approved exclusion reason". The eligible source universe is **explicit** (`_SOURCE_PAYLOADS`);
only approved steps can contribute flags to the snip_qc verdict.

---

## ➕ How to add a new QC exclusion flag

Example: adding `focus_flag` from the `focus_qc` stage.

**1. Add to policy** (in `contract.py` or config override):
```python
SNIP_QC_EXCLUSION_FLAGS = (
    ...existing...,
    "focus_flag",          # ← new flag
)
```

**2. Register the source** (in `flag_input_resolver.py`):
```python
from data_pipeline.quality_control.focus_qc.contract import FOCUS_QC_PAYLOAD_COLUMNS

_SOURCE_PAYLOADS = {
    ...existing...,
    "focus_qc": FOCUS_QC_PAYLOAD_COLUMNS,   # ← new source
}
```

**3. Register the step in `PIPELINE_STEPS`** (`paths.py`): ensure `focus_qc` has a per-well
CSV artifact (this is the source step's own registration — the resolver finds it from there).

**4. No manual edit to `snip_qc.smk` rule inputs required.** The rule's DAG gating inputs
(`_snipqc_source_shards`) are driven entirely by the resolver — they are the concrete paths
returned by `resolve_snip_qc_flag_sources`. Once `_SOURCE_PAYLOADS` and `PIPELINE_STEPS` are
updated (steps 2–3), the new source CSV and its `.validated` sentinel are automatically
included as rule inputs. Do not add them by hand.

If any step is missing, the resolver fails loud at startup with a clear message.

---

## ✅ Conformance checklist

- [ ] `flag_input_resolver.py` is the only `snip_qc` module importing `paths.py` — verified by test
- [ ] `_SOURCE_PAYLOADS` imports `*_PAYLOAD_COLUMNS` from source contracts — no column strings typed manually
- [ ] `resolve_snip_qc_flag_sources` is pure: no disk reads, no side effects
- [ ] `resolved_sources` JSON contains both `exclusion_flags` and `resolved_sources` keys
- [ ] `cmd_snip_qc` reads `exclusion_flags` from the JSON, not from `SNIP_QC_EXCLUSION_FLAGS`
- [ ] `write_snip_qc_resolved_sources_for_well` rule docstring states: source inputs gate DAG execution, not resolver inputs
- [ ] `resolved_sources` artifact is NOT in `PATH_MODE_MERGED` — per-well only
- [ ] Boolean flags coerced explicitly at load time; NA or unknown values fail loud
- [ ] Resolver error messages name the specific unresolved flags and list eligible steps
- [ ] `SNIP_QC_EXCLUSION_FLAGS` in `contract.py`; config may override; both planning and runtime use same resolved policy
- [ ] No reason→flag_column rename layer: `qc_fail_reasons` tokens ARE flag-column names
