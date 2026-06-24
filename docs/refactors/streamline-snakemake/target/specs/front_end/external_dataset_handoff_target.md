# External Dataset Handoff — the outside-world → frame_inventory target (🟢 TARGET)

**Status:** target spec, mdcolon 2026-06-23. The locked design for how an **external researcher with
their own organized data** enters the pipeline and runs it from segmentation onward. Spec only — no
implementation in this pass.

**Companion to:**
- `frame_inventory_handoff_contract.md` — the drop-in seam (the immutable frame key, the one-file +
  sentinel model, the native-vs-drop-in producer split). This doc is *how arbitrary external data
  reaches that seam*, grounded in what the code actually enforces.
- `front_end_naming_and_frame_inventory_flow.md` — the native ingest lineages + the fan.
- `plate_metadata_ingest_and_entity_qc.md` — the biology root (the three-layer L1/L2/L3 doctrine).

**Decision provenance:** `_WIP_external_handoff_decisions.md` (same folder) — the threaded decision log
with the rejected alternatives and the scar tissue. This spec is its clean distillation.

---

> ## One pipeline. Many entrances. Same contracts at the gate.

The external handoff does **not** create a second pipeline. It creates a **stricter entrance into the
same pipeline.** There is no "external mode" — after the entrance, no stage can tell which producer
ran. The doctrine, in one breath:

> **The metadata says what the biology means. The manifest says what images exist. The strict
> validator makes the image claims touch disk. `paths.py` owns where pipeline artifacts live. The
> consumers enforce biological completeness only when they need it.**

---

## 1. Intended user and the promise

**The user:** another **researcher** — a scientist with little-to-mild computational-biology
expertise, likely **driving the pipeline with an AI assistant** (Claude or similar).

**Consequence for the design:** **errors are the UX.** Every validator fails loud with the *fix named
in the message*, because the user's AI assistant reads the error and repairs the input. We do not ship
a tutorial; we ship contracts that explain themselves. This is the existing house style — leaned into,
not invented.

**The promise:** a user who did not use our microscope, who has organized their own images and biology
however they organize it, can push that data through the pipeline from segmentation onward — by
landing it in the **same two tables the native pipeline builds internally**: `frame_inventory`
(per-frame) and `plate_metadata` (per-well biology).

```
  external researcher data
    ├─ biology metadata  ──► plate_metadata  (per well_index) ──┐
    │                                                            │  joined LATE at
    └─ images + per-frame facts ──► frame_inventory manifest ──┐ │  consolidate_features
                                                               ▼ ▼
                                          strict per-well validator (the gate)
                                                               │
                                                               ▼
                                       same segmentation → snips → features → QC
```

Two independent roots, joined late — never fused at the door.

---

## 2. Required inputs

| Input | Shape | Required? |
|---|---|---|
| **frame inventory manifest** | one `dropin_frame_inventory.csv`, all wells (§4) | ✅ always |
| **biology metadata** | long CSV/sheet OR plate `.xlsx` → well-grain long table (§3) | ✅ for any biology-aware stage; not for bare segmentation |
| **`image_root`** | a directory for resolving *relative* `source_image_path` | ⬚ optional (only if the manifest uses relative paths) |

---

## 3. Biology metadata ingest

The biology root accepts **two external shapes, one contract** (the no-plate path is the linchpin —
it is what a researcher without a 96-well plate uses):

```
  plate.xlsx (8×12 grid sheets) ──► grid ingester (BUILT) ───┐
                                                             ├──► well-grain long table ──► L2 contract
  well_metadata.csv (long)      ──► long ingester (NEW)  ────┘
```

**Accepted well-key column names** (normalize to canonical `well_index` via the identifier grammar,
e.g. `A1`→`A01`): `well_index`, `well`, `well_name`. Fail loud only if NONE is present.

**Accepted age aliases:** `start_age_hpf`, `age_hpf`. **Canonical output column is `start_age_hpf`**
(keeps the staging formula honest: `predicted_stage_hpf = start_age_hpf + elapsed_h · rate(temperature)`).

> **Long-ingest collision policy (locked, built).** `age_hpf` and `start_age_hpf` may both appear
> ONLY if they agree (else fail loud); a user-supplied `well_id` is **dropped, never trusted** (identity
> is minted downstream from `experiment_id` + `well_index` and checked at L2); after normalization, no
> two passthrough value columns may collide (fail loud).

**Canonical output columns — always emitted, even when the user omitted them:**
`experiment_id`, `well_id`, `well_index`, `genotype`, `start_age_hpf`, `temperature`, `medium`.
A field the user did not supply is emitted as **all-NA**. The table's column *shape* never depends on
which fields the user happened to include.

> **Tiny doctrine — columns are the skeleton, values are the blood.** L2 checks the skeleton (the
> canonical columns exist, identity is consistent, no duplicate keys); L2 **allows NA values**.
> Whether NA biology is acceptable is decided by the *consumer* gates, not at ingest:
> - stage prediction requires `start_age_hpf` + `temperature` (already fails loud if absent);
> - genotype-aware analysis requires `genotype`;
> - medium-sensitive analysis requires `medium`.
>
> This collapses every downstream consumer's question from "is this column missing OR present-but-NA?"
> into the single case "the column exists; the value may be NA." Shape stability is pipeline oxygen.

---

## 4. Frame inventory contract (the manifest)

The manifest is the per-frame table the pixels cannot carry. It is the **same table the native
pipeline produces internally** — `REQUIRED_FRAME_INVENTORY_COLUMNS` in
`image_materialization/frame_inventory_contract.py` is the authority. The user authors the **atoms**;
`well_id` / `image_id` are **derived** and recomputed by the validator (a user-supplied composed id is
checked against the atoms and fails loud on disagreement — never trusted).

**The four immutable atoms:** `experiment_id`, `well_index`, `channel_id`, `time_index`.

> **Do I include `well_id` / `image_id`?** No. They are **intentionally absent** from
> `REQUIRED_FRAME_INVENTORY_COLUMNS` — the drop-in manifest **omits them**, and the splitter/validator
> derives them from the atoms before writing canonical frame_inventory shards. If a user *does* supply
> them, they are recomputed and checked against the atoms (disagreement fails loud); they are never
> trusted as authored.

**Per-frame facts the user authors alongside the atoms:**
- `source_image_path` — TIFF / PNG / JPEG; **absolute OR relative to `image_root`** (§5 policy);
- `source_micrometers_per_pixel` — the SOURCE pixel size of the submitted image, **required > 0**.
  *(This is the exact frame_inventory contract column — the validator references it, never mints a
  second pixel-size name. The upstream acquisition inventory uses `micrometers_per_pixel`; do not
  conflate the two — each is named where it lives.)*
- `image_width_px`, `image_height_px` — declared dims, usually populated by the scaffold helper (§6)
  from the image headers, then self-checked against the real image header at L4 (users rarely type
  these by hand);
- `elapsed_time_s`, `acquisition_time_s` — the time block (**conditionally required** — see below).
  `elapsed_time_s` is the required temporal axis for multi-timepoint wells; `acquisition_time_s`, when
  present, is raw/source timing provenance and is **not** the downstream timing contract (a future
  validator must not require both for multi-timepoint data).

### The time / age rule (the one genuinely conditional requirement)

`predicted_stage_hpf` decomposes across both roots — `elapsed_time_s` per frame, `start_age_hpf` +
`temperature` per well. So the time block is **conditionally required by the data itself**, enforced
by a per-well grain rule (no "is this a snapshot?" flag anywhere):

> **If a well has more than one distinct `time_index` / timepoint, `elapsed_time_s` is required.**
> A single-timepoint well may omit it.

A single timepoint with multiple channels (BF + fluorescence) is still single-timepoint — the rule is
**temporal**, not a count of frames or image_ids. Some absolute embryo age (`predicted_stage_hpf`) is
derivable **iff** the user also supplied `start_age_hpf` (+ `temperature`) in the biology metadata; if
they did not, the stage-prediction consumer says so by name.

---

## 5. Strict per-well validation (the gate)

ONE validator, used identically by both producers (native + drop-in). It grows the **behavior** of the
existing `validate_frame_inventory` (today the weak schema check — resolves its `TODO(Scope 2)`) while
**moving the implementation** into `frame_inventory_validation.py` (§7). It runs as an
ordered sequence of contract rules; it is **non-mutating** — it writes only the `.validated` sentinel
on pass, or an errors report + raises on fail (the one-file model). It never rewrites the manifest.

```
validate_frame_inventory(input_csv, output_flag, *,
                         image_root=None, check_sources=False, validation_scope="per_well")
  L0  schema          required columns present, dtypes        (io/validators.validate_dataframe_schema)
  L1  uniqueness      identity-anchored image_id key unique    (existing)
  L2  derived ids     well_id / image_id recompute from atoms  (existing assert_derived_ids_consistent)
  L3  grain  (ALWAYS, SCOPE-AWARE) one experiment_id (both scopes);
                      per_well: one well_index;
                      merged:  many wells allowed, per-well checks applied grouped by well_id;
                      BF contiguous 0..N-1; channels rectangular (same time_index set);
                      multi-timepoint ⇒ elapsed_time_s required

  (BF is the required reference segmentation channel — `REQUIRED_CHANNEL = "BF"` in the contract.
   Wells without a BF channel are not accepted by this entrance; the contiguity rule is anchored to it.)
  L4  sources (check_sources) — the image/folder contract (below)
  → PASS: write .validated sentinel
  → FAIL: write errors report + raise (per_well → {well_id}_frame_inventory.errors.md;
          merged → {experiment_id}_frame_inventory.errors.md;
          fallback frame_inventory.errors.md when the id can't be derived / table unreadable)
```

> **`grain` names the row; `validation_scope` names the gate.** The L3 scope argument is
> `validation_scope` (`"per_well"` | `"merged"`), NEVER `grain` — `grain` is reserved across the
> feature/QC specs for row identity grain (`snip_id`, `physical_embryo_id`, `well_id`).

> **The merged node CANNOT run the per-well "one well_index" rule.** A merged experiment-level table
> has one `experiment_id` but MANY wells; `validation_scope="merged"` allows many wells and applies
> the BF-contiguity / rectangularity / multi-timepoint-⇒-`elapsed_time_s` checks **grouped by
> `well_id`**. `validation_scope="per_well"` asserts exactly one well. Both assert one experiment.

**`check_sources` is a mode flag (one validator, two modes):** the **Python default is conservative
(`check_sources=False`)** so no caller silently inherits strict L4 during the refactor; the
"drop-in strict" stance is enforced by each call site passing `--check-sources` explicitly, not by
the default. Proving the claimed images are real is the point of the drop-in (and native per-well)
entrance, so those nodes pass `check_sources=True`. The **merged node explicitly passes
`check_sources=False`** (re-opening every image for the aggregate view is wasteful — L0–L3 still run,
grouped). L3 grain checks always run, at the node's scope.

The validator is wired at two `frame_inventory.smk` nodes: the per-well node
`validate_frame_inventory_for_well` (`check_sources=True`, `validation_scope=per_well`, `image_root`)
and the merged node `validate_frame_inventory` (`check_sources=False`, `validation_scope=merged`).
**Native per-well L4 is justified by audit:** the native YX1 per-well shard already writes a real
`source_image_path` + dims + µm/px, so `check_sources=True` there is meaningful (it catches a
corrupt/missing materialized image). Keyence per-well L4 is gated on the same per-scope confirmation
when Keyence materialization lands.

### L4 — the image / folder contract (gated by `check_sources=True`)
For each row: `source_image_path` resolves (policy below) → file **exists** → image **opens**
(TIFF/PNG/JPEG) → real pixel dims **==** declared `image_width_px` / `image_height_px` →
`source_micrometers_per_pixel` **> 0**.

> **There is NO separate "well folder contract" object.** L4 *is* the image/folder contract — an
> optional stricter inspection mode on the frame_inventory validator. Ownership, locked:
> `paths.py` owns **where** pipeline artifacts live (shard / sentinel layout under `per_well/{well_id}/`);
> the frame_inventory validator owns **what** images claim to be (the table); L4 **proves** the claimed
> image paths are real. *The table makes the claim. The image root gives relative claims a home. The
> strict validator makes the claim touch disk.*

### Path-resolution policy (accept both; never rewrite the CSV)
```
if path.is_absolute():
    resolved = path                      # validate directly — absolute paths are NOT forced inside image_root
else:
    if image_root is None:               # relative + check_sources=True + no root → FAIL LOUD
        raise ValueError("relative source_image_path requires image_root when check_sources=True")
    resolved = image_root / path
    assert resolved.resolve().is_relative_to(image_root.resolve())   # relative paths may not escape via ..
```
Relative paths are resolved against `image_root` **during validation**; the source CSV is **not
rewritten** by the validator. (A normalized-absolute-path artifact, if ever wanted, is a separate
explicit build step — not the validator's job.)

---

### Downstream trust boundary — temporal coherence

The strict per-well frame_inventory validator is the **single temporal-coherence gate** for native and
drop-in data. By the time segmentation, tracking, and `physical_embryo_registry` run, the validated
per-well frame_inventory shard has already proven:
- one `experiment_id` per shard;
- one `well_index` / `well_id` per shard;
- unique `image_id` rows;
- BF `time_index` values are contiguous;
- all present channels share the same `time_index` set;
- wells with more than one distinct `time_index` have `elapsed_time_s`;
- source image paths resolve and image dimensions match declared dimensions when `check_sources=True`.

**Downstream products do not re-check these frame-grain invariants.** `physical_embryo_registry` trusts
frame_inventory temporal coherence and adds **only identity coherence**: one tracked animal identity,
one `physical_embryo_id`, a stable identity spine. It does **not** re-check BF contiguity, channel
rectangularity, elapsed-time presence, or image source validity — that would be duplicated contract
enforcement of an invariant the gate already owns.

Likewise **biology completeness** is not enforced at the door: L2 plate metadata allows NA, and the
deferred `entity_metadata_completeness_qc` (L3) flags a *registered* embryo missing a required field —
conditional on `physical_embryo_registry` existence, at `snip_id` grain. The §3 decision to emit
canonical nullable biology columns is precisely what lets L3 check `isna()` on a guaranteed-present
column instead of playing "missing column vs. present-but-NA." External/drop-in biology simply arrives
sparser; L3 is the gate that catches it, and needs no external variant because it lives past the
agnostic seam.

> **Doctrine:** *Validate an invariant where it lives. Trust it downstream. Do not make every consumer
> become a tiny customs office.* The manifest makes claims; the validator makes claims touch disk;
> downstream consumers trust upstream gates and add only their own invariants.

This trust is only sound because the gate cannot be skipped. **Any entrypoint that feeds
segmentation / tracking / `physical_embryo_registry` without a validated per-well frame_inventory
shard is outside contract** — a later "run tracking on a folder" shortcut would break the trust
boundary and is not permitted. There is no second door; downstream stages do not defend themselves
because nothing reaches them un-gated.

---

## 6. Scaffold helper (Phase 1 — emit a starter manifest, do not reorganize)

For the "researcher + AI assistant" user, a blank CSV is a worse start than a filled-in draft. Phase 1
ships a small documented helper (`scaffold_dropin_inventory.py`) that reads an image directory, reads
each image header for the real dims, and **emits a starter `dropin_frame_inventory.csv`** the user
edits (filling `source_micrometers_per_pixel` and correcting any inferred frame atoms —
`channel_id`, `time_index`). It is the inverse read direction expressed as **free functions, not a
class**; it **does not move or reorganize the user's images** in Phase 1 (manifest-is-truth makes
reorganization never required).

> Age does **not** live here. `start_age_hpf` is biology metadata (§3), not a frame_inventory column —
> the scaffold helper only touches the manifest. Two roots; the vines do not cross the trellis.

---

## 7. Files — ADD / CHANGE / UNTOUCHED

Source code grouped by **function**; orchestration grouped by **workflow entrance**. No new source
package; no class; no `validation/` subpackage (see §9).

**ADD**
- `metadata_ingest/frame_inventory/frame_inventory_validation.py` — the public gate (sequences L0–L4).
- `metadata_ingest/frame_inventory/frame_inventory_validation_rules.py` — the product-specific contract
  rules (BF-contiguity, rectangularity, elapsed-time, source-readability, declared-dims, path-resolution).
- `metadata_ingest/frame_inventory/scaffold_dropin_inventory.py` — image dir → starter manifest.
- `metadata_ingest/well_discovery/discover_wells_from_handoff.py` — drop-in twin of
  `discover_wells_from_scope_metadata.py` (one experiment_id, derive well_id, reuse the discovered-wells contract).
- `metadata_ingest/well_discovery/split_dropin_inventory.py` — `split_dropin_inventory_by_well()`.
- `pipeline_orchestrator/rules/dropin_handoff.smk` — external-entry orchestration rules. **The main
  Snakefile / orchestration include list must `include:` this file** — otherwise it is a decorative
  canoe that no workflow can reach.
- mirror tests under `tests/data_pipeline/...` for each.

**CHANGE**
- `metadata_ingest/frame_inventory/frame_inventory.py` — product/table ops only (keeps
  `merge_frame_inventory_shards` + the reader); the validator MOVES OUT (import in `tasks.py` repoints
  — a live-code refactor, do it green before growing the gate).
- `metadata_ingest/plate/plate_metadata_loader.py` — implement the long-table ingester; age-alias;
  emit canonical nullable biology columns.
- `metadata_ingest/plate/plate_metadata_contract.py` — confirm biology columns required-as-columns,
  values nullable.
- `pipeline_orchestrator/tasks.py` — `validate-frame-inventory` gains `--image-root` + `--check-sources`;
  new thin verbs for discover-from-handoff / split / scaffold; repoint the validator import. The long
  biology path adds **no new ingest verb** — `ingest-plate-metadata` already exists, and the
  long-vs-grid branching happens inside `plate_metadata_loader`, not at the verb layer.
- `pipeline_orchestrator/rules/frame_inventory.smk` — per-well node `check_sources=True` + image_root;
  merged node `check_sources=False`.

**UNTOUCHED (by design)**
- `image_materialization/materialized_image_paths.py` — pure path functions; recommended target layout,
  not a gate. No class.
- `image_materialization/frame_inventory_contract.py` — the column manifests + derived-id helpers; the
  rules module IMPORTS from here, never duplicates the column vocabulary.
- `io/validators.py::validate_dataframe_schema` — the generic L0 primitive; reused, unchanged.

---

## 8. Non-goals

- **No zero-setup UX.** The user can run the pipeline (envs, cluster). The bridge is data-shape, not
  infrastructure.
- **No user-configurable layout** — the canonical tree is fixed; no `LayoutSpec` class.
- **No separate folder-contract object** — L4 source checks are the folder contract.
- **No second pipeline / "external mode"** — one stricter entrance into the same pipeline.

---

## 9. Future work (do NOT build now)

- **Scaffold helper Phase 3:** copy/symlink images into the canonical `materialized_image_paths` tree
  → a fully self-describing dataset.
- **Promote validation to a `validation/` subpackage** ONLY if the rules module exceeds ~400–500 lines,
  source validation needs multiple backends, grain/source/schema rules need independent test fixtures,
  a second public gate appears, or drop-in validation becomes a public API surface separate from native.
  None hold today — a package is not earned by long filenames, only by a real subsystem.
- **A normalized-absolute-path manifest artifact**, if path portability ever needs materializing
  (a build step, never the validator).

---

## ✅ DECISIONS (this doc)
1. External handoff = a **stricter entrance** to the same pipeline; two roots joined late; no external mode.
2. **Biology accepts long OR plate.xlsx → well-grain long;** finishing the long ingester is the no-plate path.
3. Accept well-key aliases (`well_index`/`well`/`well_name` → `well_index`); accept age aliases
   (`start_age_hpf`/`age_hpf` → canonical `start_age_hpf`).
4. **Long-ingest emits canonical nullable biology columns even when absent;** L2 validates skeleton, consumers validate completeness.
5. Manifest = atoms + path + calib + dims + time block; `well_id`/`image_id` derived, recomputed by the gate.
6. **Time rule is temporal:** >1 distinct `time_index` ⇒ `elapsed_time_s` required.
7. **`source_micrometers_per_pixel`** is the frame_inventory contract column (validator > 0); no second vocabulary.
8. Strict `validate_frame_inventory` **behavior is preserved/grown** as the public gate (ordered L0–L4,
   **non-mutating** — sentinel/errors only), but the **implementation moves** out of `frame_inventory.py`
   into `frame_inventory_validation.py`.
9. **`check_sources` mode flag:** default `True` (drop-in); merged node explicitly `False`.
10. **Path policy:** absolute validated directly (not forced under `image_root`); relative resolved against
    `image_root` (fail loud if absent under `check_sources`), no `..` escape; CSV never rewritten.
11. L4 source checks ARE the image/folder contract — no separate folder-contract object.
12. **Flat validation modules** (`frame_inventory_validation.py` + `frame_inventory_validation_rules.py`);
    **no `validation/` subpackage** yet.
13. Drop-in discovery/split code lives in `well_discovery/` (beside the twin); orchestration in a sibling
    `dropin_handoff.smk`; **no new source package, no `stitched_handoff/`/`dropin/` kingdom.**
14. **Scaffold helper Phase 1:** emit a starter manifest, free functions, no image reorganization.
15. `materialized_image_paths.py` stays pure functions — recommended layout, not a gate, no class.

## 🔗 RELATED-DOC UPDATES NEEDED
- `frame_inventory_handoff_contract.md`: add the time block to "Required core columns" and mark it
  drop-in-conditional (temporal rule); cross-link this doc as the "how external data reaches the seam"
  companion.
- `plate_metadata_ingest_and_entity_qc.md`: note the long ingester is also the external no-plate path;
  the L2 contract is already grain/format-agnostic (skeleton required, values nullable).
- `README.md` (target index): add this doc under `specs/front_end/`.
