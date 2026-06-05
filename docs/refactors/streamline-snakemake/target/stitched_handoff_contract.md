# Stitched Handoff Contract — the drop-in seam (🟢 TARGET)

**Status:** active spec, mdcolon 2026-06-04. Defines the **one explicit contract** for the
stitched-image handoff so an external dataset can be organized and pushed through the pipeline
from segmentation onward — the way any standard image-processing pipeline declares its input
format.
**Companion to:** `front_end_naming_and_flow.md` (the front-end ingest + fan spec — this doc is
its post-fan "drop-in here" counterpart) and `per_well_throughline_findings.md` (the north-star
findings doc; the stitched image tree is **off-registry** there).
**Scope of THIS doc:** the stitched-image **directory tree** (the pixel store), the
**`frame_inventory`** table that travels with it, the **immutable frame key**, the **strict
shared validator** (`validate_frame_inventory_well`), and a **worked walkthrough** of both
producers — native microscope ingest **and** external drop-in.

> **Vocabulary note (2026-06-04):** this doc adopts **`frame_inventory`** as the name of the
> per-well validated table, replacing the older `frame_contract`. The code rename
> (`frame_contract` → `frame_inventory`: 37 Snakefile refs + 3 Python modules + schema) is a
> **Scope-2 migration**, logged in Refactor Items — **not** executed in this doc pass.

---

## 🎯 WHY THIS DOC EXISTS — the one un-named handoff

Every other handoff in this refactor was made an **explicit, named contract**:
`scope_metadata_mapped.csv`, `discovered_wells.txt`, the per-well shards, the `STAGES` registry.
The **stitched image tree is the exception** — by design it is *off-registry* (an image
directory, not a tabular artifact; findings doc, Lean MVP Contract), so it has had no written
contract. Its layout lives only as an opaque-string path builder in code (`stitched_ff_builder.py`).

That is the gap this doc closes. The stitched seam is the **natural drop-in point** because it
is the **first microscope-agnostic artifact** in the pipeline:

```
  per-microscope ingest ──► FAN ──►  STITCHED HANDOFF  ──► segmentation ─► snips ─► features ─► QC
   (yx1 / keyence,                    (this doc:           (everything downstream is
    raw reads)                         agnostic, drop-in)   microscope-agnostic already)
```

> **The load-bearing claim:** everything **upstream** of the stitched tree is
> microscope-specific (raw reads, series→well mapping). Everything **downstream** of the
> `frame_inventory` is microscope-agnostic and already consumes it. So a user who can produce
> **(1) a stitched tree + (2) a valid frame inventory** can run the pipeline from segmentation —
> *without* a Keyence or YX1 microscope, without our ingest stages.

---

## 🧱 THE FOUR THINGS — what each artifact answers (lock this vocabulary)

The stitched image tree **is not the contract and not the metadata. It is the pixel store.**
Keeping these four roles distinct is what makes the seam legible:

| Thing | Answers | What's actually in it |
|---|---|---|
| **stitched image tree** = **`image_id`s on disk** | *Where are the image files?* | one image file per `image_id` (`{well_id}_{channel_id}_t{time_index:04d}.{ext}`). The path/filename **encodes the frame key** — the tree IS the set of `image_id`s realized on disk. Pixels only; no calibration, no timing. |
| **scope metadata** (`scope_metadata_mapped.csv`) | *What frames did the front-end EXPECT, with what facts?* | one experiment-grain table (native only): one row per expected frame, carrying the frame-key atoms **plus** the facts pixels can't hold — `source_micrometers_per_pixel`, dims, channel. |
| **`frame_inventory.csv`** (built → validated) | *What per-frame image rows do we trust?* | one **per-well** table: the frame-key atoms + `source_image_path` + calibration + dims + derived `image_id`. = expected facts JOINED to observed image files. **This is the only thing segmentation reads.** |
| **`.validated` sentinel** | *Has this well's inventory passed the gate?* | a near-empty flag file; its *existence* = passed. Carries no detail (detail goes to the fail-only errors file). |

Mental model: **`image_id`s on disk (pixels) → expected facts table → joined+trusted per-frame
table.** There is **one** per-frame table per well (`frame_inventory`); its validation status is a
**sentinel**, not a different filename. (We do **not** keep a separate "candidate" file — see
One-File Model.)

---

## 🪨 THE IMMUTABLE FRAME KEY — the invariant in stone letters

> **The frame key is the four RAW ATOMS, immutable after scope-metadata mapping:**
> **`experiment_id + well_index + channel_id + time_index`**
> **Every stage may CHECK, ENRICH, or FILTER-BY-FAILING-LOUDLY. No stage may silently
> reassign, renumber, infer a different time axis, change `channel_id`, change a well's
> `well_index`, drop an expected frame, or mint an inconsistent `well_id`/`image_id`.**

**The key is the raw atoms; `well_id` and `image_id` are DERIVED compositions of them** — not
extra key fields. (Listing `well_id` *and* `experiment_id` in the key would be redundant:
`well_id` already contains `experiment_id`.)

```
well_id  = {experiment_id}_{well_index}                  # composition of two atoms
image_id = {well_id}_{channel_id}_t{time_index:04d}      # = {experiment_id}_{well_index}_{channel_id}_t{time_index:04d}
```

> **Naming caveat (mdcolon).** It is the *combination of the four raw atoms* that is unique —
> `image_id` is the composed **name** for that combination, a convenience, not a separate
> identity. The atoms are the truth; the composed ids are derived labels.

The validator checks that any composed id present (`well_id`, `image_id`) agrees with the atoms;
if absent it derives and writes them into the validated `frame_inventory`. This invariant is *why*
the same validator can serve both producers: it never trusts a producer to have composed the ids
correctly — it recomputes them from the atoms and fails loud on disagreement. (Generalizes the
`validate_well_id` fail-loud rule — `well_id_throughline_refactor_plan.md`, Scope 2 — to the whole
per-frame key.)

---

## ⭐ NORTH STAR — submission surface vs. operational spine

The **public contract wants to be one dataset-level thing** ("here is my dataset"), but the
**DAG spine is per-well**. These are not in conflict once you separate the two levels:

> **A stitched handoff may be authored as a dataset-level submission, but the pipeline converts
> it at the gate into per-well validated `frame_inventory` shards. Those shards — not the
> submission file — are the post-fan DAG spine.**

| Level | What it is | Who reads it |
|---|---|---|
| **Submission surface** (ergonomic) | the stitched image tree **+ one dataset-level `dropin_frame_inventory.csv`** (all wells) | the **gate only** — `discover_wells_from_handoff` + `build_frame_inventory_well` |
| **Operational spine** (correct) | **per-well** validated `{well_id}_frame_inventory.csv` shards | **everything downstream** — segmentation reads its well's shard, never the big file |

**The rule:** *big file allowed at the door; per-well shards required inside the house.* The
dataset-level file is **ingress-only** — read only by discovery and the per-well build, never by
a well-local compute stage (that would recreate the merge-wall the per-well spine exists to avoid
— findings doc, "spine rule ☠️").

---

## 📦 THE CONTRACT — submission surface (two components) → per-well spine

The **submission surface** has two required **components** (a directory tree is not a single
"artifact" like a CSV — hence *components*). The gate fans them into per-well **artifacts**.

### (1) The stitched image tree — `image_id`s on disk (off-registry, keyed on `well_id`)

Each file **is an `image_id`**: the filename is the composed frame key, so the tree is just the
set of `image_id`s realized as pixels on disk.

```
{output_root}/built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/{well_id}_{channel}_t{time_index:04d}.{ext}
                                                          └─ GLOBAL well_id   └─ BF, GFP…   └─────────── image_id ───────────┘.{ext}
                                                                                            (= {well_id}_{channel}_t{time_index:04d})
```

> **Recommended layout vs. source of truth (decided 2026-06-04).** The tree above is the
> **recommended** layout (portable, browsable, self-describing). But the **validator's source of
> truth is `source_image_path`** in the inventory — images may live anywhere that path resolves
> (absolute or relative to `image_root`). An external user need **not** reorganize into this exact
> tree. Filename mismatch against the derived `image_id` is a **warning**, not a hard failure.

| Rule | Value | Why |
|---|---|---|
| **One image per `(well_id, channel, time_index)`** | a single flat-field-corrected, stitched image | one file = one frame |
| **Image format is flexible** | **TIFF, PNG, or JPEG** — preferred TIFF/PNG | the loader keys on `source_image_path`, not a fixed suffix (see Image Format) |
| **`well_id` is GLOBAL, per well** | `{experiment_id}_{well}` e.g. `20250912_B01` | segmentation keys on `well_id` — strict, non-negotiable |
| **`channel` is a controlled token** | `BF`, `GFP`, … | used in the frame key + joins |
| **`time_index` is the T dimension** | contiguous, 0-based, `_t{n:04d}` | microscopy convention; matches the on-disk filename |
| **Per-well `.done` sentinel** | **native stitcher only** — NOT user-authored | a drop-in user authors zero dotfiles (see Sentinels) |

> **🖼️ Image Format.** Input images can be **TIFF, PNG, or JPEG** — preferred TIFF/PNG. The
> pipeline is *not* TIFF-only; `source_image_path` carries the actual extension. **The pipeline's
> own main image OUTPUT is PNG** (viewable directly in a VS Code session over the cluster).

This tree is **off-registry** — its own path helper, not a `STAGES` row, keyed on `well_id`
(`front_end_naming_and_flow.md`, Decision 7).

### (2) The frame inventory (the per-frame table the pixels can't carry)

The pixels do **not** know their calibration, timing, or biology. The `frame_inventory` is the
per-frame table that does. **This table IS the metadata** — there is no separate metadata file
travelling alongside; the inventory is the per-frame scope metadata joined to the image paths.

> **🔑 The drop-in file is the SAME TABLE the native pipeline builds internally.** This is the
> clearest way to understand the contract: a `dropin_frame_inventory.csv` is *exactly* what
> `build_frame_inventory_well` produces in the native path — the frame-key atoms + image path +
> calibration + dims. The native pipeline **derives** it (join `scope_metadata_mapped` to the
> stitched tree); a drop-in user **authors** it directly. Same columns, same meaning, same
> validator. The only difference is *who filled it in*. So "what does a drop-in dataset look
> like?" = "what does the native pipeline's per-well inventory look like, written by hand?"

**At ingress (drop-in) it is ONE dataset-level file (all wells):**
```
dropin_frame_inventory.csv      ← user authors this: rows for ALL wells, the submission surface
                                  (identical schema to the native per-well inventory, just unsplit)
```
**The gate fans it into per-well spine artifacts (what the DAG consumes):**
```
frame_inventory/{exp}/per_well/{well_id}/{well_id}_frame_inventory.csv            ← per-well shard (well_id IN the filename)
frame_inventory/{exp}/per_well/{well_id}/{well_id}_frame_inventory.csv.validated  ← sentinel: PASSED the gate (existence = trusted)
frame_inventory/{exp}/per_well/{well_id}/{well_id}_frame_inventory.errors.md      ← ONLY on FAILURE (what broke); absent on pass
frame_inventory/{exp}/{exp}_frame_inventory.csv                                   ← merged view (off-spine; nothing downstream reads it)
```

> **`well_id` in the filename (decided 2026-06-04).** The shard is `{well_id}_frame_inventory.csv`,
> not just `frame_inventory.csv` under a `{well_id}/` dir. The file **names itself** — grep a
> `well_id` across the tree and every artifact for that well (csv, sentinel, report) carries the
> tag. Matches the front-end doc's "a `well_id` is the same string in a log, a filename, a row, or
> a path." It also lets **build** and **validate** be separate rules without an output collision
> (see One-File Model).

> **Segmentation reads `…/{well_id}/{well_id}_frame_inventory.csv` — NOT `dropin_frame_inventory.csv`.**
> The big file is ingress-only; it never appears as a well-local `input:`.

(`frame_inventory/` is the leaning family name; today's code writes under `experiment_metadata/`
— findings #5, family choice still open.)

**Required core columns** (the minimal set — everything else is enrichment):

**The user authors ATOMS + calibration + dims. The pipeline DERIVES the composed ids** (`well_id`,
`image_id`) — exactly like the native path, where the user never composes an id either. This keeps
the drop-in file the *same table* the native pipeline builds internally.

| Column | Meaning | User authors? | Source when WE generate it |
|---|---|---|---|
| `experiment_id` | global experiment id — **atom** | ✅ | constant per run |
| `well_index` | local well label (`B01`) — **atom** | ✅ | from scope metadata |
| `channel_id` | controlled channel token — **atom** | ✅ | from scope metadata |
| `time_index` | the **T dimension** (0-based, contiguous) — **atom** | ✅ | from scope metadata (was `frame_index`/`time_int` — see Naming) |
| `source_image_path` | image path (TIFF/PNG/JPEG); **absolute OR relative to `image_root`** | ✅ | built from the stitched-tree layout |
| `source_micrometers_per_pixel` | calibration (µm/px) — **required, > 0** | ✅ | from scope metadata calibration |
| `image_width_px` | declared width | ✅ | from the image header |
| `image_height_px` | declared height | ✅ | from the image header |
| `well_id` *(derived)* | `{experiment_id}_{well_index}` | ⛔ **do not author** | composed + written by the build step |
| `image_id` *(derived)* | `{well_id}_{channel_id}_t{time_index:04d}` | ⛔ **do not author** | composed + written by the build step |

> **`well_id` and `image_id` are derived AND written, never authored.** The build step composes both
> from the atoms and writes them into the shard. **If the user *does* supply one** (e.g. pastes a
> `well_id` column), the validator **recomputes it from the atoms and fails loud on disagreement** —
> it never *trusts* a user-supplied composed id. So the contract asks for atoms only; supplying a
> composed id is tolerated-but-checked, not required.

> **Why dims are required despite being derivable:** `image_width_px`/`image_height_px` are
> **intentionally duplicated** from the image header so the validator can detect mismatches
> (header says 2048×2048 but the manifest claims 2048×1024 → fail loud). They are a self-check,
> not new information.

> **`source_micrometers_per_pixel` is the SOURCE pixel size** of the submitted image as stored —
> not a desired resampling target. It is the one genuinely external scientific fact the user
> supplies; everything else is the file's coordinates or encoded in the path/filename.

> **📁 Relative vs. absolute paths (accept both, prefer relative).** `source_image_path` may be
> absolute *or* relative to an `image_root` (see `StitchedHandoffSpec`). Relative makes the dataset
> **portable**; the validator **canonicalizes to absolute** in the shard, so the spine is always
> unambiguous. Validation errors if a path is relative and `image_root` is `None`, or if a given
> `image_root` doesn't exist.

**Enrichment columns** (genotype, treatment, medium, temperature, start_age_hpf, timing) come from
the **plate lineage** and join far downstream at `consolidate_features` — **not** part of this
handoff, **not** required for a drop-in segmentation run (`front_end_naming_and_flow.md`: the plate
lineage is a separate root).

#### Format: CSV all the way through this seam (decided 2026-06-04)

**Decision: CSV in, CSV out.** The user authors a CSV; the per-well shard is
`{well_id}_frame_inventory.csv`. **Grounded in the code:** `build_frame_contract` writes `.csv`,
the Snakefile wires `.csv` (37 refs), and segmentation's reader (`run_per_well.py:27-29`) is
format-agnostic (`if .parquet → read_parquet else read_csv`). **CSV is the pipeline's standard for
contract tables** (parquet only for the big binary snip manifest). A small inspectable text table
at a contract boundary is worth far more than a parquet micro-optimization — debugging a handoff
failure means *reading the shard*. Parquet stays a free future swap (segmentation already accepts
it). *This supersedes the earlier "CSV-in / parquet-internal (leaning)" note.*

---

## 🔁 THE ONE-FILE MODEL — built vs. validated by sentinel, not by name

There is **one** per-well inventory file. Its validation status is carried by a **sentinel**, not
by a second filename. (We do **not** write a separate "candidate" file — that would duplicate the
per-frame key across two near-identical tables.) This matches the convention the codebase already
uses everywhere: `X.csv` + `.X.validated`.

```
build_frame_inventory_well[well_id]
    → {well_id}_frame_inventory.csv                 (built; not yet validated — no sentinel)

validate_frame_inventory_well[well_id]              ← THE shared gate (native + drop-in, identical)
    reads {well_id}_frame_inventory.csv
    → {well_id}_frame_inventory.csv.validated       (on PASS — sentinel only)
    → {well_id}_frame_inventory.errors.md + raises  (on FAIL — the report exists ONLY when useful)

segment_and_track_per_well[well_id]
    depends on the .validated sentinel → reads only {well_id}_frame_inventory.csv
```

**Two rules, not one** (build → validate), and validate writes **only the sentinel** (on pass) or
**an errors report + raise** (on fail) — it never rewrites the csv. This keeps the validator its own
**exposed DAG node** (the drawbridge — it can be inspected and reused), gives clean per-well
staleness, and avoids the Snakemake "two rules can't share one output" collision (the `{well_id}_`
filename makes each well's file distinct; build owns the `.csv`, validate owns the `.validated`).

> **Why fail-only report, not an always-written one (mdcolon's complexity rule).** The validator
> already *computes* every failure reason — dumping them to `{well_id}_frame_inventory.errors.md`
> before raising is ~2 lines and is the whole "drop-in user sees what broke" value. A *success*
> report is formatting nobody reads + an extra happy-path output to wire. So: errors file on
> failure (cheap, useful), sentinel on success (the flag), no redundant success `.md`.

> **The validator is the promotion step.** An inventory with no `.validated` sentinel is *built
> but untrusted*. The sentinel is the only thing that marks it safe. The filename stays neutral
> (`frame_inventory`, not `validated_frame_inventory`); the sentinel records status.

---

## 🔀 TWO PRODUCERS, ONE VALIDATOR — native and drop-in

The **only** divergence is **who builds `frame_inventory.csv`**. After the build, the pipeline
cannot tell which producer ran — that is the design goal, and it realizes the locked requirement:
*the same validator runs end-to-end (microscope) or drop-in.*

**Native ingest (pipeline-generated images):**
```
scope_metadata_mapped.csv  (expected: well/channel/time/calibration — METADATA-ONLY, no images)
   ↓  discover_wells_from_metadata (checkpoint) → discovered_wells.txt
   ⟱ FAN ⟱   (fan is metadata-only, so active_wells exists BEFORE stitching — findings #13)
   ↓  stitch_well[well_id]  →  images on disk  +  .well_{well_id}.done
   ↓  build_frame_inventory_well[well_id]:  join expected scope rows + OBSERVE the stitched tree
   │      → {well_id}_frame_inventory.csv
   ↓  validate_frame_inventory_well[well_id]            ◄── shared gate
   ↓  segment_and_track_per_well[well_id]
```

**External drop-in (user already has images; no stitcher runs):**
```
dropin_frame_inventory.csv  (one big file, ALL wells — the user's expected+observed in one)
   ↓  discover_wells_from_handoff (checkpoint):
   │      • exactly one experiment_id (else RAISE)
   │      • every well_id global (else RAISE)
   │      → discovered_wells.txt
   ⟱ FAN ⟱   (NO stitch step — user already has images)
   ↓  build_frame_inventory_well[well_id]:  SELECT this well's rows from the big file
   │      → {well_id}_frame_inventory.csv
   ↓  validate_frame_inventory_well[well_id]            ◄── SAME shared gate, no source branch
   ↓  segment_and_track_per_well[well_id]
```

> **Same `discovered_wells.txt`, same build output, same gate, same shards, same segmentation.**
> `discover_wells_from_handoff` is the **drop-in twin** of `discover_wells_from_metadata` — both
> read a **table** (never images) and emit the identical `discovered_wells.txt`, so `active_wells`
> exists before any per-well work and **run-one-well-at-a-time works identically in both paths.**

**Why discovery is metadata-only (load-bearing for per-well):** the fan must produce the well list
*before* image work, or you can't fan until stitching finishes and single-well runs get worse.
Verified locked (findings #13): `discover_wells_from_metadata` reads `scope_metadata_mapped.csv`,
"canonical metadata, no images." The drop-in twin reads the user's table — also no images at
discovery. The **stitched tree is observed at `build_frame_inventory_well`, not at discovery.**

**Snakemake input-declaration note:** Snakemake can't declare "every path inside a CSV" as an
`input:` before reading the CSV. Fine — the **build + gate are the declared DAG nodes**; they open
every file and fail loud. Individual images are validated *inside* the gate, not declared as graph
edges.

**Do we wait for all stitching? (native only.)** If the stitcher loops all wells in one job
(`execution=single`, findings doc), native validation waits for that job's per-well `.done`
sentinels, then fans. Drop-in has no stitch job, so it fans immediately after discovery. (The
`fanout` vs `execution` distinction: same per-well `fanout`, different `execution`.)

> **The build/validate split is NOT post-MVP — it is the design.** Earlier drafts called the
> candidate/validate split "post-MVP." That's superseded: `build_frame_inventory_well` (per
> producer) → `validate_frame_inventory_well` (shared) is the MVP shape, because it's exactly what
> makes one validator serve both producers. What *is* deferred: making the native `build` itself
> richer (e.g. a separate observed-inventory table) — see Open.

---

## 🔒 THE SHARED VALIDATOR — `validate_frame_inventory_well` (STRICT, file-level)

One validator, used identically by both producers. It is **strict and file-level**, not
schema-only — an external dataset has had none of our upstream guarantees, and "check the deck
going in" is the whole point.

> **Today there is exactly ONE shared validator and it is weak:** `io/validators.py::
> validate_dataframe_schema` checks only (1) required columns present, (2) no nulls. No stage
> opens images or checks dims/calibration/contiguity. This gate = that existing structural check
> **plus** a new shared file-level check. Promoting it is the work (Refactor Items).

```
validate_frame_inventory_well[well_id]:   (operates on the RAW ATOMS: experiment_id, well_index, channel_id, time_index)
  [x] schema — required columns present, correct dtypes        (existing validate_dataframe_schema)
  [x] per-well purity — exactly one experiment_id and one well_index, matching the rule wildcard
  [x] uniqueness — one row per (well_index, channel_id, time_index)
  [x] well_id (derived) recomputes to {experiment_id}_{well_index}; reject a bare local label (B01)
  [x] image_id (if present) matches the recomputed key; else derive + write it
  [x] BF present AND contiguous 0..N-1 (BF defines the segmentation timeline)
  [x] all channels share the SAME time_index set (rectangular — see Channel Sync)
  [x] every source_image_path EXISTS (resolved via image_root if relative)
  [x] every image OPENS (TIFF/PNG/JPEG) and real dims == declared (image_width_px/height_px)
  [x] source_micrometers_per_pixel present and > 0 for every row
  [~] WARN if source_image_path basename != derived image_id (recommended layout, not required)
  → on PASS: writes .validated sentinel (no report)
  → on FAIL: writes {well_id}_frame_inventory.errors.md, then RAISES (no sentinel) — fail loud
```

- **Sentinel on pass, errors report only on fail.** On success the `.validated` sentinel is the
  whole signal; on failure a human gets `{well_id}_frame_inventory.errors.md` (what broke) and the
  rule raises. No redundant success report (mdcolon's complexity rule).
- **Channel rule:** BF is **required** and contiguous (defines the timeline). Other channels are
  optional, but **if present** must be contiguous **and rectangular** — see Channel Sync.
- **`well_id` strictness is load-bearing:** segmentation keys on global `well_id`; a bare `B01` is
  rejected here (mirrors `validate_well_id` — Scope 2).
- **One experiment per submission:** discovery already raised if the drop-in file mixed
  experiments; the validator re-asserts single-experiment purity per well.

### 🔁 Channel sync — all channels same length (decided 2026-06-04, stricter)

**Every present channel must have the SAME `time_index` set** (rectangular: BF 0..N, GFP 0..N — not
BF 0..100 / GFP 0..80). This is **stricter** than "BF-only contiguous," chosen deliberately: it is
**more defensive and honest** — if the user *claims* a channel exists, a ragged time span is almost
always a mistake (a dropped or mis-numbered frame), and failing loud at the door beats a silent
misalignment deep in a multi-channel stage. *(If a future stage genuinely needs ragged channels,
relax this then — but default to honest.)*

---

## 📐 WORKED WALKTHROUGH — how an OUTSIDE USER drops in (no microscope)

The user starts **after** the fan. They never touch raw files or microscope code.

**Step 1 — lay out images (recommended layout; PNG shown).** Any resolvable location works, but
the recommended self-describing tree is:
```
.../stitched_ff_images/my_experiment_B01/BF/my_experiment_B01_BF_t0000.png
.../stitched_ff_images/my_experiment_B01/BF/my_experiment_B01_BF_t0001.png
```
- The recommended folder name uses the **derived** `well_id` (`my_experiment_B01`) as a
  self-describing label on disk — but that's just the folder name; in the **manifest** the user
  authors the atom `well_index` (`B01`) and `experiment_id`, and the pipeline composes `well_id`.
- `time_index` = 0-based frame order.

**Step 2 — author ONE dataset-level `dropin_frame_inventory.csv`** (rows for ALL wells). **Author
the atoms only** — no `well_id`, no `image_id` (the pipeline derives both):
```csv
experiment_id,well_index,channel_id,time_index,source_image_path,source_micrometers_per_pixel,image_width_px,image_height_px
my_experiment,B01,BF,0,stitched_ff_images/my_experiment_B01/BF/my_experiment_B01_BF_t0000.png,3.25,2048,2048
my_experiment,B01,BF,1,stitched_ff_images/my_experiment_B01/BF/my_experiment_B01_BF_t0001.png,3.25,2048,2048
```
(paths shown relative to an `image_root`; absolute also accepted). The only genuinely new fact the
user supplies is `source_micrometers_per_pixel`; everything else is an atom or a file coordinate.
The pipeline composes `well_id = my_experiment_B01` and `image_id` from these rows.

**Step 3 — run the pipeline.** `discover_wells_from_handoff` splits the file → `discovered_wells.txt`;
`build_frame_inventory_well` selects each well's rows → `{well_id}_frame_inventory.csv`;
`validate_frame_inventory_well` checks every file and, on pass, writes `.validated` (on failure it
writes `{well_id}_frame_inventory.errors.md` and raises). **`.validated` present → segmentation runs.**
One big file in, per-well validated shards out, fail-loud at the door.

> **Why this is reasonable:** the required core is tiny — the four frame-key **atoms**
> (`experiment_id`, `well_index`, `channel_id`, `time_index`), the image path, calibration, and
> dims. Most are file coordinates or encoded in the path/filename; `well_id`/`image_id` are
> **derived** (the user never composes them). The user authors **one scientific fact**
> (`micrometers_per_pixel`) and gets the **atoms** right. The strict gate turns "did I organize it
> correctly?" into a single pass/fail at the door instead of a deep segmentation failure.

---

## 🧱 SMALL DATACLASSES — clipboards, not a mayor

Encode the contract at the boundary with **small, frozen dataclasses that hold expectations and
locate files** — *not* a "dataset brain" that runs the pipeline.

> **A dataclass here is a clipboard, not a mayor.** It carries what-is-required and
> where-things-live. It must NOT grow `discover()`, `validate()`, `segment()`, `merge()` methods.

```python
# What the USER must author: atoms + image path + calibration + dims.
# well_id / image_id are DERIVED (composed + written by build) — NOT in this required set.
REQUIRED_FRAME_INVENTORY_COLUMNS: tuple[str, ...] = (   # a TUPLE — never a mutable default
    "experiment_id", "well_index", "channel_id", "time_index",   # the four frame-key atoms
    "source_image_path", "source_micrometers_per_pixel",
    "image_width_px", "image_height_px",
)
DERIVED_FRAME_INVENTORY_COLUMNS: tuple[str, ...] = ("well_id", "image_id")  # composed from atoms; checked-if-supplied

@dataclass(frozen=True)
class StitchedHandoffSpec:           # the dataset-level INPUT (ingress)
    experiment_id: str
    manifest_path: Path              # the dataset-level dropin_frame_inventory.csv
    image_root: Path | None = None   # resolves relative source_image_path; None ⇒ paths must be absolute

@dataclass(frozen=True)
class FrameInventorySpec:             # what a valid inventory REQUIRES
    required_columns: tuple[str, ...] = REQUIRED_FRAME_INVENTORY_COLUMNS
    allowed_image_suffixes: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    required_channel: str = "BF"

@dataclass(frozen=True)
class WellHandoff:                    # ONE well's operational unit (the important one)
    experiment_id: str
    well_id: str
    image_root: Path | None
    candidate_manifest_path: Path     # input to the gate ({well_id}_frame_inventory.csv, pre-sentinel)
    validated_frame_inventory_path: Path
    report_path: Path
    validated_sentinel_path: Path
```

- The per-row schema is **constants**, not a per-row dataclass — the validator is dataframe-centric,
  so a `FrameInventoryRow` would be instantiated per-row for no gain (keep it only as doc if useful).
- `WellHandoff` **receives** `well_id`, never mints it (minting stays in `identifiers/`). It carries
  **both** the input (`candidate_manifest_path`) and outputs, so it's the single object a per-well
  entrypoint needs.
- **Explicit signatures, no globals** (consistent with the findings-doc "no haunted globals" rule):
  ```python
  def validate_frame_inventory_well(*, well: WellHandoff, spec: FrameInventorySpec) -> None: ...
  ```

### Where this lives — a real data contract, not just orchestration

```
data_pipeline/stitched_handoff/                 # 'stitched' = the product (already-stitched images), not the act
    __init__.py
    contract.py     # REQUIRED_FRAME_INVENTORY_COLUMNS, FrameInventorySpec, derive_image_id(), normalize_source_paths()
    paths.py        # stitched_image_path(), stitched_well_done_path(), WellHandoff  (OFF-registry image paths)
    validate.py     # validate_frame_inventory_well()  — the SHARED gate
    build.py        # build_frame_inventory_well() (native: join+observe; drop-in: select-rows)
    split.py        # split_dropin_inventory_by_well(), discover_wells_from_handoff()
```

**Kingdom boundaries:**
- `identifiers/` mints/validates `well_id`, `image_id`. `stitched_handoff/` **imports** them.
- `pipeline_orchestrator/lib/paths.py` = the registry for **tabular** artifacts (the per-well
  `frame_inventory` shard path comes from there). **Off-registry image paths** live in
  `stitched_handoff/paths.py` (findings-doc "image trees are off-registry").
- `stitched_handoff/` is its own module (not in `lib/`) because *"what a valid external stitched
  dataset looks like"* is a real **data contract**, not workflow glue.

---

## ✅ DECISIONS (this doc)
1. **Stitched seam = the public drop-in point** — first microscope-agnostic artifact.
2. **Four distinct roles:** pixel tree / scope metadata / `frame_inventory` / `.validated` sentinel.
3. **Immutable frame key = four RAW ATOMS** `experiment_id + well_index + channel_id + time_index`;
   `well_id`/`image_id` are **derived compositions**, not key fields. Check/enrich/fail-loud, never
   silently reassign; the validator recomputes the derived ids from the atoms.
4. **Submission surface vs. operational spine** — one dataset-level file in, per-well shards out;
   the big file is **ingress-only**.
5. **One-file model:** one `{well_id}_frame_inventory.csv` per well; **validatedness = sentinel**,
   not a separate "candidate" file. Build and validate are **two rules** (validation stays an
   exposed DAG step); validate writes **`.validated` on pass, `.errors.md` + raise on fail** — no
   always-written success report.
6. **`well_id` in the filename** (`{well_id}_frame_inventory.csv`) — self-describing; also avoids the
   build/validate output collision.
7. **CSV all the way** through this seam (grounded: code writes/reads CSV; standard for contract
   tables). Parquet = free future swap. *Supersedes the old "parquet-internal" lean.*
8. **One shared validator** for native + drop-in; the only divergence is `build_frame_inventory_well`.
9. **Discovery is metadata-only** (both producers) → `active_wells` exists before stitching;
   run-one-well works in both paths.
10. **User authors ATOMS only** (`experiment_id`, `well_index`, `channel_id`, `time_index` + path/
    calibration/dims); **`well_id` and `image_id` are derived + written, never authored** (if a user
    supplies one, the validator recomputes from atoms and fails loud on mismatch). `well_id` is
    strict/global/per-well as a derived id.
11. **Manifest is truth; layout recommended** — images resolve via `source_image_path`
    (abs or rel to `image_root`); filename mismatch = warning.
12. **Paths absolute OR relative-to-`image_root`** (prefer relative); validator canonicalizes.
13. **One experiment per submission** (mixed experiments → raise at discovery).
14. **All channels same `time_index` set** (rectangular); BF required + contiguous.
15. **`.done` sentinel is native-stitcher only** — drop-in users author zero dotfiles.
16. **Small dataclasses** (`StitchedHandoffSpec`, `FrameInventorySpec`, `WellHandoff`) — clipboards;
    schema as a **tuple** constant; explicit signatures, no globals.
17. **New module `data_pipeline/stitched_handoff/`** (contract/paths/validate/build/split).
18. **Adopt `frame_inventory`** as the validated-table name (rename from `frame_contract`).
19. **`time_index` is canonical** — the T dimension; collapse `frame_index`+`time_int`.

## 🪧 OPEN (carried, not decided here)
- **Family name** for the inventory: dedicated `frame_inventory/` vs today's `experiment_metadata/`
  (findings #5, leaning dedicated).
- **`.validated` sentinel suffix** (leading vs trailing dot) — front-end-doc audit; normalize before
  wiring `validated_path`.
- **Native `build` richness:** observe the tree **inline** in `build_frame_inventory_well` (lean,
  default) vs. a **separate per-well observed-inventory table** (more inspectable, finer image
  staleness, but another table — the duplication mdcolon flagged). Drop-in tips lean: the user file
  *is* the observation. **Leaning inline; revisit if image-staleness granularity matters.**
- **Per-well `.done` location:** `built_image_data/{exp}/.well_{well_id}.done` (flat, today) vs
  `…/stitched_ff_images/{well_id}/.done` (self-contained). Leaning self-contained; confirm against
  sentinel-handling code first.

## 🔧 REFACTOR ITEMS (this doc surfaces)
- **Rename `frame_contract` → `frame_inventory`** (Scope-2): **37 Snakefile refs**, 3 Python modules
  (`auxiliary_masks/inference.py`, `materialize_auxiliary_masks.py`, `…/frame_contract/build_frame_contract.py`),
  `schemas/frame_contract.py`, and the `frame_contract/` module dir. Regenerate artifacts.
- **Collapse `frame_index` + `time_int` → `time_index`** across schemas + builders (Scope-2).
- **Split `build_frame_contract` (build+validate fused, experiment-grain) into per-well
  `build_frame_inventory_well` + shared `validate_frame_inventory_well`.**
- **Promote the shared validator** from `validate_dataframe_schema` (columns+nulls only) to strict
  file-level (paths exist, images open, dims match, µm/px>0, BF + rectangular channels, purity,
  recompute-derived-ids-from-atoms; `.validated` on pass, `.errors.md` + raise on fail).
- **Add `discover_wells_from_handoff`** — drop-in twin of `discover_wells_from_metadata`, reading the
  dataset-level CSV → `discovered_wells.txt`; enforce single-experiment.
- **Create `data_pipeline/stitched_handoff/`** (contract/paths/validate/build/split) with the small
  dataclasses + `split_dropin_inventory_by_well()`.

## 🔗 RELATED-DOC UPDATES NEEDED
- `data_output_structure.md`: `frame_manifest.csv`/`frame_contract` → `frame_inventory`; fix path/
  calibration column names to match code; add this drop-in contract to "Practical Flow for Scientists."
- `front_end_naming_and_flow.md`: rename `validate_frame_contract_well` →
  `validate_frame_inventory_well`; cross-link this doc from the post-fan flow as the "drop-in here"
  companion.
- `per_well_throughline_findings.md`: note `frame_contract` → `frame_inventory` vocabulary; the
  immutable-frame-key invariant; the build→validate(sentinel) per-well rhythm.
