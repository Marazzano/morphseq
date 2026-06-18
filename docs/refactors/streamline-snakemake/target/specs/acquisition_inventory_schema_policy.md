# Acquisition Inventory Schema Policy

**Status:** doctrine. Locked 2026-06-18. Governs what the per-scope acquisition inventory carries, what
the validator enforces, and how scope-divergent facts converge on shared names for downstream use.

This generalizes the same discipline the model-backend docs already state for detection
(`specs/detect-seg-track/adapter_seams.md`, `detection_world.md`): *the shared validator allows unknown
backend-specific columns but only hard-validates the shared required blocks.* The acquisition inventory
is the front-end instance of that rule.

---

## Role

The acquisition inventory is the **system of record for what each microscope ACQUIRED** — one row per
acquired coordinate, before any materialization. It should **capture everything the scope makes
available**, even fields nothing downstream consumes yet. Capturing is cheap and the inventory is the
audit trail; re-deriving lost facts later is expensive.

---

## Two tiers

Every acquisition-inventory column is one of two tiers.

### Tier 1 — SHARED, HARD-CHECKED

Columns whose **meaning is identical across scopes**, so cross-scope and downstream code can rely on
them. The acquisition validator **hard-checks ONLY these**: required-present, types, calibration `> 0`,
and cell-key uniqueness. Both the YX1 and Keyence extractors already emit them (sometimes under
different raw names, reconciled at extraction):

```text
experiment_id
position_index   (+ raw_position_label pre-mapping)
well_index
channel          (normalized token)   + raw_channel_name
time_index
elapsed_time_s   (DERIVED — see Time Atom below)
micrometers_per_pixel
image_width_px
image_height_px
microscope_id
```

### Tier 2 — SCOPE-SPECIFIC, SOFT

Whatever a given scope happens to know. Tier-2 columns are **ALLOWED** to be present (no
"unexpected column" error), but **NOT required, NOT type-checked, and NOT forced downstream**. Converge
on shared NAMES where it is natural (e.g. `objective_magnification` exists in both scopes), but tolerate
divergence where the fact is genuinely scope-native. Downstream ignores Tier-2 columns unless a stage
deliberately opts in.

**YX1** (from the nd2 `Microscope` struct + tensor axes — currently UNDER-extracted; the extractor reads
only `objectiveName` today):

```text
objective_magnification   objective_name   objective_numerical_aperture
zoom_magnification         immersion_refractive_index   pinhole_diameter_um
modality_flags
n_z   x_um   y_um   channel_index   source_nd2_path   acquisition_time_s
```

**Keyence** (from the BZ-X `meta` dict — exposure/gain/binning currently ignored):

```text
objective_magnification (Objective)   exposure_time   camera_gain   binning   lens_id
experiment_time_s   frame_interval_s   absolute_start_time   source_file
```

> **Deferred work (specified, not yet built):** fully expanding both extractors to capture the Tier-2
> optical fields above. The validator already allows unknown Tier-2 columns, so extraction can grow
> without any contract change. Keyence BZ-X TIFF-tag layout is proprietary and needs reverse-engineering
> on real files; YX1 needs the richer `nd2.frame_metadata(...).channels[i].microscope` read.

### The validator rule

```text
hard-check the Tier-1 shared block;
allow + ignore Tier-2 scope-specific columns.
```

One shared validator, scope-divergent inputs. A column graduates from Tier 2 to Tier 1 only by an
explicit decision (it must become cross-scope meaningful and get a converged name) — see the time atom.

---

## The Time Atom (worked example of a promoted Tier-2 → Tier-1 column)

Time is a **fundamental pipeline unit**, produced **differently by each scope**. It is the case that
forced this policy: the new `frame_inventory` contract initially dropped time, and
`compute_stage_predictions` needs `elapsed_time_s`.

### Per-artifact ownership

Conceptual test — *what is each artifact a record of?*

```text
acquisition_inventory  = what the microscope ACQUIRED
  → OWNS the raw time atom (scope-specific) AND DERIVES elapsed_time_s
    (rebased per-well to the well's first frame) via the shared, scope-neutral helper
    metadata_ingest/time_helpers.py::add_elapsed_time_columns

frame_inventory        = what got MATERIALIZED into a trusted frame
  → CARRIES time THROUGH (inherits; the materialize_well backend SELECTS the columns,
    it does NOT derive — the pixel stage stays pixel-focused)

downstream stages      → read ONE name: elapsed_time_s
```

This mirrors the detection doctrine exactly: scope-specific code diverges *before* the contract; the
contract is shared; downstream reads one name.

### The shared derived name and its scope-specific inputs

The converged downstream name is **`elapsed_time_s`** (already the output of `time_helpers`). The raw
input atom differs per scope, but the helper parameterizes its input column, so one helper serves all:

```text
YX1:      acquisition_time_s (per-frame ND2 timestamp, already in the YX1 acquisition inventory)
            → add_elapsed_time_columns(experiment_time_col="acquisition_time_s",
                                       group_cols=[per-well])
Keyence:  experiment_time_s / frame_interval_s
            → add_elapsed_time_columns(...)   (default input columns)
```

`add_elapsed_time_columns` rebases elapsed time to each group's first frame, so grouping per well gives
"seconds since this well's first frame" — what per-well downstream wants (t=0 → 0.0).

### What the contract carries (decision: derived + raw, auditable)

The contract requires `elapsed_time_s` (canonical, downstream-facing) **plus** the raw atom(s) it was
derived from, so the derivation is re-derivable and auditable. For the YX1 MVP that is
`elapsed_time_s` + `acquisition_time_s`. Time columns are NOT part of the per-frame unique key (time is
not identity).

---

## See also

- `specs/pipeline_file_philosophy.md` — the validator-ownership and section-banner doctrine.
- `specs/detect-seg-track/adapter_seams.md`, `detection_world.md` — the same shared-vs-backend rule on
  the model side; downstream model products import the frame-identity block from the inventory contract
  owner (`image_materialization/frame_inventory_contract.py`).
- `src/data_pipeline/metadata_ingest/time_helpers.py` — the shared time derivation (reuse; do not
  re-implement).
