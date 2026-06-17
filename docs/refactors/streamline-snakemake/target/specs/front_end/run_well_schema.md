# Run Well Schema — discovered vs eligible vs runnable (🟢 TARGET)

**Status:** companion contract/test note, mdcolon 2026-06-17. This doc defines the
well-selection state model that sits between `discover_wells`, acquisition resolution, and per-well
execution. It also names the canonical Keyence reacquisition test case.

**Companion to:** `front_half_reorg_roadmap.md`, `acquisition_inventory_flow.md`,
`frame_inventory_handoff_contract.md`, and `well_id_throughline_refactor_plan.md`.

---

## The Three Well States

The pipeline must keep three ideas separate:

```text
discovered_wells
  wells that physically exist in validated canonical metadata

eligible_wells
  discovered wells that have a resolved active acquisition source for stitch

run_wells
  wells requested for this run after target filtering and eligibility filtering
```

The target selection equation is:

```text
run_wells = discovered_wells ∩ target_wells ∩ eligible_wells
```

or, operationally:

```text
run_wells = discovered_wells ∩ target_wells ∩ {well_id : active_for_stitch == true}
```

`discovered_wells` is not a quality-control or acquisition-resolution artifact. It answers only:

```text
does this physical well identity exist in canonical metadata?
```

`eligible_wells` answers a later question:

```text
does this discovered well have exactly one resolved active acquisition source for stitch?
```

---

## Mapping Validity Is Not Eligibility

Duplicate handling depends on which layer produced the duplicate.

### Mapping duplicate

A mapping duplicate means raw positions cannot be assigned to wells unambiguously.

Example:

```text
two YX1 stage positions both map to B03
```

That is a structural mapping failure. It should fail before `scope_metadata_mapped.csv` and before
`discover_wells`.

```text
mapping uniqueness
  = structural validity before discovery
```

### Acquisition duplicate

An acquisition duplicate means the well identity is known, but there is more than one candidate raw
image source for the same acquisition cell.

Example:

```text
Keyence well B03 was re-acquired into the same claimed grid cells
```

That is not an undiscovered well. It is a discovered well with unresolved acquisition evidence.

```text
acquisition eligibility
  = resolved active acquisition before stitch
```

So the state is:

```text
discovered = yes
eligible   = no
run        = no
```

---

## Keyence Reacquisition Handling

For Keyence, `eligible_wells` earns its existence because a re-acquired well is not a mapping error
and not an undiscovered well.

The canonical test case is:

```text
20260414_b9d2_14hpf_plate02_B03
```

This well was re-acquired: it has 6 tiles where the modal/expected count is 3. That should become
the test case for Keyence acquisition eligibility.

### 1. Ingest Records All Raw Evidence

`ingest_scope_metadata` writes every raw Keyence TIFF plane into:

```text
acquisition_inventory__keyence.csv
```

This artifact is evidence only. It does not filter, guess, collapse, or choose between acquisitions.

### 2. Keyence Checks Raw Acquisition-Cell Uniqueness

The raw Keyence acquisition-cell key is:

```text
well_id
position_index
z_index
channel_index
time_index_claimed
```

A normal Z stack produces many rows because `z_index` differs. That is valid.

A reacquisition conflict occurs when multiple rows have the same full cell key. That means the
pipeline has more than one candidate image for the same well, position, Z plane, channel, and
claimed timepoint.

### 3. `resolve_acquisitions` Classifies The Conflict

`resolve_acquisitions` decides what kind of duplication occurred:

```text
exact duplicate rows        -> collapse
byte-identical TIFFs        -> collapse as true duplicate
different bytes             -> real reacquisition conflict
partial/non-rectangular set -> malformed acquisition conflict
```

It does not pretend the well is missing. It explains why the well is not currently stitchable.

### 4. Expected B03 Outputs

`discovered_wells.txt` should include B03:

```text
20260414_b9d2_14hpf_plate02_B03
```

because the well physically exists.

`acquisition_conflicts__keyence.csv` should record the duplicated raw acquisition cells and
candidate acquisition groups.

`well_acquisition_summary__keyence.csv` should mark B03 inactive:

```text
well_id: 20260414_b9d2_14hpf_plate02_B03
active_for_stitch: false
quarantine_reason: unresolved duplicate grid-cell acquisition / reacquisition conflict
```

`resolved_acquisition_inventory__keyence.csv` should exclude B03 until a valid resolution is chosen.

### 5. The Well Runner Skips B03 But Runs The Rest

The runnable set is:

```text
run_wells = discovered_wells ∩ target_wells ∩ eligible_wells
```

So B03 remains visible, explained, and skipped. Other clean wells still run.

---

## Keyence Test Plan

Use `20260414_b9d2_14hpf_plate02_B03` as the load-bearing reacquisition test.

### Inventory Test

Build or fixture `acquisition_inventory__keyence.csv` for the experiment.

Assertions:

```text
all raw TIFF planes are present
B03 has duplicated acquisition-cell keys
clean wells do not have duplicated acquisition-cell keys
a normal Z stack is not classified as a duplicate because z_index differs
```

### Conflict Detection Test

Run the Keyence conflict detector over the inventory.

Assertions:

```text
acquisition_conflicts__keyence.csv contains B03
the duplicated key is the full acquisition-cell key
candidate acquisition groups are named
the conflict records enough evidence to explain 6 observed tiles vs expected/modal 3
```

### Resolution Test

Run `resolve_acquisitions` with no manual resolution sidecar.

Assertions:

```text
well_acquisition_summary__keyence.csv has one row for B03
B03 active_for_stitch == false
B03 quarantine_reason is non-empty and names reacquisition / duplicate grid-cell conflict
resolved_acquisition_inventory__keyence.csv has no active rows for B03
clean wells are active_for_stitch == true
```

### Run-Well Selection Test

Run the well selector with:

```text
discovered_wells.txt includes B03
target_wells includes B03 and at least one clean well
well_acquisition_summary__keyence.csv marks B03 inactive
```

Assertions:

```text
B03 is absent from run_wells
the clean target well is present in run_wells
an inactive target well produces an explicit skip/quarantine reason, not silent disappearance
```

### Stitch Scheduling Test

Build the per-well stitch DAG for the experiment.

Assertions:

```text
stitch_well[B03] is not requested by the normal run_wells fan
stitch_well[clean_well] is requested
if stitch_well[B03] is forced manually, it fails loud because no resolved active acquisition rows exist
```

### Manual Resolution Test

Provide an explicit `acquisition_resolution__keyence.csv` selecting one acquisition group for B03.

Assertions:

```text
B03 active_for_stitch flips to true
resolved_acquisition_inventory__keyence.csv contains only the selected acquisition group
run_wells includes B03 when B03 is targeted
stitch_well[B03] consumes the resolved inventory, not the raw/conflicted inventory
```

---

## Implementation Implications

- `discover_wells` remains microscope-agnostic and emits physical well identities only.
- `validate_physical_well_mapping` enforces structural mapping uniqueness before discovery.
- Keyence acquisition conflict detection happens after raw inventory exists, not inside discovery.
- `well_acquisition_summary__{scope}.csv` is the shared eligibility contract.
- Shared orchestration reads only `well_id`, `active_for_stitch`, and `quarantine_reason`.
- Keyence-specific stitch code consumes `resolved_acquisition_inventory__keyence.csv`.
- No target path may use `drop_duplicates(..., keep="first")` to resolve an acquisition conflict.

The final rule:

```text
YX1 duplicate well mapping
  -> invalid mapping; fail before discovery

Keyence duplicate raw acquisition for the same well
  -> discovered but ineligible until resolved
```
