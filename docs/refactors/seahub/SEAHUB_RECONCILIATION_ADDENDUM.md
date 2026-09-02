# SeaHub reconciliation addendum

## Scope

Reconciliation remediation is limited to records currently classified as
`unmatched_stage` or `unmatched_condition`. Existing path/role exclusions and
files explicitly marked `not collected`, `not sequenced`, `abandoned`, or
`not used` remain excluded. No source metadata is edited in place.

## Canonical stage crosswalk

Retain every original token in `stage_source_label`; write the normalized age
to `stage_hpf` and record how it was obtained. The owner-supplied crosswalk is
normative:

- Cleavage: `1-cell=0.00`, `2-cell=0.75`, `4-cell=1.00`, `8-cell=1.25`,
  `16-cell=1.50`, `32-cell=1.75`, and `64-cell=2.00` hpf.
- Blastula: `128-cell=2.25`, `256-cell=2.50`, `512-cell=2.75`,
  `1k-cell=3.00`, `high=3.33`, `oblong=3.67`, `sphere=4.00`, `dome=4.33`,
  and `30%-epiboly=4.67` hpf.
- Gastrula: `50%-epiboly=5.25`, `germ-ring=5.67`, `shield=6.00`,
  `75%-epiboly=8.00`, `90%-epiboly=9.00`, and `bud=10.00` hpf.
- Segmentation/pharyngula: `1-somite=10.33`, `5-somite=11.67`,
  `8-somite≈13.0`, `10-somite=14.0`, **`12-somite=15.0`**,
  `14-somite=16.0`, `18-somite=18.0`, `20-somite=19.0`,
  `26-somite=22.0`, and `prim-5=24.0` hpf.

The `12-somite=15.0 hpf` entry is a linear interpolation between the supplied
10-somite and 14-somite anchors.

Accept common aliases case-insensitively, including `12s` → `12-somite`,
`18s` → `18-somite`, and `Shield` → `shield`.

For reconciliation:

- Direct hpf matches are `stage_exact`.
- Matches made through the morphological crosswalk are `stage_crosswalk`.
- Preserve `stage_source_label`, normalized `stage_hpf`, the crosswalk label,
  and `stage_match_delta_hpf`.
- Any nearest-hpf tolerance must be explicit and configured rather than hidden
  in string-normalization logic.

## Stage-field precedence

Use the structured metadata columns as authoritative:

1. `stage_collected` for collection/imaging age.
2. `stage_addition` for perturbation-addition age.
3. Stages parsed from `collection_name` only as validation or fallback.

Record disagreements rather than discarding the row. This avoids the CHEM15
failure in which `stage_addition=shield` is valid but the collection name
encodes the same timing as `6hpf-treatment`.

## Experiment-field correction

When `collection_name` begins with an unambiguous `GENE#` or `CHEM#` token that
disagrees with metadata `expt`, use the collection-name prefix for matching
while retaining the original value. Emit:

- `metadata_experiment_original`
- `metadata_experiment_effective`
- `metadata_experiment_corrected` (boolean)
- `metadata_experiment_correction_source="collection_name_prefix"`

All corrected matches remain visibly flagged in downstream manifests. The raw
workbook is never modified.

## Audit contract

Every included row must expose `metadata_match_status`, `stage_match_method`,
the raw and normalized stage values, and any experiment-correction flag.
Every excluded FOV must appear in `dropped_fovs.csv` with an explicit reason;
there are no silent drops.
