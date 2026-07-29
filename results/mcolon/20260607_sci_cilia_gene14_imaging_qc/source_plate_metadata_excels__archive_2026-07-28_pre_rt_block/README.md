# Archive: plate metadata Excels as of 2026-07-28, before the rt_block migration

Byte-for-byte copies of `../source_plate_metadata_excels/` taken immediately before the
migration described below. Kept so the migration is reversible and so the original
formula-driven sheets remain inspectable.

**Do not read these for analysis** — they are superseded. The live files are one directory up.

## What the migration changed

1. **Added an `rt_block` sheet** to all 25 workbooks, in the same grid format as
   `hash_plate_num` (row 0 = column numbers, column 0 = row letters, cells = the RT block for
   that imaging well).

   With this, the three sheets together carry everything needed to build a sequencing
   `embryo_ID` = `GENE14_{hash_plate}_{hash_well}_{rt_block}`, all keyed by imaging well:

   | sheet | gives |
   |---|---|
   | `hash_plate_num` | hash plate (and which wells the plate placed) |
   | `image_to_hash_map` | hash well (only for wells that moved) |
   | `rt_block` | RT block |

   No gene or timepoint lookup is needed to resolve an embryo.

2. **Baked `image_to_hash_map` from formulas to static values.** These cells were Excel
   formulas like `=$A2 & TEXT(B$1+5, "00")`, whose results existed only as Excel's cached
   values. openpyxl cannot evaluate formulas, so *any* save through it dropped those caches and
   the sheet read back blank — silently turning `A9` into `A3` and reformatted plates into
   identity plates. Since adding a sheet requires a save, baking was not optional.

   Values came from Excel's own cache where present (184 cells), and otherwise from evaluating
   the formula grammar directly. That evaluator was validated at 184/184 against the real
   cached values before being trusted for the rest.

## Verification at migration time

- `rt_block` present in all 25 files; populated on 1972/1972 crosswalk rows
- all 911 pre-existing `image_to_hash_map` values preserved exactly; none lost or altered
- 0 formulas remaining in any workbook
- no sheets dropped from any file
- plate03 reformatting intact (`A3 → A09`, `C1 → C06`)
- `20260324_cep290_18hpf_24hpf_plate02` — the one plate holding two collection events — split
  per well into 49 × Bl2 (18hpf) and 47 × Bl3 (24hpf); each imaging well belongs to exactly one
  event, so a single grid suffices
- imaging→sequencing resolution: 589 rows from the Excel sheets, matching the independent
  `RT_BLOCK_BY_GENE_AND_TIMEPOINT` code table exactly (0 disagreements)

## Related

Consumer and full design notes: `results/mcolon/20260727_gene14_clean/0_shared/`
(`2_seq_imaging_crosswalk.py`, `SEQ_IMAGING_CROSSWALK.md`).
