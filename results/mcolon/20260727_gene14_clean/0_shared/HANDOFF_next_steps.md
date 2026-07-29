# HANDOFF — GENE14 `0_shared` pipeline

Written 2026-07-28. Read this before touching anything in `0_shared/`.

The identity plumbing (imaging well → sequencing `embryo_ID`) now works and is committed. What
remains is **one real data bug** and **a reorganization** of the script layout.

---

## 1. The thing that was fixed (don't undo it)

An imaging well now resolves to a sequencing embryo using **only the plate metadata Excel**. No
gene lookup, no timepoint lookup, no tie-break.

```
imaging (experiment, well)
    --hash_plate_num-------->  hash_plate     ALSO decides which wells the plate placed
    --image_to_hash_map----->  hash_well      only for wells that physically moved
    --rt_block-------------->  rt_block       NEW sheet, added 2026-07-28
    =>  embryo_ID = GENE14_{hash_plate}_{hash_well}_{rt_block}
```

That triple **is** what `embryo_ID` is made of, and `embryo_ID` is unique (546/546), so a hit is
exact. Commit `df91bc6b` added the `rt_block` sheet to all 25 Excels and rekeyed the resolver.
Resolution went **367 → 589** imaging rows, and every resolved row now sits on the plate that was
actually imaged (was 228/367 — the old key returned wrong-plate embryos for ~118 wells).

Two invariants worth not relearning the hard way:

- **Parse `embryo_ID`; do not read the `hash_plate`/`hash_well`/`rt_block` columns** of
  `GENE14_embryo_metadata.tsv`. Those columns have carried errors the ID string did not (11
  cep290 embryos had `hash_plate` corrected P18→P02 while their ID still said P18, collapsing two
  distinct embryos onto one coordinate). The upstream fix landed 2026-07-28 and the columns now
  agree 546/546, but we keep parsing the ID because it is the identity. There is a dated
  `TEMPORARY` note at the top of `2_seq_imaging_crosswalk.py` to remove once you trust the columns.
- **`image_to_hash_map` cells were Excel formulas.** openpyxl cannot evaluate formulas, so any
  save through it dropped their cached values and the sheet read back blank — silently turning
  reformatted plates into identity plates. They are now baked to static values (0 formulas remain),
  so the files are safe to write to. Originals are archived at
  `../../20260607_sci_cilia_gene14_imaging_qc/source_plate_metadata_excels__archive_2026-07-28_pre_rt_block/`.

Full background: `SEQ_IMAGING_CROSSWALK.md` next to this file.

---

## 2. RESOLVED — crispant plate-map formula loss and plate correction

The apparent A4/A7 collision was one symptom of two problems:

- The RT-block migration evaluator handled `&`/`TEXT(...)` formulas but missed 38
  `CONCATENATE(...)` formulas in the crispant 24/48 hpf workbooks, baking them blank.
- The archived crispant 48 hpf workbook already assigned imaging columns 4-7 to the wrong hash
  plate (P18 rather than P04).

The live workbooks now encode crispant 48 hpf columns 1-3 as P18/10-12 and columns 4-7 as
P04/1-4. The missing crispant 24 hpf control mappings and the cep290 30 hpf plate01 H row were
also restored. The live static maps match all 1,710 archived formulas, and the builder now checks
coordinate uniqueness plus MorphSeq/sequencing collection-time agreement.

---

## 3. Reorganization to do

Current layout, after renumbering:

| script | does | note |
|---|---|---|
| `0_load_mcclintock.R` | load mcclintock CDS / coldata | **merge these two** |
| `1_cell_type_lineage.R` | cell-type lineage table | **merge these two** |
| `2_seq_imaging_crosswalk.py` | the resolver library (imported, not run) | fine as-is |
| `3_build_embryo_id_map.py` | writes `embryo_id_map.csv` | **blocked by §2** |
| `4_attach_morphseq_labels.py` | joins phenotype labels | was `3_`, renumbered |
| `5_celltype_gate.py` | cell-type gating | was `4_`, renumbered |

### 3a. Combine 0 + 1 into one "load sequencing data" script

They are both "get the sequencing side into memory and write a table". One script, one output
contract. Suggested: `0_load_sequencing.R`, producing the embryo table and the cell-type lineage
table together. After that step the only sequencing identity in play is `seq_embryo_ID`.

### 3b. Naming — be explicit about which world an ID belongs to

The single biggest source of confusion in this work was two different things both called
"embryo id". Fix the names everywhere:

| current | use instead | why |
|---|---|---|
| `embryo_id` (imaging) | `morphseq_embryo_id` | e.g. `20260415_cep290_18hpf_plate03_C01_e01` |
| `embryo_ID` (sequencing) | `seq_embryo_ID` | e.g. `GENE14_P18_F10_Bl2` |
| `physical_embryo_id` | keep | e.g. `cep290_18hpf_plate01_F10` |

The capital-vs-lowercase `embryo_id` / `embryo_ID` distinction is *not* a safe way to tell them
apart. Rename at the boundaries where each is read.

### 3c. The mapping is its own step, and it comes before labels

The chain to make explicit in script order:

```
morphseq_embryo_id  ->  (hash_plate, hash_well, rt_block)  ->  seq_embryo_ID
```

`3_build_embryo_id_map.py` is that step and now runs before label attachment (it was written as
`5_` first, which was wrong — labels were joining before the mapping existed).

**Also note:** `imaging_to_seq_crosswalk.tsv` (168 rows) is written by the label script and mixes
the mapping with phenotype columns. It is a *filtered, label-joined* artifact, not the mapping.
Once `embryo_id_map.csv` exists, downstream code should join that instead, and the 168-row file
should either be dropped or clearly renamed as a label output.

---

## 4. `embryo_id_map.csv` — the intended output

One row per `(experiment, imaging_well)` — the plate position, not the image. A well imaged
several times (`_sci` timeseries + `_t01`/`_t02` snapshots) is ONE row, because those are the same
physical embryo.

| column | meaning |
|---|---|
| `experiment` | imaging experiment / which plate Excel this came from |
| `imaging_well` | well on the imaging plate |
| `hash_plate` | from `hash_plate_num` |
| `hash_well` | from `image_to_hash_map` (or unchanged if the well did not move) |
| `rt_block` | from `rt_block` |
| `seq_embryo_ID` | `GENE14_{hash_plate}_{hash_well}_{rt_block}`, if sequenced |
| `physical_embryo_id` | imaging-side physical identity, when the labels table has one |
| `provenance` | how `seq_embryo_ID` was obtained, or why it is blank |

`provenance` values: `excel_coordinate` (resolved), `not_in_seq_metadata` (coordinate complete but
no such sequencing embryo — expected for wells that were imaged but not sequenced),
`incomplete_coordinate` (the Excel is missing a part).

Current numbers at the physical-embryo registry grain:

```
527 rows across 21 experiments
516  excel_coordinate      (516 distinct seq embryos)
 11  not_in_seq_metadata
  0  incomplete_coordinate
```

The 11 complete coordinates without a sequencing metadata record are retained explicitly rather
than assigned a guessed ID.

---

## 5. State of the tree

- Committed: `df91bc6b` — Excel migration + resolver rekey + archive of originals.
- **Uncommitted**: the renumbering (`3_`→`4_`, `4_`→`5_`), the new `3_build_embryo_id_map.py`,
  and edits to `2_seq_imaging_crosswalk.py` (the commented-out `RT_BLOCK_BY_GENE_AND_TIMEPOINT`
  table and removal of `rt_block_for`, now that the block comes from the Excel).
- Branch is `chore/remove-committed-venvs`, which is **not** the right home for this work. Move to
  a proper branch before opening a PR.

Environment: `conda run -n segmentation_grounded_sam --no-capture-output python ...` — never bare
`python`.
