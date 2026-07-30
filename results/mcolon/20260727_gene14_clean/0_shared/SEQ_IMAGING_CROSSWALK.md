# Imaging ↔ sequencing crosswalk for GENE14

> **Historical note:** This design was superseded by
> `map_morphseq_to_sequencing.ipynb`. The standalone resolver scripts were removed.

Design notes for `2_seq_imaging_crosswalk.py`. The code holds the rules; this file holds the
reasoning, the failure modes, and the history, so the mapping never has to be re-derived by hand.

## Two ID worlds (they share no id column)

| world | id | example | grammar |
|---|---|---|---|
| imaging | `embryo_id` | `20260415_cep290_18hpf_plate03_C01_e01` | `date_gene_stage_plate_well_e##` |
| imaging | `physical_embryo_id` | `cep290_18hpf_plate01_F10` | `gene_stage_plate_well` (biology-scoped) |
| sequencing | `embryo_ID` | `GENE14_P18_F10_Bl2` | `GENE14_P{hash_plate}_{hash_well}_Bl{rt_block}` |

A sequencing `embryo_ID` is **defined by** (hash plate, hash well, RT block). It is the physical
identity — "you can only destroy an embryo once" — so physical embryo ↔ sequencing embryo is 1:1.

The imaging side has redundant images of the same fish (snapshot + `_t01`/`_t02` backups +
`_sci` timeseries, up to 3 imaging `embryo_id`s). They all collapse to **one** physical embryo
and therefore **one** sequencing embryo. Several imaging rows legitimately resolving to the same
`embryo_ID` is expected, not a bug; the caller collapses them by `ACQ_PRIORITY`.

## What maps cleanly (the easy 90%)

- **Normal plates**: the imaging well *is* the hash well. The `image_to_hash_map` sheet is blank,
  so the join is identity and `hash_plate` comes from the `hash_plate_num` sheet. 15 of 16
  phenotyped experiments are this case.
- `embryo_ID` strings are identical across every table (metadata, QC list, McClintock CDS,
  File A), so joins on the id string match with zero orphans.

## The tricky things

A naive join silently mismatches roughly 30% of the interesting embryos. Each of the following
cost real debugging time.

### 1. Reformatted plates (imaging well ≠ hash well)

Embryos were physically re-pipetted from the imaging plate onto a hash plate, so the well changes
(C1→C6, A3→A9, …). Recorded per plate in the Excel `image_to_hash_map` sheet (position = imaging
well, value = hash well) plus `hash_plate_num` (value = hash plate).

Reformatted plates: crispants `260319_p1`/`p2`, `260320_p4`; cep290 `260414_plate_3` (= plate03);
b9d2 `14hpf_plate02`.

**Rule:** populated `image_to_hash_map` → use it; blank → identity. The code detects this from
the sheet contents rather than a hardcoded list, so it cannot drift.

### 2. `hash_plate_num` is populated for *both* regimes

It gives the hash plate for every embryo, normal or reformatted. It must be read for identity
plates too — it is what distinguishes P02 from P18.

*Original bug:* reading it only for reformatted plates left 15 embryos unresolved.

### 3. plate03 (cep290 18hpf) is irregular

8 rows × 5 imaging columns, with imaging **column 4 deliberately skipped**
(cols 1,2,3,5,6 → hash \*06,\*07,\*09,\*10,\*11). There is no imaging F4.

Read the sheet on its own terms. Do **not** validate plate03 against File A: File A recorded
`imaging_well=F4` (etc.) for the rescued embryos, but those File-A imaging wells are part of the
original error. The Excel is truth. (plate03 has 0 phenotype predictions anyway.)

### 4. The `embryo_ID` string is the key — do not key on (gene, timepoint)

`embryo_ID` is literally `GENE14_{hash_plate}_{hash_well}_{rt_block}`, and it is unique, so that
triple identifies a sequencing embryo exactly: no gene, no timepoint, no tie-break.

**History — why this file used to do something worse.** 11 rescued cep290 embryos had their
`hash_plate` *column* corrected P18→P02 while their `embryo_ID` *string* still said P18. The
columns then collided (`P02/F10/Bl2` and `P02/H6/Bl2` each named two distinct embryos) even
though the IDs never did. That drove a `(gene, timepoint, hash_plate, hash_well)` key plus an
`id_plate` tie-break, and a plate-less fallback that silently returned **wrong-plate embryos for
~118 imaging wells**.

The metadata was corrected upstream on 2026-07-28; columns now agree with the ID strings 546/546.
The code parses the coordinate out of `embryo_ID` anyway — the ID is the identity, so parsing is
immune to that class of column drift recurring.

Switching to the parsed-ID key took resolution from 367 to **555** imaging rows, and every
resolved row now sits on the plate that was actually imaged (was 228/367).

### 5. Collection time ≠ imaging time for "30to48" plates

plate01/plate02 of cep290 and b9d2 carry both a 30hpf and a 48hpf snapshot (`_t01`/`_t02`) plus an
`_sci` timeseries — redundant backups of the same embryos, collected across 30–48 hpf. The
sequencing `timepoint` for these is 48.

**Rule:** collection time = plate start age, except experiments containing `30to48` → 48.

`predicted_stage_hpf` stays each embryo's true age; collection time is only for structure and for
keying to the sequencing side. Without this, a `_t01` snapshot labeled 30 fails to match its
embryo, which sequencing filed under collection 48.

### 6. RT block is the third coordinate and is recorded in each plate workbook

An RT block covers one contiguous run of hash columns = one gene at one collection time. Block
does **not** depend on hash plate (cep290@18 is Bl2 on both P02 and P18).

The `rt_block` sheet was added to every live plate workbook from the bench RT plate map and
verified against the sequencing metadata. All 27 (experimental gene, timepoint) groups are
single-block:

| gene | 14hpf | 18hpf | 24hpf | 30hpf | 48hpf |
|---|---|---|---|---|---|
| cep290 | — | Bl2 | Bl3 | Bl4 | Bl5 |
| b9d2 | Bl6 | Bl7 | — | Bl8 | Bl9 |
| foxj1a / ift88 / sspo | — | Bl1 | Bl1 | Bl1 | Bl1 |

The three crispant targets were pooled into one block, so they share Bl1 at every timepoint.

Controls (WT sibling + AB) ride in **each experiment's own block**, which is why a control can
resolve from a cep290 or b9d2 imaging plate. That is correct, not a mismatch — `target=Control`
erases which experiment the control came from, and the `strain` column recovers it
(`unknown` = WT sibling, `AB` = AB control).

## The resolution key, put together

```
imaging (experiment, well)
  --image_to_hash_map, or identity when blank-->  hash_well
  --hash_plate_num-->                             hash_plate
  --rt_block------------------------------------>  rt_block

HashCoordinate(hash_plate, hash_well, rt_block)  ->  embryo_ID
```

The coordinate *is* what `embryo_ID` is made of, so a hit is exact and a miss is an honest `None`.

## Provenance

Copied verbatim from
`results/mcolon/20260715_gene14_hooke_dact/scripts/core/seq_imaging_crosswalk.py` on 2026-07-27 so
this folder is self-contained and the archive can be retired. Rekeyed onto the parsed `embryo_ID`
on 2026-07-28.
