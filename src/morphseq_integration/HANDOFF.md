# HANDOFF — morphseq ↔ sequencing integration

Written 2026-07-30. Context for a second agent picking up this work.

## 0. Mission

Build `src/morphseq_integration/` (peer package to `data_pipeline`) that joins **morphology** and
**sci-PLEX sequencing** at embryo resolution, so transcriptional state can be read as a function of
morphology and vice versa. It replaces a fragmented 2025-Q1 notebook chain whose paradigm was
`results/nlammers/20250303/generate_morphseq_dataset.ipynb`.

**Immediate goal: integrate the GENE7 (20250612) dataset into the full morph-seq set.** Nick will
direct the work from there.

**Use the LEGACY embeddings, not the current pipeline's.** A live investigation (see §7) found the
current pipeline renders embryos ~30% smaller in area and ~4x more saturated than the version that
produced the legacy latents. Track 1 (analysis) proceeds on legacy data; track 2 (old/new QC
comparison) is separate work Nick is directing elsewhere. The saturation root cause is **still open
and explicitly on hold** — do not spend time on it.

## 1. Environment

Only `points-ml` has the full set (pandas + pyarrow + matplotlib + skimage + nbclient):

```bash
PYTHONPATH=/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src:. \
  /net/trapnell/vol1/home/nlammers/micromamba/envs/points-ml/bin/python <script>
```

`morphseq-env` has no pyarrow (cannot read analysis_ready parquet). `vae-env-cluster` has pyarrow
but no nbclient. Do not use bare `python`. Tests: the repo's `tests/conftest.py` already puts `src/`
on the path, so `tests/morphseq_integration/` will just work.

## 2. Legacy embeddings — VERIFIED PRESENT

`/net/trapnell/vol1/home/nlammers/projects/data/morphseq/legacy/20241107_ds_sweep01_optimum/`
**38 files**, named `morph_latents_{experiment_id}.csv`. (Nick's message said `moprhseq` — typo in
the message only; the path on disk is correct.)

All six GENE7 plates present:

| file | rows | analysis_ready exists |
|---|---|---|
| `morph_latents_20250612_24hpf_ctrl_atf6.csv` | 95 | yes |
| `morph_latents_20250612_24hpf_wfs1_ctcf.csv` | 93 | yes |
| `morph_latents_20250612_30hpf_ctrl_atf6.csv` | 93 | yes |
| `morph_latents_20250612_30hpf_wfs1_ctcf.csv` | 95 | yes |
| `morph_latents_20250612_36hpf_ctrl_atf6.csv` | 95 | yes |
| `morph_latents_20250612_36hpf_wfs1_ctcf.csv` | 96 | yes |

Schema (203 columns): `experiment_date`, `embryo_id`, `snip_id`, + 200 latent columns.
**No other metadata.**

### 2a. Compatibility issues — read all of these before writing the parser

1. **Embryo index is off by one.** Legacy `..._A01_e00_t0000`; new `..._A01_e01_BF_t0000`. Legacy
   only ever contains `e00` — one embryo per well, always. Do not assume `e00 == e01`; **join on
   `well_id`, never by string-patching the index.**

2. **No channel token in legacy IDs.** Legacy has no `BF` segment. Any regex must handle both
   grammars.

3. **Latent naming differs, and the mapping is UNVERIFIED.**
   - Legacy: `z_mu_n_00–19` (20 nuisance) + `z_mu_b_00–79` (80 biological), plus matching
     `z_sigma_n_*` / `z_sigma_b_*`.
   - New `analysis_ready`: flat `z_mu_00–99` + `z_sigma_00–99`.
   - **Both are 100 dims and both report `embedding_model_name = 20241107_ds_sweep01_optimum` — the
     same checkpoint.** The obvious guess is `z_mu_00–19 <-> z_mu_n_*` and
     `z_mu_20–99 <-> z_mu_b_*`, but **this is a guess. Verify before relying on it** (see §6.3).
     `data_pipeline/analysis_ready/contract.py` already documents both families and provides
     `select_latent_columns(..., prefer_biological=True)` — reuse it rather than re-deriving. The old
     notebooks fit PCA on `z_mu_b` only.

4. **Latent CSVs carry no metadata.** The companion is
   `models/legacy/20241107_ds_sweep01_optimum/embryo_stats_df.csv` — has `snip_id`, `embryo_id`,
   `experiment_date`, `experiment_time`, `temperature`, `medium`, `short_pert_name`, `control_flag`,
   `phenotype`, `predicted_stage_hpf`, `surface_area_um`, `length_um`, `width_um`, `train_cat`,
   `recon_mse`, and UMAP coords. This is what the old notebook merged as `morph_df`.

5. **6 of 38 experiments have no `analysis_ready` counterpart**: `20250215` (7217 rows),
   `20250622_chem_35C_T00_1223_check`, `20250622_chem_35C_T01_1605`, `20250623_chem_28C_T02_1259`,
   `20250623_chem_34C_T02_1231`, `20250623_chem_35C_T02_1204`. Legacy covers experiments the new
   pipeline has not processed — the parser must not require a pipeline match.

6. **Do not assume a single timepoint.** `20250215` has 7217 rows for one experiment — almost
   certainly a timelapse. The GENE7 plates are single-frame (`t0000` only), but the parser must
   handle `_tNNNN` generally. *Unverified — confirm before generalizing.*

7. **Row counts differ from the new pipeline, for good reasons.** Example
   `20250612_30hpf_ctrl_atf6`: legacy 93 wells, new 95 wells, 93 shared. The two new-only wells are
   **C06 and C07** — exactly two of the nine whole-frame mask blowouts found in the QC audit (area
   5–7x the upper cut, whole-FOV masks, all co-flagged out-of-focus). **The legacy pipeline
   correctly excluded them; the new one did not.** Also, new resolves 2 embryos in wells `E09` and
   `F12` where legacy has 1. So legacy = 1 row/well; new = 1+ rows/well.

## 3. What I built — keep this, extend it

`src/morphseq_integration/` — **7 files, ~714 lines, WRITTEN BUT NEVER EXECUTED, zero tests.**
Untracked in git. Nothing outside this directory was touched.

| file | role |
|---|---|
| `identifiers.py` | Seq sample-id grammar. `normalize_hash_well` (`A1`->`A01`), `to_sequencing_well` (`A01`->`A1`), `format_hash_plate` (`18`->`P18`), `build_seq_sample_id`, `strip_rt_block` (drops `_Bl1`), `is_blank` |
| `hash_map.py` | Per-well hash map for one experiment. Prefers `plate_metadata.csv`, falls back to the workbook via the pipeline's own `load_plate_metadata_pages`. `experiments_with_hash_map()` lists paired experiments |
| `crosswalk.py` | `build_crosswalk`, `validate_crosswalk` (1-to-1 both directions), `summarize` |
| `experiment_key.py` | Loads/validates the curated key; fails loud on dupes and blanks |
| `experiment_sequencing_key.csv` | The curated key — 6 GENE7 rows seeded |
| `paths.py` | Pipeline root: arg -> `$MORPHSEQ_PIPELINE_ROOT` -> cluster default |
| `__init__.py` | Re-exports |

Output shape of `build_crosswalk()` — one row per imaging well:

```
experiment_id  well_id  well_index  sci_expt  hash_plate  hash_well  seq_sample_id  pairing_status  hash_map_source
```

`pairing_status` is one of `paired`, `blank_well`, `no_hash_map`. The crosswalk is **identity
only** — no latents, no stages, no QC. Morphology joins on `well_id`; sequencing joins on
`seq_sample_id`.

### Design decisions Nick locked in — do not revisit

- **The crosswalk reads ONLY morphseq's own metadata.** Do *not* depend on
  `seahub/metadata/collection_metadata.xlsx` (explicitly rejected) or the lost
  `metadata/experiment_metadata.csv` (gone from the cluster; not in git history).
- **One curated fact per paired experiment: `experiment_id -> sci_expt`**, in
  `experiment_sequencing_key.csv`, living in `src/` (his choice, not the repo `metadata/` dir).
  Everything else derives — `image_to_hash_map` and `hash_plate_num` are 8x12 sheets already in the
  plate workbooks, and the pipeline's generic plate loader carries them through untouched.
- **Join key is `well_id`, not `snip_id`.** A hash well identifies a *well*; the pipeline resolves
  multiple embryos per well. The old notebook hardcoded `_e00_t0000` and silently hid that
  ambiguity.
- **Keep it simple and move fast.** `paths.py` and the `ExperimentKey` dataclass are more
  scaffolding than he wanted — first thing to cut if it is in the way.

### Known rough edges in what I wrote

- `is_blank()` in `identifiers.py` is tangled; should collapse to
  `pd.isna(v) or (isinstance(v, str) and not v.strip())`.
- `summarize()` uses `pivot_table`, so status columns are absent when a status has no rows — the
  "did it work" view can come back missing `paired`.
- The hash map currently resolves via the **workbook fallback**, not `plate_metadata.csv`, because
  the acquisition CSVs are from 2026-07-26 and predate Nick's 2026-07-29 sheet edits.
  `hash_map_source` records which was used. A pipeline re-run of the six experiments would flip that
  path.
- `build_crosswalk()` has never run. Expect breakage.

## 4. What to add — backward-compatible legacy parser

New module, e.g. `morphseq_integration/legacy_morph.py`. Requirements:

1. **Load one or many `morph_latents_*.csv`**, deriving `experiment_id` from the filename (not from
   `experiment_date`, which happens to match but is a weaker contract).
2. **Parse both ID grammars into a canonical spine.** Emit `experiment_id`, `well_id`, `well_index`,
   `local_embryo_index`, `time_index`, plus the original `snip_id` verbatim for provenance. Mint
   `well_id` with `data_pipeline.shared.identifiers.build_well_id` — never roll it inline.
3. **Normalize latent names to one convention.** Recommend keeping the legacy `z_mu_b_*` /
   `z_mu_n_*` split as canonical (it carries more information) and providing an adapter for flat
   names, using `select_latent_columns` from the analysis_ready contract.
4. **Join the metadata companion** (`embryo_stats_df.csv`) on `snip_id`, with an indicator so
   missing metadata is visible rather than silently NaN.
5. **`source` column** on every row (`legacy` vs `pipeline`) so a merged frame can always be split
   back apart.
6. **Fail loud** on: unparseable `snip_id`, duplicate `(well_id, local_embryo_index, time_index)`,
   latent column count != 100.

Then a small `assemble.py` that produces the master table: identity spine + morph latents + seq
latents + stages + `pairing_status`, written as parquet.

## 5. GENE7 pairing — verified from both sides, use it

**20250612 = GENE7**, collected 2025-06-12: atf6 / wfs1a;wfs1b / ctcf crispants x 24/28.5/34/35 C x
24/30/36 hpf.

| imaging experiment | `hash_plate_num` | seq `hash_plate` | timepoint | targets |
|---|---|---|---|---|
| 20250612_24hpf_ctrl_atf6 | 1 | P01 | 24 | Control, atf6 |
| 20250612_24hpf_wfs1_ctcf | 2 | P02 | 24 | wfs1a;wfs1b, ctcf |
| 20250612_30hpf_ctrl_atf6 | 18 | P18 | 30 | Control, atf6 |
| 20250612_30hpf_wfs1_ctcf | 4 | P04 | 30 | wfs1a;wfs1b, ctcf |
| 20250612_36hpf_ctrl_atf6 | 5 | P05 | 36 | Control, atf6 |
| 20250612_36hpf_wfs1_ctcf | 6 | P06 | 36 | wfs1a;wfs1b, ctcf |

Seq side: `/net/seahub_zfish/vol1/data/preprocessed/GENE7/GENE7_embryo_metadata.tsv` — 672 embryos,
of which **576 have `imaging = True`, exactly 6 x 96.** Every hash well has an imaging counterpart.
The projected CDS is at `annotated/v2.2.1/GENE7/GENE7_projected_cds_v2.2.1`.

Note `embryo_ID` there is `GENE7_P18_A1_Bl1` (RT-block suffix), while the sample key is
`GENE7_P18_A1`. `strip_rt_block` handles it.

## 6. Validation gates — must pass before trusting anything

1. **Seq set equality (the one that matters).** Mint sample IDs for all 576 paired wells; strip RT
   blocks from `GENE7_embryo_metadata.tsv`; assert the 576 minted IDs **exactly equal** the 576
   `imaging=True` embryos. Exact equality both directions. Until this passes, the sample-id grammar
   is plausible, not correct.
2. **Crosswalk is 1-to-1.** No duplicate `well_id`, no duplicate `seq_sample_id` among `paired`
   rows. `validate_crosswalk` does this — make sure it is exercised.
3. **Latent ordering.** Establish the flat <-> `n`/`b` mapping empirically before mixing legacy and
   pipeline latents. Suggested test: for the 93 shared wells of `20250612_30hpf_ctrl_atf6`,
   correlate each new `z_mu_ii` against each legacy `z_mu_n/b_jj` and check the permutation is
   clean. Note the images differ (scale + saturation), so expect strong-but-imperfect correlation;
   you are recovering *ordering*, not equality.
4. **Legacy/new well reconciliation.** Confirm legacy-only and new-only wells are explainable (for
   `30hpf_ctrl_atf6` the new-only wells C06/C07 are known mask blowouts). Do not silently
   inner-join the discrepancy away.

## 7. Context you will need but should not re-derive

- **`experiment_metadata.csv` is gone** from the cluster and from git history. Its only
  irreplaceable column was `sci_experiment`. Everything else it held is superseded: `temperature` is
  now a per-well 8x12 sheet (the GENE7 plates run 4 temperatures *within* one plate, which the old
  scalar could not represent), `microscope` -> `raw_image_data/{Keyence,YX1}`, `use_flag` ->
  orchestrator manifests, `has_sci_data` is equivalent to "workbook has an `image_to_hash_map`".
- **~56 imaging experiments carry a hash map** (`experiments_with_hash_map()`); only the 6 GENE7
  rows are keyed. The rest need Nick's curation.
- **QC: `sa_outlier_flag` is over-firing on these plates.** 96 of 106 failures; 87 "too small"; 75
  of those within 20% of the cut. Root cause is a reference-population mismatch
  (`k_lower = 0.9 x p5` against a pooled ~28.5 C wildtype curve; cold-reared embryos are genuinely
  smaller). Nick's chosen interim policy: **ignore `sa_outlier_flag` when it fires alone** —
  recovers 76 snips, 81.9% -> 94.9%, and all 9 genuine whole-frame blowouts stay rejected because
  they co-flag `focus_flag`. Implement as `use_snip | (qc_fail_reasons == "sa_outlier_flag")` **in
  the integration layer, not in `snip_qc/build.py`** — keeps the shared pipeline contract untouched.
  Caveat: ~4 of the 76 are genuinely thin/broken masks; adding `AND area >= 0.8 x cut` drops all
  four. Full analysis: `results/nlammers/20260729_morphseq_integration/`.
- **The current pipeline's snips are degraded vs legacy** — `target_pixel_size_um: 7.8` in
  `config.yaml` vs 6.5 in the legacy pipeline (area ratio 0.700 measured, 0.694 predicted), plus
  ~4x more saturation from an unresolved cause. Confined to the snip raster and everything computed
  on it (snips, `legacy_embeddings`, UNet aux masks, `fraction_alive`); `mask_geometry`,
  `curvature_metrics`, `pose_kinematics`, `stage_predictions`, and `surface_area_qc` are computed at
  full-frame native resolution and are unaffected. **This is why track 1 uses legacy embeddings.**
  The saturation cause is ON HOLD by Nick's instruction — a promising untested lead is that the
  snips are inverted, so "blown out" regions are *dark* in the original, making this plausibly a
  background-subtraction / background-estimate effect rather than highlight clipping;
  `estimate_background_stats_full_frame` computes background outside the mask in full-frame space,
  which is a different definition from the old build's.
- **Out of scope for now:** the `src/seq/hooke_latent_projections/project_ccs_data.py` port for
  transcriptional pseudostage. Nick called staging "nice to have, non-essential." It also needs a
  Hooke refit — `bead_expt_linear` was fit on REF1/GENE1/GENE2/GENE3 + HF2@28C and filters embryos
  to experiments in the model metadata, so GENE7 gets dropped outright. And no
  `GENE7_counts_table.csv` exists in `sci-PLEX/ccs_data_cell_type_broad/` yet.

## 8. First moves

1. Run `build_crosswalk()`; fix what breaks.
2. Add validation gate §6.1 (seq set equality). This is the highest-value single test.
3. Write `legacy_morph.py` per §4.
4. Resolve latent ordering (§6.3).
5. Produce a GENE7 master table: 576 paired wells x (legacy morph latents + seq metadata + QC policy
   flag), and report coverage honestly — how many wells have morph, how many have seq, how many have
   both.
6. Add `tests/morphseq_integration/`.

## 9. Working style Nick has asked for

- **Do not start writing code until he explicitly says go.** Asking "should I do X?" is a real
  question — stop and wait for the answer. Design discussion is not a green light. He has stated
  this is important to him.
- **Never rename/move/edit files without explicit consent**, even an obvious-looking fix. Surface it
  and let him decide.
- **When told to run something, do only that** — no unrequested dry-runs, verification passes, or
  side checks.
- Reading and investigating to answer a question is always fine.
- Work on `main`; do not create feature branches; do not push (he pushes).
