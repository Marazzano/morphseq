# Cilia-Mutant Experiment Registry

A durable reference answering: **which experiments contain cilia-mutant data, for
which gene, how many embryos, and at what stages.**

Genes tracked: **cep290, tmem67, rpgrip1l, b9d2**.

## Files

- `cilia_mutant_experiment_registry.csv` — one row per `(experiment_id, gene)`.
- `build_registry.py` — regenerates the CSV from current source files.
- `README.md` — this file.

## How to refresh

```bash
source /net/trapnell/vol1/home/mdcolon/proj/morphseq/.venv/bin/activate
python results/mcolon/20260715_lab_meeting_scratch/registry/build_registry.py
```

Rerun whenever new plate-metadata workbooks or build06 outputs land. The script
prints the per-gene summary and the exact genotype token forms it matched.

## Summary

| Gene      | # experiments |
|-----------|---------------|
| cep290    | 36            |
| b9d2      | 20            |
| tmem67    | 5             |
| rpgrip1l  | 1             |

**Total unique experiments: 59** (62 registry rows; a few experiments carry more
than one cilia gene, e.g. `20251205` has both cep290 and tmem67, and `20250509`
has both cep290 and rpgrip1l).

Coverage: 50 rows have build06 embryo-level detail; 12 rows are plate-metadata
only (no build06 file yet); 24 rows are build06-only (see the experiment-id
caveat below).

## How it was built

**Source 1 — plate metadata genotype sheets** (`metadata/plate_metadata/*.xlsx`,
priority source for *presence*). For each workbook the sheet named `genotype`
(case-insensitive) is read as a plate grid, and **every cell** is scanned. If any
cell contains a gene token (case-insensitive substring), the experiment is marked
for that gene. Files with `_backup_` or `_fixed` in the name are skipped
(189 workbooks scanned). `experiment_id` here is the **leading date token** of the
filename (e.g. `20260210` from `20260210_well_metadata.xlsx`).

**Source 2 — build06 output**
(`morphseq_playground/metadata/build06_output/df03_final_output_with_latents_*.csv`,
priority source for *embryo counts / stages*). The `genotype` column is
substring-matched per gene; for each matching experiment we report unique
`embryo_id` count (restricted to `use_embryo_flag == True`) and
`predicted_stage_hpf` min/max. `experiment_id` here comes from the CSV's own
`experiment_id` column.

### Substring-matching method & token forms matched

Matching on the bare gene token sidesteps zygosity-suffix and spelling variation.
The distinct token forms actually present in the genotype sheets were:

- **cep290**: `cep290_wildtype`, `cep290_het`, `cep290_heterozygous`,
  `cep290_homo`, `cep290_homozygous`, `cep290_crispant`, `cep290_unknown`,
  `cep290s` (placeholder note "tbd cep290s")
- **tmem67**: `tmem67`, `tmem67_wildtype`, `tmem67_heterozygous`,
  `tmem67_homozygous`, `tmem67_unknown`, `tmem67s` (placeholder "tbd tmem67s")
- **rpgrip1l**: `rpgrip1l_wildtype`, `rpgrip1l_heterozygous`, `rpgrip1l_mutant`,
  `rpgrip1l_unknown`
- **b9d2**: `b9d2`, `b9d2_wildtype`, `b9d2_wt`, `b9d2_het`, `b9d2_heterozygous`,
  `b9d2_homo`, `b9d2_homozygous`, `b9d2_uncertain`, `b9d2_unknown`,
  `b9d2s` (placeholder "b9d2s to genotype")

## Caveats

- **`rpgrip1l` vs `rpgrip1`**: we search only for the exact token `rpgrip1l`.
  Because `rpgrip1l` already contains `rpgrip1`, no bare-`rpgrip1` search is done,
  so there is no false-positive risk. Only one experiment (`20250509`) carries it.
- **Plural / placeholder tokens** (`cep290s`, `tmem67s`, `b9d2s`) are genotyping-
  pending notes ("tbd cep290s", "b9d2s to genotype"). These are genuine cilia-
  mutant experiments awaiting genotype calls, so they are correctly included — not
  false positives.
- **Experiment-id granularity mismatch (why some rows look duplicated).** The two
  sources key experiments differently. Plate-metadata `experiment_id` is the
  leading **date only** (per spec), so several per-plate / per-stage workbooks that
  share a date (e.g. `20260324_cep290_18hpf_plate01`,
  `20260324_cep290_30hpf_plate02`, ...) all collapse to a single date row
  (`20260324`), shown as *metadata-only*. build06, however, keys on its own full
  compound `experiment_id` (`20260324_cep290_30hpf_plate01`, etc.), shown as
  *build06-only*. So for the 2025-10 and 2026-03/04 cilia batches you will see a
  coarse date row (plate side) plus several fine per-plate rows (build06 side)
  describing the same underlying work. Embryo counts and stages are on the
  fine-grained build06 rows.
- **Not all experiments have build06 coverage**, and vice-versa — the `in_plate_metadata`
  and `in_build06` boolean columns record exactly which sources saw each row.
- **`n_embryos` / stages are blank** when no build06 file exists. A few build06
  files have all `predicted_stage_hpf` missing (e.g. `20250425`), leaving stage
  columns blank while `n_embryos` is still populated.

## Column reference (`cilia_mutant_experiment_registry.csv`)

| column | meaning |
|--------|---------|
| `experiment_id` | date token (plate side) or full build06 id |
| `gene` | one of cep290 / tmem67 / rpgrip1l / b9d2 |
| `in_plate_metadata` | gene token found in a genotype sheet |
| `in_build06` | gene found in a build06 genotype column |
| `n_embryos` | unique `embryo_id` with `use_embryo_flag=True` (build06); blank if no build06 |
| `stage_min_hpf` / `stage_max_hpf` | `predicted_stage_hpf` range (build06) |
| `plate_metadata_file` | source workbook basename (if any) |
| `notes` | e.g. metadata-only / build06-only flags |
