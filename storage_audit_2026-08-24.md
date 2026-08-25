# Home-directory storage audit

**Audit date:** 2026-08-24  
**Scope:** `/net/trapnell/vol1/home/nlammers`  
**Mode:** Read-only inspection. No existing file or directory was created, edited, moved, deleted, compressed, archived, or cleaned. This Markdown report is the only created artifact.

## Executive summary

- The audit establishes a **known, non-overlapping lower bound of 62.079 TiB** in the home directory. The true total is higher because several branches containing millions of objects did not finish within bounded scans.
- Known raw-image storage is at least **49.554 TiB**. This excludes the unmeasured remainder of MorphSeq raw Keyence data, so it is also a lower bound.
- The largest single product is `projects/data/morphseq/raw_image_data/YX1` at **39.659 TiB**. It contains 90 very large ND2 files plus a small TIFF component.
- The next-largest product families are `projects/data/killi_dynamics` at **10.453 TiB** and `projects/data/pecfin_dynamics` at **4.298 TiB**.
- The clearest high-volume detritus/review candidates are:
  - **2.454 TiB** in Killi Cellpose `_Archive` output;
  - **812.17 GiB** of `monocle.bpcells.*.tmp` products, including **569.63 GiB at least one year old**;
  - **304.52 GiB** in a training `old_sweep`;
  - **106.37 GiB** in a processed sci-PLEX `_Archive`;
  - **39.78 GiB** of Nextflow `work` data;
  - a **suspected 370.82 GiB duplicate ND2 pair** that requires a full checksum before any action;
  - caches, editor runtimes, logs, and environments that collectively add tens to hundreds of GiB.
- A directory named `raw_image_data/ignore` occupies **2.335 TiB**, almost entirely one `YX1/20250515_part1` acquisition. Its location makes it a high-priority retention review, but it is raw data and should not be deleted merely because it is named `ignore`.
- The backing GPFS volume is 210 TiB, 99% used, with about 3.9 TiB free. That is a cluster-wide filesystem figure, not this user's quota or usage.

## How to read the figures

- `du -x -B1` supplied allocated-byte measurements. `-x` prevents traversal onto another filesystem.
- **Exact** means the bounded traversal completed.
- **Lower bound** means every included component completed, but one or more sibling branches did not.
- File-type totals use apparent file sizes from filesystem metadata. They can differ slightly from allocated `du` totals.
- Binary units are used: 1 GiB = 2^30 bytes and 1 TiB = 2^40 bytes.
- Parent and child rows are intentionally both shown for navigation. They must not be added together.

## Footprint overview

| Product family | Measured footprint | Status | Interpretation |
|---|---:|---|---|
| `projects/data/morphseq` | **>=46.206 TiB** | Lower bound | Dominated by raw microscopy; several high-object branches unresolved |
| `projects/data/killi_dynamics` | **10.453 TiB** | Exact | Raw ND2 plus large Cellpose, Zarr, and segmentation products |
| `projects/data/pecfin_dynamics` | **4.298 TiB** | Exact | Raw ND2 plus Zarr and Cellpose-derived data |
| Home outside `projects` and `micromamba` | **1.013 TiB** | Exact | Mostly BPCells temp products, dated TIFF acquisitions, sequencing output, and caches |
| All repositories under `projects/repositories` | **23.83 GiB** | Exact product sum | Active MorphSeq checkout is the only repository above ~10 GiB |
| `micromamba` components | **<=144.62 GiB component sum** | Upper bound, not a root total | Separate component scans can double-count Conda hard links |
| **Known home-directory total** | **>=62.079 TiB** | Lower bound | Excludes unresolved branch content; actual use is higher |

The lower bound is conservative: unresolved content is omitted rather than estimated.

## 1. MorphSeq data: the dominant footprint

### 1.1 Raw image data

| Path | Allocated size | Status | Notes |
|---|---:|---|---|
| `projects/data/morphseq/raw_image_data/YX1` | **39.659 TiB** | Exact | 69 dated entries; one is empty and 68 nonempty acquisitions are each above ~40 GiB |
| `projects/data/morphseq/raw_image_data/ignore` | **2.335 TiB** | Exact | Dominated by one acquisition below |
| `.../raw_image_data/ignore/YX1/20250515_part1` | **2.312 TiB** | Exact | Raw-data retention/archive decision required |
| `projects/data/morphseq/raw_image_data/Keyence` | **>=1.998 TiB** | Lower bound | Exact 2024–2026 totals plus three measured 2023 acquisitions; more 2023 content and `Keyence/ignore` remain unmeasured |
| **Known MorphSeq raw-image total** | **>=43.992 TiB** | Lower bound | Actual total is higher |

#### YX1 by acquisition year

| Year | Entries | Allocated size |
|---|---:|---:|
| 2023 | 3 | 4.744 TiB |
| 2024 | 13 | 4.648 TiB |
| 2025 | 27 | 18.053 TiB |
| 2026 | 26 | 12.214 TiB |
| **Total** | **69** | **39.659 TiB** |

Largest YX1 dated products include:

| Acquisition | Allocated size |
|---|---:|
| `20231206` | 2.634 TiB |
| `20231110` | 1.755 TiB |
| `20250912` | 1.409 TiB |
| `20250512` | 1.296 TiB |
| `20251121` | 1.294 TiB |
| `20250305` | 1.273 TiB |
| `20250501` | 1.169 TiB |
| `20250711` | 1.147 TiB |
| `20251113` | 1.126 TiB |
| `20251106` | 1.059 TiB |

All other nonempty YX1 dated products are still above the requested ~10 GiB needle-mover threshold.

#### Raw Keyence coverage

| Partition | Coverage | Allocated size |
|---|---|---:|
| 2023 | 3 acquisitions measured; additional acquisitions unresolved | **>=487.84 GiB** |
| 2024 | 13 acquisitions | 173.07 GiB |
| 2025 | About 50 acquisition directories plus tiny mapping files | 1.104 TiB |
| 2026 | 43 acquisitions | 254.12 GiB |
| **Known Keyence total** | Plus unmeasured 2023 and `Keyence/ignore` | **>=1.998 TiB** |

The completed 2025 Keyence acquisitions are generally ~16.6–26.7 GiB each. The 2026 acquisitions are mostly ~3–7 GiB each and are collectively significant rather than individually above 10 GiB. The high object count prevented an exact total within the audit bound.

#### MorphSeq raw file types

Across `YX1` and the outer `raw_image_data/ignore` branch:

| Type | File count | Apparent size |
|---|---:|---:|
| ND2 | 90 | ~41.97 TiB |
| TIFF | 3,629 | ~22.98 GiB |
| GCI/link/spreadsheet/XML metadata | 345 | Negligible |

Raw Keyence is organized as many TIFF acquisitions, but a complete extension aggregation was not attempted after its bounded size scan expired.

### 1.2 Built image data

| Path | Allocated size | Notes |
|---|---:|---|
| `projects/data/morphseq/built_image_data` | **1.346 TiB** | Exact parent total |
| `.../Keyence_stitched_z` | **1.075 TiB** | Largest built representation |
| `.../stitched_FF_images` | **~253.32 GiB** | Inferred as parent remainder after exact sibling totals; only negligible directory overhead affects the inference |
| `.../Keyence` | **24.16 GiB** | Exact |

These are derived image products. Before archiving or deleting any copy, retain the raw acquisition, pipeline version, parameters, and enough metadata to reproduce the representation.

### 1.3 Training data and models

| Path | Allocated size | Status/notes |
|---|---:|---|
| `projects/data/morphseq/training_data/models` | **690.71 GiB** | Exact |
| `.../models/training_outputs` | **357.85 GiB** | Exact |
| `.../models/old_sweep` | **304.52 GiB** | Exact; strong derived-detritus candidate |
| `.../models/hydra_outputs` | **24.94 GiB** | Exact |
| Current masks/snips outside `_Archive` | ~4.22 GiB combined | Exact |
| `training_data/_Archive/bf_embryo_snips` | **20.50 GiB** | Exact |
| Other `training_data/_Archive` variants | Unresolved | Z05, uncropped, masks, and dated datasets are high-object branches |

The known non-overlapping training-data lower bound is about **715.4 GiB**. The actual value is higher because archived variants were excluded from the lower-bound arithmetic when their scans timed out.

### 1.4 sci-PLEX

| Path | Allocated size |
|---|---:|
| `projects/data/morphseq/sci-PLEX` | **156.11 GiB** |
| `.../processed_sci_data` | 155.63 GiB |
| `.../processed_sci_data/_Archive` | **106.37 GiB** |
| `.../gap16_no_ctrls_projected_comb_cds_v2.0.2` | 15.88 GiB |
| `.../lmx1b_combined_analysis` | 13.91 GiB |
| `.../reference_cds_v2.0.2` | 12.31 GiB |

The `_Archive` subtree is the clear review candidate. These appear to be processed single-cell products, so confirm that the underlying counts, metadata, code, and package versions are retained before removing derived objects.

### 1.5 Unresolved MorphSeq branches

The following branches exceeded their bounds and are omitted from the 46.206 TiB lower bound except for explicitly completed children:

- `raw_image_data/Keyence`: unmeasured 2023 acquisitions and `Keyence/ignore`.
- `segmentation/{focus_v0_0100_predictions,mask_v0_0100_predictions,yolk_v1_0050_predictions,via_v1_0100_predictions,bubble_v0_0100_predictions}`. Model binaries themselves are only ~797 MiB; prediction outputs cause the high object count.
- `pipeline/output`, including `.stale_contract_quarantine_20260726`, `work_directories`, acquisition/object/feature extraction, QC, and analysis-ready products.
- `pipeline/stale_output`, containing three July 2026 SeaHub cleanup/reset products.
- `training_data/_Archive` variants other than the measured 20.50 GiB `bf_embryo_snips` child.

These are important follow-up targets because their inability to complete a bounded metadata walk indicates very high object counts even where byte totals are unknown.

## 2. Killi dynamics: 10.453 TiB

| Path | Allocated size | Classification |
|---|---:|---|
| `projects/data/killi_dynamics/built_data` | **5.995 TiB** | Derived |
| `.../built_data/cellpose_output` | **5.022 TiB** | Derived model output |
| `.../cellpose_output/tdTom-bright-log-v5/_Archive` | **2.454 TiB** | Archived derived output; strong review candidate |
| `.../built_data/zarr_image_files` | **951.49 GiB** | Derived/chunked image representation |
| `projects/data/killi_dynamics/raw_data` | **2.869 TiB** | Primary ND2 acquisitions |
| `projects/data/killi_dynamics/segmentation` | **1.584 TiB** | Derived predictions |
| `.../segmentation/tdTom-bright-log-v5/cellpose_output/20251023` | **1.572 TiB** | Derived prediction product |
| `.../built_data/mips` | 33.79 GiB | Derived projection images |
| `.../built_data/mask_stacks` | 10.85 GiB | Derived masks |

The 2.454 TiB Cellpose `_Archive` contains three large dated products:

| Archived dataset | Allocated size |
|---|---:|
| `20250716` | 738.56 GiB |
| `20250621` | 886.55 GiB |
| `20250731` | 887.95 GiB |

The active Cellpose branch and `segmentation/.../cellpose_output` have similar model naming but different dated contents. This suggests successive pipeline layouts, not proven duplication. Compare manifests/content before treating them as redundant.

### Killi raw acquisitions

| Acquisition | Allocated size |
|---|---:|
| `20251023` | 700.02 GiB |
| `20250731` | 613.77 GiB |
| `20260612_wt_pair_tdTom` | 562.83 GiB |
| `20260610_wt_pair_tdTom` | 505.71 GiB |
| `20250621` | 312.60 GiB |
| `20250716` | 242.70 GiB |

Ten ND2 files account for essentially the entire Killi raw-data footprint.

## 3. Pecfin dynamics: 4.298 TiB

| Path | Allocated size | Classification |
|---|---:|---|
| `projects/data/pecfin_dynamics/raw_data` | **2.577 TiB** | Primary ND2/TIFF acquisitions |
| `projects/data/pecfin_dynamics/built_data` | **1.721 TiB** | Derived |
| `.../built_data/zarr_image_files` | **1.306 TiB** | Derived/chunked image representation |
| `.../built_data/cellpose_output` | **232.99 GiB** | Derived predictions |
| `.../built_data/mask_stacks` | **192.27 GiB** | Derived masks |

Largest raw acquisitions:

| Acquisition | Allocated size |
|---|---:|
| `20250225` | 795.81 GiB |
| `20240223` | 743.88 GiB |
| `20240620` | 504.17 GiB |
| `20240619` | 371.26 GiB |
| `20240424` | 161.25 GiB |

Forty-three ND2 files account for ~2.574 TiB apparent. Another 9,147 TIFF files total less than 1 GiB, so large ND2 acquisitions—not TIFF count—drive Pecfin raw storage.

### Suspected raw-data duplicate

The following files have the same name and exact size but are separate inodes, so they consume separate storage:

- `projects/data/pecfin_dynamics/raw_data/20240223/wt_tdTom_timelapse_long.nd2`
- `projects/data/pecfin_dynamics/raw_data/20240223/additional_files/wt_tdTom_timelapse_long.nd2`

Each is 398,165,942,272 bytes (**370.82 GiB**). Their modification times are about one hour apart. This is a suspected duplicate, not a confirmed duplicate. A full cryptographic checksum and provenance check are required before considering either copy removable.

## 4. Storage outside `projects`

The exact total outside `projects` and `micromamba` is **1.013 TiB**.

### 4.1 BPCells temporary products: 812.17 GiB

`tmp_files/nobackup` contains 264 immediate `monocle.bpcells.*.tmp` products totaling **812.17 GiB**. These are temporary products rather than raw imagery.

| Modification-age bucket | Products | Allocated size |
|---|---:|---:|
| At least 365 days | 217 | **569.63 GiB** |
| 90–179 days | 29 | 84.26 GiB |
| 30–89 days | 13 | 69.57 GiB |
| Under 30 days | 5 | 88.71 GiB |

Largest individual temporary products:

| Product | Allocated size |
|---|---:|
| `monocle.bpcells.20260730.52c64b071dfc.tmp` | 23.52 GiB |
| `monocle.bpcells.20260730.1da434b36fa66.tmp` | 23.52 GiB |
| `monocle.bpcells.20260730.52c661ea8d4f_r.tmp` | 17.53 GiB |
| `monocle.bpcells.20260730.1da4325b408ee_r.tmp` | 17.53 GiB |

The largest date groups are `20250202` (~95.6 GiB), `20260730` (~88.7 GiB), `20250213` (~82.4 GiB), `20250214` (~80.1 GiB), and `20250205` (~61.6 GiB). Review active processes and whether these temporary matrices are referenced before cleanup; age and `.tmp` naming alone are not proof that they are unused.

### 4.2 Dated TIFF image experiments: 120.40 GiB

| Experiment | Allocated size |
|---|---:|
| `20250716/20250716_chem4_35C_T00_1045` | 26.73 GiB |
| `20250716/20250716_chem4_34C_T01_1014` | 26.67 GiB |
| `20250716/20250716_chem4_28C_T01_1400` | 25.08 GiB |
| `20250716/20250716_chem4_28C_T00_1158` | 25.02 GiB |
| `20250721/20250721_chem5_28C_T00_1257` | 16.90 GiB |

These directories contain 18,108 TIFF files with ~119.24 GiB apparent size. They appear to duplicate experiment names also present under MorphSeq raw Keyence data; for example, the July 2025 experiment directories appear in both locations. This is a **suspected cross-location duplicate set**, not confirmed duplication. Verify file manifests or checksums before acting.

### 4.3 Sequencing run: 70.42 GiB

`sci_3lvl_runs/230828_lmx1b_crispant` contains:

| Component | Allocated size | Interpretation |
|---|---:|---|
| `work` | **39.78 GiB** | Nextflow intermediate/work data; review after validating final outputs |
| `demux_out` | **20.45 GiB** | Compressed FASTQ/FQ output |
| `lmx1b` | **10.17 GiB** | Includes BAM/analysis data |

File metadata show ~34.4 GiB of BAM files and ~34.2 GiB of compressed FASTQ/FQ files across the full run. `work` is the most likely reproducible detritus, while BAM/FASTQ retention depends on the project's archival policy.

### 4.4 Caches: 15.14 GiB

| Cache | Allocated size |
|---|---:|
| `.cache/pip` | **9.12 GiB** |
| `.cache/huggingface` | **3.69 GiB** |
| `.cache/wandb` | **1.40 GiB** |
| `.cache/torch` | **0.83 GiB** |
| Other `.cache` content | ~0.10 GiB |

These are generally reproducible/downloadable, but clearing them can require network access, exact-version retrieval, or rerunning jobs. Review application-specific cache policies first.

## 5. Environments, editor state, and repository detritus

### 5.1 Micromamba/Conda

Measured separately:

| Component | Standalone scan |
|---|---:|
| `micromamba/envs` (17 named environments) | 86.64 GiB |
| `micromamba/pkgs` package cache | 39.61 GiB |
| Content excluding `envs` and `pkgs` | 18.36 GiB |
| Included standalone `micromamba/vae-env` | 10.99 GiB |
| **Component sum** | **144.62 GiB** |

The component sum is an upper bound, not an exact root total, because Conda commonly hard-links files from `pkgs` into environments. A whole-tree pass did not finish within ten minutes. Candidate actions should use Micromamba/Conda-aware environment and package-cache review rather than manual file deletion.

### 5.2 VS Code and remote runtime state

| Component | Allocated size |
|---|---:|
| `.vscode` | 3.81 GiB |
| `.vscode-server` | 4.65 GiB |
| `.vscode-remote-containers` | 1.28 GiB |
| **Combined** | **9.73 GiB** |

`.vscode/servers` alone is ~3.38 GiB and retains five Stable server versions, each roughly 0.48–0.80 GiB. `.vscode-server` contains ~2.63 GiB of extensions and ~2.02 GiB of data. Old versions and unused extensions are plausible small-item detritus.

### 5.3 User libraries and runtimes

| Component | Allocated size |
|---|---:|
| `.local` | 4.37 GiB |
| `R/x86_64-pc-linux-gnu-library/4.4` | 1.45 GiB |
| `projects/repositories/.venv` | 1.27 GiB |
| Active MorphSeq checkout `.venv` | 0.22 GiB |

These are reproducible only if environment specifications and package sources remain available.

### 5.4 Active MorphSeq repository: 13.39 GiB

| Component | Allocated size | Notes |
|---|---:|---|
| `logs` | **7.76 GiB** | 2,567 files; almost all bytes are `.err` files from the last 90 days |
| `.git` | **2.34 GiB** | Almost entirely packed Git objects |
| `results` | **1.65 GiB** | Mostly `results/mcolon` and `results/nlammers` |
| `src` | 0.95 GiB | Source/package content |

Logs are the clearest repository-local detritus. Do not manually remove `.git/objects`; use Git-aware maintenance only after confirming repository integrity and remote/back-up coverage.

The other repository products are each below 2 GiB. All repository products together occupy ~23.83 GiB.

### 5.5 Negligible trash

`.Trash-1001` is only about 145 KiB. Emptying trash would not materially affect the storage problem.

## 6. Candidate review order

This order balances likely reclaimable space against data value. It is **not** an instruction to delete.

1. **Archive/tier primary MorphSeq YX1 raw acquisitions (39.659 TiB)** if the cluster has a designated archival store. These are the largest objects but likely irreplaceable, so deletion should be the last resort.
2. **Validate `raw_image_data/ignore/YX1/20250515_part1` (2.312 TiB).** Determine why primary raw data live under `ignore`, whether a canonical copy exists, and whether archival is appropriate.
3. **Review Killi Cellpose `_Archive` (2.454 TiB).** This is archived derived data inside a derived-output hierarchy and is the largest plausibly reclaimable product.
4. **Review BPCells temporary products (812.17 GiB), starting with the 569.63 GiB at least one year old.** Confirm no live R sessions/jobs or notebooks reference them.
5. **Review training `old_sweep` (304.52 GiB)** and archived training variants. Preserve winning checkpoints, configs, metrics, seeds, and code revisions before removing failed/redundant sweeps.
6. **Checksum the suspected Pecfin duplicate pair (potentially 370.82 GiB redundant).** Same size/name and distinct inodes are suggestive but insufficient.
7. **Review sci-PLEX `processed_sci_data/_Archive` (106.37 GiB).** Confirm that raw counts and a reproducible processing recipe exist.
8. **Review Nextflow `work` (39.78 GiB)** after confirming final demultiplexed and analysis outputs are complete and backed up.
9. **Review Zarr/Cellpose/segmentation representations.** Killi and Pecfin have multiple TiB of derived stores; retain only representations that cannot be regenerated economically or exactly.
10. **Tidy caches, logs, editor versions, and unused environments.** Individually smaller, they are lower-risk ways to recover tens to perhaps over 100 GiB.

## 7. Safety checklist before any deletion or archive move

- Identify the canonical copy and responsible project owner.
- For raw data, confirm an independent archival copy and verify it with a checksum before removing cluster storage.
- For suspected duplicates, compare full cryptographic hashes; matching names and sizes are not enough.
- For derived products, record source inputs, code commit, environment, model version, parameters, and output manifest.
- Reproduce a representative subset before deleting expensive derived outputs.
- Check for active jobs, open workflows, symlinks, notebooks, and manifests that reference the path.
- Prefer application-aware cleanup for Nextflow, Conda/Micromamba, Git, pip, W&B, and editor runtimes.
- Stage decisions by product boundary; do not use broad wildcard or recursive deletion commands.
- Re-measure after each approved cleanup step rather than acting on the entire list at once.

## 8. Audit limitations

- `projects` contains millions of objects. An initial exhaustive home walk was stopped without modifying data after it remained inside `projects` for about an hour.
- The audit then used exact product-level traversals with 2–10 minute bounds. Completed figures are exact allocated sizes; expired branches are clearly marked and excluded from lower-bound totals.
- `projects/data/morphseq` did not finish as one product. Its 46.206 TiB figure is a sum of non-overlapping completed children and is therefore a lower bound.
- No consistent exact total was obtained for raw Keyence, MorphSeq prediction outputs, pipeline output/stale output, and several archived training-image variants.
- The GPFS quota wrapper attempted to create temporary IPC state and was blocked. It was not bypassed, preserving the requirement that this report be the only created artifact.
- The filesystem was not snapshotted. Sizes can change if other jobs write concurrently.
- Hard links can make sums of separately scanned paths exceed physical use. This is explicitly accounted for in the micromamba section and is why its component sum is not added to an exact root total.
- The volume-wide `df` result (210 TiB total, 207 TiB used, 3.9 TiB available, 99% full) includes other users and should not be interpreted as a personal quota.

## Bottom line

The storage crisis is primarily raw microscopy: at least **49.554 TiB**, led by MorphSeq YX1 ND2 acquisitions. However, the audit also identifies multiple high-confidence review pools that do not require sacrificing canonical raw data: a **2.454 TiB derived archive**, **812.17 GiB of BPCells temporary products**, a **304.52 GiB old training sweep**, a **106.37 GiB processed-data archive**, **39.78 GiB of Nextflow work data**, and smaller caches/logs/runtime detritus. The safest near-term strategy is to validate and remove or externally archive those derived/temp candidates first, then establish archival tiers for the tens of TiB of primary ND2 data.
