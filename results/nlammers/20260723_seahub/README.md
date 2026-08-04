# SeaHub sequence-image exploration

This directory is a self-contained, read-only exploration of:

- `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/image_data`
- `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/seahub/metadata/collection_metadata.xlsx`

No source image, source metadata, or code outside this directory is modified.

## Contents

- `01_seahub_registry_and_segmentation.ipynb`: exploratory entry point.
- `seahub_workflow.py`: registry, XLSX parsing, metadata matching,
  GroundingDINO inference, QC, position assignment, and cropping.
- `test_seahub_workflow.py`: focused dependency-light tests.
- `outputs/image_registry.csv`: one row per JPG/JPEG.
- `outputs/image_metadata_reconciliation.csv`: registry plus match status and
  matched collection metadata.
- `outputs/segmentation_smoke_test/`: a real eight-embryo GroundingDINO result
  for `GENE16 / 24hpf_A_foxc1a.jpg`.

## Beta-code inventory

`results/mcolon/20260408_segmenting_sequence_images/01_test_gdino.py`:

1. Loads a fine-tuned GroundingDINO checkpoint on CPU.
2. Loads PNG test images and predicts `"individual embryo"` boxes.
3. Converts normalized `cxcywh` boxes to normalized `xyxy`.
4. Draws detection previews.
5. Writes a raw detection CSV.

It does not parse the SeaHub tree, read collection metadata, enforce an
eight-instance layout, assign within-FOV sequencing positions, or write
individual embryo snips. The code here adds those pieces while retaining the
same checkpoint, prompt, and initial thresholds.

## Current inventory and reconciliation

The 2026-07-23 scan found 1,925 JPGs:

- 1,062 active 1280×960 eight-embryo FOVs.
- 330 additional eight-embryo FOVs below abandoned/not-used paths.
- 524 existing single-embryo images, mostly 256×576 or 576×256 and including
  explicit `_cropped`/`_uncropped` pairs.
- 9 review images (phenotype grids, a close-up, or irregular legacy images).

Of the 1,062 active FOVs, 550 have a unique exact match to
`collection_metadata`; 9 have duplicate exact metadata rows; 268 have no
condition match; and 235 have no stage match. Ambiguous and low-confidence
matches are deliberately not promoted to default segmentation candidates.

The matcher uses:

- experiment ID from the path (`GENE*` or `CHEM*`);
- collected stage from the filename or a legacy `condition_stage` directory;
- perturbation tokens with separator/order normalization;
- prior/addition stage where present, which disambiguates numbered chemical
  images.

Every row retains `metadata_match_status`, score, and candidate count so a
manual correction table can be added later without hiding uncertainty.
Collected age is exported as the numeric `stage_hpf` field (for example,
`24.0`), while `stage_source_label` preserves the original token. Somite or
named stages are left null in `stage_hpf` rather than assigned an unstated
hours-post-fertilization conversion.

## Embryo positions and outputs

For exactly eight detections, positions are assigned as:

```text
1  2  3  4
5  6  7  8
```

That is top row left-to-right, followed by bottom row left-to-right. The
filename's letter/replicate label is retained separately as `fov_label`.

Snips are bounding-box instance crops with 8% padding, not pixelwise masks.
Each passing embryo is written in both RGB and grayscale; the manifest exposes
the files as `snip_color_path` and `snip_grayscale_path`. This matches the
capability in the beta code, which is a detector despite its
segmentation-oriented directory name. A count mismatch writes a QC preview
but no partial crop set.

## Real smoke test

The fine-tuned model returned exactly eight detections for
`GENE16 / 24hpf_A_foxc1a.jpg` using the beta thresholds:

- box threshold: `0.15`
- text threshold: `0.10`
- detections before/after NMS: `8 / 8`
- CPU model load: about 70 seconds
- CPU inference plus crop writing: about 10 seconds

See `outputs/segmentation_smoke_test/snips_contact_sheet.jpg`,
`snips_grayscale_contact_sheet.jpg`, `embryo_manifest.csv`, and
`segmentation_qc.csv`.

## Additional cross-experiment validation

Three additional exact-match FOVs were processed together with one model load:

| Experiment | Image | Stage (hpf) | Raw / NMS detections | Result |
|---|---|---:|---:|---|
| GENE6 | `24hpf_A_pbx1b.jpg` | 24 | 9 / 8 | count pass; review |
| GENE13 | `48hpf_A_lhx2a.jpg` | 48 | 8 / 8 | pass |
| CHEM17 | `72hpf_A_BGJ398.jpg` | 72 | 8 / 8 | pass |

The extra GENE6 proposal was an overlapping duplicate that NMS removed.
Subsequent review identified two bona fide errors in the GENE6/Pbx crops,
which the current count-only FOV QC does not capture.
Each FOV has eight RGB and eight grayscale snips under
`outputs/segmentation_validation_3_experiments/`, along with a combined
manifest, QC table, previews, and per-FOV contact sheets. Every contact-sheet
panel is labeled in its upper-left corner with position and detection
confidence (for example, `P6 0.233`).

## Environment obstacles

- `openpyxl` is absent from `morphseq-env`, so the support module contains a
  small dependency-free reader for this workbook's simple tabular XLSX data.
- `segmentation_sandbox/models/GroundingDINO` is empty, although the beta
  script assumes that checkout. The loader uses the existing read-only
  GroundingDINO checkout under `mdcolon/proj/image_segmentation` and the exact
  saved fine-tuning config beside the checkpoint.
- `morphseq-env` lacks the GroundingDINO text-model stack. The loader reuses
  already-installed, read-only `transformers` packages from `points-ml`; it
  does not install anything.
- The compiled GroundingDINO operator is incompatible with the current
  environment. The source's valid pure-PyTorch CPU fallback works and was
  used for the smoke test.
- Full-corpus inference was not run. At roughly 10 seconds per FOV after model
  load, the 550 uniquely reconciled candidates would take roughly 1.5 hours
  serially, before review of count mismatches.

## Running

Production SGE scripts require an explicit, unique run identifier; there is no
fallback to a prior dated output tree. For example:

```bash
qsub -v SEAHUB_RUN_ID=20260731_prod01 ...
```

The identifier selects matching fresh detection and bundle roots. It should
begin with the operational date (`YYYYMMDD`); otherwise also pass
`SEAHUB_OPERATIONAL_DATE=YYYYMMDD`. Reusing a nonempty bundle, detection
partition, or merged detection output fails loudly.

After the full-corpus GroundingDINO partitions have been merged, generate the
cleaned source-mask census used for both SeaHub scale calibration and the
authoritative downstream mask handoff with one GPU:

```bash
qsub -v SEAHUB_RUN_ID=20260731_prod01 \
  -hold_jid <groundingdino-merge-job-id> \
  results/nlammers/20260723_seahub/submit_seahub_sam2_areas.sge
```

This runs one box-prompted SAM2 call per source FOV and writes
`scale_calibration/sam2_areas/sam2_mask_areas.csv` plus one cleaned full-FOV
binary PNG per embryo under `scale_calibration/sam2_areas/masks/`. Cleanup keeps
the prompt-associated connected component and fills holes. The CSV is replaced
atomically after every FOV, so a terminated job retains a valid partial
checkpoint. The output directory must be fresh and empty.
After that job succeeds, submit `submit_seahub_plan_and_launch.sge` with
`-hold_jid <sam2-area-job-id>` and the same `SEAHUB_RUN_ID`; planning requires
both the completed CSV and its `_SUCCESS` sentinel.

Open the notebook with the `morphseq-env` kernel. Its registry and
reconciliation cells are safe to run directly. The expensive model cell is
guarded by `RUN_SEGMENTATION = False`; set it to `True` after selecting the
desired candidate rows.

Focused tests:

```bash
cd /net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/results/nlammers/20260723_seahub
PYTHONDONTWRITEBYTECODE=1 conda run -n morphseq-env --no-capture-output \
  python -m unittest -v test_seahub_workflow.py
```
