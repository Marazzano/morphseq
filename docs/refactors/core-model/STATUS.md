# Core-model status

Generated `2026-08-28T23:58:44+00:00` by `scripts/status.py`; do not edit by hand.

## Git

- Branch: `core-model-refactor`
- HEAD: `5599df65ad98f189180652930c062a4f5ef7258d`
- Dirty tree: **yes** (39 entries)
- Upstream: `origin/core-model-refactor`
- Unpushed commits: **1**

```text
5599df65 docs(core): authorize pre-A3 metric mechanism work
```

Dirty entries:

```text
 M results/nlammers/20260723_seahub/submit_seahub_back_half_rerun_20260828.sge
 D src/build/.DS_Store
 D src/build/_Archive/__init__.py
 D src/build/_Archive/build01A_compile_keyence_images_cytometer.py
 D src/build/_Archive/build01A_compile_keyence_images_refactor.py
 D src/build/_Archive/build02A_adjust_ff_contrast.py
 D src/build/_Archive/build02B_segment_bf_main_v2.py
 D src/build/_Archive/build03A_process_embryos_main_par.py
 D src/build/_Archive/export_sample_yx1_stacks.py
 D src/build/_Archive/keyence_export_utils.py
 D src/build/_Archive/yx1_export_utils.py
 D src/build/__init__.py
 D src/build/benchmark_focus_stacker.py
 D src/build/build01AB_stitch_keyence_z_slices.py
 D src/build/build01A_compile_keyence_images.py
 D src/build/build01A_compile_keyence_torch.py
 D src/build/build01B_compile_yx1_images_torch.py
 D src/build/build02B_segment_bf_main.py
 D src/build/build03A_process_embryos_main_par.py
 D src/build/build03A_process_images.py
 D src/build/build03A_process_images.py.backup_pre_sam2_refactor
 D src/build/build03B_export_z_snips.py
 D src/build/build04.py.backup_20250913
 D src/build/build04_perform_embryo_qc.py
 D src/build/build05_make_training_snips.py
 D src/build/build_utils.py
 D src/build/data_classes.py
 D src/build/export_utils.py
 D src/build/file_checks.py
 D src/build/infer_developmental_age.py
 D src/build/merge_well_metadata_to_output.py
 D src/build/patch_build04_use_embryo_flag.py
 D src/build/pipeline_objects.py
 D src/build/qc/__init__.py
 D src/build/qc/embryo_flags.py
 D src/build/qc_utils.py
 D src/build/run_experiment_manager.sh
 D src/build/utils/__init__.py
 D src/build/utils/curvature_utils.py
```

| Commit of interest | Revision | Provenance | Ancestor of HEAD | Ancestor of `origin/main` |
|---|---|---|---:|---:|
| Rendering defaults restored | `37aeb639` | Added by Codex for task (ii)2; evidence: `docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md:639-653`; added 2026-08-27 | yes | yes |

## Tests

- Command: `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python -m pytest tests/core -q --color=no --disable-warnings --tb=short -p scripts.status --status-json <status-json>`
- Duration: **100.55s** (limit 600.0s)
- Result: **PASS** (pytest exit 0)
- Counts: **70 passed · 0 failed · 0 skipped · 0 xfailed · 0 xpassed · 0 not run**

## Decisions

- Ledger: `docs/refactors/core-model/DECISIONS.md`

> **UNVERIFIED DECISIONS: D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33, D34, D35, D36**

**34 decisions · 0 with tests (0 passing) · unverified: [D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33, D34, D35, D36]**

| Decision | Marked tests | Verification |
|---|---:|---|
| D1 | 0 | unverified |
| D2 | 0 | unverified |
| D4 | 0 | unverified |
| D5 | 0 | unverified |
| D6 | 0 | unverified |
| D7 | 0 | unverified |
| D8 | 0 | unverified |
| D9 | 0 | unverified |
| D10 | 0 | unverified |
| D11 | 0 | unverified |
| D12 | 0 | unverified |
| D13 | 0 | unverified |
| D15 | 0 | unverified |
| D16 | 0 | unverified |
| D17 | 0 | unverified |
| D18 | 0 | unverified |
| D19 | 0 | unverified |
| D20 | 0 | unverified |
| D21 | 0 | unverified |
| D22 | 0 | unverified |
| D23 | 0 | unverified |
| D24 | 0 | unverified |
| D25 | 0 | unverified |
| D26 | 0 | unverified |
| D27 | 0 | unverified |
| D28 | 0 | unverified |
| D29 | 0 | unverified |
| D30 | 0 | unverified |
| D31 | 0 | unverified |
| D32 | 0 | unverified |
| D33 | 0 | unverified |
| D34 | 0 | unverified |
| D35 | 0 | unverified |
| D36 | 0 | unverified |

## Environment

- Python: `3.10.16` at `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python`
- `pyarrow` imports: **yes (25.0.1)**

| Package | Version |
|---|---|
| `pytest` | `9.0.2` |
| `torch` | `2.5.1` |
| `torchvision` | `0.20.1` |
| `pandas` | `2.2.3` |
| `numpy` | `1.26.4` |
| `pyarrow` | `25.0.1` |
| `pytorch-lightning` | `2.5.1` |
| `hydra-core` | `1.3.2` |
| `omegaconf` | `2.3.0` |

## Data

Skipped (`--with-data` was not supplied).

## Generator

- Total no-data runtime: **105.46s**
