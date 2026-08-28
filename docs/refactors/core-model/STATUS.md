# Core-model status

Generated `2026-08-28T01:10:16+00:00` by `scripts/status.py`; do not edit by hand.

## Git

- Branch: `core-model-refactor`
- HEAD: `d830391f9f7132d8086a53b5470129847e07e9d2`
- Dirty tree: **yes** (15 entries)
- Upstream: `origin/core-model-refactor`
- Unpushed commits: **1**

```text
d830391f Pegged background statistics for snips
```

Dirty entries:

```text
 M AGENTS.md
 M docs/refactors/core-model/DECISIONS.md
 M docs/refactors/core-model/PLAN.md
 M docs/refactors/core-model/README.md
 M docs/refactors/core-model/contracts/MANIFEST_SCHEMA.md
 M docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md
M  src/data_pipeline/object_extraction/snip_processing/augmentation.py
MM src/data_pipeline/object_extraction/snip_processing/entrypoints/run_snip_processing.py
A  src/data_pipeline/object_extraction/snip_processing/inventory_contract.py
A  src/data_pipeline/object_extraction/snip_processing/provenance.py
MM src/data_pipeline/pipeline_orchestrator/rules/snip_processing.smk
M  src/data_pipeline/pipeline_orchestrator/tasks.py
A  tests/data_pipeline/object_extraction/snip_processing/test_rendering_provenance.py
M  tests/data_pipeline/object_extraction/snip_processing/test_run_snip_processing.py
?? results/nlammers/20260723_seahub/submit_seahub_back_half_rerun.sge
```

| Commit of interest | Revision | Provenance | Ancestor of HEAD | Ancestor of `origin/main` |
|---|---|---|---:|---:|
| Rendering defaults restored | `37aeb639` | Added by Codex for task (ii)2; evidence: `docs/refactors/core-model/reports/GROUND_TRUTH_2026-08-27.md:639-653`; added 2026-08-27 | yes | yes |

## Tests

- Command: `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python -m pytest tests/core -q --color=no --disable-warnings --tb=short -p scripts.status --status-json <status-json>`
- Duration: **106.50s** (limit 600.0s)
- Result: **PASS** (pytest exit 0)
- Counts: **5 passed · 0 failed · 0 skipped · 0 xfailed · 0 xpassed · 0 not run**

## Decisions

- Ledger: `docs/refactors/core-model/DECISIONS.md`

> **UNVERIFIED DECISIONS: D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33**

**31 decisions · 0 with tests (0 passing) · unverified: [D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33]**

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

- Total no-data runtime: **112.54s**
