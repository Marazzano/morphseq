# Core-model status

Generated `2026-08-25T05:14:58+00:00` by `scripts/status.py`; do not edit by hand.

## Git

- Branch: `core-model-refactor`
- HEAD: `3ecc4818994de5c2ab9db1cafa85f1af60c64044`
- Dirty tree: **yes** (19 entries)
- Upstream: `origin/core-model-refactor`
- Unpushed commits: **2**

```text
3ecc4818 Merge branch 'core-model-refactor' of github.com:nlammers371/morphseq into core-model-refactor
4a457843 Doc to audit mem usage
```

Dirty entries:

```text
M  AGENTS.md
A  DECISIONS.md
M  docs/refactors/core-model/.DS_Store
 D docs/refactors/core-model/OUTSTANDING_PIPELINE_ISSUES.md
A  docs/refactors/core-model/PLAN.md
 D docs/refactors/core-model/SNIP_IMAGE_REGRESSION_STATUS.md
A  docs/refactors/core-model/THREAD_BRIEFS.md
 D docs/refactors/core-model/TRAINING_READINESS_REMAINDER.md
 D docs/refactors/core-model/UPSTREAM_PIPELINE_STATE.md
 D docs/refactors/core-model/plans/AGENT_BRIEFS_PHASE1.md
 M tests/conftest.py
?? docs/refactors/core-model/STATUS.md
?? docs/refactors/core-model/evidence/AGENT_BRIEFS_PHASE1.md
?? docs/refactors/core-model/evidence/OUTSTANDING_PIPELINE_ISSUES.md
?? docs/refactors/core-model/evidence/SNIP_IMAGE_REGRESSION_STATUS.md
?? docs/refactors/core-model/evidence/TRAINING_READINESS_REMAINDER.md
?? docs/refactors/core-model/evidence/UPSTREAM_PIPELINE_STATE.md
?? docs/refactors/core-model/reports/STUDY_stage_lineage.md
?? scripts/status.py
```

| Commit of interest | Revision | Ancestor of HEAD | Ancestor of `origin/main` |
|---|---|---:|---:|
| Phase 1 adapter | `3c0986e6` | commit unavailable | commit unavailable |
| Phase 1 datasets | `96f3991d` | commit unavailable | commit unavailable |
| Phase 1 integration | `c3ca21a2` | commit unavailable | commit unavailable |
| Phase 1 completion | `0431e6d9` | commit unavailable | commit unavailable |
| Rendering defaults restored | `37aeb639` | yes | yes |

## Tests

- Command: `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python -m pytest tests/core -q --color=no --disable-warnings --tb=short -p scripts.status --status-json <status-json>`
- Duration: **6.02s** (limit 6.0s)
- Result: **FAIL** (pytest exit 124)
- Counts: **0 passed · 0 failed · 0 skipped · 0 xfailed · 0 xpassed · 0 not run**

> Pytest exceeded the 6.0s status budget.

## Decisions

- Ledger: `DECISIONS.md`

> **UNVERIFIED DECISIONS: D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25**

**23 decisions · 0 with tests (0 passing) · unverified: [D1, D2, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25]**

> **Ledger divergence:** Canonical `docs/refactors/core-model/DECISIONS.md` has not received the reconciled ledger; using merge-staged `DECISIONS.md`.

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

## Environment

- Python: `3.10.16` at `/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python`
- `pyarrow` imports: **no (ModuleNotFoundError: No module named 'pyarrow')**

| Package | Version |
|---|---|
| `pytest` | `9.0.2` |
| `torch` | `2.5.1` |
| `torchvision` | `0.20.1` |
| `pandas` | `2.2.3` |
| `numpy` | `1.26.4` |
| `pyarrow` | `not installed` |
| `pytorch-lightning` | `2.5.1` |
| `hydra-core` | `1.3.2` |
| `omegaconf` | `2.3.0` |

## Data

Skipped (`--with-data` was not supplied).

## Generator

- Total no-data runtime: **8.15s**
