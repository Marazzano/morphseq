# Ground-truth audit — core-model refactor

**Date:** 2026-08-27
**Repo:** `/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq`
**Branch / HEAD at audit time:** `core-model-refactor` @ `748b6b60`, working tree clean
**Environment for all Python/pytest commands:** `morphseq-env`
(`/net/trapnell/vol1/home/nlammers/micromamba/envs/morphseq-env/bin/python`, Python 3.10.16)

**Method.** Every row below is settled by a command run against the repository and its
output, quoted verbatim. No document was used as evidence for any claim about code. Where a
question could not be answered by measurement, the row says so rather than inferring.

---

## Headline

Three things are true at once, and conflating them is what produced the current mess:

1. **The Phase 1 confabulation is confirmed.** The four commits do not exist in any object
   store or reflog on this machine. Phase 1 was not built.
2. **The two audits are *not* confabulated — they are *stale*.** Every Phase 0 finding
   checked below is exactly correct at `6f6e0f3f^`, at the precise line numbers cited. The
   findings were then fixed by `6f6e0f3f` (2026-08-18), but the audit documents were
   committed six days later (`ec1a24ae`, 2026-08-24) without revalidation. Their file:line
   citations are real; their tense is wrong.
3. **`STATUS.md` is the one document that got this right.** Its generator hardcodes the four
   prose-derived SHAs, but it *does* check them against git and correctly prints
   `commit unavailable` for all four. The generator's real defect is elsewhere: a 6.0-second
   pytest budget that makes every status file report a false test failure.

Two claims in the audit brief itself are also refuted: `5976f8d2` is **not** an ancestor of
`core-model-refactor`, and `794adf46` did **not** merge.

---

## Claim table

| # | Claim | Verdict |
|---|---|---|
| 1 | `src/core` inventory; package imports cleanly | **Partially verified** — 61/78 modules import; 17 fail (15 environmental, 2 real but dead) |
| 2 | Core test suite is 58/58 | **Refuted** — the suite is **5 tests**, 5 passed |
| 3a | `model_configs.py` still imports `data.dataset_configs` | **Refuted (stale)** — now `src.core.data.dataset_configs` |
| 3b | Hydra paths in `training.py` / `training_cluster.py` still absolute or stale | **Refuted (stale)** — relative and valid; `training_cluster.py` is a 19-line retired stub |
| 3c | Run scripts still invoke `src.run.*` | **Refuted (stale)** — all live scripts use `src.core.run.training` |
| 4 | `_pixel_scale` still a literal `128*288` near line 185 | **Refuted (stale)** — computed from `self.input_dim[-2:]` |
| 5a | Margin term still present in metric loss | **Refuted (stale)** — no `margin` anywhere in `src/core` |
| 5b | `accumulate_grad_batches` still in config | **Refuted for the training path (stale)**; still present, and still inert, in 7 LDM/diffusion YAMLs |
| 5c | `metricVAE` lacks the logvar clamp `VAE` has | **Refuted (stale)** — clamp present in both `metricVAE` branches |
| 6 | `contrastive_transform` accepts and ignores `target_size` | **VERIFIED — live bug** |
| 7 | Metric Hydra config sets `dataconfig.target: "BasicDataset"` | **Refuted (stale)** — now `target_name: "NTXentDataset"` |
| 8 | `src/core/data/` still ignored by git | **Refuted** — un-ignored by `b76528eb`; all 5 files tracked |
| 9a | `5976f8d2` is an ancestor of `core-model-refactor` | **REFUTED — the brief is wrong; the original claim was right** |
| 9b | `37aeb639` is on `origin/main` | **Verified** |
| 9c | What is actually unpushed | **Verified** — 1 substantive commit (`794adf46`) + 3 duplicate commits + 2 stale `diffusion-dev` commits |
| 10 | Worktree/branch inventory | **Verified** — 7 worktrees, 10 local branches |
| 11 | `794adf46` records the 8 named rendering parameters and 2 inventory scale columns | **Verified for content; REFUTED for "merged"** |
| 12 | SeaHub µm/px is a hardcoded 7.8 | **Both true** — a real calibration module exists with 7.8 as documented fallback, *and* the API/CLI defaults are a hardcoded 7.8 |
| 13 | `sa_outlier_flag` compares against own track or population | **Verified — population/stage-matched**, confirming the pose-confound tech-debt note |

---

## Code state

### Claim 1 — `src/core` inventory and import health

```
$ find src/core -type f -name '*.py' | wc -l
86
```

86 files, 78 importable modules across 11 subpackages: `data/` (5), `diffusion/` (30),
`functions/` (11), `lightning/` (5), `losses/` (7), `models/` (17), `run/` (6), plus
`config_files/` and `hydra_configs/` (YAML only).

Recursive import sweep (`pkgutil.walk_packages` over `src.core`, each module imported in
isolation):

```
IMPORT OK   : 61
IMPORT FAIL : 17
  FAIL src.core.diffusion.ldm.models.autoencoder
       ImportError: cannot import name 'VectorQuantizer2' from 'taming.modules.vqvae.quantize'
  FAIL src.core.diffusion.ldm.models.diffusion.classifier
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.models.diffusion.ddim
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.models.diffusion.ddpm
       ModuleNotFoundError: No module named 'pytorch_lightning.utilities.distributed'
  FAIL src.core.diffusion.ldm.models.diffusion.plms
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.attention
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.diffusionmodules.model
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.diffusionmodules.openaimodel
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.diffusionmodules.util
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.encoders.modules
       ModuleNotFoundError: No module named 'clip'
  FAIL src.core.diffusion.ldm.modules.image_degradation
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.ldm.modules.losses
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.diffusion.losses
       ModuleNotFoundError: No module named 'ldm'
  FAIL src.core.functions.core_utils_segmentation
       ModuleNotFoundError: No module named 'skimage'
  FAIL src.core.functions.image_utils
       ModuleNotFoundError: No module named 'skimage'
  FAIL src.core.models._extra_files.ldm_model_configs
       ImportError: cannot import name 'SplitArchitectureAELDM' from 'src.core.models.model_components.arch_configs'
  FAIL src.core.run.registry
       NameError: name 'vae_loss_basic' is not defined
```

Classification of the 17 failures:

- **13 environmental, in the vendored `ldm` diffusion tree** — the tree expects a top-level
  `ldm` package, `taming`, `clip`, and a pre-2.0 `pytorch_lightning` API. Not on the core
  training path.
- **2 environmental** — `skimage` is absent from `morphseq-env`.
- **2 genuine in-repo breakage, both in dead code.** Neither is reachable from the training
  path, and neither appears in any planning document:

```
$ python -c "import src.core.run.registry"
  File ".../src/core/run/registry.py", line 3, in <module>
    "VAELossBasic": vae_loss_basic.VAELossBasic,
NameError: name 'vae_loss_basic' is not defined

$ grep -rn 'registry' src/core/run/*.py src/core/models/*.py src/core/lightning/*.py
src/core/run/registry.py:1:# 1) your registry mapping names → classes
```

The only match is `registry.py`'s own comment — nothing imports it. Likewise
`_extra_files/ldm_model_configs.py:9` imports `SplitArchitectureAELDM`, which is commented
out at its definition site:

```
$ grep -rn 'SplitArchitectureAELDM' src/core/
src/core/models/model_configs.py:139:#     ddconfig: SplitArchitectureAELDM = field(default_factory=SplitArchitectureAELDM)
src/core/models/model_components/arch_configs.py:77:# class SplitArchitectureAELDM(ArchitectureAELDM): # adds split logic
src/core/models/_extra_files/ldm_model_configs.py:9:from src.core.models.model_components.arch_configs import SplitArchitectureAELDM, ArchitectureAELDM
```

**Conclusion:** the live training path (`data`, `losses`, `lightning`, `models`, `run`)
imports cleanly. Two orphaned modules are unconditionally broken on import and should be
deleted or repaired; no document mentions either.

### Claim 2 — Core test suite: 5 tests, not 58

```
$ python -m pytest tests/core -v --no-header
============================= test session starts ==============================
collecting ... collected 5 items

tests/core/test_imports.py::CoreImportTests::test_metric_hydra_config_builds_concrete_training_data_config PASSED [ 20%]
tests/core/test_phase0_imports_and_config.py::Phase0ImportAndConfigTests::test_trainer_hardware_can_be_configured_for_cpu PASSED [ 40%]
tests/core/test_phase0_model_smoke.py::Phase0ModelSmokeTests::test_basic_vae_cpu_forward_backward PASSED [ 60%]
tests/core/test_phase0_model_smoke.py::Phase0ModelSmokeTests::test_metric_vae_cpu_forward_backward_with_paired_views PASSED [ 80%]
tests/core/test_phase0_model_smoke.py::Phase0ModelSmokeTests::test_pixel_scale_uses_configured_image_geometry PASSED [100%]

======================== 5 passed in 136.57s (0:02:16) =========================
```

`tests/core/` contains exactly three files:

```
$ find tests/core -type f -name '*.py' | sort
tests/core/test_imports.py
tests/core/test_phase0_imports_and_config.py
tests/core/test_phase0_model_smoke.py
```

**58/58 is refuted.** The real number is 5/5. All five are Phase 0 tests — they cover
config composition, CPU trainer config, and synthetic forward/backward smoke. There is no
Phase 1 test, no manifest test, and no test that touches pipeline output.

Note the runtime: **136.57s**. This matters for the `STATUS.md` defect below.

### Claim 3a — `model_configs.py` data-config import — REFUTED (stale)

```
$ grep -n "import" src/core/models/model_configs.py | head -10
1:from dataclasses import  field, asdict
2:from pydantic.dataclasses import dataclass # as pydantic_dataclass
3:from typing import Any, Literal
4:from src.core.models.model_utils import deep_merge, prune_empty
5:from omegaconf import OmegaConf, DictConfig
6:# from src.losses.legacy_loss_functions import VAELossBasic
7:from src.core.losses.loss_configs import BasicLoss, MetricLoss
8:from src.core.data.dataset_configs import BaseDataConfig, NTXentDataConfig
```

Line 8 is already the corrected path. The audit's claim was true at `6f6e0f3f^`:

```
$ git show 6f6e0f3f^:src/core/models/model_configs.py | grep -n 'dataset_configs'
8:from data.dataset_configs import BaseDataConfig, NTXentDataConfig
```

Same line number, exactly as the integration audit cites (`:299-307`). The audit read real
code; the code then changed.

### Claim 3b — Hydra paths — REFUTED (stale)

```
$ grep -n 'config_path\|config_name\|@hydra' src/core/run/training.py
23:@hydra.main(version_base="1.1",
24:            config_path="../hydra_configs",
25:            config_name="base")

$ grep -n 'config_path\|config_name\|@hydra' src/core/run/training_cluster.py
(no output)

$ grep -rn 'config_path' src/core/ --include=*.py
src/core/run/training.py:24:            config_path="../hydra_configs",
```

`../hydra_configs` is relative and resolves to a directory that exists:

```
$ find src/core/hydra_configs -type f | sort
src/core/hydra_configs/base.yaml
src/core/hydra_configs/base_cluster.yaml
src/core/hydra_configs/base_cluster_metric.yaml
src/core/hydra_configs/model/metric_vae_timm.yaml
src/core/hydra_configs/model/vae_timm.yaml
src/core/hydra_configs/model/vae_timm_no_pips.yaml
```

`training_cluster.py` is now a 19-line retired stub:

```
$ wc -l src/core/run/training_cluster.py
19 src/core/run/training_cluster.py

$ sed -n '1,16p' src/core/run/training_cluster.py
"""Retired compatibility entrypoint for cluster training.

Use the single supported entrypoint and select the former cluster metric config:

    python -m src.core.run.training --config-name base_cluster_metric
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "src.core.run.training_cluster is retired. Use "
        "`python -m src.core.run.training --config-name base_cluster_metric`."
    )
```

The audit's claim was true at `6f6e0f3f^` — an absolute path pointing at `src/hydra_configs`,
a directory that does not exist today, in a 54-line file matching the cited `:32-48`:

```
$ git show 6f6e0f3f^:src/core/run/training_cluster.py | sed -n '30,36p'
OmegaConf.register_new_resolver("ancestor", _abs_ancestor)

@hydra.main(version_base="1.1",
            config_path="/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/hydra_configs",
            config_name="base_cluster_metric")

$ git show 6f6e0f3f^:src/core/run/training_cluster.py | wc -l
54
```

### Claim 3c — run scripts invoking `src.run.*` — REFUTED (stale)

```
$ grep -n 'python\|-m src' src/core/run/run_calls/*.sh
src/core/run/run_calls/20250504_ntxent_squeeze.sh:6:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/20250504_ntxent_squeeze.sh:16:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/20250504_ntxent_squeeze.sh:26:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run01.sh:6:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run01.sh:21:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run02.sh:6:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run02.sh:20:#python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run00.sh:6:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/ntxent_run00.sh:20:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/20250504_ntxent_percep_wt_sweep.sh:6:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/20250504_ntxent_percep_wt_sweep.sh:13:python -m src.core.run.training --config-name base_cluster_metric --run \
src/core/run/run_calls/20250504_ntxent_percep_wt_sweep.sh:19:python -m src.core.run.training --config-name base_cluster_metric --run \
```

Every live invocation uses `src.core.run.training`. Repo-wide, the only `src.run.*` string in
executable code is inside a script that refuses to run:

```
$ cat src/core/run/run_calls/20250504_ldm_run.sh
#!/usr/bin/env bash
set -euo pipefail

echo >&2 "Retired: the referenced src.run.training_ldm_cluster entrypoint and its Hydra config no longer exist."
echo >&2 "This script is retained as provenance only; use src.core.run.training with a supported config."
exit 2
```

True at `6f6e0f3f^`:

```
$ git show 6f6e0f3f^:src/core/run/run_calls/ntxent_run00.sh | grep -n 'python -m'
6:python -m src.run.training_cluster --run \
20:python -m src.run.training_cluster --run \
```

### Claim 4 — `_pixel_scale` literal `128*288` — REFUTED (stale)

```
$ grep -n '_pixel_scale' src/core/losses/loss_functions.py
184:    def _pixel_scale(self) -> float:
229:        pixel_loss_w = pixel_loss.mean() * self._pixel_scale()

$ sed -n '184,190p' src/core/losses/loss_functions.py
    def _pixel_scale(self) -> float:
        height, width = self.input_dim[-2:]
        pixel_count = height * width
        if self.reconstruction_loss != "L1":
            return pixel_count / 100
        else:
            return pixel_count / 10 / 100
```

Geometry is derived from `self.input_dim`, not hardcoded. The literal survives only in
deprecated stacks, none of which is `src/core`:

```
$ grep -rn '128 \* 288\|128\*288' --include=*.py src/
src/vae/models/morph_iaf_vae/morph_iaf_vae_model.py:297:        recon_weight = (128*288) / (recon_x.shape[2]*recon_x.shape[3])
src/vae/models/seq_vae/seq_vae_model.py:304:        recon_weight = (128*288)
src/_Archive/vae/models/_Archive/morph_iaf_vae/morph_iaf_vae_model.py:294:        recon_weight = (128*288) / (recon_x.shape[2]*recon_x.shape[3])
src/_Archive/vae/models/seq_vae/seq_vae_model.py:304:        recon_weight = (128*288)
src/legacy/vae/models/morph_iaf_vae/morph_iaf_vae_model.py:297:        recon_weight = (128*288) / (recon_x.shape[2]*recon_x.shape[3])
src/legacy/vae/models/seq_vae/seq_vae_model.py:304:        recon_weight = (128*288)
```

The audit was right at `6f6e0f3f^`, at exactly the lines it cites (`:185-187`):

```
$ git show 6f6e0f3f^:src/core/losses/loss_functions.py | grep -n -A6 'def _pixel_scale'
183:    def _pixel_scale(self) -> float:
184-        if self.reconstruction_loss != "L1":
185-            return (128 * 288) / 100
186-        else:
187-            return (128 * 288) / 10 / 100
```

There is now a regression test pinning the fix
(`tests/core/test_phase0_model_smoke.py::test_pixel_scale_uses_configured_image_geometry`),
which passed above.

### Claim 5a — margin term in metric loss — REFUTED (stale)

```
$ grep -rn 'margin' src/core/losses/
(no output)

$ grep -rn 'margin' src/core/ --include=*.py | grep -v diffusion
(no output)

$ grep -rn 'margin' src/core/hydra_configs/
(no output)
```

No `margin` field on `MetricLoss` either. The live logit form is affine in distance with no
offset term:

```
$ grep -n 'dist_normed' src/core/losses/loss_functions.py
345:        dist_normed = -(dist_matrix / sigma).pow(0.5) / temperature
372:        return self._nt_xent_loss_multiclass(dist_normed, target_matrix)
```

The audit's cited form existed at `6f6e0f3f^`, at line 342, matching its `:340-342` citation:

```
$ git show 6f6e0f3f^:src/core/losses/loss_functions.py | grep -n 'margin'
342:        dist_normed = (-(dist_matrix / sigma).pow(0.5) + self.cfg.margin) / temperature
```

The audit's substantive point — that the margin was inert because it shifted every logit
uniformly under an affine kernel — was correct, and the term has since been removed. **No
action remains.**

### Claim 5b — `accumulate_grad_batches` — REFUTED for the training path (stale)

Not a field on `LitTrainConfig`, and absent from every Hydra config:

```
$ grep -rn 'accumulate' src/core/lightning/train_config.py src/core/hydra_configs/*.yaml
(no output)
```

It does still appear in seven LDM/diffusion YAMLs:

```
$ grep -rn 'accumulate_grad_batches' --include=*.py --include=*.yaml src/
src/core/config_files/ldm_ae/ldmAE_test.yaml:34:    accumulate_grad_batches: 2
src/core/config_files/ldm_ae/autoencoder_kl_8x8x64.yaml:53:    accumulate_grad_batches: 2
src/core/config_files/ldm_ae/autoencoder_kl_8x8x64_nl.yaml:52:    accumulate_grad_batches: 2
src/core/diffusion/configs/autoencoder/autoencoder_kl_64x64x3.yaml:54:    accumulate_grad_batches: 2
src/core/diffusion/configs/autoencoder/autoencoder_kl_32x32x4.yaml:53:    accumulate_grad_batches: 2
src/core/diffusion/configs/autoencoder/autoencoder_kl_8x8x64.yaml:53:    accumulate_grad_batches: 2
src/core/diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml:54:    accumulate_grad_batches: 2
src/_Archive/vae/test_config.yaml:54:    accumulate_grad_batches: 2
```

and it is read by no Python anywhere in `src/` — so where it survives it remains dead config,
exactly as the audit said:

```
$ grep -rn 'accumulate_grad_batches' --include=*.py src/
(no output — never referenced in any .py under src/)
```

True at `6f6e0f3f^` at the cited line `train_config.py:40`:

```
$ git show 6f6e0f3f^:src/core/lightning/train_config.py | grep -n 'accumulate'
40:    accumulate_grad_batches: int = 2
```

### Claim 5c — `metricVAE` logvar clamp — REFUTED (stale)

Both `metricVAE` branches clamp, identically to `VAE`:

```
$ grep -rn 'clamp' src/core/models/legacy_models.py
38:        mu, logvar = encoder_output.embedding, encoder_output.log_covariance.clamp(min=-10.0, max=5.0)
91:            logvar = encoder_output.log_covariance.clamp(min=-10.0, max=5.0)
107:            logvar = enc.log_covariance.clamp(min=-10.0, max=5.0)  # (2B, D)

$ grep -n '^class ' src/core/models/legacy_models.py
12:class VAE(nn.Module):
59:class metricVAE(nn.Module):
```

Line 38 is in `VAE`; lines 91 and 107 are both inside `metricVAE` (vanilla and NT-Xent
branches respectively). The audit cited `:90` and `:106` as unclamped — off by exactly one
line, consistent with the clamp having been inserted. Confirmed at `6f6e0f3f^`:

```
$ git show 6f6e0f3f^:src/core/models/legacy_models.py | grep -n 'log_covariance'
38:        mu, logvar = encoder_output.embedding, encoder_output.log_covariance.clamp(min=-10.0, max=5.0)
90:            mu, logvar = encoder_output.embedding, encoder_output.log_covariance
106:            logvar = enc.log_covariance  # (2B, D)
```

The fix is pinned by a passing assertion
(`self.assertLessEqual(model_output.logvar.max().item(), 5.0)` in
`test_metric_vae_cpu_forward_backward_with_paired_views`, with the test encoder deliberately
emitting `logvar_value=20.0`).

### Claim 6 — `contrastive_transform` ignores `target_size` — **VERIFIED, live bug**

```
$ sed -n '12,38p' src/core/data/data_transforms.py
def contrastive_transform(target_size=None):  # (size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(brightness=0.3)
    data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.RandomAffine(degrees=15, scale=tuple([0.7, 1.3])),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          # transforms.RandomGrayscale(p=0.2),
                                          # GaussianBlur(kernel_size=5),
                                          transforms.ToTensor()])
    return data_transforms


def basic_transform(target_size=None):
    if target_size is not None:
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((target_size[0], target_size[1])),
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    return data_transform
```

`target_size` is accepted at line 12 and referenced nowhere in the body — no `Resize`, no
mention. `basic_transform` (line 26) *does* honour it. Callers pass it in good faith:

```
$ grep -rn 'target_size' src/core/ --include=*.py --include=*.yaml
src/core/run/run_utils.py:472:    target_size = (input_dim[1], input_dim[2])
src/core/run/run_utils.py:484:            transform_kwargs={"target_size": target_size},
src/core/run/run_utils.py:491:            transform_kwargs={"target_size": target_size},
src/core/data/data_transforms.py:12:def contrastive_transform(target_size=None):  # (size, s=1):
src/core/data/data_transforms.py:26:def basic_transform(target_size=None):
src/core/data/data_transforms.py:27:    if target_size is not None:
src/core/data/data_transforms.py:30:            transforms.Resize((target_size[0], target_size[1])),
```

**This is the one Phase 0 finding that is still live.** Any metric/contrastive run at an input
size other than the images' native size silently trains on unresized images. It is not covered
by any test in `tests/core`.

### Claim 7 — metric Hydra `dataconfig.target` — REFUTED (stale)

```
$ grep -n 'target' src/core/hydra_configs/model/metric_vae_timm.yaml
1:config_target: src.core.models.model_configs.metricVAEConfig
41:  target_name: "NTXentDataset"
```

The key was renamed and the value corrected. Both changes are visible in history:

```
$ git log -p --oneline -3 -- src/core/hydra_configs/model/metric_vae_timm.yaml | grep -E '^(commit|[+-].*target)'
-  target: "BasicDataset"
+  target_name: "NTXentDataset"
-config_target: src.models.model_configs.metricVAEConfig
+config_target: src.core.models.model_configs.metricVAEConfig
```

Pinned by a passing test asserting
`model_cfg.dataconfig.target_name == "NTXentDataset"` and
`type(model_cfg.dataconfig) is NTXentDataConfig`.

### Claim 8 — `src/core/data/` gitignored — REFUTED

```
$ git check-ignore -v src/core/data/dataset_classes.py
exit=1
$ git check-ignore -v src/core/data/
exit=1
$ git check-ignore -v src/core/data/pipeline_contracts.py
exit=1
```

(`git check-ignore` exit 1 = not ignored.) All five files are tracked:

```
$ git ls-files src/core/data/
src/core/data/__init__.py
src/core/data/data_transforms.py
src/core/data/dataset_classes.py
src/core/data/dataset_configs.py
src/core/data/dataset_utils.py
```

The rule was removed by `b76528eb`:

```
$ git show b76528eb -- .gitignore
@@ -81,7 +81,6 @@ src/build/__pychache__/
 # Ignore __pycache__ directories (recursively)
 **/__pycache__/
-src/core/data/
 .aider*
```

This is the fix the three worktree agents each rediscovered independently — see Claim 10.

---

## Git state

### Claim 9a — is `5976f8d2` an ancestor of `core-model-refactor`? — **NO**

The audit brief asserts this claim "also appears false — that commit is in the history."
**That is wrong.** The commit object is *resolvable*, which is not the same as being an
ancestor:

```
$ git merge-base --is-ancestor 5976f8d2 core-model-refactor; echo "exit=$?"
exit=1

$ git log --oneline -1 5976f8d2
5976f8d2 Merge pull request #31 from nlammers371/feat/fp-snip-products

$ git branch --contains 5976f8d2 -a
  remotes/origin/HEAD -> origin/main
  remotes/origin/main

$ git log --oneline --ancestry-path 5976f8d2..core-model-refactor
(no output)
```

Exit 1 from `--is-ancestor` means not an ancestor. `git branch --contains` lists only
`origin/main` — `core-model-refactor` is absent. The merge base is far older:

```
$ git merge-base 5976f8d2 core-model-refactor
14c8ab1089ec9b70b809d53694bc6bb463f2a3cd

$ git log --oneline -1 14c8ab10
14c8ab10 More figures around cross-modal prediction
```

The topology makes the trap visible — `core-model-refactor` branched from the left-hand
lineage at `a37a3441`/`14c8ab10`, *before* PR 31 merged on the right:

```
$ git log --oneline --graph origin/main -12
*   7b028916 Merge branch 'feat/fp-snip-products'
|\
| * 09f79707 CORRECTION: the trailing-blank H label IS repaired, and my rewrite was wrong
* | 71b4e741 Merge branch 'feat/fp-snip-products'
|\|
| * 8b755251 Re-express main's ND2 indexing tests against the surviving API
* | 5976f8d2 Merge pull request #31 from nlammers371/feat/fp-snip-products
|\|
| *   48fdbd92 Merge main (SeaHub + 9 upstream commits) into feat/fp-snip-products
| |\
| |/
|/|
* |   a37a3441 Merge remote-tracking branch 'origin/main'
|\ \
| * | 14c8ab10 More figures around cross-modal prediction
| * | 409d9c83 Patch to resolve issues with SeaHUB embryo image extraction. Still needs testing.
| * | 9c47e248 Plots on plots
| * | 37aeb639 Added logic to handle SeaHUB scaling. Lots of preliminary work around GENE 7 seq-morph integration as well
```

`core-model-refactor` is **194 commits behind `origin/main`**:

```
$ git rev-list --left-right --count origin/main...core-model-refactor
194	15
```

**Consequence:** the original claim was correct and was wrongly dismissed. All PR-31 snip
rendering work — everything merged into `origin/main` after `14c8ab10` — is absent from the
branch the core refactor is being built on. Any statement of the form "the rendering fix is in
our ancestry" needs rechecking against this.

### Claim 9b — is `37aeb639` on `origin/main`? — **YES**

```
$ git merge-base --is-ancestor 37aeb639 origin/main; echo "exit=$?"
exit=0

$ git merge-base --is-ancestor 37aeb639 core-model-refactor; echo "exit=$?"
exit=0

$ git log --oneline -1 37aeb639
37aeb639 Added logic to handle SeaHUB scaling. Lots of preliminary work around GENE 7 seq-morph integration as well
```

On both branches. This one checks out. Note `37aeb639` predates the branch point, which is
why it is present while `5976f8d2` is not.

### Claim 9c — what is actually unpushed

```
$ git log --oneline --branches --not --remotes
794adf46 Record snip rendering provenance
5d9df0f8 Track core training data package
8c57cd26 Track core training data package
eed13d84 Track core training data package
cc0cec4d Example figure
b01f896b Added p val option to plotting code. Added custom notebook for pbx analysis

$ git rev-list --left-right --count origin/core-model-refactor...core-model-refactor
0	0
```

- `core-model-refactor` is **exactly in sync** with its upstream.
- `main` reports `[ahead 4, behind 194]`, but those four commits are already published via
  `origin/core-model-refactor`, so they are not at risk.
- **Only one substantive unpushed commit exists: `794adf46`** (851 insertions of snip
  provenance code).
- The three `Track core training data package` commits are byte-identical duplicates of the
  already-merged `b76528eb` — same tree, same parent:

```
$ for c in 8c57cd26 5d9df0f8 eed13d84 b76528eb; do echo "$c  tree=$(git rev-parse $c^{tree})  parent=$(git rev-parse --short $c^)"; done
8c57cd26  tree=3dcdf349904c4f42202a5b8eb7ad5b50528c5f2c  parent=a24ca2d3
5d9df0f8  tree=3dcdf349904c4f42202a5b8eb7ad5b50528c5f2c  parent=a24ca2d3
eed13d84  tree=3dcdf349904c4f42202a5b8eb7ad5b50528c5f2c  parent=a24ca2d3
b76528eb  tree=3dcdf349904c4f42202a5b8eb7ad5b50528c5f2c  parent=a24ca2d3
```

- `cc0cec4d` / `b01f896b` are old `diffusion-dev` commits, unrelated to this refactor.

### The four Phase 1 commits do not exist

```
$ for c in 3c0986e6 96f3991d c3ca21a2 0431e6d9; do echo "--- $c ---"; git cat-file -t $c; done
--- 3c0986e6 ---
fatal: Not a valid object name 3c0986e6
--- 96f3991d ---
fatal: Not a valid object name 96f3991d
--- c3ca21a2 ---
fatal: Not a valid object name c3ca21a2
--- 0431e6d9 ---
fatal: Not a valid object name 0431e6d9
```

Absent from every reflog, including all six per-worktree reflogs:

```
$ git reflog --all | wc -l
401

$ find .git -name HEAD -path '*logs*'
.git/logs/HEAD
.git/worktrees/morphseq-snip-provenance.nobIjL/logs/HEAD
.git/worktrees/morphseq-audit-regen-scope/logs/HEAD
.git/worktrees/morphseq-phase1b/logs/HEAD
.git/worktrees/morphseq-phase1-accept/logs/HEAD
.git/worktrees/morphseq-phase1c/logs/HEAD
.git/worktrees/morphseq-phase1a/logs/HEAD

$ for c in 3c0986e6 96f3991d c3ca21a2 0431e6d9; do echo "--- $c ---"; grep -rl "$c" .git/logs || echo "  (no hit in .git/logs)"; done
--- 3c0986e6 ---
  (no hit in .git/logs)
--- 96f3991d ---
  (no hit in .git/logs)
--- c3ca21a2 ---
  (no hit in .git/logs)
--- 0431e6d9 ---
  (no hit in .git/logs)
```

401 reflog entries were searched. **Phase 1 was not built.** Confirmed.

### Claim 10 — worktree and branch inventory

```
$ git worktree list
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq                    748b6b60 [core-model-refactor]
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-audit-regen-scope  3ecc4818 [audit/regen-scope]
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-phase1-accept      3ecc4818 [thread/phase1-accept]
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-phase1a            8c57cd26 [slice/phase1a]
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-phase1b            5d9df0f8 [slice/phase1b]
/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq-phase1c            eed13d84 [slice/phase1c]
/tmp/morphseq-snip-provenance.nobIjL                                               794adf46 [thread/snip-provenance]
```

| Worktree | Branch | Ahead of `core-model-refactor` | Uncommitted / untracked |
|---|---|---|---|
| `morphseq` (primary) | `core-model-refactor` @ `748b6b60` | — | clean |
| `morphseq-phase1a` | `slice/phase1a` @ `8c57cd26` | `8c57cd26` (dup of `b76528eb`) | `?? src/core/data/pipeline_contracts.py` (348 lines, untracked) |
| `morphseq-phase1b` | `slice/phase1b` @ `5d9df0f8` | `5d9df0f8` (dup) | ` D src/core/data/dataset_classes.py` (uncommitted deletion) |
| `morphseq-phase1c` | `slice/phase1c` @ `eed13d84` | `eed13d84` (dup) | clean |
| `morphseq-phase1-accept` | `thread/phase1-accept` @ `3ecc4818` | none | clean |
| `morphseq-audit-regen-scope` | `audit/regen-scope` @ `3ecc4818` | none | `?? docs/refactors/core-model/reports/AUDIT_regeneration_scope.md` |
| `/tmp/morphseq-snip-provenance.nobIjL` | `thread/snip-provenance` @ `794adf46` | `794adf46` | clean |

Verbatim per-worktree status:

```
$ git -C .../morphseq-phase1a status --porcelain=v1 -uall
?? src/core/data/pipeline_contracts.py

$ git -C .../morphseq-phase1b status --porcelain=v1 -uall
 D src/core/data/dataset_classes.py

$ git -C .../morphseq-phase1c status --porcelain=v1 -uall
(no output)

$ git -C .../morphseq-phase1-accept status --porcelain=v1 -uall
(no output)

$ git -C .../morphseq-audit-regen-scope status --porcelain=v1 -uall
?? docs/refactors/core-model/reports/AUDIT_regeneration_scope.md
```

This exactly matches the brief's description and corroborates the write-into-gitignored-dir
diagnosis: three agents each independently produced the *same* `.gitignore` fix (identical
tree hash and parent), which is what you would expect if each one wrote files into
`src/core/data/`, found git blind to them, and fixed the ignore rule. The only Phase 1
artifact that survived is one untracked 348-line file:

```
$ wc -l .../morphseq-phase1a/src/core/data/pipeline_contracts.py
348 .../morphseq-phase1a/src/core/data/pipeline_contracts.py

$ head -8 .../morphseq-phase1a/src/core/data/pipeline_contracts.py
"""Schema declarations for pipeline artifacts consumed by the core manifest.

These declarations intentionally describe the compatibility boundary observed by core.  They do
not import :mod:`src.data_pipeline`; the manifest adapter is the only module allowed to do that.
Validation is diagnostic and returns a complete report instead of failing on the first schema
difference.
"""
```

`morphseq-phase1b` additionally holds an **uncommitted deletion of `dataset_classes.py`** —
a destructive, unreviewed change sitting in a worktree. It should be restored or the worktree
discarded before anything else touches that tree.

Local branches (10) and their tracking state:

```
$ git for-each-ref --format='%(refname:short) | %(objectname:short) | %(upstream:short) | %(upstream:track)' refs/heads
audit/regen-scope | 3ecc4818 |  |
core-model-refactor | 748b6b60 | origin/core-model-refactor |
diffusion-dev | cc0cec4d | origin/diffusion-dev | [ahead 1157, behind 874]
main | 183903c3 | origin/main | [ahead 4, behind 194]
refactor_20231201 | 563cfd3c | origin/refactor_20231201 |
slice/phase1a | 8c57cd26 |  |
slice/phase1b | 5d9df0f8 |  |
slice/phase1c | eed13d84 |  |
thread/phase1-accept | 3ecc4818 |  |
thread/snip-provenance | 794adf46 |  |
```

Six branches have no upstream at all.

---

## Pipeline state

### Claim 11 — `794adf46` snip rendering provenance

**Part 1 — did it merge? NO.** The brief calls this "the one piece of last night's work that
merged." It did not:

```
$ git merge-base --is-ancestor 794adf46 core-model-refactor; echo "exit=$?"
exit=1

$ git log --oneline core-model-refactor..thread/snip-provenance
794adf46 Record snip rendering provenance

$ git ls-tree core-model-refactor -- src/data_pipeline/object_extraction/snip_processing/provenance.py
(no output — file absent on core-model-refactor)

$ git ls-tree thread/snip-provenance -- src/data_pipeline/object_extraction/snip_processing/provenance.py
100644 blob 7c5e51a7abb6ee688a15341f40a07420c0721bfe	src/data_pipeline/object_extraction/snip_processing/provenance.py
```

It lives only on `thread/snip-provenance`, in a worktree under **`/tmp`** — the single most
at-risk piece of unpushed work in the repository. Its scope:

```
$ git show --stat 794adf46
commit 794adf462f3bcac20c0da0c45d50fadaaecbc9bf
Author: nlammers371 <nlammers@t003.grid.gs.washington.edu>
Date:   Mon Aug 24 22:23:40 2026 -0700

    Record snip rendering provenance

 .../snip_processing/augmentation.py                |   8 +-
 .../object_extraction/snip_processing/contract.py  |  15 +-
 .../entrypoints/run_snip_processing.py             | 154 +++++++++-
 .../snip_processing/inventory_contract.py          | 198 ++++++++++++
 .../object_extraction/snip_processing/io.py        |  27 +-
 .../object_extraction/snip_processing/ops.py       |   2 +
 .../pipelines/merge_snip_manifests.py              |  19 +-
 .../snip_processing/pipelines/snip_processing.py   | 114 ++++++-
 .../snip_processing/provenance.py                  | 331 +++++++++++++++++++++
 9 files changed, 851 insertions(+), 17 deletions(-)
```

**Part 2 — does it record what it claims? YES, all eight.** `build_rendering_contract`
emits every named parameter:

```
$ sed -n '71,143p' .../snip_processing/provenance.py
# [abridged: lines 82-99 (docstring + frame_shape unpacking) and the frame_shape
#  dict body elided at the two `...` markers; all other lines verbatim]
def build_rendering_contract(
    *,
    target_micrometers_per_pixel: float,
    blend_radius_micrometers: float,
    frame_shape: tuple[int, int],
    mask_contract: dict[str, Any],
    orientation_contract: dict[str, Any],
    clahe_enabled: bool,
    background_contract: dict[str, Any],
    file_encoding: dict[str, Any],
) -> dict[str, Any]:
    ...
        "rendering": {
            "target_micrometers_per_pixel": float(target_micrometers_per_pixel),
            "blend_radius_micrometers": float(blend_radius_micrometers),
            "mask": mask_contract,
            "orientation": orientation_contract,
            "clahe": {
                "enabled": bool(clahe_enabled),
                "implementation": "skimage.exposure.equalize_adapthist",
                "call_overrides": {},
                "resolved_library_defaults": _call_defaults(
                    skimage.exposure.equalize_adapthist
                ),
            },
            "background": background_contract,
            "frame_shape": {...},
            "resampling": {
                "physical_scale": {
                    "image": {
                        "implementation": "skimage.transform.rescale",
                        "order": 1,
                        "kernel": "linear",
                        "mode": "reflect",
                        "preserve_range": True,
                        "anti_aliasing": "library_auto",
                    },
                    "embryo_and_yolk_masks": {
                        "implementation": "skimage.transform.resize",
                        "order": 1,
                        "kernel": "linear",
                        "mode": "reflect",
                        "anti_aliasing": "library_auto",
                    },
                },
                "rotation": {
                    "implementation": "cv2.warpAffine",
                    "kernel": "INTER_LINEAR",
                    "border_mode": "BORDER_CONSTANT",
                    "border_value": 0,
                },
            },
            "file_encoding": file_encoding,
        },
```

| Required item | Field | Present |
|---|---|---|
| target µm/px | `rendering.target_micrometers_per_pixel` | yes |
| blend radius | `rendering.blend_radius_micrometers` | yes |
| mask source | `rendering.mask` (+ `observed_mask_sources`, source/version columns) | yes |
| orientation policy | `rendering.orientation` (PCA major-axis + 180° disambiguation, evidence source) | yes |
| CLAHE | `rendering.clahe` (enabled, impl, overrides, resolved library defaults) | yes |
| background model | `rendering.background` (model, mode, sampling, noise dist + seed) | yes |
| resampling kernel | `rendering.resampling` (physical scale + rotation, order/kernel/mode) | yes |
| encoding | `rendering.file_encoding` | yes |

Also captured: `contract_version`, `git` identity, and `numpy`/`scipy`/`scikit-image`/`opencv`
versions. It is genuinely wired, not dead code, and called with live values:

```
$ grep -rn 'build_rendering_contract' .../src/data_pipeline/ | grep -v 'provenance.py'
.../entrypoints/run_snip_processing.py:47:    build_rendering_contract,
.../entrypoints/run_snip_processing.py:237:    rendering_contract = build_rendering_contract(
.../pipelines/snip_processing.py:34:    build_rendering_contract,
.../pipelines/snip_processing.py:232:    rendering_contract = build_rendering_contract(

$ sed -n '232,241p' .../pipelines/snip_processing.py
    rendering_contract = build_rendering_contract(
        target_micrometers_per_pixel=target_pixel_size_um,
        blend_radius_micrometers=blend_radius_um,
        frame_shape=output_shape_hw,
        mask_contract={
            "artifact": "segmentation_tracking.exported_mask_path",
            "source_column": "source_backend",
            "version_columns": ["source_model", "model_release"],
        },
```

**Part 3 — the two inventory scale columns: YES, and this is the substantive fix.**

```
$ grep -n 'micrometers_per_pixel' .../snip_processing/inventory_contract.py
33:    "source_micrometers_per_pixel",
34:    "snip_micrometers_per_pixel",
114:        "source_micrometers_per_pixel",
115:        "snip_micrometers_per_pixel",
134:                and df.loc[valid, "source_micrometers_per_pixel"].isna().any()
138:                    f"{scope_label}: source_micrometers_per_pixel is required for every "
```

The columns existed by name before, but `source_micrometers_per_pixel` was written as a
literal `None`. On `core-model-refactor`:

```
$ sed -n '254,255p' src/data_pipeline/object_extraction/snip_processing/entrypoints/run_snip_processing.py
            "source_micrometers_per_pixel": None,
            "snip_micrometers_per_pixel": float(target_pixel_size_um),
```

After `794adf46`, it carries a real value joined from the frame contract, and a null is a
hard error:

```
$ grep -n 'micrometers_per_pixel' .../pipelines/snip_processing.py
81:        usecols=["image_id", "image_micrometers_per_pixel", "well_index", "well_id", "time_int"],
87:    if merged["image_micrometers_per_pixel"].isna().any():
96:            "Missing image_micrometers_per_pixel after join with frame_contract for rows: "
172:                    "source_micrometers_per_pixel": float(row["image_micrometers_per_pixel"]),
173:                    "snip_micrometers_per_pixel": float(target_pixel_size_um),
```

This directly closes the gap `PIPELINE_RECON` measured ("all 133 readable inventories lack
both requested pixel-scale fields") — **for snips rendered from here on.** It is retrospective
for nothing. And because it is unmerged and parked in `/tmp`, it currently closes nothing at
all.

### Claim 12 — SeaHub µm/px: a real calibration path *and* a hardcoded 7.8

Both halves of the question are true, in different places.

**There is a real calibration module** — `src/data_pipeline/acquisition/seahub/scale_calibration.py`:

```
$ sed -n '1,17p' src/data_pipeline/acquisition/seahub/scale_calibration.py
"""Conservative source-FOV pixel-scale calibration for SeaHub images.

SeaHub source images contain eight embryos in one field of view (FOV).  Scale is therefore
estimated once for the FOV and must be broadcast unchanged to all embryos cropped from it.  The
estimate uses the median area of valid embryo masks and a versioned, stage-specific strict-WT
surface-area reference::

    raw_um_per_px = sqrt(strict_wt_p50_area_um2 / median_valid_mask_area_px)

The raw value is deliberately only half-weighted relative to the stage prior, then constrained to
within 20% of that prior and to the absolute interval [3.5, 9.0] um/px.  FOVs with fewer than four
valid masks use the stage prior without attempting a mask-derived estimate.  An unresolved or
unreferenced stage retains the explicit 7.8 um/px global fallback so reconciliation's intentional
stage-failure pass-through remains processable.
"""
```

Its constants:

```
$ sed -n '30,36p' src/data_pipeline/acquisition/seahub/scale_calibration.py
DEFAULT_REFERENCE_VERSION = "v1"
ABSOLUTE_MIN_UM_PER_PX = 3.5
ABSOLUTE_MAX_UM_PER_PX = 9.0
RELATIVE_BOUND_FRACTION = 0.20
MIN_VALID_MASKS = 4
RAW_SCALE_WEIGHT = 0.50
GLOBAL_FALLBACK_UM_PER_PX = 7.8
```

**The assignment site** — 7.8 is reached only on the unresolved/unreferenced-stage branch, and
it is labelled as a placeholder in the output row:

```
$ sed -n '396,413p' src/data_pipeline/acquisition/seahub/scale_calibration.py
        if reference_row is None:
            prior = np.nan
            strict_wt_p50 = np.nan
            raw_scale = np.nan
            applied_raw_weight = 0.0
            unbounded = float(config.global_fallback_um_per_px)
            status = (
                "global_fallback_unresolved_stage"
                if not np.isfinite(stage_hpf)
                else "global_fallback_unreferenced_stage"
            )
            method = "global_fallback"
            issue = (
                "unresolved_stage"
                if not np.isfinite(stage_hpf)
                else f"stage_not_in_reference:{stage_hpf:g}"
            )
            reference_source = "global_placeholder_7.8_um_per_px"
```

The real branch:

```
$ sed -n '421,430p' src/data_pipeline/acquisition/seahub/scale_calibration.py
        elif n_valid_masks >= int(config.min_valid_masks):
            prior = float(reference_row["stage_prior_um_per_px"])
            strict_wt_p50 = float(reference_row["strict_wt_p50_area_um2"])
            raw_scale = float(np.sqrt(strict_wt_p50 / median_area))
            applied_raw_weight = float(config.raw_scale_weight)
            unbounded = prior + applied_raw_weight * (raw_scale - prior)
            status = "mask_area_regularized"
            method = "strict_wt_p50_over_fov_median_mask_area"
```

**But hardcoded 7.8 defaults also survive on the API and CLI surfaces:**

```
$ sed -n '109,115p' src/data_pipeline/acquisition/seahub/integration.py
@dataclass(frozen=True)
class SeaHubIntegrationConfig:
    """Frozen operational choices for a SeaHub integration bundle."""

    operational_date: str = "20260723"
    micrometers_per_pixel: float = 7.8

$ grep -n 'micrometers-per-pixel' src/data_pipeline/acquisition/seahub/cli.py
213:    parser.add_argument("--micrometers-per-pixel", type=float, default=7.8)
```

The code claims production cannot silently take that path:

```
$ sed -n '395,400p' src/data_pipeline/acquisition/seahub/integration.py  # docstring of _validated_fov_scale_calibration, defined at :389
    """Return one auditable scale row for every included source FOV.

    Direct API callers may omit calibration and retain the explicit legacy 7.8
    placeholder.  The production CLI requires a mask manifest and supplies the
    provisional mask-derived table, so production cannot silently take this path.
    """
```

and the CLI does wire the calibrator in:

```
$ grep -n 'calibrate_reconciled_source_fovs\|fov_scale_calibration=' src/data_pipeline/acquisition/seahub/cli.py
21:from .scale_calibration import calibrate_reconciled_source_fovs
103:    scale_calibration = calibrate_reconciled_source_fovs(
113:        fov_scale_calibration=scale_calibration,
137:    scale_calibration = calibrate_reconciled_source_fovs(
147:        fov_scale_calibration=scale_calibration,
```

**Accurate summary:** SeaHub µm/px is *not* a bare hardcoded 7.8. There is a real, versioned,
bounded, per-FOV calibration path, and every row carries a `calibration_status` /
`calibration_method` saying which branch produced it. 7.8 remains as (a) a documented global
fallback for unresolved/unreferenced stages, and (b) a hardcoded default on the direct-API and
CLI argument surfaces. **Which branch actually produced any given stored SeaHub corpus cannot
be determined from source — it is recorded per-row in `fov_scale_calibration.csv`
(`integration.py:1429-1430`) and must be read from the data.** I did not open any produced
bundle, so I am not asserting what the stored corpus used.

### Claim 13 — `sa_outlier_flag` compares against a population reference, not the embryo's own track — **VERIFIED**

```
$ sed -n '146,161p' src/data_pipeline/quality_control/surface_area_outlier_detection.py
    # Interpolate reference values at each embryo's stage
    # Reference curves are pre-filled/extrapolated, so simple interp is sufficient
    df['_ref_p5'] = np.interp(df[stage_col], ref_df['stage_hpf'], ref_df['p5'])
    df['_ref_p95'] = np.interp(df[stage_col], ref_df['stage_hpf'], ref_df['p95'])

    # Calculate thresholds
    upper_threshold = k_upper * df['_ref_p95']
    lower_threshold = k_lower * df['_ref_p5']

    # Two-sided flagging
    too_large = df[sa_col] > upper_threshold
    too_small = df[sa_col] < lower_threshold
    df['sa_outlier_flag'] = too_large | too_small
```

The comparison is against `ref_df['p5']` / `ref_df['p95']` — percentile bands of an external,
pre-built, stage-indexed **population** reference, interpolated at each frame's own
`stage_hpf`. There is no groupby on embryo or track, and no reference to the embryo's own
history anywhere in the function. It is a **population, stage-matched** comparison, applied
per-frame and independently.

The required reference columns confirm the reference is a population curve:

```
$ grep -n "required_cols = " src/data_pipeline/quality_control/surface_area_outlier_detection.py
138:    required_cols = ['stage_hpf', 'p5', 'p95']
```

**This is exactly the condition
`docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md` describes.**
The doc exists and is directly on point:

```
$ ls -la docs/data_pipeline/specs/target/specs/tech_debt/surface_area_qc_pose_confound.md
-rw-r--r-- 1 nlammers nlammers_g 7109 Jul  8 11:53 .../surface_area_qc_pose_confound.md
```

Because each frame is judged against a population band rather than the embryo's own
trajectory, anything that changes apparent projected area without changing the embryo —
pose, roll, orientation, focus — moves a frame toward the flag. A within-track comparison
would be largely immune to this; the population comparison is not. This also compounds the
known temperature bias in `k_lower=0.9`.

Two additional facts worth recording, since they bear on how much weight this flag can carry:

```
$ grep -n 'sa_outlier_flag' src/data_pipeline/quality_control/applicability.py
20:    "sa_outlier_flag": "surface_area_qc_applicability",

$ sed -n '28,32p' src/data_pipeline/quality_control/surface_area_outlier_detection.py
``src/build/build04_perform_embryo_qc.py`` still calls ``compute_sa_outlier_flag`` to produce the
legacy ``sa_outlier_flag`` ground truth the legacy-drift gate compares against. Do NOT import it
```

---

## Cross-check of `reports/PIPELINE_RECON.md`

Three headline figures, re-derived from the shipped CSVs rather than from the prose:

```
$ python - <<'EOF'
import pandas as pd
q = pd.read_csv("qc_availability_by_experiment.csv")
s = pd.read_csv("snips_by_experiment.csv")
print("present==True :", int(q['present'].sum()))
print("readable==True:", int(q['readable'].sum()))
no_qc = q.loc[~q['present'].astype(bool), 'experiment_id']
inv = set(s['experiment_id'])
overlap = [e for e in no_qc if e in inv]
print("experiments with present==False:", len(no_qc))
print("...of which inventory-bearing:", len(overlap))
print("sum of snips for those:", int(s.loc[s['experiment_id'].isin(overlap),'snips'].sum()))
print("total snips:", int(s['snips'].sum()))
EOF
present==True : 105
readable==True: 105
experiments with present==False: 43
...of which inventory-bearing: 28
sum of snips for those: 167603
total snips: 699505
```

```
$ python -c "import pandas as pd; print(pd.read_csv('qc_availability_by_experiment.csv')['experiment_id'].nunique())"
148
```

| RECON claim | Re-derived | Match |
|---|---|---|
| "148 deduplicated experiment IDs" | 148 unique `experiment_id` | yes |
| "105/105 present files opened successfully" | present=105, readable=105 | yes |
| "28 inventory-bearing experiments (167,603 snips) have no QC artifact" | 28 / 167,603 | yes |
| "43 incomplete experiments" (per `TRAINING_READINESS_REMAINDER` §3) | 43 with `present==False` | yes |
| "699,505 audited snips" (per `PLAN.md` §2) | 699,505 | yes |

**`PIPELINE_RECON.md` and its tables reproduce exactly.** The brief's assessment of this
document is correct; it is the one to build on. Caveat on scope, not accuracy: it was
generated on 2026-08-18/19 from a Mac mount
(`/media/nick/gs_cluster/projects/data/morphseq/pipeline/output`), so it is a snapshot of
that date, and I did not re-measure against live data on the cluster.

---

## Documents to correct or retire

### 1. `evidence/TRAINING_READINESS_REMAINDER.md` — **RETIRE**, do not edit

The load-bearing factual claims are fabricated:

| Statement in doc | Truth |
|---|---|
| "inspected ... at `0431e6d9`" | commit does not exist (`fatal: Not a valid object name`) |
| "**58/58 tests**" | suite is 5 tests; 5 passed |
| "Move commits `3c0986e6`, `96f3991d`, `c3ca21a2`, `0431e6d9`" | none exist in any object store or reflog |
| "The Phase 1 bridge is substantially implemented and unit-tested" | no Phase 1 code is committed anywhere; one untracked 348-line file survives |
| "Re-run the 58 core tests after synchronization" | there is nothing to synchronize |
| "local `origin/core-model-refactor`: `b76528eb`" | `origin/core-model-refactor` is at `748b6b60` |

Retire it rather than patch it: the §2–§5 planning content (corpus acceptance, cohort freeze,
science-policy artifacts, acceptance-run ordering) is genuinely useful and is *not* derived
from the fabricated commits — it should be lifted into a fresh document that carries no claim
about implementation state. **Note one item in it is now independently confirmed true:** "all
133 readable inventories lack both requested pixel-scale fields" matches the RECON tables and
is exactly what `794adf46` addresses going forward.

### 2. `PLAN.md` §1 — **CORRECT** (the header claim is false)

> "**The code is essentially done. The corpus is not.**"
> "Phase 0 and Phase 1 are implemented and unit-tested — 58/58 core tests pass at `0431e6d9`"

Phase 1 does not exist. The sentence should read, approximately: *Phase 0 is done and
regression-tested at 5/5; Phase 1 is unstarted; the corpus is unresolved.* §1's claim that
"what remains on the code side is four small hardenings, named run configs, and one real
end-to-end acceptance run" understates the remaining work by the entire Phase 1 data layer.

Note that PLAN.md correctly attributes the claim (*"per `evidence/TRAINING_READINESS_REMAINDER.md`,
2026-08-24"*) — the citation discipline worked; it was the cited document that lied.
§2's corpus table and figures cross-check clean against RECON and should be kept.

One additional correction for §2: it says the scale/blend fix was "restored `37aeb639`,
2026-08-02". `37aeb639` is on both `origin/main` and `core-model-refactor` — but everything
merged after `14c8ab10`, **including the entire PR-31 snip rendering line (`5976f8d2`)**, is
*not* in this branch's ancestry (Claim 9a). Any "the fix is in our ancestry" reasoning must be
re-derived per-commit.

### 3. `STATUS.md` and `scripts/status.py` — **CORRECT the generator; the output is mostly sound**

The brief's hypothesis is half right, and the half it gets wrong is worth stating plainly:
**STATUS.md did not launder the confabulation.** Its generator checks the SHAs against git and
correctly reports them as unavailable:

```
$ grep -n -A8 'def ancestor_status' scripts/status.py
193:def ancestor_status(revision: str, target: str) -> str:
194-    if not revision_exists(revision):
195-        return "commit unavailable"
```

which produced, in `STATUS.md`:

| Commit of interest | Revision | Ancestor of HEAD | Ancestor of `origin/main` |
|---|---|---:|---:|
| Phase 1 adapter | `3c0986e6` | commit unavailable | commit unavailable |
| Phase 1 datasets | `96f3991d` | commit unavailable | commit unavailable |
| Phase 1 integration | `c3ca21a2` | commit unavailable | commit unavailable |
| Phase 1 completion | `0431e6d9` | commit unavailable | commit unavailable |

**The evidence that Phase 1 did not exist was sitting in a generated file since 2026-08-25.**
Nobody read it.

Two real defects remain:

**(a) The pytest budget produces a permanent false failure.** `TEST_TIMEOUT_SECONDS = 6.0`
(`scripts/status.py:46`), and on timeout the generator synthesizes `exit_code = 124`
(`:290`). The suite actually takes **136.57s**, so every status file reports:

> - Duration: **6.02s** (limit 6.0s)
> - Result: **FAIL** (pytest exit 124)
> - Counts: **0 passed · 0 failed · 0 skipped**

A generated status file that always says FAIL/0-tests trains readers to ignore it — which is
plausibly *why* the `commit unavailable` rows went unread. Raise the budget past the real
runtime, and distinguish "timed out" from "failed" in the rendered result.

**(b) `COMMITS_OF_INTEREST` is a hardcoded prose-derived literal** (`scripts/status.py:53-59`).
The verdict column is computed, but the *input list* is transcribed from
`TRAINING_READINESS_REMAINDER.md`. That is the narrow sense in which the brief's concern is
valid: a generated file should not carry an unsourced list of SHAs. Either derive the list
from git, or annotate each entry with where the SHA came from.

**(c) The file is stale**: it reports HEAD `3ecc4818` and "Dirty tree: **yes** (19 entries)";
HEAD is now `748b6b60` and the tree is clean. Regenerate after fixing (a).

### 4. `audits/CORE_REFACTOR_PHASE0_AUDIT.md` — **MARK AS HISTORICAL; do not treat as current**

**This document is accurate, not confabulated.** Every claim spot-checked reproduces exactly
at `6f6e0f3f^`, at the line numbers cited. It should be relabelled — "audited at `6f6e0f3f^`,
2026-08-18; findings 1–5 closed by `6f6e0f3f`" — not retired or distrusted.

Findings now **closed**, each with the confirming pre-fix evidence quoted above:

| Finding | Cited | Confirmed at `6f6e0f3f^` | Status now |
|---|---|---|---|
| `_pixel_scale` literal `(128*288)` | `loss_functions.py:185-187` | yes, exact lines | fixed, regression-tested |
| margin inert in NT-Xent | `loss_functions.py:342` | yes, exact line | term removed entirely |
| `metricVAE` unclamped logvar | `legacy_models.py:90, :106` | yes, exact lines | clamped at `:91, :107`, tested |
| `accumulate_grad_batches` dead config | `train_config.py:40` | yes, exact line | removed from `LitTrainConfig` |
| sweep configs / `training_cluster.py:32-48` | 54-line file | yes | now a 19-line retired stub |

Its "hard-coded `128×288` geometry" section needs partial correction: the `_pixel_scale`
bullet is closed, but the other cited sites — `arch_configs.py:18, :47, :62`,
`arch_spec.py:77` — were not in scope here and remain **unverified**.

Its `tv_weight`-unsummed finding was also out of scope and is **unverified**.

### 5. `audits/NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md` — **MARK AS HISTORICAL; one finding still live**

Same disposition, same reason: accurate when written, verified at `6f6e0f3f^`.

Closed:
- `:299-307` — `from data.dataset_configs import ...` at `model_configs.py:8`. Confirmed at
  `6f6e0f3f^`, same line. Now `src.core.data.dataset_configs`. (Its supporting claim that
  `src/data/dataset_configs.py` exists as a downstream-analysis shim is **still true** —
  `ls` confirms the file — so the collision hazard it describes was real.)
- `:315` — scripts invoking `src.run.training_cluster`. Confirmed at `6f6e0f3f^`. All live
  scripts now use `src.core.run.training`.

**Still live — the only open Phase 0 finding in either audit:**
- `:274-275, :413` — `contrastive_transform(target_size=...)` accepts and ignores the
  argument. Verified above against current code. `run_utils.py:484` and `:491` pass it. This
  should be carried forward into whatever replaces the retired readiness doc, and it wants a
  test.

Its `:217` claim (both datasets inherit `torchvision.datasets.ImageFolder`) and its
recommendations at `:526-528` were **not checked here** and remain unverified.

### 6. `DECISIONS.md` — **decisions stand; two documented inconsistencies unverified**

Not audited here beyond noting that `STATUS.md` reports all 23 decisions as carrying zero
marked tests:

> **23 decisions · 0 with tests (0 passing) · unverified: [D1, D2, D4, ...]**

Since the human-ratified decisions are real, the gap is in test marking, not in the ledger.
`TRAINING_READINESS_REMAINDER.md` alleges two internal inconsistencies (optical covariates;
ImageFolder boundary status) — I did **not** verify either, and they inherit that document's
credibility, so re-derive before acting.

### 7. `contracts/MANIFEST_SCHEMA.md` and `plans/AGENT_BRIEFS_PHASE1.md` — **no correction needed**

Both are specifications, correctly labelled as such by the brief. Nothing in this audit
contradicts either. `AGENT_BRIEFS_PHASE1.md` describes work that was never executed — it
remains a valid brief, and the surviving untracked `pipeline_contracts.py` in
`morphseq-phase1a` is a partial, unreviewed attempt at it.

---

## Loose ends this audit created

Report-only, per instructions — nothing below has been fixed.

1. **`794adf46` is unpushed and lives in `/tmp`.** 851 insertions of working provenance code
   in `/tmp/morphseq-snip-provenance.nobIjL`. `/tmp` is not durable. This is the highest-risk
   item found.
2. **`morphseq-phase1b` holds an uncommitted deletion of `src/core/data/dataset_classes.py`.**
   A live module, deleted, unreviewed, sitting in a worktree.
3. **`morphseq-phase1a` holds the only surviving Phase 1 artifact**, untracked:
   `src/core/data/pipeline_contracts.py`, 348 lines.
4. **`slice/phase1a`, `slice/phase1b`, `slice/phase1c` are redundant** — identical tree and
   parent to the already-merged `b76528eb`. The worktrees hold value (items 2–3); the commits
   do not.
5. **`src/core/run/registry.py` and `src/core/models/_extra_files/ldm_model_configs.py` fail
   on import unconditionally.** Both are unreferenced dead code. No document mentions either.
6. **`contrastive_transform` silently ignores `target_size`** (Claim 6) — untested, live.
7. **`core-model-refactor` is 194 commits behind `origin/main`** and does not contain the PR-31
   snip rendering line. Whether that matters is a decision, not a fact, but it should be a
   conscious one.
