# Tier 1 — Depth Through-Line Findings & Change Log (🟢 GREEN)

**Status:** Tier 1 of `data_flow_test_plan.md` is **GREEN** — 2026-06-26, mdcolon + Opus.
**Result:** one YX1 embryo's data crossed *every* seam from raw ND2 to the `snip_qc` verdict, on
CPU, one well (`20250912_B01`), one timepoint — a **continuous from-raw rebuild** (not a tail top-up).

This doc satisfies the §5 reporting contract: every change made to get data flowing is recorded
**before** the tier is marked green. *A run result without a change log is folklore with a timestamp.*

---

## The run

- **Target:** `through_line` (new named target; `rule all` untouched, still stops at merged frame_masks).
- **Fixture:** `20250912` / `B01` / 1 timepoint, overlay `config_smoke_through_line_20250912_B01.yaml`.
- **Mode:** continuous from-raw — `--forcerun materialize_image_product_for_well` so the whole per-well
  spine (30 jobs) regenerated from the raw ND2 in one pass (mdcolon chose this over a tail top-up so a
  green verdict reflects bytes that flowed the whole length *together*; the §2 ledger had found the
  prior B01 shards internally inconsistent across earlier smokes).
- **Device:** CPU throughout (`AUTO_MODE_CHOSEN -> CPU`; `unet_snip.device: cpu`). No GPU spent.
- **Timepoint limiter:** through config (`SMOKE_FRAME_CAP_ACTIVE: first 1 of 113 time_index`) — no
  ad-hoc `if time_index == 0` in any stage (§93 precondition satisfied via config alone).

### Success criteria (§3) — all observed on disk
| Criterion | Result |
|---|---|
| snip_qc verdict + `.validated` | ✅ `…/snip_qc/per_well/20250912_B01/20250912_B01_snip_qc.parquet` (+`.validated`) |
| `use_snip` populated, `qc_fail_reasons` sane | ✅ 1 snip, `use_snip=True`, `qc_fail_reasons=''` (clean live embryo) |
| No stage produced 0 rows; PKs unique; validators accept | ✅ frame_inventory 46 / detections 1 / masks 1 / registry 1 / snip_inventory 1 / snip_qc 1; `image_id`+`snip_id` unique |
| Every intermediate `.validated` sentinel | ✅ full chain frame_inventory → … → snip_qc (13 sentinels) |
| Resolver-doctrine proof | ✅ resolved_sources JSON exists; 3 sources (death/mask_quality/surface_area), all paths exist, declared sources match the exclusion map AND the DAG inputs to `build_snip_qc_for_well` |

> **Row-count note (not a failure):** frame_inventory carries 46 rows = 45 z_stack planes + 1 BF
> projection. Detection→snip_qc consumes only the **projection** frame (1 row), so the snip-grain
> through-line is genuinely one timepoint. Follow-up observation: the z_stack rows span time_index
> {0,1,2} (a residue of an earlier 3-tp materialization of the z_stack product that the cap did not
> re-truncate, since z_stack was already on disk). Harmless to the depth proof — the projection
> stream, which is what flows downstream, is 1 timepoint. Worth a clean re-materialize before Tier 2.

---

## CHANGE LOG (§5) — every change that made data flow

### Change 1 — add the `through_line` named target (the missing REQUEST)
1. **What:** new `rule through_line` + `_through_line_targets()` in `Snakefile`; new overlay
   `config_smoke_through_line_20250912_B01.yaml`; fixed the stale comment that forward-referenced it.
2. **Why:** the snip_qc rules were fully wired (snip_qc.smk) but **nothing requested them** — `rule all`
   stops at merged frame_masks, so the terminal was unreachable. (§6 DECIDED: add a separate named
   target, do not extend `rule all`.)
3. **Doctrine:** rule/step naming + config. Mirrors `front_half` (aggregate over discovered wells,
   well-set chosen by `target_wells`); `rule all` default behavior untouched.
4. **Permanence:** permanent (canonical target). The config overlay is dev-smoke scaffolding.
5. **Verified:** `snakemake -n through_line` plans the full spine to `validate_snip_qc_for_well`;
   then the real run reached it (3/3 terminal jobs green).

### Change 2 — `unet_snip` model route is ENVIRONMENT, not data layout (checkpoint-not-found)
1. **What:** `pipeline_orchestrator/tasks.py::cmd_snip_auxiliary_masks` — env `--models-root`
   (from `env.yaml.paths.models_root`) is authoritative and **replaces** any config `models_root`.
   `config.yaml` `unet_snip`: moved the `segmentation/` family segment OUT of `models_root` and INTO
   each per-family `checkpoint` key (`checkpoint: "segmentation/<family>"`); `models_root` is now an
   env-overridden placeholder, documented as such.
2. **Why:** the run failed at `build_snip_auxiliary_masks_for_well` —
   `FileNotFoundError: UNet checkpoint not found: …/models/bubble_v0_0100`. The checkpoints live at
   `…/models/segmentation/<family>`; the CLI was building `…/models/<family>` because of a config/env
   models_root collision.
3. **Doctrine:** the **config(science) vs env.yaml(machine) kingdom boundary**. Model weights may live
   anywhere on a machine, so the model root is an *environment* fact (env.yaml); the science config
   owns only the per-family checkpoint sub-route. **(mdcolon correction:** an earlier attempt rebased a
   relative config `models_root` onto `output_root` — rejected as data-relative-path pollution that
   only coincidentally works. Reverted to env-root-authoritative.)** This is a local instance of the
   broader gap recorded in `tech_debt/model_routing_pattern.md` (no general model-routing pattern yet);
   the fix here does NOT solve the general case, it just keeps this seam env-authoritative.
4. **Permanence:** permanent (canonical fix); no relative-path machinery added.
5. **Verified:** all four checkpoints (via/yolk/focus/bubble) resolve+exist through the real loader;
   `unet` + `tasks_parser` tests green; the live run's aux-masks step then ran on CPU.

### Change 3 — snip_qc accepts pandas nullable BooleanDtype (dtype seam)
1. **What:** `quality_control/snip_qc/build.py` — the flag-column dtype gate changed from
   `dtype != bool` to `not pd.api.types.is_bool_dtype(...)` (accepts numpy bool AND pandas nullable
   `boolean`), with a dtype-naming error message.
2. **Why:** the run failed at `build_snip_qc_for_well` —
   `ValueError: flag column 'viability_dead_flag' must be boolean dtype`. `inputs.py` (the canonical
   producer of `qc_flags_df`) deliberately coerces every flag to pandas nullable `"boolean"` and
   guarantees no NA; `build.py` demanded numpy plain `bool` — a producer/consumer dtype mismatch at the
   snip_qc internal seam. Unit-green because the build tests construct plain-bool fixtures; only the
   real `inputs.py → build.py` path exposed it.
3. **Doctrine:** validator/contract consistency (one product, producer and consumer must agree on
   dtype). The non-null guarantee is preserved (both `inputs.py` and `build.py` still raise on NA).
4. **Permanence:** permanent (canonical fix).
5. **Verified:** 96 `quality_control` tests green; the live run's `build_snip_qc_for_well` then
   produced the verdict (1/3 → 3/3).

---

## ⛔ STOP — GPU GATE (do not proceed to Tier 2 autonomously)

Tier 1 is green on CPU. Per `data_flow_test_plan.md` §"GPU GATE", **PAUSE and hand back to mdcolon**
to verify this through-line result + this change log before any GPU is requested for Tier 2 (WIDTH,
all wells). mdcolon explicitly approves crossing this gate.

**Follow-up to consider before Tier 2:** a clean re-materialize of B01 so the z_stack stream is also
1-timepoint (cosmetic; does not affect the snip-grain through-line, which is already 1 tp).
