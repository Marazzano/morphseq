# Plan — Keyence z-slice stitching, illumination, and resolution

Status (2026-07-16): **P0, P1, P1b, P2 IMPLEMENTED. P4.1 ATTEMPTED AND REVERTED — see P4.**
Output target resolution set to **6.5 µm/px** (write policy z_stack + snip_processing), per user.
NOTE: motion_blur_qc's `qc_micrometers_per_pixel` is deliberately **15.0889, not 6.5** — it is an
operating point, not an output scale. See P2.

Verification: full test run on `tests/data_pipeline/acquisition/{image_materialization,image_building}`
gives **15 failed / 184 passed** vs a clean-tree baseline of **15 failed / 178 passed** — the failure
SET is identical (zero regressions), +6 new passing tests. Those 15 are PRE-EXISTING failures on
`main` (stale fixtures missing `stage_x_nm/y/z`, plus product-shard merge tests); they are NOT from
this work and were left alone.

**NOT re-run yet:** plate01 still has the old images on disk. P0 changes pixels, so a re-run is
required before inspecting output.

---

## Still open / discovered along the way (NOT actioned)

- **Tile-wide gain saturates highlights.** On A02 the tile medians are 33607 / 31302 / 17251, so
  tile2 gets gain **1.81** and **2.18% of its pixels clip at uint16 max**. Tile-wide medians reflect
  CONTENT (embryo vs background), not just illumination, so a content-driven median gap is
  "corrected" as if it were illumination. Pre-existing (the gain predates this work); flagged, not
  changed.
- ~~motion_blur_qc threshold needs a one-time re-check~~ — **resolved**: QC is pinned at the
  resolution the threshold was tuned at (15.0889), so the operating point is preserved (+0.0027 ncc
  shift). See P2.
- **motion_blur_qc + snip_processing tests do not collect at all** on main (namespace collision:
  `tests/data_pipeline/` has no `__init__.py` and shadows the real package). Pre-existing.

Context: found while inspecting `20260702_hotchem_24hpf_plate01` (job 22309236, 870/870 steps,
2h00m, zero errors). All findings below are measured on that run's real output, not inferred.

---

## P0 — `_feather_composite` never fades tiles OUT (visible seams)

**File:** `src/data_pipeline/acquisition/image_building/utils/frame_tiler.py` (~L396-401)

Tiles are placed left→right and `placed_extent` holds only **already-placed (left) neighbours**, so
the trailing branch

```python
if ov > 0 and min(end, pe) == end:
    weight[-ov:] = np.linspace(1.0, 0.0, ov)
```

requires `pe >= end` — never true in a monotonic strip. **It never fires.** The comment
"next tile handles it" is wrong: the next tile sets only its own *leading* ramp and never goes back
to fade the previous tile out. Tiles fade IN but never OUT.

Result: in each overlap the left tile keeps weight 1 while the right ramps 0→1; the blend drifts to
the midpoint then **jumps to the right tile's value the instant the left tile ends** — a hard step of
~half the tile difference. Effectively a hard cut.

**Evidence (A02, plate01):**
- Synthetic tiles (100,120,100), jump at overlap exit: **production 10.00 / notebook 0.00**.
- Real FF frame: largest jump in the whole image is **row 1334, magnitude 20.4**; predicted overlap
  exit is **row 1335**. Overlap *centres* show only 0.51 / 0.45.
- Same-z, same final frame, seam step as % of dynamic range:

  | measured at | nb hard cut | nb feather+gain | **materialized** |
  |---|---|---|---|
  | overlap centres | 0.44, 4.00% | 0.96, 2.11% | **0.68, 4.08%** |

  The shipped image tracks the **hard cut**, not the feathered version.

**Why the earlier "+2/-2 seams fixed" verification was a FALSE PASS:** `seam_positions()` measures the
overlap **centre** (x=816, 1488), which genuinely is smooth. The real discontinuity is at the overlap
**exit** (x = i*step + tile_w = 960, 1632) — 144px away.

**Fix:** mirror the already-correct reference implementation, `composite_feather` in
`results/nlammers/20260702_hotchem_qc/keyence_illumination_correction_compare.ipynb`, which sets both
ramps symmetrically (`if i>0: w[:overlap]=linspace(0,1)`; `if i<n-1: w[-overlap:]=linspace(1,0)`).

**Verify at overlap EXITS, not centres.** Requires re-running plate01.

---

## P1 — Write policy: PNG optionality + fixed target resolution

**File:** `src/data_pipeline/acquisition/image_materialization/materialized_image_write_policy.py`

### Decisions (user, 2026-07-16)
- **Keep jpg as the DEFAULT for z-slices** (raw data is not archived yet).
- **Build in PNG optionality** — a future where heavy raw is archived and only PNGs are kept.
- **Default target resolution = 6.5 µm/px.**

### Defect being fixed
The policy is **blind to physical scale**. `resolve_image_write_policy(config, product_key)` takes no
image metadata; `downsample_image` applies a pure integer factor. `_PRODUCT_DEFAULTS` is keyed by
**product key only** (no scope), and YX1's `_build_yx1_write_policy` passes it straight through. So
both scopes get `÷4` from different natives:

| scope | native | after blind ÷4 |
|---|---|---|
| Keyence | 3.7744 µm/px | **15.09 µm/px** |
| YX1 (`20260502_zfpm_pilot`) | 3.2308 µm/px | **12.92 µm/px** |

A 17% physical-scale mismatch baked silently into the archive. The legacy pipeline **already did this
correctly** — `build03B_export_z_snips.py:316` used `outscale=5.66` and rescaled
`px_dim_raw / outscale` per image. The rewrite lost fixed-resolution targeting and replaced it with ÷4.

### TRAP 1 — format→method coupling silently disables downsampling
`_default_downsample_method(fmt, factor)` returns `"none"` for any non-jpg format, and
`resolve_image_write_policy` calls it whenever `downsample_method` is not **explicitly** in the
overrides — **overwriting the product default**. Verified:

```
override {file_format: png, jpeg_quality: None}
  -> downsample_method='none', factor=4    >>> DOWNSAMPLING SILENTLY OFF
```

Flipping to PNG via config alone yields full-res PNGs and **no error**. Must be fixed as part of
adding PNG optionality (make the default method format-agnostic, or key it off factor only).

### TRAP 2 — non-integer scale breaks the int contract
Targeting 6.5 µm/px gives non-integer scales (Keyence 1.72, YX1 2.01). Exact edits:

- **L91-92** `ImageWritePolicy.downsample_factor: int` → `float`.
- **L149** `expected_downsampled_dims`: `factor = int(factor)` → float. L155-156 `area_resize` already
  does `round(w/factor)` — **works unchanged for floats**.
- **L162-169** `block_mean` divisibility (`width_px % factor`) — must explicitly reject non-integer.
- **L177** `downsample_image`: `factor = int(...)` → float. L188 `reshape(h//factor, ...)` is int-only.
- **L245ish** `_validate_policy`: `factor = int(...)` → float coercion; keep the `< 1` guard.
- **L80-84** `_DEFAULT_METHOD_BY_FORMAT_FACTOR` is keyed by the `(format, factor)` **tuple** — a float
  like `("jpg", 1.72)` never hits and falls through to `"block_mean"`, which then fails loud on
  non-divisible dims. **This dict cannot survive non-integer factors.**
- **L100** `resolve_image_write_policy` needs native µm/px threaded in (new param) + both callers
  (keyence materializer, yx1 `_build_yx1_write_policy` L588-605).
- **L23-31 / L35-43** add `target_micrometers_per_pixel` to `_VALID_POLICY_KEYS` **and**
  `MATERIALIZED_IMAGE_WRITE_POLICY_COLUMNS` (frame_inventory provenance columns).

`expected_downsampled_dims` has **no external callers** — the int contract is contained.

### Changing output dims — downstream impact is minimal (verified)
- Recorded dims are **read back from the written file's header** (`materialize_well_keyence.py`
  L333-337), not computed → adaptive.
- `frame_inventory_validation_rules` L285-291 is a **self-check** (declared vs actual header) → passes.
- L403 focus_index_map shape check is **projection-only** (z_stack rows must have
  `focus_index_map_path` NA) → untouched by a z-slice-only change.
- **The only real risk is behavioral:** `motion_blur_qc.bad_z_pair_ncc_threshold: 0.90` was tuned at
  the current resolution. See P2.

### OPEN QUESTION — which knob is "6.5"?
"Default target resolution 6.5 µm/px" was stated in a write-policy context, so it is applied here to
**materialized z-slices**. But `snip_processing.target_pixel_size_um` currently defaults to **7.8**
(`pipelines/snip_processing.py:42`) — coarser than both legacy values (6.5 FF / 5.66 z) and possibly
never deliberately chosen. **Confirm whether snip target should also become 6.5.** Setting z-slice
storage = snip target = 6.5 means snips need no upsampling from materialized z-slices (see P4).

---

## P2 — motion_blur_qc: downsample on the fly

**Files:** `src/data_pipeline/quality_control/motion_blur_qc/{config.py,compute.py}`

`BF__z_stack` has exactly **one consumer**: motion_blur_qc (surfaced via snip_qc's `motion_blur_flag`).
Detection runs on projections only (`run_frame_detection.py:104`). Segmentation/focus_qc use the FF.

The plumbing already exists: frame_inventory carries **`image_micrometers_per_pixel`** (verified
populated: z_stack rows = 15.0889, FF rows = 3.7744), and the load boundary already returns it —
it's just discarded:

```python
z_planes, _z_rows = load_z_stack_images_from_image_id(...)   # compute.py:126
```

**IMPLEMENTED.** Added `qc_micrometers_per_pixel` to `MOTION_BLUR_QC_DEFAULTS` / `MotionBlurQCConfig`
and `_resample_planes_to_target()` in `compute.py`, which resizes planes from
`_z_rows["image_micrometers_per_pixel"]` via the existing `resize_image_to_shape` (already picks
`INTER_AREA` when shrinking). Nothing else needed — `compute_mask_pixel_motion_metrics` already does
`resize_binary_mask_to_shape(mask, stack.shape[1:3])`, so the mask follows the stack.

### The default is 15.0889, NOT 6.5 — this is the whole point of the knob
`bad_z_pair_ncc_threshold=0.90` was tuned against Keyence z-slices as materialized under the old
blind `downsample_factor=4` (native 3.7744 => **15.0889 µm/px**). Pinning QC there keeps that operating
point intact while the write policy independently moves storage to 6.5 µm/px. **Storage resolution is
a storage decision; QC resolution is a QC decision** — that separation is why this knob exists. Setting
it to the 6.5 storage target would MOVE the operating point and force a re-tune, i.e. exactly the
disruption the design was meant to avoid.

**Bonus correctness fix:** the 0.90 threshold previously applied to Keyence at 15.09 µm/px and would
have hit YX1 at 12.92 µm/px — the same number silently meaning different things per scope. A fixed QC
target makes it **scope-invariant**.

### Verified: operating point preserved, NO re-tune needed
QC resamples the new 6.5 µm/px storage back to **(428, 180)** — the exact grid the old on-disk planes
used. On real A02 planes:

| | ncc_mean | ncc_min | flag |
|---|---|---|---|
| OLD (15.09 jpg, on disk) | 0.9924 | 0.9877 | False |
| NEW (6.5 stored -> resampled to 15.09) | 0.9951 | 0.9933 | False |

**Shift +0.0027 against a 0.90 threshold — negligible.** The small positive direction is expected: a
finer JPEG averaged back down carries less blocking noise (measured ratio 1.427 vs FF's 1.001) than
one encoded straight at 15.09. Caveat: the "stored finer" planes in that check were upscaled from the
on-disk ones as a stand-in, so it validates grid alignment and the JPEG-path shift, not new detail.

Behaviour table (verified): source 6.5 -> 428x180; native 3.774 -> 248x105; source == 15.0889 -> no-op;
source coarser (20.0) -> no-op, never upsamples. Mixed calibrations in one stack -> fail loud.
`None` disables resampling entirely.

---

## P3 — z-snips for model training (design, not yet scoped)

**Plan:** z-slices are used for model training **in addition to** FF. High-res full-frame z-slices are
**not** needed on disk right now; what's needed is **masked z-slice snips at a designated resolution**.

The machinery already exists and is FF-only:
`snip_processing/pipelines/snip_processing.py` already does fixed-resolution snips —
`target_pixel_size_um` (default 7.8), `output_shape_hw` [576, 256] (matches legacy `outshape`), reads
each frame's `image_micrometers_per_pixel` and calls
`extract_embryo_crop(image, mask, yolk, output_shape, pixel_size_um, target_pixel_size_um)`.
**The write policy is the only place in the pipeline that ignores physical scale.**

**Gap:** extend the snip step to load z-planes per `image_id`
(`load_z_stack_images_from_image_id` already returns exactly that) and run the same
crop/rotate/mask at the same target. One mask + one rotation per stack — correct for a z-stack snip.

**Constraint: source resolution must be >= snip target.** With z-slices stored at 6.5 µm/px and snip
target 6.5, this is satisfied with no upsampling. (At today's 15.09 it is NOT — 2.7x upsampling.)

### Storage evidence (measured, one real YX1 plane, `20260502_zfpm_pilot`)
Native 3.2308 µm/px, 2189x2189. **T=83, P=96, Z=9 → 71,712 z-frames/experiment** vs Keyence's 1,344
(96 wells x 14 z x **1** timepoint). It is the **timepoints**, not the resolution, that dominate.

| encoding @ 5.66 µm/px (1250x1250) | per frame | per exp | x66 YX1 exps |
|---|---|---|---|
| PNG | 508 KB | 36.4 GB | **2.41 TB** |
| JPG q85 | 75 KB | 5.4 GB | 0.36 TB |
| PNG @ native (2189x2189) | 1,887 KB | — | ~9 TB |

**Volume is 93% full: 15 TB free of 210 TB.** PNG full-frame z for YX1 = ~16% of remaining headroom
— hence the decision to keep jpg for now.

Legacy never materialized full-frame YX1 z-slices at all (`build01B` writes only FF `_stitch.jpg`);
z-snips were pulled **straight from the nd2** (`imObject.voxel_size()`). Snips at 576x256 are ~35x
smaller than 1250x1250 full frames, so z-snips are only GB-scale even for YX1.

**Fork to decide when scoped:** (a) snip from raw at export time (legacy; no full-frame cost; breaks
the materialization boundary), (b) store z at >= target (chosen for now via 6.5 µm/px + jpg),
(c) transient high-res z per well (no permanent cost; snips non-reproducible without re-run).

---

## P4 — conda/import overhead (LOW priority; earlier claims were overstated)

**Measured warm (3 reps, seconds):**
```
direct python, no import   : 0.05  0.01  0.01
conda run, no import       : 3.11  0.56  0.43     <- conda itself: ~0.4s warm
tasks CLI --help           : 4.74  1.01  0.61     <- the 678 bookkeeping jobs
materialize_well_keyence   : 29.71 21.18 13.65    <- the 192 real jobs
torch                      : 17.41  9.54  4.76
```

**`conda run` is NOT the villain (~0.4s warm).** The cost is **imports**: `materialize_well_keyence`
takes **13-30s** because it pulls `torch` at module level via `focus_stack_group`. The `tasks` CLI is
correctly lazy (`torch loaded by tasks import: False`).

Honest total: ~192x15s + ~678x1s ≈ 3,560s / 4 cores ≈ **15 min of 120** → **12-25%** depending on
cache. **This was never the 5-hour explanation** (that remains cluster contention: plate02 midday got
84% in 5h19m on the same 870-step DAG that finished in 2h00m overnight).

**Uniquely bad for single-timepoint Keyence — CONFIRMED.** Job counts are per-well x per-product,
never per-timepoint (192/192/192/96/96/96 + 4 = 870). That count is **identical** for a 1-timepoint
Keyence plate and an 83-timepoint YX1 run — only the work *inside* each materialize job scales with T.
So the fixed ~15s import is amortized over ~83x more work on YX1 and bites ~1% there vs ~25% here.

### P4.1 was ATTEMPTED and REVERTED (2026-07-16). Do not retry as specified.

My "~80% of the benefit for ~5% of the work" estimate was **wrong on both halves**:

**The benefit is ~4%, not ~20%.** Of the 192 materialize jobs, ~96 are projection jobs that genuinely
focus-stack and need torch regardless. Only the ~96 z_stack jobs can skip it:
`96 x ~13s / 4 cores ≈ 5 min of a 120 min run`. On YX1 it is ~1%.

**The only approach that actually works is unsafe.** Deferring the import inside the materializer
does NOTHING on its own: `materialize_well_keyence` imports `display_polarity`, which triggers
`image_building/shared/__init__.py`, which eagerly imports `focus_stack_group` -> torch. Python always
runs a package `__init__` on any submodule import, so the ONLY way to break the chain is to make that
`__init__` lazy — and that introduces an **order-dependent name collision**, measured:

| import order | eager `__init__` (main) | lazy `__init__` (attempted) |
|---|---|---|
| attribute first | function | function |
| **submodule imported first** | **function** | **MODULE** <- silently different |

`focus_stack_group` is BOTH a submodule and a function in that package. The eager `__init__` rebinds
the name to the function after importing the submodule, so the function always wins. A PEP-562 lazy
`__getattr__` is never consulted once Python auto-binds the submodule attribute, so
`from ...shared import focus_stack_group` starts returning a MODULE as soon as anything imports the
submodule first — which the lazy materializer import itself does. Unfixable without renaming either
the submodule or the exported function (both API breaks).

Pushing the laziness down into `focus_stack_group.py` / `log_focus.py` (deferring `import torch`
inside their functions) would avoid the collision, but `log_focus` uses torch pervasively in
module-level annotations and signatures — invasive surgery on a core primitive for ~4%.

**Verdict: not worth it. Trading a latent correctness hazard in a shared primitive for ~4% is a bad
trade.** If per-job startup ever becomes the bottleneck, do the batch dispatcher instead.

**Options, ranked:**
1. ~~Lazy-import torch in the materialize path~~ — **REVERTED, see above.**
2. **`EXECUTION_RUN_BATCH` dispatcher** — one job per experiment, import once, loop wells in-process
   (870 → ~10 jobs). Amortizes torch AND conda across the whole experiment, sidestepping the name
   collision entirely. The flag exists in `paths.py`; the dispatcher does not. Loses per-well
   retry/resume; needs internal parallelism to keep the 4 cores busy. **The real lever if needed.**
3. **Bypass `conda run`** — saves ~0.4s/job ≈ 1.5 min total. **Not worth it.**

---

## Suggested order
1. **P0 feather fix** — visible defect, small change, requires plate01 re-run.
2. **P1 write policy** — PNG optionality + trap 1 + trap 2 + 6.5 µm/px target. Resolve the OPEN
   QUESTION on the snip knob first.
3. **P2 motion_blur_qc** on-the-fly downsample + threshold re-check (pairs naturally with P1).
4. **P4.1 lazy torch import** — cheap, independent.
5. **P3 z-snips** — needs scoping; largest design surface.

P0 + P1 + P2 all land in the same re-run, so batch them before re-running plate01.
