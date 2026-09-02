# Z-Stack Focus QC + Motion-Blur QC (TARGET STUB)

**Status:** target stub, mdcolon 2026-06-30.

This document defines the two post-segmentation QC products that consume already-materialized
z-stack/projection pixels:

- `focus_qc`
- `motion_blur_qc`

It deliberately does **not** specify microscope acquisition details, upstream extraction behavior, or
materialized image layout. Those belong to upstream specs. This QC spec starts at the shared pipeline
handoff:

```text
validated snip_inventory + validated frame_masks + validated frame_inventory
```

The only question here is: given canonical snips, masks, and inventory-addressed pixels, what QC
summary rows do we write?

---

## One-Sentence Decision

`focus_qc` and `motion_blur_qc` are ordinary per-well QC products, cloned from the
`mask_quality_qc` pattern: they run on a well shard, load pixels through that well's
`frame_inventory`, compute embryo-local metrics inside each snip mask, emit
`SNIP_ID_SPINE_COLUMNS + *_PAYLOAD_COLUMNS`, validate against `physical_embryo_registry` with
`check_sources=True`, and feed `snip_qc` through named flag columns.

---

## Scope Boundary

**In scope here:**

- QC product contracts for `focus_qc` and `motion_blur_qc`.
- Compute inputs and transient metric definitions.
- Per-well and merged on-disk QC CSV outputs.
- `snip_qc` hook names.

**Out of scope here:**

- how z-stacks are deposited;
- acquisition inventory schema;
- image encoding policy;
- selected-plane outputs.

QC consumes the post-materialization contract. If a run has `BF__projection__focus_stack` and
`BF__z_stack` rows in `frame_inventory`, QC can use them. If it does not, that is an upstream
materialization/input availability problem, not something this QC product repairs.

Saturation / crushed-brightness QC is also out of scope for `focus_qc`. It belongs with the general
mask/image-quality product (`mask_quality_qc` or its direct successor), not in this z-stack focus
product.

---

## Shared QC Pattern

Both products mirror `src/data_pipeline/quality_control/mask_quality_qc/`:

```text
src/data_pipeline/quality_control/{focus_qc,motion_blur_qc}/
  __init__.py
  contract.py      # schema + validator
  compute.py       # pure metric/flag math
  config.py        # thresholds / params
  entrypoint.py    # thin filesystem adapter
```

Entrypoint inputs:

```text
snip_inventory_csv
frame_masks_csv
frame_inventory_csv
physical_embryo_registry_csv
output_csv
optional config overrides
```

Rules:

- copy `SNIP_ID_SPINE_COLUMNS` from the snip universe;
- use `SNIP_FRAME_PROVENANCE_COLUMNS` only to find pixels through `frame_inventory`;
- never parse IDs or infer `physical_embryo_id` from filenames;
- never emit frame provenance, image encoding fields, or raw pixel paths as QC payload;
- validate with `validate_snip_grain_identity_columns(..., grain="snip_id", check_sources=True)`;
- fail loud on missing required pixel products instead of writing neutral flags.

The table shape is fixed:

```text
QC_TABLE_COLUMNS = SNIP_ID_SPINE_COLUMNS + QC_PAYLOAD_COLUMNS
```

Use the column-family pattern documented in `pipeline_file_philosophy.md` and
`features/targets/feature_world.md`: spine columns are imported from the identity minting site,
payload columns are owned by this QC product, and `*_TABLE_COLUMNS` is the emitted contract. These
QC products intentionally have no emitted provenance block.

---

## focus_qc

### What It Decides

Whether the already-materialized projection image has enough internal embryo structure to trust the
snip. The current target is the "ghost / structureless embryo" failure mode from the
`results/mcolon/20260625_cross_experiment_focus_motion_qc` review: the embryo body can be bright,
large, and well masked, but empty inside because it is badly out of focus.

This is not saturation QC. It should not emit `saturation_flag`, `top_spread_p99_p90`, JPEG quality,
downsample factor, projection path, or z-stack provenance.

### Pixel And Mask Inputs

For each snip in the well shard:

1. start from the snip row in `snip_inventory`;
2. use its frame provenance (`image_id`, `time_index`, `channel_id`, as available in the contract) to
   find the projection row in the per-well `frame_inventory`;
3. load the projection image from `frame_inventory.source_image_path`;
4. load the snip mask from `frame_masks`;
5. align the mask to the loaded image shape if needed, using nearest-neighbor mask semantics.

The MVP focus metric uses the projection image plus mask only. It may operate on a snip-shaped crop,
because the metric is measured inside the embryo interior, but the crop must be raw image evidence:
pixel-equivalent to the projection row named by `frame_inventory`, unaugmented, and paired with the
same mask coordinate transform. It must not consume the processed snip image from `snip_processing` if
that image has been CLAHE/noise augmented.

Default implementation: load the projection via `frame_inventory` and make the crop transiently inside
`focus_qc`. If a future raw-snip crop product is promoted, `focus_qc` can consume it only after that
crop is a contracted, unaugmented, inventory-addressed pixel product.

The crop should include a small local context band around the mask. The normalization in the research
script used mask plus dilated local context because the metric thresholds (`grad > 0.02`,
`interior_strong_edge_fraction < 0.50`) live on a locally normalized 0-1 image, not on raw camera
intensity. Mask-only normalization can stretch tiny within-embryo variation into apparent structure;
raw-255 normalization is more punitive on bright/dorsal cases; whole-crop normalization is close but
less explicit about the reference population. The local context band gives a stable exposure reference
near the embryo while still keeping the verdict embryo-local.

This preserves the run grain the pipeline needs: the rule executes per `well_id`, and the verdict is
calculated per `snip_id` / embryo within that well.

### Metrics And Verdict

Compute for every snip:

```text
interior_mask = erode(mask, 12 px), fallback to 6 px, then full mask if too small
local_image = raw projection crop around mask + local dilated context
normalized_image = robust 1st-99th percentile rescale over mask + local context
gradient_image = Sobel(Gaussian(normalized_image, sigma=1.0))
interior_strong_edge_fraction
interior_n_px
focus_flag = interior_strong_edge_fraction < interior_strong_edge_fraction_threshold
```

Current target defaults from the 2026-06-30 fine-tuning review:

```text
normalization_mode = "local_context"
strong_edge_sobel_threshold = 0.02
interior_strong_edge_fraction_threshold = 0.50
min_interior_pixels = 200
```

`interior_strong_edge_fraction` is the fraction of eroded-interior pixels whose local gradient
exceeds `strong_edge_sobel_threshold`. The hard erosion strips the silhouette boundary, because the
body outline is strong even in bad focus and otherwise drowns out the internal structure signal.

This cutoff is a deliberate compromise, not a claim that every excluded embryo is technically
out-of-focus. The fine-tuning review showed that `local_context`, `grad > 0.02`,
`interior_strong_edge_fraction < 0.50` removes a small low-information tail (~2.1% overall in the
7,500-embryo sample, concentrated in 20251125) while catching the whole-embryo ghost/structureless
anchors. Some bright/dorsal embryos with genuinely low internal information will be removed. That is
acceptable for the first pipeline gate: this product is allowed to be conservative and exclude
low-information snips rather than preserve every borderline bright/dorsal case.

Calibration/review reports should show not-dead snips separately so dead embryos do not masquerade as
focus failures, but death status is not part of the `focus_qc` contract.

The research scripts also computed exploratory companions such as `interior_std`,
`interior_grad_p90`, `interior_lap_density`, and straight-axis band summaries for partial defocus.
Those are not canonical payload unless promoted deliberately. Write them as review/debug artifacts
outside the stable QC table until a downstream consumer or a persisted verdict requires them.

### Contract

```text
FOCUS_QC_PAYLOAD_COLUMNS = (
    "interior_strong_edge_fraction",
    "interior_n_px",
    "focus_flag",
)

FOCUS_QC_TABLE_COLUMNS = SNIP_ID_SPINE_COLUMNS + FOCUS_QC_PAYLOAD_COLUMNS
```

`focus_flag` is a required non-null boolean. Metrics are required and numeric. A flag without its
supporting metric is not an acceptable QC product.

---

## motion_blur_qc

### What It Decides

Whether adjacent z planes disagree inside the embryo mask strongly enough to indicate inter-slice
motion blur.

This is a snip-grain QC product even though the source z-stack rows are frame/plane-grain. The mask is
what makes the verdict embryo-specific.

**Forward note for Pass 2:** the Pass-1 z-stack materialization policy may write `BF__z_stack` at
`downsample_factor != 1` (current target: JPEG q85, factor 4). `motion_blur_qc` must therefore align
the embryo mask onto the loaded z-plane dimensions with nearest-neighbor mask semantics before any
mask-pixel NCC calculation. The MVP threshold is fixed at `0.90`; the snip-level flag threshold is
`mask_pixel_bad_pair_frac > 0.10`.

### Pixel And Mask Inputs

For each snip in the well shard:

1. start from the snip row in `snip_inventory`;
2. use its frame provenance to find the corresponding `BF__z_stack` rows in the per-well
   `frame_inventory`;
3. sort z-stack rows by `z_index`;
4. load planes from `frame_inventory.source_image_path`;
5. load the snip mask from `frame_masks`;
6. align the mask to the loaded z-plane shape if needed, using nearest-neighbor mask semantics.

This lookup should be framed as "which materialized z-stack rows does this well's `frame_inventory`
declare for this snip's image/time/channel?", not as a filesystem crawl for z-stack files. Missing or
ambiguous rows are contract failures.

QC should not care whether z-stack planes are JPEG, PNG, or TIFF. Encoding, JPEG quality,
downsample factor, and dimensions are `frame_inventory` facts. They are useful for loading and
validation, but they are not QC payload columns.

### Metrics And Verdict

No grid is persisted in the MVP. For each adjacent z-plane pair:

```text
mask_pixel_pair_ncc[z] = NCC(z_plane[z][mask], z_plane[z+1][mask])
bad_z_pair = mask_pixel_pair_ncc < 0.90
mask_pixel_bad_pair_frac = fraction(valid bad_z_pair values)
motion_blur_flag = mask_pixel_bad_pair_frac > 0.10
```

Evaluation policy:

- require at least two z planes;
- require a non-empty mask after any alignment;
- require at least one adjacent pair with nonzero in-mask variance;
- exclude flat/constant pairs from the denominator, but count them in the output;
- fail loud if no valid adjacent pair remains.
- do not add a minimum mask-pixel threshold in the MVP; if the mask exists and is non-empty, evaluate
  it.

This replaces the older whole-frame/tile `bad_pair_frac` gate, which was diluted by background and
missed visible embryo motion.

### Contract

```text
MOTION_BLUR_QC_PAYLOAD_COLUMNS = (
    "mask_pixel_ncc_mean",
    "mask_pixel_ncc_min",
    "mask_pixel_ncc_p05",
    "mask_pixel_bad_pair_frac",
    "mask_pixel_longest_bad_run",
    "n_z_planes",
    "n_z_pairs",
    "n_valid_z_pairs",
    "n_flat_z_pairs",
    "n_mask_pixels",
    "motion_blur_flag",
)

MOTION_BLUR_QC_TABLE_COLUMNS = SNIP_ID_SPINE_COLUMNS + MOTION_BLUR_QC_PAYLOAD_COLUMNS
```

`motion_blur_flag` is a required non-null boolean. Metrics are required and numeric.

---

## On-Disk Products

These are ordinary `quality_control` stage products. Add two `PIPELINE_STEPS` rows matching the
existing `mask_quality_qc` row shape:

```python
"focus_qc": {
    "stage": "quality_control",
    "product_dir": "focus_qc",
    "fanout": PER_WELL_THEN_MERGE,
    "execution": EXECUTION_PER_WELL,
    "artifacts": {"focus_qc": {
        PATH_MODE_PER_WELL: "{well_id}_focus_qc.csv",
        PATH_MODE_MERGED:   "{experiment_id}_focus_qc.csv",
    }},
},
"motion_blur_qc": {
    "stage": "quality_control",
    "product_dir": "motion_blur_qc",
    "fanout": PER_WELL_THEN_MERGE,
    "execution": EXECUTION_PER_WELL,
    "artifacts": {"motion_blur_qc": {
        PATH_MODE_PER_WELL: "{well_id}_motion_blur_qc.csv",
        PATH_MODE_MERGED:   "{experiment_id}_motion_blur_qc.csv",
    }},
},
```

Resolved paths:

```text
quality_control/{exp}/focus_qc/per_well/{well_id}/{well_id}_focus_qc.csv
quality_control/{exp}/focus_qc/{exp}_focus_qc.csv

quality_control/{exp}/motion_blur_qc/per_well/{well_id}/{well_id}_motion_blur_qc.csv
quality_control/{exp}/motion_blur_qc/{exp}_motion_blur_qc.csv
```

Sentinels are derived through `validated_path(...)`; they are not separate registry artifacts. Rules
must call the existing `rule_artifact` / `rule_validated` helpers and must not string-build paths.

No focus/motion intermediate grids, maps, debug overlays, copied z-stack planes, or encoding fields
are written under `quality_control/`.

---

## snip_qc Wiring

`snip_qc` consumes the final boolean flags:

```text
"focus"      -> "focus_flag"
"motion_blur" -> "motion_blur_flag"
```

Current wiring:

- `focus_flag` is registered in `snip_qc/flag_input_resolver.py` and
  `SNIP_QC_EXCLUSION_FLAGS`.
- `motion_blur_flag` is registered in `snip_qc/flag_input_resolver.py` and
  `SNIP_QC_EXCLUSION_FLAGS`, so default `snip_qc` runs require validated per-well
  `motion_blur_qc` shards.

If saturation becomes a hard exclusion, wire `saturation_flag` from `mask_quality_qc` / general
image-quality QC, not from `focus_qc`.

---

## Build Order

1. Confirm `frame_inventory` exposes the materialized projection and z-stack rows needed by these QC
   modules.
2. Add `focus_qc` as a `mask_quality_qc`-style product: contract, compute, config, entrypoint,
   `PIPELINE_STEPS`, rules, tests.
3. Add `motion_blur_qc` the same way.
4. Wire both into `snip_qc` only after their output contracts exist. Done for `focus_flag` and
   `motion_blur_flag`.

---

## Locked Decisions

1. QC persists per-snip summaries: metrics plus flags, not flags alone.
2. QC recomputes focus/motion metrics from durable pixels and masks; no persisted grid product.
3. QC reads pixels through `frame_inventory`; it never infers paths or encoding from config.
4. `focus_qc` emits only `focus_flag` and focus metrics.
5. `motion_blur_qc` emits only `motion_blur_flag` and motion metrics.
6. Saturation belongs to general mask/image-quality QC, not `focus_qc`.
7. Microscope-specific z-stack deposition details do not belong in this QC spec.
