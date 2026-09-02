# Handoff: z-stack materialization (downsample default) + motion_blur_qc build-out

**From:** Sonnet, 2026-06-30. **To:** Opus, next session.

## Where things stand

`focus_qc` (Chunk B) is fully built, tested, and wired into `snip_qc` — see commits
`c0e81ac4`..`0ce49e97` on `mdcolon/20260222_docs_snakemake_remake`. `motion_blur_qc` (Chunk C) is
**deliberately deferred** per `docs/.../plans/wiggly-wondering-lovelace.md` — it must NOT be wired
into `snip_qc`'s `SNIP_QC_EXCLUSION_FLAGS` / `_SOURCE_PAYLOADS` until z-stack materialization is
proven live on real data. That gate is still closed.

User's ask for this session, in order:
1. Set `downsample_factor: 4` as the **default** for z-stack materialization (not just an override).
2. Run z-stack materialization through for real and **submit it as a cluster job** (SGE), not run
   it interactively.
3. Review the `motion_blur_qc` spec section together and fine-tune defaults before implementing.

## 1. The encoding/downsample layer does not exist yet — build it first

I verified this directly: `src/data_pipeline/image_materialization/` has no `image_encoding.py`,
no `ImageEncodingConfig`, no `resolve_image_encoding_config` — none of it is wired. The current
YX1 z-stack write path (`materialize_well_yx1.py` ~line 409) calls `skio.imsave` directly with no
downsample and no format/quality knobs; `materialized_image_paths.z_stack_frame_path` defaults to
`ext="png"`. So "set downsample to 4" is not a config flip — the resolver module described in
`docs/.../front_end/image_encoding_materialization_config.md` has to be built first, then threaded
into the materializer, before any downsample setting does anything.

That spec doc already exists and is fairly complete (config shape, defaults table, path policy,
frame_inventory encoding columns, build order). Read it before writing code — most of the design
decisions are already made there. The one thing to change: its locked MVP recommendation currently
shows `downsample_factor: 1` for `BF__z_stack`; the user now wants **`downsample_factor: 4`** as
the default (not 1, and not 2 — go straight to 4). Update the doc's `Defaults` and `Locked MVP
Recommendation` sections to say 4, and explain why in the doc (storage: the spec's own probes
showed JPEG q85 z-stack at ~140GB across just 3 experiments at downsample 1 — 4x downsample drops
the pixel count 16x, which is the actual lever, more than the JPEG quality knob).

### Naming — user flagged `image_encoding.py` as a bad name, do not just use it

Same complaint pattern as `materialized_image_loaders.py` earlier this session (see git log
`dde2411a` and the conversation that produced it) — a name like `image_encoding.py` undersells
what the module does. It's not a codec utility; it's a **per-product policy resolver +
write-time application**, the encoding-policy sibling of `materialized_image_paths.py` (path
grammar) and the newly-added `materialized_image_readers.py` (read side). Candidates worth
weighing with the user before committing:
- `materialized_image_encoding.py` — matches the `materialized_image_*` family already
  established (`materialized_image_paths.py`, `materialized_image_readers.py`).
- Split `resolve_*` (pure config resolution, e.g. `image_encoding_policy.py`) from the actual
  pixel-write helper — mirrors the `resolve_*`/`load_*` split that worked well for the readers
  module.
**Do not pick silently.** Ask the user with concrete options the way the readers-module naming was
resolved earlier (AskUserQuestion with 2-3 named candidates) — they care about this and will push
back on a guess.

### Build order for this piece (from the existing spec, adjusted for downsample=4 default)

1. Add the resolver/writer module (name TBD above) with `ImageEncodingConfig`,
   `resolve_image_encoding_config(config, product_key)`, `suffix_for_encoding`,
   `apply_downsample`, `prepare_image_for_write`, `write_encoded_image`. Unit tests.
2. Add `image_materialization.products` defaults to `config.yaml`, with
   `BF__z_stack: {file_format: jpg, jpeg_quality: 85, downsample_factor: 4, pixel_dtype: uint8}`.
3. Thread the resolved encoding into the YX1 z-stack materializer
   (`materialize_well_yx1.py`, the `z_stack` branch ~line 401-432) — replace the bare
   `skio.imsave` with `write_encoded_image` after `apply_downsample`.
4. Update `materialized_image_paths.z_stack_frame_path` (and `materialized_image_path`) to accept
   a dynamic suffix from the resolver instead of a hardcoded `ext="png"` default.
5. Extend `frame_inventory_contract.py` + its L4 source validator with the encoding columns
   (`image_file_format`, `pixel_dtype`, `downsample_factor`, `jpeg_quality`, and updated
   `image_width_px`/`image_height_px` semantics — at downsample 4 these are POST-downsample
   dimensions, the spec needs to be explicit that `native_height_px`/`native_width_px` carry the
   pre-downsample values if those columns are added).
6. Smoke: there's already a ready-made overlay config —
   `src/data_pipeline/pipeline_orchestrator/config_smoke_zstack_20250912.yaml` (B01 + C01,
   `smoke_max_time_indices: 3`, both `BF__projection__focus_stack` and `BF__z_stack` products).
   Use it as the base for the real run.

**Important consequence of downsample 4 for `motion_blur_qc` (section 3 below):** the spec's "no
mask alignment needed at downsample 1" assumption (which is what let `focus_qc` skip alignment
entirely) does NOT hold for z-stacks once downsample=4 is the default — mask vs. z-plane pixel
dims will differ by design. `compute_motion_blur_qc` MUST do nearest-neighbor mask alignment to
the loaded z-plane shape, exactly as the spec already says (`z_stack_focus_motion_blur_qc_and_slice_selection.md`,
"align the mask to the loaded z-plane shape if needed, using nearest-neighbor mask semantics").
Do not let this slip because focus_qc didn't need it.

## 2. Submit the z-stack materialization run as an SGE job

There's a proven SGE submission pattern already in the repo to mirror:
`src/data_pipeline/pipeline_orchestrator/submit_tier2_through_line.sge` (GPU node, conda env
`segmentation_grounded_sam`, `PYTHONPATH` exported manually because the Snakefile imports
`data_pipeline` at parse time, `snakemake --configfile config.yaml <overlay>.yaml --cores N
--rerun-triggers mtime --keep-going --printshellcmds <target>`).

For this run: base it on `config_smoke_zstack_20250912.yaml` (already has the right
`target_wells`/`products` block) layered with whatever encoding-default config change you make in
step 1. Z-stack materialization itself is CPU-only per that config's own comment
(`focus_stack projection requires GPU; z_stack is CPU-only`) — only request GPU if you're also
materializing the focus_stack projection product in the same run (you probably are, since
`focus_qc` needs it too). Write a new `submit_zstack_materialization.sge` (or extend the tier2 one)
rather than running interactively — the user explicitly asked for a submitted job, not a foreground
run. Don't reuse `submit_tier2_through_line.sge` unmodified; its target is `through_line` which
pulls in everything downstream through `snip_qc`, not just materialization.

Pick a real target: probably `frame_inventory` merged artifact for B01+C01 (or whatever wells are
configured) is enough to prove z-stack rows exist and validate before going further — don't pull
the whole `through_line` target into this job.

## 3. motion_blur_qc spec — fine-tuning review (do this WITH the user, not solo)

Full spec section: `docs/data_pipeline/specs/target/specs/quality_control/z_stack_focus_motion_blur_qc_and_slice_selection.md`,
`## motion_blur_qc` (~line 210 onward). Current locked defaults, as written:

```text
mask_pixel_pair_ncc[z] = NCC(z_plane[z][mask], z_plane[z+1][mask])
bad_z_pair = mask_pixel_pair_ncc < 0.90
mask_pixel_bad_pair_frac = fraction(valid bad_z_pair values)
motion_blur_flag = mask_pixel_bad_pair_frac > 0.10
```

Evaluation policy: require ≥2 z planes; require non-empty mask after alignment; require ≥1
adjacent pair with nonzero in-mask variance; exclude flat/constant pairs from the denominator but
count them in the output; fail loud if no valid adjacent pair remains.

Contract (already drafted, matches what the Chunk B plan referenced):
```text
MOTION_BLUR_QC_PAYLOAD_COLUMNS = (
    "mask_pixel_ncc_mean", "mask_pixel_ncc_min", "mask_pixel_ncc_p05",
    "mask_pixel_bad_pair_frac", "mask_pixel_longest_bad_run",
    "n_z_pairs", "n_valid_z_pairs", "n_flat_z_pairs", "n_mask_pixels",
    "motion_blur_flag",
)
```

### Things worth scrutinizing with the user before implementing (not yet decided — surface these,
don't silently implement)

- **`ncc_min_threshold=0.90` and `bad_pair_frac_threshold=0.10` were carried over from the
  research script** (per the project memory: "Best metrics: ncc_min/bad_pair_frac_ncc... Threshold:
  ncc_min < 0.85 OR bad_pair_frac_frac > 0.10" — note the **0.85 vs 0.90** discrepancy between the
  user's earlier motion-artifact-detection memory and this spec's locked `0.90`. Flag this
  explicitly and ask which is current — don't silently pick one.
- **Downsample interaction**: at downsample=4, NCC over a much smaller pixel count per snip mask
  is noisier — small masks could have very few in-mask pixels per z-plane, making `mask_pixel_ncc`
  unstable. Worth asking: should there be a `min_mask_pixels_for_ncc` guard (analogous to
  `focus_qc`'s `min_interior_pixels`) so tiny/edge snips don't produce a flaky verdict instead of
  failing loud or excluding cleanly?
- **`mask_pixel_longest_bad_run`** is in the contract but has no defined computation in the spec
  prose above it — needs an explicit definition (longest consecutive run of `bad_z_pair=True` along
  the sorted z-axis, presumably) before compute.py can be written. Pin this with the user.
- **Flat-pair exclusion criterion** ("exclude flat/constant pairs from the denominator") has no
  numeric threshold defined (e.g. "in-mask variance < eps") — needs a concrete cutoff, mirroring
  how `focus_qc` defines `min_interior_pixels` precisely rather than vaguely.
- **Honest framing, same as focus_qc**: this is a heuristic too (adjacent-z-plane disagreement
  inside a possibly-coarse mask), not a ground-truth motion detector. Worth carrying the same
  "calibration-subject" language into `config.py`'s docstring that `focus_qc/config.py` uses.
- Per the locked Chunk-C boundary: build `motion_blur_qc`'s contract/config/compute with **unit
  tests on synthetic z-stack rows only**. Do NOT wire `motion_blur_flag` into
  `SNIP_QC_EXCLUSION_FLAGS` or add `motion_blur_qc` to `flag_input_resolver._SOURCE_PAYLOADS` until
  the materialization run in section 2 is proven live AND motion_blur_qc validates on real data.

## Suggested order for the next session

1. Resolve the encoding-module naming question with the user (don't guess).
2. Build the encoding resolver + wire into YX1 z-stack materializer, default `downsample_factor: 4`.
3. Update `frame_inventory_contract.py` + L4 validation for the new encoding columns.
4. Submit the SGE job for B01+C01 z-stack + projection materialization; confirm `frame_inventory`
   rows validate with the new encoding columns once it completes.
5. THEN sit down with the user on the motion_blur_qc fine-tuning questions above before writing
   `compute.py` — several numeric/semantic gaps need pinning, not guessing.
6. Build `motion_blur_qc` (Chunk C) to the same 5-file template as `focus_qc`, including the
   z-stack reader functions in `materialized_image_readers.py`
   (`resolve_z_stack_rows_from_image_id` / `load_z_stack_images_from_image_id` — specced in the
   approved plan but not yet built since focus_qc only exercised the projection path).
7. Leave the `snip_qc` wire-in for a separate, later step — explicitly gated as described above.
