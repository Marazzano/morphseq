     Plan: z-stack write-policy + materialization (Pass 1) — motion_blur_qc deferred to Pass 2

     Context
  
     The Snakemake streamline refactor has focus_qc (Chunk B) built and wired into snip_qc. The next
     chunk needs z-stack pixels persisted so post-segmentation motion_blur_qc can run — but PNG
     z-stacks explode storage (~619 GB across 3 experiments per the spec's probes). The fix is a
     per-product write policy (format/quality/downsample/dtype) that lets BF__z_stack materialize as
     JPEG q85 at downsample_factor 4 (16× fewer pixels — the real storage lever). This policy layer
     does not exist in code yet — only in spec docs. So "set downsample to 4" is not a config flip;
     the policy module must be built, threaded into the YX1 materializer, and the frame_inventory
     contract extended to record on-disk reality before any of it takes effect.

     Two passes — strict dependency chain, do not reverse

     ▎ write policy → materialized z-stack files → validated frame_inventory rows → motion_blur_qc

     Pass 1 (THIS plan, approved): write policy + materializer threading + config + frame_inventory
     contract & L4 + SGE smoke + docs. Produce real BF__z_stack rows at ds4 and prove they validate.

     Pass 2 (NOT this plan): build motion_blur_qc against the validated real row shape — only after
     Pass 1's rows exist and the user resolves the open QC-threshold/guard questions. Do not implement
     any motion_blur_qc paths, rules, commands, tests, or snip_qc wiring in Pass 1. Reason: motion_blur
     depends on the exact post-downsample z-stack row shape; writing it against imagined rows risks
     discovering L4 changed dim meaning or z grouping.

     Decisions locked with the user

     - Module name: materialized_image_write_policy.py (type ImageWritePolicy), the "how products
     are encoded/written" sibling to materialized_image_paths.py (where written) and
     materialized_image_readers.py (how read back). Not the spec's vague image_encoding.py.
     - Dim columns: source_ prefix = native/provenance, plain name = on-disk reality — reuses the
     existing source_image_path / source_micrometers_per_pixel convention.
     - SGE run: target front_half, wells B01+C01, GPU node; "purely a write + transform op."
     - downsample_factor: 4 default for BF__z_stack.

     Design doctrine — keep the policy MINIMAL (user direction)

     The write policy declares format, scale, dtype — the minimal facts to write the file and later
     interpret the derived bytes. It is NOT an image-transformation ontology. No compression_mode, no
     intensity_mapping. Five fields only; format implies the writer route.

     ▎ policy says format/scale/dtype → writer handles the route → inventory records the derived bytes.

     Pipeline: downsample → convert dtype → route-specific write. No extra knobs until a real fork appears.

     1. convert_pixel_dtype = standard FIXED dtype-range conversion only, NOT per-image/per-plane
     min/max contrast stretch (the latter injects artificial adjacent-z NCC similarity). uint16→uint8
     = fixed range; uint8→uint8 / uint16→uint16 = identity. This is a function rule, not a
     config field; the docstring states "fixed conversion, no per-image normalization," and a test
     asserts two planes with equal relative intensities map to equal uint8 (i.e. not per-plane stretched).
     2. Downsample FIRST, then dtype-convert (downsample on the richer dtype before quantization).
     3. downsample_method IS a recorded field because none/block_mean/decimation are materially
     different transforms — but it has a documented default so configs needn't state it (base
     default = none; BF__z_stack default = block_mean). block_mean requires native dims
     divisible by factor → fail loud otherwise; none/factor-1 is identity. Output shape comes
     from one func expected_downsampled_dims(w, h, factor, method) that both downsample_image and L4
     call — no duplicated shape math, no implicit source == image*factor assertion.
     4. pixel_dtype semantics: dtype handed to the encoder / expected on read-back (uint8 for jpg) —
     NOT the raw acquisition dtype. Document in the contract comment.
     5. Readers unchanged: Pass-2 motion_blur_qc and all consumers read the recorded source_image_path;
     never reconstruct paths or re-derive encoding from config.

     Code facts verified this session (handoff was partly wrong)

     - Materializer: scope/yx1/materialize_well_yx1.py (not top-level). Writes = the two
     skio.imsave calls at lines 355 (projection) / 419 (z_stack), inside
     materialize_yx1_product_for_well (def 168). No config flows in today — add a param.
     - z-stack readers already exist (resolve_z_stack_rows_from_image_id readers.py:97,
     load_z_stack_images_from_image_id readers.py:235; encoding-agnostic via
     cv2.imread(IMREAD_UNCHANGED)). Do not rebuild — Pass 2 consumes them.
     - ext is already a validated path param (default "png"); .jpg/.tif already in
     ALLOWED_IMAGE_SUFFIXES (frame_inventory_contract.py:107). suffix_for_policy feeds ext=.
     - dtype divergence: projection writes uint8 (img_as_ubyte); z_stack writes raw uint16 ND2 slice.
     - L4 dim self-check is the crux: validate_sources (frame_inventory_validation_rules.py:255-262)
     asserts declared image_*_px == PIL header dims. At ds4 the file is 1/4 size, so this fails today
     unless image_*_px becomes on-disk dims.

     ---
     §1. Build materialized_image_write_policy.py

     New file: src/data_pipeline/image_materialization/materialized_image_write_policy.py.

     @dataclass(frozen=True)
     class ImageWritePolicy:
         file_format: Literal["png", "jpg", "tif"]
         downsample_factor: int                              # 1 | 4 (2 allowed)
         downsample_method: Literal["none", "block_mean"]    # recorded; has a default
         pixel_dtype: Literal["uint8", "uint16"]             # OUTPUT dtype / read-back dtype
         jpeg_quality: int | None = None                     # required iff jpg; None otherwise

     def resolve_image_write_policy(config: dict, product_key: str) -> ImageWritePolicy: ...
     def suffix_for_policy(policy) -> str: ...                       # "png"/"jpg"/"tif" → ext=
     def expected_downsampled_dims(w, h, factor, method) -> tuple[int,int]: ...  # ONE source of truth
     def downsample_image(image, policy) -> np.ndarray: ...         # none = identity; block_mean per method
     def convert_pixel_dtype(image, pixel_dtype) -> np.ndarray: ...  # FIXED dtype conversion, no normalization
     def prepare_image_for_write(image, policy) -> np.ndarray:       # downsample → convert dtype
         image = downsample_image(image, policy)
         return convert_pixel_dtype(image, policy.pixel_dtype)
     def write_image(image, path, policy) -> None:                  # prepare, then route on file_format
         image = prepare_image_for_write(image, policy)
         # jpg → write_jpeg(quality=policy.jpeg_quality); png → write_png; tif → write_tif

     - resolve_image_write_policy keys off the canonical product_key (BF__z_stack,
     BF__projection__focus_stack) from existing build_image_product_key /
     image_product_key_for_resolved_product. Merges image_materialization.write_policies[<key>]
     (NEW, §3) over the locked defaults; unknown override keys → fix-named ValueError (mirror
     focus_qc/config.py::resolve_config). Fills downsample_method from the per-product default when
     omitted.
     - Format implies the route + its validation (no compression_mode):
       - jpg → suffix .jpg/.jpeg, pixel_dtype == uint8, jpeg_quality not None, lossy.
       - png → suffix .png, jpeg_quality is None, lossless (uint8; uint16 if writer supports/validates).
       - tif → suffix .tif/.tiff, jpeg_quality is None, uint8 or uint16.
     - convert_pixel_dtype: fixed dtype-range conversion only (uint16→uint8 by range; identity for
     matching). Docstring: "fixed conversion, no per-image min/max normalization."
     - expected_downsampled_dims: none/factor-1 = identity; block_mean requires divisibility by
     factor, else fail loud. Both downsample_image and L4 call it — no duplicated shape math.
     - Unit tests: resolve defaults/overrides (incl. downsample_method default fill) + unknown-key
     rejection; per-format route validation (jpg requires uint8+quality, png/tif reject quality);
     expected_downsampled_dims block_mean divisibility + fail-loud; downsample shape; fixed-scale
     dtype test (equal relative intensities → equal uint8, NOT per-plane stretched); round-trip per format.

     Defaults table (locked)

     default:                     png, factor 1, method none,       uint8
     BF__z_stack:                 jpg q85, factor 4, method block_mean, uint8
     BF__projection__focus_stack: png, factor 1, method none,       uint8

     §2. Thread policy into the YX1 materializer

     scope/yx1/materialize_well_yx1.py:
     - Add config (or pre-resolved write_policies: dict[str, ImageWritePolicy]) param to
     materialize_yx1_product_for_well (line 168); forward through materialize_yx1_well (133) and the
     run_materialize_well.py caller.
     - Both write branches (projection 345-355, z_stack 409-419): resolve policy for the branch's
     product_key, suffix = suffix_for_policy(policy), pass ext=suffix to
     projection_frame_path/z_stack_frame_path, then write_image(raw, out_path, policy) (which
     internally does downsample → convert dtype → route-write). Replaces the bare skio.imsave.
     - Update both _frame_inventory_row(...) calls (384, 422): plain image_width_px/image_height_px =
     post-downsample dims actually written; source_image_width_px/source_image_height_px = native
     img_w/img_h from acquisition inventory (277-278); plus image_file_format, pixel_dtype,
     downsample_factor, downsample_method, jpeg_quality (null for non-jpg). Update
     _frame_inventory_row signature + _EMITTED_COLUMNS.

     §3. Config: per-product write-policy map

     config.yaml image_materialization: keep the existing list-form products: (selects WHICH
     products to build — orthogonal). Add a sibling map keyed by canonical product_key, consumed only by
     resolve_image_write_policy:
     image_materialization:
       write_policies:
         BF__z_stack:
           file_format: jpg
           jpeg_quality: 85
           downsample_factor: 4
           downsample_method: block_mean
           pixel_dtype: uint8
     Defaults live in the resolver (incl. the downsample_method default); the map only overrides. Do NOT
     overload the list-form products.

     §4. Extend frame_inventory contract + L4 validation

     frame_inventory_contract.py:
     - Add to REQUIRED_FRAME_INVENTORY_COLUMNS (40-54): source_image_width_px,
     source_image_height_px, image_file_format, pixel_dtype, downsample_factor,
     downsample_method. Add jpeg_quality as nullable (follow CONSTRUCTION_PROVENANCE_COLUMNS
     nullable-declared pattern, 79-81). Update inline comments (52-53): image_*_px now =
     on-disk/post-downsample; document pixel_dtype = encoder/read-back dtype, not raw acquisition.
     - DOWNSTREAM_FRAME_IDENTITY_BLOCK (127-137) already carries image_*_px; leave source_image_*_px
     out unless a downstream consumer needs native dims (note it; MVP = out).

     frame_inventory_validation_rules.py::validate_sources (224-272):
     - Existing PIL header check (255-262) now validates against plain on-disk image_*_px — correct
     once those are post-downsample. Add: image_*_px == expected_downsampled_dims(source_image_*_px, downsample_factor) (single-source shape doctrine from §1); source_image_path suffix matches
     image_file_format; jpg ⇒ non-null jpeg_quality, non-jpg ⇒ null; downsample_factor >= 1.

     §6. Submit the SGE materialization smoke

     New src/data_pipeline/pipeline_orchestrator/submit_zstack_materialization.sge, modeled on
     submit_tier2_through_line.sge (conda segmentation_grounded_sam, manual PYTHONPATH=src export,
     GPU -l gpgpu=TRUE,cuda=1). Do NOT reuse the tier2 script — its target is through_line (pulls
     through snip_qc); this run is write+transform only.
     snakemake --configfile config.yaml config_smoke_zstack_20250912.yaml \
       --cores 8 --rerun-triggers mtime --keep-going --printshellcmds front_half
     - Wells B01+C01, both products (z_stack jpg q85 ds4 + projection png), cap 3 tp. GPU (focus_stack
     projection needs it). After completion: inspect per-well validated frame_inventory shards —
     confirm BF__z_stack rows exist, carry the new encoding/dim columns, and pass L4 (on-disk dims
     == file header; image_*_px == expected_downsampled_dims(source_*_px, 4)). Spot-check a written
     JPEG plane is uint8, 1/4 dims, visually intact.

     §7. Doc updates

     - image_encoding_materialization_config.md: rename symbols to write-policy vocabulary
     (ImageWritePolicy 5-field shape, resolve_image_write_policy, …); drop compression_mode/
     intensity_mapping framing; BF__z_stack default downsample_factor 1 → 4 + method block_mean in Config Shape / Defaults / Locked MVP, with the 16×-pixel-reduction rationale; state
     the fixed dtype conversion (no per-image normalization) rule, the downsample→convert→route
     order, the block_mean divisibility/fail-loud doctrine, pixel_dtype semantics, and
     source_=provenance / plain=on-disk dim naming.
     - z_stack_focus_motion_blur_qc_and_slice_selection.md: add a forward-note that at ds≠1 motion_blur
     (Pass 2) MUST do nearest-neighbor mask→z-plane alignment; flag the 0.90-vs-0.85 NCC threshold as
     pending user confirmation. (No motion_blur code this pass.)

     ---
     Deferred to Pass 2 (do NOT touch in Pass 1)

     motion_blur_qc build-out (clone of focus_qc/focus_qc.smk, PIPELINE_STEPS entry, tasks command,
     tests) AND any snip_qc wiring. Gated on: (a) real ds4 z-stack rows validated by §6, AND (b) user
     resolving — NCC threshold 0.90 vs 0.85 (spec vs memory), min_mask_pixels_for_ncc guard,
     mask_pixel_longest_bad_run definition, flat-pair variance epsilon. The standing TODO at
     snip_qc/contract.py:38-40 already forbids adding motion_blur_flag to SNIP_QC_EXCLUSION_FLAGS
     until then (it would make motion_blur a hard DAG gate for every snip_qc run).

     Execution order (Pass 1)

     1. §1 module + unit tests (incl. fixed-scale + divisibility tests).
     2. §2 thread into materializer (downsample-first) → §3 config → §4 contract + L4.
     3. §6 SGE smoke; confirm rows validate with new columns. This is the gate.
     4. §7 docs alongside the code they describe.

     Verification

     - materialized_image_write_policy unit tests: resolve/suffix; expected_downsampled_dims
     divisibility + fail-loud; downsample shape 1/2/4; fixed-scale uint16→uint8 (no per-plane stretch);
     round-trip per format.
     - Run with PYTHONPATH=src + --import-mode=importlib (repo requires both):
     conda run -n segmentation_grounded_sam --no-capture-output python -m pytest ....
     - §6 SGE run is the end-to-end proof: real B01+C01 z-stack JPEGs at ds4 land on disk; per-well
     frame_inventory validates L0–L4 with new encoding/dim columns; readers still load via recorded
     source_image_path.

