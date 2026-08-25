eady to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Step 3 — stitch_well_candidate beside legacy

 Context

 Steps 1 (well_discovery/) and 2 (frame_inventory_contract.py) are shipped and verified. Step 3 is
 the first risky cut: grow a per-well YX1 stitcher beside the legacy experiment-grain stitcher,
 prove it on well B01 of experiment 20250912, and emit a candidate frame-inventory CSV. The legacy
 path stays green. No Snakemake wiring, no well_runner, no registry row — standalone CLI only
 for the B01 smoke.

 The roadmap constraint: candidate images write to isolated paths (under candidate/) so they
 can never collide with legacy output. The materializer emits frame-inventory rows as it writes
 image files (no post-hoc tree scan).

 ---
 Architecture: three clean layers

 env.yaml
   → DATA_ROOT (Snakefile)
   → BUILT_IMAGE_DATA_DIR = DATA_ROOT / "built_image_data" (Snakefile / paths.py)

 paths.py   owns: done sentinel paths, frame_inventory CSV shard paths, merged CSV paths
 layout.py  owns: materialized image file paths under built_image_data/

 frame_inventory.csv  is the bridge: tabular (registry-shaped) but carries source_image_path
                      which points into the off-registry layout.py tree

 layout.py receives built_image_data_dir already resolved — it does not know about DATA_ROOT
 or the "built_image_data" stage folder name. That is orchestration, not layout.

 ---
 Locked image layout (2026-06-17)

 built_image_data/
   {experiment_id}/
     materialized_images/
       {well_id}/
         projection/          ← one 2D frame per well × channel × time
           {channel_id}/
             {image_id}.png
         z_stack/             ← one 2D frame per well × channel × z × time  (future)
           {channel_id}/
             {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}.png

 projection/ vs z_stack/ encodes image product shape, not method. channel_id stays
 optical/biological (BF, GFP). The method (focus_stack, max_projection) lives in the
 frame_inventory CSV, not the path.

 Candidate prefix: materialized_images/candidate/{well_id}/projection/... — structural
 isolation from the live tree (materialized_images/{well_id}/projection/... at Step 6).

 ---
 Candidate output for Step 3 (standalone, no registry row)

 Step 3 does NOT add a stitch_well_candidate registry row. The candidate runs standalone and
 writes to explicit CLI-supplied paths. No well_runner, no per_well/ shape.

 images:         built_image_data/{exp}/materialized_images/candidate/{well_id}/projection/BF/*.png
 frame_inventory: --frame-inventory-csv (explicit CLI arg, written wherever caller says)
 done sentinel:   --done-flag (explicit CLI arg)

 Future shape note (Step 5–6, when fan is real): the frame_inventory shard must move to the
 standard per-well registry path so well_runner can collect and merge it:
 built_image_data/{exp}/per_well/{well_id}/frame_inventory__{well_id}.csv
 At that point a stitch_well_candidate registry row is added with fanout=PER_WELL and standard
 per_well/ layout. For Step 3 this is deferred.

 ---
 What exists (reuse)

 ┌──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┬──────────────────┐
 │                    Symbol                    │                                 File                                 │       Role       │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ _get_stack, _focus_stack,                    │                                                                      │ YX1 image        │
 │ _determine_bf_channel                        │ image_building/scope/yx1/stitched_ff_builder.py                      │ primitives —     │
 │                                              │                                                                      │ import directly  │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ LoG_focus_stacker, im_rescale                │ image_building/shared/log_focus.py                                   │ Focus stacking   │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ validate_yx1_acquisition_inventory           │ metadata_ingest/scope/yx1/acquisition_inventory.py                   │ Input contract   │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ REQUIRED_FRAME_INVENTORY_COLUMNS,            │                                                                      │                  │
 │ derive_well_id,                              │ image_materialization/stitched/contracts/frame_inventory_contract.py │ Output contract  │
 │ assert_derived_ids_consistent                │                                                                      │                  │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ build_image_id, validate_well_id             │ shared/identifiers/constructors.py + validators.py                   │ ID grammar       │
 └──────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┴──────────────────┘

 ---
 Files to create / edit

 1. src/data_pipeline/image_materialization/stitched/layout.py (new)

 One generic constructor; one thin projection wrapper now; z_stack wrapper deferred.

 ALLOWED_IMAGE_PRODUCT_TYPES = {"projection", "z_stack"}

 def materialized_image_path(
     built_image_data_dir: Path,
     *,
     experiment_id: str,
     well_id: str,
     channel_id: str,
     time_index: int,
     image_product_type: str,      # "projection" | "z_stack"
     z_index: int | None = None,
     ext: str = "png",
     candidate: bool = False,
 ) -> Path:
     """Resolve the pixel-file path for one materialized frame.

     Pure path construction — no disk I/O. `built_image_data_dir` is the stage root
     (DATA_ROOT / "built_image_data"), resolved by the caller before this is called.

     projection: z_index must be None.   filename = {image_id}.{ext}
     z_stack:    z_index must be an int. filename = {well_id}_{channel_id}_z{z_index:04d}_t{time_index:04d}.{ext}
     candidate=True inserts 'candidate/' between 'materialized_images/' and '{well_id}/'.
     """

 Enforced:
 - image_product_type in ALLOWED_IMAGE_PRODUCT_TYPES (fail loud with list of known types)
 - projection + non-None z_index → raise
 - z_stack + z_index=None → raise
 - well_id validated via validate_well_id
 - image_id = build_image_id(well_id, channel_id, time_index) (projection filename)
 - ext validated against ALLOWED_IMAGE_SUFFIXES from frame_inventory_contract

 Thin wrapper (now):
 def projection_frame_path(
     built_image_data_dir: Path,
     *,
     experiment_id: str,
     well_id: str,
     channel_id: str,
     time_index: int,
     ext: str = "png",
     candidate: bool = False,
 ) -> Path:
     return materialized_image_path(
         built_image_data_dir,
         experiment_id=experiment_id,
         well_id=well_id,
         channel_id=channel_id,
         time_index=time_index,
         image_product_type="projection",
         z_index=None,
         ext=ext,
         candidate=candidate,
     )

 # z_stack_frame_path deferred — add when z_index enters the identity grammar

 2. src/data_pipeline/image_materialization/stitched/scope/__init__.py (new, empty)

 3. src/data_pipeline/image_materialization/stitched/scope/yx1/__init__.py (new, empty)

 4. src/data_pipeline/image_materialization/stitched/scope/yx1/materialize_yx1_stitched_images.py (new)

 Two named primitive functions (simple, no class, no dispatch table):

 def materialize_ff_projection(stack_zyx: np.ndarray, *, device: str) -> np.ndarray:
     """Focus-stack a Z-stack into one 2D frame (BF / brightfield projection method).

     Returns uint8 2D image. This is the projection_method='focus_stack' primitive.
     Does not know about well_id, time_index, or paths — pure image math.
     """
     norm, _, _ = im_rescale(stack_zyx)
     ff, _ = LoG_focus_stacker(norm.astype(np.float32), filter_size=3, device=device)
     arr = np.clip(ff.cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff), 0, 65535)
     return skimage.util.img_as_ubyte(arr.astype(np.uint16))


 def materialize_max_projection(stack_zyx: np.ndarray) -> np.ndarray:
     """Max-project a Z-stack into one 2D frame (fluorescence projection method).

     Returns same dtype as input. This is the projection_method='max_projection' primitive.
     Deferred — defined here now so the naming convention is locked; used when GFP lands.
     """
     return stack_zyx.max(axis=0)

 Per-well orchestrator (calls the primitives, writes files, emits inventory rows):

 def materialize_yx1_well(
     *,
     experiment_id: str,
     well_id: str,
     well_index: str,                               # local label e.g. "B01"
     well_acquisition_inventory_df: pd.DataFrame,   # pre-filtered to ONE well; owns position_index
     nd2_path: Path,
     built_image_data_dir: Path,                    # BUILT_IMAGE_DATA_DIR from caller
     device: str = "cuda",
     candidate: bool = True,
 ) -> pd.DataFrame:                                 # frame-inventory rows (flat schema)

 Entry guard — fail immediately:
 assert derive_well_id(experiment_id, well_index) == well_id
 assert not well_acquisition_inventory_df.empty
 assert well_acquisition_inventory_df["position_index"].nunique() == 1
 assert well_acquisition_inventory_df["source_nd2_path"].nunique() == 1

 Implementation:
 1. position_index = int(well_acquisition_inventory_df["position_index"].iloc[0])
 2. Open ND2 (nd2_path), build dask array; _determine_bf_channel for channel index.
 3. For each time_index in well_acquisition_inventory_df["time_index"].unique():
   - Extract Z-stack: _get_stack(dask_arr, t=time_index, w=position_index).
   - Project: ff = materialize_ff_projection(stack, device=device).
   - Resolve path: layout.projection_frame_path(built_image_data_dir, experiment_id=experiment_id, well_id=well_id, channel_id="BF",
 time_index=time_index, candidate=candidate).
   - Write: skio.imsave(path, ff, check_contrast=False) (create parent dirs).
   - Append row to frame-inventory list.
 4. Return pd.DataFrame(rows).

 Frame-inventory flat schema (all columns, decided now):
 experiment_id             atom / key
 well_index                atom / key
 channel_id                atom / key   ("BF" for Step 3)
 time_index                atom / key
 z_index                   pd.NA for projection rows; int for z_stack rows
 image_product_type        "projection"
 projection_method         "focus_stack"
 source_image_path         str path from layout.projection_frame_path(...)
 source_micrometers_per_pixel
 image_width_px
 image_height_px
 Derived (well_id, image_id) are NOT emitted — contract validator recomputes and checks them.

 Step 3 writes BF only — documented in module docstring, not encoded in function name.

 5. src/data_pipeline/pipeline_orchestrator/tasks.py (edit)

 Thin dispatcher only. Explicit --built-image-data-dir arg — no reconstruction from output_root:

 def cmd_materialize_yx1_well_candidate(args: argparse.Namespace) -> None:
     from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import load_yx1_acquisition_inventory
     from data_pipeline.image_materialization.stitched.scope.yx1.materialize_yx1_stitched_images import materialize_yx1_well

     acq_df = load_yx1_acquisition_inventory(Path(args.acquisition_inventory_csv))
     well_rows = acq_df[
         (acq_df["experiment_id"] == args.experiment)
         & (acq_df["well_index"] == args.well_index)
             (acq_df["experiment_id"] == args.experiment)
             & (acq_df["well_index"] == args.well_index)
         ]
         inv_df = materialize_yx1_well(
             experiment_id=args.experiment,
             well_id=args.well_id,
             well_index=args.well_index,
             well_acquisition_inventory_df=well_rows,
             nd2_path=Path(args.nd2_path),
             built_image_data_dir=Path(args.built_image_data_dir),   # explicit; no reconstruction
             device=getattr(args, "device", "cuda"),
         )
         inv_df.to_csv(args.frame_inventory_csv, index=False)
         Path(args.done_flag).touch()

     CLI args for the B01 smoke:
     --experiment          20250912
     --well-id             20250912_B01
     --well-index          B01
     --acquisition-inventory-csv   <path to acquisition_inventory__yx1.csv>
     --nd2-path            <path to .nd2 file>
     --built-image-data-dir        <BUILT_IMAGE_DATA_DIR>
     --frame-inventory-csv <output CSV path>
     --done-flag           <output sentinel path>

     No --output-root — ambiguous. Caller passes the already-resolved stage dir.

     ---
     Tests to add

     tests/data_pipeline/image_materialization/stitched/test_layout.py (new)
     - projection_frame_path(..., candidate=True) path contains materialized_images/candidate/.
     - projection_frame_path(..., candidate=False) path omits candidate/.
     - Filename is canonical image_id form: {well_id}_{channel_id}_t{time_index:04d}.png.
     - materialized_image_path(..., image_product_type="projection", z_index=1) raises.
     - materialized_image_path(..., image_product_type="z_stack", z_index=None) raises.
     - materialized_image_path(..., image_product_type="bad_type") raises with known-types list.
     - Bare well_index ("B01") instead of global well_id raises via validate_well_id.
     - No disk I/O in any test.

     tests/data_pipeline/image_materialization/stitched/scope/yx1/test_materialize_yx1.py (new)
     - Unit tests mock ND2 + image ops — no GPU or real data required.
     - materialize_ff_projection on a synthetic Z-stack returns a 2D uint8 array.
     - materialize_yx1_well returned DataFrame has exactly the flat schema columns.
     - No duplicate rows on (experiment_id, well_index, channel_id, time_index).
     - source_image_path values match layout.projection_frame_path(...) for those atoms.
     - Mismatched well_id/well_index/experiment_id raises at entry guard.
     - Ambiguous position_index (nunique > 1) raises at entry guard.

     ---
     Verify (gate before committing Step 3)

     1. Unit tests pass:
     PYTHONPATH=src conda run -n segmentation_grounded_sam --no-capture-output python -m pytest \
       tests/data_pipeline/image_materialization/stitched/ -v
     2. B01 smoke (standalone CLI):
       - Images land under {BUILT_IMAGE_DATA_DIR}/20250912/materialized_images/candidate/20250912_B01/projection/BF/.
       - frame_inventory__20250912_B01.csv exists with flat schema, correct row count.
       - Done sentinel written.
       - Legacy {BUILT_IMAGE_DATA_DIR}/20250912/stitched_ff_images/ untouched (no collision).
     3. Snakemake dry-run parses: snakemake -n (no registry change in Step 3, so this should be trivial).

     ---
     Commit message

     feat: add stitch_well_candidate beside legacy; layout.py + per-well YX1 materializer (B01 smoke)

     ---
     What this step does NOT do

     - No stitch_well_candidate registry row (Step 5, when fan is real).
     - No Snakemake rule. No well_runner. No per_well/ shard layout.
     - No fluorescence / max-projection (BF only; materialize_max_projection stub defined but not called).
     - No comparison gate (Step 4). No fan (Step 5). No promotion (Step 6).
     - No repointing of segmentation / frame_contract consumers.
     - No layout.py → subpackage promotion.

● Now implementing. I'll work through the files in order, starting with layout.py and the __init__.py files, then the materializer, then tests.

  Read 1 file, listed 3 directories (ctrl+o to expand)