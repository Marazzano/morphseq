"""Pipeline-wide artifact-path registry (the ORCHESTRATION kingdom).

WHY THIS FILE EXISTS — it answers one question: *where does each file the pipeline produces
land?* Both the Snakefile (at parse time) and the Python entrypoints (at run time) import from
here, so rule-output paths and code-output paths are guaranteed identical.

THE VOCABULARY (used consistently across the whole refactor — do not blur these):

    stage     a PHASE of the pipeline, and the top-level FOLDER under output_root.
              e.g. ``acquisition``, ``object_extraction``, ``quality_control``.
              This is "where am I in the pipeline" from a user's seat. MANY steps share a stage.

    step      one unit of work WITHIN a stage — a function / ``tasks.py`` verb's OUTPUT SLOT.
              e.g. ``ingest_scope_metadata``, ``frame_inventory``. These are the
              REGISTRY KEYS (``PIPELINE_STEPS`` below).
              A step key names a logical output slot, not a Snakemake action.
              Where a step is touched by multiple rules, use a neutral noun-like name rather than
              a producer verb — e.g. ``materialize_well`` (the live per-well producer) and
              ``validate_frame_inventory_for_well`` both reference the single ``frame_inventory``
              step. Some front-end keys remain verb-shaped because they are locked by
              front_end_naming_and_flow.md.

    artifact  one FILE a step produces. A step can produce several, so each step's ``artifacts``
              is a dict (e.g. ``mapping`` -> ``position_well_mapping.csv``). The ``.validated``
              sentinel and ``.provenance.json`` sidecar are DERIVED from an artifact (helpers
              below), not separate artifacts.

So the layout is::

    output_root / {stage} / {experiment} / [{product_dir}/] [per_well/{well_id}/] {artifact}
       │             │           │               │                  │                 │
      root         STAGE      experiment    PRODUCT DIR       per-well shape       a step's
    (env.yaml)  (regime)     (caller)      (step's folder)   (step's fanout)        FILE

``product_dir`` is optional. When present it is the named product folder directly under
``{stage}/{experiment}/``, separating distinct product families within a regime (e.g.
``frame_inventory/`` and ``materialized_images/`` both live inside ``acquisition/``).
When absent, artifacts land directly under ``{stage}/{experiment}/``.

⛔ NO-LEAKAGE BOUNDARY (the hard rule for this file). ``paths.py`` only ever **SUBSTITUTES**
caller-provided identity tokens (``experiment_id``, ``well_id``) into filename templates. It
**NEVER MINTS OR DERIVES** an identity value — no ``f"{exp}_{well}"``, no splitting a well_id, no
stuffing one token into another's slot. Minting/parsing identity is the IDENTITY kingdom's job
(``shared/identifiers/``); anything in here that needs to *build* an id must import from there.
Identity flows *into* orchestration, never the other way.

THREE THINGS KEPT DELIBERATELY SEPARATE (see
docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md and
target/front_end_naming_and_flow.md):

    Root      output_root            env.yaml (machine-specific)    -> passed in here as `root`
    Layout    {stage}/{exp}/...      THIS FILE (a code contract)    -> PIPELINE_STEPS + helpers
    Selection which wells/features   config.yaml (a science choice) -> NOT here

⚠️ FORWARD DECLARATION (2026-06-06, updated 2026-06-07). This registry names the **TARGET**
artifacts (``frame_inventory``, ``discovered_wells.txt``, the ``ingest_*``/``join_*`` step keys).
The live root Snakefile now uses these registry rows for the front-end metadata/discovery flow,
while the back half still emits the legacy ``frame_contract.csv`` family until Scope 5 lands.
The registry remains the **contract those scopes converge onto**; wire each remaining step to this
registry as it is reconstructed.

Coverage: **front-end steps only** for now (the two metadata ingest lineages, well discovery,
and the post-fan frame-inventory tail). Back-half steps (segmentation, snips, aux,
feature_extraction, QC, analysis_ready) get rows when Scope 5 specifies their stage/grain. Spec:
target/front_end_naming_and_flow.md (§ PATHS.PY REGISTRY ROWS).
Audit: target/frame_inventory_well_runner_audit.md (review of the registry + the frame_inventory
adapter + well-runner that consume it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# The fixed sub-directory name for one well's shard. Part of the path contract, not a magic
# string scattered in the helpers.
PER_WELL_DIRNAME = "per_well"

# Grain of a step's outputs (its "fanout") — and which path_modes each grain permits:
#   "experiment"          -> one artifact per experiment. Allowed path_mode: experiment.
#   "per_well_then_merge" -> a per-well shard at per_well/{well_id}/, concatenated to an
#                            experiment-level merged view. Allowed path_mode: per_well | merged.
EXPERIMENT = "experiment"
PER_WELL_THEN_MERGE = "per_well_then_merge"

# How to resolve a path for a given call. Each step's `fanout` constrains which of these is legal
# (enforced by _normalize_path_mode):
#   "experiment" -> {stage}/{exp}/{file}                      (the experiment-grain artifact)
#   "per_well"   -> {stage}/{exp}/per_well/{well_id}/{file}   (one well's shard)
#   "merged"     -> {stage}/{exp}/{file}                      (the concatenated experiment view)
# Note: "experiment" and "merged" land in the same directory, but they are kept distinct so a
# per_well_then_merge step's merged view is requested as "merged" (explicit intent), never
# "experiment".
PATH_MODE_EXPERIMENT = "experiment"
PATH_MODE_PER_WELL = "per_well"
PATH_MODE_MERGED = "merged"

# Which path_modes each fanout allows.
_ALLOWED_PATH_MODES: dict[str, tuple[str, ...]] = {
    EXPERIMENT: (PATH_MODE_EXPERIMENT,),
    PER_WELL_THEN_MERGE: (PATH_MODE_PER_WELL, PATH_MODE_MERGED),
}

# Execution model — how many processes compute a step's per-well shards.
# This is metadata for the Snakemake rule author; it does not affect path construction.
#
#   EXECUTION_PER_WELL  — one job per well shard (the default; most steps). The rule uses
#                         expand() over run wells; each invocation writes one well's shard.
#   EXECUTION_RUN_BATCH — one job for the whole run set, writes ALL well shards before
#                         exiting. The rule is a batch rule whose inputs are the full run set
#                         and whose outputs are all run-well shards. Used when job-startup cost
#                         (model load, GPU init) dominates per-well encode cost — e.g. SAM2
#                         segmentation, legacy VAE embeddings.
#
# Only PER_WELL_THEN_MERGE steps may use EXECUTION_RUN_BATCH (a batch step that does not
# produce per-well shards is incoherent). EXPERIMENT-grain steps always use EXECUTION_PER_WELL.
EXECUTION_PER_WELL  = "per_well"
EXECUTION_RUN_BATCH = "run_batch"


# ──────────────────────────────────────────────────────────────────────────────────────────
# THE REGISTRY (front-end steps — TARGET names)
# ──────────────────────────────────────────────────────────────────────────────────────────
# *** THIS REGISTRY IS THE CONTRACT. ***
# Every artifact path in the pipeline is computed from this one dict — the Snakefile reads it at
# parse time, the Python entrypoints read it at run time. If a filename or stage is wrong HERE,
# it is wrong EVERYWHERE, consistently; if two places ever disagree about where a file lives, the
# bug is that one of them bypassed this registry. NEVER hardcode an artifact path elsewhere — add
# or edit a row here and call the helper functions below. Adding a step = adding one row.
#
# Per row (keyed by STEP):
#   stage     : the pipeline phase / top-level folder under output_root (many steps share one)
#   fanout    : the step's grain (EXPERIMENT | PER_WELL_THEN_MERGE)
#   artifacts : {artifact_key: template}. A template is EITHER a single string (same filename in
#               every legal path_mode) OR a {path_mode: string} dict when the filename differs by
#               mode (e.g. a per-well shard names the well, the merged view names the experiment).
#               Templates substitute caller-provided tokens only ({experiment_id}, {well_id}, and
#               any format_vars like {scope}) — see the NO-LEAKAGE BOUNDARY above. Sidecars
#               (.validated, .provenance.json) are DERIVED by the helpers, not listed here.
PIPELINE_STEPS: dict[str, dict] = {
    # ── PLATE LINEAGE (Excel — authored design + plate geometry) ──────────────
    # ingest_metadata/ groups the non-pixel acquisition-side ingest tables (plate + scope metadata,
    # the mapped scope table, and the per-coordinate acquisition_inventory). "ingest_metadata"
    # (not "acquisition_metadata") because the parent regime is already acquisition/ — the qualifier
    # adds info: these are the tables EMITTED BY ingest.
    "ingest_plate_metadata": {
        "stage": "acquisition",
        "product_dir": "ingest_metadata",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {"csv": "plate_metadata.csv"},
    },

    # ── SCOPE LINEAGE (raw microscope file — acquisition facts) ───────────────
    # ingest_scope_metadata emits two artifacts from ONE raw read; both ride one step-level
    # product_dir because both belong under ingest_metadata/ (no artifact-level override needed).
    "ingest_scope_metadata": {
        "stage": "acquisition",
        "product_dir": "ingest_metadata",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        # {scope} -> format_vars={"scope": "yx1" | "keyence"}; the ONLY raw read.
        # acquisition_inventory: the maximal per-coordinate record emitted from that one read
        # (YX1: record-only, scope-shaped — see target/acquisition_inventory_flow.md).
        "artifacts": {
            "raw": "scope_metadata__{scope}.csv",
            "acquisition_inventory": "acquisition_inventory__{scope}.csv",
        },
    },
    # well_identities/ owns position->well identity resolution (the map) and its output (the well
    # list). The MAPPED scope table is a consumer of that identity, not a member — it stays in
    # ingest_metadata/ with the other scope-metadata tables.
    "map_positions_to_wells": {
        "stage": "acquisition",
        "product_dir": "well_identities",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        # .provenance.json via provenance_path().
        "artifacts": {"mapping": "position_well_mapping.csv"},
    },
    "apply_position_to_well_mapping": {  # CONVERGENCE LINE; well_id comes from the position map.
        "stage": "acquisition",
        "product_dir": "ingest_metadata",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        # .validated via validated_path().
        "artifacts": {"mapped": "scope_metadata_mapped.csv"},
    },

    # ── FAN POINT (well discovery — checkpoint) ───────────────────────────────
    "discover_wells": {
        "stage": "acquisition",
        "product_dir": "well_identities",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {"wells": "discovered_wells.txt"},  # one well_id per line
    },

    # ── KEYENCE EXPERIMENT-GRAIN PRE-STEP (stitch map, generated once) ──────
    # Experiment-grain pre-step that samples ~50 well/time pairs, aligns each, takes the median
    # tile coords, and writes the master_params JSON. Consumed per-well by materialize_well via
    # PreComputeStitchParams(master_params_path=...) — no re-alignment needed per frame.
    # {scope} -> format_vars={"scope": SCOPE_TOKEN} (always "keyence" when this step is used).
    "keyence_stitch_map": {
        "stage": "acquisition",
        "product_dir": "ingest_metadata",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {"master_params": "keyence_stitch_map__{scope}.json"},
    },

    # ── PER-WELL MATERIALIZATION (pixel action — done sentinel only) ──────────
    # materialize_well[well_id] writes pixel files and the per-well frame-inventory shard.
    # DOCTRINE: materialize_well owns pixel materialization state (done sentinel under
    # materialized_images/). The frame-inventory CSV path is owned by the frame_inventory step —
    # the action may create the file, but the product owns the contract path.
    "materialize_well": {
        "stage": "acquisition",
        "product_dir": "materialized_images",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "done": {
                PATH_MODE_PER_WELL: "{well_id}.materialize_well.done",
            },
        },
    },

    # Resolved per-product execution commitments. The product key is a filename-level token
    # supplied by the caller; no new path mode is needed. Nests under frame_inventory/ because these
    # are scaffolding for building the canonical frame_inventory, not an independently-consumed
    # product. (Step KEY stays `resolved_product_plans`; only the on-disk folder is grouped.)
    "resolved_product_plans": {
        "stage": "acquisition",
        "product_dir": "frame_inventory/resolved_product_plans",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "json": {
                PATH_MODE_PER_WELL: "{product_key}_resolved_product_plan.json",
            },
        },
    },

    # Product-grain frame-inventory shards (one per image product). These are NOT the canonical
    # per-well frame_inventory; assembly unions them into the canonical shard. They live under
    # frame_inventory/product_inventories/ — "product_inventories" because each file is an inventory
    # FOR one image product, not a pixel/product output. (Step KEY stays `frame_inventory_products`.)
    "frame_inventory_products": {
        "stage": "acquisition",
        "product_dir": "frame_inventory/product_inventories",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "inventory": {
                PATH_MODE_PER_WELL: "{well_id}_{product_key}_frame_inventory.csv",
            },
        },
    },

    # Per-well manifest of the active validated product inventories available for assembly into the
    # canonical frame_inventory shard. Folder/filename use the semantic name `available_products`
    # (what the manifest MEANS) rather than the implementation-flavored "discovered_product_shards"
    # (the action that produced it). (Step KEY stays `discovered_product_shards` until the later
    # code-vocabulary commit.)
    "discovered_product_shards": {
        "stage": "acquisition",
        "product_dir": "frame_inventory/available_products",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "csv": {
                PATH_MODE_PER_WELL: "{well_id}_available_products.csv",
            },
        },
    },

    # ── POST-FAN — frame inventory joins the spine ────────────────────────────
    # "frame_inventory" is the product contract (a noun), not a Snakemake action. Several rules
    # touch it: materialize_well WRITES the per-well shard, validate_frame_inventory_for_well
    # writes sentinels, and merge_frame_inventory builds the experiment-level view. The registry
    # keeps ONE noun-like step for the product; actions (verbs) are rules, not registry rows.
    # The per_well shard names the WELL; the merged view names the EXPERIMENT.
    "frame_inventory": {
        "stage": "acquisition",
        "product_dir": "frame_inventory",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "inventory": {
                PATH_MODE_PER_WELL: "{well_id}_frame_inventory.csv",
                PATH_MODE_MERGED: "{experiment_id}_frame_inventory.csv",
            },
        },
    },

    # ── OBJECT EXTRACTION — detections ────────────────────────────────────────
    # Per-well detection shards feed per-well frame_masks. The merged experiment table is useful
    # for audit/reporting, but segmentation consumes the per-well shard for the same well.
    "frame_detections": {
        "stage": "object_extraction",
        "product_dir": "frame_detections",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "frame_detections": {
                PATH_MODE_PER_WELL: "{well_id}_frame_detections.csv",
                PATH_MODE_MERGED: "{experiment_id}_frame_detections.csv",
            },
        },
    },

    # ── OBJECT EXTRACTION — masks ─────────────────────────────────────────────
    # `frame_masks` is the target segmentation product. It fans out per well because SAM2
    # consumes one well's ordered frame view at a time, then merges to an experiment-level table.
    # `prompt_seeds` is a per-well audit sidecar for the detection->segmentation handoff; it is not
    # required as a merged experiment artifact.
    # execution=RUN_BATCH: SAM2 loads its model once and processes all run wells before exiting.
    "frame_masks": {
        "stage": "object_extraction",
        "product_dir": "frame_masks",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_RUN_BATCH,
        "artifacts": {
            "frame_masks": {
                PATH_MODE_PER_WELL: "{well_id}_frame_masks.csv",
                PATH_MODE_MERGED: "{experiment_id}_frame_masks.csv",
            },
            "prompt_seeds": {
                PATH_MODE_PER_WELL: "{well_id}_prompt_seeds.csv",
            },
        },
    },

    # ── OBJECT EXTRACTION — physical embryo registry ─────────────────────────
    # `physical_embryo_registry` is the identity-origination boundary: the one place track_id ->
    # physical_embryo_id is resolved. It mints one row per distinct (well_id, track_id) from
    # `frame_masks` (the DETECTED set, upstream of the valid/invalid QC split), fans out per well,
    # then merges to an experiment-level table whose validator enforces GLOBAL physical_embryo_id
    # uniqueness. execution=PER_WELL: cheap CPU per well (a dataframe drop-duplicates + mint chain),
    # not a batch model — no per-job startup cost to amortize.
    "physical_embryo_registry": {
        "stage": "object_extraction",
        "product_dir": "physical_embryo_registry",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "physical_embryo_registry": {
                PATH_MODE_PER_WELL: "{well_id}_physical_embryo_registry.csv",
                PATH_MODE_MERGED: "{experiment_id}_physical_embryo_registry.csv",
            },
        },
    },

    # ── OBJECT EXTRACTION — physical_embryo_registry report (TERMINAL) ────────
    # Embryos-per-well distribution + plate-layout heatmap, over the registry's own merged output.
    # Consumed by nothing; only the `reports` aggregate target requests it (viz/report_world.md).
    "physical_embryo_registry_report": {
        "stage": "object_extraction",
        "product_dir": "physical_embryo_registry/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            "embryos_per_well_png": "{experiment_id}_embryos_per_well.png",
            "embryos_per_well_plate_png": "{experiment_id}_embryos_per_well_plate.png",
            "embryos_per_well_over_time_png": "{experiment_id}_embryos_per_well_over_time.png",
        },
    },

    # ── OBJECT EXTRACTION — stage rollup report (TERMINAL) ────────────────────
    # One HTML+PDF page gathering every object_extraction per-step report PNG (discovered from
    # the registry by stage; see viz/stage_report.py). Consumed by nothing; requested only by
    # the `reports` aggregate target. A rollup embeds the leaf reports' PNGs — it renders
    # nothing itself, so it is a DAG leaf that depends on those leaves.
    "object_extraction_rollup_report": {
        "stage": "object_extraction",
        "product_dir": "report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "index_html": "{experiment_id}_object_extraction_report.html",
            "index_pdf": "{experiment_id}_object_extraction_report.pdf",
        },
    },

    # ── OBJECT EXTRACTION — snip inventory ───────────────────────────────────
    # `snip_inventory` is the per-embryo crop table. Fans out per well (one snip_processing job
    # per well), merges to an experiment-level table. Pixel files live beside the per-well shard
    # under the same well directory but are not tracked as registry artifacts.
    "snip_inventory": {
        "stage": "object_extraction",
        "product_dir": "snips",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "snip_inventory": {
                PATH_MODE_PER_WELL: "{well_id}_snip_inventory.csv",
                PATH_MODE_MERGED: "{experiment_id}_snip_inventory.csv",
            },
        },
    },

    # ── OBJECT EXTRACTION — snip auxiliary masks ─────────────────────────────
    # `snip_auxiliary_masks` runs the UNet auxiliary-mask families (via/yolk/focus/bubble/
    # foreground) per snip crop, AFTER physical_embryo_registry/snip_processing. One row per
    # (snip_id, auxiliary_mask_type); masks are native snip resolution (co-keyed + co-resolution
    # with the embryo crop), replacing the retired full-frame `auxiliary_masks` step. PNG masks
    # live beside the per-well shard; only the manifest CSV is a tracked artifact.
    "snip_auxiliary_masks": {
        "stage": "object_extraction",
        "product_dir": "snip_auxiliary_masks",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "manifest": {
                PATH_MODE_PER_WELL: "{well_id}_snip_auxiliary_masks.csv",
                PATH_MODE_MERGED: "{experiment_id}_snip_auxiliary_masks.csv",
            },
        },
    },

    # ── FEATURES — latent embeddings ─────────────────────────────────────────
    # `latent_embeddings` is the per-snip morphological embedding table (one row per snip_id,
    # z_mu_* / z_sigma_* columns). The PRODUCT name names the artifact, not the backend:
    # `latent_embeddings`, not `legacy_vae` (method provenance is config/code, not paths).
    # execution=RUN_BATCH: the legacy VAE loads once (in the Python-3.9 model env) and writes
    # ALL run-well shards before exiting — model load dominates per-well encode cost. The encode
    # body runs under the model interpreter (MODEL_RUN), not the normal RUN env.
    "latent_embeddings": {
        "stage": "feature_extraction",
        "product_dir": "latent_embeddings",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_RUN_BATCH,
        "artifacts": {
            "latents": {
                PATH_MODE_PER_WELL: "{well_id}_latents.parquet",
                PATH_MODE_MERGED: "{experiment_id}_latents.parquet",
            },
        },
    },

    # ── FEATURES — mask geometry ─────────────────────────────────────────────
    # `mask_geometry` is the first computed-feature product: one row per snip_id with micron-aware
    # geometry (area/perimeter/length/width/centroid), decoded from the canonical frame_masks RLE.
    # The PRODUCT name names the artifact, not the method (`mask_geometry`, not `sam2_geometry`).
    # execution=PER_WELL: cheap CPU per well (decode + measure a handful of masks) — no batch model
    # to amortize. Code lives under feature_extraction/; the on-disk stage matches it.
    "mask_geometry": {
        "stage": "feature_extraction",
        "product_dir": "mask_geometry",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "mask_geometry": {
                PATH_MODE_PER_WELL: "{well_id}_mask_geometry.csv",
                PATH_MODE_MERGED: "{experiment_id}_mask_geometry.csv",
            },
        },
    },

    # ── FEATURES — mask_geometry report (TERMINAL) ────────────────────────────
    # Feature histogram grid (renderer D) + area_um2 value-quartile gallery, over mask_geometry's
    # own merged output. Consumed by nothing; see viz/report_world.md.
    "mask_geometry_report": {
        "stage": "feature_extraction",
        "product_dir": "mask_geometry/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            "geometry_feature_grid_png": "{experiment_id}_geometry_feature_grid.png",
            "area_um2_quartile_gallery_png": "{experiment_id}_area_um2_quartile_gallery.png",
        },
    },

    # ── FEATURES — stage rollup report (TERMINAL) ────────────────────────────
    # One HTML+PDF page gathering every feature_extraction per-step report PNG. See the
    # object_extraction_rollup_report comment + viz/stage_report.py.
    "feature_extraction_rollup_report": {
        "stage": "feature_extraction",
        "product_dir": "report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "index_html": "{experiment_id}_feature_extraction_report.html",
            "index_pdf": "{experiment_id}_feature_extraction_report.pdf",
        },
    },

    # ── FEATURES — curvature metrics ─────────────────────────────────────────
    # Centerline length + curvature summaries per snip, from the same canonical frame_masks RLE.
    "curvature_metrics": {
        "stage": "feature_extraction",
        "product_dir": "curvature_metrics",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "curvature_metrics": {
                PATH_MODE_PER_WELL: "{well_id}_curvature_metrics.csv",
                PATH_MODE_MERGED: "{experiment_id}_curvature_metrics.csv",
            },
        },
    },

    # ── FEATURES — curvature_metrics report (TERMINAL) ────────────────────────
    # Feature histogram grid (renderer D) + baseline_deviation_normalized value-quartile gallery
    # with the geodesic spine overlaid (renderer E+overlay). Consumed by nothing; see
    # viz/report_world.md.
    "curvature_metrics_report": {
        "stage": "feature_extraction",
        "product_dir": "curvature_metrics/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            "feature_grid_png": "{experiment_id}_curvature_feature_grid.png",
            "gallery_png": "{experiment_id}_curvature_centerline_gallery.png",
        },
    },

    # ── FEATURES — pose & kinematics ─────────────────────────────────────────
    # Orientation/bbox per mask + displacement/speed within each track (frame timing from
    # frame_inventory). One row per snip; first frame per track has null kinematics.
    "pose_kinematics": {
        "stage": "feature_extraction",
        "product_dir": "pose_kinematics",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "pose_kinematics": {
                PATH_MODE_PER_WELL: "{well_id}_pose_kinematics.csv",
                PATH_MODE_MERGED: "{experiment_id}_pose_kinematics.csv",
            },
        },
    },

    # ── FEATURES — stage predictions ─────────────────────────────────────────
    # Kimmel1995 developmental stage (hpf) per snip from plate_metadata + frame timing. No masks.
    "stage_predictions": {
        "stage": "feature_extraction",
        "product_dir": "stage_predictions",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "stage_predictions": {
                PATH_MODE_PER_WELL: "{well_id}_stage_predictions.csv",
                PATH_MODE_MERGED: "{experiment_id}_stage_predictions.csv",
            },
        },
    },

    # ── FEATURES — fraction alive ────────────────────────────────────────────
    # Continuous viability fraction per snip from the embryo mask (RLE) vs a per-snip VIA mask.
    "fraction_alive": {
        "stage": "feature_extraction",
        "product_dir": "fraction_alive",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "fraction_alive": {
                PATH_MODE_PER_WELL: "{well_id}_fraction_alive.csv",
                PATH_MODE_MERGED: "{experiment_id}_fraction_alive.csv",
            },
        },
    },

    # ── QUALITY CONTROL — surface area QC ─────────────────────────────────────
    # Stage-binned two-sided area outlier flag per snip (area_um2 from mask_geometry vs a
    # packaged wildtype p5/p95 reference interpolated at predicted_stage_hpf).
    "surface_area_qc": {
        "stage": "quality_control",
        "product_dir": "surface_area_qc",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "surface_area_qc": {
                PATH_MODE_PER_WELL: "{well_id}_surface_area_qc.csv",
                PATH_MODE_MERGED: "{experiment_id}_surface_area_qc.csv",
            },
        },
    },

    # ── QUALITY CONTROL — surface_area_qc report (TERMINAL) ───────────────────
    # Area-vs-stage scatter against the reference band (renderer F) + stage-banded quartile
    # gallery (renderer E). Consumed by nothing; see viz/report_world.md.
    "surface_area_qc_report": {
        "stage": "quality_control",
        "product_dir": "surface_area_qc/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            "vs_stage_png": "{experiment_id}_surface_area_qc_vs_stage.png",
            "gallery_png": "{experiment_id}_surface_area_qc_gallery.png",
        },
    },

    # ── QUALITY CONTROL — mask quality QC ─────────────────────────────────────
    # Structural mask-trustworthiness flags per snip (edge / discontinuous / overlapping),
    # decoded from canonical frame_masks. Overlap is computed per image between distinct
    # physical embryos; no persisted composite flag.
    "mask_quality_qc": {
        "stage": "quality_control",
        "product_dir": "mask_quality_qc",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "mask_quality_qc": {
                PATH_MODE_PER_WELL: "{well_id}_mask_quality_qc.csv",
                PATH_MODE_MERGED: "{experiment_id}_mask_quality_qc.csv",
            },
        },
    },

    # ── QUALITY CONTROL — focus QC ─────────────────────────────────────────────
    # Interior structural-edge-content heuristic per snip (ghost/structureless embryo
    # detection). Reads pixels via frame_inventory (materialized_image_readers), masks via
    # canonical frame_masks.
    "focus_qc": {
        "stage": "quality_control",
        "product_dir": "focus_qc",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "focus_qc": {
                PATH_MODE_PER_WELL: "{well_id}_focus_qc.csv",
                PATH_MODE_MERGED: "{experiment_id}_focus_qc.csv",
            },
        },
    },

    # ── QUALITY CONTROL — motion blur QC ─────────────────────────────────────
    # Adjacent z-plane mask-pixel NCC per snip. Reads z-stack pixels via frame_inventory
    # (materialized_image_readers), masks via canonical frame_masks.
    "motion_blur_qc": {
        "stage": "quality_control",
        "product_dir": "motion_blur_qc",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "motion_blur_qc": {
                PATH_MODE_PER_WELL: "{well_id}_motion_blur_qc.csv",
                PATH_MODE_MERGED: "{experiment_id}_motion_blur_qc.csv",
            },
        },
    },

    # ── QUALITY CONTROL — death detection (per-snip flags) ────────────────────
    # Two-mode death QC per snip: viability_dead_flag (per frame) + persistence_dead_flag
    # (per animal, broadcast time_index >= D). Consumes fraction_alive + frame timing.
    "death_detection_qc": {
        "stage": "quality_control",
        "product_dir": "death_detection",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "death_detection_qc": {
                PATH_MODE_PER_WELL: "{well_id}_death_detection_qc.csv",
                PATH_MODE_MERGED: "{experiment_id}_death_detection_qc.csv",
            },
        },
    },

    # ── QUALITY CONTROL — death event (per physical embryo) ───────────────────
    # Animal-level event table (one row per persistence-dead physical_embryo_id): the
    # lead-time-adjusted death_event_time_index + death_event_stage_hpf. Same product_dir as
    # death_detection (one product, two grains); distinct artifact filenames.
    "death_event": {
        "stage": "quality_control",
        "product_dir": "death_detection",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "death_event": {
                PATH_MODE_PER_WELL: "{well_id}_death_event.csv",
                PATH_MODE_MERGED: "{experiment_id}_death_event.csv",
            },
        },
    },

    # ── QUALITY CONTROL — death_detection report (TERMINAL) ───────────────────
    # Three artifacts, all recomputed from death_detection's own merged inputs
    # (death_detection_qc flags + fraction_alive trace): whole-experiment survival curve, per-embryo
    # mortality curtain, called-death time histogram. Consumed by nothing; the worked example in
    # viz/report_world.md.
    "death_detection_report": {
        "stage": "quality_control",
        "product_dir": "death_detection/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            "experiment_png": "{experiment_id}_alive_embryos_experiment.png",
            "curtain_png": "{experiment_id}_mortality_curtain.png",
            "death_time_png": "{experiment_id}_death_time_histogram.png",
            "well_survival_png": "{experiment_id}_well_survival_over_time.png",
        },
    },

    # ── QUALITY CONTROL — stage rollup report (TERMINAL) ─────────────────────
    # One HTML+PDF page gathering every quality_control per-step report PNG. See the
    # object_extraction_rollup_report comment + viz/stage_report.py.
    "quality_control_rollup_report": {
        "stage": "quality_control",
        "product_dir": "report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "index_html": "{experiment_id}_quality_control_report.html",
            "index_pdf": "{experiment_id}_quality_control_report.pdf",
        },
    },

    # ── QUALITY CONTROL — snip QC verdict ─────────────────────────────────────
    # The final per-snip operational verdict: use_snip + qc_fail_reasons, ORed from the MVP
    # exclusion flags (death_detection_qc, surface_area_qc, mask_quality_qc).
    "snip_qc": {
        "stage": "quality_control",
        "product_dir": "snip_qc",
        "fanout": PER_WELL_THEN_MERGE,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "verdict": {
                PATH_MODE_PER_WELL: "{well_id}_snip_qc.parquet",
                PATH_MODE_MERGED: "{experiment_id}_snip_qc.parquet",
            },
            # Auxiliary per-well planning artifact: the resolver output (exclusion_flags +
            # ResolvedFlagSource list) serialized as JSON so build_snip_qc_for_well receives
            # the exact same source plan the DAG was declared with. NOT merged across wells
            # and NOT the snip_qc verdict product.
            "resolved_sources": {
                PATH_MODE_PER_WELL: "{well_id}_snip_qc_resolved_sources.json",
            },
        },
    },

    # ── QUALITY CONTROL — snip_qc report (TERMINAL) ───────────────────────────
    # Exclusion-reason fraction over time_index, two views (all snips; not-dead snips only) — the
    # whole-experiment health view. Consumed by nothing; see viz/report_world.md.
    "snip_qc_report": {
        "stage": "quality_control",
        "product_dir": "snip_qc/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this (see comment above EXECUTION_PER_WELL)
        "artifacts": {
            # ONE artifact: all-snips + not-dead-only as side-by-side panels sharing a y-axis and
            # legend, not two separate PNGs — the point is reading both at a glance for comparison.
            "exclusion_reasons_png": "{experiment_id}_exclusion_reasons_over_time.png",
        },
    },

    # ── ANALYSIS READY — the final merged analysis table ──────────────────────
    # OPTIONAL downstream product (snip_qc stays the proven through-line terminal). One row per
    # snip_id: the identity spine + every per-snip feature payload + the snip_qc verdict + the
    # per-well plate_metadata broadcast by well_id. Merged-level fan-in join, no per-well shard.
    "analysis_ready": {
        "stage": "analysis_ready",
        "product_dir": "analysis_ready",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,  # EXPERIMENT-grain steps always use this sentinel
        "artifacts": {
            "analysis_ready": {
                PATH_MODE_EXPERIMENT: "{experiment_id}_analysis_ready.parquet",
            },
        },
    },

    # ── ANALYSIS READY — report (TERMINAL) ────────────────────────────────────
    # The ONE step whose input surface is the whole joined DAG (embeddings + plate_metadata +
    # predicted_stage_hpf + genotype), so genotype/stage-colored PCA belongs here. Consumed by
    # nothing; see viz/report_world.md. Three artifacts: side-by-side latent PCA/UMAP, the
    # experiment survival curve over predicted_stage_hpf, and the per-genotype survival panel.
    "analysis_ready_report": {
        "stage": "analysis_ready",
        "product_dir": "analysis_ready/report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "latent_projection_png": "{experiment_id}_latent_projection.png",
            "survival_over_stage_png": "{experiment_id}_survival_over_stage.png",
            "genotype_survival_panel_png": "{experiment_id}_genotype_survival_panel.png",
            "well_survival_over_stage_png": "{experiment_id}_well_survival_over_stage.png",
        },
    },

    # ── ANALYSIS READY — stage rollup (TERMINAL) ──────────────────────────────
    # One HTML+PDF page gathering every analysis_ready per-step report PNG (today just
    # analysis_ready_report's 4 artifacts) — same pattern as the other 3 stage rollups.
    "analysis_ready_rollup_report": {
        "stage": "analysis_ready",
        "product_dir": "report",
        "fanout": EXPERIMENT,
        "execution": EXECUTION_PER_WELL,
        "artifacts": {
            "index_html": "{experiment_id}_analysis_ready_report.html",
            "index_pdf": "{experiment_id}_analysis_ready_report.pdf",
        },
    },
}

# ── HELPERS: step name -> artifact path ──────────────────────────────────────────────────────
# Two private lookups turn registry keys into concrete values, failing loudly on a typo:
#   _lookup_step("discover_wells")         -> {"stage": ..., "fanout": ..., "artifacts": ...}
#   _resolve_filename("ingest_scope_metadata", "raw", "experiment", {"scope": "yx1"})
#                                          -> "scope_metadata__yx1.csv"
#
# Four public functions build paths from them (root + experiment come from the caller):
#   step_dir        -> the FOLDER, no filename
#       step_dir(ROOT, "discover_wells", "20250912")
#           -> {ROOT}/acquisition/20250912
#       step_dir(ROOT, "frame_inventory", "20250912", path_mode="per_well",
#                well_id="20250912_B01")
#           -> {ROOT}/acquisition/20250912/frame_inventory/per_well/20250912_B01
#
#   artifact_path   -> FOLDER + filename  (the one you'll call most)
#       artifact_path(ROOT, "discover_wells", "wells", "20250912")
#           -> {ROOT}/acquisition/20250912/discovered_wells.txt
#       artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
#                     path_mode="per_well", well_id="20250912_B01")
#           -> {ROOT}/acquisition/20250912/frame_inventory/per_well/20250912_B01/20250912_B01_frame_inventory.csv
#
#   validated_path  -> artifact_path + ".validated"        (the sentinel beside it)
#       validated_path(ROOT, "apply_position_to_well_mapping", "mapped", "20250912")
#           -> {ROOT}/acquisition/20250912/scope_metadata_mapped.csv.validated
#
#   provenance_path -> artifact_path + ".provenance.json"  (the sidecar beside it)
#       provenance_path(ROOT, "map_positions_to_wells", "mapping", "20250912")
#           -> {ROOT}/acquisition/20250912/position_well_mapping.csv.provenance.json


def known_steps() -> tuple[str, ...]:
    """Return all registered step names (sorted) — handy for tests and docs generation."""
    return tuple(sorted(PIPELINE_STEPS))


def known_artifacts(step: str) -> tuple[str, ...]:
    """Return a step's registered artifact keys (sorted)."""
    return tuple(sorted(_lookup_step(step)["artifacts"]))


def execution_mode(step: str) -> str:
    """Return the execution model for a step: EXECUTION_PER_WELL or EXECUTION_RUN_BATCH.

    This is metadata for Snakemake rule authors — it does not affect path construction.
    EXECUTION_PER_WELL means one job per well shard (use expand() over run wells).
    EXECUTION_RUN_BATCH means one job for the whole run set, writing all well shards before
    exiting (model-load cost dominates per-well encode cost — e.g. SAM2, legacy VAE).
    """
    return _lookup_step(step)["execution"]


def _lookup_step(step: str) -> dict:
    """Return a step's registry row, or raise KeyError naming the known steps."""
    try:
        return PIPELINE_STEPS[step]
    except KeyError:
        raise KeyError(
            f"Unknown step {step!r}. Known front-end steps: {sorted(PIPELINE_STEPS)}. "
            f"Back-half steps are not in the registry yet (added in Scope 5)."
        ) from None


def _normalize_path_mode(step: str, path_mode: Optional[str]) -> str:
    """Resolve and validate ``path_mode`` against the step's fanout.

    ``path_mode=None`` means "use the obvious one": an EXPERIMENT step has exactly one legal mode
    (``experiment``), so None resolves to it — callers of experiment-grain steps need not pass
    path_mode. A PER_WELL_THEN_MERGE step has TWO legal modes (``per_well`` vs ``merged``) with no
    neutral default, so the caller MUST choose; None (or an illegal value) raises a message that
    spells out the choice. This makes ``fanout`` executable, not decorative.
    """
    fanout = _lookup_step(step)["fanout"]
    allowed = _ALLOWED_PATH_MODES.get(fanout)
    if allowed is None:
        raise ValueError(f"Step {step!r} has unknown fanout {fanout!r}.")

    if path_mode is None:
        if fanout == EXPERIMENT:
            return PATH_MODE_EXPERIMENT
        # PER_WELL_THEN_MERGE: two legal modes, no neutral default — make the user pick, say why.
        raise ValueError(
            f"Step {step!r} has fanout={PER_WELL_THEN_MERGE!r}. "
            f"Pass path_mode={PATH_MODE_PER_WELL!r} or path_mode={PATH_MODE_MERGED!r} explicitly."
        )

    if path_mode not in allowed:
        if fanout == PER_WELL_THEN_MERGE:
            raise ValueError(
                f"Step {step!r} has fanout={PER_WELL_THEN_MERGE!r}, so the caller must choose "
                f"path_mode={PATH_MODE_PER_WELL!r} for a well shard or path_mode={PATH_MODE_MERGED!r} "
                f"for the experiment-level file. Got path_mode={path_mode!r}."
            )
        raise ValueError(
            f"Step {step!r} has fanout={fanout!r}, so path_mode must be {allowed[0]!r}; "
            f"got {path_mode!r}."
        )
    return path_mode


def _resolve_filename(
    step: str,
    artifact: str,
    path_mode: str,
    format_vars: Optional[dict] = None,
) -> str:
    """Fill in an artifact's filename template for the given ``path_mode``.

    The template is either a single string (same filename in every mode) or a
    ``{path_mode: string}`` dict (filename differs by mode). Raises KeyError naming the step's
    known artifacts if ``artifact`` is unknown, and ValueError (naming the missing token + the
    template) if a ``{token}`` in the template was not supplied.
    """
    spec = _lookup_step(step)
    try:
        template = spec["artifacts"][artifact]
    except KeyError:
        raise KeyError(
            f"Step {step!r} has no artifact {artifact!r}. "
            f"Known artifacts: {sorted(spec['artifacts'])}."
        ) from None
    if isinstance(template, dict):
        try:
            template = template[path_mode]
        except KeyError:
            raise ValueError(
                f"Artifact {artifact!r} of step {step!r} has no filename for path_mode "
                f"{path_mode!r}; defined for {sorted(template)}."
            ) from None
    try:
        return template.format(**(format_vars or {}))
    except KeyError as missing:
        missing_token = missing.args[0]
        raise ValueError(
            f"Filename template {template!r} for {step}/{artifact} needs token {missing_token!r} "
            f"which was not supplied (got format_vars={format_vars!r}). Pass it via format_vars, "
            f"or well_id=/experiment_id= as appropriate."
        ) from None


# Three directory concepts in the layout, each built in exactly ONE place so they cannot drift:
#   _experiment_step_dir   {stage}/{exp}              the experiment base
#   _per_well_step_dir      {stage}/{exp}/per_well    where ALL a step's per-well shards live
#   step_dir(..., well_id)  {stage}/{exp}/per_well/{well_id}   one well's shard dir
# The public composers (step_dir, per_well_step_dir) build downward from these bricks — no caller
# ever has to strip a level off with ``.parent`` to name the per-well directory.


def _experiment_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """The experiment-level directory for a step: ``{stage}/{exp}[/{product_dir}]``.

    The ONE place the base directory is built. When the step has a ``product_dir`` key, it is
    appended after the experiment id (``{stage}/{exp}/{product_dir}``), giving each product family
    its own named folder within a regime. When absent, artifacts land directly under
    ``{stage}/{exp}/``.
    """
    spec = _lookup_step(step)
    base = Path(root) / spec["stage"] / str(experiment_id)
    product_dir = spec.get("product_dir")
    return base / product_dir if product_dir else base


def _per_well_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """The directory holding ALL of a step's per-well shards: ``{stage}/{exp}/per_well``.

    The ONE place ``PER_WELL_DIRNAME`` is joined to the experiment base. Asserts the step
    actually fans out per-well (``_normalize_path_mode`` fails loud on an experiment-grain step,
    which has no per-well directory). A specific shard dir is this plus ``{well_id}`` (step_dir);
    the merge lists this directly.
    """
    _normalize_path_mode(step, PATH_MODE_PER_WELL)  # guard: per_well_then_merge steps only
    return _experiment_step_dir(root, step, experiment_id) / PER_WELL_DIRNAME


def per_well_step_dir(root: PathLike, step: str, experiment_id: str) -> Path:
    """Return the directory holding ALL per-well shards for a per_well_then_merge step.

    The named path concept ``{stage}/{exp}/per_well`` — agnostic to who reads it (the merge lists
    it to discover which wells have shards; a cleanup might walk it; a report might count it). It
    is the per-well sibling of ``step_dir``'s experiment view, NOT a specific well's shard
    (that needs ``well_id`` via ``step_dir``). Fails loud on an experiment-grain step.
    """
    return _per_well_step_dir(root, step, experiment_id)


def step_dir(
    root: PathLike,
    step: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
) -> Path:
    """Return the directory that holds a step's artifact, before the filename is appended.

    The directory lives under the step's ``stage`` folder. ``per_well`` mode requires ``well_id``
    and yields ``{stage}/{exp}/per_well/{well_id}``; ``experiment``/``merged`` modes yield
    ``{stage}/{exp}``. ``path_mode=None`` resolves to the step's only legal mode for an
    experiment-grain step, but a per_well_then_merge step must be told ``per_well`` or ``merged``
    (see _normalize_path_mode). Composes from the directory bricks above so the per-well shard dir
    is always ``per_well_step_dir(...) / {well_id}`` — the two never drift.
    """
    mode = _normalize_path_mode(step, path_mode)
    if mode == PATH_MODE_PER_WELL:
        if not well_id:
            raise ValueError(
                f"path_mode='per_well' for step {step!r} requires well_id (the GLOBAL "
                f"{{experiment_id}}_{{well_index}} key)."
            )
        return _per_well_step_dir(root, step, experiment_id) / str(well_id)
    # experiment | merged -> the experiment base (the per_well_then_merge merged view lands beside
    # an experiment-grain artifact); the distinction is intent, enforced upstream by fanout.
    return _experiment_step_dir(root, step, experiment_id)


def artifact_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Resolve the absolute path to one of a step's artifacts.

    Examples (see target/front_end_naming_and_flow.md for the full set)::

        artifact_path(ROOT, "ingest_scope_metadata", "raw", "20250912",
                      format_vars={"scope": "yx1"})
        #   -> {ROOT}/acquisition/20250912/scope_metadata__yx1.csv

        artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
                      path_mode="per_well", well_id="20250912_B01")
        #   -> {ROOT}/acquisition/20250912/frame_inventory/per_well/20250912_B01/20250912_B01_frame_inventory.csv

        artifact_path(ROOT, "frame_inventory", "inventory", "20250912",
                      path_mode="merged")
        #   -> {ROOT}/acquisition/20250912/frame_inventory/20250912_frame_inventory.csv
    """
    mode = _normalize_path_mode(step, path_mode)
    directory = step_dir(root, step, experiment_id, path_mode=mode, well_id=well_id)
    # Build the template namespace from SUPPLIED values only — never fabricate identity (see the
    # NO-LEAKAGE BOUNDARY). experiment_id is always available; well_id only if the caller passed
    # it. The merged template names the experiment (it uses {experiment_id}); the per_well
    # template names the well (it uses {well_id}). A template that references a token we did not
    # supply raises a clear ValueError in _resolve_filename.
    fmt = dict(format_vars or {})
    reserved = {"experiment_id", "well_id"}
    bad = reserved & set(fmt)
    if bad:
        raise ValueError(
            f"Do not pass identity tokens via format_vars: {sorted(bad)}. "
            f"Use experiment_id= and well_id= instead."
        )
    fmt["experiment_id"] = str(experiment_id)
    if well_id is not None:
        fmt["well_id"] = str(well_id)
    return directory / _resolve_filename(step, artifact, mode, fmt)


def validated_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.validated`` sentinel beside an artifact (trailing-suffix form).

    ⚠️ Today ``apply_position_to_well_mapping`` writes a LEADING-dot form (``.scope_metadata_mapped.validated``)
    while this helper assumes the TARGET trailing form (``scope_metadata_mapped.csv.validated``).
    That mismatch is an open audit item (front_end doc, "Sentinel-suffix audit still open") to be
    normalized on-disk when the step is reconstructed — this helper deliberately emits the target
    convention.
    """
    base = artifact_path(
        root, step, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".validated")


def provenance_path(
    root: PathLike,
    step: str,
    artifact: str,
    experiment_id: str,
    *,
    path_mode: Optional[str] = None,
    well_id: Optional[str] = None,
    format_vars: Optional[dict] = None,
) -> Path:
    """Return the ``{artifact}.provenance.json`` sidecar beside an artifact."""
    base = artifact_path(
        root, step, artifact, experiment_id,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )
    return base.with_name(base.name + ".provenance.json")
