"""External drop-in entrance — DRAFT rule file (NOT yet included in the Snakefile).

╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  DRAFT ONLY — DO NOT add to the Snakefile `include:` list yet.                              ║
║                                                                                                ║
║  These rules INTENTIONALLY write the CANONICAL artifacts — `DISCOVERED_WELLS_TXT`, the per-well ║
║  frame_inventory shards, and the `.validated` sentinels. If included ALONGSIDE the native       ║
║  producers they WILL collide (Snakemake forbids two rules producing the same output), even with ║
║  the draft-distinct rule names below. This file is safe ONLY while un-included.                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

   Final integration needs a mode-exclusive producer-selection refactor (Step 8) that registers
   exactly one producer family:

       front_end:
         mode: native   # native | dropin
       dropin:
         enabled: true
         frame_inventory_csv: /path/to/dropin_frame_inventory.csv
         image_root: /path/to/images

   - native mode registers the native discover/materialize producers;
   - dropin mode registers the dropin discover/split producers below;
   - BOTH target the same canonical discovered_wells + frame_inventory artifacts;
   - unknown mode fails loud; a dry-run matrix (native + dropin) gates that refactor.

This file is the reviewable, code-level entrance. The drop-in twins it drives
(discover_wells_from_handoff, split_dropin_inventory_by_well) and the strict gate are all built and
tested; only the DAG wiring is deferred.

> **Ingress path ownership (locked).** ``dropin_frame_inventory.csv`` + ``image_root`` are
> config/CLI-supplied INGRESS, NOT ``PIPELINE_STEPS`` artifacts. The canonical pipeline artifacts begin
> at the per-well frame_inventory shards (post-split). The ingress manifest is read only by discovery +
> split — never by a well-local compute stage. There is no stitch step (the user already has images).

The rule NAMES below are draft-distinct (``*_dropin``) to make the collision explicit; under producer
selection they collapse onto the native names (``discover_wells`` checkpoint, the per-well shard
producer, ``validate_frame_inventory_for_well``) so downstream consumes drop-in shards transparently.
"""

DROPIN_CONFIG = config.get("dropin", {})

# Ingress (config/CLI; one blessed path source — never a raw string scattered in rules).
DROPIN_MANIFEST_CSV = str(DROPIN_CONFIG.get("frame_inventory_csv", ""))
DROPIN_IMAGE_ROOT = str(DROPIN_CONFIG.get("image_root", ""))

DROPIN_FRAME_INVENTORY_STEP = "frame_inventory"
DROPIN_FRAME_INVENTORY_ARTIFACT = "inventory"


def _dropin_shard(experiment, *, well_id):
    return rule_artifact(
        DROPIN_FRAME_INVENTORY_STEP, DROPIN_FRAME_INVENTORY_ARTIFACT, experiment,
        path_mode=PATH_MODE_PER_WELL, well_id=well_id,
    )


def _dropin_validated(experiment, *, well_id):
    return rule_validated(
        DROPIN_FRAME_INVENTORY_STEP, DROPIN_FRAME_INVENTORY_ARTIFACT, experiment,
        path_mode=PATH_MODE_PER_WELL, well_id=well_id,
    )


# 1) Discover wells from the ingress manifest → the SAME discovered_wells.txt contract.
#    (Under producer selection this becomes the `discover_wells` checkpoint, replacing the native one.)
checkpoint discover_wells_dropin:
    input:
        manifest = DROPIN_MANIFEST_CSV,
    output:
        wells = DISCOVERED_WELLS_TXT,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks discover-wells-from-handoff \
          --manifest-csv "{input.manifest}" \
          --output-wells "{output.wells}"
        """


# 2) Select THIS well's rows from the ingress manifest → exactly its declared shard.
#    Per-well + race-free: the rule writes ONLY {well_id}'s shard (Option B), never the whole
#    directory — a rule writes what it promises. Fanned over the discovery checkpoint's well list.
#    (One experiment per submission; DISCOVERED_WELLS_TXT is the single-experiment front-end target.)
rule split_dropin_inventory:
    input:
        manifest = DROPIN_MANIFEST_CSV,
    output:
        shard = str(_dropin_shard("{experiment}", well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks split-dropin-inventory \
          --manifest-csv "{input.manifest}" \
          --well-id "{wildcards.well_id}" \
          --output-csv "{output.shard}"
        """


# 3) Validate each shard through the SAME strict gate (check_sources=True, per_well).
rule validate_dropin_frame_inventory_for_well:
    input:
        shard = str(_dropin_shard("{experiment}", well_id="{well_id}")),
    output:
        validated = str(_dropin_validated("{experiment}", well_id="{well_id}")),
    params:
        image_root = DROPIN_IMAGE_ROOT,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.shard}" \
          --output-flag "{output.validated}" \
          --image-root "{params.image_root}" \
          --check-sources "true" \
          --validation-scope "per_well"
        """
