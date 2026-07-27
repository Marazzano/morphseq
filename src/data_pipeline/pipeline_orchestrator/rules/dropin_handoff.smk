"""External drop-in entrance — the dropin PRODUCER family (registered ONLY in dropin mode).

Selected by ``front_end.mode: dropin`` (Snakefile) — mode-exclusive with the native producers
(`materialize_well_native.smk`). It lands a user's ``dropin_frame_inventory.csv`` on the SAME
canonical spine the native path uses and hands off to the SAME strict gate, so nothing downstream
of the seam can tell which producer ran:

    dropin_frame_inventory.csv (ingress; config-supplied, NOT a registry artifact)
      → discover_wells (checkpoint)         → discovered_wells.txt        [canonical]
      → split_dropin_inventory (per well)   → {well_id}_frame_inventory.csv shard  [canonical]
      → validate_frame_inventory_for_well   → {well_id}_frame_inventory.csv.validated [canonical]
      → segment_and_track_per_well …

No stitch step (the user already has images). The rule NAMES are the canonical ones
(``discover_wells``, ``validate_frame_inventory_for_well``) so the well-runner fan + merge rules in
frame_inventory.smk consume drop-in shards transparently. This file reuses frame_inventory.smk's path
helpers (``_frame_inventory_artifact``, ``_materialize_well_validated``), so it MUST be included after it.

Ingress (``DROPIN_MANIFEST_CSV`` + ``DROPIN_IMAGE_ROOT``) is owned by the Snakefile's `front_end`/`dropin`
config block — one blessed path source, never a raw string scattered in rules.
"""


# 1) Discover wells from the ingress manifest → the canonical discovered_wells.txt.
checkpoint discover_wells:
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


# 2) Select THIS well's rows from the ingress manifest → exactly its canonical shard.
#    Per-well + race-free (Option B): writes ONLY {well_id}'s shard, never the whole directory.
rule split_dropin_inventory:
    input:
        manifest = DROPIN_MANIFEST_CSV,
    output:
        shard = str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks split-dropin-inventory \
          --manifest-csv "{input.manifest}" \
          --well-id "{wildcards.well_id}" \
          --output-csv "{output.shard}" \
          --image-root "{DROPIN_IMAGE_ROOT}"
        """


# 3) Validate each shard through the SAME strict gate (check_sources=True, per_well). Same rule name
#    + same canonical .validated sentinel the native producer writes → merge stays producer-agnostic.
rule validate_frame_inventory_for_well:
    input:
        inventory = str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated = str(_materialize_well_validated("{experiment}", well_id="{well_id}")),
    params:
        image_root = DROPIN_IMAGE_ROOT,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}" \
          --image-root "{params.image_root}" \
          --check-sources "true" \
          --validation-scope "per_well"
        """
