"""Native per-well producers — registered ONLY in native front-end mode.

Extracted from frame_inventory.smk so the native producer family can be mode-exclusively included
(mirroring dropin_handoff.smk). These two rules produce the CANONICAL per-well frame_inventory shard
and its strict .validated sentinel from the microscope ingest. In dropin mode the dropin_handoff.smk
producers write the same canonical artifacts instead, and this file is NOT included.

Reuses the shared helpers defined in frame_inventory.smk (included before this file):
``_frame_inventory_artifact``, ``_materialize_well_done``, ``_materialize_well_validated``.
"""


rule write_resolved_product_plan_for_well:
    """Write one resolved product commitment for one well/product_key."""
    input:
        discovered_wells=DISCOVERED_WELLS_TXT,
    output:
        resolved_product_plan=str(_resolved_product_plan(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks write-resolved-product-plan-for-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --scope "yx1" \
          --product-key "{wildcards.product_key}" \
          --output-json "{output.resolved_product_plan}" \
          --config-yaml "{CONFIG_YAML}"
        """


rule materialize_image_product_for_well:
    """Materialize exactly one resolved image product into a product frame-inventory shard."""
    input:
        acquisition_inventory_csv=SCOPE_ACQUISITION_INVENTORY_CSV,
        position_well_mapping_csv=POSITION_WELL_MAPPING_CSV,
        resolved_product_plan=str(_resolved_product_plan(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    output:
        inventory=str(_frame_inventory_product_artifact(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    params:
        device=lambda wc: str(config.get("image_building", {}).get("device", "cuda")),
        smoke_max=lambda wc: int(
            config.get("image_materialization", {}).get("smoke_max_time_indices", 0)
        ),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-image-product-for-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --scope "yx1" \
          --product-key "{wildcards.product_key}" \
          --resolved-product-plan-json "{input.resolved_product_plan}" \
          --acquisition-inventory-csv "{input.acquisition_inventory_csv}" \
          --position-well-mapping-csv "{input.position_well_mapping_csv}" \
          --built-image-data-dir "{BUILT_IMAGE_DATA_DIR}" \
          --frame-inventory-product-csv "{output.inventory}" \
          --candidate "false" \
          --device "{params.device}" \
          --smoke-max-time-indices "{params.smoke_max}"
        """


rule validate_frame_inventory_product_for_well:
    """Strictly validate one product frame-inventory shard."""
    input:
        inventory=str(_frame_inventory_product_artifact(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    output:
        validated=str(_frame_inventory_product_validated(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}" \
          --image-root "{BUILT_IMAGE_DATA_DIR}" \
          --check-sources "true" \
          --validation-scope "per_well"
        """


rule materialize_well:
    """Step 6 — the LIVE per-well YX1 materializer.

    Fanned over discovered_wells.txt (one well per job). Reads the experiment's acquisition
    inventory + position→well mapping, materializes the configured product set (Step 6: BF /
    projection / focus_stack), writes pixel files into the live acquisition tree
    (candidate=False), and emits the per-well frame-inventory shard the validator consumes.
    The ND2 source travels inside the acquisition inventory (source_nd2_path) — not a CLI arg.

    DOCTRINE: the inventory CSV path is owned by the frame_inventory registry step. The action
    (materialize_well) writes the file; the product (frame_inventory) owns the contract path.
    The done sentinel is owned by materialize_well under materialized_images/.
    """
    input:
        acquisition_inventory_csv=SCOPE_ACQUISITION_INVENTORY_CSV,
        position_well_mapping_csv=POSITION_WELL_MAPPING_CSV,
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        done=str(_materialize_well_done("{experiment}", well_id="{well_id}")),
    params:
        device=lambda wc: str(config.get("image_building", {}).get("device", "cuda")),
        smoke_max=lambda wc: int(
            config.get("image_materialization", {}).get("smoke_max_time_indices", 0)
        ),
    shell:
        # well_index is derived from well_id inside the task (identity parser) — not passed here.
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --scope "yx1" \
          --acquisition-inventory-csv "{input.acquisition_inventory_csv}" \
          --position-well-mapping-csv "{input.position_well_mapping_csv}" \
          --built-image-data-dir "{BUILT_IMAGE_DATA_DIR}" \
          --frame-inventory-csv "{output.inventory}" \
          --done-flag "{output.done}" \
          --candidate "false" \
          --device "{params.device}" \
          --smoke-max-time-indices "{params.smoke_max}"
        """


rule validate_frame_inventory_for_well:
    input:
        # Consume the materializer-emitted shard via the frame_inventory contract path.
        inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        done=str(_materialize_well_done("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_materialize_well_validated("{experiment}", well_id="{well_id}")),
    shell:
        # Per-well node = STRICT: open every image, self-check dims/µm-px, one well. The native YX1
        # shard writes a real absolute source_image_path + dims + µm/px, so check_sources=True is
        # meaningful here (catches a corrupt/missing materialized image). image_root is passed for
        # consistency; native paths are absolute so it is not consulted for resolution.
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}" \
          --image-root "{BUILT_IMAGE_DATA_DIR}" \
          --check-sources "true" \
          --validation-scope "per_well"
        """
