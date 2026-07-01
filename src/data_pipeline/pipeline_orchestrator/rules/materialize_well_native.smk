"""Native per-well producers — registered ONLY in native front-end mode.

Native mode now writes product-grain frame_inventory shards first, discovers the active validated
product shards on disk, and assembles those shards into the CANONICAL per-well frame_inventory path.
In dropin mode the dropin_handoff.smk producers write the same canonical artifacts directly, and
this file is NOT included.

Reuses the shared helpers defined in frame_inventory.smk (included before this file):
``_frame_inventory_artifact``, ``_materialize_well_validated``, product-shard path helpers, and
``_discovered_product_shards_artifact``.
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
          --scope "{SCOPE_TOKEN}" \
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
        **(
            {"master_params_json": str(KEYENCE_STITCH_MAP_JSON)}
            if MICROSCOPE == "Keyence"
            else {}
        ),
    output:
        inventory=str(_frame_inventory_product_artifact(
            "{experiment}", well_id="{well_id}", product_key="{product_key}"
        )),
    params:
        device=lambda wc: str(config.get("image_building", {}).get("device", "cuda")),
        smoke_max=lambda wc: int(
            config.get("image_materialization", {}).get("smoke_max_time_indices", 0)
        ),
        master_params_arg=(
            lambda wc, input: f'--master-params-path "{input.master_params_json}"'
            if MICROSCOPE == "Keyence"
            else ""
        ),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-image-product-for-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --scope "{SCOPE_TOKEN}" \
          --product-key "{wildcards.product_key}" \
          --resolved-product-plan-json "{input.resolved_product_plan}" \
          --acquisition-inventory-csv "{input.acquisition_inventory_csv}" \
          --position-well-mapping-csv "{input.position_well_mapping_csv}" \
          --built-image-data-dir "{BUILT_IMAGE_DATA_DIR}" \
          --frame-inventory-product-csv "{output.inventory}" \
          --config-yaml "{CONFIG_YAML}" \
          --candidate "false" \
          --device "{params.device}" \
          --smoke-max-time-indices "{params.smoke_max}" \
          {params.master_params_arg}
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


def _frame_inventory_product_validated_for_config(wc):
    return [
        str(_frame_inventory_product_validated(
            wc.experiment, well_id=wc.well_id, product_key=product_key
        ))
        for product_key in IMAGE_PRODUCT_KEYS
    ]


rule discover_product_shards_for_well:
    """Discover validated product shards active on disk for one well.

    Configured product sentinels are DAG triggers only. The command scans the product-shard
    directory and includes any CSV that has a matching ``.csv.validated`` sidecar.
    """
    input:
        validated_product_shards=_frame_inventory_product_validated_for_config,
    output:
        discovered=str(_discovered_product_shards_artifact("{experiment}", well_id="{well_id}")),
    params:
        products_dir=lambda wc: rule_step_dir(
            FRAME_INVENTORY_PRODUCTS_STEP,
            wc.experiment,
            path_mode=PATH_MODE_PER_WELL,
            well_id=wc.well_id,
        ),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks discover-product-shards-for-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --frame-inventory-products-dir "{params.products_dir}" \
          --output-csv "{output.discovered}"
        """


rule assemble_well_frame_inventory:
    """Assemble discovered product shards into the canonical per-well frame_inventory CSV."""
    input:
        discovered=str(_discovered_product_shards_artifact("{experiment}", well_id="{well_id}")),
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks assemble-well-frame-inventory \
          --discovered-product-shards-csv "{input.discovered}" \
          --output-csv "{output.inventory}"
        """


rule validate_frame_inventory_for_well:
    input:
        # Consume the assembled canonical shard via the frame_inventory contract path.
        inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_materialize_well_validated("{experiment}", well_id="{well_id}")),
    shell:
        # Per-well node = STRICT: open every image, self-check dims/µm-px, one well.
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}" \
          --image-root "{BUILT_IMAGE_DATA_DIR}" \
          --check-sources "true" \
          --validation-scope "per_well"
        """
