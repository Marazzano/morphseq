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
    """Write one resolved product commitment for one well/product_key.

    The merged config is a `params:` value, NOT a file `input:`: the resolved plan is the stable,
    content-guarded artifact that captures only the config fields this well/product needs, and
    downstream rules depend on THAT plan — not on raw config. Taking config as a file input here
    would make the config file's mtime a rerun trigger, so any config write (even a no-op rewrite)
    would invalidate every resolved plan and cascade through the whole DAG. Mirrors how
    fraction_alive / snip_auxiliary_masks pass config via params. (A genuine config change still
    propagates: it changes the resolved plan's CONTENT, which re-triggers real downstream work.)"""
    input:
        discovered_wells=DISCOVERED_WELLS_TXT,
    params:
        config_yaml=str(CONFIG_YAML),
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
          --config-yaml "{params.config_yaml}"
        """


def _resolved_product_plans_for_well(wc):
    return [
        str(_resolved_product_plan(wc.experiment, well_id=wc.well_id, product_key=key))
        for key in IMAGE_PRODUCT_KEYS
    ]


# Snakemake allows a FUNCTION for input but not for output, so the product shards are expanded
# here as literal wildcard-carrying paths. IMAGE_PRODUCT_KEYS is a DAG-build-time constant
# (Snakefile), so this list is fixed for the run -- which is exactly what makes a per-well
# multi-output rule expressible at all.
_MULTI_PRODUCT_INVENTORIES = [
    str(_frame_inventory_product_artifact(
        "{experiment}", well_id="{well_id}", product_key=key
    ))
    for key in IMAGE_PRODUCT_KEYS
]


# Exactly ONE of these two rules is registered per run, so no ruleorder is needed and no shard has
# two possible producers. Keyence gets the multi-product rule because it reads MANY raw TIFFs per
# frame (tiles x z planes): under the per-product shape, a plan requesting both
# BF__projection__focus_stack and BF__z_stack ran two processes that each read the same planes,
# focus-stacked them, and solved the same tile transform. Every other scope keeps the per-product
# rule -- YX1 reads one ND2 per well and has no such duplication to remove.
if MICROSCOPE == "Keyence":

    rule materialize_image_products_for_well:
        """Materialize ALL of one Keyence well's image products in ONE process.

        The product fanout happens at the WRITE boundary, not at the front of the job: the backend
        receives the whole product plan, prepares each (channel_id, time_index) frame once, and
        emits every product from those shared materials.

        Each product still writes its OWN shard at its OWN existing path, so
        validate_frame_inventory_product_for_well / discover_product_shards_for_well /
        assemble_well_frame_inventory are untouched: a shard must stay single-product, because
        PER_WELL_PRODUCT_CHECKS deliberately cannot answer cross-channel questions.
        """
        input:
            acquisition_inventory_csv=SCOPE_ACQUISITION_INVENTORY_CSV,
            position_well_mapping_csv=POSITION_WELL_MAPPING_CSV,
            resolved_product_plans=_resolved_product_plans_for_well,
            master_params_json=str(KEYENCE_STITCH_MAP_JSON),
        output:
            inventories=_MULTI_PRODUCT_INVENTORIES,
        params:
            device=lambda wc: str(config.get("image_building", {}).get("device", "cuda")),
            smoke_max=lambda wc: int(
                config.get("image_materialization", {}).get("smoke_max_time_indices", 0)
            ),
            # The three repeatable args are positionally ZIPPED by the task, so they are built from
            # ONE iteration over IMAGE_PRODUCT_KEYS -- the Nth key, plan and output csv belong to
            # the same product. Do not reorder one without the others.
            product_args=lambda wc, input, output: " ".join(
                f'--product-key "{key}" '
                f'--resolved-product-plan-json "{plan}" '
                f'--frame-inventory-product-csv "{out_csv}"'
                for key, plan, out_csv in zip(
                    IMAGE_PRODUCT_KEYS, input.resolved_product_plans, output.inventories
                )
            ),
        # gpu=1: as for the per-product rule -- this process runs LoG_focus_stacker's conv2d on the
        # GPU and holds that memory for the job's duration.
        resources:
            gpu=1,
        shell:
            """
            {MATERIALIZATION_RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-image-products-for-well \
              --experiment "{wildcards.experiment}" \
              --well-id "{wildcards.well_id}" \
              --scope "{SCOPE_TOKEN}" \
              {params.product_args} \
              --acquisition-inventory-csv "{input.acquisition_inventory_csv}" \
              --position-well-mapping-csv "{input.position_well_mapping_csv}" \
              --built-image-data-dir "{BUILT_IMAGE_DATA_DIR}" \
              --config-yaml "{CONFIG_YAML}" \
              --candidate "false" \
              --device "{params.device}" \
              --smoke-max-time-indices "{params.smoke_max}" \
              --input-root "{INPUTS_DIR}" \
              --master-params-path "{input.master_params_json}"
            """

else:

  rule materialize_image_product_for_well:
      """Materialize exactly one resolved image product into a product frame-inventory shard.

      The live path for every scope EXCEPT Keyence (which uses the multi-product rule above), so
      the Keyence-only master_params plumbing that used to be conditional here is now unreachable
      and has been dropped.
      """
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
      # gpu=1: this process runs LoG_focus_stacker's conv2d on the GPU and holds that memory for the
      # job's duration (real torch compute, not incidental import — see the comment on MATERIALIZATION_RUN
      # above). Do not copy onto a future model-server client rule.
      resources:
          gpu=1,
      shell:
          # MATERIALIZATION_RUN, not RUN: this task runs the per-well materializer ->
          # LoG_focus_stacker, a genuine torch/conv2d GPU compute path (Phase 0 finding — see
          # spec §1 "materialization is a THIRD compute zone"). Under runner == "conda" this is
          # identical to RUN (materialization has no separate conda env today); under
          # runner == "pixi" it resolves to the `materialization` pixi env instead of `pipeline`.
          """
          {MATERIALIZATION_RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-image-product-for-well \
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
            --input-root "{INPUTS_DIR}"
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
          --validation-scope "per_well_product"
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
