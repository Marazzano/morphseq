"""Native-grid per-embryo fluorescence intensity.

WHY THIS IS NOT IN snip_processing.smk. A snip is rendered through INTER_AREA, a local average that
mixes photons across pixel boundaries -- biasing both saturation counts and the tails of the
distribution -- and it constant-fills outside the source, which would contaminate any annulus near a
frame edge. So photometry happens on the native uint16 raster and this step reads NO rendered snip
and needs NO snip transform. It carries snip_transform_id as a join key only.

That independence is visible in the DAG: these rules depend on frame_masks + frame_inventory, the
same inputs the geometry gate uses, and NOT on snip_materialization. Intensity and snips can run in
parallel, and an intensity failure cannot take snips down.

RAW ONLY. Nothing here subtracts a background. The null is pooled over a whole well and is not
knowable while measuring the first embryo; pooling is a well-grain estimator and lives in
feature_extraction. Keeping them apart is what lets the null be re-estimated later without re-reading
a single pixel.
"""

CHANNEL_INTENSITY_STEP = "channel_intensity"


def _channel_intensity_artifact(
    experiment: str, *, path_mode: str, well_id: str | None = None,
    source_image_product_key: str | None = None,
):
    format_vars = (
        {"source_image_product_key": source_image_product_key}
        if source_image_product_key is not None
        else None
    )
    return rule_artifact(
        CHANNEL_INTENSITY_STEP, "channel_intensity", experiment,
        path_mode=path_mode, well_id=well_id, format_vars=format_vars,
    )


def _intensity_source_products() -> list[str]:
    """Which source image products get measured.

    EXPLICIT CONFIG, NOT "every product". Intensity off a CLAHE'd raster is not a measurement, and
    a BF projection carries no dosage -- measuring everything would produce rows that look
    quantitative and are not. A caller names the quantitative products or nothing runs.
    """
    return list(config.get("channel_intensity", {}).get("source_image_products", []))


def _channel_intensity_artifacts_for_run(wc):
    # WELLS x SOURCE PRODUCTS. wells_for_experiment takes the wildcards object and triggers the
    # discover_wells checkpoint, which is what makes the well list correct rather than guessed.
    return [
        str(_channel_intensity_artifact(
            wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w,
            source_image_product_key=k,
        ))
        for w in wells_for_experiment(wc)
        for k in _intensity_source_products()
    ]


rule channel_intensity_per_well:
    """Measure every embryo-time in one well, for ONE source image product.

    Fanout is well x SOURCE PRODUCT because the source product changes what the number MEANS, not
    merely where it came from.

    FLUORESCENCE HAS NO MASKS OF ITS OWN: frame_masks carries BF rows only, since detection and
    segmentation are BF-only. An RFP frame finds its mask through the shared (well_id, time_index),
    and the entrypoint asserts the two grids agree rather than assuming it.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL, well_id="{well_id}",
        )),
        frame_masks_validated=str(_frame_masks_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        intensity=str(_channel_intensity_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}",
            source_image_product_key="{source_image_product_key}",
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks channel-intensity \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-csv "{output.intensity}" \
          --source-image-product-key "{wildcards.source_image_product_key}"
        """


rule merge_channel_intensity:
    """Row-stack the per-well x per-product shards. A DUMB CONCAT, deliberately.

    No estimator belongs here. Pooling the background null is a separate rule precisely because
    folding an estimator into a merge would make a structural operation semantic -- and the merged
    table stays product-aware without learning anything, since source_image_product_key is a column
    on every row.

    Consumes {input.per_well} directly rather than re-deriving the shard list: the rule already
    declares exactly the files it needs, and computing a second list in the shell body would be a
    second source of truth that could disagree with the DAG's own edges.

    required_columns IS THE POINT OF THE CONTRACT IMPORT. pd.concat will happily UNION two drifted
    schemas into a table full of NaN without complaining, so a shard from an older extractor would
    silently become a column of nulls. Supplying the owner's contract turns that into a loud
    failure -- and it matters most for embryo_clipped_px, since a merge that dropped saturation
    would hide exactly the compression that destroys a dosage comparison while looking like clean
    data.
    """
    input:
        per_well=_channel_intensity_artifacts_for_run,
    output:
        merged=str(_channel_intensity_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import concat_well_shards_to_file
from data_pipeline.object_extraction.channel_intensity.contract import CHANNEL_INTENSITY_COLUMNS
from pathlib import Path
shards = [Path(p) for p in '{input.per_well}'.split()]
concat_well_shards_to_file(shards, '{output.merged}', required_columns=CHANNEL_INTENSITY_COLUMNS, sort_columns=['experiment_id', 'well_id', 'time_index'])
"
        """
