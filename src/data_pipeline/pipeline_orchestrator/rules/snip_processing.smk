"""Snip-processing product-family rules.

Consumes the per-well frame_masks shard and the per-well frame_inventory shard,
extracts per-embryo crops, mints snip_inventory rows, and emits the per-well
snip_inventory shard + pixel files under the same per-well directory.

Yolk masks are optional: when absent the rotation step falls back to a
mass-distribution heuristic and extraction uses a zero yolk mask.
"""

from data_pipeline.object_extraction.snip_processing.snip_frame_shape import (
    resolve_snip_frame_shape as _resolve_snip_frame_shape,
)
from data_pipeline.object_extraction.snip_processing.defaults import (
    DEFAULT_BLEND_RADIUS_UM,
    DEFAULT_TARGET_PIXEL_SIZE_UM,
)

SNIP_INVENTORY_STEP = "snip_inventory"

from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
)

# THE GEOMETRY GATE. snip_geometry derives the canonical transform ONCE per embryo-time from the BF
# mask; every render job resolves that shared recipe onto its own product grid. Per-well fanout with
# NO product dimension, deliberately: one embryo-time has one canonical transform, and a product
# wildcard here would derive it N times -- the exact drift the gate exists to close.
SNIP_GEOMETRY_STEP = "snip_geometry"


def _snip_geometry_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(SNIP_GEOMETRY_STEP, "snip_transforms", experiment, path_mode=path_mode, well_id=well_id)

# Upstream identity source: physical_embryo_registry owns the track_id -> physical_embryo_id
# resolution. snip_processing JOINS the PER-WELL shard (not the merged table) — the crop loop is
# per-well and joins on (well_id, track_id), so it depends only on this well's registry; depending
# on the merged table would force every well to finish before any well could crop.
PHYSICAL_EMBRYO_REGISTRY_STEP = "physical_embryo_registry"


def _physical_embryo_registry_per_well(experiment: str, *, well_id: str):
    return rule_artifact(PHYSICAL_EMBRYO_REGISTRY_STEP, "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _physical_embryo_registry_per_well_validated(experiment: str, *, well_id: str):
    return rule_validated(PHYSICAL_EMBRYO_REGISTRY_STEP, "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snip_inventory_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None, snip_product_key: str | None = None):
    # The product key rides in format_vars -- the sanctioned channel for filename-level tokens that
    # are not identity. paths.py only SUBSTITUTES it; it never mints one (the no-leakage rule).
    fmt = {"snip_product_key": snip_product_key} if snip_product_key is not None else None
    return rule_artifact(SNIP_INVENTORY_STEP, artifact, experiment, path_mode=path_mode, well_id=well_id, format_vars=fmt)


def _legacy_default_snip_inventory(experiment: str, *, well_id: str):
    """The product-free path ~24 existing call sites still read. A symlink, never a real file."""
    return rule_artifact(
        SNIP_INVENTORY_STEP, "legacy_default_snip_inventory", experiment,
        path_mode=PATH_MODE_PER_WELL, well_id=well_id,
    )

def _snip_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None, snip_product_key: str | None = None):
    fmt = {"snip_product_key": snip_product_key} if snip_product_key is not None else None
    return rule_validated(SNIP_INVENTORY_STEP, "snip_inventory", experiment, path_mode=path_mode, well_id=well_id, format_vars=fmt)

def _snip_inventory_provenance(experiment: str, *, path_mode: str, well_id: str | None = None, snip_product_key: str | None = None):
    return str(_snip_inventory_artifact(
        experiment, "snip_inventory", path_mode=path_mode, well_id=well_id,
        snip_product_key=snip_product_key,
    )) + ".provenance.json"

def _snip_inventory_artifacts_for_run(wc):
    # WELLS x PRODUCTS, over the AUTHORITATIVE shards -- never the compatibility alias, which would
    # otherwise double-count BF through both paths.
    return [
        str(_snip_inventory_artifact(
            wc.experiment, "snip_inventory",
            path_mode=PATH_MODE_PER_WELL, well_id=w, snip_product_key=k,
        ))
        for w in wells_for_experiment(wc)
        for k in SNIP_PRODUCT_KEYS
    ]

def _snip_inventory_validated_for_run(wc):
    return [
        str(_snip_inventory_validated(
            wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w, snip_product_key=k,
        ))
        for w in wells_for_experiment(wc)
        for k in SNIP_PRODUCT_KEYS
    ]

def _snip_inventory_provenance_for_run(wc):
    return [
        _snip_inventory_provenance(
            wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w, snip_product_key=k,
        )
        for w in wells_for_experiment(wc)
        for k in SNIP_PRODUCT_KEYS
    ]

def _snip_inventory_snips_dir(experiment: str, well_id: str) -> str:
    """Per-well pixel directory: sits beside the shard CSV under per_well/{well_id}/."""
    return rule_step_dir(SNIP_INVENTORY_STEP, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id) + "/snips"


rule snip_geometry_per_well:
    """Derive the canonical snip transform once per embryo-time for one well.

    READS MASKS, NOT PIXELS. The rotation angle comes from the mask's PCA orientation and the crop
    center from its extent, so this rule needs frame_masks + frame_inventory (for calibration) +
    the registry, and no materialized image product. That is also what lets sibling render jobs run
    in PARALLEL: none waits on another's artifacts, only on this table.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
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
        physical_embryo_registry=str(_physical_embryo_registry_per_well(
            "{experiment}", well_id="{well_id}"
        )),
        physical_embryo_registry_validated=str(_physical_embryo_registry_per_well_validated(
            "{experiment}", well_id="{well_id}"
        )),
    output:
        snip_transforms=str(_snip_geometry_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    params:
        target_pixel_size_um=lambda wc: float(
            config.get("snip_processing", {}).get("target_pixel_size_um", 7.8)
        ),
        output_height_px=lambda wc: int(_resolve_snip_frame_shape(config)[0]),
        output_width_px=lambda wc: int(_resolve_snip_frame_shape(config)[1]),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-geometry \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-path "{output.snip_transforms}" \
          --target-pixel-size-um {params.target_pixel_size_um} \
          --output-height-px {params.output_height_px} \
          --output-width-px {params.output_width_px}
        """


rule snip_materialization_per_well:
    """Extract per-embryo crops from validated frame_masks for one well.

    Reads frame_masks + frame_inventory shards, JOINS physical_embryo_id from
    the per-well physical_embryo_registry shard (identity is no longer minted
    here), builds embryo_id / snip_id via shared/identifiers, runs the
    extraction/rotation/augmentation stack, and emits the per-well
    snip_inventory shard + pixel files. Yolk masks are optional; rotation
    degrades gracefully without them.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
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
        physical_embryo_registry=str(_physical_embryo_registry_per_well(
            "{experiment}", well_id="{well_id}"
        )),
        physical_embryo_registry_validated=str(_physical_embryo_registry_per_well_validated(
            "{experiment}", well_id="{well_id}"
        )),
        # THE GATE. Geometry is derived once by snip_geometry; this rule resolves that shared
        # recipe onto its own product grid and may not derive its own.
        snip_transforms=str(_snip_geometry_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
            snip_product_key="{snip_product_key}",
        )),
        provenance=_snip_inventory_provenance(
            "{experiment}", path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}", snip_product_key="{snip_product_key}",
        ),
    params:
        snip_product_key=lambda wc: wc.snip_product_key,
        snips_dir=lambda wc: _snip_inventory_snips_dir(wc.experiment, wc.well_id),
        target_pixel_size_um=lambda wc: float(
            config.get("snip_processing", {}).get(
                "target_pixel_size_um", DEFAULT_TARGET_PIXEL_SIZE_UM
            )
        ),
        # Crop output (H, W) comes from the single snip_frame_shape source of truth so the snip
        # image, the saved embryo mask, and the per-snip via mask all share one grid.
        output_height_px=lambda wc: int(_resolve_snip_frame_shape(config)[0]),
        output_width_px=lambda wc: int(_resolve_snip_frame_shape(config)[1]),
        background_noise_scale=lambda wc: float(
            config.get("snip_processing", {}).get("background_noise_scale", 0.1)
        ),
        blend_radius_um=lambda wc: float(
            config.get("snip_processing", {}).get(
                "blend_radius_um", DEFAULT_BLEND_RADIUS_UM
            )
        ),
        # Legacy microscopy/checkpoint behavior stays ON by default. SeaHub's
        # runtime overlay explicitly sets this false for already-normalized
        # inverted 8-bit source images.
        apply_clahe=lambda wc: str(
            config.get("snip_processing", {}).get("apply_clahe", True)
        ).lower(),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-processing \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.snip_inventory}" \
          --snip-transform-table-csv "{input.snip_transforms}" \
          --snip-product-key "{params.snip_product_key}" \
          --snips-dir "{params.snips_dir}" \
          --output-root "{DATA_ROOT}" \
          --target-pixel-size-um "{params.target_pixel_size_um}" \
          --output-height-px "{params.output_height_px}" \
          --output-width-px "{params.output_width_px}" \
          --background-noise-scale "{params.background_noise_scale}" \
          --blend-radius-um "{params.blend_radius_um}" \
          --apply-clahe "{params.apply_clahe}"
        """


rule legacy_default_snip_inventory_alias:
    """Point the product-free path at the DEFAULT BF product's inventory.

    A separate rule rather than a post-write side effect inside the render job, so the dependency is
    visible in the DAG: legacy consumers depend on THIS, which depends on the default BF shard. The
    RFP job never sees the legacy path at all, which is what makes "exactly one product owns the
    alias" structural instead of a convention.

    TODO(deprecate-legacy-snip-inventory-alias): remove with the last product-unaware consumer.
    """
    input:
        default_inventory=lambda wc: str(_snip_inventory_artifact(
            wc.experiment, "snip_inventory",
            path_mode=PATH_MODE_PER_WELL, well_id=wc.well_id,
            snip_product_key=DEFAULT_BF_SNIP_PRODUCT_KEY,
        )),
    output:
        legacy_inventory=str(_legacy_default_snip_inventory("{experiment}", well_id="{well_id}")),
    run:
        from data_pipeline.object_extraction.snip_processing.legacy_snip_paths import (
            link_legacy_flat_path,
        )

        link_legacy_flat_path(
            canonical_path=Path(input.default_inventory),
            legacy_path=Path(output.legacy_inventory),
            snip_product_key=DEFAULT_BF_SNIP_PRODUCT_KEY,
        )


rule validate_snip_inventory_for_well:
    """Validate the per-well snip_inventory shard and write a .validated sentinel."""
    input:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
            snip_product_key="{snip_product_key}",
        )),
        provenance=_snip_inventory_provenance(
            "{experiment}", path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}", snip_product_key="{snip_product_key}",
        ),
    output:
        validated=str(_snip_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}",
            snip_product_key="{snip_product_key}",
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-snip-inventory \
          --input-csv "{input.snip_inventory}" \
          --provenance-json "{input.provenance}" \
          --output-flag "{output.validated}"
        """


rule merge_snip_inventory:
    """Row-stack per-well snip_inventory shards into the experiment-level merged table."""
    input:
        per_well=_snip_inventory_artifacts_for_run,
        # Wait on each shard's .validated sentinel too, not just the artifact — otherwise the merge
        # can fire in the window after the shard CSV exists but before validation writes the
        # sentinel, and collect_well_shard_paths (which requires both) sees zero validated shards
        # and raises "no shards to concatenate". Matches the SAFE merge rules (e.g. merge_frame_masks).
        per_well_validated=_snip_inventory_validated_for_run,
        per_well_provenance=_snip_inventory_provenance_for_run,
    output:
        merged=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_MERGED,
        )),
        merged_provenance=_snip_inventory_provenance(
            "{experiment}", path_mode=PATH_MODE_MERGED,
        ),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import concat_well_shards_to_file
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import SNIP_INVENTORY_COLUMNS
from data_pipeline.object_extraction.snip_processing.provenance import merge_rendering_sidecars
from pathlib import Path
# THE SHARDS SNAKEMAKE ALREADY RESOLVED, not a re-derivation. collect_well_shard_paths builds
# per-well paths from the registry and knows nothing about the product dimension, so it cannot fill
# {{snip_product_key}} -- and re-deriving a list the rule already has as input: is a second source
# of truth regardless. input.per_well is wells x products by construction.
shards = [Path(p) for p in '{input.per_well}'.split()]
concat_well_shards_to_file(shards, '{output.merged}', required_columns=SNIP_INVENTORY_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
merge_rendering_sidecars(
    [Path(p) for p in '{input.per_well_provenance}'.split()],
    Path('{output.merged_provenance}'),
)
"
        """
