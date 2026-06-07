"""Frame-inventory product-family rules.

Thin scaffold only.

``paths.py`` already declares the target product family:

    step:     frame_inventory
    artifact: inventory
    modes:    per_well | merged

The live root Snakefile still produces the legacy frame-contract outputs
(``frame_contract.csv`` / ``wells.txt``) directly, so there are no live
``frame_inventory`` Snakemake rules to extract yet. Keep this file as the
product-family landing zone and add only frame-inventory actions here when
that contract is wired:

    build_frame_inventory_for_well
    validate_frame_inventory_for_well
    merge_frame_inventory
    write_frame_inventory_provenance

Do not split detection/segmentation here. Do not bulk-import stale rule
fragments. All future frame-inventory paths in this file must come from
``data_pipeline.pipeline_orchestrator.orchestration.paths``.
"""
