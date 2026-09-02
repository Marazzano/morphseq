"""The physical embryo registry — the identity-origination boundary.

This package answers one question and owns it exclusively: *given validated frame
masks (with track identity), which physical embryos exist, what are their stable
IDs, and where did each one come from?* It is the only place in the pipeline where
``track_id → physical_embryo_id`` resolution happens.

It also AUTHORS the identity-carrying spine law (``snip_identity_contract``): the
parent ``physical_embryo_id`` must travel with every derived snip/embryo-grain row
and be validated against every derived ID. This world owns the meaning of physical
embryo identity; every downstream product merely calls the shared validator.

See docs/data_pipeline/specs/target/specs/detect-seg-track/targets/
physical_embryo_registry_world.md.
"""
