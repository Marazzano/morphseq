"""morphseq_integration — join the morphology and sequencing modalities at embryo resolution.

A peer package to ``data_pipeline``. ``data_pipeline`` owns imaging: raw frames through to the
per-snip ``analysis_ready`` table. This package owns the *bridge* to sci-PLEX sequencing, so that
transcriptional state can be read as a function of morphology and vice versa.

The bridge is one small idea. Every morphseq plate workbook that has a paired sequencing run carries
two extra 8x12 sheets — ``image_to_hash_map`` (which hash well each imaging well went into) and
``hash_plate_num`` (which hash plate). Combined with one curated fact per experiment, the imaging
experiment's sci-PLEX name, that is enough to mint the sequencing sample key for every imaging well.

    build_crosswalk()  ->  well_id <-> seq_sample_id, one row per imaging well

That crosswalk is identity only. Morphology joins onto ``well_id``; sequencing joins onto
``seq_sample_id``.

    build_master_table()  ->  the crosswalk + legacy morphology latents + sequencing metadata

Morphology comes from the LEGACY VAE latents (``legacy_morph``), not the current pipeline's, because
the pipeline's snip raster is degraded relative to the build that trained that checkpoint. The master
table deliberately carries no QC from the current pipeline for the same reason — see ``assemble``.
"""

from __future__ import annotations

from .assemble import (
    build_master_table,
    coverage_summary,
    load_seq_metadata,
    verify_seq_pairing,
    write_master_table,
)
from .crosswalk import (
    ALL_STATUSES,
    CROSSWALK_COLUMNS,
    STATUS_BLANK_WELL,
    STATUS_NO_HASH_MAP,
    STATUS_PAIRED,
    build_crosswalk,
    summarize,
    validate_crosswalk,
)
from .experiment_key import ExperimentKey, load_experiment_key
from .hash_map import experiments_with_hash_map, load_experiment_hash_map
from .identifiers import (
    build_seq_sample_id,
    format_hash_plate,
    normalize_hash_well,
    strip_rt_block,
    to_sequencing_well,
)
from .legacy_morph import (
    attach_legacy_metadata,
    available_experiments,
    load_legacy_latents,
    load_many_legacy_latents,
    parse_legacy_snip_id,
    to_flat_latent_names,
)
from .paths import ENV_PIPELINE_ROOT, PipelinePaths, default_paths

__all__ = [
    "ALL_STATUSES",
    "CROSSWALK_COLUMNS",
    "ENV_PIPELINE_ROOT",
    "STATUS_BLANK_WELL",
    "STATUS_NO_HASH_MAP",
    "STATUS_PAIRED",
    "ExperimentKey",
    "PipelinePaths",
    "attach_legacy_metadata",
    "available_experiments",
    "build_crosswalk",
    "build_master_table",
    "build_seq_sample_id",
    "coverage_summary",
    "default_paths",
    "experiments_with_hash_map",
    "format_hash_plate",
    "load_experiment_hash_map",
    "load_experiment_key",
    "load_legacy_latents",
    "load_many_legacy_latents",
    "load_seq_metadata",
    "normalize_hash_well",
    "parse_legacy_snip_id",
    "strip_rt_block",
    "summarize",
    "to_flat_latent_names",
    "to_sequencing_well",
    "validate_crosswalk",
    "verify_seq_pairing",
    "write_master_table",
]
