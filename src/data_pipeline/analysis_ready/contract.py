"""analysis_ready contract — composed, NOT re-declared.

The assembled analysis-ready table is one row per ``snip_id``. Its column set is the UNION of the
identity spine, every included feature product's payload, the snip_qc verdict, and the (well-
broadcast) plate metadata fields. Per the stub doctrine, **every column group below is IMPORTED from
its mint site** — this file declares no column-name literals of its own, so a rename upstream flows
here automatically and the products can never silently drift apart.

The one group that is not a static tuple is the latent embeddings: ``legacy_embeddings`` matches its
payload dynamically by the ``z_mu_*`` / ``z_sigma_*`` prefix (the width is model-dependent), so we
import the one static column it owns (``embedding_model_name``) and carry the prefix rule as a
documented predicate rather than a literal list.
"""

from __future__ import annotations

# ── identity spine + frame provenance (the base every row carries) ───────────────────────────────
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_ID_SPINE_COLUMNS,
    SNIP_FRAME_PROVENANCE_COLUMNS,
)

# ── per-snip feature payloads — imported from each product's own contract ─────────────────────────
from data_pipeline.feature_extraction.curvature_metrics.contract import CURVATURE_PAYLOAD_COLUMNS
from data_pipeline.feature_extraction.stage_predictions.contract import (
    STAGE_PREDICTION_PAYLOAD_COLUMNS,
    STAGE_PREDICTION_PROVENANCE_COLUMNS,
)
from data_pipeline.feature_extraction.mask_geometry.contract import MASK_GEOMETRY_PAYLOAD_COLUMNS
from data_pipeline.feature_extraction.pose_kinematics.contract import POSE_KINEMATICS_PAYLOAD_COLUMNS
from data_pipeline.feature_extraction.fraction_alive.contract import FRACTION_ALIVE_PAYLOAD_COLUMNS
from data_pipeline.feature_extraction.legacy_embeddings.contract import EMBEDDING_MODEL_NAME_COL

# ── QC verdict payload ────────────────────────────────────────────────────────────────────────────
from data_pipeline.quality_control.snip_qc.contract import SNIP_QC_PAYLOAD_COLUMNS

# ── plate metadata fields (broadcast well_id → snips) ─────────────────────────────────────────────
from data_pipeline.acquisition.metadata_ingest.plate.plate_metadata_contract import (
    REQUIRED_PLATE_METADATA_FIELDS,
)

# The spine every analysis-ready row must carry (5 IDs, additive one-per-level, + frame provenance).
ANALYSIS_READY_SPINE_COLUMNS: tuple[str, ...] = (
    SNIP_ID_SPINE_COLUMNS + SNIP_FRAME_PROVENANCE_COLUMNS
)

# Static per-snip feature payloads (the latent z_mu_*/z_sigma_* block is dynamic — see below).
ANALYSIS_READY_FEATURE_PAYLOAD_COLUMNS: tuple[str, ...] = (
    CURVATURE_PAYLOAD_COLUMNS
    + STAGE_PREDICTION_PAYLOAD_COLUMNS
    + STAGE_PREDICTION_PROVENANCE_COLUMNS
    + MASK_GEOMETRY_PAYLOAD_COLUMNS
    + POSE_KINEMATICS_PAYLOAD_COLUMNS
    + FRACTION_ALIVE_PAYLOAD_COLUMNS
    + (EMBEDDING_MODEL_NAME_COL,)
)

# The dynamic latent block is identified by prefix, not enumerated (width is model-dependent).
# NOTE two families: the pipeline encode step writes flat z_mu_00…; the disentangled models
# (assess_vae_results) additionally split z_mu_b_* (BIOLOGICAL) vs z_mu_n_* (nuisance). The
# biological subset (z_mu_b_*) is what UMAP is fit on when available.
LATENT_COLUMN_PREFIXES: tuple[str, ...] = ("z_mu_", "z_sigma_")
BIOLOGICAL_LATENT_PREFIX: str = "z_mu_b_"

# Plate broadcast policy: assemble carries ALL non-spine columns present in the plate CSV (the real
# schema is wider than the contract's required set — e.g. chem_perturbation, mold_type,
# series_number_map). REQUIRED_PLATE_METADATA_FIELDS is imported as the must-exist GUARANTEE (assemble
# fails loud if any is missing); the rest are carried opportunistically, like the z_mu_* block. The
# spine keys (experiment_id/well_id) are dropped on join since they already live in the spine.
ANALYSIS_READY_REQUIRED_PLATE_COLUMNS: tuple[str, ...] = REQUIRED_PLATE_METADATA_FIELDS
_SPINE_SET = frozenset(ANALYSIS_READY_SPINE_COLUMNS)


def broadcast_plate_columns(plate_columns: "tuple[str, ...] | list[str]") -> list[str]:
    """Non-spine plate columns to broadcast onto snips, given a plate table's actual columns.

    Carries everything the plate CSV has except the spine keys (which the snip already owns). The
    required fields are asserted present by the assemble step, not filtered here.
    """
    return [c for c in plate_columns if c not in _SPINE_SET]


def is_latent_column(name: str) -> bool:
    """True if ``name`` is a dynamic latent column (z_mu_* / z_sigma_*)."""
    return any(name.startswith(p) for p in LATENT_COLUMN_PREFIXES)


def select_latent_columns(
    columns: "tuple[str, ...] | list[str]", *, prefer_biological: bool = True
) -> list[str]:
    """The latent-mean columns to fit UMAP on, given a table's actual columns.

    When ``prefer_biological`` and any ``z_mu_b_*`` columns exist, return ONLY those (the
    disentangled biological subset, matching assess_vae_results). Otherwise fall back to all
    ``z_mu_*`` mean columns (the flat pipeline output). z_sigma_* are never fit on.
    """
    z_mu = [c for c in columns if c.startswith("z_mu_")]
    if prefer_biological:
        bio = [c for c in z_mu if c.startswith(BIOLOGICAL_LATENT_PREFIX)]
        if bio:
            return bio
    return z_mu


# The contract-pinned column core (spine + static feature payloads + QC + REQUIRED plate fields).
# The full assembled table additionally carries the dynamic latent (z_mu_*/z_sigma_*) block and any
# extra (non-required) plate columns the CSV happens to have — both resolved at assemble time.
ANALYSIS_READY_CORE_COLUMNS: tuple[str, ...] = (
    ANALYSIS_READY_SPINE_COLUMNS
    + ANALYSIS_READY_FEATURE_PAYLOAD_COLUMNS
    + SNIP_QC_PAYLOAD_COLUMNS
    + ANALYSIS_READY_REQUIRED_PLATE_COLUMNS
)

__all__ = [
    "ANALYSIS_READY_SPINE_COLUMNS",
    "ANALYSIS_READY_FEATURE_PAYLOAD_COLUMNS",
    "ANALYSIS_READY_REQUIRED_PLATE_COLUMNS",
    "ANALYSIS_READY_CORE_COLUMNS",
    "LATENT_COLUMN_PREFIXES",
    "BIOLOGICAL_LATENT_PREFIX",
    "SNIP_QC_PAYLOAD_COLUMNS",
    "broadcast_plate_columns",
    "is_latent_column",
    "select_latent_columns",
]
