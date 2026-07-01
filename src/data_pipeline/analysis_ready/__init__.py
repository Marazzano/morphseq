"""analysis_ready — STUB (not yet wired into the DAG).

The legacy analysis_ready subsystem (flat ``snip_id``/``time_int`` vocabulary, off-DAG loaders /
validators / assemble) was retired: it duplicated the identity spine and re-declared feature/QC
columns that their products already own. This stub is what replaces it.

Intended role (decided, not yet built): an OPTIONAL downstream product — ``snip_qc`` remains the
proven through-line terminal. When built, analysis_ready joins, keyed on the identity spine, the
wide per-snip feature columns with the ``snip_qc`` verdict into one analysis table for notebooks /
embeddings.

Doctrine for whoever wires this: **import spine and payload columns from their mint sites; never
re-declare them here.** The building blocks already exist and are re-exported below so a future
``contract.py`` composes from live sources:

  - identity spine + frame provenance  ← ``segmentation.physical_embryo_registry.snip_identity_contract``
  - snip_qc verdict payload            ← ``quality_control.snip_qc.contract`` (``use_snip``, reasons)
  - per-snip feature columns           ← each feature product's own ``contract.py``
    (mask_geometry, curvature_metrics, pose_kinematics, stage_predictions, fraction_alive)

Add the feature-product contract imports when the analysis_ready column set is decided; do not
inline a column list.
"""

from __future__ import annotations

from data_pipeline.quality_control.snip_qc.contract import SNIP_QC_PAYLOAD_COLUMNS
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_FRAME_PROVENANCE_COLUMNS,
    SNIP_ID_SPINE_COLUMNS,
)

# The identity spine every analysis-ready row must carry — imported, not re-declared. The feature
# payload (from the feature-product contracts) and the QC verdict payload get appended here once the
# analysis_ready column set is chosen and the product is wired.
ANALYSIS_READY_SPINE_COLUMNS: tuple[str, ...] = SNIP_ID_SPINE_COLUMNS + SNIP_FRAME_PROVENANCE_COLUMNS

__all__ = [
    "ANALYSIS_READY_SPINE_COLUMNS",
    "SNIP_QC_PAYLOAD_COLUMNS",
    "SNIP_ID_SPINE_COLUMNS",
    "SNIP_FRAME_PROVENANCE_COLUMNS",
]
