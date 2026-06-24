"""Product-specific frame_inventory contract rules (L3 grain + L4 sources).

This module holds the strict, frame_inventory-specific contract rules that
``frame_inventory_validation.py`` sequences: BF-contiguity, channel rectangularity, the per-well
temporal rule, source-path resolution / readability, and the declared-dimension self-check. It
IMPORTS the column vocabulary from ``image_materialization/frame_inventory_contract.py`` — it never
duplicates the contract's column names or the REQUIRED_CHANNEL.

STUB (Step 1): the rules land in Step 2. Kept as an explicit seam so the public gate's import target
exists and the green refactor is a clean, reviewable move with no behaviour change.
"""

from __future__ import annotations

# Rules implemented in Step 2:
#   L3 (scope-aware grain):
#     - one experiment_id (both scopes)
#     - per_well: one well_id/well_index; BF contiguous 0..N-1; channels share one time_index set;
#       >1 distinct time_index => elapsed_time_s required
#     - merged: many wells allowed; the per-well temporal checks applied grouped by well_id
#   L4 (gated by check_sources):
#     - resolve source_image_path (absolute direct; relative under image_root, no `..` escape)
#     - file exists; image opens (ALLOWED_IMAGE_SUFFIXES); real dims == declared; um/px > 0
