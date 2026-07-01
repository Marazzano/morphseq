"""feature_extraction — per-product feature packages.

Each feature product owns its own folder (compute + contract + entrypoint): mask_geometry,
curvature_metrics, pose_kinematics, stage_predictions, fraction_alive. There is no package-level
facade — import from the product you need. The former ``core/`` re-export facade and the
whole-experiment ``consolidate_features`` path were retired (superseded by the per-product folders
and their per-well DAG rules).
"""
