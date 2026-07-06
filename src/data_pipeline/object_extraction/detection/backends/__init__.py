"""Detection backends — each translates one detector's native output into shared frame_detections rows.

Every backend exposes the same ``detect_frame(model, image_path, *, identity_row, detector_model_id,
config)`` adapter so the shared router (``detection/run_frame_detection.py``) stays backend-agnostic.
"""
