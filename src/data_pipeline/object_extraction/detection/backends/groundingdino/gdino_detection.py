"""
GroundingDINO Embryo Detection

Zero-shot object detection using GroundingDINO to identify embryos in microscopy images.
Provides seed detections for SAM2 mask propagation.

Key Functions:
    - load_groundingdino_model: Initialize model from config
    - detect_embryos: Run detection on a single image
    - calculate_containment: fraction of the smaller box inside the other (what IoU cannot see)
    - filter_detections: resolve raw boxes to one per object -- size/coverage bounds, then group
      same-object boxes by IoU OR containment, keeping the largest in each group. Confidence
      gating happens once, at detect_embryos via box_threshold.
    - select_seed_frame: Choose best frame for SAM2 initialization

Example Usage:
    ```python
    import torch
    from pathlib import Path

    # Load model
    model = load_groundingdino_model(
        config_path="GroundingDINO_SwinT_OGC.py",
        weights_path="groundingdino_swint_ogc.pth",
        device="cuda"
    )

    # Detect embryos
    detections = detect_embryos(
        model=model,
        image_path=Path("embryo_image.jpg"),
        text_prompt="individual embryo",
        box_threshold=0.35,
        text_threshold=0.25
    )

    # Resolve to one box per object. EVERY knob is required (PIPELINE_PHILOSOPHY P4a);
    # pass None/NaN to declare a bound deliberately absent.
    filtered = filter_detections(
        detections,
        iou_threshold=0.5,
        containment_threshold=0.85,
        min_detection_area_um2=600_000.0,
        max_detection_area_um2=6_500_000.0,
        max_frame_coverage=0.90,
        frame_area_um2=frame_width_px * frame_height_px * um_per_px ** 2,
    )
    ```

Detection Format:
    Each detection is a dict with:
    - box_xyxy: [x_min, y_min, x_max, y_max] in normalized coords [0, 1]
    - confidence: float confidence score
    - phrase: str matched text phrase
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from data_pipeline.utils.cuda_diagnostics import resolve_device

warnings.filterwarnings("ignore")

from data_pipeline.models.groundingdino import load_groundingdino_model as _load_groundingdino_model


def load_groundingdino_model(
    config_path: str,
    weights_path: str,
    device: str = "cuda"
) -> "torch.nn.Module":
    """
    Load GroundingDINO model.

    Args:
        config_path: Path to model config file (e.g., "GroundingDINO_SwinT_OGC.py")
        weights_path: Path to model weights (e.g., "groundingdino_swint_ogc.pth")
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Loaded GroundingDINO model

    Raises:
        ImportError: If GroundingDINO is not installed
        FileNotFoundError: If config or weights files don't exist

    Example:
        >>> model = load_groundingdino_model(
        ...     "models/GroundingDINO_SwinT_OGC.py",
        ...     "weights/groundingdino_swint_ogc.pth"
        ... )
    """
    device = resolve_device(device)
    config_path = Path(config_path)
    weights_path = Path(weights_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Support loading from a repo checkout (no pip install required). We infer the repo root
    # from the config path's typical location under `groundingdino/config/...`.
    repo_dir = config_path.parent.parent.parent
    return _load_groundingdino_model(
        repo_dir=repo_dir,
        config_path=config_path,
        weights_path=weights_path,
        device=device,
    )


def detect_embryos(
    model: "torch.nn.Module",
    image_path: Path,
    device: str = "cuda",
    text_prompt: str = "individual embryo",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25
) -> List[Dict]:
    """
    Detect embryos in an image using GroundingDINO.

    Args:
        model: Loaded GroundingDINO model
        image_path: Path to input image
        text_prompt: Text prompt for detection
        box_threshold: Box confidence threshold
        text_threshold: Text matching threshold

    Returns:
        List of detection dicts, each containing:
        - box_xyxy: [x_min, y_min, x_max, y_max] normalized [0, 1]
        - confidence: detection confidence score
        - phrase: matched text phrase

    Raises:
        FileNotFoundError: If image file doesn't exist

    Example:
        >>> detections = detect_embryos(
        ...     model,
        ...     Path("image.jpg"),
        ...     text_prompt="individual embryo"
        ... )
        >>> len(detections)
        3
        >>> detections[0]["confidence"]
        0.87
    """
    device = resolve_device(device)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Import required GroundingDINO utilities
    try:
        from groundingdino.util.inference import predict
        from groundingdino.util.utils import get_phrases_from_posmap
        import groundingdino.datasets.transforms as T
    except ImportError:
        raise ImportError("GroundingDINO utilities not available")

    # Load and transform image.
    # GroundingDINO's transforms expect a PIL image (uses `.size` as (w, h)).
    image_source = cv2.imread(str(image_path))
    if image_source is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_source)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image_pil, None)

    # Run detection
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=str(device),
    )

    # Convert to standard format
    detections = []
    for box, confidence, phrase in zip(boxes, logits, phrases):
        # `groundingdino.util.inference.predict` returns boxes in (cx, cy, w, h)
        # normalized to [0, 1]. Convert to (x_min, y_min, x_max, y_max) normalized.
        cx, cy, bw, bh = [float(v) for v in box.cpu().numpy().tolist()]
        x0 = max(0.0, min(1.0, cx - (bw / 2.0)))
        y0 = max(0.0, min(1.0, cy - (bh / 2.0)))
        x1 = max(0.0, min(1.0, cx + (bw / 2.0)))
        y1 = max(0.0, min(1.0, cy + (bh / 2.0)))
        detections.append({
            "box_xyxy": [x0, y0, x1, y1],
            "confidence": float(confidence.cpu().numpy()),
            "phrase": phrase
        })

    return detections


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [x_min, y_min, x_max, y_max] in normalized coords
        box2: [x_min, y_min, x_max, y_max] in normalized coords

    Returns:
        IoU value between 0 and 1

    Example:
        >>> box1 = [0.2, 0.2, 0.6, 0.6]
        >>> box2 = [0.4, 0.4, 0.8, 0.8]
        >>> iou = calculate_iou(box1, box2)
        >>> 0.0 < iou < 1.0
        True
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_containment(box1: List[float], box2: List[float]) -> float:
    """Fraction of the SMALLER box that lies inside the other: ``intersection / min(area)``.

    This is the metric IoU cannot express. A small box nested in a large one has low IoU *by
    construction* — the intersection is divided by the large union — so NMS never fires on it.
    Measured on the pbx pilot: nested yolk/head fragments scored IoU 0.11-0.38 while their
    containment was 0.994-1.000.

    Symmetric by construction (``max`` over both choices of denominator), so there is no "which box
    came first" ordering effect.

    Note ``containment >= IoU`` always, since ``min(area) <= union``; the two differ only in what
    they are lenient about. IoU is lenient when boxes are similar-sized and offset; containment is
    lenient when one box is much smaller and inside the other.

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        Containment in 0..1; 0.0 when the boxes do not overlap or either has zero area.

    Example:
        >>> outer = [0.0, 0.0, 1.0, 1.0]
        >>> inner = [0.1, 0.1, 0.2, 0.2]        # wholly inside, but only 1% of the area
        >>> round(calculate_containment(outer, inner), 3)
        1.0
        >>> round(calculate_iou(outer, inner), 3)   # IoU cannot see it
        0.01
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    smaller = min(area1, area2)

    return intersection / smaller if smaller > 0 else 0.0


def _is_bounded(value) -> bool:
    """True when a bound was actually chosen. None/NaN mean 'deliberately not bounded' (P4a)."""
    return value is not None and not math.isnan(float(value))


def _box_frame_coverage(box: List[float]) -> float:
    """Fraction of the frame the box covers. Boxes are NORMALIZED, so this is just its area."""
    return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))


def _box_area_um2(box: List[float], frame_area_um2: float) -> float:
    """Physical box area. Boxes are normalized, so coverage x frame area gives um^2."""
    return _box_frame_coverage(box) * frame_area_um2


def filter_detections(
    detections: List[Dict],
    *,
    iou_threshold,
    containment_threshold,
    min_detection_area_um2,
    max_detection_area_um2,
    max_frame_coverage,
    frame_area_um2=None,
) -> List[Dict]:
    """Resolve raw candidate boxes to one box per real object.

        raw detections
          -> STEP 1  size / coverage bounds  -> potential detections
          -> STEP 2  group boxes describing the SAME object (IoU or containment, transitively)
          -> STEP 3  keep the LARGEST box in each group

    EVERY knob is a required keyword argument, with no default — config owns the values and their
    provenance (PIPELINE_PHILOSOPHY P4a). Pass ``None`` or ``float('nan')`` to declare a bound
    deliberately absent; that check is then skipped. This is what keeps an option from becoming
    dead code: a caller cannot fail to notice a filter exists, only choose not to use it.

    Confidence gating already happened at detection time (``box_threshold`` in ``detect_embryos``);
    re-applying a stricter cutoff here would silently drop real detections that already cleared the
    detector's bar (a 0.45 filter over a 0.35 gate used to do exactly that — see the SAM3 exemplar
    review's F05 recovery investigation). Confidence policy lives in one place.

    Confidence is also NOT used to choose between grouped boxes: measured on the pilot, a nested
    yolk fragment out-scored the whole embryo in 2 of 7 wells, which would seed SAM2 on the yolk.
    The largest box wins instead.

    Args:
        detections: candidate dicts with ``box_xyxy`` (NORMALIZED 0-1) and ``confidence``.
        iou_threshold: group two similar-sized boxes on one object. NaN/None disables.
        containment_threshold: group a box nested inside another. NaN/None disables.
        min_detection_area_um2: reject boxes smaller than an embryo. NaN/None disables.
        max_detection_area_um2: reject boxes larger than an embryo. NaN/None disables.
        max_frame_coverage: reject boxes covering ~the whole frame. Scale-free. NaN/None disables.
        frame_area_um2: physical area of the full frame. REQUIRED when either um^2 bound is set;
            passing None with a um^2 bound raises rather than silently skipping the check.

    Returns:
        The kept subset of ``detections`` (input dicts, not copies).

    Raises:
        ValueError: a um^2 bound was requested but ``frame_area_um2`` is missing or non-positive.

    Example:
        >>> dets = [
        ...     {"box_xyxy": [0.2, 0.2, 0.6, 0.9], "confidence": 0.45},   # embryo
        ...     {"box_xyxy": [0.3, 0.7, 0.5, 0.9], "confidence": 0.50},   # yolk, inside it
        ... ]
        >>> kept = filter_detections(
        ...     dets, iou_threshold=0.5, containment_threshold=0.85,
        ...     min_detection_area_um2=None, max_detection_area_um2=None,
        ...     max_frame_coverage=None)
        >>> [d["confidence"] for d in kept]   # the larger box wins, not the more confident
        [0.45]
    """
    if not detections:
        return []

    wants_physical = _is_bounded(min_detection_area_um2) or _is_bounded(max_detection_area_um2)
    if wants_physical and not (frame_area_um2 and float(frame_area_um2) > 0):
        raise ValueError(
            "filter_detections: min/max_detection_area_um2 was set but frame_area_um2 is "
            f"{frame_area_um2!r}. A physical bound cannot be applied without the frame's physical "
            "area — pass frame_area_um2, or set the bound to NaN to declare it deliberately absent."
        )

    # ── STEP 1 — size / coverage bounds ────────────────────────────────────────────────────────
    potential_detections = []
    for index, detection in enumerate(detections):
        box = detection["box_xyxy"]
        if _is_bounded(max_frame_coverage) and _box_frame_coverage(box) > float(max_frame_coverage):
            continue
        if wants_physical:
            area_um2 = _box_area_um2(box, float(frame_area_um2))
            if _is_bounded(min_detection_area_um2) and area_um2 < float(min_detection_area_um2):
                continue
            if _is_bounded(max_detection_area_um2) and area_um2 > float(max_detection_area_um2):
                continue
        potential_detections.append(index)

    # ── STEP 2 — group boxes that describe the same object ─────────────────────────────────────
    # Transitive on purpose: a fragment may be nested in a sibling that is itself absorbed. A
    # greedy sequential loop drops that link and leaves the fragment orphaned (measured: a 3-box
    # well where the smallest box was 1.00 inside the middle box but only 0.85 inside the largest).
    same_object = {index: set() for index in potential_detections}
    for position, left in enumerate(potential_detections):
        for right in potential_detections[position + 1:]:
            box_l, box_r = detections[left]["box_xyxy"], detections[right]["box_xyxy"]
            grouped = (
                (_is_bounded(iou_threshold)
                 and calculate_iou(box_l, box_r) >= float(iou_threshold))
                or (_is_bounded(containment_threshold)
                    and calculate_containment(box_l, box_r) >= float(containment_threshold))
            )
            if grouped:
                same_object[left].add(right)
                same_object[right].add(left)

    # ── STEP 3 — one representative per group: the largest box ─────────────────────────────────
    # Two genuinely separate embryos do not contain each other, so they form separate groups and
    # BOTH survive. That is why this is a component walk and not "keep the single biggest box".
    kept_indices = []
    visited = set()
    for index in potential_detections:
        if index in visited:
            continue
        group, stack = [], [index]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            stack.extend(same_object[node] - visited)
        kept_indices.append(
            max(group, key=lambda i: _box_frame_coverage(detections[i]["box_xyxy"]))
        )

    return [detections[index] for index in sorted(kept_indices)]


def select_seed_frame(
    frame_detections: Dict[str, List[Dict]],
    min_detections: int = 1
) -> Optional[str]:
    """
    Select best seed frame for SAM2 initialization.

    Chooses frame with:
    1. At least min_detections embryos
    2. Highest average detection confidence

    Args:
        frame_detections: Dict mapping frame_id to list of detections
        min_detections: Minimum number of detections required

    Returns:
        Frame ID of selected seed frame, or None if no suitable frame

    Example:
        >>> frame_detections = {
        ...     "frame_0000": [{"confidence": 0.8}, {"confidence": 0.7}],
        ...     "frame_0001": [{"confidence": 0.9}, {"confidence": 0.85}],
        ...     "frame_0002": [{"confidence": 0.6}],
        ... }
        >>> seed = select_seed_frame(frame_detections, min_detections=2)
        >>> seed
        'frame_0001'
    """
    candidates = {}

    for frame_id, detections in frame_detections.items():
        if len(detections) >= min_detections:
            avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
            candidates[frame_id] = avg_confidence

    if not candidates:
        return None

    # Return frame with highest average confidence
    return max(candidates.items(), key=lambda x: x[1])[0]


def convert_boxes_to_sam2_format(
    detections: List[Dict],
    image_height: int,
    image_width: int
) -> np.ndarray:
    """
    Convert GroundingDINO boxes to SAM2 prompt format.

    SAM2 expects boxes in absolute pixel coordinates as numpy array.

    Args:
        detections: List of detection dicts with normalized box_xyxy
        image_height: Image height in pixels
        image_width: Image width in pixels

    Returns:
        Numpy array of shape (N, 4) with absolute pixel coordinates

    Example:
        >>> detections = [{"box_xyxy": [0.2, 0.3, 0.6, 0.7], "confidence": 0.9}]
        >>> boxes = convert_boxes_to_sam2_format(detections, 1000, 1000)
        >>> boxes.shape
        (1, 4)
        >>> boxes[0]
        array([200., 300., 600., 700.])
    """
    if not detections:
        return np.array([]).reshape(0, 4)

    boxes = []
    for det in detections:
        box_norm = det["box_xyxy"]
        # Convert normalized [0, 1] to absolute pixels
        box_abs = [
            box_norm[0] * image_width,
            box_norm[1] * image_height,
            box_norm[2] * image_width,
            box_norm[3] * image_height,
        ]
        boxes.append(box_abs)

    return np.array(boxes, dtype=np.float32)
