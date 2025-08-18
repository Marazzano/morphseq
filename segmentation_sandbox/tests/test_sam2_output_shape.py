import json
from scripts.detection_segmentation.sam2_utils import GroundedSamAnnotations


def test_format_and_conversion_helpers():
    # Fake seed detections with different bbox key names
    seed_detections = [
        {"box_xyxy": [10, 20, 30, 40], "confidence": 0.95, "label": "embryo"},
        {"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9, "label": "embryo"}
    ]

    formatted = GroundedSamAnnotations._format_seed_detections(None, seed_detections)

    # Check formatted detections structure
    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert 'original' in formatted[0]
    assert 'bbox_xyxy' in formatted[0]

    # Fake sam2_results keyed by numeric indices (as might be returned by some predictor)
    sam2_numeric = {
        0: {"0": {"embryo_id": "VID_e01", "snip_id": "VID_e01_s0000"}},
        "1": {"0": {"embryo_id": "VID_e01", "snip_id": "VID_e01_s0001"}}
    }

    image_ids_ordered = ["VID_t0000", "VID_t0001"]

    converted = GroundedSamAnnotations._convert_sam2_results_to_image_ids_format(None, sam2_numeric, image_ids_ordered)

    # After conversion, keys should be image ids
    assert set(converted.keys()) == set(image_ids_ordered)

    # Build a minimal video structure as the pipeline would
    seed_frame_info = {
        "video_id": "VID",
        "seed_frame": "VID_t0000",
        "seed_frame_index": 0,
        "num_embryos": 1,
        "detections": formatted,
        "is_first_frame": True,
        "all_frames": image_ids_ordered,
        "embryo_ids": ["VID_e01"],
        "requires_bidirectional_propagation": False,
        "bbox_format": "xyxy",
        "bbox_units": "pixels"
    }

    video_structure = {
        "video_id": "VID",
        "well_id": "VID",
        "seed_frame_info": seed_frame_info,
        "num_embryos": 1,
        "frames_processed": len(converted),
        "sam2_success": True,
        "processing_timestamp": "2025-08-14T00:00:00",
        "requires_bidirectional_propagation": False,
        "image_ids": converted
    }

    # Top-level simulated results
    results = {
        "experiments": {
            "EXP": {
                "experiment_id": "EXP",
                "videos": {"VID": video_structure}
            }
        },
        "snip_ids": ["VID_e01_s0000", "VID_e01_s0001"]
    }

    # Basic shape assertions
    assert 'experiments' in results
    exp = results['experiments']['EXP']
    assert 'videos' in exp
    vid = exp['videos']['VID']
    assert 'seed_frame_info' in vid
    assert 'image_ids' in vid
    assert isinstance(vid['image_ids'], dict)
    assert list(vid['image_ids'].keys()) == image_ids_ordered

    # Ensure seed_frame_info has expected fields
    s = vid['seed_frame_info']
    for k in ('seed_frame', 'seed_frame_index', 'detections', 'bbox_format', 'bbox_units'):
        assert k in s

    # Print JSON for manual inspection if needed (pytest will capture it)
    print(json.dumps(results, indent=2))
