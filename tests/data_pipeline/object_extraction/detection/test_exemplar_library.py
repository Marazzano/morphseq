from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.object_extraction.detection.exemplar_library import (
    exemplars_for_sam3_prompt,
    load_exemplar_manifest,
    resolve_exemplar_set,
    validate_exemplar_manifest,
)


def _manifest_rows():
    return [
        {
            "exemplar_id": "zf_default_early_001",
            "concept_label": "zebrafish embryo",
            "prompt_role": "positive",
            "prompt_type": "box",
            "reference_image_path": "images/20250912_B01_BF_t0000.png",
            "bbox_x_min_px": 18,
            "bbox_y_min_px": 24,
            "bbox_x_max_px": 76,
            "bbox_y_max_px": 82,
            "notes": "early timepoint; one embryo in multi-embryo well",
        },
        {
            "exemplar_id": "zf_default_neg_001",
            "concept_label": "zebrafish embryo",
            "prompt_role": "negative",
            "prompt_type": "box",
            "reference_image_path": "images/20250912_B01_BF_t0002.png",
            "bbox_x_min_px": 2,
            "bbox_y_min_px": 3,
            "bbox_x_max_px": 20,
            "bbox_y_max_px": 22,
            "notes": "debris/background",
        },
    ]


def test_loads_and_resolves_model_neutral_exemplar_manifest(tmp_path: Path):
    set_root = tmp_path / "zebrafish_default"
    (set_root / "images").mkdir(parents=True)
    (set_root / "images/20250912_B01_BF_t0000.png").touch()
    (set_root / "images/20250912_B01_BF_t0002.png").touch()
    pd.DataFrame(_manifest_rows()).to_csv(set_root / "manifest.csv", index=False)

    exemplar_set = resolve_exemplar_set("zebrafish_default", library_root=tmp_path)

    assert exemplar_set.name == "zebrafish_default"
    assert len(exemplar_set.exemplars) == 2
    first = exemplar_set.exemplars[0]
    assert first.box_xyxy == (18.0, 24.0, 76.0, 82.0)
    assert first.reference_image_path == set_root / "images/20250912_B01_BF_t0000.png"

    payload = exemplars_for_sam3_prompt(exemplar_set.exemplars)
    assert payload[0] == {
        "exemplar_id": "zf_default_early_001",
        "role": "positive",
        "type": "box",
        "image_path": str(set_root / "images/20250912_B01_BF_t0000.png"),
        "box_xyxy": [18.0, 24.0, 76.0, 82.0],
    }


def test_rejects_polymorphic_legacy_box_or_mask_column():
    df = pd.DataFrame([
        {
            "exemplar_id": "bad",
            "concept_label": "zebrafish embryo",
            "prompt_role": "positive",
            "prompt_type": "box",
            "reference_image_path": "img.png",
            "reference_box_or_mask": "[1, 2, 3, 4]",
        }
    ])

    with pytest.raises(ValueError, match="missing required exemplar columns"):
        validate_exemplar_manifest(df)


def test_rejects_invalid_box_order(tmp_path: Path):
    rows = _manifest_rows()
    rows[0]["bbox_x_min_px"] = 76
    rows[0]["bbox_x_max_px"] = 18
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="bbox_x_min_px"):
        load_exemplar_manifest(manifest)
