"""Focused tests for the SeaHub exploratory workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import seahub_workflow as workflow


class SeahubWorkflowTests(unittest.TestCase):
    def test_metadata_reader(self):
        metadata = workflow.load_collection_metadata()
        self.assertGreater(len(metadata), 1_000)
        self.assertIn("collection_name", metadata)
        self.assertIn("collection_date", metadata)
        self.assertIn("GENE16", set(metadata["expt"].dropna()))

    def test_observed_name_conventions(self):
        standard = workflow.parse_image_name(
            "genetic_perturbations/GENE16/Batch 1/images/"
            "24hpf_A_foxc1a.jpg"
        )
        self.assertEqual(standard["stage_hpf"], 24.0)
        self.assertEqual(standard["stage_source_label"], "24hpf")
        self.assertEqual(standard["fov_label"], "A")
        self.assertEqual(standard["perturbation_parsed"], "foxc1a")

        legacy = workflow.parse_image_name(
            "genetic_perturbations/GENE1/Batch 3/tbx6_24hpf/"
            "tbx6_20240717_A05_uncropped.jpg"
        )
        self.assertEqual(legacy["stage_hpf"], 24.0)
        self.assertEqual(legacy["source_embryo_label"], "A05")
        self.assertEqual(legacy["crop_variant"], "uncropped")

        chemical = workflow.parse_image_name(
            "chemical_perturbations/CHEM13/images/"
            "16_60hpf_SB505124_96hpf.jpg"
        )
        self.assertEqual(chemical["stage_addition_hpf"], 60.0)
        self.assertEqual(chemical["stage_hpf"], 96.0)
        self.assertEqual(chemical["perturbation_parsed"], "SB505124")

    def test_position_assignment(self):
        boxes = np.array(
            [
                [0.10, 0.10, 0.20, 0.35],
                [0.30, 0.12, 0.40, 0.37],
                [0.50, 0.11, 0.60, 0.36],
                [0.70, 0.13, 0.80, 0.38],
                [0.11, 0.55, 0.21, 0.85],
                [0.31, 0.57, 0.41, 0.87],
                [0.51, 0.56, 0.61, 0.86],
                [0.71, 0.58, 0.81, 0.88],
            ]
        )
        np.testing.assert_array_equal(
            workflow.assign_embryo_positions(boxes), np.arange(1, 9)
        )

    def test_mock_detection_to_eight_snips(self):
        boxes = np.array(
            [
                [0.02, 0.05, 0.23, 0.45],
                [0.27, 0.05, 0.48, 0.45],
                [0.52, 0.05, 0.73, 0.45],
                [0.77, 0.05, 0.98, 0.45],
                [0.02, 0.55, 0.23, 0.95],
                [0.27, 0.55, 0.48, 0.95],
                [0.52, 0.55, 0.73, 0.95],
                [0.77, 0.55, 0.98, 0.95],
            ]
        )
        scores = np.linspace(0.95, 0.80, 8)
        phrases = ["individual embryo"] * 8
        details = {
            "box_threshold": 0.15,
            "text_threshold": 0.10,
            "raw_detection_count": 8,
            "nms_detection_count": 8,
            "selected_detection_count": 8,
        }
        with tempfile.TemporaryDirectory(dir=workflow.WORK_DIR) as temporary:
            temporary_path = Path(temporary)
            image_path = temporary_path / "mock_fov.jpg"
            Image.new("RGB", (400, 300), "white").save(image_path)
            images = pd.DataFrame(
                [
                    {
                        "image_id": "mock-image",
                        "image_path": str(image_path),
                        "experiment_id": "GENE_TEST",
                        "metadata_collection_name": "GENE_TEST_demo_24hpf",
                    }
                ]
            )
            with patch.object(
                workflow,
                "_predict_threshold_sweep",
                return_value=(boxes, scores, phrases, details),
            ):
                manifest, qc = workflow.run_grounding_dino_segmentation(
                    images,
                    temporary_path / "output",
                    model=object(),
                )
            self.assertEqual(len(manifest), 8)
            self.assertEqual(
                sorted(manifest["embryo_position"].tolist()), list(range(1, 9))
            )
            self.assertTrue(
                manifest["snip_color_path"]
                .map(lambda value: Path(value).is_file())
                .all()
            )
            self.assertTrue(
                manifest["snip_grayscale_path"]
                .map(lambda value: Path(value).is_file())
                .all()
            )
            with Image.open(manifest.iloc[0]["snip_color_path"]) as color:
                self.assertEqual(color.mode, "RGB")
            with Image.open(manifest.iloc[0]["snip_grayscale_path"]) as grayscale:
                self.assertEqual(grayscale.mode, "L")
            self.assertEqual(qc.iloc[0]["segmentation_qc_status"], "pass")

    def test_contact_sheet_has_confidence_labels(self):
        with tempfile.TemporaryDirectory(dir=workflow.WORK_DIR) as temporary:
            temporary_path = Path(temporary)
            records = []
            for position in range(1, 9):
                color_path = temporary_path / f"color_{position}.jpg"
                grayscale_path = temporary_path / f"gray_{position}.jpg"
                Image.new("RGB", (80 + position, 100), "white").save(color_path)
                Image.new("L", (80 + position, 100), 255).save(grayscale_path)
                records.append(
                    {
                        "embryo_position": position,
                        "detection_confidence": 0.5 + position / 100,
                        "snip_color_path": str(color_path),
                        "snip_grayscale_path": str(grayscale_path),
                    }
                )
            manifest = pd.DataFrame.from_records(records)
            color_output = workflow.save_embryo_contact_sheet(
                manifest,
                temporary_path / "color_sheet.jpg",
                variant="color",
            )
            grayscale_output = workflow.save_embryo_contact_sheet(
                manifest,
                temporary_path / "grayscale_sheet.jpg",
                variant="grayscale",
            )
            with Image.open(color_output) as color:
                self.assertEqual(color.mode, "RGB")
                self.assertLess(
                    min(
                        sum(color.getpixel((x, y)))
                        for x in range(8, 50)
                        for y in range(8, 30)
                    ),
                    100,
                )
            with Image.open(grayscale_output) as grayscale:
                self.assertEqual(grayscale.mode, "L")
                self.assertLess(
                    min(
                        grayscale.getpixel((x, y))
                        for x in range(8, 50)
                        for y in range(8, 30)
                    ),
                    40,
                )

    def test_output_guard(self):
        with self.assertRaises(ValueError):
            workflow._require_work_output("/tmp/not-a-seahub-output")


if __name__ == "__main__":
    unittest.main()
