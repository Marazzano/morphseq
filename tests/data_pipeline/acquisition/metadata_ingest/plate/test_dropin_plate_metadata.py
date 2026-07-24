from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.plate.dropin_plate_metadata import (
    ingest_dropin_plate_metadata,
)
from data_pipeline.shared.identifiers import build_well_id


def test_dropin_plate_metadata_preserves_extension_columns(tmp_path):
    experiment = "20260723_seahub_GENE1_shard001"
    source = tmp_path / "plate.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "well_index": "A01",
                "well_id": build_well_id(experiment, "A01"),
                "genotype": "ctrl",
                "start_age_hpf": 24.0,
                "temperature": 28.5,
                "medium": "E3",
                "source_embryo_id": "seahub_fov_p01",
                "image_kind": "single_z",
            }
        ]
    ).to_csv(source, index=False)
    output = tmp_path / "canonical.csv"
    flag = tmp_path / "canonical.csv.validated"
    landed = ingest_dropin_plate_metadata(
        input_csv=source,
        experiment_id=experiment,
        output_csv=output,
        output_flag=flag,
    )
    assert landed["source_embryo_id"].tolist() == ["seahub_fov_p01"]
    assert landed["image_kind"].tolist() == ["single_z"]
    assert output.exists()
    assert flag.exists()
