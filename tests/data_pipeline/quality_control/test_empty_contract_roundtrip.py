"""Empty QC shards remain valid after CSV discards their in-memory dtypes."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.quality_control.death_detection.contract import (
    DEATH_DETECTION_QC_TABLE_COLUMNS,
    validate_death_detection_qc,
)
from data_pipeline.quality_control.focus_qc.contract import (
    FOCUS_QC_TABLE_COLUMNS,
    validate_focus_qc,
)
from data_pipeline.quality_control.mask_quality_qc.contract import (
    MASK_QUALITY_QC_TABLE_COLUMNS,
    validate_mask_quality_qc,
)
from data_pipeline.quality_control.motion_blur_qc.contract import (
    MOTION_BLUR_QC_TABLE_COLUMNS,
    validate_motion_blur_qc,
)
from data_pipeline.quality_control.snip_qc.contract import (
    SNIP_QC_TABLE_COLUMNS,
    validate_snip_qc,
)
from data_pipeline.quality_control.surface_area_qc.contract import (
    SURFACE_AREA_QC_TABLE_COLUMNS,
    validate_surface_area_qc,
)


EMPTY_QC_CONTRACTS = (
    ("mask_quality_qc", MASK_QUALITY_QC_TABLE_COLUMNS, validate_mask_quality_qc),
    ("focus_qc", FOCUS_QC_TABLE_COLUMNS, validate_focus_qc),
    ("motion_blur_qc", MOTION_BLUR_QC_TABLE_COLUMNS, validate_motion_blur_qc),
    ("death_detection_qc", DEATH_DETECTION_QC_TABLE_COLUMNS, validate_death_detection_qc),
    ("surface_area_qc", SURFACE_AREA_QC_TABLE_COLUMNS, validate_surface_area_qc),
    ("snip_qc", SNIP_QC_TABLE_COLUMNS, validate_snip_qc),
)


@pytest.mark.parametrize("name,columns,validator", EMPTY_QC_CONTRACTS)
def test_empty_qc_csv_roundtrip_is_valid(tmp_path, name, columns, validator):
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(columns=columns).to_csv(path, index=False)

    reloaded = pd.read_csv(path)
    assert reloaded.empty
    validator(reloaded)


@pytest.mark.parametrize("name,columns,validator", EMPTY_QC_CONTRACTS)
def test_empty_qc_still_requires_complete_schema(name, columns, validator):
    incomplete = pd.DataFrame(columns=columns[:-1])
    with pytest.raises(ValueError, match="missing required"):
        validator(incomplete)
