"""death_detection integration test (guardrail #5) — BOTH output grains, in one run.

Drives the real entrypoint over CSVs and asserts:
  - death_detection_qc: one row per snip_id, full snip spine, both bool flags;
  - death_event: one row per persistence-dead physical_embryo_id, physical-embryo spine ONLY
    (no embryo_id), with death_event_time_index + death_event_stage_hpf.
Catches the classic bug where one output is correct and the other drifts to the wrong grain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.feature_extraction.stage_inference import predict_stage_hpf
from data_pipeline.quality_control.death_detection.contract import (
    DEATH_DETECTION_QC_TABLE_COLUMNS,
    DEATH_EVENT_TABLE_COLUMNS,
    validate_death_detection_qc,
    validate_death_event,
)
from data_pipeline.quality_control.death_detection.entrypoint import run_death_detection
from data_pipeline.shared.identifiers import (
    build_embryo_id,
    build_image_id,
    build_physical_embryo_id,
    build_snip_id,
    build_well_id,
)

EXP = "20250912"
WELL = build_well_id(EXP, "B01")
CHANNEL = "BF"


def _build_inputs(tmp_path):
    """Two animals in one well: phys 1 clearly dies, phys 2 stays alive."""
    fa_rows, timing_rows, inv_rows = [], [], []
    traces = {1: [1.0, 1.0, 1.0, 0.1, 0.05, 0.05], 2: [1.0, 0.99, 0.98, 0.99, 0.98, 0.97]}
    times = list(range(6))
    # uniform 1h frames; timing is per (well, time_index)
    for t in times:
        timing_rows.append(
            {"experiment_id": EXP, "well_id": WELL, "time_index": t, "elapsed_time_s": t * 3600.0}
        )
    for phys_index, fractions in traces.items():
        phys = build_physical_embryo_id(WELL, phys_index)
        for t, frac in zip(times, fractions):
            image_id = build_image_id(WELL, CHANNEL, t)
            embryo_id = build_embryo_id(phys, image_id)
            snip_id = build_snip_id(embryo_id, image_id)
            common = {
                "experiment_id": EXP,
                "well_id": WELL,
                "physical_embryo_id": phys,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "image_id": image_id,
                "time_index": t,
                "channel_id": CHANNEL,
            }
            fa_rows.append({**common, "fraction_alive": frac})
            inv_rows.append({**common, "mask_id": image_id + f"_m{phys_index}"})

    paths = {}
    for name, rows in (
        ("fraction_alive", fa_rows),
        ("frame_inventory", timing_rows),
        ("snip_inventory", inv_rows),
    ):
        p = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths[name] = p

    plate_metadata = pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "start_age_hpf": 24.0,
                "temperature": 28.0,
            }
        ]
    )
    paths["plate_metadata"] = tmp_path / "plate_metadata.csv"
    plate_metadata.to_csv(paths["plate_metadata"], index=False)

    registry = pd.DataFrame(
        {"physical_embryo_id": [build_physical_embryo_id(WELL, i) for i in (1, 2)]}
    )
    paths["registry"] = tmp_path / "registry.csv"
    registry.to_csv(paths["registry"], index=False)
    return paths


def test_both_grains_emitted_correctly(tmp_path):
    paths = _build_inputs(tmp_path)
    out_qc = tmp_path / "death_detection_qc.csv"
    out_event = tmp_path / "death_event.csv"

    run_death_detection(
        fraction_alive_csv=paths["fraction_alive"],
        frame_inventory_csv=paths["frame_inventory"],
        plate_metadata_csv=paths["plate_metadata"],
        snip_inventory_csv=paths["snip_inventory"],
        physical_embryo_registry_csv=paths["registry"],
        output_qc_csv=out_qc,
        output_death_event_csv=out_event,
        config_overrides={"lead_time_hr": 0.0},
    )

    # ── snip-grain QC table ──
    qc = pd.read_csv(out_qc)
    assert list(qc.columns) == DEATH_DETECTION_QC_TABLE_COLUMNS
    assert qc["snip_id"].is_unique and len(qc) == 12  # 2 animals x 6 frames
    assert qc["viability_dead_flag"].dtype == bool and qc["persistence_dead_flag"].dtype == bool
    # the validator (registry verifier) must pass
    validate_death_detection_qc(qc, physical_embryo_registry_df=pd.read_csv(paths["registry"]), check_sources=True)
    # only the dying animal carries persistence-true snips
    dead_phys = build_physical_embryo_id(WELL, 1)
    assert qc[qc["physical_embryo_id"] == dead_phys]["persistence_dead_flag"].any()
    alive_phys = build_physical_embryo_id(WELL, 2)
    assert not qc[qc["physical_embryo_id"] == alive_phys]["persistence_dead_flag"].any()

    # ── physical-embryo-grain death_event table ──
    event = pd.read_csv(out_event)
    assert list(event.columns) == DEATH_EVENT_TABLE_COLUMNS
    assert "embryo_id" not in event.columns          # animal-level: must NOT carry channel id
    assert len(event) == 1                            # only the persistence-dead animal
    assert event["physical_embryo_id"].iloc[0] == dead_phys
    assert event["physical_embryo_id"].is_unique
    validate_death_event(event, physical_embryo_registry_df=pd.read_csv(paths["registry"]), check_sources=True)
    # The decline candidate marks the frame BEFORE the steep drop (diff t=2->t=3), so the
    # inflection is t=2; with lead_time_hr=0 the called-death frame D == 2. Stage is calculated
    # directly from well metadata + frame timing, independently of the dying animal's snip rows.
    death_d = int(event["death_event_time_index"].iloc[0])
    assert death_d == 2
    assert event["death_event_stage_hpf"].iloc[0] == pytest.approx(
        predict_stage_hpf(24.0, death_d * 3600.0, 28.0)
    )
