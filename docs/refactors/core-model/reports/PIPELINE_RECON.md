# Pipeline output reconnaissance for core-model refactor

Generated 2026-08-18T05:36:45.198131+00:00 by `/home/nick/miniconda3/envs/morphseq-env/bin/python` with pandas 2.2.3. This was a read-only survey of `/media/nick/gs_cluster/projects/data/morphseq/pipeline/output`. Section 7 and its dependent conclusions were rerun 2026-08-19T03:31:41.073578+00:00 with PyArrow 25.0.1.

## Scope and blocking findings

- Explicit cohort: **148 deduplicated experiment IDs** from 3 manifests.
- Newest selected experiment by ID: `20260724_hotfish_36hpf_plate02`.
- Pipeline path authority: `src/data_pipeline/pipeline_orchestrator/orchestration/paths.py` (loaded directly to avoid package initializer side effects).
- QC Parquet is now readable: **105/105 present files** opened successfully. The remaining availability blocker is upstream: **28 inventory-bearing experiments (167,603 snips) have no QC artifact**, so those rows remain unresolved rather than being treated as QC failures.

<details><summary>Exact ordered experiment IDs</summary>

```text
20250612_24hpf_ctrl_atf6
20250612_24hpf_wfs1_ctcf
20250612_30hpf_ctrl_atf6
20250612_30hpf_wfs1_ctcf
20250612_36hpf_ctrl_atf6
20250612_36hpf_wfs1_ctcf
20240813_24hpf
20240813_30hpf
20240813_36hpf
20240813_extras
20260702_hotchem_24hpf_plate01
20260702_hotchem_24hpf_plate02
20260702_hotchem_30hpf_plate01
20260702_hotchem_30hpf_plate02
20260702_hotchem_36hpf_plate01
20260702_hotchem_36hpf_plate02
20260320_cilia_crispant_48hpf
20260324_cep290_18hpf_24hpf_plate02
20260324_cep290_18hpf_plate01
20260324_cep290_24hpf_plate01
20260324_cep290_24hpf_plate02
20260324_cep290_30hpf_plate01
20260324_cep290_30hpf_plate02
20260331_b9d2_18hpf_plate01
20260331_b9d2_18hpf_plate02
20260414_b9d2_14hpf_plate01
20260414_b9d2_14hpf_plate02
20260414_b9d2_30hpf_plate01
20260414_b9d2_30hpf_plate02
20260415_b9d2_30to48hpf_plate01_t02
20260415_b9d2_30to48hpf_plate02_t02
20260415_cep290_18hpf_plate03
20260415_cep290_30to48hpf_plate02_t01
20230525
20230531
20230602
20230613
20230615
20230620
20230622
20230627
20230629
20230831
20240509_18ss
20240509_24hpf
20240510
20240717
20240718
20240724
20240725
20240726
20250622_chem_28C_T00_1425
20250622_chem_28C_T01_1658
20250622_chem_34C_T00_1256
20250622_chem_34C_T01_1632
20250622_chem_35C_T00_1223_check
20250622_chem_35C_T01_1605
20250623_chem_28C_T02_1259
20250623_chem_34C_T02_1231
20250623_chem_35C_T02_1204
20250624_chem02_28C_T00_1356
20250624_chem02_28C_T01_1808
20250624_chem02_34C_T00_1243
20250624_chem02_34C_T01_1739
20250624_chem02_35C_T00_1216
20250624_chem02_35C_T01_1711
20250625_chem02_28C_T02_1332
20250625_chem02_34C_T02_1301
20250625_chem02_35C_T02_1228
20250703_chem3_28C_T00_1325
20250703_chem3_34C_T00_1131
20250703_chem3_34C_T01_1457
20250703_chem3_35C_T00_1101
20250703_chem3_35C_T01_1437
20250716_chem4_28C_T00_1158
20250716_chem4_28C_T01_1400
20250716_chem4_34C_T00_1014
20250716_chem4_35C_T00_1045
20250721_chem5_28C_T00_1257
20250721_chem5_28C_T01_1401
20250721_chem5_35C_T00_1023
20260319_cilia_crispant_18hpf
20260319_cilia_crispant_24hpf
20260319_cilia_crispant_30hpf
20260416_cep290_30to48hpf_plate01_t02
20260416_cep290_30to48hpf_plate02_t02
20260320
20231110
20231206
20231218
20240306
20240307
20240404
20240411
20240418
20240509
20240522
20240530
20240626
20240812
20241022
20241023
20250126
20250215
20250305
20250415
20250416
20250425
20250501
20250512
20250515_part2
20250519
20250711
20250912
20251017_part1
20251017_part2
20251020
20251104
20251106
20251112
20251113
20251119
20251121
20251125
20251205
20251207_pbx
20251212
20260122
20260202
20260206
20260208
20260210
20260213
20260219
20260223
20260224
20260228
20260304
20260306
20260319
20260417_irx_pilot
20260418_irx_pilot
20260724_hotfish_24hpf_plate01
20260724_hotfish_24hpf_plate02
20260724_hotfish_30hpf_plate01
20260724_hotfish_30hpf_plate02
20260724_hotfish_36hpf_plate01
20260724_hotfish_36hpf_plate02
```
</details>

## 1. Availability and schema drift

| source | selected | present | readable | schemas |
| --- | --- | --- | --- | --- |
| inventory | 148 | 133 | 133 | 2 |
| plate | 148 | 144 | 144 | 38 |
| qc | 148 | 105 | 105 | 3 |
| stage | 148 | 109 | 109 | 2 |

Inventory schema flag: **133/133** readable inventories lack `source_micrometers_per_pixel`; **133/133** lack `snip_micrometers_per_pixel`. A further **15** selected experiments have no readable merged inventory.

Schema IDs expand to these exact ordered column lists:

- `I01` (inventory): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "image_id", "time_index", "channel_id", "mask_id", "track_id", "image_path", "processed_snip_path", "embryo_mask", "embryo_mask_snip_path", "crop_x_min_px", "crop_y_min_px", "crop_x_max_px", "crop_y_max_px", "crop_width_px", "crop_height_px", "is_valid_snip", "error_message"]`
- `I02` (inventory): `["snip_id", "embryo_id", "physical_embryo_id", "experiment_id", "well_id", "image_id", "time_index", "channel_id", "mask_id", "track_id", "image_path", "processed_snip_path", "embryo_mask", "embryo_mask_snip_path", "crop_x_min_px", "crop_y_min_px", "crop_x_max_px", "crop_y_max_px", "crop_width_px", "crop_height_px", "is_valid_snip", "error_message"]`
- `S01` (stage): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "image_id", "time_index", "channel_id", "predicted_stage_hpf", "model_version", "stage_prediction_status"]`
- `S02` (stage): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "image_id", "time_index", "channel_id", "predicted_stage_hpf", "model_version"]`
- `Q01` (QC; 101 experiments): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "persistence_dead_flag", "viability_dead_flag", "focus_flag", "focus_qc_applicability", "discontinuous_mask_flag", "edge_flag", "overlapping_mask_flag", "motion_blur_flag", "motion_blur_qc_applicability", "sa_outlier_flag", "surface_area_qc_applicability", "use_snip", "qc_fail_reasons"]`
- `Q02` (QC; 2 experiments): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "use_snip", "qc_fail_reasons"]`
- `Q03` (QC; 2 experiments): `["experiment_id", "well_id", "physical_embryo_id", "embryo_id", "snip_id", "persistence_dead_flag", "viability_dead_flag", "death_detection_qc_applicability", "focus_flag", "focus_qc_applicability", "discontinuous_mask_flag", "edge_flag", "overlapping_mask_flag", "motion_blur_flag", "motion_blur_qc_applicability", "sa_outlier_flag", "surface_area_qc_applicability", "use_snip", "qc_fail_reasons"]`
- `P01` (plate): `["well_index", "medium", "image_to_hash_map", "hash_plate_num", "mold_type", "genotype", "strain", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature", "morph_seq_qc", "experiment_id", "well_id"]`
- `P02` (plate): `["well_index", "medium", "mold_type", "genotype", "strain", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature", "morph_seq_qc", "experiment_id", "well_id"]`
- `P03` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature", "image_to_hash_plate_num", "hash_to_image_map", "qc", "morph_seq_qc", "experiment_id", "well_id"]`
- `P04` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature", "qc", "experiment_id", "well_id"]`
- `P05` (plate): `["well_index", "genotype", "chem_perturbation", "start_age_hpf", "temperature", "sequenced", "qc", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P06` (plate): `["well_index", "pair", "genotype", "sequenced", "strain", "chem_perturbation", "start_age_hpf", "qc", "temperature", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P07` (plate): `["well_index", "genotype", "sequenced", "pair", "strain", "chem_perturbation", "start_age_hpf", "qc", "temperature", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P08` (plate): `["well_index", "genotype", "sequenced", "strain", "chem_perturbation", "start_age_hpf", "qc", "temperature", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P09` (plate): `["well_index", "genotype", "sequenced", "strain", "pair", "chem_perturbation", "start_age_hpf", "qc", "temperature", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P10` (plate): `["well_index", "genotype", "sequenced", "strain", "chem_perturbation", "pair", "start_age_hpf", "qc", "temperature", "image_to_hash_map", "hash_plate_num", "medium", "experiment_id", "well_id"]`
- `P11` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_morph", "start_age_hpf", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P12` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "start_age_morph", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P13` (plate): `["well_index", "mold_type", "medium", "genotype", "chem_perturbation", "start_age_hpf", "start_age_morph", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P14` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well", "image_to_hash_map", "image_to_hash_plate_num", "hash_to_image_map", "qc", "image_notes", "morph_seq_qc", "temperature", "experiment_id", "well_id"]`
- `P15` (plate): `["well_index", "temperature", "medium", "experiment_id", "well_id", "genotype", "start_age_hpf"]`
- `P16` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well", "image_to_hash_map", "image_to_hash_plate_num", "hash_to_image_map", "qc", "morph_seq_qc", "temperature", "experiment_id", "well_id"]`
- `P17` (plate): `["well_index", "medium", "mold_type", "genotype", "strain", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P18` (plate): `["well_index", "medium", "mold_type", "genotype", "strain", "chem_perturbation", "embryos_per_well", "experiment_id", "well_id", "start_age_hpf", "temperature"]`
- `P19` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "start_age_morph", "embryos_per_well", "temperature", "tricane", "pair", "experiment_id", "well_id"]`
- `P20` (plate): `["well_index", "genotype", "start_age_hpf", "medium", "chem_perturbation", "mold_type", "embryos_per_well", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P21` (plate): `["well_index", "genotype", "start_age_hpf", "mold_type", "chem_perturbation", "embryos_per_well", "medium", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P22` (plate): `["well_index", "genotype_map_orig", "genotype", "start_age_hpf", "chem_perturbation", "embryos_per_well", "mold_type", "medium", "qc", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P23` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P24` (plate): `["well_index", "genotype", "start_age_hpf", "medium", "embryos_per_well", "chem_perturbation", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P25` (plate): `["well_index", "genotype", "chem_perturbation", "start_age_hpf", "medium", "embryos_per_well", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P26` (plate): `["well_index", "medium", "mold_type", "genotype", "series_number_map", "chem_perturbation", "start_age_hpf", "embryos_per_well", "qc", "image_notes", "morph_seq_qc", "temperature", "experiment_id", "well_id"]`
- `P27` (plate): `["well_index", "orig_genotype", "medium", "mold_type", "genotype", "series_number_map", "chem_perturbation", "start_age_hpf", "embryos_per_well", "qc", "image_notes", "temperature", "experiment_id", "well_id"]`
- `P28` (plate): `["well_index", "medium", "mold_type", "genotype", "series_number_map", "chem_perturbation", "start_age_hpf", "embryos_per_well", "qc", "image_notes", "temperature", "experiment_id", "well_id"]`
- `P29` (plate): `["well_index", "medium", "mold_type", "genotype", "series_number_map", "chem_perturbation", "start_age_hpf", "embryos_per_well", "image_notes", "temperature", "experiment_id", "well_id"]`
- `P30` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "start_age_morph", "embryos_per_well", "experiment_id", "well_id", "temperature"]`
- `P31` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "start_age_morph", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P32` (plate): `["well_index", "medium", "mold_type", "genotype", "start_age_hpf", "series_number_map", "start_age_morph", "embryos_per_well", "chem_perturbation", "temperature", "experiment_id", "well_id"]`
- `P33` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "pair", "series_number_map", "start_age_morph", "embryos_per_well", "temperature", "experiment_id", "well_id"]`
- `P34` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "start_age_morph", "embryos_per_well", "temperature", "tricane", "experiment_id", "well_id"]`
- `P35` (plate): `["well_index", "medium", "genotype", "chem_perturbation", "start_age_hpf", "series_number_map", "temperature", "experiment_id", "well_id"]`
- `P36` (plate): `["well_index", "medium", "mold_type", "genotype", "chem_perturbation", "series_number_map", "start_age_hpf", "start_age_morph", "embryos_per_well", "temperature", "tricane", "pair", "experiment_id", "well_id"]`
- `P37` (plate): `["well_index", "genotype", "temperature", "strain", "chem_perturbation", "start_age_hpf", "medium", "experiment_id", "well_id"]`
- `P38` (plate): `["well_index", "genotype", "chem_perturbation", "start_stage_hpf", "temperature", "sequenced", "qc", "image_notes", "image_to_hash_map", "hash_plate_num", "experiment_id", "well_id", "start_age_hpf", "medium"]`

<details><summary>Per experiment × source availability, rows, schema, and drift</summary>

| experiment_id | source | status | rows | schema | reference_experiment | + vs ref | − vs ref | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20250612_24hpf_ctrl_atf6 | inventory | readable | 97 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_24hpf_ctrl_atf6 | stage | readable | 97 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_24hpf_ctrl_atf6 | qc | readable | 97 | Q03 | 20260724_hotfish_36hpf_plate01 | ["death_detection_qc_applicability"] | [] |  |
| 20250612_24hpf_ctrl_atf6 | plate | readable | 96 | P01 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["image_notes", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250612_24hpf_wfs1_ctcf | inventory | readable | 96 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_24hpf_wfs1_ctcf | stage | readable | 96 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_24hpf_wfs1_ctcf | qc | readable | 96 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_24hpf_wfs1_ctcf | plate | readable | 96 | P02 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250612_30hpf_ctrl_atf6 | inventory | readable | 97 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_30hpf_ctrl_atf6 | stage | readable | 97 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_30hpf_ctrl_atf6 | qc | readable | 97 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_30hpf_ctrl_atf6 | plate | readable | 96 | P02 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250612_30hpf_wfs1_ctcf | inventory | readable | 97 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_30hpf_wfs1_ctcf | stage | readable | 97 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_30hpf_wfs1_ctcf | qc | readable | 97 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_30hpf_wfs1_ctcf | plate | readable | 96 | P02 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250612_36hpf_ctrl_atf6 | inventory | readable | 102 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_36hpf_ctrl_atf6 | stage | readable | 102 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_36hpf_ctrl_atf6 | qc | readable | 102 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_36hpf_ctrl_atf6 | plate | readable | 96 | P02 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250612_36hpf_wfs1_ctcf | inventory | readable | 97 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250612_36hpf_wfs1_ctcf | stage | readable | 97 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_36hpf_wfs1_ctcf | qc | readable | 97 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250612_36hpf_wfs1_ctcf | plate | readable | 96 | P02 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240813_24hpf | inventory | readable | 48 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240813_24hpf | stage | readable | 48 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_24hpf | qc | readable | 48 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_24hpf | plate | readable | 96 | P03 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240813_30hpf | inventory | readable | 50 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240813_30hpf | stage | readable | 50 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_30hpf | qc | readable | 50 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_30hpf | plate | readable | 96 | P03 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240813_36hpf | inventory | readable | 50 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240813_36hpf | stage | readable | 50 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_36hpf | qc | readable | 50 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_36hpf | plate | readable | 96 | P03 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240813_extras | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240813_extras | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_extras | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240813_extras | plate | readable | 96 | P04 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20260702_hotchem_24hpf_plate01 | inventory | readable | 104 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_24hpf_plate01 | stage | readable | 104 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_24hpf_plate01 | qc | readable | 104 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_24hpf_plate01 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260702_hotchem_24hpf_plate02 | inventory | readable | 117 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_24hpf_plate02 | stage | readable | 117 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_24hpf_plate02 | qc | readable | 117 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_24hpf_plate02 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260702_hotchem_30hpf_plate01 | inventory | readable | 124 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_30hpf_plate01 | stage | readable | 124 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_30hpf_plate01 | qc | readable | 124 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_30hpf_plate01 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260702_hotchem_30hpf_plate02 | inventory | readable | 107 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_30hpf_plate02 | stage | readable | 107 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_30hpf_plate02 | qc | readable | 107 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_30hpf_plate02 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260702_hotchem_36hpf_plate01 | inventory | readable | 101 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_36hpf_plate01 | stage | readable | 101 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_36hpf_plate01 | qc | readable | 101 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_36hpf_plate01 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260702_hotchem_36hpf_plate02 | inventory | readable | 99 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260702_hotchem_36hpf_plate02 | stage | readable | 99 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_36hpf_plate02 | qc | readable | 99 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260702_hotchem_36hpf_plate02 | plate | readable | 96 | P05 | 20260724_hotfish_36hpf_plate01 | [] | ["image_notes", "start_stage_hpf"] |  |
| 20260320_cilia_crispant_48hpf | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260320_cilia_crispant_48hpf | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260320_cilia_crispant_48hpf | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260320_cilia_crispant_48hpf | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_18hpf_24hpf_plate02 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_18hpf_24hpf_plate02 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_18hpf_24hpf_plate02 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_18hpf_24hpf_plate02 | plate | readable | 96 | P07 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_18hpf_plate01 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_18hpf_plate01 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_18hpf_plate01 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_18hpf_plate01 | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_24hpf_plate01 | inventory | readable | 101 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_24hpf_plate01 | stage | readable | 101 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_24hpf_plate01 | qc | readable | 101 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_24hpf_plate01 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_24hpf_plate02 | inventory | readable | 101 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_24hpf_plate02 | stage | readable | 101 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_24hpf_plate02 | qc | readable | 101 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_24hpf_plate02 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_30hpf_plate01 | inventory | readable | 98 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_30hpf_plate01 | stage | readable | 98 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_30hpf_plate01 | qc | readable | 98 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_30hpf_plate01 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260324_cep290_30hpf_plate02 | inventory | readable | 72 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260324_cep290_30hpf_plate02 | stage | readable | 72 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_30hpf_plate02 | qc | readable | 72 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260324_cep290_30hpf_plate02 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260331_b9d2_18hpf_plate01 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260331_b9d2_18hpf_plate01 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260331_b9d2_18hpf_plate01 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260331_b9d2_18hpf_plate01 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260331_b9d2_18hpf_plate02 | inventory | readable | 70 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260331_b9d2_18hpf_plate02 | stage | readable | 70 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260331_b9d2_18hpf_plate02 | qc | readable | 70 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260331_b9d2_18hpf_plate02 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260414_b9d2_14hpf_plate01 | inventory | readable | 107 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260414_b9d2_14hpf_plate01 | stage | readable | 107 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_14hpf_plate01 | qc | readable | 107 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_14hpf_plate01 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260414_b9d2_14hpf_plate02 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260414_b9d2_14hpf_plate02 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_14hpf_plate02 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_14hpf_plate02 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260414_b9d2_30hpf_plate01 | inventory | readable | 106 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260414_b9d2_30hpf_plate01 | stage | readable | 106 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_30hpf_plate01 | qc | readable | 106 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_30hpf_plate01 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260414_b9d2_30hpf_plate02 | inventory | readable | 72 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260414_b9d2_30hpf_plate02 | stage | readable | 72 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_30hpf_plate02 | qc | readable | 72 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260414_b9d2_30hpf_plate02 | plate | readable | 96 | P09 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260415_b9d2_30to48hpf_plate01_t02 | inventory | readable | 109 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate01_t02 | stage | readable | 109 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate01_t02 | qc | readable | 109 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate01_t02 | plate | readable | 96 | P10 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260415_b9d2_30to48hpf_plate02_t02 | inventory | readable | 58 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate02_t02 | stage | readable | 58 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate02_t02 | qc | readable | 58 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_b9d2_30to48hpf_plate02_t02 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260415_cep290_18hpf_plate03 | inventory | readable | 81 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260415_cep290_18hpf_plate03 | stage | readable | 81 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_cep290_18hpf_plate03 | qc | readable | 81 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_cep290_18hpf_plate03 | plate | readable | 96 | P08 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260415_cep290_30to48hpf_plate02_t01 | inventory | readable | 93 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260415_cep290_30to48hpf_plate02_t01 | stage | readable | 93 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_cep290_30to48hpf_plate02_t01 | qc | readable | 93 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260415_cep290_30to48hpf_plate02_t01 | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20230525 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230525 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230525 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230525 | plate | readable | 96 | P11 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230531 | inventory | readable | 7,846 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230531 | stage | readable | 7,846 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230531 | qc | readable | 7,846 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230531 | plate | readable | 96 | P11 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230602 | inventory | readable | 8,998 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230602 | stage | readable | 8,998 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230602 | qc | readable | 8,998 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230602 | plate | readable | 96 | P11 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230613 | inventory | readable | 5,421 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230613 | stage | readable | 5,421 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230613 | qc | readable | 5,421 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230613 | plate | readable | 96 | P12 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230615 | inventory | readable | 4,216 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230615 | stage | readable | 4,216 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230615 | qc | readable | 4,216 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230615 | plate | readable | 96 | P12 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230620 | inventory | readable | 5,460 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230620 | stage | readable | 5,460 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230620 | qc | readable | 5,460 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230620 | plate | readable | 96 | P12 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230622 | inventory | readable | 3,582 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230622 | stage | readable | 3,582 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230622 | qc | readable | 3,582 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230622 | plate | readable | 96 | P12 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230627 | inventory | readable | 4,193 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230627 | stage | readable | 4,193 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230627 | qc | readable | 4,193 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230627 | plate | readable | 96 | P13 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230629 | inventory | readable | 3,918 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230629 | stage | readable | 3,918 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230629 | qc | readable | 3,918 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230629 | plate | readable | 96 | P12 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20230831 | inventory | readable | 65 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20230831 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230831 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20230831 | plate | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509_18ss | inventory | readable | 25 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240509_18ss | stage | readable | 25 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509_18ss | qc | readable | 25 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509_18ss | plate | readable | 96 | P14 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "sequenced", "start_stage_hpf"] |  |
| 20240509_24hpf | inventory | readable | 32 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240509_24hpf | stage | readable | 32 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509_24hpf | qc | readable | 32 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509_24hpf | plate | readable | 96 | P14 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "sequenced", "start_stage_hpf"] |  |
| 20240510 | inventory | readable | 59 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240510 | stage | readable | 59 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240510 | qc | readable | 59 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240510 | plate | readable | 96 | P15 | 20260724_hotfish_36hpf_plate01 | [] | ["chem_perturbation", "hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240717 | inventory | readable | 68 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240717 | stage | readable | 68 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240717 | qc | readable | 68 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240717 | plate | readable | 96 | P16 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "sequenced", "start_stage_hpf"] |  |
| 20240718 | inventory | readable | 99 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240718 | stage | readable | 99 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240718 | qc | readable | 99 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240718 | plate | readable | 96 | P16 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "sequenced", "start_stage_hpf"] |  |
| 20240724 | inventory | readable | 56 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240724 | stage | readable | 56 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240724 | qc | readable | 56 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240724 | plate | readable | 96 | P16 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "sequenced", "start_stage_hpf"] |  |
| 20240725 | inventory | readable | 64 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240725 | stage | readable | 64 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240725 | qc | readable | 64 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240725 | plate | readable | 96 | P16 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "sequenced", "start_stage_hpf"] |  |
| 20240726 | inventory | readable | 60 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240726 | stage | readable | 60 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240726 | qc | readable | 60 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240726 | plate | readable | 96 | P16 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "hash_to_image_map", "image_to_hash_plate_num", "mold_type", "morph_seq_qc"] | ["hash_plate_num", "image_notes", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_28C_T00_1425 | inventory | readable | 91 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_28C_T00_1425 | stage | readable | 91 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_28C_T00_1425 | qc | readable | 91 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_28C_T00_1425 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_28C_T01_1658 | inventory | readable | 86 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_28C_T01_1658 | stage | readable | 86 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_28C_T01_1658 | qc | readable | 86 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_28C_T01_1658 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_34C_T00_1256 | inventory | readable | 78 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_34C_T00_1256 | stage | readable | 78 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_34C_T00_1256 | qc | readable | 78 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_34C_T00_1256 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_34C_T01_1632 | inventory | readable | 79 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_34C_T01_1632 | stage | readable | 79 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_34C_T01_1632 | qc | readable | 79 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_34C_T01_1632 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_35C_T00_1223_check | inventory | readable | 79 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_35C_T00_1223_check | stage | readable | 79 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_35C_T00_1223_check | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_35C_T00_1223_check | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250622_chem_35C_T01_1605 | inventory | readable | 85 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250622_chem_35C_T01_1605 | stage | readable | 85 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_35C_T01_1605 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250622_chem_35C_T01_1605 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250623_chem_28C_T02_1259 | inventory | readable | 86 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250623_chem_28C_T02_1259 | stage | readable | 86 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_28C_T02_1259 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_28C_T02_1259 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250623_chem_34C_T02_1231 | inventory | readable | 83 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250623_chem_34C_T02_1231 | stage | readable | 83 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_34C_T02_1231 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_34C_T02_1231 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250623_chem_35C_T02_1204 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250623_chem_35C_T02_1204 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_35C_T02_1204 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250623_chem_35C_T02_1204 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_28C_T00_1356 | inventory | readable | 104 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_28C_T00_1356 | stage | readable | 104 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_28C_T00_1356 | qc | readable | 104 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_28C_T00_1356 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_28C_T01_1808 | inventory | readable | 97 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_28C_T01_1808 | stage | readable | 97 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_28C_T01_1808 | qc | readable | 97 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_28C_T01_1808 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_34C_T00_1243 | inventory | readable | 78 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_34C_T00_1243 | stage | readable | 78 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_34C_T00_1243 | qc | readable | 78 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_34C_T00_1243 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_34C_T01_1739 | inventory | readable | 75 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_34C_T01_1739 | stage | readable | 75 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_34C_T01_1739 | qc | readable | 75 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_34C_T01_1739 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_35C_T00_1216 | inventory | readable | 81 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_35C_T00_1216 | stage | readable | 81 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_35C_T00_1216 | qc | readable | 81 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_35C_T00_1216 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250624_chem02_35C_T01_1711 | inventory | readable | 84 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250624_chem02_35C_T01_1711 | stage | readable | 84 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_35C_T01_1711 | qc | readable | 84 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250624_chem02_35C_T01_1711 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250625_chem02_28C_T02_1332 | inventory | readable | 96 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250625_chem02_28C_T02_1332 | stage | readable | 96 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_28C_T02_1332 | qc | readable | 96 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_28C_T02_1332 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250625_chem02_34C_T02_1301 | inventory | readable | 82 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250625_chem02_34C_T02_1301 | stage | readable | 82 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_34C_T02_1301 | qc | readable | 82 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_34C_T02_1301 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250625_chem02_35C_T02_1228 | inventory | readable | 84 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250625_chem02_35C_T02_1228 | stage | readable | 84 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_35C_T02_1228 | qc | readable | 84 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250625_chem02_35C_T02_1228 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250703_chem3_28C_T00_1325 | inventory | readable | 61 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250703_chem3_28C_T00_1325 | stage | readable | 61 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_28C_T00_1325 | qc | readable | 61 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_28C_T00_1325 | plate | readable | 96 | P18 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250703_chem3_34C_T00_1131 | inventory | readable | 58 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250703_chem3_34C_T00_1131 | stage | readable | 58 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_34C_T00_1131 | qc | readable | 58 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_34C_T00_1131 | plate | readable | 96 | P18 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250703_chem3_34C_T01_1457 | inventory | readable | 59 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250703_chem3_34C_T01_1457 | stage | readable | 59 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_34C_T01_1457 | qc | readable | 59 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_34C_T01_1457 | plate | readable | 96 | P18 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250703_chem3_35C_T00_1101 | inventory | readable | 65 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250703_chem3_35C_T00_1101 | stage | readable | 65 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_35C_T00_1101 | qc | readable | 65 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_35C_T00_1101 | plate | readable | 96 | P18 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250703_chem3_35C_T01_1437 | inventory | readable | 58 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250703_chem3_35C_T01_1437 | stage | readable | 58 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_35C_T01_1437 | qc | readable | 58 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250703_chem3_35C_T01_1437 | plate | readable | 96 | P18 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250716_chem4_28C_T00_1158 | inventory | readable | 87 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250716_chem4_28C_T00_1158 | stage | readable | 87 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_28C_T00_1158 | qc | readable | 87 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_28C_T00_1158 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250716_chem4_28C_T01_1400 | inventory | readable | 94 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250716_chem4_28C_T01_1400 | stage | readable | 94 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_28C_T01_1400 | qc | readable | 94 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_28C_T01_1400 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250716_chem4_34C_T00_1014 | inventory | readable | 94 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250716_chem4_34C_T00_1014 | stage | readable | 94 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_34C_T00_1014 | qc | readable | 94 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_34C_T00_1014 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250716_chem4_35C_T00_1045 | inventory | readable | 91 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250716_chem4_35C_T00_1045 | stage | readable | 91 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_35C_T00_1045 | qc | readable | 91 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250716_chem4_35C_T00_1045 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250721_chem5_28C_T00_1257 | inventory | readable | 107 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250721_chem5_28C_T00_1257 | stage | readable | 107 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_28C_T00_1257 | qc | readable | 107 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_28C_T00_1257 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250721_chem5_28C_T01_1401 | inventory | readable | 107 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250721_chem5_28C_T01_1401 | stage | readable | 107 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_28C_T01_1401 | qc | readable | 107 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_28C_T01_1401 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250721_chem5_35C_T00_1023 | inventory | readable | 108 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250721_chem5_35C_T00_1023 | stage | readable | 108 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_35C_T00_1023 | qc | readable | 108 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250721_chem5_35C_T00_1023 | plate | readable | 96 | P17 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260319_cilia_crispant_18hpf | inventory | readable | 92 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260319_cilia_crispant_18hpf | stage | readable | 92 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_18hpf | qc | readable | 92 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_18hpf | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260319_cilia_crispant_24hpf | inventory | readable | 99 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260319_cilia_crispant_24hpf | stage | readable | 99 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_24hpf | qc | readable | 99 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_24hpf | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260319_cilia_crispant_30hpf | inventory | readable | 102 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260319_cilia_crispant_30hpf | stage | readable | 102 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_30hpf | qc | readable | 102 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319_cilia_crispant_30hpf | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260416_cep290_30to48hpf_plate01_t02 | inventory | readable | 109 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate01_t02 | stage | readable | 109 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate01_t02 | qc | readable | 109 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate01_t02 | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260416_cep290_30to48hpf_plate02_t02 | inventory | readable | 70 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate02_t02 | stage | readable | 70 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate02_t02 | qc | readable | 70 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260416_cep290_30to48hpf_plate02_t02 | plate | readable | 96 | P06 | 20260724_hotfish_36hpf_plate01 | ["pair", "strain"] | ["image_notes", "start_stage_hpf"] |  |
| 20260320 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260320 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260320 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260320 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20231110 | inventory | readable | 5,852 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20231110 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231110 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231110 | plate | readable | 96 | P20 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20231206 | inventory | readable | 7,587 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20231206 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231206 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231206 | plate | readable | 96 | P21 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20231218 | inventory | readable | 3,536 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20231218 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231218 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20231218 | plate | readable | 96 | P22 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "genotype_map_orig", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240306 | inventory | readable | 10,837 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240306 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240306 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240306 | plate | readable | 96 | P23 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240307 | inventory | readable | 4,972 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240307 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240307 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240307 | plate | readable | 96 | P23 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240404 | inventory | readable | 3,222 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240404 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240404 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240404 | plate | readable | 96 | P24 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240411 | inventory | readable | 4,464 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240411 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240411 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240411 | plate | readable | 96 | P24 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240418 | inventory | readable | 9,062 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240418 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240418 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240418 | plate | readable | 96 | P25 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20240509 | inventory | readable | 4,387 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240509 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240509 | plate | readable | 96 | P26 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240522 | inventory | readable | 9,284 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240522 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240522 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240522 | plate | readable | 96 | P27 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "orig_genotype", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240530 | inventory | readable | 10,692 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240530 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240530 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240530 | plate | readable | 96 | P28 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240626 | inventory | readable | 11,088 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240626 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240626 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240626 | plate | readable | 96 | P26 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "morph_seq_qc", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "sequenced", "start_stage_hpf"] |  |
| 20240812 | inventory | readable | 7,575 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20240812 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240812 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20240812 | plate | readable | 96 | P23 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20241022 | inventory | readable | 3,737 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20241022 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20241022 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20241022 | plate | readable | 96 | P29 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20241023 | inventory | readable | 4,687 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20241023 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20241023 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20241023 | plate | readable | 96 | P29 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map"] | ["hash_plate_num", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250126 | inventory | readable | 7,745 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250126 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250126 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250126 | plate | readable | 96 | P30 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250215 | inventory | readable | 7,644 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250215 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250215 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250215 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250305 | inventory | readable | 32,962 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250305 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250305 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250305 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250415 | inventory | readable | 4,267 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250415 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250415 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250415 | plate | readable | 96 | P32 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250416 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250416 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250416 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250416 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250425 | inventory | readable | 13,429 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250425 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250425 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250425 | plate | readable | 96 | P33 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250501 | inventory | readable | 24,718 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250501 | stage | readable | 24,718 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250501 | qc | readable | 24,718 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250501 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250512 | inventory | readable | 24,569 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250512 | stage | readable | 24,569 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250512 | qc | readable | 24,569 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250512 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250515_part2 | inventory | readable | 3,510 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250515_part2 | stage | readable | 3,510 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250515_part2 | qc | readable | 3,510 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250515_part2 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250519 | inventory | readable | 4,179 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250519 | stage | readable | 4,179 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250519 | qc | readable | 4,179 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250519 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250711 | inventory | readable | 19,723 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250711 | stage | readable | 19,723 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250711 | qc | readable | 19,723 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250711 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20250912 | inventory | readable | 15,572 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20250912 | stage | readable | 15,572 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250912 | qc | readable | 15,572 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20250912 | plate | readable | 96 | P31 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251017_part1 | inventory | readable | 6,794 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251017_part1 | stage | readable | 6,794 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251017_part1 | qc | readable | 6,794 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251017_part1 | plate | readable | 96 | P34 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251017_part2 | inventory | readable | 7,559 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251017_part2 | stage | readable | 7,559 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251017_part2 | qc | readable | 7,559 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251017_part2 | plate | readable | 96 | P34 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251020 | inventory | readable | 5,216 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251020 | stage | readable | 5,216 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251020 | qc | readable | 5,216 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251020 | plate | readable | 96 | P34 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251104 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251104 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251104 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251104 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251106 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251106 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251106 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251106 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251112 | inventory | readable | 6,991 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251112 | stage | readable | 6,991 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251112 | qc | readable | 6,991 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251112 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251113 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251113 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251113 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251113 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251119 | inventory | readable | 10,440 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251119 | stage | readable | 10,440 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251119 | qc | readable | 10,440 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251119 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251121 | inventory | readable | 20,857 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251121 | stage | readable | 20,857 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251121 | qc | readable | 20,857 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251121 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251125 | inventory | readable | 21,934 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251125 | stage | readable | 21,934 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251125 | qc | readable | 21,934 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251125 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251205 | inventory | readable | 12,291 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251205 | stage | readable | 12,291 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251205 | qc | readable | 12,291 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251205 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251207_pbx | inventory | readable | 14,760 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251207_pbx | stage | readable | 14,760 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251207_pbx | qc | readable | 14,760 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251207_pbx | plate | readable | 96 | P35 | 20260724_hotfish_36hpf_plate01 | ["series_number_map"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20251212 | inventory | readable | 11,437 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20251212 | stage | readable | 11,437 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251212 | qc | readable | 11,437 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20251212 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260122 | inventory | readable | 20,098 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260122 | stage | readable | 20,098 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260122 | qc | readable | 20,098 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260122 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260202 | inventory | readable | 16,943 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260202 | stage | readable | 16,943 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260202 | qc | readable | 16,943 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260202 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260206 | inventory | readable | 14,700 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260206 | stage | readable | 14,700 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260206 | qc | readable | 14,700 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260206 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260208 | inventory | readable | 31,680 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260208 | stage | readable | 31,680 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260208 | qc | readable | 31,680 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260208 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260210 | inventory | readable | 15,892 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260210 | stage | readable | 15,892 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260210 | qc | readable | 15,892 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260210 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260213 | inventory | readable | 29,010 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260213 | stage | readable | 29,010 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260213 | qc | readable | 29,010 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260213 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260219 | inventory | readable | 31,713 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260219 | stage | readable | 31,713 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260219 | qc | readable | 31,713 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260219 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260223 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260223 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260223 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260223 | plate | readable | 96 | P36 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260224 | inventory | absent | — | — | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260224 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260224 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260224 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260228 | inventory | readable | 17,012 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260228 | stage | readable | 17,012 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260228 | qc | readable | 17,012 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260228 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260304 | inventory | readable | 37,409 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260304 | stage | readable | 37,409 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260304 | qc | readable | 37,409 | Q03 | 20260724_hotfish_36hpf_plate01 | ["death_detection_qc_applicability"] | [] |  |
| 20260304 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260306 | inventory | readable | 26,979 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260306 | stage | readable | 26,979 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260306 | qc | readable | 26,979 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260306 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260319 | inventory | readable | 10,269 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260319 | stage | readable | 10,269 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319 | qc | readable | 10,269 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260319 | plate | readable | 96 | P19 | 20260724_hotfish_36hpf_plate01 | ["embryos_per_well", "mold_type", "pair", "series_number_map", "start_age_morph", "tricane"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260417_irx_pilot | inventory | readable | 4,968 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260417_irx_pilot | stage | readable | 4,968 | S02 | 20260724_hotfish_36hpf_plate01 | [] | ["stage_prediction_status"] |  |
| 20260417_irx_pilot | qc | readable | 4,968 | Q02 | 20260724_hotfish_36hpf_plate01 | [] | ["persistence_dead_flag", "viability_dead_flag", "focus_flag", "focus_qc_applicability", "discontinuous_mask_flag", "edge_flag", "overlapping_mask_flag", "motion_blur_flag", "motion_blur_qc_applicability", "sa_outlier_flag", "surface_area_qc_applicability"] |  |
| 20260417_irx_pilot | plate | readable | 96 | P37 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260418_irx_pilot | inventory | readable | 15,251 | I02 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260418_irx_pilot | stage | readable | 15,251 | S02 | 20260724_hotfish_36hpf_plate01 | [] | ["stage_prediction_status"] |  |
| 20260418_irx_pilot | qc | readable | 15,251 | Q02 | 20260724_hotfish_36hpf_plate01 | [] | ["persistence_dead_flag", "viability_dead_flag", "focus_flag", "focus_qc_applicability", "discontinuous_mask_flag", "edge_flag", "overlapping_mask_flag", "motion_blur_flag", "motion_blur_qc_applicability", "sa_outlier_flag", "surface_area_qc_applicability"] |  |
| 20260418_irx_pilot | plate | readable | 96 | P37 | 20260724_hotfish_36hpf_plate01 | ["strain"] | ["hash_plate_num", "image_notes", "image_to_hash_map", "qc", "sequenced", "start_stage_hpf"] |  |
| 20260724_hotfish_24hpf_plate01 | inventory | readable | 102 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_24hpf_plate01 | stage | readable | 102 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_24hpf_plate01 | qc | readable | 102 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_24hpf_plate01 | plate | readable | 96 | P38 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_24hpf_plate02 | inventory | readable | 56 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_24hpf_plate02 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_24hpf_plate02 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_24hpf_plate02 | plate | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate01 | inventory | readable | 98 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_30hpf_plate01 | stage | readable | 98 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate01 | qc | readable | 98 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate01 | plate | readable | 96 | P38 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate02 | inventory | readable | 59 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_30hpf_plate02 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate02 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_30hpf_plate02 | plate | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate01 | inventory | readable | 101 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_36hpf_plate01 | stage | readable | 101 | S01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate01 | qc | readable | 101 | Q01 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate01 | plate | readable | 96 | P38 | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate02 | inventory | readable | 61 | I01 | 20260724_hotfish_36hpf_plate02 | [] | [] |  |
| 20260724_hotfish_36hpf_plate02 | stage | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate02 | qc | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |
| 20260724_hotfish_36hpf_plate02 | plate | absent | — | — | 20260724_hotfish_36hpf_plate01 | [] | [] |  |

</details>

The same exact table, including canonical paths and full column JSON, is in [availability_schema.csv](recon_tables/availability_schema.csv).

## 2. Cohort size

Readable inventories contribute **699,505 snip rows**.

Counts by channel:

| channel_id | snips |
| --- | --- |
| BF | 699505 |

Counts by experiment are exact in [snips_by_experiment.csv](recon_tables/snips_by_experiment.csv). Counts for every physical embryo are in [physical_embryo_frame_counts.csv](recon_tables/physical_embryo_frame_counts.csv).

Unique non-null `physical_embryo_id`: **13,194**.

Frames per physical embryo:

| statistic | frames |
| --- | --- |
| count | 13194.0 |
| mean | 53.01690162194937 |
| std | 70.17993637311966 |
| min | 1.0 |
| 5% | 1.0 |
| 25% | 1.0 |
| 50% | 27.0 |
| 75% | 78.0 |
| 95% | 186.0 |
| max | 348.0 |

## 3. Identity integrity

- Duplicate `snip_id` rows: **0** across **0** IDs.
- Missing identity-spine columns: `[]`.
- Nulls by identity field: `{"channel_id": 0, "embryo_id": 0, "experiment_id": 0, "image_id": 0, "physical_embryo_id": 0, "snip_id": 0, "time_index": 0, "well_id": 0}`.
- `physical_embryo_id` grammar/parent disagreements: **0**.
- `embryo_id = physical_embryo_id + channel_id` disagreements: **0**.
- `snip_id = embryo_id + zero-padded time_index` disagreements: **0**.

## 4. Image product type

Product source: **inferred from image_path segments (inventory columns absent)**.

| image_product_type | snips |
| --- | --- |
| projection | 699505 |

| projection_method | snips |
| --- | --- |
| focus_stack | 699505 |

Only one non-null/inferred product type is present, so the focus-axis collision condition was not triggered in this cohort.
`z_position` and `z_index` are absent or entirely null in the snip inventories; a z-position distribution cannot be measured from the requested boundary.

## 5. Masks

Observed convention: processed crop `<snip_id>.png`; embryo mask `<snip_id>_embryo.png` in the same directory.
Exact convention match: **699,505/699,505**; colocated explicit paths: **699,505/699,505**; no locatable mask after explicit-path and suffix fallback checks: **0/699,505**.
Per-experiment exact coverage is in [mask_coverage_by_experiment.csv](recon_tables/mask_coverage_by_experiment.csv).

## 6. Paths and image format

| _path_kind | _image_exists | snips | fraction |
| --- | --- | --- | --- |
| relative_under_root | True | 699505 | 1.0 |

Opened **100/100** sampled files; **100/100** were 8-bit, non-interlaced grayscale at pipeline `(H, W) = (576, 256)`. Exact checks: [image_format_sample.csv](recon_tables/image_format_sample.csv).

## 7. QC

Rerun with PyArrow 25.0.1: all **105/105 present QC Parquets** were readable, yielding **531,902 QC rows** in three schema variants. There were no duplicate QC `snip_id` rows, null `use_snip` verdicts, or QC IDs absent from inventory. Each QC-bearing experiment had a complete one-to-one inventory/QC match. The other **28 inventory-bearing experiments** have no QC file, leaving **167,603/699,505 inventory snips (23.96%)** without a QC row. Missing QC remains unresolved data availability, not a negative verdict. Exact availability and schemas are in [qc_availability_by_experiment.csv](recon_tables/qc_availability_by_experiment.csv) and [qc_schema_catalog.csv](recon_tables/qc_schema_catalog.csv).

Across QC-evaluable rows, `use_snip` passed **185,204** and failed **346,698**, for a row-weighted pass rate of **34.82%**. The unweighted experiment-level pass rate had median **72.41%** (5th–95th percentile: **20.41%–95.49%**); the much lower row-weighted rate reflects large, low-pass experiments.

| `is_valid_snip` | `use_snip=True` | `use_snip=False` | no QC row |
| --- | ---: | ---: | ---: |
| True | 185,204 | 346,698 | 167,603 |
| False | 0 | 0 | 0 |
| missing | 0 | 0 | 0 |

Every inventory row has `is_valid_snip=True`. Consequently, on QC-evaluable rows the strict gate `is_valid_snip ∧ use_snip` is exactly the `use_snip` verdict: **185,204 pass (34.82%)** and **346,698 are removed (65.18%)**. It must not be evaluated by filling the 167,603 missing QC values with `False`.

| QC failure-reason token | rows |
| --- | ---: |
| `sa_outlier_flag` | 241,274 |
| `focus_flag` | 173,605 |
| `persistence_dead_flag` | 115,150 |
| `viability_dead_flag` | 110,782 |
| `motion_blur_flag` | 70,583 |
| `edge_flag` | 51,708 |
| `overlapping_mask_flag` | 45,759 |
| `discontinuous_mask_flag` | 8,129 |

Failure-reason counts are multi-label and therefore exceed the number of failed rows. Strict-gate removal is strongly nonuniform by experiment: the experiment-level removal-rate 5th percentile, median, and 95th percentile are **4.51%**, **27.59%**, and **79.59%**, respectively, a **75.08-percentage-point** central-90% spread. Experiment identity explains **14.38%** of the row-level binary strict-gate variance (η²). Under the declared 5-percentage-point spread screen, QC exclusion is explicitly **experiment-correlated**, not approximately uniform.

Exact results are in [qc_by_experiment.csv](recon_tables/qc_by_experiment.csv), [qc_fail_reason_counts.csv](recon_tables/qc_fail_reason_counts.csv), [valid_use_snip_crosstab.csv](recon_tables/valid_use_snip_crosstab.csv), and [qc_rerun_summary.csv](recon_tables/qc_rerun_summary.csv).

## 8. Staging

| status | count |
| --- | --- |
| predicted | 511118 |
| — | 20219 |
| missing_start_age_hpf | 898 |

Finite predicted-stage coverage against all inventory rows: **531,337/699,505 (75.96%)**.
Survivors of the precursor gate `valid ∧ status==predicted ∧ finite stage`: **511,118**. Adding `use_snip=True` yields **176,466 definite full-gate survivors** (**25.23%** of all inventory rows; **34.53%** of precursor survivors). Rows without QC are unresolved and are not counted as failures. Exact per-experiment survivor counts are in [combined_metric_gate_by_experiment.csv](recon_tables/combined_metric_gate_by_experiment.csv).
Per-experiment coverage: [staging_by_experiment.csv](recon_tables/staging_by_experiment.csv). Stage histogram: [stage_histogram.csv](recon_tables/stage_histogram.csv).

## 9. Metric-group candidates

`short_pert_name` exists anywhere in selected plate metadata: **False**.
Every plate column's exact null count/rate, cardinality, and full JSON value set is in [plate_column_profiles.csv](recon_tables/plate_column_profiles.csv).

| column | null_rate | cardinality | value preview |
| --- | --- | --- | --- |
| chem_perturbation | 72.40% | 53 | ["DMSO", "DMSO_6", "Fgf_025", "Fgf_050", "Fgf_075", "Fgf_100", "Fgf_150", "Shh_025", "Shh_050", "Shh_075", "Shh_100", "TGFB-i", "Wnt-i", "bmp_i_11", "bmp_i_12",… |
| embryos_per_well | 76.91% | 3 | [0.0, 1.0, 2.0] |
| experiment_id | 0.00% | 144 | [20230525, 20230531, 20230602, 20230613, 20230615, 20230620, 20230622, 20230627, 20230629, 20231110, 20231206, 20231218, 20240306, 20240307, 20240404, 20240411,… |
| genotype | 10.21% | 99 | ["H2B-mScarlet", "Uncertain", "ab", "ab-ctrl-inj", "ab-lmx1b", "ab/tbxta", "ab_inj_ctrl", "atf6", "b9d2_het", "b9d2_heterozygous", "b9d2_homo", "b9d2_homozygous… |
| genotype_map_orig | 99.31% | 3 | ["tbxta", "tbxta/wik", "wik"] |
| hash_plate_num | 84.75% | 3 | [1.0, 18.0, 2.0] |
| hash_to_image_map | 100.00% | 0 | [] |
| image_notes | 99.99% | 1 | ["rounded somites"] |
| image_to_hash_map | 93.45% | 96 | ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11"… |
| image_to_hash_plate_num | 96.18% | 6 | [1.0, 18.0, 2.0, 4.0, 5.0, 6.0] |
| medium | 2.08% | 9 | ["EM", "EM3", "MC05", "MC1", "MC10", "MC_01", "MC_010", "MC_015", "not recorded"] |
| mold_type | 63.89% | 5 | [0, 1, 3, 4, "none"] |
| morph_seq_qc | 100.00% | 0 | [] |
| orig_genotype | 99.32% | 4 | ["Uncertain", "tbx16", "wik", "wik/tbx16"] |
| pair | 85.76% | 24 | ["ab", "ab_spawn", "b9d2_P8_P6_F1s", "b9d2_pair_1", "b9d2_pair_2", "b9d2_pair_4", "b9d2_pair_5", "b9d2_pair_6", "b9d2_pair_7", "b9d2_pair_8", "cep290_P1", "cep2… |
| qc | 92.04% | 2 | [0.0, 1.0] |
| sequenced | 86.44% | 3 | [0.0, 1.0, 2.0] |
| series_number_map | 63.95% | 96 | [1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 2.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 3.0, 30.0, 31.0, 32.0, 33.0, … |
| start_age_hpf | 12.33% | 56 | [10.0, 10.75, 106.0, 11.0, 11.66, 12.0, 13.0, 14.0, 14.5, 15.0, 15.5, 16.0, 18.0, 21.0, 22.0, 24.0, 24.71, 25.0, 25.25, 25.33333333333334, 25.54333333333334, 26… |
| start_age_morph | 93.06% | 9 | ["10ss", "11ss", "12ss", "13ss", "5ss", "7ss", "8ss", "e_90", "t_bud"] |
| start_stage_hpf | 97.92% | 3 | [24.83333333, 30.91666667, 36.833333333333336] |
| strain | 63.66% | 3 | ["ab", "b9d2_het", "cep290_het"] |
| temperature | 5.56% | 15 | [19.0, 22.0, 24.0, 25.0, 28.0, 28.5, 29.0, 30.0, 32.0, 33.5, 34.0, 34.2, 34.4, 35.0, 35.1] |
| tricane | 98.61% | 1 | [50.0] |
| well_id | 0.00% | 13824 | ["20230525_A01", "20230525_A02", "20230525_A03", "20230525_A04", "20230525_A05", "20230525_A06", "20230525_A07", "20230525_A08", "20230525_A09", "20230525_A10",… |
| well_index | 0.00% | 96 | ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11"… |

Candidate-to-curated-class mapping (full exact sets are in [metric_candidate_mapping.csv](recon_tables/metric_candidate_mapping.csv)):

| candidate | cohort_cardinality | matched_count |
| --- | --- | --- |
| genotype | 98 | 0 |
| strain | 3 | 0 |
| chem_perturbation | 53 | 0 |
| genotype_map_orig | 3 | 0 |
| orig_genotype | 4 | 0 |
| genotype__strain | 25 | 1 |

## 10. Pixel scale and intensity

Neither requested µm/px column contains measurable values in readable inventories, so scale distributions and mixed-scale detection are blocked by schema, not treated as homogeneous scale.

Per-experiment side-by-side sampled intensity distributions are in [intensity_by_experiment.csv](recon_tables/intensity_by_experiment.csv); per-image measurements are in [intensity_samples.csv](recon_tables/intensity_samples.csv).

<details><summary>Per-experiment intensity medians (distribution quantiles are in CSV)</summary>

| experiment_id | sample_n | mean_median | std_median | min_median | max_median | saturated_255_fraction_median |
| --- | --- | --- | --- | --- | --- | --- |
| 20230531 | 200 | 28.574293348524307 | 39.53140842993818 | 1.0 | 254.0 | 0.0 |
| 20230602 | 200 | 23.882425944010414 | 39.037019136514594 | 0.0 | 254.0 | 0.0 |
| 20230613 | 200 | 25.857560899522568 | 39.01578719879123 | 0.0 | 254.0 | 0.0 |
| 20230615 | 200 | 24.21575927734375 | 35.837647451298416 | 0.0 | 254.0 | 0.0 |
| 20230620 | 200 | 26.129370795355904 | 41.18435882855741 | 0.0 | 254.0 | 0.0 |
| 20230622 | 200 | 24.56432088216146 | 39.7515188256208 | 0.0 | 254.0 | 0.0 |
| 20230627 | 200 | 18.678178575303818 | 39.22288153740885 | 0.0 | 254.0 | 0.0 |
| 20230629 | 200 | 20.85040961371528 | 37.980442239697666 | 0.0 | 254.0 | 0.0 |
| 20230831 | 65 | 28.27862548828125 | 50.38509827285809 | 0.0 | 255.0 | 4.7471788194444445e-05 |
| 20231110 | 200 | 23.85867648654514 | 36.506065185101605 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20231206 | 200 | 28.101521809895836 | 43.14229786664728 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20231218 | 200 | 28.86471896701389 | 39.74914516850694 | 2.0 | 255.0 | 6.781684027777777e-06 |
| 20240306 | 200 | 27.501634385850693 | 43.37326098058266 | 1.0 | 255.0 | 6.781684027777777e-06 |
| 20240307 | 200 | 30.440636528862846 | 46.27118175219299 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20240404 | 200 | 31.756622314453125 | 42.823185843664504 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20240411 | 200 | 30.23218790690104 | 42.30893231431 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20240418 | 200 | 34.19603135850694 | 42.99341369921527 | 11.0 | 254.0 | 0.0 |
| 20240509 | 200 | 34.93871392144097 | 44.2331597640266 | 12.0 | 255.0 | 6.781684027777777e-06 |
| 20240509_18ss | 25 | 22.95870632595486 | 35.200739169097005 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20240509_24hpf | 32 | 24.025156656901043 | 37.86188445920693 | 2.0 | 254.0 | 0.0 |
| 20240510 | 59 | 28.297322591145832 | 45.070671223065304 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20240522 | 200 | 33.09411960177951 | 42.30742447962898 | 12.0 | 255.0 | 6.781684027777777e-06 |
| 20240530 | 200 | 35.361111111111114 | 47.566019853632156 | 9.0 | 255.0 | 6.781684027777777e-06 |
| 20240626 | 200 | 39.71266004774306 | 50.841024982572335 | 10.0 | 255.0 | 1.6954210069444445e-05 |
| 20240717 | 68 | 23.152225070529514 | 38.20654269752815 | 4.0 | 255.0 | 6.781684027777777e-06 |
| 20240718 | 99 | 29.639600965711807 | 45.48213854832548 | 4.0 | 255.0 | 1.3563368055555555e-05 |
| 20240724 | 56 | 23.72850884331597 | 38.36675830281892 | 4.0 | 254.0 | 0.0 |
| 20240725 | 64 | 28.736891004774307 | 46.63938163983967 | 3.0 | 255.0 | 1.3563368055555555e-05 |
| 20240726 | 60 | 32.81823052300347 | 50.61241770296274 | 1.0 | 255.0 | 2.712673611111111e-05 |
| 20240812 | 200 | 24.806410047743057 | 37.94783926215009 | 0.0 | 254.0 | 0.0 |
| 20240813_24hpf | 48 | 25.123246934678818 | 40.50304480470682 | 1.5 | 255.0 | 6.781684027777777e-06 |
| 20240813_30hpf | 50 | 25.952880859375 | 42.43963594441714 | 4.0 | 255.0 | 6.781684027777777e-06 |
| 20240813_36hpf | 50 | 27.45164998372396 | 44.465032260874736 | 4.5 | 255.0 | 6.781684027777777e-06 |
| 20241022 | 200 | 31.27524142795139 | 35.287439568882846 | 14.0 | 255.0 | 6.781684027777777e-06 |
| 20241023 | 200 | 36.74708048502604 | 43.64076315034818 | 4.0 | 255.0 | 6.781684027777777e-06 |
| 20250126 | 200 | 28.913272433810764 | 40.17546308903468 | 9.0 | 255.0 | 1.3563368055555555e-05 |
| 20250215 | 200 | 23.91002400716146 | 34.443258596763414 | 5.0 | 254.0 | 0.0 |
| 20250305 | 200 | 33.173428005642364 | 46.31688853062899 | 2.0 | 255.0 | 1.3563368055555555e-05 |
| 20250415 | 200 | 25.24203152126736 | 34.085125852012546 | 0.0 | 254.0 | 0.0 |
| 20250425 | 200 | 28.213663736979168 | 36.05746288753953 | 13.0 | 255.0 | 6.781684027777777e-06 |
| 20250501 | 200 | 50.78669230143229 | 56.64608788161856 | 9.0 | 255.0 | 6.781684027777777e-06 |
| 20250512 | 200 | 27.08289252387153 | 38.57441292599557 | 6.0 | 255.0 | 1.3563368055555555e-05 |
| 20250515_part2 | 200 | 45.34105088975694 | 57.7502744204327 | 11.0 | 255.0 | 6.781684027777777e-06 |
| 20250519 | 200 | 30.11276584201389 | 34.830874783377084 | 6.0 | 254.0 | 0.0 |
| 20250612_24hpf_ctrl_atf6 | 97 | 32.432834201388886 | 36.69177849904294 | 2.0 | 252.0 | 0.0 |
| 20250612_24hpf_wfs1_ctcf | 96 | 27.310621473524307 | 35.3633435585769 | 5.0 | 254.0 | 0.0 |
| 20250612_30hpf_ctrl_atf6 | 97 | 26.453145345052082 | 39.78416461510388 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20250612_30hpf_wfs1_ctcf | 97 | 30.41070556640625 | 37.47434066372356 | 8.0 | 255.0 | 6.781684027777777e-06 |
| 20250612_36hpf_ctrl_atf6 | 102 | 28.357326931423614 | 42.21845561696389 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20250612_36hpf_wfs1_ctcf | 97 | 28.020460340711807 | 41.088955985087125 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20250622_chem_28C_T00_1425 | 91 | 25.386210123697918 | 35.622724690130525 | 7.0 | 254.0 | 0.0 |
| 20250622_chem_28C_T01_1658 | 86 | 23.55193413628472 | 37.50984570341723 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20250622_chem_34C_T00_1256 | 78 | 27.453630235460068 | 36.9356210395062 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20250622_chem_34C_T01_1632 | 79 | 30.905904134114582 | 37.65280163964047 | 12.0 | 255.0 | 6.781684027777777e-06 |
| 20250622_chem_35C_T00_1223_check | 79 | 27.643663194444443 | 36.66812991253162 | 7.0 | 255.0 | 6.781684027777777e-06 |
| 20250622_chem_35C_T01_1605 | 85 | 25.280680338541668 | 37.82817906606501 | 9.0 | 255.0 | 6.781684027777777e-06 |
| 20250623_chem_28C_T02_1259 | 86 | 28.539004855685764 | 45.82990224700332 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20250623_chem_34C_T02_1231 | 83 | 30.456380208333332 | 45.681406189042065 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20250624_chem02_28C_T00_1356 | 104 | 25.334865993923614 | 34.850356384813814 | 10.0 | 254.0 | 0.0 |
| 20250624_chem02_28C_T01_1808 | 97 | 27.995313856336807 | 36.25930976432085 | 8.0 | 255.0 | 6.781684027777777e-06 |
| 20250624_chem02_34C_T00_1243 | 78 | 21.98491414388021 | 36.968575356600276 | 1.5 | 255.0 | 6.781684027777777e-06 |
| 20250624_chem02_34C_T01_1739 | 75 | 29.325636121961807 | 38.1024269084191 | 9.0 | 255.0 | 6.781684027777777e-06 |
| 20250624_chem02_35C_T00_1216 | 81 | 26.845296223958332 | 36.516603995295085 | 8.0 | 255.0 | 6.781684027777777e-06 |
| 20250624_chem02_35C_T01_1711 | 84 | 25.248009575737846 | 38.447949054087715 | 7.0 | 255.0 | 6.781684027777777e-06 |
| 20250625_chem02_28C_T02_1332 | 96 | 28.76639133029514 | 44.260473104655716 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20250625_chem02_34C_T02_1301 | 82 | 32.99485609266493 | 43.97010389616518 | 10.5 | 255.0 | 6.781684027777777e-06 |
| 20250625_chem02_35C_T02_1228 | 84 | 22.819478352864586 | 40.664903781825245 | 1.0 | 255.0 | 6.781684027777777e-06 |
| 20250703_chem3_28C_T00_1325 | 61 | 23.27178955078125 | 36.52884440462716 | 4.0 | 255.0 | 6.781684027777777e-06 |
| 20250703_chem3_34C_T00_1131 | 58 | 27.576812744140625 | 36.658549755996404 | 8.0 | 255.0 | 6.781684027777777e-06 |
| 20250703_chem3_34C_T01_1457 | 59 | 30.46393500434028 | 38.18100153548197 | 11.0 | 255.0 | 6.781684027777777e-06 |
| 20250703_chem3_35C_T00_1101 | 65 | 28.55465359157986 | 36.33808762559233 | 11.0 | 255.0 | 6.781684027777777e-06 |
| 20250703_chem3_35C_T01_1437 | 58 | 23.54885525173611 | 39.43392732244834 | 5.0 | 255.0 | 6.781684027777777e-06 |
| 20250711 | 200 | 34.61719089084201 | 53.87106404668587 | 2.0 | 255.0 | 2.0345052083333332e-05 |
| 20250716_chem4_28C_T00_1158 | 87 | 26.22540961371528 | 36.77137341680481 | 4.0 | 255.0 | 6.781684027777777e-06 |
| 20250716_chem4_28C_T01_1400 | 94 | 27.004007975260414 | 37.74907929417107 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20250716_chem4_34C_T00_1014 | 94 | 29.54309760199653 | 37.89377741370549 | 7.0 | 255.0 | 6.781684027777777e-06 |
| 20250716_chem4_35C_T00_1045 | 91 | 28.593790690104168 | 37.84580480223344 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20250721_chem5_28C_T00_1257 | 107 | 26.323384602864582 | 36.7784307660942 | 2.0 | 255.0 | 6.781684027777777e-06 |
| 20250721_chem5_28C_T01_1401 | 107 | 26.587965223524307 | 37.136309275213634 | 3.0 | 255.0 | 6.781684027777777e-06 |
| 20250721_chem5_35C_T00_1023 | 108 | 30.06847466362847 | 36.75924648849707 | 5.5 | 255.0 | 6.781684027777777e-06 |
| 20250912 | 200 | 19.68880886501736 | 35.20868279998952 | 0.0 | 254.0 | 0.0 |
| 20251017_part1 | 200 | 37.60581461588542 | 50.92140333379708 | 1.0 | 255.0 | 3.0517578125e-05 |
| 20251017_part2 | 200 | 38.162143283420136 | 56.094840439135226 | 7.0 | 255.0 | 1.3563368055555555e-05 |
| 20251020 | 200 | 25.90765380859375 | 39.702375157441985 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20251112 | 200 | 32.076805962456596 | 44.42590847235272 | 7.0 | 255.0 | 6.781684027777777e-06 |
| 20251119 | 200 | 25.959672715928818 | 32.7106200925825 | 9.0 | 254.0 | 0.0 |
| 20251121 | 200 | 31.97048611111111 | 54.906823108793745 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20251125 | 200 | 27.859290228949654 | 43.41344209191085 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20251205 | 200 | 30.230733235677086 | 47.04049302919856 | 3.0 | 255.0 | 1.3563368055555555e-05 |
| 20251207_pbx | 200 | 32.993235270182296 | 54.17141911556177 | 0.0 | 255.0 | 1.3563368055555555e-05 |
| 20251212 | 200 | 27.252733018663193 | 44.85977898221741 | 4.0 | 255.0 | 1.3563368055555555e-05 |
| 20260122 | 200 | 27.822750515407986 | 40.26263651400211 | 0.0 | 254.0 | 0.0 |
| 20260202 | 200 | 18.69068060980903 | 37.5083015515151 | 0.0 | 254.0 | 0.0 |
| 20260206 | 200 | 22.497599283854168 | 44.46592776821316 | 1.0 | 255.0 | 1.3563368055555555e-05 |
| 20260208 | 200 | 23.691175672743057 | 27.900801235359978 | 6.0 | 254.0 | 0.0 |
| 20260210 | 200 | 27.81271023220486 | 52.64952919233929 | 1.0 | 255.0 | 4.0690104166666664e-05 |
| 20260213 | 200 | 21.20124308268229 | 39.72249122450263 | 0.0 | 254.0 | 0.0 |
| 20260219 | 200 | 19.76569620768229 | 38.600748283930656 | 0.0 | 254.0 | 0.0 |
| 20260228 | 200 | 34.382008870442704 | 48.20741629529586 | 6.0 | 255.0 | 6.781684027777777e-06 |
| 20260304 | 200 | 33.74122111002604 | 46.68951315417405 | 5.0 | 254.5 | 3.3908420138888887e-06 |
| 20260306 | 200 | 36.920386420355904 | 52.910959571746425 | 2.0 | 255.0 | 6.781684027777777e-06 |
| 20260319 | 200 | 30.04760064019097 | 39.85311359799363 | 3.0 | 254.0 | 0.0 |
| 20260319_cilia_crispant_18hpf | 92 | 22.98932223849826 | 36.14297676442102 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260319_cilia_crispant_24hpf | 99 | 30.612508138020832 | 40.46036037638605 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260319_cilia_crispant_30hpf | 102 | 28.847703721788193 | 41.8778506988676 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260324_cep290_24hpf_plate01 | 101 | 22.18386501736111 | 38.13453737903453 | 0.0 | 254.0 | 0.0 |
| 20260324_cep290_24hpf_plate02 | 101 | 27.129659016927082 | 40.31635791389427 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260324_cep290_30hpf_plate01 | 98 | 25.180908203125 | 39.78068018182544 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260324_cep290_30hpf_plate02 | 72 | 25.89739312065972 | 40.040924563019 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260331_b9d2_18hpf_plate02 | 70 | 26.18834771050347 | 34.47318910120718 | 1.0 | 254.0 | 0.0 |
| 20260414_b9d2_14hpf_plate01 | 107 | 24.455796983506943 | 33.19587989132344 | 1.0 | 255.0 | 6.781684027777777e-06 |
| 20260414_b9d2_30hpf_plate01 | 106 | 27.056959364149307 | 39.137487426898105 | 0.5 | 255.0 | 6.781684027777777e-06 |
| 20260414_b9d2_30hpf_plate02 | 72 | 26.489288330078125 | 38.87681542687855 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260415_b9d2_30to48hpf_plate01_t02 | 109 | 32.121622721354164 | 48.164495408545186 | 0.0 | 255.0 | 1.3563368055555555e-05 |
| 20260415_b9d2_30to48hpf_plate02_t02 | 58 | 32.06872219509549 | 48.19872155658364 | 0.0 | 255.0 | 1.0172526041666666e-05 |
| 20260415_cep290_18hpf_plate03 | 81 | 25.531392415364582 | 33.362516690745444 | 0.0 | 254.0 | 0.0 |
| 20260415_cep290_30to48hpf_plate02_t01 | 93 | 23.954847547743057 | 39.19150090846921 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260416_cep290_30to48hpf_plate01_t02 | 109 | 27.124891493055557 | 48.18356704082159 | 0.0 | 255.0 | 1.3563368055555555e-05 |
| 20260416_cep290_30to48hpf_plate02_t02 | 70 | 27.37503390842014 | 48.43564038505035 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260417_irx_pilot | 200 | 25.92840576171875 | 37.12760247708653 | 1.0 | 254.0 | 0.0 |
| 20260418_irx_pilot | 200 | 20.79594930013021 | 40.21059913121356 | 0.0 | 254.0 | 0.0 |
| 20260702_hotchem_24hpf_plate01 | 104 | 22.036434597439236 | 36.78635191010381 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260702_hotchem_24hpf_plate02 | 117 | 27.82876247829861 | 37.56203972951253 | 0.0 | 254.0 | 0.0 |
| 20260702_hotchem_30hpf_plate01 | 124 | 27.76397026909722 | 38.145735086941976 | 1.0 | 255.0 | 6.781684027777777e-06 |
| 20260702_hotchem_30hpf_plate02 | 107 | 28.14410400390625 | 39.29684917313201 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260702_hotchem_36hpf_plate01 | 101 | 24.29645453559028 | 43.33714246143003 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260702_hotchem_36hpf_plate02 | 99 | 25.004930284288193 | 42.3327582300225 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260724_hotfish_24hpf_plate01 | 102 | 22.325154622395836 | 40.77483465913885 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260724_hotfish_24hpf_plate02 | 56 | 45.25823635525174 | 42.42545873327107 | 0.0 | 254.0 | 0.0 |
| 20260724_hotfish_30hpf_plate01 | 98 | 23.614034016927086 | 42.2220568910902 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260724_hotfish_30hpf_plate02 | 59 | 59.03325059678819 | 44.67071208651118 | 0.0 | 254.0 | 0.0 |
| 20260724_hotfish_36hpf_plate01 | 101 | 24.444010416666668 | 44.66318013844595 | 0.0 | 255.0 | 6.781684027777777e-06 |
| 20260724_hotfish_36hpf_plate02 | 61 | 69.24222140842014 | 48.79995794084001 | 0.0 | 255.0 | 2.712673611111111e-05 |

</details>
Experiment η² (fraction of sampled statistic variance explained by experiment): `{"mean": 0.263657265793611, "std": 0.22196801250347398, "saturated_255_fraction": 0.04716591346202481}`. With η²≥0.10 treated as a material batch signal, intensity is **strongly experiment-correlated and learnable as a batch effect**.

## 11. I/O throughput

| measurement | mean_ms | median_ms | p95_ms |
| --- | --- | --- | --- |
| cold_read_decode | 57.38105421109746 | 32.195532228797674 | 69.96134493965657 |
| warm_read_decode | 5.948353524630268 | 4.5331125147640705 | 10.914673400111504 |
| warm_file_read | 2.728088409639895 | 2.7103458996862173 | 3.630404779687524 |
| decode_from_memory | 1.2586798255021374 | 1.1351711582392454 | 1.7522440990433121 |
| resize_576x256_to_288x128 | 0.3100008595113953 | 0.2987708430737257 | 0.35950632300227886 |

Warm throughput is **168.1 images/s/worker**. At 128 reads for a metric batch of 64, one worker spends approximately **0.761 s/batch** in warm read+decode. The separated medians indicate **per-file read latency/I/O** dominates. “Cold” is a first-pass measurement on a shared mount; the script cannot guarantee or flush the system page cache.

## 12. Group-split feasibility

| split | physical_embryos | snips | embryo_fraction | snip_fraction |
| --- | --- | --- | --- | --- |
| train | 10555 | 560256 | 80.00% | 80.09% |
| eval | 1319 | 70318 | 10.00% | 10.05% |
| test | 1320 | 68931 | 10.00% | 9.85% |

The split is deterministic and group-disjoint at `physical_embryo_id`; exact assignments are in [group_split_assignments.csv](recon_tables/group_split_assignments.csv).

| candidate | total_groups | groups_too_small |
| --- | --- | --- |
| chem_perturbation | 53 | 15 |
| genotype | 93 | 22 |
| genotype__strain | 23 | 2 |
| genotype_map_orig | 3 | 2 |
| orig_genotype | 4 | 2 |
| strain | 3 | 0 |

A group is flagged when any split has fewer than two physical embryos. Exact candidate/value/split counts: [metric_group_split_feasibility.csv](recon_tables/metric_group_split_feasibility.csv).

## Design-changing findings, audit comparison, and unmeasured items

The three findings most likely to change the bridge design are:

1. **Preflight must be cohort-wide and schema-aware:** 15 selected experiments lack a merged inventory, 39 lack merged staging, and both requested scale fields are absent from all 133 readable inventories.
2. **QC materially and nonuniformly changes cohort composition:** only 34.82% of QC-evaluable rows pass, experiment identity explains 14.38% of strict-gate variance, and 28 inventory-bearing experiments lack QC entirely.
3. **Intensity normalization/batch controls need an explicit decision:** the largest experiment η² among mean/std/saturation is 0.264, which is a material experiment signal by the stated screen.

Comparison with `NEW_PIPELINE_CORE_INTEGRATION_AUDIT.md`:

- The image-format claim is confirmed (100.00% conforming before the interlace check).
- The audit already warned that a live inventory lacked the two scale fields; this broader cohort determines whether that was isolated or systemic. It is an extension of the warning, not inherently a contradiction.
- The audit says `short_pert_name` is not guaranteed; this cohort confirms it is absent.
- No measured result contradicts the recommended inventory + stage + QC + plate boundary. The successful Parquet rerun removes the environment blocker, while the 28 missing QC artifacts and experiment-correlated exclusion strengthen the need for fail-loud preflight and cohort reporting.

Items not measured, with reasons:

- Strict QC and combined-gate eligibility for 167,603 inventory snips: 28 inventory-bearing experiments have no QC artifact, so these rows remain unresolved rather than failed.
- Direct product-type provenance: the inventory column is absent; path-segment inference is reported separately.
- `z_position` distribution: no populated source column at the snip-inventory boundary.
- µm/px distributions and mixed-scale determination: requested fields absent/unpopulated.
- Truly cold I/O: the shared filesystem page cache cannot be flushed safely; first-pass and warm measurements are both reported.
