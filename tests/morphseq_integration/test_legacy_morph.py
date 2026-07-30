"""Legacy latent parsing: the historical snip_id grammar and the latent-name correspondence."""

from __future__ import annotations

import pandas as pd
import pytest

from morphseq_integration.legacy_morph import (
    EXPECTED_LATENT_DIMS,
    SPINE_COLUMNS,
    attach_legacy_metadata,
    classify_latent_columns,
    flat_latent_name,
    legacy_latent_name,
    load_legacy_latents,
    parse_legacy_snip_id,
    to_flat_latent_names,
)


def _legacy_latent_columns() -> list[str]:
    """The 200-column legacy latent block: n_00-19 then b_20-99, for mu and sigma."""
    columns = []
    for kind in ("z_mu", "z_sigma"):
        columns += [f"{kind}_n_{i:02d}" for i in range(20)]
        columns += [f"{kind}_b_{i:02d}" for i in range(20, 100)]
    return columns


def _flat_latent_columns() -> list[str]:
    return [f"{kind}_{i:02d}" for kind in ("z_mu", "z_sigma") for i in range(100)]


def _write_latents_csv(directory, experiment_id: str, snip_ids: list[str]) -> None:
    spine = pd.DataFrame(
        {
            "experiment_date": experiment_id,
            "snip_id": snip_ids,
            "embryo_id": [s.rsplit("_t", 1)[0] for s in snip_ids],
        }
    )
    latents = pd.DataFrame(
        {
            column: [float(position + row) for row in range(len(snip_ids))]
            for position, column in enumerate(_legacy_latent_columns())
        }
    )
    pd.concat([spine, latents], axis=1).to_csv(
        directory / f"morph_latents_{experiment_id}.csv", index=False
    )


class TestParseLegacySnipId:
    def test_parses_the_legacy_grammar_and_shifts_to_one_based(self):
        """Legacy _e00 is the first embryo; the canonical contract is one-based."""
        assert parse_legacy_snip_id("20250612_30hpf_ctrl_atf6_A01_e00_t0000") == (
            "20250612_30hpf_ctrl_atf6",
            "A01",
            1,
            0,
        )

    def test_handles_a_second_embryo_and_a_late_timepoint(self):
        assert parse_legacy_snip_id("20250624_chem02_35C_T00_1216_H12_e01_t0074") == (
            "20250624_chem02_35C_T00_1216",
            "H12",
            2,
            74,
        )

    def test_requires_a_canonical_padded_well_label(self):
        """Every real legacy id is zero-padded, so an unpadded label is corruption, not a variant."""
        with pytest.raises(ValueError):
            parse_legacy_snip_id("20240812_A1_e00_t0000")

    def test_rejects_the_current_pipeline_grammar(self):
        """A current snip_id carries a channel token; parsing it here would misread the well."""
        with pytest.raises(ValueError, match="cannot parse legacy snip_id"):
            parse_legacy_snip_id("20250612_30hpf_ctrl_atf6_A01_e01_BF_t0000")

    @pytest.mark.parametrize(
        "snip_id",
        [
            "20250612_A01_e00",  # no timepoint
            "20250612_A01_t0000",  # no embryo token
            "20250612_Z99_e00_t0000",  # well outside the plate
            "20250612_A13_e00_t0000",  # column outside 1-12
            "",
        ],
    )
    def test_rejects_malformed_ids(self, snip_id):
        with pytest.raises(ValueError):
            parse_legacy_snip_id(snip_id)


class TestClassifyLatentColumns:
    def test_reads_the_legacy_split_off_the_column_names(self):
        latents = classify_latent_columns(["snip_id", *_legacy_latent_columns()])
        assert latents.is_legacy_naming is True
        assert latents.n_dims == EXPECTED_LATENT_DIMS
        assert latents.split_index == 20
        assert latents.mu_columns[0] == "z_mu_n_00"
        assert latents.mu_columns[19] == "z_mu_n_19"
        assert latents.mu_columns[20] == "z_mu_b_20"
        assert latents.mu_columns[-1] == "z_mu_b_99"

    def test_orders_by_dimension_not_lexically(self):
        """Column order must follow the dimension index, and the b_/n_ prefix breaks sort order."""
        shuffled = list(reversed(_legacy_latent_columns()))
        latents = classify_latent_columns(shuffled)
        assert latents.mu_columns[0] == "z_mu_n_00"
        assert latents.mu_columns[-1] == "z_mu_b_99"

    def test_recognizes_the_flat_pipeline_naming(self):
        latents = classify_latent_columns(["snip_id", *_flat_latent_columns()])
        assert latents.is_legacy_naming is False
        assert latents.n_dims == EXPECTED_LATENT_DIMS
        assert latents.split_index is None

    def test_rejects_a_frame_carrying_both_namings(self):
        """Mixed naming makes dimension identity ambiguous, so it must fail rather than guess."""
        with pytest.raises(ValueError, match="BOTH legacy"):
            classify_latent_columns([*_legacy_latent_columns(), *_flat_latent_columns()])

    def test_rejects_a_frame_with_no_latents(self):
        with pytest.raises(ValueError, match="no z_mu"):
            classify_latent_columns(["snip_id", "experiment_date"])

    def test_rejects_mismatched_mu_and_sigma_blocks(self):
        columns = [c for c in _legacy_latent_columns() if c != "z_sigma_b_99"]
        with pytest.raises(ValueError, match="different dimension indices"):
            classify_latent_columns(columns)


class TestLatentNameCorrespondence:
    """The flat name z_mu_{ii} corresponds to legacy dimension {ii}: both number 0-99 continuously,
    with the prefix marking the nuisance/biological split at 20."""

    @pytest.mark.parametrize(
        ("index", "expected"), [(0, "z_mu_n_00"), (19, "z_mu_n_19"), (20, "z_mu_b_20"), (99, "z_mu_b_99")]
    )
    def test_legacy_name_places_the_split_at_twenty(self, index, expected):
        assert legacy_latent_name("z_mu", index) == expected

    def test_flat_name_is_zero_padded(self):
        assert flat_latent_name("z_mu", 5) == "z_mu_05"
        assert flat_latent_name("z_sigma", 99) == "z_sigma_99"

    def test_the_two_namings_round_trip_for_every_dimension(self):
        for kind in ("z_mu", "z_sigma"):
            for index in range(100):
                legacy = legacy_latent_name(kind, index)
                assert to_flat_latent_names([legacy])[legacy] == flat_latent_name(kind, index)

    def test_rename_map_preserves_dimension_index(self):
        renames = to_flat_latent_names(_legacy_latent_columns())
        assert renames["z_mu_n_00"] == "z_mu_00"
        assert renames["z_mu_b_20"] == "z_mu_20"
        assert renames["z_mu_b_99"] == "z_mu_99"
        assert len(renames) == 200

    def test_rename_map_ignores_non_latent_columns(self):
        assert to_flat_latent_names(["snip_id", "well_id", "temperature"]) == {}


class TestLoadLegacyLatents:
    def test_builds_the_canonical_spine(self, tmp_path):
        experiment_id = "20250612_30hpf_ctrl_atf6"
        _write_latents_csv(
            tmp_path,
            experiment_id,
            [f"{experiment_id}_A01_e00_t0000", f"{experiment_id}_A02_e00_t0000"],
        )
        frame = load_legacy_latents(experiment_id, legacy_root=tmp_path)

        assert list(frame.columns[: len(SPINE_COLUMNS)]) == list(SPINE_COLUMNS)
        assert len(frame) == 2
        assert frame["well_id"].tolist() == [
            f"{experiment_id}_A01",
            f"{experiment_id}_A02",
        ]
        # physical_embryo_id is re-minted one-based through the canonical constructor.
        assert frame["physical_embryo_id"].iloc[0] == f"{experiment_id}_A01_e01"
        assert frame["local_embryo_index"].tolist() == [1, 1]
        # The original id is preserved verbatim for provenance.
        assert frame["snip_id"].iloc[0] == f"{experiment_id}_A01_e00_t0000"
        assert set(frame["source"]) == {"legacy"}

    def test_keeps_legacy_latent_names_by_default(self, tmp_path):
        _write_latents_csv(tmp_path, "20250101", ["20250101_A01_e00_t0000"])
        frame = load_legacy_latents("20250101", legacy_root=tmp_path)
        assert "z_mu_n_00" in frame.columns
        assert "z_mu_b_20" in frame.columns
        assert "z_mu_00" not in frame.columns

    def test_renames_to_flat_on_request(self, tmp_path):
        _write_latents_csv(tmp_path, "20250101", ["20250101_A01_e00_t0000"])
        frame = load_legacy_latents("20250101", legacy_root=tmp_path, flat_latent_names=True)
        assert "z_mu_00" in frame.columns
        assert "z_mu_20" in frame.columns
        assert "z_mu_n_00" not in frame.columns
        assert sum(c.startswith("z_mu_") for c in frame.columns) == EXPECTED_LATENT_DIMS

    def test_preserves_latent_values_through_the_flat_rename(self, tmp_path):
        """The rename must move names, not data: dimension 20's value stays with dimension 20."""
        _write_latents_csv(tmp_path, "20250101", ["20250101_A01_e00_t0000"])
        legacy = load_legacy_latents("20250101", legacy_root=tmp_path)
        flat = load_legacy_latents("20250101", legacy_root=tmp_path, flat_latent_names=True)
        for index in (0, 19, 20, 99):
            assert (
                legacy[legacy_latent_name("z_mu", index)].iloc[0]
                == flat[flat_latent_name("z_mu", index)].iloc[0]
            )

    def test_handles_multiple_timepoints(self, tmp_path):
        snip_ids = [f"20240812_A01_e00_t{t:04d}" for t in range(5)]
        _write_latents_csv(tmp_path, "20240812", snip_ids)
        frame = load_legacy_latents("20240812", legacy_root=tmp_path)
        assert frame["time_index"].tolist() == [0, 1, 2, 3, 4]
        assert frame["well_id"].nunique() == 1

    def test_handles_multiple_embryos_per_well(self, tmp_path):
        _write_latents_csv(
            tmp_path,
            "20250624",
            ["20250624_A01_e00_t0000", "20250624_A01_e01_t0000"],
        )
        frame = load_legacy_latents("20250624", legacy_root=tmp_path)
        assert frame["local_embryo_index"].tolist() == [1, 2]
        assert frame["well_id"].nunique() == 1

    def test_fails_on_duplicate_identity(self, tmp_path):
        _write_latents_csv(
            tmp_path, "20250101", ["20250101_A01_e00_t0000", "20250101_A01_e00_t0000"]
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_legacy_latents("20250101", legacy_root=tmp_path)

    def test_fails_when_the_filename_and_ids_disagree(self, tmp_path):
        """A mis-copied file would otherwise be silently relabelled by the filename."""
        frame = pd.DataFrame({"snip_id": ["20240101_A01_e00_t0000"]})
        for column in _legacy_latent_columns():
            frame[column] = 0.0
        frame.to_csv(tmp_path / "morph_latents_20250101.csv", index=False)
        with pytest.raises(ValueError, match="different experiment_id"):
            load_legacy_latents("20250101", legacy_root=tmp_path)

    def test_fails_on_a_truncated_latent_block(self, tmp_path):
        frame = pd.DataFrame({"snip_id": ["20250101_A01_e00_t0000"]})
        for column in _legacy_latent_columns():
            if column not in ("z_mu_b_99", "z_sigma_b_99"):
                frame[column] = 0.0
        frame.to_csv(tmp_path / "morph_latents_20250101.csv", index=False)
        with pytest.raises(ValueError, match="latent dimensions"):
            load_legacy_latents("20250101", legacy_root=tmp_path)

    def test_missing_experiment_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no legacy latents"):
            load_legacy_latents("does_not_exist", legacy_root=tmp_path)


class TestAttachLegacyMetadata:
    def test_flags_uncovered_rows_rather_than_dropping_them(self, tmp_path):
        """GENE7 has no companion coverage at all; those rows must survive, visibly unmatched."""
        _write_latents_csv(tmp_path, "20250101", ["20250101_A01_e00_t0000"])
        latents = load_legacy_latents("20250101", legacy_root=tmp_path)
        joined = attach_legacy_metadata(latents, metadata=pd.DataFrame(columns=["snip_id"]))
        assert len(joined) == 1
        assert joined["has_legacy_metadata"].tolist() == [False]
        assert "temperature" in joined.columns
        assert joined["temperature"].isna().all()

    def test_joins_matching_metadata_and_marks_coverage(self, tmp_path):
        snip_ids = ["20250101_A01_e00_t0000", "20250101_A02_e00_t0000"]
        _write_latents_csv(tmp_path, "20250101", snip_ids)
        latents = load_legacy_latents("20250101", legacy_root=tmp_path)
        metadata = pd.DataFrame(
            {"snip_id": [snip_ids[0]], "temperature": [28.5], "short_pert_name": ["ctrl"]}
        )
        joined = attach_legacy_metadata(latents, metadata=metadata)

        assert len(joined) == 2
        assert joined.set_index("snip_id")["has_legacy_metadata"].to_dict() == {
            snip_ids[0]: True,
            snip_ids[1]: False,
        }
        assert joined.loc[joined.snip_id == snip_ids[0], "temperature"].iloc[0] == 28.5
        # A partial metadata frame joins what it has rather than demanding the full column set.
        assert "short_pert_name" in joined.columns
        assert "recon_mse" not in joined.columns

    def test_rejects_a_fan_out_join(self, tmp_path):
        _write_latents_csv(tmp_path, "20250101", ["20250101_A01_e00_t0000"])
        latents = load_legacy_latents("20250101", legacy_root=tmp_path)
        metadata = pd.DataFrame(
            {"snip_id": ["20250101_A01_e00_t0000"] * 2, "temperature": [28.5, 34.0]}
        )
        with pytest.raises(ValueError, match="duplicate snip_id"):
            attach_legacy_metadata(latents, metadata=metadata)
