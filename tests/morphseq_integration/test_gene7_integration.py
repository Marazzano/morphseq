"""The GENE7 validation gates, run against the live cluster data.

These skip when the cluster mounts are absent so the suite still runs off-cluster. They are the
regression form of the gates that qualified this bridge:

    seq set equality   the minted sample ids exactly equal the sequencing corpus's imaging embryos
    1-to-1 crosswalk   no well or sample is claimed twice
    reconciliation     every legacy/pipeline well discrepancy is explainable
    latent ordering    flat z_mu_{ii} corresponds to legacy dimension {ii}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphseq_integration import legacy_morph as lm
from morphseq_integration.assemble import (
    build_master_table,
    coverage_summary,
    load_seq_metadata,
    seq_metadata_path,
    verify_seq_pairing,
)
from morphseq_integration.crosswalk import STATUS_PAIRED, build_crosswalk, validate_crosswalk
from morphseq_integration.paths import ENV_PIPELINE_ROOT, default_paths

GENE7_EXPERIMENTS = [
    "20250612_24hpf_ctrl_atf6",
    "20250612_24hpf_wfs1_ctcf",
    "20250612_30hpf_ctrl_atf6",
    "20250612_30hpf_wfs1_ctcf",
    "20250612_36hpf_ctrl_atf6",
    "20250612_36hpf_wfs1_ctcf",
]

EXPECTED_PAIRED_WELLS = 6 * 96  # every hash well of all six plates
N_CURATED_EXCLUSIONS = 14  # image-QC exclusions, see excluded_wells.csv
EXPECTED_ANALYZED_WELLS = EXPECTED_PAIRED_WELLS - N_CURATED_EXCLUSIONS


def _have_pipeline() -> bool:
    try:
        default_paths()
    except (FileNotFoundError, OSError):
        return False
    return True


def _have_legacy() -> bool:
    try:
        lm.default_legacy_root()
    except (FileNotFoundError, OSError):
        return False
    return True


def _have_seq() -> bool:
    try:
        return seq_metadata_path("GENE7").is_file()
    except OSError:
        return False


needs_pipeline = pytest.mark.skipif(
    not _have_pipeline(), reason=f"pipeline tree unavailable (set ${ENV_PIPELINE_ROOT})"
)
needs_legacy = pytest.mark.skipif(not _have_legacy(), reason="legacy latents unavailable")
needs_seq = pytest.mark.skipif(not _have_seq(), reason="seahub sequencing metadata unavailable")


@pytest.fixture(scope="module")
def crosswalk() -> pd.DataFrame:
    return build_crosswalk(GENE7_EXPERIMENTS)


@needs_pipeline
class TestCrosswalk:
    def test_covers_every_well_of_all_six_plates(self, crosswalk):
        assert len(crosswalk) == EXPECTED_PAIRED_WELLS
        assert set(crosswalk["experiment_id"]) == set(GENE7_EXPERIMENTS)
        assert (crosswalk["pairing_status"] == STATUS_PAIRED).all()

    def test_is_one_to_one(self, crosswalk):
        validate_crosswalk(crosswalk)
        paired = crosswalk.loc[crosswalk["pairing_status"] == STATUS_PAIRED]
        assert paired["well_id"].is_unique
        assert paired["seq_sample_id"].is_unique

    def test_hash_plates_match_the_curated_pairing(self, crosswalk):
        """The imaging plate -> hash plate assignment, verified from both sides."""
        expected = {
            "20250612_24hpf_ctrl_atf6": "P01",
            "20250612_24hpf_wfs1_ctcf": "P02",
            "20250612_30hpf_ctrl_atf6": "P18",
            "20250612_30hpf_wfs1_ctcf": "P04",
            "20250612_36hpf_ctrl_atf6": "P05",
            "20250612_36hpf_wfs1_ctcf": "P06",
        }
        observed = crosswalk.groupby("experiment_id")["hash_plate"].unique()
        for experiment_id, hash_plate in expected.items():
            assert observed[experiment_id].tolist() == [hash_plate]


@needs_pipeline
@needs_seq
class TestSeqSetEquality:
    """The gate that makes the sample-id grammar correct rather than plausible."""

    def test_minted_ids_exactly_equal_the_sequencing_corpus(self, crosswalk):
        report = verify_seq_pairing(crosswalk)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["sci_expt"] == "GENE7"
        assert row["n_minted"] == EXPECTED_PAIRED_WELLS
        assert row["n_seq_imaging"] == EXPECTED_PAIRED_WELLS
        assert row["n_shared"] == EXPECTED_PAIRED_WELLS
        assert bool(row["equal"]) is True

    def test_set_equality_holds_in_both_directions(self, crosswalk):
        minted = set(crosswalk.loc[crosswalk["pairing_status"] == STATUS_PAIRED, "seq_sample_id"])
        seq_keys = set(load_seq_metadata("GENE7")["seq_sample_id"])
        assert minted == seq_keys
        assert not minted - seq_keys
        assert not seq_keys - minted

    def test_a_wrong_hash_plate_would_be_caught(self, crosswalk):
        """Negative control: the gate must fail on a corrupted pairing, or it proves nothing."""
        corrupted = crosswalk.copy()
        corrupted.loc[corrupted["hash_plate"] == "P18", "seq_sample_id"] = "GENE7_P99_A1"
        with pytest.raises(ValueError, match="do not match the sequencing corpus"):
            verify_seq_pairing(corrupted)

    def test_sequencing_metadata_is_sample_unique(self):
        seq = load_seq_metadata("GENE7")
        assert len(seq) == EXPECTED_PAIRED_WELLS
        assert seq["seq_sample_id"].is_unique


@needs_legacy
class TestLegacyLatents:
    def test_every_legacy_file_parses(self):
        """All 38 embedded experiments, including the multi-timepoint and multi-embryo ones."""
        experiments = lm.available_experiments()
        assert len(experiments) >= 38
        for experiment_id in experiments:
            frame = lm.load_legacy_latents(experiment_id)
            assert len(frame) > 0
            assert frame["well_id"].notna().all()
            assert (frame["local_embryo_index"] >= 1).all()

    def test_all_six_gene7_plates_are_embedded(self):
        embedded = set(lm.available_experiments())
        assert set(GENE7_EXPERIMENTS) <= embedded

    def test_latent_block_is_the_expected_checkpoint_shape(self):
        frame = lm.load_legacy_latents("20250612_30hpf_ctrl_atf6")
        latents = lm.classify_latent_columns(list(frame.columns))
        assert latents.n_dims == lm.EXPECTED_LATENT_DIMS
        assert latents.is_legacy_naming is True
        assert latents.split_index == 20

    def test_timelapse_experiments_carry_many_timepoints(self):
        """_tNNNN handling must be general, not special-cased to the single-frame plates."""
        frame = lm.load_legacy_latents("20240812")
        assert frame["time_index"].nunique() > 1

    def test_companion_does_not_cover_gene7(self):
        """embryo_stats_df.csv is the Nov-2024 training table and stops at 20250215. GENE7
        morphology metadata therefore comes from the plate workbook and the seq side instead."""
        latents = lm.load_legacy_latents("20250612_30hpf_ctrl_atf6")
        joined = lm.attach_legacy_metadata(latents)
        assert not joined["has_legacy_metadata"].any()
        assert len(joined) == len(latents)


@needs_pipeline
@needs_legacy
class TestLegacyPipelineReconciliation:
    """Legacy/new well discrepancies must be explainable, never silently inner-joined away."""

    EXPERIMENT = "20250612_30hpf_ctrl_atf6"
    KNOWN_NEW_ONLY = {"C06", "C07"}  # whole-frame mask blowouts the legacy build excluded
    KNOWN_MULTI_EMBRYO = {"E09", "F12"}  # new resolves 2 embryos where legacy found 1

    @pytest.fixture(scope="class")
    def frames(self):
        from data_pipeline.shared.identifiers import parse_physical_embryo_id

        parquet = default_paths().analysis_ready_parquet(self.EXPERIMENT)
        if not parquet.is_file():
            pytest.skip(f"analysis_ready parquet absent for {self.EXPERIMENT}")
        legacy = lm.load_legacy_latents(self.EXPERIMENT)
        new = pd.read_parquet(parquet)
        new["local_embryo_index"] = [
            parse_physical_embryo_id(value)[1] for value in new["physical_embryo_id"]
        ]
        return legacy, new

    def test_no_legacy_only_wells(self, frames):
        legacy, new = frames
        assert not set(legacy["well_id"]) - set(new["well_id"])

    def test_new_only_wells_are_the_known_blowouts(self, frames):
        legacy, new = frames
        new_only = {
            well_id.rsplit("_", 1)[1] for well_id in set(new["well_id"]) - set(legacy["well_id"])
        }
        assert new_only == self.KNOWN_NEW_ONLY

    def test_multi_embryo_wells_are_the_known_ones(self, frames):
        _, new = frames
        multi = {
            well_id.rsplit("_", 1)[1]
            for well_id in new.loc[new["local_embryo_index"] > 1, "well_id"]
        }
        assert multi == self.KNOWN_MULTI_EMBRYO

    def test_legacy_is_one_row_per_well(self, frames):
        legacy, _ = frames
        assert legacy["well_id"].is_unique

    def test_latent_dimension_ordering_corresponds(self, frames):
        """Flat z_mu_{ii} corresponds to legacy dimension {ii}.

        Both tables number 0-99 continuously, the prefix marking the split at 20, so the mapping is
        established by construction. The empirical check here is the SIGN of the diagonal: under a
        wrong permutation the diagonal correlations would be sign-random, and they are not.

        Magnitudes are deliberately not asserted tightly. The current pipeline's snips are degraded
        relative to the build that trained this checkpoint (smaller raster, more saturation), and
        these latent dimensions are mutually correlated (legacy self-off-diagonal |r| ~ 0.36), so
        per-dimension argmax is not a reliable recovery signal at n~93.
        """
        legacy, new = frames
        legacy_flat = legacy.rename(columns=lm.to_flat_latent_names(list(legacy.columns)))
        new_first = new.loc[new["local_embryo_index"] == 1]
        merged = legacy_flat.merge(new_first, on="well_id", suffixes=("_leg", "_new"))
        assert len(merged) == len(legacy), "the well join must stay 1-to-1"

        names = [lm.flat_latent_name("z_mu", index) for index in range(100)]
        left = merged[[f"{name}_leg" for name in names]].to_numpy(float)
        right = merged[[f"{name}_new" for name in names]].to_numpy(float)
        left_z = (left - left.mean(0)) / left.std(0)
        right_z = (right - right.mean(0)) / right.std(0)
        correlation = (left_z.T @ right_z) / len(merged)
        diagonal = np.diag(correlation)

        # Sign agreement: chance would be ~50%.
        assert (diagonal > 0).sum() >= 90
        # The diagonal is the dominant structure overall.
        off_diagonal = correlation - np.diag(diagonal)
        assert np.abs(diagonal).mean() > 2 * np.abs(off_diagonal).mean()
        # Biological dimensions survive the raster change far better than nuisance ones.
        assert np.abs(diagonal[20:]).mean() > np.abs(diagonal[:20]).mean()


@needs_pipeline
@needs_legacy
@needs_seq
class TestMasterTable:
    @pytest.fixture(scope="class")
    def master(self) -> pd.DataFrame:
        """The analysis-facing table: curated image-QC exclusions already dropped."""
        return build_master_table(GENE7_EXPERIMENTS)

    @pytest.fixture(scope="class")
    def master_all(self) -> pd.DataFrame:
        """Every paired well, exclusions retained — for checking the exclusion step itself."""
        return build_master_table(GENE7_EXPERIMENTS, exclusions="keep")

    def test_one_row_per_imaging_well(self, master_all):
        assert len(master_all) == EXPECTED_PAIRED_WELLS
        assert master_all["well_id"].is_unique

    def test_curated_exclusions_are_dropped_by_default(self, master, master_all):
        """Exclusions apply to BOTH modalities, so the analyzed well set is identical everywhere."""
        from morphseq_integration import excluded_well_ids

        assert len(master) == EXPECTED_ANALYZED_WELLS
        assert master["well_id"].is_unique
        assert not (set(master["well_id"]) & excluded_well_ids())
        assert len(master_all) - len(master) == N_CURATED_EXCLUSIONS

    def test_flag_mode_retains_and_marks(self):
        flagged = build_master_table(GENE7_EXPERIMENTS, exclusions="flag")
        assert len(flagged) == EXPECTED_PAIRED_WELLS
        assert int(flagged["curated_excluded"].sum()) == N_CURATED_EXCLUSIONS

    def test_every_well_has_sequencing(self, master):
        assert master["has_seq"].all()

    def test_morphology_coverage_is_high_and_accounted_for(self, master_all):
        """Wells without legacy morphology are wells the legacy build excluded, not join failures."""
        assert master_all["has_morph"].sum() == 567
        assert (~master_all["has_morph"]).sum() == 9

    def test_no_morph_without_seq(self, master):
        """Sequencing covers every hash well, so a morph-only row would mean a broken join."""
        assert not (master["has_morph"] & ~master["has_seq"]).any()

    def test_carries_the_legacy_latent_block(self, master):
        latents = lm.classify_latent_columns(list(master.columns))
        assert latents.n_dims == lm.EXPECTED_LATENT_DIMS
        assert latents.is_legacy_naming is True

    def test_carries_no_pipeline_qc(self, master):
        """QC from the current pipeline was computed on the degraded raster and does not apply."""
        for column in ("use_snip", "qc_fail_reasons", "sa_outlier_flag", "focus_flag"):
            assert column not in master.columns

    def test_experimental_design_is_balanced(self, master_all):
        """Independent sanity check on the join: 4 targets x 3 temperature arms x 3 timepoints."""
        design = pd.crosstab(master_all["target"], master_all["timepoint"])
        assert sorted(design.columns) == [24, 30, 36]
        assert (design.loc[[t for t in design.index if t.endswith("hot")]] == 24).all().all()
        assert (design.loc[[t for t in design.index if t.endswith("cold")]] == 12).all().all()

    def test_all_four_temperatures_appear_within_every_plate(self, master):
        """The reason the retired scalar experiment_metadata.temperature could not represent GENE7."""
        per_plate = master.groupby("experiment_id")["temp"].nunique()
        assert (per_plate == 4).all()

    def test_coverage_summary_is_honest(self, master):
        summary = coverage_summary(master)
        assert len(summary) == len(GENE7_EXPERIMENTS)
        assert summary["n_wells"].sum() == EXPECTED_ANALYZED_WELLS
        assert summary["n_both"].sum() == master["has_morph"].sum()
        assert (summary["n_both"] <= summary["n_wells"]).all()
