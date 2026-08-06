"""PCA and reference-spline fitting over the legacy morphology latents.

Ported from ``results/nlammers/20250310/fit_morph_spline_v2.ipynb`` (cells 9, 12, 14), extended to
project the GENE7 cohort onto the same basis.

Pipeline
--------
1. **PCA** on ``z_mu_b_*`` (biological latents only), ``n_components=5``, fit on the pooled
   reference + 20240813 rows — the notebook's ``morph_pca.fit(pd.concat([ref, hf]))``. GENE7 is
   *transformed* on this basis, never fit into it, so adding GENE7 cannot move the axes the reference
   trajectory is defined in.

2. **Reference spline** via ``spline_fit_wrapper`` (``src/core/functions/spline_fitting_v2.py``): 50
   bootstrap ``LocalPrincipalCurve`` fits of 1000 resampled points each, anchored to sampled
   early/late endpoints, averaged pointwise into 2500 spline points. Not a classical spline — a local
   principal curve through PCA space.

   Two variants are produced, matching the notebook:

   ``unweighted``  reference + 20240813@28.5C pooled, uniform sampling weights.
   ``weighted``    same set, but the 20240813 controls are up-weighted to ``ALPHA`` of the sampling
                   mass. The notebook's rationale (cell 13): the reference trajectory systematically
                   diverges from the 28.5C cohort around 24 hpf, which distorts results for the other
                   temperature arms; up-weighting makes the curve flow through the plates' own
                   controls. ``ALPHA`` is acknowledged in the notebook as ad-hoc.

   GENE7 is NOT in either fitting set (per the chosen option (a)), so the curve is a reference
   defined independently of the data being measured against it.

Stage axis is ``predicted_stage_hpf`` throughout; the notebook's degree-2 polynomial surface and its
``mdl_stage_hpf`` re-parameterization are deliberately omitted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# results/nlammers/<this dir>/morph_pca_spline.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.core.functions.spline_fitting_v2 import spline_fit_wrapper  # noqa: E402

from legacy_reference import (  # noqa: E402
    HOTFISH_CONTROL_TEMPERATURE,
    ReferenceSets,
    biological_latent_columns,
)

N_COMPONENTS = 5
PCA_COLUMNS: tuple[str, ...] = tuple(f"PCA_{p:02}_bio" for p in range(N_COMPONENTS))

# Spline fitting parameters (notebook cell 14).
ALPHA = 0.25  # target fraction of sampling mass from the 20240813 control arm
N_BOOTS = 50
N_SPLINE_POINTS = 2500
BOOT_SIZE = 1000

STAGE_COLUMN = "predicted_stage_hpf"
SPLINE_STAGE_COLUMN = "timepoint"  # what spline_fit_wrapper expects

SOURCE_REFERENCE = "reference"
SOURCE_HOTFISH = "20240813"
SOURCE_GENE7 = "GENE7"


@dataclass
class MorphPCA:
    """A fitted PCA basis plus the cohorts projected into it."""

    pca: PCA
    latent_columns: tuple[str, ...]
    reference: pd.DataFrame
    hotfish: pd.DataFrame
    gene7: pd.DataFrame | None = None

    @property
    def explained_variance(self) -> pd.DataFrame:
        ratio = self.pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "component": np.arange(1, len(ratio) + 1),
                "explained_variance_ratio": ratio,
                "cumulative": np.cumsum(ratio),
            }
        )

    def transform(self, frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        """Project ``frame``'s latents onto the fitted basis, carrying metadata through."""
        return _project(self.pca, frame, self.latent_columns, source=source)


def _carry_columns(frame: pd.DataFrame) -> list[str]:
    """Metadata worth carrying into the PCA frame, when present."""
    candidates = [
        "snip_id",
        "embryo_id",
        "well_id",
        "experiment_date",
        "experiment_id",
        "temperature",
        "short_pert_name",
        "predicted_stage_hpf",
        "surface_area_um",
        # GENE7 / sequencing-side fields
        "target",
        "perturbation",
        "timepoint_seq",
        "seq_sample_id",
    ]
    return [column for column in candidates if column in frame.columns]


def _project(
    pca: PCA, frame: pd.DataFrame, latent_columns: "tuple[str, ...]", *, source: str
) -> pd.DataFrame:
    missing = [column for column in latent_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: {len(missing)} latent column(s) absent, e.g. {missing[:5]}. "
            "All cohorts must be expressed in the same z_mu_b_* block."
        )
    coordinates = pca.transform(frame.loc[:, list(latent_columns)])
    out = pd.DataFrame(coordinates, columns=list(PCA_COLUMNS))
    carried = _carry_columns(frame)
    out[carried] = frame.loc[:, carried].to_numpy()
    out["source"] = source
    # spline_fit_wrapper keys on a 'timepoint' column; floor to whole hours as the notebook did.
    if STAGE_COLUMN in out.columns:
        out[SPLINE_STAGE_COLUMN] = np.floor(pd.to_numeric(out[STAGE_COLUMN], errors="coerce"))
    return out


def fit_morph_pca(
    sets: ReferenceSets,
    *,
    n_components: int = N_COMPONENTS,
    gene7: pd.DataFrame | None = None,
) -> MorphPCA:
    """Fit PCA on reference + 20240813 pooled, then project every cohort.

    GENE7 is transformed only. Fitting it in would let the perturbation cohort reshape the axes that
    the reference trajectory is expressed in, which would make "distance from the reference curve"
    depend on which perturbations happened to be included.
    """
    latent_columns = sets.latent_columns
    fitting_block = pd.concat(
        [sets.reference.loc[:, list(latent_columns)], sets.hotfish.loc[:, list(latent_columns)]]
    )
    pca = PCA(n_components=n_components)
    pca.fit(fitting_block)

    result = MorphPCA(
        pca=pca,
        latent_columns=latent_columns,
        reference=_project(pca, sets.reference, latent_columns, source=SOURCE_REFERENCE),
        hotfish=_project(pca, sets.hotfish, latent_columns, source=SOURCE_HOTFISH),
    )
    if gene7 is not None:
        result.gene7 = _project(pca, gene7, latent_columns, source=SOURCE_GENE7)
    return result


def build_spline_fitting_set(fitted: MorphPCA) -> pd.DataFrame:
    """Reference plus the 20240813 control arm — the notebook's ``fit_pca_df`` (cell 12)."""
    controls = fitted.hotfish.loc[
        np.isclose(
            pd.to_numeric(fitted.hotfish["temperature"], errors="coerce"),
            HOTFISH_CONTROL_TEMPERATURE,
        )
    ]
    if controls.empty:
        raise ValueError(
            f"no 20240813 embryos at {HOTFISH_CONTROL_TEMPERATURE}C — the control arm is the "
            "spline anchor, so an empty selection means temperatures were not attached."
        )
    return pd.concat([fitted.reference, controls], ignore_index=True)


def compute_spline_weights(fitting_set: pd.DataFrame, *, alpha: float = ALPHA) -> np.ndarray:
    """Sampling weights giving the 20240813 rows ``alpha`` of the total mass (notebook cell 14).

    The control arm is ~24 snips against ~14,200 reference snips, so uniform sampling would make it
    invisible to the bootstrap. These weights become ``p=`` in ``spline_fit_wrapper``'s resample.
    """
    is_hotfish = (fitting_set["source"] == SOURCE_HOTFISH).to_numpy()
    n_total = len(is_hotfish)
    n_hotfish = int(is_hotfish.sum())
    if n_hotfish == 0 or n_hotfish == n_total:
        raise ValueError(
            "the fitting set must mix reference and 20240813 rows for alpha-weighting to mean "
            f"anything (got {n_hotfish} hotfish of {n_total})."
        )
    weights = np.empty(n_total, dtype=float)
    weights[is_hotfish] = alpha * n_total / n_hotfish
    weights[~is_hotfish] = (1.0 - alpha) * n_total / (n_total - n_hotfish)
    return weights


def fit_reference_splines(
    fitted: MorphPCA,
    *,
    alpha: float = ALPHA,
    n_boots: int = N_BOOTS,
    n_spline_points: int = N_SPLINE_POINTS,
    boot_size: int = BOOT_SIZE,
) -> dict[str, pd.DataFrame]:
    """Fit both spline variants. Returns ``{"unweighted": df, "weighted": df}``.

    Each frame has the ``PCA_*_bio`` columns plus ``*_se`` bootstrap standard errors.
    """
    fitting_set = build_spline_fitting_set(fitted)
    shared = dict(
        n_boots=n_boots,
        n_spline_points=n_spline_points,
        stage_col=SPLINE_STAGE_COLUMN,
        boot_size=boot_size,
    )
    return {
        "unweighted": spline_fit_wrapper(fitting_set, obs_weights=None, **shared),
        "weighted": spline_fit_wrapper(
            fitting_set, obs_weights=compute_spline_weights(fitting_set, alpha=alpha), **shared
        ),
    }


def gene7_latents_from_master(master: pd.DataFrame) -> pd.DataFrame:
    """Rename GENE7's legacy latent block to the companion's naming and keep what PCA needs.

    ``build_master_table`` emits ``z_mu_n_*`` / ``z_mu_b_*`` already, so this mostly guards that the
    biological block is present and disambiguates the two ``timepoint`` meanings: the sequencing
    ``timepoint`` (collection hpf) is preserved as ``timepoint_seq`` so it is not overwritten by the
    floored morphological stage that ``spline_fit_wrapper`` keys on.
    """
    out = master.copy()
    if "timepoint" in out.columns:
        out = out.rename(columns={"timepoint": "timepoint_seq"})
    if "temp" in out.columns and "temperature" not in out.columns:
        out["temperature"] = pd.to_numeric(out["temp"], errors="coerce")
    latent_columns = biological_latent_columns(list(out.columns))
    if not latent_columns:
        raise ValueError("master table carries no z_mu_b_* columns.")
    return out
