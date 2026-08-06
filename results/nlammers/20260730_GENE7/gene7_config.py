"""Single source of truth for figure output and contrast exclusions across the GENE7 notebooks.

Everything that renders a figure imports from here, so:

  * repointing the output root is a ONE-LINE change (``FIGURE_ROOT`` below), and
  * the excluded-contrast list cannot drift between notebooks.

Layout under ``FIGURE_ROOT``, one subfolder per producing notebook:

    synthesis/        gene7_synthesis.ipynb        -- the authoritative narrative
    lda_contrasts/    gene7_lda_contrasts.ipynb    -- axis construction and validation (feeds §1)
    edger_results/    gene7_edger_results.ipynb    -- full regression detail (feeds §1-§5)
    unsupervised/     gene7_unsupervised.ipynb     -- the label-free arm (held separate)
    morphospace/      gene7_morphospace.ipynb      -- PCA and reference spline (feeds §0)
    cohort_axes/      gene7_cohort_axes.ipynb      -- intrinsic per-cohort axes
    slides/           render_slide_figures.py      -- standalone slide-ready renders

WRITING SOMEWHERE ELSE (e.g. a synced Drive folder): set ``FIGURE_ROOT`` to any path this host can
actually see, or export ``GENE7_FIGURE_ROOT``. A local Google Drive mount on a laptop is NOT visible
from the cluster, so those figures have to be copied or rsynced after the fact.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --- where figures go -------------------------------------------------------------------------
FIGURE_ROOT = Path(os.environ.get("GENE7_FIGURE_ROOT", HERE / "figures"))

# --- contrasts dropped from every analysis and every figure -----------------------------------
# atf6 | 34C | 30hpf is driven by single-point leverage in BOTH groups: one crispant at s_z = 4.37
# with the next highest at 0.61 (gap +3.76 against a next gap of +0.52; robust z = 11.1), and one
# control at 0.99 against a next of -0.06 (robust z = 9.8). The binary indicator resolves 2 cell
# types there while the morphology score resolves 10 and the within-crispant slope 11 -- a
# regression hanging off one embryo, not a phenotype.
#
# The call is made on the shape of the MORPHOLOGY distribution alone, never on a transcriptional
# outcome, so it does not select on the quantity being measured. It is applied at load time in the
# notebooks rather than in the fitting scripts, so data/edger/*.csv remain a complete record.
EXCLUDED_CONTRASTS = ["atf6 vs ctrl | 34C | 30hpf"]


# --- the temperature colour scale, shared by every panel that colours by temperature -----------
# RdBu_r is diverging, so it has a white point, and a diverging map only means anything if that
# white point is anchored to something. Scaling to the data's own min/max anchors it to nothing: the
# white point lands at (min+max)/2, which is an artefact of which cohorts happen to be in the frame
# and MOVES when a contrast is dropped -- the same 34C dot changes colour between two panels of the
# same figure set.
#
# Anchoring at 28C fixes the meaning: white is standard rearing temperature and colour reads as
# deviation from it, blue colder and red warmer, identically in every panel. The range is symmetric
# about 28 so equal deviations get equal saturation.
REFERENCE_TEMPERATURE = 28.0
TEMPERATURE_SCALE = "RdBu_r"


def temperature_limits(values, *, reference: float = REFERENCE_TEMPERATURE):
    """``(vmin, vmax)`` for a temperature colour map, symmetric about ``reference``."""
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    span = float(np.max(np.abs(finite - reference))) if finite.size else 1.0
    return reference - (span or 1.0), reference + (span or 1.0)


def figure_dir(notebook: str) -> Path:
    """Create and return the output folder for one notebook's figures."""
    target = FIGURE_ROOT / notebook
    target.mkdir(parents=True, exist_ok=True)
    return target


def drop_excluded(*frames, column: str = "contrast"):
    """Filter ``EXCLUDED_CONTRASTS`` out of any frames that carry a contrast column.

    Returns the frames in the order given, so a notebook can rebind them in one statement.
    Frames without the column pass through untouched.
    """
    out = []
    for frame in frames:
        if frame is not None and column in getattr(frame, "columns", ()):
            frame = frame[~frame[column].isin(EXCLUDED_CONTRASTS)].reset_index(drop=True)
        out.append(frame)
    return out[0] if len(out) == 1 else tuple(out)
