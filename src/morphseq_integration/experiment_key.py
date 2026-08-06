"""The explicit imaging-experiment -> sequencing-experiment key.

This is the ONE curated fact the crosswalk cannot derive. Everything else it needs
(``image_to_hash_map``, ``hash_plate_num``) already lives per-well in the morphseq plate
workbooks and flows through the pipeline's own plate metadata.

The key is hand-maintained on purpose. It was previously inferred by date from an external
collection-metadata workbook, which is fragile: dates go ambiguous when two sci experiments are
collected the same day, and blank for anything predating that workbook. A curator writing one
line per paired imaging experiment is both simpler and auditable.

Schema (``experiment_sequencing_key.csv``):
    experiment_id  pipeline experiment id, e.g. 20250612_24hpf_ctrl_atf6
    sci_expt       sci-PLEX experiment name, e.g. GENE7
    notes          free text; provenance, caveats, who confirmed it

One ``sci_expt`` per imaging experiment. Duplicate ``experiment_id`` rows fail loud rather than
silently taking the last one. (If a single imaging plate ever spans two sci experiments, this key
grows a ``hash_plate`` column and the join gains that field — not a reason to loosen it now.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KEY_FILENAME = "experiment_sequencing_key.csv"
KEY_COLUMNS: tuple[str, ...] = ("experiment_id", "sci_expt", "notes")
_REQUIRED_VALUES: tuple[str, ...] = ("experiment_id", "sci_expt")


def default_key_path() -> Path:
    """Path to the key shipped alongside this module."""
    return Path(__file__).resolve().parent / KEY_FILENAME


@dataclass(frozen=True)
class ExperimentKey:
    """Validated imaging -> sequencing experiment key."""

    table: pd.DataFrame

    def sci_expt_for(self, experiment_id: str) -> str | None:
        """The sci experiment paired with ``experiment_id``, or ``None`` if unmapped."""
        hit = self.table.loc[self.table["experiment_id"] == str(experiment_id), "sci_expt"]
        return None if hit.empty else str(hit.iloc[0])

    @property
    def experiment_ids(self) -> list[str]:
        """Every imaging experiment with a key entry, in file order."""
        return self.table["experiment_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.table)


def load_experiment_key(key_path: str | Path | None = None) -> ExperimentKey:
    """Read and validate the experiment key.

    Args:
        key_path: Override the shipped key (useful for tests and one-off curation branches).

    Raises:
        FileNotFoundError: if the key file does not exist.
        ValueError: on a missing required column, a blank required value, or a duplicate
            ``experiment_id``.
    """
    path = Path(key_path) if key_path is not None else default_key_path()
    if not path.is_file():
        raise FileNotFoundError(f"[morphseq_integration] experiment key not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")

    missing = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"[morphseq_integration] {path.name} is missing required column(s) {missing}. "
            f"Expected header: {','.join(KEY_COLUMNS)}"
        )

    df = df.loc[:, list(KEY_COLUMNS)].copy()
    for column in KEY_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    # Drop wholly blank lines (trailing newlines, spacer rows) before validating.
    df = df.loc[~(df["experiment_id"] == "")].reset_index(drop=True)

    blank = df.loc[df["sci_expt"] == "", "experiment_id"].tolist()
    if blank:
        raise ValueError(
            f"[morphseq_integration] {path.name} has rows with an experiment_id but no sci_expt: "
            f"{blank}. Fill in the sci experiment, or delete the row until it is known."
        )

    duplicated = df.loc[df.duplicated(subset=["experiment_id"], keep=False), "experiment_id"]
    if not duplicated.empty:
        raise ValueError(
            f"[morphseq_integration] {path.name} has duplicate experiment_id rows: "
            f"{sorted(set(duplicated))}. Each imaging experiment maps to exactly one sci_expt."
        )

    return ExperimentKey(table=df)
