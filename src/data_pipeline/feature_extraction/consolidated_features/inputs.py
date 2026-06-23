"""consolidated_features input assembly — load upstream feature shards via the path registry.

Per the snip_qc doctrine: this module owns the small, explicit in-memory assembly of upstream
feature tables. It resolves each product's per-well shard through ``orchestration.paths``
(``artifact_path``) — it does NOT take raw CSV path args and does NOT duplicate the registry. The
entrypoint passes the explicit list of source feature steps to merge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.pipeline_orchestrator.orchestration.paths import (
    PATH_MODE_PER_WELL,
    artifact_path,
    known_artifacts,
)


def load_feature_shards(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    feature_steps: list[str],
) -> dict[str, pd.DataFrame]:
    """Return ``{step -> DataFrame}`` for each per-well feature shard named in ``feature_steps``.

    The artifact key is inferred via ``known_artifacts(step)`` when the step has exactly one
    artifact (true for every feature product); a step with multiple artifacts fails loud and must
    be passed explicitly upstream.
    """
    tables: dict[str, pd.DataFrame] = {}
    for step in feature_steps:
        artifacts = known_artifacts(step)
        if len(artifacts) != 1:
            raise ValueError(
                f"consolidated_features: step {step!r} has artifacts {artifacts}; cannot infer a "
                "single artifact key. Pass the artifact key explicitly."
            )
        path = artifact_path(
            output_root, step, artifacts[0], experiment_id,
            path_mode=PATH_MODE_PER_WELL, well_id=well_id,
        )
        if not Path(path).exists():
            raise ValueError(
                f"consolidated_features: feature shard for step {step!r} not found at {path}."
            )
        tables[step] = pd.read_csv(path)
    return tables
