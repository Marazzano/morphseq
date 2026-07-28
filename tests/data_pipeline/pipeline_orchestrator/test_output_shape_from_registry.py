"""Repo-wide guard: a rule's `output:` shape is DERIVED from the registry, never hand-written.

See src/data_pipeline/docs/PIPELINE_OVERVIEW.md §B3 ("Execution grain").

A per-well shard set can be named two ways, and the choice IS the job count:

    symbolic  output: ".../per_well/{well_id}/{well_id}_x.csv"   Snakemake resolves -> N jobs
    resolved  output: [".../A01/A01_x.csv", ...]                 already resolved   -> 1 job

Both write the same N shards to the same N paths — only the process count differs, so paths.py
cannot see the difference and neither can a reader. That invisibility is why `execution` drifted
into decoration: `frame_masks` and `latent_embeddings` both declared EXECUTION_RUN_BATCH while
their rules fanned out per well, paying the SAM2 / VAE model load N times instead of once, and
nothing objected.

The intended fix is a constructor, not a validator: rules would call
`well_runner.step_outputs(step, ...)`, which reads `paths.execution_mode(step)` and emits the shape
that mode implies, making the wrong pairing unconstructible. This test pins the one remaining
bypass — hand-typing the wildcard into `output:` — so the registry cannot silently stop governing
job count again.

STATUS (2026-07-25): **`step_outputs()` has not been written.** Every rule in this repo therefore
hand-writes its wildcard and sits on `_GRANDFATHERED` below; there is currently no way for a rule
to comply with the doctrine this module states. Until the constructor exists, the list is the only
honest place for a new per-well rule, and this test's real value is the second check —
`test_batch_steps_are_never_grandfathered` — which does bite.

DEVELOPMENT NOTE. This is a test-time doctrine check, never a runtime gate: a scratch rule with a
hand-written output runs fine locally and only trips here. Pre-existing per-well rules are
grandfathered by `_GRANDFATHERED` below; that list may only SHRINK. New rules use step_outputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "data_pipeline"
_RULES_DIR = _SRC_ROOT / "pipeline_orchestrator" / "rules"

# The symbolic well, hand-typed. paths.WILDCARD_WELL_ID is the ONE spelling; a rule that writes it
# literally into `output:` is choosing its own fanout instead of asking the registry for it.
_WILDCARD_LITERAL = "{well_id}"

# Rules whose `output:` predates step_outputs. They are correct today — each declares
# EXECUTION_PER_WELL and fans out per well, which agrees with the registry — but they assert that
# shape by hand rather than deriving it. Converting them is mechanical and not urgent; the two
# steps that MATTER (the EXECUTION_RUN_BATCH pair) must never appear here.
#
# This list may only shrink. Adding to it re-opens the exact drift the module docstring describes.
_GRANDFATHERED: frozenset[str] = frozenset({
    "assemble_well_frame_inventory",
    "build_curvature_metrics_for_well",
    "build_death_detection_for_well",
    "build_focus_qc_for_well",
    "build_fraction_alive_for_well",
    "build_mask_geometry_for_well",
    "build_mask_quality_qc_for_well",
    "build_motion_blur_qc_for_well",
    "build_physical_embryo_registry_for_well",
    "build_pose_kinematics_for_well",
    "build_snip_auxiliary_masks_for_well",
    # The two resident-model-server client rules (2026-07-25). They are per-well by design and
    # agree with the registry (both steps are EXECUTION_PER_WELL): a server changes WHERE the model
    # lives, not how many jobs Snakemake runs. Each is a thin socket client paired with a
    # service() rule that loads the model once per run. They are listed here for the same reason as
    # every other entry -- step_outputs() does not exist yet (see the note above this list) -- and
    # they come off together with their in-process twins when it does.
    "build_snip_auxiliary_masks_for_well_served",
    "frame_detections_per_well_served",
    "build_snip_qc_for_well",
    "build_stage_predictions_for_well",
    "build_surface_area_qc_for_well",
    "discover_product_shards_for_well",
    "frame_detections_per_well",
    # frame_masks_per_well joined this list on 2026-07-25, when frame_masks was retagged
    # EXECUTION_RUN_BATCH -> EXECUTION_PER_WELL. It was excluded before only because a RUN_BATCH
    # step may never be grandfathered; now that the registry agrees the rule is per-well, it is an
    # ordinary hand-written-wildcard case like every other entry here.
    "frame_masks_per_well",
    "materialize_image_product_for_well",
    "snip_processing_per_well",
    "split_dropin_inventory",
    "validate_curvature_metrics_for_well",
    "validate_death_detection_qc_for_well",
    "validate_death_event_for_well",
    "validate_focus_qc_for_well",
    "validate_fraction_alive_for_well",
    "validate_frame_detections_for_well",
    "validate_frame_inventory_for_well",
    "validate_frame_inventory_product_for_well",
    "validate_frame_masks_for_well",
    "validate_latent_embeddings_for_well",
    "validate_mask_geometry_for_well",
    "validate_mask_quality_qc_for_well",
    "validate_motion_blur_qc_for_well",
    "validate_physical_embryo_registry_for_well",
    "validate_pose_kinematics_for_well",
    "validate_snip_auxiliary_masks_for_well",
    "validate_snip_inventory_for_well",
    "validate_snip_qc_for_well",
    "validate_stage_predictions_for_well",
    "validate_surface_area_qc_for_well",
    # latent_embeddings is the one step still DECLARING run_batch while its rule fans out per
    # well. It is exempt here only because step_outputs() does not exist (see the note above this
    # list) -- not because the pairing is correct. It is CPU-bound, so unlike the GPU steps it was
    # not given a resident server; the honest fix is to retag the registry row PER_WELL.
    "encode_latent_embeddings_for_well",
    "write_resolved_product_plan_for_well",
    "write_snip_qc_resolved_sources_for_well",
})


def _rule_output_blocks(text: str) -> list[tuple[str, str]]:
    """(rule_name, output_block_text) for every rule that has an `output:` section."""
    blocks = []
    for block in re.split(r"\n(?=rule \w+:)", text):
        header = re.match(r"rule (\w+):", block)
        if not header:
            continue
        output_match = re.search(
            r"\n\s*output:\s*\n(.*?)\n\s*(?:input:|params:|shell:|run:|script:|resources:|threads:)",
            block,
            re.DOTALL,
        )
        if output_match:
            blocks.append((header.group(1), output_match.group(1)))
    return blocks


def _rule_files() -> list[Path]:
    files = sorted(_RULES_DIR.glob("*.smk"))
    assert files, f"no rule files found under {_RULES_DIR} — test would vacuously pass"
    return files


@pytest.mark.parametrize("rule_file", _rule_files(), ids=lambda p: p.name)
def test_output_shape_is_not_hand_written(rule_file: Path) -> None:
    """No rule hand-types the `{well_id}` wildcard into `output:` — shape comes from the registry."""
    offenders = [
        name
        for name, block in _rule_output_blocks(rule_file.read_text(encoding="utf-8"))
        if _WILDCARD_LITERAL in block and name not in _GRANDFATHERED
    ]
    assert not offenders, (
        f"{rule_file.name}: rule(s) {offenders} hand-write '{_WILDCARD_LITERAL}' in output:, "
        f"choosing per-well fanout instead of deriving it from the registry. Use "
        f"well_runner.step_outputs(root, step, artifact, experiment, config) so paths.execution_mode "
        f"decides the shape. See PIPELINE_OVERVIEW.md §B3."
    )


def test_batch_steps_are_never_grandfathered() -> None:
    """A RUN_BATCH step may never sit on the grandfather list — that is the drift this test exists for."""
    import sys

    sys.path.insert(0, str(_SRC_ROOT.parent))
    from data_pipeline.pipeline_orchestrator.orchestration import paths

    batch_steps = {
        step
        for step in paths.PIPELINE_STEPS
        if paths.execution_mode(step) == paths.EXECUTION_RUN_BATCH
    }
    assert batch_steps, "no RUN_BATCH steps in the registry — has execution been removed?"

    leaked = sorted(step for step in batch_steps if step in _GRANDFATHERED)
    assert not leaked, (
        f"RUN_BATCH step(s) {leaked} are grandfathered. A batch step MUST derive its output shape: "
        f"a hand-written wildcard makes Snakemake fan out one job per well, so the model loads N "
        f"times — the exact cost EXECUTION_RUN_BATCH exists to avoid."
    )
