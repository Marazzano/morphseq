"""Assert that a run actually got the pipeline's run policy, not Snakemake's defaults.

WHY THIS EXISTS. `profiles/default/config.yaml` holds five settings where the Snakemake default is
actively wrong for this pipeline (a docstring edit re-running SAM2 across every well; one bad well
killing a 96-well run). Those values are policy for every run, not per-invocation choices.

HOW THE PROFILE IS FOUND. Snakemake 7.32.4 auto-discovers `profiles/default` -- searching both
relative to the current directory AND relative to the Snakefile, in that order (see the
`--workflow-profile` help text). Because the profile sits beside this workflow, it loads on any
normal invocation regardless of cwd. That is GOOD, and it means the policy is hard to lose by
accident; an earlier comment in profiles/default/config.yaml claiming "Snakemake 7 does not
auto-discover workflow profiles" is simply wrong.

WHAT THIS GUARD IS ACTUALLY FOR, then, is not the accidental case. It is:
  * `--workflow-profile none`, which silently reverts every value (verified: rerun_triggers
    {mtime}->all five, gpu budget gone);
  * a profile that is moved, renamed, or has a typo'd key, where discovery quietly finds nothing;
  * a future Snakemake upgrade changing discovery semantics under us;
  * an override that a caller believes is narrow but that drops a value the pipeline depends on.
In every one of those the failure is otherwise silent -- the run simply behaves differently, and
the first evidence is a re-run of SAM2 across every well or several models on one card. This
module converts that into a hard failure at second zero, before any rule is defined.

NOT EVERY POLICY VALUE IS CHECKABLE. `keep-going` and `rerun-incomplete` are arguments to
`Workflow.execute()`, passed straight through to `JobScheduler.__init__` (scheduler.py:129) and
never stored on the Workflow object -- so at parse time their real values simply cannot be read.
sys.argv is the only available signal, and argv cannot see a value that came from the profile, so
an argv check would fire on correct runs and teach people to ignore it. They are therefore left
unverified here and rely on the profile alone. The two settings below ARE readable, and a run
that lost the profile loses all five together, so checking these two detects that case anyway.

NOT A REPLACEMENT FOR THE PROFILE. The profile remains the single source of the values; this only
verifies they arrived. Values are checked against the Workflow object, i.e. what Snakemake
actually resolved -- so an explicit CLI override still passes, since overriding on purpose is
legitimate and the profile documents itself as overridable.

FUTURE (NOT SCHEDULED): a Snakemake 8 upgrade is worth doing on its own merits, but it is NOT a
prerequisite for anything here and this guard does not depend on it. Auto-discovery already works
on the pinned 7.32.4. The guard stays useful across an upgrade precisely because it verifies the
resolved values rather than the mechanism that supplied them.
"""

from __future__ import annotations

from typing import Any


#: Settings that live on the Workflow object and so are readable at parse time.
#: Each entry is (attribute, expected, why-the-default-hurts).
REQUIRED_WORKFLOW_SETTINGS: tuple[tuple[str, Any, str], ...] = (
    (
        "rerun_triggers",
        frozenset({"mtime"}),
        "the default set also includes params/input/code/software-env, any of which re-runs SAM2 "
        "across every well after an unrelated code edit",
    ),
)

#: Scheduler resource budgets that must be present. A rule's `resources: gpu=1` declaration is
#: INERT without a matching budget: Snakemake treats the resource as unlimited and will start
#: `--cores N` SAM2 jobs on one card.
REQUIRED_GLOBAL_RESOURCES: tuple[tuple[str, int, str], ...] = (
    (
        "gpu",
        1,
        "frame_masks declares `resources: gpu=1`, but that declaration does nothing without a "
        "budget; without it --cores N starts N SAM2 jobs on a single card",
    ),
)

PROFILE_HINT = (
    "Pass the profile explicitly:\n"
    "    --profile src/data_pipeline/pipeline_orchestrator/profiles/default\n"
    "or run from the directory containing profiles/ (what the submit scripts do via cd)."
)


def check_run_policy(workflow: Any, *, strict: bool = True) -> list[str]:
    """Return a list of run-policy violations for `workflow`; raise on any when `strict`.

    Reads the resolved Workflow object rather than sys.argv, so it does not care HOW the policy
    arrived (profile, auto-discovery, or an explicit flag) -- only that it did.
    """

    problems: list[str] = []

    for attr, expected, why in REQUIRED_WORKFLOW_SETTINGS:
        actual = getattr(workflow, attr, None)
        if actual != expected:
            problems.append(
                f"{attr}: expected {expected!r}, got {actual!r}\n    why it matters: {why}"
            )

    budgets = getattr(workflow, "global_resources", None) or {}
    for name, expected, why in REQUIRED_GLOBAL_RESOURCES:
        actual = budgets.get(name)
        if actual != expected:
            problems.append(
                f"resource budget {name!r}: expected {expected!r}, got {actual!r}\n"
                f"    why it matters: {why}"
            )

    if problems and strict:
        raise RuntimeError(
            "This run did not receive the pipeline's run policy "
            "(profiles/default/config.yaml).\n\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\n"
            + PROFILE_HINT
        )

    return problems
