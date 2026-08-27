#!/usr/bin/env python3
"""Generate computed core-model project status.

The default path inspects git, runs the bounded core test suite, cross-references
``pytest.mark.decision`` markers, and records the Python environment. Optional
data inspection consumes an explicit JSON artifact manifest; it never discovers
experiments or artifacts by globbing the pipeline output tree.

Data-config shape::

    {"experiments": [{"experiment_id": "opaque-id", "artifacts": {
      "snip_inventory": "/absolute/inventory.csv",
      "stage_predictions": "/absolute/stages.csv",
      "snip_qc": "/absolute/qc.parquet",
      "plate_metadata": "/absolute/plate.csv"}}]}
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs" / "refactors" / "core-model"
DECISIONS_PATH = DOCS_ROOT / "DECISIONS.md"
INCOMING_DECISIONS_PATH = REPO_ROOT / "DECISIONS.md"
STATUS_PATH = DOCS_ROOT / "STATUS.md"
TEST_TARGETS = ("tests/core",)
TEST_TIMEOUT_SECONDS = 6.0
REQUIRED_DATA_ARTIFACTS = (
    "snip_inventory",
    "stage_predictions",
    "snip_qc",
    "plate_metadata",
)
COMMITS_OF_INTEREST = (
    ("Phase 1 adapter", "3c0986e6"),
    ("Phase 1 datasets", "96f3991d"),
    ("Phase 1 integration", "c3ca21a2"),
    ("Phase 1 completion", "0431e6d9"),
    ("Rendering defaults restored", "37aeb639"),
)
PACKAGE_DISTRIBUTIONS = (
    "pytest",
    "torch",
    "torchvision",
    "pandas",
    "numpy",
    "pyarrow",
    "pytorch-lightning",
    "hydra-core",
    "omegaconf",
)


# These hooks are active only in the pytest subprocess launched with
# ``-p scripts.status``. Keeping collection and outcomes in the same process
# avoids guessing test results from console text.
_PLUGIN_DECISIONS: dict[str, list[str]] = {}
_PLUGIN_OUTCOMES: dict[str, str] = {}
_PLUGIN_COLLECTION_ERRORS: list[str] = []


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("morphseq-status")
    group.addoption(
        "--status-json",
        action="store",
        default=None,
        help="Write machine-readable test and decision-marker results.",
    )


def pytest_collection_modifyitems(
    session: Any, config: Any, items: Sequence[Any]
) -> None:
    del session, config
    for item in items:
        decision_ids: list[str] = []
        for marker in item.iter_markers(name="decision"):
            decision_ids.extend(str(value) for value in marker.args)
        _PLUGIN_DECISIONS[item.nodeid] = list(dict.fromkeys(decision_ids))
        _PLUGIN_OUTCOMES[item.nodeid] = "not_run"


def pytest_collectreport(report: Any) -> None:
    if report.failed:
        _PLUGIN_COLLECTION_ERRORS.append(str(report.longrepr))


def pytest_runtest_logreport(report: Any) -> None:
    current = _PLUGIN_OUTCOMES.get(report.nodeid, "not_run")
    if report.failed:
        _PLUGIN_OUTCOMES[report.nodeid] = "failed"
    elif report.skipped:
        if current != "failed":
            _PLUGIN_OUTCOMES[report.nodeid] = (
                "xfailed" if hasattr(report, "wasxfail") else "skipped"
            )
    elif report.when == "call" and report.passed:
        if current != "failed":
            _PLUGIN_OUTCOMES[report.nodeid] = (
                "xpassed" if hasattr(report, "wasxfail") else "passed"
            )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    output_value = session.config.getoption("--status-json")
    if not output_value:
        return
    payload = {
        "exit_code": int(exitstatus),
        "collected": len(_PLUGIN_OUTCOMES),
        "outcomes": _PLUGIN_OUTCOMES,
        "decisions": _PLUGIN_DECISIONS,
        "collection_errors": _PLUGIN_COLLECTION_ERRORS,
    }
    Path(output_value).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Inspect explicitly configured source artifacts and include row counts.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=(
            Path(os.environ["MORPHSEQ_STATUS_DATA_CONFIG"])
            if "MORPHSEQ_STATUS_DATA_CONFIG" in os.environ
            else None
        ),
        help=(
            "JSON file with an ordered 'experiments' list; each entry must contain "
            "experiment_id and explicit paths for all four source artifacts."
        ),
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=Path.home() / ".cache" / "morphseq" / "status-data.json",
        help="Cache for source hashes and row counts.",
    )
    return parser.parse_args()


def run_command(args: Sequence[str], *, timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return run_command(("git", *args))


def git_text(*args: str, fallback: str = "unavailable") -> str:
    result = git_command(*args)
    if result.returncode != 0:
        return fallback
    return result.stdout.strip() or fallback


def revision_exists(revision: str) -> bool:
    return git_command("cat-file", "-e", f"{revision}^{{commit}}").returncode == 0


def ancestor_status(revision: str, target: str) -> str:
    if not revision_exists(revision):
        return "commit unavailable"
    if not revision_exists(target):
        return "target unavailable"
    result = git_command("merge-base", "--is-ancestor", revision, target)
    if result.returncode == 0:
        return "yes"
    if result.returncode == 1:
        return "no"
    return "error"


def collect_git_status() -> dict[str, Any]:
    branch = git_text("branch", "--show-current", fallback="detached HEAD")
    head = git_text("rev-parse", "HEAD")
    dirty_result = git_command("status", "--porcelain=v1", "--untracked-files=all")
    dirty_lines = dirty_result.stdout.splitlines() if dirty_result.returncode == 0 else []

    upstream_result = git_command(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    if upstream:
        unpushed_result = git_command(
            "log", "--format=%h %s", f"{upstream}..HEAD"
        )
        unpushed = (
            unpushed_result.stdout.splitlines() if unpushed_result.returncode == 0 else []
        )
        unpushed_error = (
            None if unpushed_result.returncode == 0 else unpushed_result.stderr.strip()
        )
    else:
        unpushed = []
        unpushed_error = "no upstream is configured"

    commits = []
    for label, revision in COMMITS_OF_INTEREST:
        commits.append(
            {
                "label": label,
                "revision": revision,
                "head": ancestor_status(revision, "HEAD"),
                "origin_main": ancestor_status(revision, "origin/main"),
            }
        )
    return {
        "branch": branch,
        "head": head,
        "dirty": dirty_lines,
        "upstream": upstream,
        "unpushed": unpushed,
        "unpushed_error": unpushed_error,
        "commits": commits,
    }


def run_tests() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="morphseq-status-") as temp_dir:
        result_path = Path(temp_dir) / "pytest-results.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *TEST_TARGETS,
            "-q",
            "--color=no",
            "--disable-warnings",
            "--tb=short",
            "-p",
            "scripts.status",
            "--status-json",
            str(result_path),
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=TEST_TIMEOUT_SECONDS,
                check=False,
            )
            timed_out = False
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            exit_code = 124
        duration = time.perf_counter() - started

        if result_path.exists():
            payload = json.loads(result_path.read_text())
        else:
            payload = {
                "collected": 0,
                "outcomes": {},
                "decisions": {},
                "collection_errors": [],
            }
        payload.update(
            {
                "command": command,
                "duration_seconds": duration,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "output": (stdout + "\n" + stderr).strip(),
                "declared_decisions": discover_decision_markers(),
            }
        )
        return payload


def decision_ids_from_decorators(decorators: Sequence[ast.expr]) -> list[str]:
    decision_ids: list[str] = []
    for decorator in decorators:
        if not isinstance(decorator, ast.Call):
            continue
        function = decorator.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "decision"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "mark"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "pytest"
        ):
            continue
        for argument in decorator.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                decision_ids.append(argument.value)
    return list(dict.fromkeys(decision_ids))


def discover_decision_markers() -> dict[str, list[str]]:
    markers: dict[str, list[str]] = {}
    for target in TEST_TARGETS:
        target_path = REPO_ROOT / target
        paths = (
            sorted(target_path.rglob("test_*.py"))
            if target_path.is_dir()
            else [target_path]
        )
        for path in paths:
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except (OSError, SyntaxError):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decision_ids = decision_ids_from_decorators(node.decorator_list)
                    if decision_ids:
                        markers[f"{relative}::{node.name}"] = decision_ids
                elif isinstance(node, ast.ClassDef):
                    class_ids = decision_ids_from_decorators(node.decorator_list)
                    for child in node.body:
                        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        function_ids = decision_ids_from_decorators(child.decorator_list)
                        decision_ids = list(dict.fromkeys([*class_ids, *function_ids]))
                        if decision_ids:
                            markers[f"{relative}::{node.name}::{child.name}"] = decision_ids
    return markers


def parse_ratified_decisions(path: Path) -> list[str]:
    text = path.read_text()
    match = re.search(r"^## Ratified\s*$([\s\S]*?)(?=^##\s|\Z)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"No '## Ratified' section found in {path}")
    decision_ids = re.findall(r"^\|\s*(D\d+)\s*\|", match.group(1), re.MULTILINE)
    duplicates = [
        decision_id
        for decision_id, count in Counter(decision_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate decision IDs in {path}: {', '.join(duplicates)}")
    if not decision_ids:
        raise ValueError(f"No decision rows found in {path}")
    return decision_ids


def resolve_decision_ledger() -> tuple[Path, list[str], str | None]:
    candidates = [DECISIONS_PATH]
    if INCOMING_DECISIONS_PATH != DECISIONS_PATH and INCOMING_DECISIONS_PATH.exists():
        candidates.append(INCOMING_DECISIONS_PATH)

    parsed: list[tuple[Path, list[str]]] = []
    failures: list[str] = []
    for path in candidates:
        try:
            parsed.append((path, parse_ratified_decisions(path)))
        except (OSError, ValueError) as exc:
            failures.append(str(exc))

    if not parsed:
        raise ValueError("; ".join(failures))
    if len(parsed) > 1:
        baseline_path, baseline_ids = parsed[0]
        for path, decision_ids in parsed[1:]:
            if decision_ids != baseline_ids or path.read_text() != baseline_path.read_text():
                raise ValueError(
                    f"divergent decision ledgers: {baseline_path} and {path}"
                )

    selected_path, decision_ids = parsed[0]
    warning = None
    if selected_path != DECISIONS_PATH:
        warning = (
            f"Canonical `{DECISIONS_PATH.relative_to(REPO_ROOT)}` has not received the "
            f"reconciled ledger; using merge-staged `{selected_path.relative_to(REPO_ROOT)}`."
        )
    return selected_path, decision_ids, warning


def correlate_decisions(
    decision_ids: Sequence[str], test_results: dict[str, Any]
) -> dict[str, Any]:
    tests_by_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
    outcomes = test_results.get("outcomes", {})
    for nodeid, marked_ids in test_results.get("decisions", {}).items():
        for decision_id in marked_ids:
            tests_by_decision[decision_id].append(
                {"nodeid": nodeid, "outcome": outcomes.get(nodeid, "not_run")}
            )

    collected_nodeids = set(test_results.get("decisions", {}))
    for nodeid, marked_ids in test_results.get("declared_decisions", {}).items():
        collected = any(
            collected_nodeid == nodeid or collected_nodeid.startswith(f"{nodeid}[")
            for collected_nodeid in collected_nodeids
        )
        if collected:
            continue
        for decision_id in marked_ids:
            tests_by_decision[decision_id].append(
                {"nodeid": nodeid, "outcome": "not_run"}
            )

    known = set(decision_ids)
    unknown_markers = sorted(set(tests_by_decision) - known)
    passing: list[str] = []
    with_tests: list[str] = []
    for decision_id in decision_ids:
        marked_tests = tests_by_decision.get(decision_id, [])
        if marked_tests:
            with_tests.append(decision_id)
            if all(test["outcome"] == "passed" for test in marked_tests):
                passing.append(decision_id)
    unverified = [decision_id for decision_id in decision_ids if decision_id not in passing]
    return {
        "tests_by_decision": dict(tests_by_decision),
        "with_tests": with_tests,
        "passing": passing,
        "unverified": unverified,
        "unknown_markers": unknown_markers,
    }


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_environment() -> dict[str, Any]:
    try:
        pyarrow = importlib.import_module("pyarrow")
        pyarrow_import = f"yes ({getattr(pyarrow, '__version__', 'unknown version')})"
    except Exception as exc:  # an installed but broken engine must remain visible
        pyarrow_import = f"no ({type(exc).__name__}: {exc})"
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "pyarrow_import": pyarrow_import,
        "packages": {
            distribution: package_version(distribution)
            for distribution in PACKAGE_DISTRIBUTIONS
        },
    }


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_data_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "fingerprints": {}, "results": {}}
    try:
        payload = load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"version": 1, "fingerprints": {}, "results": {}}
    if payload.get("version") != 1:
        return {"version": 1, "fingerprints": {}, "results": {}}
    payload.setdefault("fingerprints", {})
    payload.setdefault("results", {})
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(path: Path, cache: dict[str, Any]) -> str:
    resolved = str(path.resolve())
    stat = path.stat()
    digest = sha256_file(path)
    cache["fingerprints"][resolved] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }
    return digest


def count_csv_rows(path: Path) -> int:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        count = sum(1 for _ in csv.reader(handle, delimiter=delimiter))
    return max(0, count - 1)


def count_artifact_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return count_csv_rows(path)
    if suffix == ".parquet":
        try:
            parquet = importlib.import_module("pyarrow.parquet")
        except Exception as exc:
            raise RuntimeError(
                f"cannot read Parquet {path}: pyarrow is unavailable ({exc})"
            ) from exc
        return int(parquet.ParquetFile(path).metadata.num_rows)
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    raise ValueError(f"unsupported artifact format for row count: {path}")


def parse_data_config(path: Path) -> list[dict[str, Any]]:
    payload = load_json_object(path)
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(f"{path} must contain a non-empty ordered 'experiments' list")

    experiments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_experiment in enumerate(raw_experiments):
        if not isinstance(raw_experiment, dict):
            raise ValueError(f"experiments[{index}] in {path} must be an object")
        experiment_id = raw_experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError(f"experiments[{index}] has no non-empty experiment_id")
        if experiment_id in seen_ids:
            raise ValueError(f"duplicate experiment_id in {path}: {experiment_id}")
        seen_ids.add(experiment_id)

        raw_artifacts = raw_experiment.get("artifacts")
        if not isinstance(raw_artifacts, dict):
            raise ValueError(f"experiment {experiment_id} has no 'artifacts' object")
        artifacts: dict[str, Path] = {}
        for artifact in REQUIRED_DATA_ARTIFACTS:
            raw_value = raw_artifacts.get(artifact)
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(
                    f"experiment {experiment_id} has no explicit {artifact} path"
                )
            artifact_path = Path(raw_value)
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            artifacts[artifact] = artifact_path
        experiments.append(
            {"experiment_id": experiment_id, "artifacts": artifacts}
        )
    return experiments


def collect_data_status(config_path: Path, cache_path: Path) -> dict[str, Any]:
    experiments = parse_data_config(config_path)
    cache = load_data_cache(cache_path)
    sources: list[dict[str, Any]] = []
    cache_key_sources: list[dict[str, Any]] = []

    for experiment in experiments:
        experiment_id = experiment["experiment_id"]
        for artifact in REQUIRED_DATA_ARTIFACTS:
            path = experiment["artifacts"][artifact]
            record: dict[str, Any] = {
                "experiment_id": experiment_id,
                "artifact": artifact,
                "path": str(path.resolve()),
                "available": path.is_file(),
            }
            if path.is_file():
                try:
                    record["sha256"] = source_hash(path, cache)
                except OSError as exc:
                    record["error"] = f"hash failed: {exc}"
            cache_key_sources.append(
                {
                    "experiment_id": experiment_id,
                    "artifact": artifact,
                    "path": record["path"],
                    "sha256": record.get("sha256"),
                    "available": record["available"],
                    "error": record.get("error"),
                }
            )
            sources.append(record)

    cache_key = hashlib.sha256(
        json.dumps(cache_key_sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cached_result = cache["results"].get(cache_key)
    if cached_result is not None:
        result = dict(cached_result)
        result["cache_hit"] = True
    else:
        for record in sources:
            if not record["available"] or record.get("error"):
                continue
            try:
                record["rows"] = count_artifact_rows(Path(record["path"]))
            except Exception as exc:
                record["error"] = f"row count failed: {type(exc).__name__}: {exc}"
        result = {
            "config": str(config_path.resolve()),
            "cache_key": cache_key,
            "sources": sources,
            "cache_hit": False,
        }
        cache["results"][cache_key] = {
            key: value for key, value in result.items() if key != "cache_hit"
        }
    write_json_atomic(cache_path, cache)
    result["ok"] = all(
        source["available"] and not source.get("error")
        for source in result["sources"]
    )
    return result


def markdown_code_block(lines: Iterable[str]) -> list[str]:
    values = list(lines)
    if not values:
        return []
    return ["```text", *values, "```", ""]


def render_git_section(status: dict[str, Any]) -> list[str]:
    dirty = status["dirty"]
    lines = [
        "## Git",
        "",
        f"- Branch: `{status['branch']}`",
        f"- HEAD: `{status['head']}`",
        f"- Dirty tree: **{'yes' if dirty else 'no'}** ({len(dirty)} entries)",
        f"- Upstream: `{status['upstream'] or 'none'}`",
    ]
    if status["unpushed_error"]:
        lines.append(f"- Unpushed commits: unavailable ({status['unpushed_error']})")
    else:
        lines.append(f"- Unpushed commits: **{len(status['unpushed'])}**")
    lines.append("")
    if status["unpushed"]:
        lines.extend(markdown_code_block(status["unpushed"]))
    if dirty:
        lines.extend(["Dirty entries:", ""])
        lines.extend(markdown_code_block(dirty))
    lines.extend(
        [
            "| Commit of interest | Revision | Ancestor of HEAD | Ancestor of `origin/main` |",
            "|---|---|---:|---:|",
        ]
    )
    for commit in status["commits"]:
        lines.append(
            f"| {commit['label']} | `{commit['revision']}` | {commit['head']} | "
            f"{commit['origin_main']} |"
        )
    lines.append("")
    return lines


def render_test_section(results: dict[str, Any]) -> list[str]:
    counts = Counter(results.get("outcomes", {}).values())
    lines = [
        "## Tests",
        "",
        f"- Command: `{' '.join(results['command'][:-1])} <status-json>`",
        f"- Duration: **{results['duration_seconds']:.2f}s** "
        f"(limit {TEST_TIMEOUT_SECONDS:.1f}s)",
        f"- Result: **{'PASS' if results['exit_code'] == 0 else 'FAIL'}** "
        f"(pytest exit {results['exit_code']})",
        f"- Counts: **{counts['passed']} passed · {counts['failed']} failed · "
        f"{counts['skipped']} skipped · {counts['xfailed']} xfailed · "
        f"{counts['xpassed']} xpassed · {counts['not_run']} not run**",
        "",
    ]
    if results.get("timed_out"):
        lines.extend([f"> Pytest exceeded the {TEST_TIMEOUT_SECONDS:.1f}s status budget.", ""])
    if results.get("collection_errors"):
        lines.extend(
            [f"> Collection errors: {len(results['collection_errors'])}", ""]
        )
    if results["exit_code"] != 0 and results.get("output"):
        output_lines = results["output"].splitlines()[-40:]
        lines.extend(["Pytest output (last 40 lines):", ""])
        lines.extend(markdown_code_block(output_lines))
    return lines


def render_decision_section(
    decision_ids: Sequence[str],
    correlation: dict[str, Any],
    ledger_path: Path,
    ledger_warning: str | None,
) -> list[str]:
    unverified = correlation["unverified"]
    unverified_text = ", ".join(unverified) if unverified else "none"
    lines = [
        "## Decisions",
        "",
        f"- Ledger: `{ledger_path.relative_to(REPO_ROOT)}`",
        "",
        f"> **UNVERIFIED DECISIONS: {unverified_text}**",
        "",
        f"**{len(decision_ids)} decisions · {len(correlation['with_tests'])} with tests "
        f"({len(correlation['passing'])} passing) · unverified: [{unverified_text}]**",
        "",
    ]
    if ledger_warning:
        lines.extend([f"> **Ledger divergence:** {ledger_warning}", ""])
    if correlation["unknown_markers"]:
        lines.extend(
            [
                "> Decision markers absent from the ledger: "
                + ", ".join(correlation["unknown_markers"]),
                "",
            ]
        )
    lines.extend(["| Decision | Marked tests | Verification |", "|---|---:|---|"])
    for decision_id in decision_ids:
        tests = correlation["tests_by_decision"].get(decision_id, [])
        if not tests:
            result = "unverified"
        elif all(test["outcome"] == "passed" for test in tests):
            result = "passing"
        else:
            result = ", ".join(sorted({test["outcome"] for test in tests}))
        lines.append(f"| {decision_id} | {len(tests)} | {result} |")
    lines.append("")
    if correlation["with_tests"]:
        lines.extend(["Marked tests:", ""])
        for decision_id in correlation["with_tests"]:
            for test in correlation["tests_by_decision"][decision_id]:
                lines.append(
                    f"- {decision_id}: `{test['nodeid']}` — {test['outcome']}"
                )
        lines.append("")
    return lines


def render_environment_section(environment: dict[str, Any]) -> list[str]:
    lines = [
        "## Environment",
        "",
        f"- Python: `{environment['python']}` at `{environment['executable']}`",
        f"- `pyarrow` imports: **{environment['pyarrow_import']}**",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for package, version in environment["packages"].items():
        lines.append(f"| `{package}` | `{version}` |")
    lines.append("")
    return lines


def render_data_section(
    *, requested: bool, result: dict[str, Any] | None, error: str | None
) -> list[str]:
    lines = ["## Data", ""]
    if not requested:
        return [*lines, "Skipped (`--with-data` was not supplied).", ""]
    if error:
        return [*lines, f"> **Data status failed:** {error}", ""]
    assert result is not None
    available = sum(1 for source in result["sources"] if source["available"])
    lines.extend(
        [
            f"- Config: `{result['config']}`",
            f"- Source cache key: `{result['cache_key']}`",
            f"- Cache hit: **{'yes' if result['cache_hit'] else 'no'}**",
            f"- Availability: **{available}/{len(result['sources'])} artifacts**",
            "",
            "| Experiment | Artifact | Available | Rows | SHA-256 | Error |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for source in result["sources"]:
        sha = source.get("sha256", "—")
        if sha != "—":
            sha = f"`{sha[:12]}…`"
        lines.append(
            f"| `{source['experiment_id']}` | `{source['artifact']}` | "
            f"{'yes' if source['available'] else 'no'} | {source.get('rows', '—')} | "
            f"{sha} | {source.get('error', '')} |"
        )
    lines.append("")
    return lines


def write_status(markdown: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(markdown)
    temporary.replace(STATUS_PATH)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    git_status = collect_git_status()
    test_results = run_tests()
    environment = collect_environment()

    ledger_error: str | None = None
    ledger_path = DECISIONS_PATH
    ledger_warning: str | None = None
    try:
        ledger_path, decision_ids, ledger_warning = resolve_decision_ledger()
        correlation = correlate_decisions(decision_ids, test_results)
    except (OSError, ValueError) as exc:
        ledger_error = str(exc)
        decision_ids = []
        correlation = {
            "tests_by_decision": {},
            "with_tests": [],
            "passing": [],
            "unverified": [],
            "unknown_markers": [],
        }

    data_result: dict[str, Any] | None = None
    data_error: str | None = None
    if args.with_data:
        if args.data_config is None:
            data_error = (
                "--with-data requires --data-config or MORPHSEQ_STATUS_DATA_CONFIG; "
                "artifact paths are never inferred"
            )
        else:
            try:
                data_result = collect_data_status(args.data_config, args.data_cache)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                data_error = f"{type(exc).__name__}: {exc}"

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Core-model status",
        "",
        f"Generated `{generated_at}` by `scripts/status.py`; do not edit by hand.",
        "",
    ]
    lines.extend(render_git_section(git_status))
    lines.extend(render_test_section(test_results))
    if ledger_error:
        lines.extend(
            ["## Decisions", "", f"> **Decision status failed:** {ledger_error}", ""]
        )
    else:
        lines.extend(
            render_decision_section(
                decision_ids,
                correlation,
                ledger_path,
                ledger_warning,
            )
        )
    lines.extend(render_environment_section(environment))
    lines.extend(
        render_data_section(
            requested=args.with_data,
            result=data_result,
            error=data_error,
        )
    )
    lines.extend(
        [
            "## Generator",
            "",
            f"- Total no-data runtime: **{time.perf_counter() - started:.2f}s**"
            if not args.with_data
            else f"- Total runtime: **{time.perf_counter() - started:.2f}s**",
            "",
        ]
    )
    write_status("\n".join(lines))

    tests_ok = test_results["exit_code"] == 0
    decisions_ok = ledger_error is None
    data_ok = not args.with_data or (
        data_error is None and data_result is not None and data_result["ok"]
    )
    return 0 if tests_ok and decisions_ok and data_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
