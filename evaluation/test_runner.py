"""Test execution for the SWE-Qwen evaluation harness.

Runs pytest suites in isolated Modal containers: clones the target repo at
``base_sha``, runs the selected tests before any patch is applied, applies the
ground-truth ``test_patch`` (verifying F2P on the reference head state), then
applies the model-generated patch and collects the final results.

Everything except the ``run_tests_in_container`` Modal function is plain
Python (importable and unit-testable without Modal): ``classify_test_outcomes``
is a pure classifier and ``collect_test_results`` shells out to a local pytest
installation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import modal

if TYPE_CHECKING:
    from evaluation.schema import TestResult

logger = logging.getLogger(__name__)

# Ground-truth F2P must reach this value after applying the reference test patch.
_GT_F2P_THRESHOLD = 1.0

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("swe-qwen-eval-test-runner")

BASE_IMAGE = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install(
        "pytest>=8.0",
        "pytest-timeout>=2.3",
        "pytest-json-report>=1.5",
        "gitpython>=3.1",
        "unidiff>=0.7",
        "pydantic>=2.10.0",
        "pydantic-settings>=2.7.0",
    )
    .apt_install("git")
)


# Persistent volumes: cloned repos and pytest caches survive across containers.
def _get_volumes() -> tuple[modal.Volume, modal.Volume]:
    from evaluation.config import EvalConfig

    config = EvalConfig()
    repo_vol_name = config.modal_volumes.get("repo_cache", "eval-repo-cache")
    test_vol_name = config.modal_volumes.get("test_cache", "eval-test-cache")
    return (
        modal.Volume.from_name(repo_vol_name, create_if_missing=True),
        modal.Volume.from_name(test_vol_name, create_if_missing=True),
    )


repo_volume, test_volume = _get_volumes()

_GIT_TIMEOUT_SECONDS = 300


# ── Outcome classification (pure) ─────────────────────────────────────────────


def classify_test_outcomes(attempts: list[str]) -> Literal["passed", "failed", "flaky", "skipped"]:
    """Classify a test's final status from the status strings of repeated runs.

    Args:
        attempts: Status strings from repeated runs of one test (per attempt).

    Returns:
        ``"passed"`` if all attempts passed; ``"failed"`` if all attempts are
        in ``{failed, errored}``; ``"skipped"`` if all attempts were skipped;
        ``"flaky"`` if statuses changed across attempts (any mix).
    """
    if not attempts:
        return "failed"
    statuses = set(attempts)
    if len(statuses) > 1:
        return "flaky"
    only = statuses.pop()
    if only == "passed":
        return "passed"
    if only == "skipped":
        return "skipped"
    return "failed"


# ── pytest execution (pure, no Modal) ─────────────────────────────────────────

_OUTCOME_TO_STATUS: dict[str, str] = {
    "passed": "passed",
    "failed": "failed",
    "skipped": "skipped",
    "error": "errored",
    "xpassed": "passed",
    "xfailed": "failed",
}


class _Attempt(TypedDict):
    status: str
    duration: float
    output: str


def _errored_attempt(message: str) -> _Attempt:
    return {"status": "errored", "duration": 0.0, "output": message}


_MISSING_ATTEMPT = _errored_attempt("test not collected by pytest")


def _quote_k_name(name: str) -> str:
    """Quote a test name for a pytest ``-k`` expression if it needs it.

    Names are passed through as-is (parametrize suffixes are preserved);
    only names containing characters the expression parser treats specially
    are wrapped in double quotes with backslash-escapes.
    """
    if any(c in name for c in " \t'\"()[]{}&|!~,;"):
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return name


def _build_k_expression(test_names: list[str]) -> str:
    """Build a pytest ``-k`` expression selecting the given test names."""
    return " or ".join(_quote_k_name(n) for n in test_names)


def _failure_text(test: dict[str, Any]) -> str:
    """Extract the most useful failure representation from a JSON report test."""
    for phase in ("call", "setup", "teardown"):
        info = test.get(phase)
        if not isinstance(info, dict):
            continue
        longrepr = info.get("longrepr")
        if longrepr:
            if isinstance(longrepr, dict):
                crash = longrepr.get("reprcrash")
                if isinstance(crash, dict):
                    loc = f"{crash.get('path', '')}:{crash.get('lineno', '')}"
                    return f"{loc}: {crash.get('message', '')}"
                return str(longrepr.get("repr_traceback") or longrepr)
            return str(longrepr)
        crash = info.get("crash")
        if isinstance(crash, dict):
            return str(crash.get("message") or crash)
    return ""


def _attempt_from_report_test(test: dict[str, Any]) -> _Attempt:
    """Map one pytest-json-report test entry to an attempt dict."""
    duration = 0.0
    for phase in ("setup", "call", "teardown"):
        info = test.get(phase)
        if isinstance(info, dict):
            duration += float(info.get("duration", 0.0) or 0.0)
    outcome = str(test.get("outcome", "error"))
    status = _OUTCOME_TO_STATUS.get(outcome, "errored")
    return {"status": status, "duration": duration, "output": _failure_text(test)}


def _load_json_report(path: Path) -> dict[str, Any] | None:
    """Load a pytest-json-report file, or None if it is missing/invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _attempts_from_report(
    report: dict[str, Any],
    test_names: list[str],
) -> tuple[dict[str, _Attempt], list[str]]:
    """Attribute JSON report tests back to the requested names.

    Returns a mapping of requested name -> attempt plus the full list of
    collected node IDs (for the ``no tests collected`` warning).
    """
    tests = report.get("tests", []) if isinstance(report.get("tests"), list) else []
    nodeids = [str(t.get("nodeid", "")) for t in tests if isinstance(t, dict)]
    if not nodeids and test_names:
        logger.warning(
            "pytest collected no tests for expression: %s",
            _build_k_expression(test_names),
        )
    by_name: dict[str, _Attempt] = {}
    used: set[int] = set()
    for name in test_names:
        for idx, test in enumerate(tests):
            if idx in used or not isinstance(test, dict):
                continue
            nodeid = str(test.get("nodeid", ""))
            if name == nodeid or nodeid.endswith(name) or name in nodeid:
                by_name[name] = _attempt_from_report_test(test)
                used.add(idx)
                break
    return by_name, nodeids


def _parse_stdout_report(stdout: str) -> dict[str, _Attempt] | None:
    """Fallback: parse ``pytest -rA`` summary lines when the JSON report is missing."""
    attempts: dict[str, _Attempt] = {}
    for line in stdout.splitlines():
        status, _, nodeid = line.partition(" ")
        if not status or not nodeid or status not in _OUTCOME_TO_STATUS:
            continue
        outcome = status
        attempts[nodeid] = {
            "status": _OUTCOME_TO_STATUS[outcome],
            "duration": 0.0,
            "output": line,
        }
    return attempts or None


def _run_pytest_once(
    repo_path: Path,
    test_names: list[str],
    timeout: int,
) -> tuple[dict[str, _Attempt], list[str]]:
    """Run pytest once in ``repo_path`` and return per-name attempts.

    ``test_names`` empty means "run the full suite" (results keyed by node ID).
    The subprocess bound is capped below the Modal function timeout so the
    function can return gracefully instead of being killed.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        ".",
        "-q",
        "--tb=short",
        "-rA",
        "--timeout",
        str(timeout),
        "--json-report",
        "--json-report-file",
        str(report_path),
    ]
    if test_names:
        cmd += ["-k", _build_k_expression(test_names)]
    if Path("/test_cache").is_dir():
        cmd += ["-o", "cache_dir=/test_cache/pytest-cache"]
    subprocess_timeout = min(max(120, timeout * max(len(test_names) * 3, 12) + 60), 240)
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "pytest timed out after %ss in %s",
            subprocess_timeout,
            repo_path,
            exc_info=True,
        )
        return ({n: _errored_attempt("pytest invocation timed out") for n in test_names}, [])
    finally:
        report_path.unlink(missing_ok=True)

    report = _load_json_report(report_path)
    if report is None:
        logger.warning("pytest JSON report missing in %s — falling back to stdout parse", repo_path)
        attempts = _parse_stdout_report(proc.stdout)
        if attempts is not None:
            return attempts, list(attempts)
        return ({n: _errored_attempt("pytest produced no report") for n in test_names}, [])
    return _attempts_from_report(report, test_names)


def collect_test_results(
    repo_path: Path,
    test_names: list[str],
    timeout: int = 30,
    max_retries: int = 2,
) -> list[TestResult]:
    """Run the given tests with retries and return per-test results.

    Args:
        repo_path: Checked-out repository root.
        test_names: Pytest node IDs (or name substrings) to run. Empty means
            run the full suite once (baseline sanity).
        timeout: Per-test timeout in seconds (``pytest --timeout``).
        max_retries: Extra runs for tests that failed/errored (flaky detection).

    Returns:
        One ``TestResult`` per requested test; status comes from
        ``classify_test_outcomes`` across attempts.
    """
    from evaluation.schema import TestResult

    names = [n for n in test_names if n]
    full_suite = not names
    attempts: dict[str, list[_Attempt]] = {}
    for round_idx in range(max_retries + 1):
        if full_suite:
            if round_idx > 0:
                break
            run_names: list[str] = []
        else:
            pending = [
                n for n, a in attempts.items() if not a or a[-1]["status"] in ("failed", "errored")
            ]
            if not pending:
                break
            run_names = pending
        per_run, nodeids = _run_pytest_once(repo_path, run_names, timeout)
        if full_suite:
            for nodeid in nodeids:
                attempts.setdefault(nodeid, []).append(per_run[nodeid])
            break
        for name in run_names:
            attempts.setdefault(name, []).append(per_run.get(name, _MISSING_ATTEMPT))

    results: list[TestResult] = []
    for name, name_attempts in attempts.items():
        statuses = [a["status"] for a in name_attempts]
        status: Literal["passed", "failed", "errored", "skipped", "flaky"]
        status = classify_test_outcomes(statuses)
        duration = next(
            (a["duration"] for a in name_attempts if a["duration"] > 0),
            name_attempts[0]["duration"],
        )
        output = name_attempts[-1]["output"]
        results.append(
            TestResult(
                name=name,
                status=status,
                duration=duration,
                output=output,
                retry_count=len(name_attempts) - 1,
            )
        )
    return results


# ── Git helpers ───────────────────────────────────────────────────────────────


def _run_git(cwd: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a git command, optionally raising on non-zero exit."""
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
    return proc


def _clone_repo(repo: str, repo_dir: Path) -> None:
    """Clone ``https://github.com/{repo}`` into ``repo_dir`` if not already cached."""
    if (repo_dir / ".git").is_dir():
        logger.info("repo %s already cached at %s", repo, repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("cloning %s -> %s", repo, repo_dir)
    proc = _run_git(
        repo_dir.parent, "clone", "--quiet", f"https://github.com/{repo}.git", repo_dir.name
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clone of {repo} failed: {proc.stderr.strip()[:300]}")


def _ensure_checked_out(repo_dir: Path, base_sha: str) -> None:
    """Checkout ``base_sha``, fetching from origin first if the commit is unknown."""
    proc = _run_git(repo_dir, "checkout", "--quiet", base_sha)
    if proc.returncode != 0:
        logger.warning(
            "checkout %s failed (%s); fetching and retrying",
            base_sha,
            proc.stderr.strip()[:200],
        )
        _run_git(repo_dir, "fetch", "--quiet", "--all")
        proc = _run_git(repo_dir, "checkout", "--quiet", base_sha)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to checkout {base_sha}: {proc.stderr.strip()[:300]}")


def _reset_to_base(repo_dir: Path, base_sha: str) -> None:
    """Revert working tree to ``base_sha`` (drops applied patches and stray files)."""
    _run_git(repo_dir, "reset", "--hard", base_sha, check=True)
    _run_git(repo_dir, "clean", "-fd", check=True)


# ── Modal function ────────────────────────────────────────────────────────────


@app.function(
    image=BASE_IMAGE,
    volumes={"/repo_cache": repo_volume, "/test_cache": test_volume},
    timeout=300,
    gpu=None,  # CPU only for test execution
)
def run_tests_in_container(  # noqa: PLR0913, PLR0917
    repo: str,
    base_sha: str,
    test_patch: str | None = None,
    generated_patch: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    timeout: int = 30,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Execute the eval test suite for one instance in an isolated container.

    1. Clone ``https://github.com/{repo}`` to ``/repo_cache/{repo}`` if not cached.
    2. Checkout ``base_sha``.
    3. Run the selected tests (``-k`` expression over ``fail_to_pass`` +
       ``pass_to_pass``) -> ``tests_before``.
    4. If ``test_patch``: apply it and re-run -> ``tests_head``.
    5. Compute ground-truth F2P/P2P from ``tests_before``/``tests_head``.
    6. Revert to ``base_sha``; if ``generated_patch``: apply it and re-run ->
       ``tests_after``.

    Args:
        repo: ``"owner/name"`` of the GitHub repository.
        base_sha: Commit to evaluate against (pre-fix state).
        test_patch: Ground-truth test changes (unified diff), or None.
        generated_patch: Model-generated fix patch (unified diff), or None.
        fail_to_pass: Test names that should fail at ``base_sha`` and pass after a fix.
        pass_to_pass: Test names that should keep passing.
        timeout: Per-test timeout in seconds.
        max_retries: Extra attempts for failed/errored tests (flaky detection).

    Returns:
        Dict with ``tests_before``/``tests_head``/``tests_after``
        (serialized ``TestResult`` lists), ``patch_application``
        (serialized ``PatchApplicationResult``) and ``ground_truth``
        (``{"f2p", "p2p", "warning"}``; empty when no ``test_patch``).
        An ``"error"`` key on repo preparation failure.
    """
    from evaluation.metrics import compute_f2p
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult

    fail_to_pass = fail_to_pass or []
    pass_to_pass = pass_to_pass or []
    test_names = [*fail_to_pass, *pass_to_pass]
    repo_dir = Path("/repo_cache") / repo

    try:
        _clone_repo(repo, repo_dir)
        _ensure_checked_out(repo_dir, base_sha)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        logger.error("repo preparation failed for %s: %s", repo, exc, exc_info=True)
        return {
            "repo": repo,
            "base_sha": base_sha,
            "error": str(exc),
            "tests_before": [],
            "tests_head": [],
            "tests_after": [],
            "patch_application": {},
            "ground_truth": {},
        }

    tests_before = collect_test_results(
        repo_dir, test_names, timeout=timeout, max_retries=max_retries
    )

    tests_head: list[TestResult] = []
    ground_truth: dict[str, Any] = {}
    if test_patch:
        patch_result = apply_patch(repo_dir, test_patch, base_sha)
        if patch_result.success:
            tests_head = collect_test_results(
                repo_dir, test_names, timeout=timeout, max_retries=max_retries
            )
            f2p, p2p, _f2p_count, _p2p_count = compute_f2p(
                tests_before, tests_head, fail_to_pass, pass_to_pass
            )
            ground_truth = {
                "f2p": f2p,
                "p2p": p2p,
                "warning": f2p < _GT_F2P_THRESHOLD,
            }
            if f2p < _GT_F2P_THRESHOLD:
                logger.warning(
                    "ground truth F2P=%.2f%% < 100%% for %s "
                    "— test patch may be incomplete or repo state drifted",
                    f2p * 100,
                    repo,
                )
        else:
            logger.warning("test_patch application failed for %s: %s", repo, patch_result.error)
    else:
        logger.info("no test_patch for %s — skipping ground truth verification", repo)

    _reset_to_base(repo_dir, base_sha)

    tests_after: list[TestResult] = []
    if generated_patch:
        patch_result = apply_patch(repo_dir, generated_patch, base_sha)
        if patch_result.success:
            tests_after = collect_test_results(
                repo_dir, test_names, timeout=timeout, max_retries=max_retries
            )
        else:
            logger.warning(
                "generated patch application failed for %s: %s",
                repo,
                patch_result.error,
            )
    else:
        patch_result = PatchApplicationResult(
            success=False,
            method_used="failed",
            error="no generated patch provided",
        )
        logger.info("no generated patch for %s — skipping patch tests", repo)

    return {
        "repo": repo,
        "base_sha": base_sha,
        "tests_before": [t.model_dump() for t in tests_before],
        "tests_head": [t.model_dump() for t in tests_head],
        "tests_after": [t.model_dump() for t in tests_after],
        "patch_application": patch_result.model_dump(),
        "ground_truth": ground_truth,
    }
