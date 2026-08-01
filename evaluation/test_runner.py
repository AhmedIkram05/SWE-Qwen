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
import os
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
    .apt_install(
        "git",
        "build-essential",  # gcc, g++, make for C extensions
        "libffi-dev",  # cffi, cryptography
        "libssl-dev",  # cryptography, pyOpenSSL
        "pkg-config",  # find libraries
        "libfreetype6-dev",  # matplotlib
        "libpng-dev",  # matplotlib
        "libjpeg-dev",  # matplotlib
        "zlib1g-dev",  # matplotlib
    )
    .pip_install(
        # Test runner deps
        "pytest>=7.0,<8.0",  # <8: old repos import _pytest.monkeypatch.notset (removed in 8)
        "pytest-timeout>=2.3",
        "pytest-json-report>=1.5",
        "gitpython>=3.1",
        "unidiff>=0.7",
        "pydantic>=2.10.0",
        "pydantic-settings>=2.7.0",
        # Heavy common deps pre-installed to avoid per-repo compile (scikit-learn needs these)
        "numpy",
        "scipy",
        "pandas",
        "joblib",
        "threadpoolctl",
        "scikit-learn",
        # Build deps
        "setuptools>=65.0",
        "wheel",
    )
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

_GIT_TIMEOUT_SECONDS = 600  # Django checkout of old commits can take 5+ min on CPU


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

    pytest ``-k`` is a boolean expression on test IDs. Full node IDs
    (``path/to/file.py::test_name``) don't work because ``:`` is not a
    valid ``-k`` operator. Strip the file path and keep only the test
    function name (plus any parametrize suffix).

    Names containing other special characters are wrapped in double quotes
    with backslash-escapes.
    """
    # Strip file path from full node ID: ``tests/foo.py::test_bar`` → ``test_bar``
    if "::" in name:
        name = name.split("::", 1)[1]
    if any(c in name for c in " \t'\"()[]{}&|!~,;:"):
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
            if name == nodeid or nodeid.endswith(name) or nodeid in name:
                by_name[name] = _attempt_from_report_test(test)
                used.add(idx)
                break
        else:
            logger.debug(
                "_attempts_from_report: unmatched name=%r (nodeids: %d)", name, len(nodeids)
            )
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


def _run_pytest_once(  # noqa: PLR0912
    repo_path: Path,
    test_names: list[str],
    timeout: int,
    python_cmd: list[str] | None = None,
) -> tuple[dict[str, _Attempt], list[str]]:
    """Run pytest once in ``repo_path`` and return per-name attempts.

    ``test_names`` empty means "run the full suite" (results keyed by node ID).
    The subprocess bound is capped below the Modal function timeout so the
    function can return gracefully instead of being killed.

    ``python_cmd`` overrides the interpreter (e.g. ``["conda", "run", "-n",
    "testbed", "python"]`` on official SWE-bench images).  Defaults to the
    venv/python heuristic below.
    """
    import time

    start_time = time.time()

    # ponytail: derive venv python from repo_path (mirrors _install_repo logic)
    # Search for .venv created by _install_repo.  The venv lives at:
    #   - {cache_dir}/{repo_name}/.venv  (cache_dir=/test_cache on Modal,
    #     or {repos_dir} on local).  On local the venv is a SIBLING of the
    #     repo dir, not in its parent chain.
    if python_cmd is None:
        python_cmd = [sys.executable]
        _venv_candidates: list[Path] = []
        _p = repo_path.parent
        for _ in range(3):
            _venv_candidates.append(_p / ".venv" / "bin" / "python")
            _p = _p.parent
        # Also try {grandparent}/{repo_name}/.venv (local nested repos like
        # sphinx-doc/sphinx where venv is at repos_dir/sphinx/.venv)
        gp = repo_path.parent.parent
        _venv_candidates.append(gp / repo_path.name / ".venv" / "bin" / "python")
        # Modal convention
        _venv_candidates.append(Path("/test_cache") / repo_path.name / ".venv" / "bin" / "python")
        for _c in _venv_candidates:
            if _c.exists():
                python_cmd = [str(_c)]
                break

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)
    cmd = [
        *python_cmd,
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
        # ponytail: per-process cache dir — 16 containers sharing one
        # pytest-cache dir contended on the volume lock; pid is unique per container
        cmd += ["-o", f"cache_dir=/test_cache/pytest-cache-{os.getpid()}"]
    subprocess_timeout = min(max(120, timeout * max(len(test_names) * 3, 12) + 60), 240)

    # Add a safety margin to the timeout
    safety_margin = 30
    effective_timeout = subprocess_timeout - safety_margin
    if effective_timeout <= 0:
        effective_timeout = subprocess_timeout // 2

    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "pytest timed out after %ss in %s",
            effective_timeout,
            repo_path,
            exc_info=True,
        )
        return ({n: _errored_attempt("pytest invocation timed out") for n in test_names}, [])
    finally:
        try:
            report_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to cleanup report file %s: %s", report_path, e)

    # Check if we're taking too long overall
    elapsed = time.time() - start_time
    modal_timeout_warn = 250
    if elapsed > modal_timeout_warn:  # Close to Modal function timeout
        logger.warning("pytest execution took %.1fs, approaching limit", elapsed)

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
    python_cmd: list[str] | None = None,
) -> list[TestResult]:
    """Run the given tests with retries and return per-test results.

    Args:
        repo_path: Checked-out repository root.
        test_names: Pytest node IDs (or name substrings) to run. Empty means
            run the full suite once (baseline sanity).
        timeout: Per-test timeout in seconds (``pytest --timeout``).
        max_retries: Extra runs for tests that failed/errored (flaky detection).
        python_cmd: Interpreter override (see ``_run_pytest_once``).

    Returns:
        One ``TestResult`` per requested test; status comes from
        ``classify_test_outcomes`` across attempts.
    """
    import time

    start_time = time.time()

    from evaluation.schema import TestResult

    names = [n for n in test_names if n]
    full_suite = not names
    attempts: dict[str, list[_Attempt]] = {}

    # Add a maximum iteration limit to prevent infinite loops
    max_iterations = max_retries + 5  # Allow some extra iterations

    for iteration_count, round_idx in enumerate(range(max_retries + 1), start=1):
        if iteration_count > max_iterations:
            logger.warning(
                "Exceeded maximum iterations (%d) in collect_test_results for %s",
                max_iterations,
                repo_path,
            )
            break

        # Check if we're taking too long overall
        elapsed = time.time() - start_time
        collect_timeout_warn = 200
        if elapsed > collect_timeout_warn:  # Close to Modal function timeout
            logger.warning("collect_test_results taking too long (%.1fs), truncating", elapsed)
            break

        if full_suite:
            if round_idx > 0:
                break
            run_names: list[str] = []
        elif round_idx == 0:
            # First iteration: run all requested tests
            run_names = names
        else:
            # Subsequent iterations: run only failed/errored tests
            pending = [
                n for n, a in attempts.items() if not a or a[-1]["status"] in ("failed", "errored")
            ]
            if not pending:
                break
            run_names = pending
        per_run, nodeids = _run_pytest_once(repo_path, run_names, timeout, python_cmd=python_cmd)
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
    # Remove stale lock files from crashed containers (Modal volumes persist state)
    lock = repo_dir / ".git" / "index.lock"
    lock.unlink(missing_ok=True)
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
    """Revert working tree to ``base_sha`` (drops applied patches and stray files).

    Raises RuntimeError if the reset fails (e.g. ``base_sha`` not present in
    this checkout's history) — a silent no-op would evaluate the wrong state.
    """
    proc = _run_git(repo_dir, "reset", "--hard", base_sha)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git reset --hard {base_sha} failed in {repo_dir}: {proc.stderr.strip()[:300]}"
        )
    _run_git(repo_dir, "clean", "-fd")


def _install_repo(repo_dir: Path, timeout: int = 900, cache_dir: str | Path | None = None) -> None:
    """Install the repo package so tests can import it.

    Uses a cached venv (default ``/test_cache`` for Modal, configurable for
    local execution) to avoid rebuilding on every run.  Tries pyproject.toml,
    setup.py, setup.cfg.  Idempotent and fast on retry.

    Args:
        repo_dir: Checked-out repository root.
        timeout: Timeout for pip install in seconds.
        cache_dir: Directory to store cached venvs.  Defaults to ``/test_cache``.
    """
    # ponytail: cache venv in persistent volume so pip install runs once per repo
    test_cache = Path(cache_dir or "/test_cache")
    repo_name = repo_dir.name
    venv_dir = test_cache / repo_name / ".venv"
    marker = venv_dir / (repo_dir.name + ".installed")

    if marker.exists():
        logger.info("reusing cached venv for %s", repo_dir)
        sys.prefix = str(venv_dir)
        # activate venv for subsequent subprocess calls
        _activate_venv(venv_dir)
        return

    if not venv_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            timeout=30,
            check=True,
        )

    pip = venv_dir / "bin" / "pip"
    proc = subprocess.run(
        [str(pip), "install", "-e", str(repo_dir), "--quiet", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode == 0:
        logger.info("pip install -e . succeeded for %s", repo_dir)
    else:
        setup_py = repo_dir / "setup.py"
        if setup_py.exists():
            python = venv_dir / "bin" / "python"
            subprocess.run(
                [str(python), "setup.py", "develop", "--quiet"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=repo_dir,
            )
        logger.warning(
            "repo installation may be incomplete for %s: %s", repo_dir, proc.stderr[:200]
        )

    marker.touch()
    _activate_venv(venv_dir)


def _activate_venv(venv_dir: Path) -> None:
    """Prepend venv bin to PATH for subsequent subprocess calls."""
    venv_bin = str(venv_dir / "bin")
    env_path = os.environ.get("PATH", "")
    os.environ["PATH"] = venv_bin + os.pathsep + env_path


# ── SWE-bench official images ────────────────────────────────────────────────


def munge_instance_id(instance_id: str) -> str:
    """SWE-bench Docker-tag safety: ``django__django-10554`` -> ``django_1776_django-10554``."""
    return instance_id.replace("__", "_1776_")


def swebench_image(instance_id: str) -> modal.Image:
    """Official prebuilt per-instance SWE-bench eval image.

    ``swebench/sweb.eval.x86_64.<instance>:latest`` has ``/testbed`` checked
    out at the instance's base commit (plus a marker "SWE-bench" commit on
    top) and a conda env ``testbed`` with the project already installed —
    zero clone, zero pip install at eval time.
    """
    return modal.Image.from_registry(
        f"swebench/sweb.eval.x86_64.{munge_instance_id(instance_id)}:latest"
    )


def _execute_instance(  # noqa: PLR0913, PLR0917
    repo_dir: Path,
    base_sha: str,
    test_patch: str | None,
    generated_patch: str | None,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout: int,
    max_retries: int,
    python_cmd: list[str] | None = None,
    reset_first: bool = True,
) -> dict[str, Any]:
    """Run the full per-instance eval test sequence against ``repo_dir``.

    Assumes ``repo_dir`` is a git checkout whose history contains
    ``base_sha`` (swebench image or cloned/installed repo).  Sequence:

    1. Hard-reset to ``base_sha`` (drops any applied patches).
    2. ``tests_before``: selected tests at base state (no retries).
    3. Apply ground-truth ``test_patch`` -> ``tests_head`` -> ground truth.
    4. Reset again; apply ``generated_patch`` -> ``tests_after`` (with retries).

    Returns the same result-dict shape as ``run_tests_in_container``.
    """
    from evaluation.metrics import compute_f2p
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult

    fail_to_pass = fail_to_pass or []
    pass_to_pass = pass_to_pass or []
    test_names = [*fail_to_pass, *pass_to_pass]

    if reset_first:
        _reset_to_base(repo_dir, base_sha)

    # ponytail: baseline runs use max_retries=0 — flaky detection only matters for the final eval
    tests_before = collect_test_results(
        repo_dir, test_names, timeout=timeout, max_retries=0, python_cmd=python_cmd
    )

    tests_head: list[TestResult] = []
    ground_truth: dict[str, Any] = {}
    error: str | None = None
    if test_patch:
        patch_result = apply_patch(repo_dir, test_patch, base_sha, skip_checkout=True)
        if patch_result.success:
            tests_head = collect_test_results(
                repo_dir, test_names, timeout=timeout, max_retries=0, python_cmd=python_cmd
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
                # Env drift / broken image: the test patch should make ALL
                # F2P tests pass.  Without this the instance would score as a
                # model failure indistinguishable from a genuine regression.
                error = "ground truth F2P<100% (env drift or missing image?)"
                logger.error("%s for %s", error, repo_dir)
        else:
            logger.warning("test_patch application failed for %s: %s", repo_dir, patch_result.error)
    else:
        logger.info("no test_patch for %s — skipping ground truth verification", repo_dir)

    _reset_to_base(repo_dir, base_sha)

    if error is not None:
        # Ground truth broken → generated-patch tests are meaningless; short-circuit.
        return {
            "repo": repo_dir.name if str(repo_dir) == "/testbed" else str(repo_dir),
            "base_sha": base_sha,
            "tests_before": [t.model_dump() for t in tests_before],
            "tests_head": [t.model_dump() for t in tests_head],
            "tests_after": [],
            "patch_application": {},
            "ground_truth": ground_truth,
            "error": error,
        }

    tests_after: list[TestResult] = []
    if generated_patch:
        patch_result = apply_patch(repo_dir, generated_patch, base_sha)
        if patch_result.success:
            tests_after = collect_test_results(
                repo_dir,
                test_names,
                timeout=timeout,
                max_retries=max_retries,
                python_cmd=python_cmd,
            )
        else:
            logger.warning(
                "generated patch application failed for %s: %s",
                repo_dir,
                patch_result.error,
            )
    else:
        patch_result = PatchApplicationResult(
            success=False,
            method_used="failed",
            error="no generated patch provided",
        )
        logger.info("no generated patch for %s — skipping patch tests", repo_dir)

    return {
        "repo": repo_dir.name if str(repo_dir) == "/testbed" else str(repo_dir),
        "base_sha": base_sha,
        "tests_before": [t.model_dump() for t in tests_before],
        "tests_head": [t.model_dump() for t in tests_head],
        "tests_after": [t.model_dump() for t in tests_after],
        "patch_application": patch_result.model_dump(),
        "ground_truth": ground_truth,
        "error": error,
    }


# ── Modal function ────────────────────────────────────────────────────────────


@app.function(
    image=BASE_IMAGE,
    volumes={"/repo_cache": repo_volume, "/test_cache": test_volume},
    timeout=1800,  # 30 min: heavy repos (scikit-learn) need 10-15 min for pip install -e .
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
        _install_repo(repo_dir)
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

    # ponytail: baseline runs use max_retries=0 — flaky detection only matters for the final eval
    tests_before = collect_test_results(repo_dir, test_names, timeout=timeout, max_retries=0)

    tests_head: list[TestResult] = []
    ground_truth: dict[str, Any] = {}
    error: str | None = None
    if test_patch:
        # Caller already checked out base_sha; skip the redundant checkout
        patch_result = apply_patch(repo_dir, test_patch, base_sha, skip_checkout=True)
        if patch_result.success:
            _install_repo(repo_dir)
            # ponytail: ground truth also uses max_retries=0 — flaky detection is for eval only
            tests_head = collect_test_results(repo_dir, test_names, timeout=timeout, max_retries=0)
            f2p, p2p, _f2p_count, _p2p_count = compute_f2p(
                tests_before, tests_head, fail_to_pass, pass_to_pass
            )
            ground_truth = {
                "f2p": f2p,
                "p2p": p2p,
                "warning": f2p < _GT_F2P_THRESHOLD,
            }
            if f2p < _GT_F2P_THRESHOLD:
                error = "ground truth F2P<100% (env drift or install incomplete?)"
                logger.error("%s for %s", error, repo)
        else:
            logger.warning("test_patch application failed for %s: %s", repo, patch_result.error)
    else:
        logger.info("no test_patch for %s — skipping ground truth verification", repo)

    _reset_to_base(repo_dir, base_sha)

    if error is not None:
        return {
            "repo": repo,
            "base_sha": base_sha,
            "tests_before": [t.model_dump() for t in tests_before],
            "tests_head": [t.model_dump() for t in tests_head],
            "tests_after": [],
            "patch_application": {},
            "ground_truth": ground_truth,
            "error": error,
        }

    tests_after: list[TestResult] = []
    if generated_patch:
        patch_result = apply_patch(repo_dir, generated_patch, base_sha)
        if patch_result.success:
            _install_repo(repo_dir)
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
        "error": error,
    }


# ── Batch Modal function ──────────────────────────────────────────────────


def _run_swebench_instance_body(  # noqa: PLR0913, PLR0917
    instance_id: str,
    base_sha: str,
    test_patch: str | None = None,
    generated_patch: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    timeout: int = 30,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Execute one SWE-bench instance inside an official swebench image.

    ``swebench/sweb.eval.x86_64.<instance>:latest`` ships ``/testbed`` (the
    repo with full git history, checked out at the instance base commit plus a
    marker "SWE-bench" commit on top) and a conda env ``testbed`` with the
    project already installed.  No clone, no pip install at eval time.

    Registered once per REPO by ``swebench_fn`` (one image per repo is enough:
    every instance's base commit exists in the image's git history, so
    ``git reset --hard`` + the ground-truth check cover per-instance state).
    """
    repo_dir = Path("/testbed")

    # ponytail: bake pytest-json-report/pytest-timeout into a custom image if
    # this per-container install ever dominates (it's ~10-20s, parallel across
    # instances, so leave it)
    try:
        probe = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "testbed",
                "python",
                "-c",
                "import pytest_jsonreport, pytest_timeout",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,  # we check returncode below
        )
        if probe.returncode != 0:
            subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "testbed",
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "pytest-json-report",
                    "pytest-timeout",
                ],
                timeout=180,
                check=True,
            )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        logger.exception("test plugin setup failed for %s", instance_id)
        return {
            "repo": instance_id,
            "base_sha": base_sha,
            "error": f"plugin setup failed: {exc}",
            "tests_before": [],
            "tests_head": [],
            "tests_after": [],
            "patch_application": {},
            "ground_truth": {},
        }

    result = _execute_instance(
        repo_dir,
        base_sha,
        test_patch,
        generated_patch,
        fail_to_pass or [],
        pass_to_pass or [],
        timeout=timeout,
        max_retries=max_retries,
        python_cmd=["conda", "run", "-n", "testbed", "python"],
    )
    result["repo"] = instance_id  # /testbed name is useless; carry the real id
    return result


_swebench_fns: dict[str, Any] = {}


def swebench_fn(repo: str, instance_id: str) -> Any:
    """Lazily register ONE Modal function per repo on an official swebench image.

    Modal 1.5.x cannot swap ``image`` per call (``with_options`` has no image
    param), and images are per-instance, so we register one function per repo
    on first use — taking any instance of that repo as the image (full git
    history + editable install make it valid for every instance of the repo).
    """
    fn = _swebench_fns.get(repo)
    if fn is None:
        fn = app.function(
            image=swebench_image(instance_id),
            volumes={"/test_cache": test_volume},
            timeout=900,  # 15 min per instance (conda install of 2 plugins + tests)
            gpu=None,  # CPU only for test execution
        )(_run_swebench_instance_body)
        _swebench_fns[repo] = fn
    return fn


@app.function(
    image=BASE_IMAGE,
    volumes={"/repo_cache": repo_volume, "/test_cache": test_volume},
    timeout=3600,  # 60 min: batch runs multiple repos, each may need 15 min for pip install
    gpu=None,
)
def run_tests_batch(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915
    repo: str,
    base_sha: str,
    test_patch: str | None,
    test_jobs: list[dict[str, Any]],
    timeout: int = 30,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Execute test suites for multiple patches in a single container.

    One container per (repo, base_sha): clone once, checkout once, install
    once, run ``tests_before`` once, then per job: apply ground-truth
    ``test_patch`` (per-job, with a shared-head fast path when all jobs share
    it), verify ground truth, reset, apply the generated patch and run
    ``tests_after``.

    Args:
        repo: GitHub repo ``"owner/name"``.
        base_sha: Commit to evaluate against.
        test_patch: Default ground-truth test changes; jobs may override via
            their own ``test_patch`` key (SWE-bench instances in the same repo
            have distinct test patches).
        test_jobs: Per-job dicts with ``generated_patch``, ``fail_to_pass``,
            ``pass_to_pass``, and optionally ``test_patch`` / ``instance_id``.
        timeout: Per-test timeout in seconds.
        max_retries: Extra attempts for failed/errored tests.

    Returns:
        List of result dicts (same shape as ``run_tests_in_container`` return),
        one per ``test_job``.
    """
    import time

    start_time = time.time()

    from evaluation.metrics import compute_f2p
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult

    repo_dir = Path("/repo_cache") / repo

    try:
        _clone_repo(repo, repo_dir)
        _ensure_checked_out(repo_dir, base_sha)
        _install_repo(repo_dir)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        logger.error("repo preparation failed for %s: %s", repo, exc, exc_info=True)
        error_result = {
            "repo": repo,
            "base_sha": base_sha,
            "error": str(exc),
            "tests_before": [],
            "tests_head": [],
            "tests_after": [],
            "patch_application": {},
            "ground_truth": {},
        }
        return [dict(error_result) for _ in test_jobs]

    # ── Shared: tests_before (same base state for all jobs) ──
    all_test_names: list[str] = []
    for job in test_jobs:
        all_test_names.extend(job.get("fail_to_pass") or [])
        all_test_names.extend(job.get("pass_to_pass") or [])
    all_test_names = list(set(all_test_names))

    # ponytail: baseline runs use max_retries=0 — flaky detection only matters for the final eval
    tests_before = collect_test_results(repo_dir, all_test_names, timeout=timeout, max_retries=0)

    # ── Per-job ground-truth verification ──
    # SWE-bench instances in the same repo have DIFFERENT test patches, so the
    # old "shared tests_head from the first job" logic was wrong for jobs 2+.
    # Fast path: when every job shares one test patch, run tests_head once.
    job_test_patches = [job.get("test_patch") or test_patch or "" for job in test_jobs]
    shared_test_patch = job_test_patches[0] if len(set(job_test_patches)) == 1 else None

    shared_tests_head: list[TestResult] = []
    if shared_test_patch:
        patch_result = apply_patch(repo_dir, shared_test_patch, base_sha, skip_checkout=True)
        if patch_result.success:
            _install_repo(repo_dir)
            # ponytail: ground truth uses max_retries=0 — flaky detection is for eval only
            shared_tests_head = collect_test_results(
                repo_dir, all_test_names, timeout=timeout, max_retries=0
            )
        else:
            logger.warning("test_patch application failed for %s: %s", repo, patch_result.error)

    # ── Per-job: reset, ground truth, apply generated patch, run tests_after ──
    results: list[dict[str, Any]] = []
    for i, job in enumerate(test_jobs):
        # Check if we're taking too long
        elapsed = time.time() - start_time
        # ponytail: fn timeout is 3600s — warn/truncate only when <5 min real margin
        # remains (calibrated to the slowest first container: pip install -e of a
        # heavy repo like scikit-learn takes 10-15 min)
        batch_timeout_warn = 3300  # 3600s fn timeout - 300s cleanup margin
        if elapsed > batch_timeout_warn:
            logger.warning("Approaching fn timeout for %s, truncating remaining jobs", repo)
            remaining_jobs = len(test_jobs) - i
            if remaining_jobs > 0:
                error_result = {
                    "repo": repo,
                    "base_sha": base_sha,
                    "error": "timeout approaching, truncated",
                    "tests_before": [t.model_dump() for t in tests_before],
                    "tests_head": [],
                    "tests_after": [],
                    "patch_application": {},
                    "ground_truth": {},
                }
                results.extend([dict(error_result) for _ in range(remaining_jobs)])
            break

        logger.info(
            "Processing job %d/%d for %s (%.1fs elapsed)", i + 1, len(test_jobs), repo, elapsed
        )
        _reset_to_base(repo_dir, base_sha)

        fail_to_pass = job.get("fail_to_pass") or []
        pass_to_pass = job.get("pass_to_pass") or []
        job_test_names = [*fail_to_pass, *pass_to_pass]

        # Ground-truth verification for THIS job's test patch
        tests_head: list[TestResult] = []
        ground_truth: dict[str, Any] = {}
        job_error: str | None = None
        job_test_patch = job_test_patches[i]
        if job_test_patch:
            if shared_tests_head:
                tests_head = shared_tests_head
            else:
                patch_result = apply_patch(repo_dir, job_test_patch, base_sha, skip_checkout=True)
                if patch_result.success:
                    _install_repo(repo_dir)
                    tests_head = collect_test_results(
                        repo_dir, job_test_names, timeout=timeout, max_retries=0
                    )
                else:
                    logger.warning(
                        "test_patch application failed for %s: %s", repo, patch_result.error
                    )
            if tests_head:
                f2p, p2p, _f2p_count, _p2p_count = compute_f2p(
                    tests_before, tests_head, fail_to_pass, pass_to_pass
                )
                ground_truth = {
                    "f2p": f2p,
                    "p2p": p2p,
                    "warning": f2p < _GT_F2P_THRESHOLD,
                }
                if f2p < _GT_F2P_THRESHOLD:
                    job_error = "ground truth F2P<100% (env drift or install incomplete?)"
                    logger.error("%s for %s", job_error, repo)

        _reset_to_base(repo_dir, base_sha)

        if job_error is not None:
            # env is broken — generated-patch tests would score as model failures
            logger.error("Skipping generated-patch tests for %s: %s", repo, job_error)
            results.append(
                {
                    "repo": repo,
                    "base_sha": base_sha,
                    "error": job_error,
                    "tests_before": [t.model_dump() for t in tests_before],
                    "tests_head": [t.model_dump() for t in tests_head],
                    "tests_after": [],
                    "patch_application": {},
                    "ground_truth": ground_truth,
                }
            )
            logger.info("Completed job %d/%d for %s (errored)", i + 1, len(test_jobs), repo)
            continue

        generated_patch = job.get("generated_patch") or ""
        if generated_patch:
            logger.info("Applying generated patch for job %d", i + 1)
            # ponytail: _reset_to_base already ran — skip the redundant checkout
            patch_result = apply_patch(repo_dir, generated_patch, base_sha, skip_checkout=True)
            if patch_result.success:
                logger.info("Generated patch applied successfully, installing repo")
                _install_repo(repo_dir)
                logger.info("Running tests_after for job %d (%d tests)", i + 1, len(job_test_names))
                tests_after = collect_test_results(
                    repo_dir, job_test_names, timeout=timeout, max_retries=max_retries
                )
                logger.info("Completed tests_after for job %d", i + 1)
            else:
                logger.warning(
                    "generated patch application failed for %s: %s",
                    repo,
                    patch_result.error,
                )
                tests_after = []
        else:
            patch_result = PatchApplicationResult(
                success=False,
                method_used="failed",
                error="no generated patch provided",
            )
            tests_after = []

        results.append(
            {
                "repo": repo,
                "base_sha": base_sha,
                "tests_before": [t.model_dump() for t in tests_before],
                "tests_head": [t.model_dump() for t in tests_head],
                "tests_after": [t.model_dump() for t in tests_after],
                "patch_application": patch_result.model_dump(),
                "ground_truth": ground_truth,
            }
        )
        logger.info("Completed job %d/%d for %s", i + 1, len(test_jobs), repo)

    return results
