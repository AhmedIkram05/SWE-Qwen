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

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


# pytest ``-k`` expression grammar (pytest 7 ``_pytest.mark.expression``)
# has no quoted-string support at all: each token must match the ident
# regex ``(\w|:|\+|\-|\.|\[|\]|\\|/)+``. Any character outside that set
# (space, ``"``, ``'``, ``(``, ``)``, ``{``, ``&``, ``|``, ``!``, ``;``, ...)
# aborts parsing with ``unexpected character`` and pytest exits rc=4, so the
# whole selector fails. The only safe strategy is to keep every character
# inside the ident set and emit bare tokens.
# ponytail: single char under `\w` etc (real names carry no back/quote)
# NOTE: `?` intentionally excluded — pytest -k expression parser rejects it
# ("unexpected character '?'", rc=4) and aborts the WHOLE selector.
_K_IDENT_RE = re.compile(r"[\w\-\+\.:/\[\]]")


def _bare_test_name(name: str) -> str:
    """Normalize a SWE-bench test name to its bare pytest identifier.

    Handles both ``path/to/file.py::test_bar`` (node ID) and the classic
    SWE-bench ``test_bar(SomeTestClass)`` format: strip the file path and the
    parenthesized class suffix. ``test_delete_cookie_samesite(DeleteCookieTests)``
    → ``test_delete_cookie_samesite``.
    """
    if "::" in name:
        name = name.split("::", 1)[1]
    if "(" in name:
        name = name.split("(", 1)[0]
    return name


def _quote_k_name(name: str) -> str:
    """Sanitize one test name into a bare pytest ``-k`` ident token.

    pytest ``-k`` is a boolean expression on test IDs and has no quoting.
    Full node IDs (``path/to/file.py::test_name``) are stripped to the final
    segment, SWE-bench ``name(TestClass)`` suffixes are dropped, then every
    character outside the ident set (from corrupted golden fragments) is
    removed. An empty result contributes nothing to the OR-expression and is
    skipped by the caller.
    """
    name = _bare_test_name(name)
    return "".join(c for c in name if _K_IDENT_RE.match(c))


def _dedup_names(names: list[str]) -> list[str]:
    """Remove names that are truncated parametrize IDs from SWE-bench.

    SWE-bench HF data stores test names from output logs, and some
    parametrize IDs are truncated (e.g. ``test_parse_noqa[noqa:`` vs
    the full ``test_parse_noqa[noqa:-expected3]``).  A valid pytest
    node ID that is parametrized *always* ends with ``]``.  Any name
    that has an opening ``[`` but no trailing ``]`` is a truncated
    prefix and can never match a real collected test.
    """
    return [n for n in names if not ("[" in n and not n.endswith("]"))]


def _build_k_expression(test_names: list[str]) -> str:
    """Build a pytest ``-k`` expression selecting the given test names."""
    tokens = [t for t in (_quote_k_name(n) for n in test_names) if t]
    return " or ".join(tokens)


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
        bare = _bare_test_name(name)
        if not bare:
            continue
        for idx, test in enumerate(tests):
            if idx in used or not isinstance(test, dict):
                continue
            nodeid = str(test.get("nodeid", ""))
            if bare == nodeid or nodeid.endswith(bare) or nodeid in name:
                by_name[name] = _attempt_from_report_test(test)
                used.add(idx)
                break
        else:
            logger.debug(
                "_attempts_from_report: unmatched name=%r (nodeids: %d)", name, len(nodeids)
            )
    if test_names and len(by_name) < len(test_names):
        logger.warning(
            "pytest matched %d/%d requested names in %s; unseen names (first 5): %s; "
            "sample nodeids (first 5): %s",
            len(by_name),
            len(test_names),
            len(nodeids),
            [n for n in test_names if n not in by_name][:5],
            nodeids[:5],
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


def _write_framework_conftest(repo_path: Path) -> None:
    """Write a root ``conftest.py`` for repos whose test suites need a
    framework bootstrap before pytest collection.

    django's suite imports ``django.conf.settings`` at import time; without
    ``DJANGO_SETTINGS_MODULE`` + ``django.setup()`` every test file errors
    with ImproperlyConfigured and nothing collects.  SWE-bench sidesteps this
    by invoking ``tests/runtests.py --settings=test_sqlite``; we keep pytest
    (the shared harness) and inject the equivalent bootstrap.  No-op for
    non-django repos or when the repo already ships a root conftest.
    """
    if not (repo_path / "tests" / "test_sqlite.py").is_file():
        return
    conftest = repo_path / "conftest.py"
    if conftest.exists():
        return
    # runtests.py registers every tests/ subpackage (dir with __init__.py, not
    # in SUBDIRS_TO_SKIP) as a top-level app label, with tests/ on sys.path so
    # ``import sessions_tests`` resolves. Mirror that so test models (e.g.
    # sessions_tests.models.CustomSession) collect under raw pytest.
    skip = {
        "__pycache__",
        "gis_tests",
        "urls",
        "wsgi",
        "runtests",
        "data",
        "import_error_package",
        "test_runner_apps",
    }
    test_labels = sorted(
        d.name
        for d in (repo_path / "tests").iterdir()
        if d.is_dir() and (d / "__init__.py").is_file() and d.name not in skip
    )
    _django_apps = [
        "django.contrib.contenttypes",
        "django.contrib.auth",
        "django.contrib.sites",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.contrib.admin.apps.SimpleAdminConfig",
        "django.contrib.staticfiles",
    ] + test_labels
    conftest.write_text(
        "import os, sys\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.test_sqlite')\n"
        "import django\n"
        "from django.conf import settings\n"
        "# Mirrors tests/runtests.py: put tests/ on sys.path and register\n"
        "# ALWAYS_INSTALLED_APPS + per-test-module app labels, so modules\n"
        "# importing contrib models or test models collect under pytest.\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))\n"
        f"settings.INSTALLED_APPS = list(settings.INSTALLED_APPS) + {_django_apps!r}\n"
        "# Mirrors runtests.py setup(): SITE_ID/ROOT_URLCONF/MIDDLEWARE and\n"
        "# no-migrations for contrib apps, so session/contrib tests run.\n"
        "settings.SITE_ID = 1\n"
        "settings.ROOT_URLCONF = 'urls'\n"
        "settings.MIDDLEWARE = [\n"
        "    'django.contrib.sessions.middleware.SessionMiddleware',\n"
        "    'django.middleware.common.CommonMiddleware',\n"
        "    'django.middleware.csrf.CsrfViewMiddleware',\n"
        "    'django.contrib.auth.middleware.AuthenticationMiddleware',\n"
        "    'django.contrib.messages.middleware.MessageMiddleware',\n"
        "]\n"
        "settings.MIGRATION_MODULES = {'auth': None, 'contenttypes': None, 'sessions': None}\n"
        "django.setup()\n"
        "# Create the test DB like runtests.py's DiscoverRunner: raw pytest\n"
        "# never builds one, so NAME-less sqlite tests error with\n"
        "# ImproperlyConfigured otherwise. In-memory DB, dies with process.\n"
        "try:\n"
        "    from django.test.runner import DiscoverRunner\n"
        "    from django.test.utils import setup_test_environment\n"
        "    setup_test_environment()\n"
        "    DiscoverRunner(verbosity=0, interactive=False).setup_databases()\n"
        "except Exception as exc:  # noqa: BLE001 — degrade: DB tests error, rest run\n"
        "    import sys as _sys\n"
        "    print(f'[conftest] DB bootstrap failed: {exc!r}', file=_sys.stderr)\n"
    )


def _derive_test_files(repo_path: Path, test_names: list[str]) -> list[str]:  # noqa: PLR0912
    """Map SWE-bench test names to repo-relative test file paths.

    Supports two name shapes found in the golden dataset:

    - ``test_x(mod.path.Class)`` (django style) → ``tests/mod/path.py``
    - ``mod/path.py::test_x[...]`` (sqlfluff nodeid style) → ``mod/path.py``

    Plain names (sympy ``test_PythonCodePrinter``) and names that don't
    resolve to an existing file (mangled golden docstrings like
    ``set_cookie()accepts...``) are dropped; the caller falls back to a
    full-dir collect when nothing resolves.

    Passing explicit files instead of ``.`` has two benefits: it bypasses
    pytest's ``python_files`` glob (django's ``tests.py`` matches neither
    ``test_*.py`` nor ``*_test.py``, so dir-walks silently skip it), and it
    cuts collection cost from the whole tree to a handful of files — the
    dominant Modal cost driver at 50-100 instances.
    """
    files: list[str] = []
    seen: set[str] = set()
    plain_names: list[str] = []
    for name in test_names:
        rel: str | None = None
        if "(" in name:
            module = name.split("(", 1)[1].rsplit(".", 1)[0].strip(")").strip()
            cands: list[str] = []
            if module.startswith("tests."):
                cands.append(module.replace(".", "/") + ".py")
            elif module:
                cands.append("tests/" + module.replace(".", "/") + ".py")
                cands.append(module.replace(".", "/") + ".py")
            for cand in cands:
                if (repo_path / cand).is_file():
                    rel = cand
                    break
        elif "::" in name:
            head = name.split("::", 1)[0]
            if head.endswith(".py") and (repo_path / head).is_file():
                rel = head
        else:
            # Plain name like test_PythonCodePrinter — resolve via grep
            # so we don't fall back to the full-dir ".".  (sympy collects
            # ~280s on a full walk; grep resolves it in <1s.)
            plain_names.append(name)
            continue
        if rel and rel not in seen:
            seen.add(rel)
            files.append(rel)

    if plain_names:
        resolved = _derive_files_from_grep(repo_path, plain_names)
        for p in resolved:
            if p not in seen:
                seen.add(p)
                files.append(p)

    return files


def _derive_files_from_grep(repo_path: Path, names: list[str]) -> list[str]:
    """Find files containing ``def <name>`` for each plain test name.

    Falls back gracefully (empty list) when ``git`` is unavailable or the
    repo is not a git worktree — the caller will use a full-dir walk instead.

    Strips parametrize suffixes (``[param]``) and class suffixes
    (``(ClassName)``) before searching — the golden dataset stores
    parametrized names like ``test_foo[a]`` but the actual function
    definition is ``def test_foo(``.
    """
    bare = []
    for n in names:
        b = _bare_test_name(n)
        if "[" in b:
            b = b.split("[", 1)[0]
        if b:
            bare.append(b)
    if not bare:
        return []
    # NOTE: ``\b`` is PCRE-only and not supported by ``git grep -E``.
    # Omitting the word boundary is safe — test names from SWE-bench are
    # unique enough.  The ``\(`` anchor at the end prevents matching a
    # name as a substring of a longer function (e.g. ``test_cookie`` won't
    # match ``test_cookie_settings`` because there's no ``(`` at that position).
    pattern = "|".join(rf"^[[:space:]]*(def |async def ).*{re.escape(n)}\(" for n in bare)
    try:
        r = subprocess.run(
            ["git", "grep", "-l", "-E", pattern, "--", "*.py"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if r.returncode not in (0, 1):
        return []
    if r.returncode == 1:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def _run_pytest_once(  # noqa: PLR0912, PLR0915
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

    _write_framework_conftest(repo_path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)
    # ponytail: collect only the files the requested tests live in — a
    # full-dir walk costs minutes on sympy/django and skips django's
    # tests.py modules (pytest python_files glob). Fall back to "." when
    # no path is derivable (plain sympy names, mangled golden docstrings).
    pytest_paths = _derive_test_files(repo_path, test_names) if test_names else []
    if not pytest_paths:
        logger.warning(
            "no test files derivable from %d names — falling back to full-dir collect (slow)",
            len(test_names),
        )
        pytest_paths = ["."]
    cmd = [
        *python_cmd,
        "-m",
        "pytest",
        *pytest_paths,
        "-q",
        "--tb=short",
        "-rA",
        "--continue-on-collection-errors",
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
    # Large repos (sympy, django) can take >5 min to import and collect.
    # Cap at 900 so big suites actually finish; the Modal function timeout
    # (3600 for batch, 900 swebench) bounds it from the outside.
    subprocess_timeout = min(max(120, timeout * max(len(test_names) * 3, 12) + 60), 900)

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
        report_path.unlink(missing_ok=True)
        return ({n: _errored_attempt("pytest invocation timed out") for n in test_names}, [])

    # Check if we're taking too long overall
    elapsed = time.time() - start_time
    modal_timeout_warn = 700

    if elapsed > modal_timeout_warn:  # Close to pytest function timeout
        logger.warning("pytest execution took %.1fs, approaching limit", elapsed)
    report = _load_json_report(report_path)
    # ponytail: report_path is a NamedTemporaryFile suffixed .json; must exist
    # until read — unlink here, after loading, not in a bare finally.
    try:
        report_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to cleanup report file %s: %s", report_path, e)
    if report is None:
        logger.warning(
            "pytest JSON report missing in %s (rc=%s) — falling back to stdout parse; "
            "stderr tail:\n%s",
            repo_path,
            proc.returncode,
            (proc.stderr or "")[-2000:],
        )
        attempts = _parse_stdout_report(proc.stdout)
        if attempts is not None:
            return attempts, list(attempts)
        return ({n: _errored_attempt("pytest produced no report") for n in test_names}, [])
    attempts, nodeids = _attempts_from_report(report, test_names)
    if not nodeids and test_names:
        # diagnostics: why did pytest collect nothing at all?
        coll_errors = []
        for c in report.get("collectors") or []:
            if not isinstance(c, dict):
                continue
            for r in c.get("result") or []:
                if not isinstance(r, dict):
                    continue
                lp = r.get("longrepr")
                if isinstance(lp, str) and lp not in coll_errors:
                    coll_errors.append(lp)
        logger.warning(
            "pytest collected 0/%d tests in %s (rc=%s); stdout tail:\n%s\nstderr tail:\n%s\n"
            "collector errors:\n%s",
            len(test_names),
            repo_path,
            proc.returncode,
            (proc.stdout or "")[-2000:],
            (proc.stderr or "")[-2000:],
            "\n---\n".join(coll_errors)[-4000:],
        )
    return attempts, nodeids


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
    # Remove names that are strict substrings of another name (corrupted
    # prefix duplicates from SWE-bench HF data truncation).
    names = _dedup_names(names)
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

        # Check if we're taking too long overall (collect can legitimately
        # take 15 min on huge repos like sympy; Modal batch timeout is 3600)
        elapsed = time.time() - start_time
        collect_timeout_warn = 900
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
    # Modal volumes persist between runs: a previously failed patch application
    # leaves the working tree dirty, and "local changes would be overwritten"
    # then fails the checkout for EVERY subsequent instance of the repo.
    _run_git(repo_dir, "reset", "--hard", "HEAD")
    _run_git(repo_dir, "clean", "-fdq")
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

    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    if marker.exists():
        # Rebuild stale venvs created without --system-site-packages (they lack
        # pytest, which only lives in the base image python).  ponytail: check
        # cfg text once, rebuild once, then fast path forever.
        stale = pyvenv_cfg.is_file() and "system_site_packages = true" not in pyvenv_cfg.read_text()
        if stale:
            logger.warning("rebuilding stale venv for %s (missing system site packages)", repo_dir)
            shutil.rmtree(venv_dir, ignore_errors=True)
            marker.unlink(missing_ok=True)
        else:
            logger.info("reusing cached venv for %s", repo_dir)
            sys.prefix = str(venv_dir)
            # activate venv for subsequent subprocess calls
            _activate_venv(venv_dir)
            return

    if not venv_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
            capture_output=True,
            timeout=30,
            check=True,
        )

    # ponytail: skip editable install for pytest source repos (the source
    # checkout registers pytest11 entry points that override the proper
    # pytest installation, causing ``TypeError: required field 'lineno'
    # missing from alias`` during test collection).  Create the marker
    # so we don't retry on the next run.
    if (repo_dir / "src/_pytest").is_dir() or (repo_dir / "_pytest").is_dir():
        logger.info("skipping editable install for %s (is the pytest source)", repo_dir.name)
        marker.touch()
        _activate_venv(venv_dir)
        return

    pip = venv_dir / "bin" / "pip"
    # ponytail: SWE-bench base images have all dependencies pre-installed
    # (venv inherits them via --system-site-packages).  --no-deps avoids
    # re-resolving/installing them — 5-10 min → 2-10 seconds per repo.
    proc = subprocess.run(
        [
            str(pip),
            "install",
            "-e",
            str(repo_dir),
            "--no-deps",
            "--quiet",
            "--disable-pip-version-check",
        ],
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
            "repo installation may be incomplete for %s. stderr tail:\n%s",
            repo_dir,
            (proc.stderr or "")[:4000],
        )

    marker.touch()
    _activate_venv(venv_dir)

    # ponytail: sympy editable install at old cached base (c4e836c) lacks
    # ``equal_valued`` which torch importers in the project venv need.
    # Pin a compatible version so the full test suite doesn't break.
    if (repo_dir / "sympy" / "__init__.py").is_file():
        python = venv_dir / "bin" / "python"
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "sympy==1.13.3"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )


def _activate_venv(venv_dir: Path) -> None:
    """Prepend venv bin to PATH for subsequent subprocess calls."""
    venv_bin = str(venv_dir / "bin")
    env_path = os.environ.get("PATH", "")
    os.environ["PATH"] = venv_bin + os.pathsep + env_path


# ── Baseline cache (T3: reuse tests_before/tests_head across variants) ────────


_BASELINE_CACHE_DIR = Path("/test_cache") / "instance_baselines"
_BASELINE_CACHE_VERSION = 2  # invalidates caches from before test_patch/gold_patch flow
_VERIFIED_DIR = Path("/test_cache") / ".verified"


def _baseline_cache_path(instance_id: str) -> Path:
    """Path to the cached baseline for *instance_id*."""
    return _BASELINE_CACHE_DIR / f"{instance_id}.json"


def _load_baseline_cache(instance_id: str, base_sha: str) -> dict[str, Any] | None:
    """Load cached tests_before/tests_head/ground_truth for *instance_id*.

    Returns None if no cache exists or the cached base_sha doesn't match.
    """
    path = _baseline_cache_path(instance_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _BASELINE_CACHE_VERSION or data.get("base_sha") != base_sha:
            logger.debug("baseline cache stale for %s (version/base_sha mismatch)", instance_id)
            return None
        return data  # noqa: TRY300
    except (OSError, json.JSONDecodeError):
        logger.warning("corrupt baseline cache for %s", instance_id)
        return None


def _save_baseline_cache(instance_id: str, data: dict[str, Any]) -> None:
    """Atomically save baseline cache for *instance_id*."""
    _BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _baseline_cache_path(instance_id)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning("failed to write baseline cache for %s", instance_id)


# ── Repo-verified marker (T4: once-per-repo ground-truth verification) ────


def _verified_marker_path(repo: str) -> Path:
    """Path to the verification marker for *repo* (slashes → underscores)."""
    return _VERIFIED_DIR / repo.replace("/", "_")


def _is_repo_verified(repo: str) -> bool:
    """Return True if this repo's environment has been verified."""
    return _verified_marker_path(repo).is_file()


def _mark_repo_verified(repo: str) -> None:
    """Write the verification marker for *repo*."""
    _VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    _verified_marker_path(repo).touch(exist_ok=True)


# ── SWE-bench official images ────────────────────────────────────────────────


def munge_instance_id(instance_id: str) -> str:
    """SWE-bench Docker-tag safety: ``django__django-10554`` -> ``django_1776_django-10554``."""
    return instance_id.replace("__", "_1776_")


def _swebench_repo_key(instance_id: str) -> str:
    """SWE-bench ``org__repo-<pr>`` -> ``org/repo``.

    Splits on the LAST dash so repos containing dashes (``sphinx-doc__sphinx``)
    survive. Used to key the once-per-repo verification marker: ``/testbed``
    is the same filesystem name in every swebench image, so the directory
    name cannot identify the repo.
    """
    return instance_id.rsplit("-", 1)[0].replace("__", "/")


def swebench_image(instance_id: str) -> modal.Image:
    """Official prebuilt per-instance SWE-bench eval image.

    ``swebench/sweb.eval.x86_64.<instance>:latest`` has ``/testbed`` checked
    out at the instance's base commit (plus a marker "SWE-bench" commit on
    top) and a conda env ``testbed`` with the project already installed —
    zero clone, zero pip install at eval time.

    The bare image ships ONLY that conda env; the harness code itself
    (``evaluation.test_runner`` + its deps) is imported at container
    hydration in the image's base python, so the eval package's own runtime
    deps are pip-installed on top.  The repo's test deps stay in the
    ``testbed`` conda env untouched.
    """
    return modal.Image.from_registry(
        f"swebench/sweb.eval.x86_64.{munge_instance_id(instance_id)}:latest"
    ).pip_install(
        # Eval-package runtime deps for the base python (module-level
        # imports: evaluation.config -> pydantic_settings; body imports:
        # evaluation.schema -> pydantic, patch_applier -> unidiff).
        "pydantic>=2.10.0",
        "pydantic-settings>=2.7.0",
        "unidiff>=0.7",
    )


@functools.lru_cache(maxsize=256)
def _swebench_image_tag_exists(tag: str) -> bool:
    """True if ``swebench/sweb.eval.x86_64.<tag>:latest`` is published on Docker Hub."""
    import urllib.request
    from urllib.error import HTTPError, URLError

    url = f"https://hub.docker.com/v2/repositories/swebench/sweb.eval.x86_64.{tag}/tags/latest"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            return resp.status == 200  # noqa: PLR2004
    except HTTPError as e:
        # A 404 is the only definitive "image does not exist". Docker Hub
        # rate-limits anonymous API calls (429) and flakes on 5xx — failing
        # closed on those silently routed every instance to the dep-starved
        # fallback containers. Modal pulls the image itself, so an unverifiable
        # API response is not a missing image.
        return e.code != 404  # noqa: PLR2004
    except (URLError, OSError):
        return True


def swebench_image_exists(instance_id: str) -> bool:
    """True if the official swebench eval image for this instance is published.

    The harness pre-flights this before registering functions: a single
    missing image (e.g. sqlfluff_1776_sqlfluff-3662) would otherwise fail
    the entire Modal app at image-build time, disabling Modal for the
    process and losing every test run.
    """
    return _swebench_image_tag_exists(munge_instance_id(instance_id))


def _execute_instance(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915
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
    instance_id: str = "",
    verify_mode: str = "all",
    gold_patch: str | None = None,
) -> dict[str, Any]:
    """Run the full per-instance eval test sequence against ``repo_dir``.

    Assumes ``repo_dir`` is a git checkout whose history contains
    ``base_sha`` (swebench image or cloned/installed repo).  Sequence:

    1. Hard-reset to ``base_sha`` (drops any applied patches).
    2. Apply ground-truth ``test_patch`` (skip_checkout) so F2P tests EXIST;
       ``tests_before`` = selected tests at base + test_patch (F2P fail, P2P pass).
       → Uses baseline cache when *instance_id* is non-empty and cache exists.
    3. Apply ``gold_patch`` on top -> ``tests_head`` -> ground truth (F2P PASS).
       → Skips ground truth when *verify_mode* is ``"once_per_repo"`` and the
         repo marker exists (first-verifier already validated the env).
       → Also uses baseline cache when available.
    4. Reset again; re-apply ``test_patch`` then ``generated_patch``
       (both skip_checkout) -> ``tests_after`` (with retries).

    Returns the same result-dict shape as ``run_tests_in_container``.
    """
    from evaluation.metrics import compute_f2p
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult, TestResult

    fail_to_pass = fail_to_pass or []
    pass_to_pass = pass_to_pass or []
    test_names = [*fail_to_pass, *pass_to_pass]
    _t0 = time.time()
    logger.info("── instance %s (%d tests) ──", instance_id or repo_dir.name, len(test_names))
    # ponytail: /testbed is the same name in every swebench image — key the
    # once-per-repo marker on the instance's real repo, not the filesystem name
    if str(repo_dir) == "/testbed":
        repo_name = _swebench_repo_key(instance_id) if instance_id else "testbed"
    else:
        repo_name = str(repo_dir)

    if reset_first:
        _reset_to_base(repo_dir, base_sha)

    # ── Load or compute tests_before + tests_head (baseline cache) ──
    tests_before: list[TestResult] = []
    tests_head: list[TestResult] = []
    ground_truth: dict[str, Any] = {}
    error: str | None = None
    cached_baseline: dict[str, Any] | None = None

    if instance_id:
        cached_baseline = _load_baseline_cache(instance_id, base_sha)

    if cached_baseline is not None:
        # ponytail: reuse cached before/head — identical across variants
        from evaluation.schema import TestResult

        tests_before = [
            TestResult.model_validate(t) for t in (cached_baseline.get("tests_before") or [])
        ]
        tests_head = [
            TestResult.model_validate(t) for t in (cached_baseline.get("tests_head") or [])
        ]
        ground_truth = cached_baseline.get("ground_truth") or {}
        logger.info(
            "baseline cache HIT for %s (%d before, %d head tests)",
            instance_id or repo_name,
            len(tests_before),
            len(tests_head),
        )

    else:
        # ponytail: baseline runs use max_retries=0 — flaky detection only matters for final eval
        test_patch_result = None
        if test_patch:
            test_patch_result = apply_patch(repo_dir, test_patch, base_sha, skip_checkout=True)
            if not test_patch_result.success:
                logger.warning(
                    "test_patch application failed for %s: %s", repo_dir, test_patch_result.error
                )

        tests_before = collect_test_results(
            repo_dir, test_names, timeout=timeout, max_retries=0, python_cmd=python_cmd
        )

        # Ground-truth verification: base + test_patch + gold_patch → F2P PASS
        repo_key = repo_name
        skip_gt = verify_mode == "once_per_repo" and _is_repo_verified(repo_key)
        if test_patch_result is not None and test_patch_result.success and not skip_gt:
            if gold_patch:
                gold_result = apply_patch(repo_dir, gold_patch, base_sha, skip_checkout=True)
                if not gold_result.success:
                    logger.warning(
                        "gold_patch application failed for %s: %s", repo_dir, gold_result.error
                    )
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
                error = "ground truth F2P<100% (env drift or missing image?)"
                logger.error("%s for %s", error, repo_dir)
            # First successful verification → mark repo as verified
            elif verify_mode == "once_per_repo" and not _is_repo_verified(repo_key):
                _mark_repo_verified(repo_key)
                logger.info("repo %s verified (marker written)", repo_key)
        elif skip_gt:
            logger.info(
                "ground truth SKIPPED for %s (repo already verified, verify_mode=%s)",
                repo_name,
                verify_mode,
            )
            ground_truth = {"f2p": 1.0, "p2p": 1.0, "warning": False, "skipped": True}
        else:
            logger.info("no test_patch for %s — skipping ground truth verification", repo_name)

        # Save baseline cache for reuse across variants
        if instance_id and not error:
            _save_baseline_cache(
                instance_id,
                {
                    "version": _BASELINE_CACHE_VERSION,
                    "base_sha": base_sha,
                    "tests_before": [t.model_dump() for t in tests_before],
                    "tests_head": [t.model_dump() for t in tests_head],
                    "ground_truth": ground_truth,
                },
            )

    _reset_to_base(repo_dir, base_sha)

    if error is not None:
        return {
            "repo": repo_name,
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
        if test_patch:
            apply_patch(repo_dir, test_patch, base_sha, skip_checkout=True)
        patch_result = apply_patch(repo_dir, generated_patch, base_sha, skip_checkout=True)
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
            # SWE-bench patch_failure semantics: don't run pytest on a broken
            # tree; mark every requested test as errored instead.
            tests_after = [
                TestResult(
                    name=n,
                    status="errored",
                    duration=0.0,
                    output="patch did not apply",
                    retry_count=0,
                )
                for n in test_names
            ]
    else:
        patch_result = PatchApplicationResult(
            success=False,
            method_used="failed",
            error="no generated patch provided",
        )
        logger.info("no generated patch for %s — skipping patch tests", repo_name)

    _elapsed = time.time() - _t0
    logger.info("  total: %.1fs for %s", _elapsed, instance_id or repo_name)
    return {
        "repo": repo_name,
        "base_sha": base_sha,
        "tests_before": [t.model_dump() for t in tests_before],
        "tests_head": [t.model_dump() for t in tests_head],
        "tests_after": [t.model_dump() for t in tests_after],
        "patch_application": patch_result.model_dump(),
        "ground_truth": ground_truth,
        "error": error,
        "latency_seconds": _elapsed,
    }


# ── Modal function ────────────────────────────────────────────────────────────


@app.function(
    image=BASE_IMAGE,
    volumes={"/repo_cache": repo_volume, "/test_cache": test_volume},
    timeout=1800,  # 30 min: heavy repos (scikit-learn) need 10-15 min for pip install -e .
    gpu=None,  # CPU only for test execution
)
def run_tests_in_container(  # noqa: PLR0913, PLR0917, PLR0912
    repo: str,
    base_sha: str,
    test_patch: str | None = None,
    gold_patch: str | None = None,
    generated_patch: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    timeout: int = 30,
    max_retries: int = 2,
    instance_id: str = "",
    verify_mode: str = "all",
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

    When *instance_id* is provided and a baseline cache exists, steps 3-5 are
    skipped in favour of cached results (T3).  When *verify_mode* is
    ``"once_per_repo"``, ground-truth verification runs only for the first
    instance of each repo (T4).

    Args:
        repo: ``"owner/name"`` of the GitHub repository.
        base_sha: Commit to evaluate against (pre-fix state).
        test_patch: Ground-truth test changes (unified diff), or None.
        gold_patch: Ground-truth fix patch (unified diff), or None.
        generated_patch: Model-generated fix patch (unified diff), or None.
        fail_to_pass: Test names that should fail at ``base_sha`` and pass after a fix.
        pass_to_pass: Test names that should keep passing.
        timeout: Per-test timeout in seconds.
        max_retries: Extra attempts for failed/errored tests (flaky detection).
        instance_id: SWE-bench instance id (used for cache key; empty = no cache).
        verify_mode: ``"all"`` (verify every instance), ``"once_per_repo"``
            (first instance per repo, then skip), or ``"none"`` (skip entirely).

    Returns:
        Dict with ``tests_before``/``tests_head``/``tests_after``
        (serialized ``TestResult`` lists), ``patch_application``
        (serialized ``PatchApplicationResult``) and ``ground_truth``
        (``{"f2p", "p2p", "warning"}``; empty when no ``test_patch``).
        An ``"error"`` key on repo preparation failure.
    """

    fail_to_pass = fail_to_pass or []
    pass_to_pass = pass_to_pass or []
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

    return _execute_instance(
        repo_dir,
        base_sha,
        test_patch,
        generated_patch,
        fail_to_pass,
        pass_to_pass,
        timeout=timeout,
        max_retries=max_retries,
        instance_id=instance_id,
        verify_mode=verify_mode,
        gold_patch=gold_patch,
    )


# ── Batch Modal function ──────────────────────────────────────────────────


def _run_swebench_instance_body(  # noqa: PLR0913, PLR0917
    instance_id: str,
    base_sha: str,
    test_patch: str | None = None,
    gold_patch: str | None = None,
    generated_patch: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    timeout: int = 30,
    max_retries: int = 2,
    verify_mode: str = "all",
) -> dict[str, Any]:
    """Execute one SWE-bench instance inside an official swebench image.

    ``swebench/sweb.eval.x86_64.<instance>:latest`` ships ``/testbed`` (the
    repo with full git history, checked out at the instance base commit plus a
    marker "SWE-bench" commit on top) and a conda env ``testbed`` with the
    project already installed.  No clone, no pip install at eval time.

    Uses the baseline cache (T3) to skip tests_before/tests_head when this
    instance has already been evaluated in the shared ``/test_cache`` volume.
    Honours *verify_mode* for once-per-repo ground-truth (T4).

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
            check=False,
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
        instance_id=instance_id,
        verify_mode=verify_mode,
        gold_patch=gold_patch,
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

    All per-repo functions wrap the SAME module-level body
    (``_run_swebench_instance_body``); without a ``name=`` override every
    registration would share one server-side tag (``__qualname__``) and the
    LAST repo would silently override the others — wrong image per repo.
    ``name`` affects only the tag; container-side lookup still resolves the
    body by ``implementation_name``.
    """
    fn = _swebench_fns.get(repo)
    if fn is None:
        fn = app.function(
            image=swebench_image(instance_id),
            volumes={"/test_cache": test_volume},
            timeout=900,  # 15 min per instance (conda install of 2 plugins + tests)
            gpu=None,  # CPU only for test execution
            name=f"_run_swebench_{repo.replace('/', '_')}",
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
    verify_mode: str = "all",
) -> list[dict[str, Any]]:
    """Execute test suites for multiple patches in a single container.

    One container per (repo, base_sha): clone once, checkout once, install
    once, run ``tests_before`` once, then per job: apply ground-truth
    ``test_patch`` + ``gold_patch`` (per-job; gold patches differ per
    instance so there is no shared-head fast path), verify ground truth,
    reset, re-apply ``test_patch`` then the generated patch and run
    ``tests_after`` (F2P tests only exist with test_patch applied).

    Uses baseline cache (T3) when jobs carry ``instance_id`` keys, and
    honours *verify_mode* for once-per-repo ground-truth (T4).

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
        verify_mode: Ground-truth verification mode.

    Returns:
        List of result dicts (same shape as ``run_tests_in_container`` return),
        one per ``test_job``.
    """
    import time

    start_time = time.time()

    from evaluation.metrics import compute_f2p
    from evaluation.patch_applier import apply_patch
    from evaluation.schema import PatchApplicationResult, TestResult

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

    # ── Per-job: reset, ground truth, apply generated patch, run tests_after ──
    job_test_patches = [job.get("test_patch") or test_patch or "" for job in test_jobs]
    job_gold_patches = [job.get("gold_patch") or "" for job in test_jobs]
    results: list[dict[str, Any]] = []
    for i, job in enumerate(test_jobs):
        elapsed = time.time() - start_time
        batch_timeout_warn = 3300
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
        job_instance_id = job.get("instance_id") or ""

        # Use baseline cache or compute tests_head
        tests_head: list[TestResult] = []
        ground_truth: dict[str, Any] = {}
        job_error: str | None = None
        job_test_patch = job_test_patches[i]
        cached = _load_baseline_cache(job_instance_id, base_sha) if job_instance_id else None

        if cached is not None:
            tests_head = [TestResult.model_validate(t) for t in (cached.get("tests_head") or [])]
            ground_truth = cached.get("ground_truth") or {}
            logger.info("baseline cache hit for job %s", job_instance_id)

        elif job_test_patch:
            skip_gt = verify_mode == "once_per_repo" and _is_repo_verified(repo)
            if skip_gt:
                ground_truth = {"f2p": 1.0, "p2p": 1.0, "warning": False, "skipped": True}
            else:
                patch_result = apply_patch(repo_dir, job_test_patch, base_sha, skip_checkout=True)
                if patch_result.success:
                    gold_patch = job_gold_patches[i]
                    if gold_patch:
                        gold_result = apply_patch(
                            repo_dir, gold_patch, base_sha, skip_checkout=True
                        )
                        if not gold_result.success:
                            logger.warning(
                                "gold_patch application failed for %s: %s",
                                repo,
                                gold_result.error,
                            )
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
                    _before_names = {t.name for t in tests_before}
                    _head_names = {t.name for t in tests_head}
                    logger.info(
                        "ground truth for %s: f2p=%.2f p2p=%.2f | before: %d/%d collected, "
                        "head: %d/%d collected | f2p missed in before: %s | f2p missed in head: %s",
                        repo,
                        f2p,
                        p2p,
                        len(tests_before),
                        len(job_test_names),
                        len(tests_head),
                        len(job_test_names),
                        [n for n in fail_to_pass if n not in _before_names][:3],
                        [n for n in fail_to_pass if n not in _head_names][:3],
                    )
                    ground_truth = {
                        "f2p": f2p,
                        "p2p": p2p,
                        "warning": f2p < _GT_F2P_THRESHOLD,
                    }
                    if f2p < _GT_F2P_THRESHOLD:
                        job_error = "ground truth F2P<100% (env drift or install incomplete?)"
                        logger.error("%s for %s", job_error, repo)
                    elif verify_mode == "once_per_repo" and not _is_repo_verified(repo):
                        _mark_repo_verified(repo)

            if job_instance_id and not job_error:
                _save_baseline_cache(
                    job_instance_id,
                    {
                        "version": 2,
                        "base_sha": base_sha,
                        "tests_before": [],
                        "tests_head": [t.model_dump() for t in tests_head],
                        "ground_truth": ground_truth,
                    },
                )

        _reset_to_base(repo_dir, base_sha)

        if job_error is not None:
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
            if job_test_patch:
                apply_patch(repo_dir, job_test_patch, base_sha, skip_checkout=True)
            patch_result = apply_patch(repo_dir, generated_patch, base_sha, skip_checkout=True)
            if patch_result.success:
                _install_repo(repo_dir)
                tests_after = collect_test_results(
                    repo_dir, job_test_names, timeout=timeout, max_retries=max_retries
                )
            else:
                logger.warning(
                    "generated patch application failed for %s: %s",
                    repo,
                    patch_result.error,
                )
                tests_after = [
                    TestResult(
                        name=n,
                        status="errored",
                        duration=0.0,
                        output="patch did not apply",
                        retry_count=0,
                    )
                    for n in job_test_names
                ]
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
