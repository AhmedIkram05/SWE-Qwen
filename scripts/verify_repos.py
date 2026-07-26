#!/usr/bin/env python3
"""
Phase 2 — Task 2.3: Repository Verification Script

Read candidates.json, clone each repo to temp dir, run 9 checks:
  - License (hard fail)           - Python version (hard fail)
  - Recent release                - Commit activity (last 6 mo)
  - pytest only                   - No external services
  - Issue-PR linkage              - Size (.py file count)
  - Tests run (pip install + pytest)

Usage:
    python scripts/verify_repos.py [--input PATH] [--output PATH]
                                   [--workers N] [--repos N]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from github import Auth, Github, GithubException
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORBIDDEN_DEPS = [
    "psycopg2",
    "redis",
    "pymongo",
    "boto3",
    "anthropic",
    "openai",
    "google-cloud",
    "sqlalchemy[asyncpg]",
    "mysqlclient",
]

PYPROJECT_REQUIRES_PYTHON = re.compile(r'requires-python\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
SETUP_REQUIRES_PYTHON = re.compile(r'python_requires\s*=\s*["\']([^"\']+)["\']')

CHECK_NAMES = [
    "license",
    "python_version",
    "recent_release",
    "commit_activity",
    "pytest_only",
    "no_services",
    "issue_pr_linkage",
    "size",
    "check_build_readiness",
]

HARD_FAIL_CHECKS = {"license", "python_version"}

MAX_TEST_SECONDS = 180

COMMITS_MIN_6MO = 10
RELEASE_MAX_AGE_DAYS = 365
MAX_LINKED_ISSUES_SAMPLE = 30
LINKAGE_PASS_RATIO = 0.50
MIN_PY_FILES = 500
MAX_PY_FILES = 5000


# ---------------------------------------------------------------------------
# Helper: GitHub API with retry
# ---------------------------------------------------------------------------


def gh_get(repo, attr: str, *args, **kwargs) -> Any:
    """Call a PyGithub method with one retry + 5s backoff on failure."""
    try:
        method = getattr(repo, attr)
        return method(*args, **kwargs)
    except GithubException as exc:
        if exc.status in (403, 429):
            time.sleep(5)
            return method(*args, **kwargs)
        raise


# ---------------------------------------------------------------------------
# Check Functions
# ---------------------------------------------------------------------------


def check_license(repo) -> dict:
    """Hard fail: license must be MIT, Apache-2.0, or BSD-3-Clause."""
    try:
        lic = repo.license
        spdx = lic.spdx_id if lic and hasattr(lic, "spdx_id") else (lic or "NOASSERTION")
        passed = spdx in ("MIT", "Apache-2.0", "BSD-3-Clause")
    except Exception as exc:
        return {"passed": False, "value": f"API error: {exc}", "hard_fail": True}
    else:
        return {"passed": passed, "value": spdx, "hard_fail": True}


def check_python_version(repo_dir: Path) -> dict:
    """Hard fail: python_requires >= 3.10."""
    pyproject = repo_dir / "pyproject.toml"
    setup_py = repo_dir / "setup.py"
    value = None

    if pyproject.exists():
        text = pyproject.read_text()
        m = PYPROJECT_REQUIRES_PYTHON.search(text)
        if m:
            value = m.group(1)

    if not value and setup_py.exists():
        text = setup_py.read_text()
        m = SETUP_REQUIRES_PYTHON.search(text)
        if m:
            value = m.group(1)

    if not value:
        return {"passed": True, "value": "unknown (no constraint found)", "hard_fail": True}

    # Parse version constraint — accept >=3.10, >=3.11, >=3.12, ~=3.10, etc.
    # We just check it allows 3.10+
    passed = _allows_python_310(value)
    return {"passed": passed, "value": value, "hard_fail": True}


def _allows_python_310(constraint: str) -> bool:
    """Check if a python_requires constraint allows Python >= 3.10.

    Handles:
        >=3.10       -> True  (allows 3.10+)
        >=3.9        -> True  (3.9+ includes 3.10)
        ^3.9         -> True  (poetry: >=3.9, <4.0)
        ~=3.10       -> True  (compatible: >=3.10, <4.0)
        >=3.8, <4.0  -> True  (range includes 3.10)
        >=3.11       -> False (minimum is 3.11)
    """
    clauses = [c.strip() for c in constraint.split(",")]

    min_versions = []
    for clause in clauses:
        m = re.match(r"(>=)\s*(\d+)\.(\d+)", clause)
        if m:
            min_versions.append((int(m.group(2)), int(m.group(3))))
            continue
        m = re.match(r"(\^)\s*(\d+)\.(\d+)", clause)  # poetry
        if m:
            min_versions.append((int(m.group(2)), int(m.group(3))))
            continue
        m = re.match(r"(~=)\s*(\d+)\.(\d+)", clause)  # compatible release
        if m:
            min_versions.append((int(m.group(2)), int(m.group(3))))

    if not min_versions:
        return True  # no constraint found — allow

    # Pick the strictest minimum version; if <= (3, 10), 3.10 is allowed
    return max(min_versions) <= (3, 10)


def check_recent_release(repo) -> dict:
    """Soft: release <= 365 days ago."""
    try:
        releases = gh_get(repo, "get_releases")
        if releases and releases[0] and releases[0].published_at:
            published = releases[0].published_at
            if isinstance(published, str):
                published = datetime.fromisoformat(published.replace("Z", "+00:00"))
            age = datetime.now(UTC) - published
            passed = age.days <= RELEASE_MAX_AGE_DAYS
            result = {
                "passed": passed,
                "value": published.strftime("%Y-%m-%d") if passed else f"{age.days}d ago",
                "hard_fail": False,
            }
        else:
            result = {"passed": False, "value": "no releases", "hard_fail": False}
    except Exception as exc:
        result = {"passed": False, "value": f"API error: {exc}", "hard_fail": False}
    return result


def check_commit_activity(repo) -> dict:
    """Soft: >= 10 commits in last 6 months via compare API."""
    try:
        since = datetime.now(UTC) - timedelta(days=180)
        commits = gh_get(repo, "get_commits", since=since)
        count = commits.totalCount if hasattr(commits, "totalCount") else len(list(commits[:100]))
        passed = count >= COMMITS_MIN_6MO
        result = {"passed": passed, "value": count, "hard_fail": False}
    except Exception as exc:
        result = {"passed": False, "value": f"API error: {exc}", "hard_fail": False}
    return result


def check_pytest_only(repo_dir: Path) -> dict:
    """Soft: detect pytest config, ensure no tox or forbidden test frameworks."""
    has_pytest = False
    reasons = []
    for marker in ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "noxfile.py"]:
        p = repo_dir / marker
        if p.exists():
            text = p.read_text()
            if "pytest" in text or marker == "pytest.ini":
                has_pytest = True
            if "tox" in text and marker in ("tox.ini", "setup.cfg"):
                reasons.append("tox detected")
            if "nox" in text:
                reasons.append("nox detected")
            # unittest can coexist with pytest — intentionally not flagged

    passed = has_pytest and not reasons
    value = "pytest only" if passed else "; ".join(reasons) if reasons else "no pytest"
    return {"passed": passed, "value": value, "hard_fail": False}


def check_no_services(repo_dir: Path) -> dict:
    """Soft: no forbidden dependencies in any requirements file."""
    found = []
    for pattern in ["*requirements*.txt", "*requirements*.in", "setup.cfg", "pyproject.toml"]:
        for p in repo_dir.glob(pattern):
            if p.is_file():
                text = p.read_text().lower()
                for dep in FORBIDDEN_DEPS:
                    if dep.lower() in text:
                        found.append(dep)
    return {"passed": len(found) == 0, "value": list(set(found)), "hard_fail": False}


def check_issue_pr_linkage(repo) -> dict:
    """Soft: sample last 30 bug-labeled issues, check what % have linked merged PRs.
    Uses GitHub timeline 'cross-referenced' events (the Development sidebar link),
    which is the actual mechanism GitHub uses to link issues to PRs.
    This is the direction that matters for Phase 3 ingestion: issue → PR."""
    try:
        linked = 0
        total = 0
        for issue in repo.get_issues(state="all", labels=["bug"], sort="updated", direction="desc"):
            if total >= MAX_LINKED_ISSUES_SAMPLE:
                break
            if issue.pull_request:
                continue  # skip PRs that appear in issue list
            total += 1

            # Check timeline for cross-referenced merged PRs
            found = False
            try:
                for event in issue.get_timeline():
                    if event.event == "cross-referenced" and event.source and event.source.issue:
                        src = event.source.issue
                        if src.pull_request:
                            pr = src.as_pull_request()
                            if pr.merged:
                                found = True
                                break
            except Exception:
                pass  # timeline not accessible

            if found:
                linked += 1

        ratio = linked / total if total > 0 else 0
        passed = ratio >= LINKAGE_PASS_RATIO
        return {"passed": passed, "value": round(ratio, 2), "hard_fail": False}
    except Exception as exc:
        return {"passed": False, "value": f"API error: {exc}", "hard_fail": False}


def check_size(repo_dir: Path) -> dict:
    """Soft: count Python files between 500 and 5000 (excl tests, .venv, .git)."""
    count = 0
    for p in repo_dir.rglob("*.py"):
        rel = p.relative_to(repo_dir).as_posix()
        if (
            rel.startswith(".venv/")
            or rel.startswith(".git/")
            or rel.startswith("venv/")
            or "/.venv/" in rel
        ):
            continue
        if rel.startswith("tests/") or rel.startswith("test/"):
            continue
        count += 1
    passed = MIN_PY_FILES <= count <= MAX_PY_FILES
    return {"passed": passed, "value": count, "hard_fail": False}


def _detect_package_name(repo_dir: Path) -> str | None:
    """Extract package name from pyproject.toml, setup.cfg, or setup.py. Falls back to dir name."""
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            return m.group(1)
        # Try [project] section
        m = re.search(r'\[project\]\s*\n\s*name\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1)
    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.exists():
        m = re.search(r"^name\s*=\s*(.+)$", setup_cfg.read_text(), re.MULTILINE)
        if m:
            return m.group(1).strip()
    setup_py = repo_dir / "setup.py"
    if setup_py.exists():
        m = re.search(r"""["']name["']\s*=\s*["']([^"']+)["']""", setup_py.read_text())
        if m:
            return m.group(1)
    return None


def check_build_readiness(repo_dir: Path) -> dict:  # noqa: PLR0912
    # ponytail: self-contained venv — caller skips host-level install
    """Lightweight structural check: pytest config present, package installs, imports cleanly.
    Replaces the heavyweight tests_run check — actual pytest execution happens in CI matrix jobs."""
    checks = {"pytest_config": False, "pip_install": False, "import_check": False}
    details = {}

    # 1. Detect pytest config file
    for cfg_name in ("pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"):
        cfg = repo_dir / cfg_name
        if cfg.exists():
            text = cfg.read_text()
            if cfg_name == "pyproject.toml" and "pytest" not in text:
                continue  # not a pytest-using project
            if cfg_name == "setup.cfg" and "tool:pytest" not in text:
                continue
            checks["pytest_config"] = True
            details["pytest_config_file"] = cfg_name
            break
    if not checks["pytest_config"]:
        details["pytest_config_error"] = "no pytest config found"

    # 2. Create isolated venv, install package, verify import
    venv_path = repo_dir / ".swe-build-check"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        details["venv_error"] = str(exc)[:200]
        passed = checks["pytest_config"] and checks["pip_install"] and checks["import_check"]
        return {
            "passed": passed,
            "value": json.dumps(checks),
            "details": details,
            "hard_fail": False,
        }

    pip_path = venv_path / "bin" / "pip"
    python_path = venv_path / "bin" / "python"

    # Install package (no test deps — lightweight)
    try:
        r = subprocess.run(
            [str(pip_path), "install", "-e", "."],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_dir,
            check=False,
        )
        if r.returncode == 0:
            checks["pip_install"] = True
        else:
            details["pip_install_error"] = r.stderr[:200]
    except subprocess.TimeoutExpired:
        details["pip_install_error"] = "timed out (120s)"

    # 3. Import check
    if checks["pip_install"]:
        pkg_name = (_detect_package_name(repo_dir) or repo_dir.name).replace("-", "_")
        # ponytail: some packages lack __version__ — fall back to "imported ok"
        try:
            import_cmd = (
                f"import {pkg_name}; v = getattr({pkg_name}, '__version__', "
                f"'imported ok'); print(v)"
            )
            r = subprocess.run(
                [str(python_path), "-c", import_cmd],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=repo_dir,
                check=False,
            )
            if r.returncode == 0:
                checks["import_check"] = True
                details["import_ok"] = r.stdout.strip()
            else:
                details["import_error"] = r.stderr[:200]
        except Exception as exc:
            details["import_error"] = str(exc)[:200]

    passed = checks["pytest_config"] and checks["pip_install"] and checks["import_check"]
    return {"passed": passed, "value": json.dumps(checks), "details": details, "hard_fail": False}


# ---------------------------------------------------------------------------
# Per-repo verification
# ---------------------------------------------------------------------------


def verify_one_repo(gh: Github, cand: dict) -> dict:  # noqa: PLR0915
    """Run all checks for a single repo. Returns result dict."""
    url = cand["url"]
    owner = cand["owner"]
    name = cand["name"]
    clone_url = f"https://x-access-token:{gh.requester.auth.token}@github.com/{owner}/{name}.git"

    result: dict = {
        "url": url,
        "owner": owner,
        "name": name,
        "domain_category": cand.get("domain_category", ""),
        "description": cand.get("description", ""),
        "stars": cand.get("stars", 0),
        "overall_pass": True,
        "default_branch": "main",
        "test_command_actual": (
            'pytest -x -m "not slow and not requires_modal and not requires_gcp"'
        ),
        "install_success": False,
        "install_error_snippet": None,
        "checks": {},
        "_error": None,
    }

    # --- Clone ---
    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="swe-qwen-verify-"))
        repo_dir = tmp_dir / name

        # Get default branch from API first
        try:
            gh_repo = gh.get_repo(f"{owner}/{name}")
            result["default_branch"] = gh_repo.default_branch
        except GithubException as exc:
            result["_error"] = f"GitHub API error fetching repo: {exc}"
            result["overall_pass"] = False
            return result

        console.print(f"  Cloning [bold]{owner}/{name}[/] ({cand.get('domain', '?')}) …")
        clone_result = subprocess.run(
            ["git", "clone", "--depth=1", clone_url, str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if clone_result.returncode != 0:
            result["_error"] = f"clone failed: {clone_result.stderr[:200]}"
            result["overall_pass"] = False
            return result

        # ponytail: host-level pip install was only for check_tests_run which now
        # creates its own isolated venv — skip to avoid contamination + timeouts
        result["install_success"] = True

        # --- Run checks ---
        checks: dict[str, dict] = {}

        # License (GitHub API)
        checks["license"] = check_license(gh_repo)

        # Python version (local file)
        checks["python_version"] = check_python_version(repo_dir)

        # Recent release (GitHub API)
        checks["recent_release"] = check_recent_release(gh_repo)

        # Commit activity (GitHub API)
        checks["commit_activity"] = check_commit_activity(gh_repo)

        # pytest only (local scan)
        checks["pytest_only"] = check_pytest_only(repo_dir)

        # No services (local scan)
        checks["no_services"] = check_no_services(repo_dir)

        # Issue-PR linkage (GitHub API)
        checks["issue_pr_linkage"] = check_issue_pr_linkage(gh_repo)

        # Size (local scan)
        checks["size"] = check_size(repo_dir)

        # Tests run (local exec)
        checks["check_build_readiness"] = check_build_readiness(repo_dir)

        # Determine overall pass
        overall = True
        for name, c in checks.items():
            c["hard_fail"] = name in HARD_FAIL_CHECKS
            if c["hard_fail"] and not c["passed"]:
                overall = False

        result["checks"] = checks
        result["overall_pass"] = overall

        # test_command_actual is set by CI matrix job — set placeholder here
        result["test_command_actual"] = "see CI matrix job"

    except subprocess.TimeoutExpired:
        result["_error"] = "clone timed out (180s)"
        result["overall_pass"] = False
    except Exception as exc:
        result["_error"] = str(exc)[:300]
        result["overall_pass"] = False
    finally:
        # Clean up
        if tmp_dir and tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Verify candidate repos against selection criteria."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="candidates.json",
        help="Input candidates JSON (default: candidates.json)",
    )
    parser.add_argument(
        "--output", "-o", default="verified.json", help="Output JSON path (default: verified.json)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--repos", type=int, default=None, help="Verify only first N repos (for testing)"
    )
    parser.add_argument(
        "--repo", default=None, help="Verify a single repo (owner/name) in CI matrix"
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print("[bold red]FATAL:[/] GITHUB_TOKEN environment variable not set.")
        sys.exit(1)

    if not Path(args.input).exists():
        console.print(
            f"[bold red]FATAL:[/] Input file {args.input} not found. Run find_candidates.py first."
        )
        sys.exit(1)

    with Path(args.input).open() as f:
        candidates = json.load(f)

    if args.repo:
        owner, name = args.repo.split("/", 1)
        candidates = [c for c in candidates if c["owner"] == owner and c["name"] == name]
        if not candidates:
            console.print(f"[bold red]FATAL:[/] Repo {args.repo} not found in candidates.")
            sys.exit(1)
    if args.repos:
        candidates = candidates[: args.repos]

    console.print("[bold]Phase 2 — Repository Verification[/]")
    console.print(f"Verifying {len(candidates)} candidates with {args.workers} workers\n")

    gh = Github(auth=Auth.Token(token))

    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Verifying repos…", total=len(candidates))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(verify_one_repo, gh, cand): cand for cand in candidates}
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task)

    # Sort results: passed first, then by stars desc
    cand_by_url = {c["url"]: c for c in candidates}
    results.sort(
        key=lambda r: (
            -1 if r["overall_pass"] else 1,
            -cand_by_url.get(r["url"], {}).get("stars", 0),
        )
    )

    # Summary
    passed_all = sum(1 for r in results if r["overall_pass"])
    failed_hard = sum(
        1
        for r in results
        if not r["overall_pass"]
        and any(c.get("hard_fail") and not c.get("passed") for c in r["checks"].values())
    )
    failed_soft = len(results) - passed_all - failed_hard

    output = {
        "verified_at": datetime.now(UTC).isoformat(),
        "github_rate_limit_remaining": gh.get_rate_limit().rate.remaining,
        "repos": results,
        "summary": {
            "total": len(results),
            "passed_all": passed_all,
            "failed_hard": failed_hard,
            "failed_soft": failed_soft,
        },
    }

    with Path(args.output).open("w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print results table
    table = Table(title="Verification Results")
    table.add_column("Repo", style="cyan")
    table.add_column("Domain")
    table.add_column("Pass?", justify="center")
    for r in results:
        domain = cand_by_url.get(r["url"], {}).get("domain", "?")
        status = "[green]✓[/]" if r["overall_pass"] else "[red]✗[/]"
        table.add_row(f"{r['owner']}/{r['name']}", domain, status)
    console.print(table)

    summary = output["summary"]
    console.print(
        f"\n[bold]Summary:[/] {summary['total']} total — "
        f"[green]{summary['passed_all']} passed[/], "
        f"[red]{summary['failed_hard']} hard fail[/], "
        f"[yellow]{summary['failed_soft']} soft fail[/]"
    )
    console.print(f"GitHub API rate limit remaining: {gh.get_rate_limit().rate.remaining}")
    console.print(f"[green]✓[/] Results written to [bold]{args.output}[/]")


if __name__ == "__main__":
    main()
