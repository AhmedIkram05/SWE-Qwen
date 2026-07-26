#!/usr/bin/env python3
"""
Phase 2 — Task 2.5: Manifest Builder

Read verified.json + selected repo list, enrich with ingestion_config,
validate against Pydantic model, write repos/manifest.json.

Usage:
    python scripts/build_manifest.py [--verified PATH] [--selection PATH]
                                     [--output PATH]

    --selection: JSON file with list of selected repo URLs, or a file with one
                 URL per line.

If --selection is omitted, all repos with overall_pass=true are included.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class CheckDetail(BaseModel):
    passed: bool
    value: Any
    hard_fail: bool = False


class VerificationInfo(BaseModel):
    verified_at: str
    all_checks_passed: bool
    check_details: dict[str, CheckDetail]


class IngestionConfig(BaseModel):
    default_branch: str
    issue_labels_to_include: list[str] = ["bug", "defect", "fix"]
    pr_merge_commits_only: bool = True
    max_issues_per_repo: int = 2000
    exclude_paths: list[str] = ["docs/", "examples/", "benchmarks/", "scripts/", "*.md", "*.rst"]
    test_directories: list[str] = ["tests/", "test/"]


class Repository(BaseModel):
    id: str
    url: str
    owner: str
    name: str
    description: str = ""
    license: str | None = None
    primary_language: str = "Python"
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    latest_release: str | None = None
    commits_last_6mo: int = 0
    python_version: str = ""
    test_command: str = "pytest -x"
    test_command_actual: str = "pytest -x"
    install_command: str = "pip install -e ."
    install_success: bool = True
    py_file_count: int = 0
    domain_category: str = ""
    issue_pr_linkage_ratio: float = 0.0
    verification: VerificationInfo
    ingestion_config: IngestionConfig


class SelectionCriteria(BaseModel):
    license: list[str] = ["MIT", "Apache-2.0", "BSD-3-Clause"]
    python_version: str = ">=3.10"
    min_stars: int = 300
    min_commits_6mo: int = 10
    max_age_days: int = 365
    test_framework: str = "pytest"
    issue_pr_linkage_min_ratio: float = 0.35
    size_range_py_files: list[int] = [500, 5000]


class ManifestSummary(BaseModel):
    total_repos: int
    by_domain: dict[str, int]
    total_py_files: int
    total_stars: int
    avg_issue_pr_linkage: float


class Manifest(BaseModel):
    version: str = "1.0"
    created_at: str = ""
    selection_criteria: SelectionCriteria = SelectionCriteria()
    repositories: list[Repository]
    summary: ManifestSummary

    @field_validator("repositories")
    @classmethod
    def check_min_1_repo(cls, v: list) -> list:
        if len(v) < 1:
            raise ValueError(f"Manifest must have at least 1 repo, got {len(v)}")
        return v

    @field_validator("created_at")
    @classmethod
    def fill_created_at(cls, v: str) -> str:
        return v or datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_manifest(
    verified_path: str,
    selection_urls: set[str] | None,
) -> Manifest:

    with Path(verified_path).open() as f:
        verified = json.load(f)

    repos_data = verified["repos"]

    # Filter to selected URLs (or all passing)
    selected = []
    for r in repos_data:
        if selection_urls and r["url"] not in selection_urls:
            continue
        if not selection_urls and not r["overall_pass"]:
            continue
        selected.append(r)

    if not selected:
        console.print("[red]No repos selected for manifest.[/]")
        sys.exit(1)

    manifest_repos: list[Repository] = []
    for r in selected:
        checks = r.get("checks", {})

        repo = Repository(
            id=f"{r['owner']}-{r['name']}",
            url=r["url"],
            owner=r["owner"],
            name=r["name"],
            description=r.get("description", ""),
            license=checks.get("license", {}).get("value"),
            stars=r.get("stars", 0) or 0,
            forks=0,
            open_issues=0,
            latest_release=checks.get("recent_release", {}).get("value")
            if checks.get("recent_release", {}).get("passed")
            else None,
            commits_last_6mo=int(checks.get("commit_activity", {}).get("value", 0) or 0),
            python_version=str(checks.get("python_version", {}).get("value", "")),
            test_command="pytest -x",
            test_command_actual=r.get("test_command_actual", "pytest -x"),
            install_command="pip install -e .",
            install_success=r.get("install_success", True),
            py_file_count=int(checks.get("size", {}).get("value", 0)),
            domain_category=r.get("domain_category", ""),
            issue_pr_linkage_ratio=float(checks.get("issue_pr_linkage", {}).get("value", 0) or 0),
            verification=VerificationInfo(
                verified_at=verified["verified_at"],
                all_checks_passed=r["overall_pass"],
                check_details={k: CheckDetail(**v) for k, v in checks.items()},
            ),
            ingestion_config=IngestionConfig(
                default_branch=r.get("default_branch", "main"),
            ),
        )
        manifest_repos.append(repo)

    # Compute summary
    by_domain: dict[str, int] = {}
    total_py_files = 0
    total_stars = 0
    total_linkage = 0.0
    for repo in manifest_repos:
        by_domain[repo.domain_category] = by_domain.get(repo.domain_category, 0) + 1
        total_py_files += repo.py_file_count
        total_stars += repo.stars
        total_linkage += repo.issue_pr_linkage_ratio

    n = len(manifest_repos)
    manifest = Manifest(
        created_at=datetime.now(UTC).isoformat(),
        repositories=manifest_repos,
        summary=ManifestSummary(
            total_repos=n,
            by_domain=by_domain,
            total_py_files=total_py_files,
            total_stars=total_stars,
            avg_issue_pr_linkage=round(total_linkage / n, 2) if n > 0 else 0.0,
        ),
    )

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build repos/manifest.json from verified candidates."
    )
    parser.add_argument(
        "--verified",
        "-v",
        default="verified.json",
        help="Path to verified.json (default: verified.json)",
    )
    parser.add_argument(
        "--selection",
        "-s",
        default=None,
        help="Path to selection file: JSON list of URLs, or plain text one per line",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="repos/manifest.json",
        help="Output path (default: repos/manifest.json)",
    )
    args = parser.parse_args()

    if not Path(args.verified).exists():
        console.print(
            f"[red]Verified file not found: {args.verified}. Run verify_repos.py first.[/]"
        )
        sys.exit(1)

    # Load selection URLs
    selection_urls: set[str] | None = None
    if args.selection:
        sel_path = Path(args.selection)
        if not sel_path.exists():
            console.print(f"[red]Selection file not found: {args.selection}[/]")
            sys.exit(1)
        raw = sel_path.read_text().strip()
        try:
            urls = json.loads(raw)
            if isinstance(urls, list):
                selection_urls = set(urls)
            elif isinstance(urls, dict) and "selected" in urls:
                selection_urls = set(urls["selected"])
            else:
                console.print(
                    "[red]Selection JSON must be a list of URLs or {'selected': [...]}[/]"
                )
                sys.exit(1)
        except json.JSONDecodeError:
            selection_urls = {line.strip() for line in raw.split("\n") if line.strip()}

    console.print("[bold]Phase 2 — Manifest Builder[/]")

    try:
        manifest = build_manifest(args.verified, selection_urls)
    except Exception as exc:
        console.print(f"[red]Error building manifest: {exc}[/]")
        sys.exit(1)

    # Validate (Pydantic does it on construction)
    try:
        manifest_dict = manifest.model_dump()
    except Exception as exc:
        console.print(f"[red]Pydantic validation failed: {exc}[/]")
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(manifest_dict, f, indent=2, default=str)

    summary = manifest.summary
    console.print(
        f"[green]✓[/] Manifest written with {summary.total_repos} repos, "
        f"{len(summary.by_domain)} domains"
    )
    console.print(f"  Domains: {summary.by_domain}")
    console.print(f"  Total Python files: {summary.total_py_files}")
    console.print(f"  Total stars: {summary.total_stars}")
    console.print(f"  Avg issue-PR linkage: {summary.avg_issue_pr_linkage}")
    console.print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
