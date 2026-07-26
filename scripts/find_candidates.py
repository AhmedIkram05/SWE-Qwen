#!/usr/bin/env python3
"""
Phase 2 — Task 2.2: Candidate Sourcing Script

Run 5 GitHub Search API queries (one per domain), apply filters,
deduplicate, assign domain, output top 50 candidates as JSON.

Usage:
    python scripts/find_candidates.py [--output PATH] [--min-stars N]

Output:
    candidates.json (or --output path) — list of candidate repo dicts
"""

import argparse
import json
import os
import sys
from pathlib import Path

from github import Auth, Github, GithubException
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# For each domain, define (text_term, topic_qualifier) pairs.
# GitHub Search API requires a text term alongside qualifiers.
# We include the topic name as text + use topic: qualifier for precise matching.
DOMAIN_TOPICS: dict[str, list[str]] = {
    "web-api": ["fastapi", "django", "flask", "starlette", "aiohttp"],
    "cli": ["cli", "click", "typer", "argparse"],
    "data-ml": ["pandas", "numpy", "scikit-learn", "polars", "duckdb"],
    "utils": ["utility", "toolkit", "python-library", "developer-tools", "utils"],
    "testing": ["testing", "pytest", "pylint", "code-quality", "unittest"],
}

QUERY_TEMPLATE = (
    "{t} topic:{t} language:python stars:>{min_stars} pushed:>2025-01-01 archived:false"
)

# Max candidate repos to collect per sub-topic query
MAX_PER_QUERY = 15

EXCLUDED_ORGS = {
    "pallets",
    "pydantic",
    "tiangolo",
    "django",
    "psf",
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "sphinx",
}

DOMAIN_TOPIC_MAP: dict[str, set[str]] = {
    "web-api": {"fastapi", "django", "flask", "starlette", "aiohttp", "api", "rest", "web"},
    "cli": {"cli", "click", "typer", "argparse", "command-line", "terminal"},
    "data-ml": {
        "pandas",
        "numpy",
        "scikit-learn",
        "polars",
        "duckdb",
        "data",
        "machine-learning",
        "ml",
    },
    "utils": {"utility", "helpers", "toolkit", "utils", "tools"},
    "testing": {"testing", "pytest", "linting", "formatting", "test", "quality"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        console.print("[bold red]FATAL:[/] GITHUB_TOKEN environment variable not set.")
        sys.exit(1)
    return token


def normalize_license(license_obj) -> str | None:
    """Extract SPDX ID from PyGithub License object or None."""
    if license_obj is None:
        return None
    if hasattr(license_obj, "spdx_id"):
        return license_obj.spdx_id
    if isinstance(license_obj, dict):
        return license_obj.get("spdx_id")
    return str(license_obj)


def assign_domain(topics: list[str], matched_domains: list[str]) -> str:
    """
    Assign repo to the domain whose topic set has the most overlap
    with the repo's GitHub topics. Fallback: first matched domain.
    """
    if not matched_domains:
        # No query matched this repo — shouldn't happen, fallback to utils
        return "utils"

    if len(matched_domains) == 1:
        return matched_domains[0]

    topic_set = {t.lower() for t in topics}
    best_domain = matched_domains[0]
    best_score = -1
    for domain in matched_domains:
        score = len(topic_set & DOMAIN_TOPIC_MAP.get(domain, set()))
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source candidate Python repos for training dataset."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="candidates.json",
        help="Output JSON path (default: candidates.json)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=300,
        help="Minimum stars filter (default: 300)",
    )
    args = parser.parse_args()

    token = get_token()
    gh = Github(auth=Auth.Token(token))
    rate_limit = gh.get_rate_limit().rate
    console.print("[bold]Phase 2 — Candidate Sourcing[/]")
    console.print(f"Token: {token[:8]}... (rate limit: {rate_limit.remaining:,})")
    console.print(f"Searching with min-stars={args.min_stars}\n")

    seen: dict[str, dict] = {}  # keyed by "owner/name"
    for domain, topics in DOMAIN_TOPICS.items():
        for topic in topics:
            query = QUERY_TEMPLATE.format(t=topic, min_stars=args.min_stars)
            console.print(f"[cyan]Searching [bold]{domain}/{topic}[/bold] …[/]")
            try:
                results = gh.search_repositories(query=query, sort="stars", order="desc")
                count = 0
                for repo in results:
                    if count >= MAX_PER_QUERY:
                        break
                    owner = repo.owner.login
                    name = repo.name
                    if owner.lower() in EXCLUDED_ORGS:
                        continue
                    key = f"{owner}/{name}"
                    if key in seen:
                        seen[key]["matched_domains"].append(domain)
                        continue
                    seen[key] = {
                        "owner": owner,
                        "name": name,
                        "url": repo.html_url,
                        "stars": repo.stargazers_count,
                        "description": repo.description or "",
                        "topics": repo.get_topics(),
                        "license": normalize_license(repo.license),
                        "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                        "domain": None,  # resolved below
                        "matched_domains": [domain],
                    }
                    count += 1
                console.print(f"  → {count} new candidates from this query")
            except GithubException as exc:
                console.print(f"[red]  API error for {domain}/{topic}: {exc}[/]")

    # Resolve domain for each candidate
    candidates: list[dict] = list(seen.values())
    for cand in candidates:
        cand["domain"] = assign_domain(cand["topics"], cand["matched_domains"])
        del cand["matched_domains"]

    # Sort by stars descending, limit to 50
    candidates.sort(key=lambda r: r["stars"], reverse=True)
    candidates = candidates[:50]

    # Write output
    with Path(args.output).open("w") as f:
        json.dump(candidates, f, indent=2)
    console.print(f"\n[green]✓[/] Wrote {len(candidates)} candidates to [bold]{args.output}[/]")

    # Domain distribution table
    table = Table(title="Domain Distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Count", justify="right")
    by_domain: dict[str, int] = {}
    for cand in candidates:
        by_domain[cand["domain"]] = by_domain.get(cand["domain"], 0) + 1
    for domain, count in sorted(by_domain.items()):
        table.add_row(domain, str(count))
    table.add_row("[bold]Total[/]", str(len(candidates)), style="bold")
    console.print(table)

    # Print remaining rate info
    remaining = gh.get_rate_limit().rate.remaining
    console.print(f"[dim]GitHub API rate limit remaining: {remaining}[/]")


if __name__ == "__main__":
    main()
