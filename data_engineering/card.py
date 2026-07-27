"""Auto-generated dataset card.

Produces a human-readable markdown card summarising dataset size, schema,
source repos, quality stats, splits, golden set, and lineage metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from data_engineering.schema import PipelineStats


def _source_repos_table(
    manifest: dict[str, Any], repo_records: dict[str, int] | None = None
) -> str:
    """Build a markdown table of source repositories.

    When *repo_records* is provided (``{repo_id: cleaned_count}``), the
    Records column shows actual per-repo counts instead of ``—``.
    """
    repos = manifest.get("repositories", [])
    lines = [
        "| Repo | Domain | Records | Stars | License |",
        "|------|--------|---------|-------|---------|",
    ]
    for r in repos:
        repo_id = r["id"]
        count = str(repo_records.get(repo_id, "—")) if repo_records else "—"
        lines.append(
            f"| {repo_id} | {r.get('domain_category', '')} | "
            f"{count} | {r.get('stars', '')} | {r.get('license', '')} |"
        )
    return "\n".join(lines)


def generate_dataset_card(
    manifest: dict[str, Any],
    stats: PipelineStats,
    run_id: str,
    git_sha: str = "",
) -> str:
    """Generate a dataset card markdown string.

    Args:
        manifest: Loaded manifest dict.
        stats: Pipeline statistics.
        run_id: Unique pipeline run ID.
        git_sha: Git commit SHA for version traceability.

    Returns:
        Complete dataset card as a markdown string.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Compute per-repo cleaned counts for the source repos table
    repo_records: dict[str, int] | None = None
    repo_raw_records: dict[str, int] | None = None  # zero-yield detection
    if stats.repo_results:
        repo_records = {r.repo_id: r.cleaned_count for r in stats.repo_results}
        repo_raw_records = {r.repo_id: r.raw_count for r in stats.repo_results}

    zero_yield: list[str] = []
    if repo_raw_records:
        zero_yield = [rid for rid, cnt in repo_raw_records.items() if cnt == 0]

    card = f"""# Dataset Card: SWE-Qwen Fine-Tuning Dataset

## Dataset Overview

- **Generated:** {now}
- **Run ID:** `{run_id}`
- **Pipeline Version (git SHA):** `{git_sha or "unknown"}`
- **Total Examples (train+val+test):** {stats.total_examples}
- **Total Raw Records Ingested:** {stats.total_raw}
- **Total Validated:** {stats.total_validated}
- **Total After Cleaning:** {stats.total_cleaned}
- **Golden Eval Subset:** {stats.golden_count}

## Schema

Each record is an ``IssueRecord`` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| ``issue_id`` | str | GitHub issue number |
| ``repo`` | str | ``owner/repo`` identifier |
| ``issue_body`` | str | Issue description text |
| ``patch_diff`` | str | Raw unified diff of the fixing PR |
| ``parsed_hunks`` | list[ParsedHunk] | Structured hunk-level diff data |
| ``test_results`` | TestResults | Final-state test outcome (post-fix) |
| ``pr_title`` | str | PR title |
| ``pr_description`` | str | PR body text |
| ``commit_messages`` | list[str] | All commit messages in the PR |
| ``files_changed`` | list[str] | All files modified by the PR |
| ``test_files_changed`` | list[str] | Subset of files in test directories |
| ``issue_labels`` | list[str] | GitHub issue labels |
| ``repo_domain`` | str | Domain category (web-api, cli, data-ml, etc.) |
| ``metadata`` | dict | Timestamps, URLs, SHAs, star counts |

## Source Repositories

{_source_repos_table(manifest, repo_records)}

## Zero-Yield Repos

Repos listed in the manifest that yielded 0 usable records (no bug-labeled issues with linked merged PRs).

| Repo |
|------|
{chr(10).join(f"| {rid} |" for rid in zero_yield) if zero_yield else "*None*"}

## Quality Stats

| Metric | Count |
|--------|-------|
| Raw records ingested | {stats.total_raw} |
| Validation errors | {stats.total_validation_errors} |
| Exact duplicates removed | {stats.dedup_stats.exact_duplicates_removed} |
| Content duplicates removed | {stats.dedup_stats.content_duplicates_removed} |
| Removed — no test file changes | {stats.clean_stats.removed_no_test_files} |
| Removed — patch too large | {stats.clean_stats.removed_patch_too_large} |
| Removed — binary diff | {stats.clean_stats.removed_binary} |
| Removed — non-Python files | {stats.clean_stats.removed_non_python} |
| Removed — empty issue body | {stats.clean_stats.removed_empty_body} |
| Removed — no F2P signal | {stats.clean_stats.removed_no_f2p_signal} |
| **Total after cleaning** | **{stats.total_cleaned}** |

## Split Ratios & Counts

| Split | Count | Ratio |
|-------|-------|-------|
| Train | {stats.train_count} | {f"{stats.train_count / stats.total_examples:.0%}" if stats.total_examples else "—"} |  # noqa: E501
| Validation | {stats.val_count} | {f"{stats.val_count / stats.total_examples:.0%}" if stats.total_examples else "—"} |  # noqa: E501
| Test | {stats.test_count} | {f"{stats.test_count / stats.total_examples:.0%}" if stats.total_examples else "—"} |  # noqa: E501
| **Total** | **{stats.total_examples}** | **100%** |

## Golden Eval Subset

- **Size:** {stats.golden_count} examples
- **Source split:** test (no data leakage)
- **Verification:** V1 F2P proxy (test file changes + F2P keywords)
- **Phase 5 upgrade:** actual test execution at base/head SHAs

## W&B Artifact Links

| Stage | Artifact |
|-------|----------|
"""
    for stage, name in stats.wandb_artifacts.items():
        card += f"| {stage} | `{name}` |\n"

    card += """
## GCS Archive Paths

| Stage | GCS Path |
|-------|----------|
"""
    for stage, path in stats.gcs_paths.items():
        card += f"| {stage} | `{path}` |\n"

    card += """

---
*Auto-generated by SWE-Qwen Data Pipeline.*
"""
    return card
