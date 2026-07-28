"""Auto-generated dataset card.

Produces a human-readable markdown card summarising dataset size, schema,
source repos, quality stats, splits, golden set, and lineage metadata.
Source: SWE-bench dataset from Hugging Face.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from data_engineering.schema import PipelineStats

# SWE-bench repo domain mapping (from swebench_ingest.py)
SWE_BENCH_PYTHON_REPOS = {
    "astropy/astropy": "data-ml",
    "django/django": "web-api",
    "matplotlib/matplotlib": "data-ml",
    "mwaskom/seaborn": "data-ml",
    "pallets/flask": "web-api",
    "psf/black": "utils",
    "pytest-dev/pytest": "testing",
    "pydantic/pydantic": "utils",
    "scipy/scipy": "data-ml",
    "sphinx-doc/sphinx": "utils",
    "sympy/sympy": "data-ml",
    "tkinter/tkinter": "utils",
    "dask/dask": "data-ml",
    "huggingface/transformers": "data-ml",
    "pallets/jinja": "web-api",
    "pandas-dev/pandas": "data-ml",
    "scikit-learn/scikit-learn": "data-ml",
    "tensorflow/tensorflow": "data-ml",
}


def _source_repos_table(
    manifest: dict[str, Any], repo_records: dict[str, int] | None = None
) -> str:
    """Build a markdown table of SWE-bench source repositories."""
    lines = [
        "| Repo | Domain | Records | Split |",
        "|------|--------|---------|-------|",
    ]
    # SWE-bench repos are known; show which split they belong to
    for repo, domain in SWE_BENCH_PYTHON_REPOS.items():
        count = str(repo_records.get(repo, "—")) if repo_records else "—"
        # Determine which SWE-bench split this repo came from
        if repo in [
            "astropy/astropy",
            "django/django",
            "matplotlib/matplotlib",
            "mwaskom/seaborn",
            "pallets/flask",
            "psf/black",
            "pytest-dev/pytest",
            "pydantic/pydantic",
            "scipy/scipy",
            "sphinx-doc/sphinx",
            "sympy/sympy",
            "tkinter/tkinter",
        ]:
            split = "Verified"
        else:
            split = "Test"
        lines.append(f"| {repo} | {domain} | {count} | {split} |")
    return "\n".join(lines)


def _swebench_splits_summary() -> str:
    """Return SWE-bench split summary for dataset card."""
    return """| Split | Examples | Test Patches | F2P Verified |
|-------|----------|--------------|--------------|
| Verified | 500 | 500 | 500 |
| Test | 2,294 | 2,294 | 2,294 |
| Dev | 225 | 225 | 225 |
| Train (Python) | ~7,863 | 0 | 0 |
| **Total** | **~10,882** | **3,019** | **3,019** |"""


def generate_dataset_card(
    manifest: dict[str, Any],
    stats: PipelineStats,
    run_id: str,
    git_sha: str = "",
    source: str = "swebench",
) -> str:
    """Generate a dataset card markdown string.

    Args:
        manifest: Empty dict for SWE-bench (kept for API compatibility).
        stats: Pipeline statistics.
        run_id: Unique pipeline run ID.
        git_sha: Git commit SHA for version traceability.
        source: Data source (must be "swebench").

    Returns:
        Complete dataset card as a markdown string.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Compute per-repo cleaned counts for the source repos table
    repo_records: dict[str, int] | None = None
    if stats.repo_results:
        repo_records = {r.repo_id: r.cleaned_count for r in stats.repo_results}

    # SWE-bench source content
    source_section = f"""## Source: SWE-bench Dataset

- **Dataset:** `SWE-bench/SWE-bench` + `SWE-bench/SWE-bench_Verified`
- **Version:** 2025-04-29 (pinned)
- **Python Repos:** 18 (from Verified + Test + Dev)
- **License:** Apache-2.0 (SWE-bench)
- **Reproducibility:** Deterministic download from Hugging Face

### SWE-bench Splits

{_swebench_splits_summary()}

### SWE-bench Repositories

{_source_repos_table(manifest, repo_records)}"""

    schema_issue_id_desc = "SWE-bench instance_id (e.g. `django__django-12345`)"
    schema_pr_desc = "Not available in SWE-bench (empty string)"
    schema_commits_desc = "Not available in SWE-bench (empty list)"
    schema_labels_desc = "Not available in SWE-bench (empty list)"
    golden_source = "Verified + Test + Dev splits (all have FAIL_TO_PASS)"
    f2p_verification = "Ground-truth F2P from SWE-bench `FAIL_TO_PASS`/`PASS_TO_PASS` fields"

    card = f"""# Dataset Card: SWE-Qwen Fine-Tuning Dataset

## Dataset Overview

- **Generated:** {now}
- **Run ID:** `{run_id}`
- **Pipeline Version (git SHA):** `{git_sha or "unknown"}`
- **Data Source:** {source}
- **Total Examples (train+val+test):** {stats.total_examples}
- **Total Raw Records Ingested:** {stats.total_raw}
- **Total Validated:** {stats.total_validated}
- **Total After Cleaning:** {stats.total_cleaned}
- **Golden Eval Subset:** {stats.golden_count}

## Schema

Each record is an ``IssueRecord`` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| ``issue_id`` | str | {schema_issue_id_desc} |
| ``repo`` | str | ``owner/repo`` identifier |
| ``issue_body`` | str | Issue description text |
| ``patch_diff`` | str | Raw unified diff of the fixing PR |
| ``parsed_hunks`` | list[ParsedHunk] | Structured hunk-level diff data |
| ``test_results`` | TestResults | Final-state test outcome (post-fix) |
| ``pr_title`` | str | {schema_pr_desc} |
| ``pr_description`` | str | {schema_pr_desc} |
| ``commit_messages`` | list[str] | {schema_commits_desc} |
| ``files_changed`` | list[str] | All files modified by the PR |
| ``test_files_changed`` | list[str] | Subset of files in test directories |
| ``issue_labels`` | list[str] | {schema_labels_desc} |
| ``repo_domain`` | str | Domain category (web-api, cli, data-ml, etc.) |
| ``metadata`` | dict | Timestamps, SHAs, version, hints, has_test_patch |

{source_section}

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
- **Source split:** {golden_source}
- **Verification:** {f2p_verification}
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
