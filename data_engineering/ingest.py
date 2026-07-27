"""GitHub API ingestion module.

Fetches issues with bug/fix labels from each repo in the manifest, resolves
linked merged PRs via timeline events, and extracts unified diffs, file
lists, and commit messages.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from threading import BoundedSemaphore
from typing import Any

from github import Github, GithubException
from github.PaginatedList import PaginatedList
from pydantic import BaseModel

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import ParsedHunk

# Global concurrency throttle: GitHub secondary rate limit kicks in when
# too many requests from the same token are in-flight at once. This
# semaphore caps total concurrent HTTP requests across all repos.
_gh_semaphore = BoundedSemaphore(1)
# Cache for PR details fetched via GraphQL: repo_full_name -> {pr_num: details}
_pr_graphql_cache: dict[str, dict[int, dict]] = {}
# Cache for patch text (pre-fetched in parallel): repo_full_name -> {pr_num: patch_diff}
_patch_cache: dict[str, dict[int, str]] = {}


class IngestionConfig(BaseModel):
    """Per-repo ingestion overrides.

    Mirrors the relevant fields from ``DataPipelineConfig`` so individual
    repos can tune label filters, issue caps, and test-directory detection.
    """

    issue_labels_to_include: list[str] = []
    max_issues_per_repo: int = 2000
    test_directories: list[str] = ["tests/", "test/"]


logger = logging.getLogger(__name__)

# ── Rate-limit helpers ─────────────────────────────────────────────────────


def _parse_unified_diff(diff_str: str) -> list[ParsedHunk]:
    """Parse a unified diff string into a list of *ParsedHunk*."""
    import unidiff

    hunks: list[ParsedHunk] = []
    try:
        patch_set = unidiff.PatchSet(diff_str)
    except Exception:
        return hunks  # best-effort; validation will catch failures
    for patched_file in patch_set:
        for hunk in patched_file:
            hunks.append(
                ParsedHunk(
                    file=patched_file.path,
                    old_start=hunk.source_start,
                    old_lines=hunk.source_length,
                    new_start=hunk.target_start,
                    new_lines=hunk.target_length,
                    diff_lines=[str(line) for line in hunk],
                )
            )
    return hunks


def github_retry(func: Callable) -> Callable:
    """Decorator: exponential backoff on GitHub API errors.

    ponytail: no proactive rate-limiter — the retry backoff handles 403/429
    reactively, letting parallel workers burst to GitHub's secondary limit
    naturally rather than serializing at 0.75 s spacing.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_delay = 60
        delay = 1
        last_exc = None
        for attempt in range(7):  # 1+2+4+8+16+32+60 ≈ 123 s max
            try:
                return func(*args, **kwargs)
            except GithubException as exc:
                last_exc = exc
                status = exc.status
                if status == 403 and "secondary rate limit" in str(exc).lower():
                    logger.warning("Secondary rate limit hit; sleeping %ss", delay)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                elif status == 429:
                    retry_after = int(exc.headers.get("Retry-After", delay))
                    logger.warning("429 received; sleeping %ss", retry_after)
                    time.sleep(retry_after)
                    delay = min(retry_after * 2, max_delay)
                elif status in (401, 404, 422):
                    raise  # non-recoverable
                else:
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            except Exception as exc:
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
        raise RuntimeError(f"GitHub API call failed after 7 retries: {last_exc}") from last_exc

    return wrapper


# ── Core fetching functions ───────────────────────────────────────────────


@github_retry
def _fetch_paginated(paginated: PaginatedList, max_items: int) -> list[Any]:
    """Fetch up to *max_items* from a PaginatedList."""
    items: list[Any] = []
    for item in paginated:
        if len(items) >= max_items:
            break
        items.append(item)
    return items


@github_retry
def fetch_issues_for_repo(
    repo_full_name: str,
    gh: Github,
    labels: list[str],  # ponytail: kept for signature compat, no longer used
    max_issues: int,
) -> list[Any]:
    """Fetch most recent issues from a repo (no label filter).

    Phase 3 already verified these repos have 60-80% issue-PR linkage on bug
    issues. Restricting to bug labels costs 80% of potential records — many
    repos don't have 2000 bug-labeled issues. Fetching all issues and relying
    on PR-link resolution gives us the full pipeline throughput.
    """
    repo = gh.get_repo(repo_full_name)
    issues = repo.get_issues(state="all", sort="updated", direction="desc")
    return _fetch_paginated(issues, max_issues)


@github_retry
def fetch_linked_prs(issue: Any, max_events: int) -> list[Any]:
    """Find merged PRs linked to *issue* via cross-reference timeline events.

    Uses ``get_page(0)`` (single page fetch) instead of iterating the full
    paginated list — cross-referenced events almost always appear on the
    first page of the timeline.
    """
    linked_prs: list[Any] = []
    try:
        timeline = issue.get_timeline()
        events = timeline.get_page(0)  # ponytail: single page, usually enough
        for event in events:
            if len(linked_prs) >= max_events:
                break
            event_type = getattr(event, "event", None)
            if event_type != "cross-referenced":
                continue
            source = getattr(event, "source", None)
            if source is None:
                continue
            src_issue = getattr(source, "issue", None)
            if src_issue is None:
                continue
            pr = getattr(src_issue, "pull_request", None)
            if pr is None:
                continue
            try:
                pr_obj = issue.repository.get_pull(src_issue.number)
            except GithubException:
                continue
            if pr_obj.merged:
                linked_prs.append(pr_obj)
    except GithubException:
        logger.warning("Failed to fetch timeline for issue %s", issue.number)
    return linked_prs


@github_retry
def fetch_pr_details(pr: Any) -> dict[str, Any]:
    """Extract diff, file list, and commit messages from a PR.

    Note: test_files_changed is NOT computed here — callers receive
    files_changed and apply their own test_dirs filtering.
    """
    files = pr.get_files()
    commits = pr.get_commits()

    files_changed: list[str] = []
    patch_diff_lines: list[str] = []
    commit_messages: list[str] = []

    for pf in files:
        files_changed.append(pf.filename)
        patch = getattr(pf, "patch", None)
        if patch:
            patch_diff_lines.append(f"--- a/{pf.filename}")
            patch_diff_lines.append(f"+++ b/{pf.filename}")
            patch_diff_lines.append(patch)

    for c in commits:
        msg = c.commit.message or ""
        commit_messages.append(msg.strip())

    return {
        "files_changed": files_changed,
        "patch_diff": "\n".join(patch_diff_lines),
        "commit_messages": commit_messages,
    }


# ── GraphQL batch helpers ──────────────────────────────────────────────────


def _run_graphql(gh: Github, query: str, variables: dict | None = None) -> dict:
    """Run a GraphQL query via PyGithub requester.

    Returns the ``data`` dict.  PyGithub's ``graphql_query`` returns
    ``(headers, data)`` and already raises on errors — we just unpack.
    For 400 GraphQL errors (expected NOT_FOUND) we extract partial data
    from the exception so callers can use what resolved.
    """
    try:
        _, data = gh.requester.graphql_query(query, variables or {})
        return data
    except GithubException as exc:
        # GraphQL errors (partial NOT_FOUND) carry usable data in exc.data
        if exc.status == 400 and isinstance(exc.data, dict):
            partial = exc.data.get("data") or {}
            if partial:
                return partial
        raise


def _graphql_resolve_issue_pr_links(
    repo_full_name: str,
    gh: Github,
    issues: list[Any],
    max_events: int,
) -> dict[int, list[int]]:
    """Batch-resolve issue → PR links via aliased GraphQL node queries.

    Replaces per-issue ``fetch_linked_prs`` REST calls with N/100 GraphQL
    queries.  Returns ``{issue_number: [pr_number, ...]}`` for merged PRs
    only.

    Falls back to individual REST calls (``fetch_linked_prs``) when GraphQL
    is unavailable (e.g., HTTP 403 on the /graphql endpoint).
    """
    owner, repo = repo_full_name.split("/", 1)
    result: dict[int, list[int]] = {}

    # Chunk into batches of 100 issues per GraphQL query
    iss_nums = [i.number for i in issues]
    batch_size = 100
    for start in range(0, len(iss_nums), batch_size):
        batch = iss_nums[start : start + batch_size]
        # Build aliased field:  alias: issue(number: N) { timelineItems ... }
        fields: list[str] = []
        for idx, num in enumerate(batch):
            fields.append(
                f"""i{idx}: issue(number: {num}) {{
                    timelineItems(first: {max_events}, itemTypes: [CROSS_REFERENCED_EVENT]) {{
                        nodes {{
                            ... on CrossReferencedEvent {{
                                source {{
                                    ... on PullRequest {{ number state merged }}
                                }}
                            }}
                        }}
                    }}
                }}"""
            )

        query = f"""
        query {{
          repository(owner: "{owner}", name: "{repo}") {{ {", ".join(fields)} }}
        }}
        """

        try:
            data = _run_graphql(gh, query)
        except Exception as exc:
            logger.warning(
                "GraphQL batch failed for %s batch %d–%d (%s); falling back to REST for this batch",
                repo_full_name,
                batch[0],
                batch[-1],
                exc,
            )
            # Fallback: individual REST timeline calls for this batch
            for issue in issues:
                if issue.number in batch:
                    try:
                        prs = fetch_linked_prs(issue, max_events)
                        linked = [p.number for p in prs]
                        if linked:
                            result[issue.number] = linked
                    except Exception:
                        continue
            continue

        repo_data = data.get("repository", {})
        if repo_data is None:
            continue

        for idx, num in enumerate(batch):
            alias = f"i{idx}"
            issue_data = repo_data.get(alias)
            if issue_data is None:
                continue
            timeline = issue_data.get("timelineItems", {}).get("nodes", [])
            pr_numbers: list[int] = []
            for node in timeline:
                src = node.get("source", {})
                typename = src.get("__typename", "")
                if typename != "PullRequest":
                    continue
                if src.get("merged"):
                    pr_numbers.append(src["number"])
            if pr_numbers:
                result[num] = pr_numbers

    return result


def _graphql_fetch_pr_details(
    gh: Github,
    repo_full_name: str,
    pr_numbers: list[int],
) -> dict[int, dict]:
    """Batch-fetch PR details (files, commits, metadata) via GraphQL.

    Returns ``{pr_num: {title, body, merged, merged_at, base_sha, head_sha,
    url, files_changed, commit_messages}}`` for **merged** PRs only.
    ``patch`` is NOT available in GraphQL — callers must fetch it via REST.
    """
    owner, repo = repo_full_name.split("/", 1)
    result: dict[int, dict] = {}
    pr_nums = list(set(pr_numbers))  # deduplicate across issues
    batch_size = 100

    for start in range(0, len(pr_nums), batch_size):
        batch = pr_nums[start : start + batch_size]
        fields: list[str] = []
        for idx, num in enumerate(batch):
            fields.append(
                f"""pr{idx}: pullRequest(number: {num}) {{
                    state merged mergedAt title body
                    baseRefOid headRefOid url
                    files(first: 128) {{
                        nodes {{ filename }}
                        pageInfo {{ hasNextPage }}
                    }}
                    commits(first: 100) {{
                        nodes {{ commit {{ message }} }}
                        pageInfo {{ hasNextPage }}
                    }}
                }}"""
            )

        query = f"""
        query {{
          repository(owner: "{owner}", name: "{repo}") {{ {", ".join(fields)} }}
        }}
        """

        try:
            data = _run_graphql(gh, query)
        except Exception:
            logger.warning(
                "GraphQL PR details failed for %s batch %d–%d",
                repo_full_name,
                batch[0],
                batch[-1],
            )
            continue

        repo_data = data.get("repository", {})
        if repo_data is None:
            continue

        for idx, num in enumerate(batch):
            alias = f"pr{idx}"
            pr_data = repo_data.get(alias)
            if pr_data is None:
                continue
            if pr_data.get("state") != "MERGED":
                continue

            files = pr_data.get("files", {})
            files_changed = [f["filename"] for f in files.get("nodes", [])]

            commits = pr_data.get("commits", {})
            commit_messages = [c["commit"]["message"].strip() for c in commits.get("nodes", [])]

            result[num] = {
                "title": pr_data.get("title", ""),
                "body": pr_data.get("body", ""),
                "merged": True,
                "merged_at": pr_data.get("mergedAt"),
                "base_sha": pr_data.get("baseRefOid", ""),
                "head_sha": pr_data.get("headRefOid", ""),
                "url": pr_data.get("url", ""),
                "files_changed": files_changed,
                "commit_messages": commit_messages,
            }

    return result


def _prefetch_patches(
    repo_full_name: str,
    gh: Github,
    pr_numbers: list[int],
    max_workers: int = 10,
) -> None:
    """Pre-fetch unified diff patches for all merged PRs in parallel.

    Populates ``_patch_cache[repo_full_name][pr_num]`` so that
    ``_build_records_for_issue`` can build records with zero REST calls.
    Uses a thread pool because ETag-304 responses cost zero rate-limit
    budget and GitHub permits moderate parallel bursts for cache warming.
    """
    pr_nums = list(set(pr_numbers))
    repo = gh.get_repo(repo_full_name)
    _patch_cache[repo_full_name] = {}
    patch_cache = _patch_cache[repo_full_name]

    def _fetch_one(pr_num: int) -> tuple[int, str] | None:
        try:
            pr = repo.get_pull(pr_num)
            if not pr.merged:
                return None
            files = pr.get_files()
            lines: list[str] = []
            for pf in files:
                patch = getattr(pf, "patch", None)
                if patch:
                    lines.append(f"--- a/{pf.filename}")
                    lines.append(f"+++ b/{pf.filename}")
                    lines.append(patch)
            return pr_num, "\n".join(lines)
        except GithubException:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(_fetch_one, pr_nums):
            if result is not None:
                pr_num, patch = result
                patch_cache[pr_num] = patch


def _build_records_for_issue(
    issue: Any,
    pr_numbers: list[int],
    repo_full_name: str,
    repo_id: str,
    repo_config: dict[str, Any],
    test_dirs: list[str],
) -> list[dict[str, Any]]:
    """Build raw records for one issue's linked PRs (zero REST calls).

    PR metadata comes from the ``_pr_graphql_cache`` and patch text from
    ``_patch_cache`` — both pre-fetched before this function is called.
    """
    pr_details = _pr_graphql_cache.get(repo_full_name, {})
    patches = _patch_cache.get(repo_full_name, {})
    records: list[dict[str, Any]] = []
    for pr_num in pr_numbers:
        gql = pr_details.get(pr_num)
        if gql is None:
            continue  # not merged or not resolved via GraphQL
        patch_diff = patches.get(pr_num)
        if patch_diff is None:
            continue  # patch not available

        test_files_changed: list[str] = []
        for f in gql["files_changed"]:
            for td in test_dirs:
                if f.startswith(td):
                    test_files_changed.append(f)
                    break

        record = {
            "issue_id": str(issue.number),
            "repo": repo_id,
            "pr_number": str(pr_num),
            "issue_body": issue.body or "",
            "patch_diff": patch_diff,
            "parsed_hunks": [],
            "test_results": {"passed": [], "failed": [], "errored": []},
            "pr_title": gql["title"],
            "pr_description": gql["body"],
            "commit_messages": gql["commit_messages"],
            "files_changed": gql["files_changed"],
            "test_files_changed": test_files_changed,
            "issue_labels": [lbl.name for lbl in issue.labels],
            "repo_domain": repo_config.get("domain_category", ""),
            "metadata": {
                "issue_url": str(issue.html_url),
                "pr_url": gql["url"],
                "merged_at": gql["merged_at"],
                "base_sha": gql["base_sha"],
                "head_sha": gql["head_sha"],
                "repo_stars": repo_config.get("stars", 0),
            },
        }
        try:
            record["parsed_hunks"] = [h.model_dump() for h in _parse_unified_diff(patch_diff)]
        except Exception:
            pass

        records.append(record)

    return records


def _process_single_issue(
    issue: Any,
    repo_id: str,
    repo_config: dict[str, Any],
    max_events: int,
    test_dirs: list[str],
) -> list[dict[str, Any]]:
    """Resolve linked PRs for one issue and build raw records.

    Runs in a thread-pool worker; ``@github_retry`` on ``fetch_linked_prs`` /
    ``fetch_pr_details`` handles rate-limiting and retry with a thread-safe
    lock.
    """
    records: list[dict[str, Any]] = []
    try:
        linked_prs = fetch_linked_prs(issue, max_events)
    except Exception:
        return []
    if not linked_prs:
        return []

    for pr in linked_prs:
        try:
            details = fetch_pr_details(pr)
        except Exception:
            continue

        # Detect test files with the actual config's test_dirs
        test_files_changed: list[str] = []
        for f in details["files_changed"]:
            for td in test_dirs:
                if f.startswith(td):
                    test_files_changed.append(f)
                    break

        record = {
            "issue_id": str(issue.number),
            "repo": repo_id,
            "pr_number": str(pr.number),
            "issue_body": issue.body or "",
            "patch_diff": details["patch_diff"],
            "parsed_hunks": [],
            "test_results": {
                "passed": [],
                "failed": [],
                "errored": [],
            },
            "pr_title": pr.title or "",
            "pr_description": pr.body or "",
            "commit_messages": details["commit_messages"],
            "files_changed": details["files_changed"],
            "test_files_changed": test_files_changed,
            "issue_labels": [lbl.name for lbl in issue.labels],
            "repo_domain": repo_config.get("domain_category", ""),
            "metadata": {
                "issue_url": str(issue.html_url),
                "pr_url": str(pr.html_url),
                "merged_at": (pr.merged_at.isoformat() if pr.merged_at else None),
                "base_sha": pr.base.sha if pr.base else "",
                "head_sha": pr.head.sha if pr.head else "",
                "repo_stars": repo_config.get("stars", 0),
            },
        }

        # Parse hunks best-effort
        try:
            record["parsed_hunks"] = [
                h.model_dump() for h in _parse_unified_diff(details["patch_diff"])
            ]
        except Exception:
            pass  # validation will catch

        records.append(record)

    return records


def ingest_repo(
    repo_config: dict[str, Any],
    gh: Github,
    config: DataPipelineConfig,
) -> list[dict[str, Any]]:
    """Ingest all issue-PR pairs for a single repo.

    Uses **batch GraphQL resolution** to replace per-issue timeline REST
    calls (~2000 calls) with N/100 GraphQL queries.  PR details (files +
    commits) still use REST since ``patch`` is not available in GitHub's
    GraphQL schema.

    Falls back transparently to per-issue REST processing when GraphQL is
    unavailable.
    """
    repo_id: str = repo_config["id"]
    # GitHub API needs owner/name, not the opaque repo_id
    repo_full_name = (
        repo_config.get("full_name")
        or f"{repo_config.get('owner', '')}/{repo_config.get('name', '')}"
    )
    if repo_full_name == "/":
        repo_full_name = repo_id  # fallback for legacy manifests without owner/name
    ic_raw = repo_config.get("ingestion_config", {})
    ic = IngestionConfig(**ic_raw) if ic_raw else IngestionConfig()
    labels = ic.issue_labels_to_include or config.issue_labels_to_include
    max_issues = min(ic.max_issues_per_repo, config.max_issues_per_repo)
    max_events = config.max_events_per_issue
    test_dirs = ic.test_directories or config.test_directories

    # Stagger repo starts so the first HTTP requests don't all fire at second 0
    jitter = random.uniform(0.5, 4.0)
    logger.debug("Staggering %s by %.1fs", repo_id, jitter)
    time.sleep(jitter)

    logger.info("Ingesting %s (max %s issues, labels=%s)", repo_full_name, max_issues, labels)

    with _gh_semaphore:
        issues = fetch_issues_for_repo(repo_full_name, gh, labels, max_issues)
    logger.info("  Fetched %s issues for %s", len(issues), repo_id)

    if not issues:
        return []

    # ── Batch-resolve issue → PR links via GraphQL ──────────────────────
    with _gh_semaphore:
        issue_pr_links = _graphql_resolve_issue_pr_links(repo_full_name, gh, issues, max_events)

    # ── Batch-fetch PR details via GraphQL (cached, avoids REST) ────────
    all_pr_nums: list[int] = []
    for pr_list in issue_pr_links.values():
        all_pr_nums.extend(pr_list)
    if all_pr_nums:
        with _gh_semaphore:
            _pr_graphql_cache[repo_full_name] = _graphql_fetch_pr_details(
                gh, repo_full_name, all_pr_nums
            )
        # Pre-fetch patch text in parallel (under semaphore to avoid bursts)
        with _gh_semaphore:
            _prefetch_patches(
                repo_full_name, gh, all_pr_nums
            )  # 10 workers, fine with parallel_workers=1

    # ── Build records per issue (PR details via REST) ───────────────────
    records: list[dict[str, Any]] = []
    gql_issues: list[Any] = []  # issues resolved via GraphQL
    rest_issues: list[Any] = []  # issues not in GraphQL result → fallback

    for issue in issues:
        if issue.number in issue_pr_links:
            gql_issues.append(issue)
        else:
            rest_issues.append(issue)

    # Batch 1: issues with GraphQL-resolved PRs → parallel per-PR processing
    pw = min(len(gql_issues), 2)
    if gql_issues and pw > 0:
        with ThreadPoolExecutor(max_workers=pw) as pool:
            futures = {
                pool.submit(
                    _build_records_for_issue,
                    issue,
                    issue_pr_links[issue.number],
                    repo_full_name,
                    repo_id,
                    repo_config,
                    test_dirs,
                ): issue.number
                for issue in gql_issues
            }
            for future in as_completed(futures):
                try:
                    records.extend(future.result())
                except Exception as exc:
                    logger.error(
                        "GraphQL-record build for issue %s failed: %s", futures[future], exc
                    )

    # Batch 2: fallback — per-issue REST timeline processing
    pw_rest = min(len(rest_issues), 2)
    if rest_issues and pw_rest > 0:
        with ThreadPoolExecutor(max_workers=pw_rest) as pool:
            futures = {
                pool.submit(
                    _process_single_issue,
                    issue,
                    repo_id,
                    repo_config,
                    max_events,
                    test_dirs,
                ): issue.number
                for issue in rest_issues
            }
            for future in as_completed(futures):
                try:
                    records.extend(future.result())
                except Exception as exc:
                    logger.error("REST fallback for issue %s failed: %s", futures[future], exc)

    logger.info(
        "  %s yielded %s records after resolving linked PRs (graphql=%d, rest_fallback=%d)",
        repo_id,
        len(records),
        len(gql_issues),
        len(rest_issues),
    )
    return records


def load_manifest(manifest_path: str) -> dict[str, Any]:
    """Load and return the repository manifest JSON."""
    import json
    from pathlib import Path

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with path.open() as f:
        return json.load(f)


def get_github_client() -> Github:
    """Create an authenticated PyGithub client.

    Disables built-in throttling (``seconds_between_requests=None``) so
    parallel workers burst to GitHub's secondary limit instead of being
    serialized at 0.25 s spacing.  ``GithubRetry`` (default, 10 retries)
    handles 403/429 reactively.      Connection pool sized for moderate concurrency — 50 parallel threads
    trigger GitHub's secondary rate limit (403 + 28 min backoff).

    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set")
    return Github(
        token,
        per_page=100,
        seconds_between_requests=None,  # ponytail: reactive retry > proactive sleep
        pool_size=10,  # 50 triggers GitHub secondary rate limit
    )


def ingest_all_repos(
    manifest: dict[str, Any],
    config: DataPipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Ingest all repos in *manifest* in parallel.

    Returns ``{repo_id: [raw_record_dicts]}``.
    """
    gh = get_github_client()
    repos = manifest.get("repositories", [])
    results: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []

    def _ingest_one(repo_cfg: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        rid = repo_cfg["id"]
        try:
            records = ingest_repo(repo_cfg, gh, config)
            return rid, records
        except Exception as exc:
            logger.error("Failed to ingest repo %s: %s", rid, exc)
            return rid, []

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as pool:
        futures = {pool.submit(_ingest_one, r): r["id"] for r in repos}
        for future in as_completed(futures):
            rid, records = future.result()
            results[rid] = records

    logger.info(
        "Ingestion complete: %d repos, %d total raw records",
        len(results),
        sum(len(v) for v in results.values()),
    )
    return results
