"""Tests for data_engineering.ingest.

Covers IngestionConfig, diff parsing, GitHub API fetching, repo ingestion
orchestration, manifest loading, and client creation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_engineering.ingest import (
    IngestionConfig,
    _parse_unified_diff,
    fetch_issues_for_repo,
    fetch_linked_prs,
    fetch_pr_details,
    get_github_client,
    ingest_repo,
    load_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_GITHUB = FIXTURES / "mock_github_responses"


# ── helpers ─────────────────────────────────────────────────────────────────


def _load_mock(name: str) -> list[dict]:
    with (MOCK_GITHUB / name).open() as f:
        return json.load(f)


def _make_issue(data: dict) -> MagicMock:
    """Build a mock PyGithub Issue from a fixture dict."""
    issue = MagicMock()
    issue.number = data["number"]
    issue.title = data.get("title", "")
    issue.body = data.get("body", "")
    issue.state = data.get("state", "open")
    issue.html_url = data.get("html_url", "")
    issue.labels = [MagicMock(name=l["name"]) for l in data.get("labels", [])]
    for lbl in issue.labels:
        lbl.name = lbl._extract_mock_name()
    issue.pull_request = None
    return issue


def _make_timeline(events: list) -> MagicMock:
    """Build a mock PaginatedList timeline that supports ``get_page(0)``."""
    tl = MagicMock()
    tl.get_page.return_value = events
    return tl


def _make_pr_files() -> list[MagicMock]:
    """Build mock PyGithub File objects from pr_files.json."""
    files = []
    for d in _load_mock("pr_files.json"):
        pf = MagicMock()
        pf.filename = d["filename"]
        pf.status = d["status"]
        pf.patch = d.get("patch", "")
        pf.additions = d["additions"]
        pf.deletions = d["deletions"]
        files.append(pf)
    return files


def _make_pr_commits() -> list[MagicMock]:
    """Build mock PyGithub Commit objects from pr_commits.json."""
    commits = []
    for d in _load_mock("pr_commits.json"):
        c = MagicMock()
        c.sha = d["sha"]
        c.commit.message = d["commit"]["message"]
        commits.append(c)
    return commits


# ── IngestionConfig ─────────────────────────────────────────────────────────


class TestIngestionConfig:
    def test_defaults(self) -> None:
        cfg = IngestionConfig()
        assert cfg.issue_labels_to_include == []
        assert cfg.max_issues_per_repo == 2000
        assert cfg.test_directories == ["tests/", "test/"]

    def test_custom_values(self) -> None:
        cfg = IngestionConfig(
            issue_labels_to_include=["bug"],
            max_issues_per_repo=100,
            test_directories=["spec/"],
        )
        assert cfg.max_issues_per_repo == 100

    def test_model_roundtrip(self) -> None:
        cfg = IngestionConfig(issue_labels_to_include=["fix"])
        d = cfg.model_dump()
        restored = IngestionConfig(**d)
        assert restored.issue_labels_to_include == ["fix"]


# ── _parse_unified_diff ─────────────────────────────────────────────────────


class TestParseUnifiedDiff:
    def test_parses_valid_diff(self) -> None:
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,3 @@\n x=1\n y=2\n+z=3\n"
        hunks = _parse_unified_diff(diff)
        assert len(hunks) == 1
        assert hunks[0].file == "foo.py"
        assert hunks[0].old_lines == 2
        assert hunks[0].new_lines == 3

    def test_parses_multi_file_diff(self) -> None:
        diff = (
            "--- a/a.py\n+++ b/a.py\n"
            "@@ -1 +1,2 @@\n-x\n+y\n+z\n"
            "--- a/b.py\n+++ b/b.py\n"
            "@@ -1 +1,2 @@\n-a\n+b\n+c\n"
        )
        hunks = _parse_unified_diff(diff)
        assert len(hunks) == 2

    def test_empty_diff(self) -> None:
        assert _parse_unified_diff("") == []

    def test_invalid_diff(self) -> None:
        assert _parse_unified_diff("not a diff at all") == []


# ── fetch_issues_for_repo ───────────────────────────────────────────────────


class TestFetchIssuesForRepo:
    def test_fetches_all_issues(self) -> None:
        mock_gh = MagicMock()
        mock_repo = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        issues_data = _load_mock("issues_page1.json")
        mock_issues = [_make_issue(d) for d in issues_data]
        mock_repo.get_issues.return_value = mock_issues

        result = fetch_issues_for_repo("owner/repo1", mock_gh, ["bug"], 100)
        assert len(result) == 2
        assert result[0].number == 42
        mock_repo.get_issues.assert_called_once()

    def test_respects_max_issues(self) -> None:
        mock_gh = MagicMock()
        mock_repo = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        issues_data = _load_mock("issues_page1.json")
        mock_issues = [_make_issue(d) for d in issues_data]
        mock_repo.get_issues.return_value = mock_issues

        result = fetch_issues_for_repo("owner/repo1", mock_gh, ["bug"], 1)
        assert len(result) == 1

    def test_empty_repo(self) -> None:
        mock_gh = MagicMock()
        mock_repo = MagicMock()
        mock_gh.get_repo.return_value = mock_repo
        mock_repo.get_issues.return_value = []

        result = fetch_issues_for_repo("owner/empty", mock_gh, ["bug"], 100)
        assert result == []


# ── fetch_linked_prs ────────────────────────────────────────────────────────


class TestFetchLinkedPRs:
    def test_detects_cross_referenced_pr(self) -> None:
        mock_issue = MagicMock()
        mock_pr = MagicMock()
        mock_repo = MagicMock()
        mock_issue.repository = mock_repo

        # Build timeline event with cross-reference
        event = MagicMock()
        event.event = "cross-referenced"
        source = MagicMock()
        src_issue = MagicMock()
        src_issue.number = 142
        src_issue.pull_request = MagicMock()
        source.issue = src_issue
        event.source = source

        mock_issue.get_timeline.return_value = _make_timeline([event])

        # Mock the PR lookup via issue.repository
        mock_repo.get_pull.return_value = mock_pr
        mock_pr.merged = True
        mock_pr.number = 142

        result = fetch_linked_prs(mock_issue, 100)
        assert len(result) == 1
        assert result[0].number == 142

    def test_skips_non_merged_pr(self) -> None:
        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_issue.repository = mock_repo

        event = MagicMock()
        event.event = "cross-referenced"
        source = MagicMock()
        src_issue = MagicMock()
        src_issue.number = 142
        src_issue.pull_request = MagicMock()
        source.issue = src_issue
        event.source = source
        mock_issue.get_timeline.return_value = _make_timeline([event])

        mock_pr = MagicMock()
        mock_pr.merged = False
        mock_repo.get_pull.return_value = mock_pr

        result = fetch_linked_prs(mock_issue, 100)
        assert len(result) == 0

    def test_skips_non_cross_referenced(self) -> None:
        mock_issue = MagicMock()

        event = MagicMock()
        event.event = "labeled"  # not cross-referenced
        mock_issue.get_timeline.return_value = _make_timeline([event])

        result = fetch_linked_prs(mock_issue, 100)
        assert result == []

    def test_uses_get_page_zero(self) -> None:
        """Verify only first page of timeline is fetched (perf optimization)."""
        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_issue.repository = mock_repo

        tl = MagicMock()
        tl.get_page.return_value = []  # empty page
        mock_issue.get_timeline.return_value = tl

        fetch_linked_prs(mock_issue, 100)
        # Must call get_page(0) — not iterate full paginated list
        tl.get_page.assert_called_once_with(0)

    def test_enforces_max_events_from_single_page(self) -> None:
        """get_page(0) returns limited results; max_events is still honored."""
        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_issue.repository = mock_repo

        events = []
        for i in range(10):
            e = MagicMock()
            e.event = "cross-referenced"
            src = MagicMock()
            si = MagicMock()
            si.number = 100 + i
            si.pull_request = MagicMock()
            src.issue = si
            e.source = src
            events.append(e)

        mock_issue.get_timeline.return_value = _make_timeline(events)
        mock_repo.get_pull.return_value.merged = True

        result = fetch_linked_prs(mock_issue, 3)
        assert len(result) == 3


# ── github_retry ────────────────────────────────────────────────────────────


class TestGithubRetry:
    def test_retries_on_secondary_rate_limit(self) -> None:
        """Backoff handles 403 secondary rate limit without proactive limiter."""
        from data_engineering.ingest import github_retry

        call_count = 0

        @github_retry
        def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                exc = GithubException(403, {"message": "secondary rate limit"})
                raise exc
            return "ok"

        from github import GithubException

        result = flaky_call()
        assert result == "ok"
        assert call_count == 2

    def test_raises_on_401(self) -> None:
        """Non-recoverable status codes are not retried."""
        from github import GithubException

        from data_engineering.ingest import github_retry

        @github_retry
        def auth_fail() -> None:
            raise GithubException(401, {"message": "bad credentials"})

        with pytest.raises(GithubException):
            auth_fail()


# ── fetch_pr_details ────────────────────────────────────────────────────────


class TestFetchPRDetails:
    def test_extracts_files_and_commits(self) -> None:
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = _make_pr_files()
        mock_pr.get_commits.return_value = _make_pr_commits()

        details = fetch_pr_details(mock_pr)

        assert "auth.py" in details["files_changed"]
        assert "tests/test_auth.py" in details["files_changed"]
        assert len(details["commit_messages"]) == 2
        assert "patch_diff" in details
        assert "--- a/auth.py" in details["patch_diff"]
        # test_files_changed removed from fetch_pr_details;
        # caller computes it from files_changed with actual test_dirs
        assert "test_files_changed" not in details

    def test_empty_pr(self) -> None:
        mock_pr = MagicMock()
        mock_pr.get_files.return_value = []
        mock_pr.get_commits.return_value = []

        details = fetch_pr_details(mock_pr)
        assert details["files_changed"] == []
        assert details["patch_diff"] == ""
        assert details["commit_messages"] == []

    def test_process_single_issue_still_computes_test_files(self) -> None:
        """Caller (_process_single_issue) recomputes test_files_changed
        from files_changed using the actual config's test_dirs."""
        from data_engineering.ingest import _process_single_issue

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.body = "Bug report"
        mock_issue.html_url = "https://github.com/o/r/issues/42"
        mock_issue.labels = []

        mock_pr = MagicMock()
        mock_pr.title = "Fix the bug"
        mock_pr.body = "Closes #42"
        mock_pr.number = 142
        mock_pr.html_url = "https://github.com/o/r/pull/142"
        mock_pr.merged_at = None
        mock_pr.base.sha = "b1"
        mock_pr.head.sha = "h1"
        mock_pr.get_files.return_value = _make_pr_files()
        mock_pr.get_commits.return_value = _make_pr_commits()

        with patch("data_engineering.ingest.fetch_linked_prs", return_value=[mock_pr]):
            records = _process_single_issue(
                mock_issue,
                "owner/repo1",
                {"domain_category": "web"},
                100,
                test_dirs=["tests/", "test/"],
            )

        assert len(records) == 1
        rec = records[0]
        assert "test_files_changed" in rec
        assert "tests/test_auth.py" in rec["test_files_changed"]


# ── ingest_repo ─────────────────────────────────────────────────────────────


class TestIngestRepo:
    def test_ingests_single_repo(self) -> None:
        """End-to-end ingest with fully mocked PyGithub + GraphQL."""
        mock_gh = MagicMock()
        mock_repo = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        issues_data = _load_mock("issues_page1.json")
        mock_issues = [_make_issue(d) for d in issues_data]
        mock_repo.get_issues.return_value = mock_issues

        pr_details = {
            142: {
                "title": "Fix session expiry",
                "body": "Closes #42",
                "merged": True,
                "merged_at": None,
                "base_sha": "base123",
                "head_sha": "head456",
                "url": "https://github.com/owner/repo1/pull/142",
                "files_changed": ["src/auth.py", "tests/test_auth.py"],
                "commit_messages": ["Fix session expiry bug"],
            }
        }

        with (
            patch(
                "data_engineering.ingest._graphql_resolve_issue_pr_links",
                return_value={42: [142], 43: [142]},
            ),
            patch("data_engineering.ingest._graphql_fetch_pr_details", return_value=pr_details),
        ):
            repo_config = {
                "id": "owner/repo1",
                "domain_category": "github.com",
                "stars": 100,
                "ingestion_config": {},
            }
            from data_engineering.config import DataPipelineConfig

            config = DataPipelineConfig(
                max_issues_per_repo=100,
                max_events_per_issue=100,
            )

            records = ingest_repo(repo_config, mock_gh, config)

        assert len(records) > 0
        record = records[0]
        assert record["repo"] == "owner/repo1"
        assert record["issue_id"] == "42"
        assert "patch_diff" in record
        assert "files_changed" in record

    def test_empty_repo_returns_empty(self) -> None:
        mock_gh = MagicMock()
        mock_repo = MagicMock()
        mock_gh.get_repo.return_value = mock_repo
        mock_repo.get_issues.return_value = []

        from data_engineering.config import DataPipelineConfig

        config = DataPipelineConfig()

        repo_config = {"id": "owner/empty", "ingestion_config": {}}
        records = ingest_repo(repo_config, mock_gh, config)
        assert records == []


# ── load_manifest ────────────────────────────────────────────────────────────


class TestLoadManifest:
    def test_loads_json_file(self) -> None:
        manifest = load_manifest(str(FIXTURES / "sample_issues.json"))
        assert isinstance(manifest, list)

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/manifest.json")


# ── get_github_client ───────────────────────────────────────────────────────


class TestGetGithubClient:
    def test_creates_client_with_token(self) -> None:
        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}):
            with patch("data_engineering.ingest.Github") as mock_gh_cls:
                get_github_client()
                mock_gh_cls.assert_called_once_with(
                    "fake-token",
                    per_page=100,
                    seconds_between_requests=None,
                    pool_size=10,
                )

    def test_raises_without_token(self) -> None:
        with patch.dict(os.environ, clear=True):
            with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
                get_github_client()
