"""Tests for SWE-bench ingestion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, TestResults
from data_engineering.swebench_ingest import (
    REPO_DOMAIN_MAP,
    SWE_BENCH_PYTHON_REPOS,
    _parse_files_from_patch,
    _parse_test_list,
    _parse_unified_diff,
    swebench_to_issue_record,
)


class TestParseHelpers:
    """Tests for helper parsing functions."""

    def test_parse_test_list_empty(self):
        assert _parse_test_list("") == []
        assert _parse_test_list("   ") == []

    def test_parse_test_list_single(self):
        assert _parse_test_list("test_foo") == ["test_foo"]

    def test_parse_test_list_multiple(self):
        assert _parse_test_list("test_foo test_bar test_baz") == [
            "test_foo",
            "test_bar",
            "test_baz",
        ]

    def test_parse_test_list_with_newlines(self):
        assert _parse_test_list("test_foo\n test_bar\n test_baz") == [
            "test_foo",
            "test_bar",
            "test_baz",
        ]

    def test_parse_files_from_patch(self):
        patch = """--- a/auth.py
+++ b/auth.py
@@ -10,7 +10,8 @@
 def login():
     token = get_token()
 -    set_session(token, ttl=300)
 +    set_session(token, ttl=3600)
     return ok()
 --- a/register.py
+++ b/register.py
@@ -1,5 +1,10 @@
 def register_user(data):
 +    if not data.get('email'):
 +        raise ValueError('Email required')
     return save_user(data)
 """
        files = _parse_files_from_patch(patch)
        assert files == ["auth.py", "register.py"]

    def test_parse_unified_diff_invalid(self) -> None:
        """Invalid diff strings should return empty list without crashing."""
        result = _parse_unified_diff("this is not a diff at all")
        assert result == []


class TestSWEBenchConstants:
    """Tests for SWE-bench constants."""

    def test_python_repos_count(self):
        assert len(SWE_BENCH_PYTHON_REPOS) == 18

    def test_python_repos_contains_expected(self):
        assert "django/django" in SWE_BENCH_PYTHON_REPOS
        assert "psf/black" in SWE_BENCH_PYTHON_REPOS
        assert "pytest-dev/pytest" in SWE_BENCH_PYTHON_REPOS
        assert "huggingface/transformers" in SWE_BENCH_PYTHON_REPOS

    def test_repo_domain_map_complete(self):
        for repo in SWE_BENCH_PYTHON_REPOS:
            assert repo in REPO_DOMAIN_MAP, f"Missing domain for {repo}"

    def test_repo_domain_map_values(self):
        domains = set(REPO_DOMAIN_MAP.values())
        assert "web-api" in domains
        assert "data-ml" in domains
        assert "utils" in domains
        assert "testing" in domains


class TestSWEBenchToIssueRecord:
    """Tests for SWE-bench example -> IssueRecord mapping."""

    @pytest.fixture
    def sample_example(self):
        return {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "base_commit": "abc123def456",
            "patch": (
                "--- a/django/views.py\n+++ b/django/views.py\n"
                "@@ -1,3 +1,4 @@\n def view():\n     pass\n+    return True\n     return False\n"
            ),
            "test_patch": (
                "--- a/tests/test_views.py\n+++ b/tests/test_views.py\n"
                "@@ -1,2 +1,4 @@\n def test_view():\n     pass\n+    assert view() == True\n"
                "+    pass\n"
            ),
            "problem_statement": "View function returns False instead of True",
            "hints_text": "Check the view logic",
            "created_at": "2024-01-15T10:00:00Z",
            "version": "4.2",
            "FAIL_TO_PASS": "tests.test_views.test_view",
            "PASS_TO_PASS": "tests.test_views.test_other",
            "environment_setup_commit": "def456abc123",
        }

    def test_maps_all_fields(self, sample_example):
        record = swebench_to_issue_record(sample_example, "web-api")

        assert isinstance(record, IssueRecord)
        assert record.issue_id == "django__django-12345"
        assert record.repo == "django/django"
        assert record.issue_body == "View function returns False instead of True"
        assert record.patch_diff == sample_example["patch"]
        assert record.pr_title == ""
        assert record.pr_description == ""
        assert record.commit_messages == []
        assert record.issue_labels == []
        assert record.repo_domain == "web-api"

    def test_maps_test_results(self, sample_example):
        record = swebench_to_issue_record(sample_example, "web-api")

        assert isinstance(record.test_results, TestResults)
        assert record.test_results.failed == ["tests.test_views.test_view"]
        assert record.test_results.passed == ["tests.test_views.test_other"]
        assert record.test_results.errored == []

    def test_maps_metadata(self, sample_example):
        record = swebench_to_issue_record(sample_example, "web-api")

        meta = record.metadata
        assert meta["base_sha"] == "abc123def456"
        assert meta["head_sha"] == "def456abc123"
        assert meta["version"] == "4.2"
        assert meta["hints"] == "Check the view logic"
        assert meta["created_at"] == "2024-01-15T10:00:00Z"
        assert meta["has_test_patch"] is True
        assert meta["instance_id"] == "django__django-12345"

    def test_maps_test_files(self, sample_example):
        record = swebench_to_issue_record(sample_example, "web-api")

        assert "tests/test_views.py" in record.test_files_changed
        assert "django/views.py" in record.files_changed

    def test_parses_hunks(self, sample_example):
        record = swebench_to_issue_record(sample_example, "web-api")

        assert len(record.parsed_hunks) > 0
        hunk = record.parsed_hunks[0]
        assert hunk.file == "django/views.py"
        assert hunk.old_start == 1
        assert hunk.new_start == 1

    def test_empty_fail_to_pass(self):
        example = {
            "instance_id": "test-1",
            "repo": "django/django",
            "base_commit": "abc123",
            "patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
            "test_patch": "",
            "problem_statement": "Test",
            "hints_text": "",
            "created_at": "2024-01-01",
            "version": "1.0",
            "FAIL_TO_PASS": "",
            "PASS_TO_PASS": "",
            "environment_setup_commit": "def456",
        }
        record = swebench_to_issue_record(example, "web-api")
        assert record.test_results.failed == []
        assert record.test_results.passed == []
        assert record.metadata["has_test_patch"] is False

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields like hints_text."""
        example = {
            "instance_id": "test-1",
            "repo": "django/django",
            "base_commit": "abc123",
            "patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
            "test_patch": "",
            "problem_statement": "Test",
            "hints_text": None,  # Can be None in dataset
            "created_at": "2024-01-01",
            "version": "1.0",
            "FAIL_TO_PASS": "",
            "PASS_TO_PASS": "",
            "environment_setup_commit": "def456",
        }
        record = swebench_to_issue_record(example, "web-api")
        assert record.metadata["hints"] == ""


class TestSaveSWEBenchSplits:
    """Tests for save_swebench_splits function."""

    def test_save_swebench_splits(self, tmp_path) -> None:
        """save_swebench_splits should write JSONL files."""
        from data_engineering.config import DataPipelineConfig
        from data_engineering.swebench_ingest import save_swebench_splits

        example = {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "base_commit": "abc123def456",
            "patch": (
                "--- a/django/views.py\n+++ b/django/views.py\n"
                "@@ -1,3 +1,4 @@\n def view():\n     pass\n+    return True\n     return False\n"
            ),
            "test_patch": "",
            "problem_statement": "View function returns False instead of True",
            "hints_text": "",
            "created_at": "2024-01-15T10:00:00Z",
            "version": "4.2",
            "FAIL_TO_PASS": "",
            "PASS_TO_PASS": "",
            "environment_setup_commit": "def456abc123",
        }
        record = swebench_to_issue_record(example, "web-api")
        config = DataPipelineConfig()
        paths = save_swebench_splits([record], tmp_path, config)
        assert "raw" in paths
        assert paths["raw"].exists()
        content = paths["raw"].read_text()
        assert "django__django-12345" in content


class TestBigQueryCache:
    """Tests for BigQuery stats cache functions."""

    def test_bq_cache_path(self) -> None:
        from data_engineering.config import DataPipelineConfig
        from data_engineering.swebench_ingest import _get_bq_stats_cache_path

        path = _get_bq_stats_cache_path(DataPipelineConfig())
        assert str(path).endswith("bigquery_repo_stats.json")

    def test_bq_load_cache_not_exists(self, tmp_path) -> None:
        from data_engineering.swebench_ingest import _load_bq_stats_cache

        result = _load_bq_stats_cache(tmp_path / "nonexistent.json")
        assert result == {}

    def test_bq_save_and_load_cache(self, tmp_path) -> None:
        from data_engineering.swebench_ingest import _load_bq_stats_cache, _save_bq_stats_cache

        data = {"django/django": {"stars": 100}}
        p = tmp_path / "cache.json"
        _save_bq_stats_cache(p, data)
        loaded = _load_bq_stats_cache(p)
        assert loaded["django/django"]["stars"] == 100


class TestFetchRepoStats:
    """Tests for BigQuery repo stats fetching."""

    @patch("google.cloud.bigquery.Client")
    def test_fetch_repo_stats(self, mock_bq_client_cls) -> None:
        from data_engineering.config import DataPipelineConfig
        from data_engineering.swebench_ingest import _fetch_repo_stats

        mock_client = mock_bq_client_cls.return_value
        mock_rows = MagicMock()
        mock_rows.__iter__.return_value = iter(
            [
                MagicMock(repo_name="django/django", stars=5000, license="BSD"),
            ]
        )
        mock_client.query.return_value.result.return_value = mock_rows

        config = DataPipelineConfig(bigquery_project="test-project")
        result = _fetch_repo_stats(config, {"django/django"})
        assert "django/django" in result
        assert result["django/django"]["stars"] == 5000


class TestAugmentWithBigQuery:
    """Tests for augment_with_bigquery function."""

    def test_augment_with_bigquery_disabled(self, tmp_path) -> None:
        """When bigquery_enabled=False, returns records unchanged."""
        from data_engineering.swebench_ingest import augment_with_bigquery

        config = DataPipelineConfig(bigquery_enabled=False, swe_bench_dir=tmp_path)
        record = IssueRecord.model_construct(
            issue_id="test-1",
            repo="owner/repo",
            issue_body="test",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
        )
        result = augment_with_bigquery([record], config)
        assert len(result) == 1
        assert "bigquery_repo_stats" not in result[0].metadata

    def test_augment_with_bigquery_cached(self, tmp_path) -> None:
        """When cache exists, no BQ query made."""
        from data_engineering.swebench_ingest import _get_bq_stats_cache_path, augment_with_bigquery

        config = DataPipelineConfig(bigquery_enabled=True, swe_bench_dir=tmp_path)
        cache_path = _get_bq_stats_cache_path(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text('{"owner/repo": {"stars": 100}}')

        record = IssueRecord.model_construct(
            issue_id="test-1",
            repo="owner/repo",
            issue_body="test",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
        )

        with patch("data_engineering.swebench_ingest._fetch_repo_stats") as mock_fetch:
            result = augment_with_bigquery([record], config)

        assert len(result) == 1
        assert result[0].metadata.get("bigquery_repo_stats") == {"stars": 100}
        mock_fetch.assert_not_called()

    def test_augment_with_bigquery_query_fails_no_cache(self, tmp_path) -> None:
        """BQ fails, no cache, returns records unchanged."""
        from data_engineering.swebench_ingest import augment_with_bigquery

        config = DataPipelineConfig(bigquery_enabled=True, swe_bench_dir=tmp_path)

        record = IssueRecord.model_construct(
            issue_id="test-1",
            repo="owner/repo",
            issue_body="test",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
        )

        with patch("data_engineering.swebench_ingest._fetch_repo_stats") as mock_fetch:
            mock_fetch.side_effect = Exception("BQ unavailable")
            result = augment_with_bigquery([record], config)

        assert len(result) == 1
        assert "bigquery_repo_stats" not in result[0].metadata

    def test_augment_with_bigquery_query_fails_with_cache(self, tmp_path) -> None:
        """BQ fails, cache exists, uses cached data."""
        from data_engineering.swebench_ingest import _get_bq_stats_cache_path, augment_with_bigquery

        config = DataPipelineConfig(bigquery_enabled=True, swe_bench_dir=tmp_path)
        cache_path = _get_bq_stats_cache_path(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text('{"owner/repo": {"stars": 100}}')

        record = IssueRecord.model_construct(
            issue_id="test-1",
            repo="owner/repo",
            issue_body="test",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-foo\n+bar\n",
        )

        with patch("data_engineering.swebench_ingest._fetch_repo_stats") as mock_fetch:
            mock_fetch.side_effect = Exception("BQ unavailable")
            result = augment_with_bigquery([record], config)

        assert len(result) == 1
        assert result[0].metadata.get("bigquery_repo_stats") == {"stars": 100}


def make_mock_dataset(items):
    """Create a mock dataset object with filter method that actually filters."""
    mock_ds = MagicMock()

    def filter_fn(fn):
        filtered = [item for item in items if fn(item)]
        return filtered

    mock_ds.filter.side_effect = filter_fn
    mock_ds.__iter__.return_value = iter(items)
    return mock_ds


class TestLoadSWEBenchSplits:
    """Tests for load_swebench_splits (mocked)."""

    @patch("data_engineering.swebench_ingest.load_dataset")
    def test_loads_all_splits(self, mock_load_dataset):
        mock_load_dataset.side_effect = [
            make_mock_dataset([{"repo": "django/django", "instance_id": "v1"}]),  # verified
            make_mock_dataset([{"repo": "django/django", "instance_id": "t1"}]),  # test
            make_mock_dataset([{"repo": "django/django", "instance_id": "d1"}]),  # dev
            make_mock_dataset(
                [
                    {"repo": "django/django", "instance_id": "tr1"},
                    {"repo": "other/repo", "instance_id": "tr2"},
                ]
            ),  # train
        ]

        from data_engineering.swebench_ingest import load_swebench_splits

        config = DataPipelineConfig()
        splits = load_swebench_splits(config)

        assert "verified" in splits
        assert "test" in splits
        assert "dev" in splits
        assert "train" in splits
        assert len(splits["verified"]) == 1
        assert len(splits["test"]) == 1
        assert len(splits["dev"]) == 1
        # Train filtered to Python repos only
        assert len(splits["train"]) == 1
        assert splits["train"][0]["repo"] == "django/django"

    @patch("data_engineering.swebench_ingest.load_dataset")
    def test_train_filters_python_repos(self, mock_load_dataset):
        mock_load_dataset.side_effect = [
            make_mock_dataset([]),  # verified
            make_mock_dataset([]),  # test
            make_mock_dataset([]),  # dev
            make_mock_dataset(
                [
                    {"repo": "django/django", "instance_id": "1"},  # Python
                    {"repo": "microsoft/vscode", "instance_id": "2"},  # Not Python
                    {"repo": "psf/black", "instance_id": "3"},  # Python
                ]
            ),
        ]

        from data_engineering.swebench_ingest import load_swebench_splits

        config = DataPipelineConfig()
        splits = load_swebench_splits(config)

        assert len(splits["train"]) == 2
        repos = {ex["repo"] for ex in splits["train"]}
        assert repos == {"django/django", "psf/black"}


class TestIngestSWEBench:
    """Integration-style test for ingest_swebench (mocked)."""

    @patch("data_engineering.swebench_ingest.load_swebench_splits")
    def test_ingest_returns_records(self, mock_load_splits):
        mock_load_splits.return_value = {
            "verified": [
                {
                    "instance_id": "v1",
                    "repo": "django/django",
                    "base_commit": "abc",
                    "patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n",
                    "test_patch": "",
                    "problem_statement": "Fix bug",
                    "hints_text": "",
                    "created_at": "2024-01-01",
                    "version": "1.0",
                    "FAIL_TO_PASS": "test_foo",
                    "PASS_TO_PASS": "test_bar",
                    "environment_setup_commit": "def",
                }
            ],
            "test": [],
            "dev": [],
            "train": [],
        }

        from data_engineering.swebench_ingest import ingest_swebench

        config = DataPipelineConfig()
        records = ingest_swebench(config)

        assert len(records) == 1
        assert records[0].issue_id == "v1"
        assert records[0].repo == "django/django"

    @patch("data_engineering.swebench_ingest.load_swebench_splits")
    def test_ingest_skips_empty_patches(self, mock_load_splits):
        """Examples with empty patches should be skipped with a warning."""
        mock_load_splits.return_value = {
            "verified": [
                {
                    "instance_id": "v1",
                    "repo": "django/django",
                    "base_commit": "abc",
                    "patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n",
                    "test_patch": "",
                    "problem_statement": "Fix bug",
                    "hints_text": "",
                    "created_at": "2024-01-01",
                    "version": "1.0",
                    "FAIL_TO_PASS": "test_foo",
                    "PASS_TO_PASS": "test_bar",
                    "environment_setup_commit": "def",
                },
                {
                    "instance_id": "v2",
                    "repo": "django/django",
                    "base_commit": "abc",
                    "patch": "",  # empty patch — should be skipped
                    "test_patch": "",
                    "problem_statement": "Empty patch",
                    "hints_text": "",
                    "created_at": "2024-01-01",
                    "version": "1.0",
                    "FAIL_TO_PASS": "",
                    "PASS_TO_PASS": "",
                    "environment_setup_commit": "def",
                },
            ],
            "test": [],
            "dev": [],
            "train": [],
        }

        from data_engineering.swebench_ingest import ingest_swebench

        config = DataPipelineConfig()
        records = ingest_swebench(config)

        assert len(records) == 1  # only v1 kept, v2 skipped
        assert records[0].issue_id == "v1"
        assert records[0].repo_domain == "web-api"
