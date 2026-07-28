"""Tests for data_engineering.clean."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering.clean import (
    clean_records,
    deduplicate,
)
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config() -> DataPipelineConfig:
    return DataPipelineConfig(
        max_patch_lines=500,
        test_directories=["tests/", "test/"],
    )


def _make_record(
    issue_id: str,
    repo: str = "owner/repo",
    test_files: list[str] | None = None,
    patch_diff: str | None = None,
    issue_body: str = "Test issue body for pipeline",
    files_changed: list[str] | None = None,
    commit_messages: list[str] | None = None,
    pr_description: str = "",
) -> IssueRecord:
    if patch_diff is None:
        patch_diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n"
    return IssueRecord(
        issue_id=issue_id,
        repo=repo,
        issue_body=issue_body,
        patch_diff=patch_diff,
        test_files_changed=test_files or [],
        files_changed=files_changed or ["foo.py"],
        commit_messages=commit_messages or ["fix: some fix"],
        pr_description=pr_description,
        issue_labels=["bug"],
    )


class TestDeduplicate:
    def test_unique_records(self) -> None:
        records = [
            _make_record("repo#1", "owner/repo"),
            _make_record(
                "repo#2",
                "owner/repo",
                patch_diff="--- a/bar.py\n+++ b/bar.py\n@@ -1,1 +1,2 @@\n-1\n+1\n+2\n",
            ),
        ]
        deduped, stats = deduplicate(records)
        assert len(deduped) == 2
        assert stats.exact_duplicates_removed == 0
        assert stats.content_duplicates_removed == 0

    def test_exact_duplicate_removed(self) -> None:
        r = _make_record("repo#1", "owner/repo")
        records = [r, r.model_copy(deep=True)]
        deduped, stats = deduplicate(records)
        assert len(deduped) == 1
        assert stats.exact_duplicates_removed == 1

    def test_content_duplicate_different_id(self) -> None:
        """Same diff hash = content duplicate even with different IDs."""
        r1 = _make_record("repo#1", "owner/repo")
        r2 = _make_record("repo#2", "owner/repo")
        r2.patch_diff = r1.patch_diff  # same content
        # Different issue_id means different `_id_hash` but same `_content_hash`
        records = [r1, r2]
        deduped, stats = deduplicate(records)
        assert len(deduped) == 1
        assert stats.content_duplicates_removed == 1

    def test_empty_input(self) -> None:
        deduped, stats = deduplicate([])
        assert deduped == []
        assert stats.total_input == 0


class TestCleanRecords:
    def test_passes_valid_record(self, config: DataPipelineConfig) -> None:
        records = [_make_record("repo#1", test_files=["test_foo.py"])]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 1
        assert stats.removed_no_test_files == 0

    def test_removes_no_test_files(self, config: DataPipelineConfig) -> None:
        records = [_make_record("repo#1", test_files=[])]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 0
        assert stats.removed_no_test_files == 1

    def test_removes_patch_too_large(self, config: DataPipelineConfig) -> None:
        big_patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,3 @@\n" + "\n".join(
            f"+line{i}" for i in range(600)
        )
        records = [_make_record("repo#1", test_files=["test_foo.py"], patch_diff=big_patch)]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 0
        assert stats.removed_patch_too_large == 1

    def test_no_f2p_signal_removed(self, config: DataPipelineConfig) -> None:
        records = [
            _make_record(
                "repo#1",
                test_files=["test_foo.py"],
                commit_messages=["chore: some cleanup"],
                pr_description="No keywords here",
            )
        ]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 0
        assert stats.removed_no_f2p_signal == 1

    def test_empty_body_removed(self, config: DataPipelineConfig) -> None:
        """Empty body should be filtered out by clean, not schema.

        Use ``model_construct`` to bypass the IssueRecord field validator
        (which rejects empty body at construction time).
        """
        from data_engineering.schema import IssueRecord

        rec = IssueRecord.model_construct(
            issue_id="repo#1",
            repo="owner/repo",
            issue_body="",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n",
            test_files_changed=["test_foo.py"],
            files_changed=["foo.py"],
            commit_messages=["fix: some fix"],
            issue_labels=["bug"],
        )
        cleaned, stats = clean_records([rec], config)
        assert len(cleaned) == 0
        assert stats.removed_empty_body == 1

    def test_empty_input(self, config: DataPipelineConfig) -> None:
        cleaned, stats = clean_records([], config)
        assert cleaned == []
        assert stats.total_input == 0

    def test_binary_diff_removed(self, config: DataPipelineConfig) -> None:
        """Binary diffs should be detected and removed."""
        binary_diff = (
            "Binary files a/foo.py and b/foo.py differ\n"
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )
        records = [_make_record("repo#1", test_files=["test_foo.py"], patch_diff=binary_diff)]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 0
        assert stats.removed_binary == 1

    def test_removes_non_python_dominated(self, config: DataPipelineConfig) -> None:
        """Records with <50% Python files should be removed."""
        records = [
            _make_record(
                "repo#1",
                test_files=["test_foo.py"],
                files_changed=["foo.js", "bar.js", "baz.js", "foo.py"],
            )
        ]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 0
        assert stats.removed_non_python == 1

    def test_training_records_without_verified_kept(self, config: DataPipelineConfig) -> None:
        """is_verified=False records survive no_test_files and no_f2p_signal."""
        rec = IssueRecord.model_construct(
            issue_id="train#1",
            repo="owner/repo",
            issue_body="Training example",
            patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n",
            test_files_changed=[],  # no test files
            files_changed=["foo.py"],
            commit_messages=["chore: cleanup"],  # no F2P keywords
            metadata={"is_verified": False},
        )
        cleaned, stats = clean_records([rec], config)
        assert len(cleaned) == 1  # kept despite no test files + no F2P

    def test_zero_output_logs_warning(self, config: DataPipelineConfig, caplog) -> None:
        """When all records are removed, a warning should be logged."""
        import logging

        caplog.set_level(logging.WARNING)
        rec = _make_record("repo#1", test_files=[])
        cleaned, stats = clean_records([rec], config)
        assert len(cleaned) == 0
        assert "0 records" in caplog.text

    def test_non_python_files_warns_but_keeps(self, config: DataPipelineConfig) -> None:
        """Mixed Python/non-Python files should warn but keep record."""
        records = [
            _make_record(
                "repo#1",
                test_files=["test_foo.py"],
                files_changed=["foo.py", "foo.js"],
            )
        ]
        cleaned, stats = clean_records(records, config)
        assert len(cleaned) == 1  # kept
        assert len(stats.warnings_non_python) >= 1
