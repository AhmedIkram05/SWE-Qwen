"""Tests for data_engineering.schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_engineering.schema import (
    CleanStats,
    DedupStats,
    GoldenSet,
    IssueRecord,
    ParsedHunk,
    PipelineResult,
    PipelineStats,
    RepoResult,
    Splits,
    TestResults,
    ValidationError,
    ValidationResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def valid_issues() -> list[dict]:
    with (FIXTURES / "sample_issues.json").open() as f:
        return json.load(f)


@pytest.fixture
def invalid_issues() -> list[dict]:
    with (FIXTURES / "sample_invalid_issues.json").open() as f:
        return json.load(f)


# ── ParsedHunk ─────────────────────────────────────────────────────────────


class TestParsedHunk:
    def test_minimal(self) -> None:
        h = ParsedHunk(
            file="foo.py",
            old_start=1,
            old_lines=5,
            new_start=1,
            new_lines=6,
            diff_lines=["+new line"],
        )
        assert h.file == "foo.py"
        assert h.old_start == 1
        assert len(h.diff_lines) == 1


# ── TestResults ────────────────────────────────────────────────────────────


class TestTestResults:
    def test_defaults(self) -> None:
        tr = TestResults()
        assert tr.passed == []
        assert tr.failed == []
        assert tr.errored == []

    def test_with_data(self) -> None:
        tr = TestResults(passed=["test_a"], failed=["test_b"])
        assert len(tr.passed) == 1


# ── IssueRecord ────────────────────────────────────────────────────────────


class TestIssueRecord:
    def test_valid_record(self, valid_issues: list[dict]) -> None:
        rec = IssueRecord(**valid_issues[0])
        assert rec.repo == "owner/repo1"
        assert "auth.py" in rec.patch_diff

    def test_empty_body_raises(self) -> None:
        data = {
            "issue_id": "test#1",
            "repo": "t/r",
            "issue_body": "   ",
            "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
        }
        with pytest.raises(ValueError):
            IssueRecord(**data)

    def test_empty_patch_diff_raises(self) -> None:
        data = {
            "issue_id": "test#2",
            "repo": "t/r",
            "issue_body": "body",
            "patch_diff": "",
        }
        with pytest.raises(ValueError):
            IssueRecord(**data)

    def test_invalid_diff_raises(self) -> None:
        data = {
            "issue_id": "test#3",
            "repo": "t/r",
            "issue_body": "body",
            "patch_diff": "not a diff at all",
        }
        with pytest.raises(ValueError):
            IssueRecord(**data)

    def test_deserialise_all_valid(self, valid_issues: list[dict]) -> None:
        for data in valid_issues:
            rec = IssueRecord(**data)
            assert isinstance(rec.parsed_hunks, list)
            assert isinstance(rec.test_results, TestResults)
            assert isinstance(rec.metadata, dict)

    def test_serialise_roundtrip(self, valid_issues: list[dict]) -> None:
        rec = IssueRecord(**valid_issues[0])
        dumped = rec.model_dump()
        restored = IssueRecord(**dumped)
        assert restored.issue_id == rec.issue_id
        assert restored.patch_diff == rec.patch_diff
        assert restored.issue_body == rec.issue_body


# ── Validation models ──────────────────────────────────────────────────────


class TestValidationError:
    def test_create(self) -> None:
        ve = ValidationError(record_id="r1", field="body", error="empty", raw_value="")
        assert ve.record_id == "r1"


class TestValidationResult:
    def test_valid(self) -> None:
        vr = ValidationResult(valid=True)
        assert vr.valid
        assert vr.record is None

    def test_invalid_with_errors(self) -> None:
        vr = ValidationResult(
            valid=False,
            errors=[ValidationError(record_id="r1", field="body", error="empty")],
        )
        assert not vr.valid
        assert len(vr.errors) == 1


# ── Stats models ───────────────────────────────────────────────────────────


class TestDedupStats:
    def test_defaults(self) -> None:
        ds = DedupStats()
        assert ds.total_input == 0
        assert ds.unique_output == 0

    def test_with_values(self) -> None:
        ds = DedupStats(
            total_input=100,
            exact_duplicates_removed=10,
            content_duplicates_removed=5,
            unique_output=85,
        )
        assert ds.unique_output == 85


class TestCleanStats:
    def test_defaults(self) -> None:
        cs = CleanStats()
        assert cs.total_output == 0
        assert cs.warnings_non_python == []

    def test_with_values(self) -> None:
        cs = CleanStats(
            total_input=100,
            removed_no_test_files=10,
            removed_patch_too_large=5,
            total_removed=15,
            total_output=85,
        )
        assert cs.total_output == 85


# ── Splits ─────────────────────────────────────────────────────────────────


class TestSplits:
    def test_defaults(self) -> None:
        s = Splits()
        assert s.train == []
        assert s.val == []
        assert s.test == []
        assert s.golden == []


# ── GoldenSet ──────────────────────────────────────────────────────────────


class TestGoldenSet:
    def test_create(self) -> None:
        gs = GoldenSet(
            f2p_verified_count=5,
            source_split="test",
        )
        assert gs.f2p_verified_count == 5


# ── Result models ──────────────────────────────────────────────────────────


class TestRepoResult:
    def test_defaults(self) -> None:
        rr = RepoResult(repo_id="owner/repo")
        assert rr.raw_count == 0
        assert rr.error is None


class TestPipelineStats:
    def test_defaults(self) -> None:
        ps = PipelineStats()
        assert ps.total_raw == 0
        assert ps.dedup_stats.total_input == 0

    def test_deserialise(self) -> None:
        data = {
            "total_raw": 100,
            "total_validated": 80,
            "total_cleaned": 60,
            "dedup_stats": {
                "total_input": 80,
                "exact_duplicates_removed": 15,
                "content_duplicates_removed": 5,
                "unique_output": 60,
            },
            "clean_stats": {
                "total_input": 60,
                "removed_no_test_files": 5,
                "total_removed": 5,
                "total_output": 55,
            },
            "train_count": 30,
            "val_count": 10,
            "test_count": 15,
            "golden_count": 5,
            "total_examples": 55,
            "repo_count": 2,
        }
        ps = PipelineStats(**data)
        assert ps.train_count == 30
        assert ps.total_examples == 55


class TestPipelineResult:
    def test_total_examples_property(self) -> None:
        splits = Splits(
            train=[IssueRecord(**self._dummy("t", i)) for i in range(10)],
            val=[IssueRecord(**self._dummy("v", i)) for i in range(3)],
            test=[IssueRecord(**self._dummy("e", i)) for i in range(2)],
        )
        pr = PipelineResult(
            run_id="test123",
            manifest_hash="abc",
            splits=splits,
            stats=PipelineStats(),
        )
        assert pr.total_examples == 15

    @staticmethod
    def _dummy(repo: str, i: int) -> dict:
        return {
            "issue_id": f"{repo}#{i}",
            "repo": repo,
            "issue_body": "body",
            "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
        }
