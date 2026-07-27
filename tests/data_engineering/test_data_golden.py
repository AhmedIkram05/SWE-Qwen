"""Tests for data_engineering.golden."""

from __future__ import annotations

from data_engineering.golden import build_golden_set
from data_engineering.schema import IssueRecord, Splits


def _rec(
    issue_id: str, test_files: list[str] | None = None, cm: list[str] | None = None
) -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo="r",
        issue_body="body",
        patch_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
        test_files_changed=test_files or [],
        commit_messages=cm or ["fix: bug"],
    )


class TestBuildGoldenSet:
    def test_builds_from_test_split(self) -> None:
        splits = Splits(
            test=[_rec("r#1", test_files=["t.py"])],
        )
        gs = build_golden_set(splits, min_size=0, source_split="test")
        assert len(gs.records) == 1
        assert gs.f2p_verified_count == 1
        assert gs.source_split == "test"

    def test_empty_when_no_qualifying(self) -> None:
        splits = Splits(test=[_rec("r#1")])  # no test_files
        gs = build_golden_set(splits, min_size=0)
        assert len(gs.records) == 0
        assert gs.f2p_verified_count == 0
