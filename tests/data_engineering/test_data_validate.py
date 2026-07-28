"""Tests for data_engineering.validate."""

from __future__ import annotations

import json
from pathlib import Path

from data_engineering.schema import IssueRecord, ValidationError
from data_engineering.validate import validate_batch

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict]:
    with (FIXTURES / name).open() as f:
        return json.load(f)


class TestValidateBatch:
    def test_all_valid(self) -> None:
        data = _load("sample_issues.json")
        valid, errors = validate_batch(data)
        assert len(valid) == len(data)
        assert all(isinstance(r, IssueRecord) for r in valid)
        assert len(errors) == 0

    def test_all_invalid_produces_errors(self) -> None:
        data = _load("sample_invalid_issues.json")
        valid, errors = validate_batch(data)
        assert len(valid) == 0
        assert len(errors) > 0
        assert all(isinstance(e, ValidationError) for e in errors)

    def test_mixed_valid_and_invalid(self) -> None:
        valid_data = _load("sample_issues.json")
        invalid_data = _load("sample_invalid_issues.json")
        data = valid_data[:2] + invalid_data[:2]

        records, errors = validate_batch(data)
        assert len(records) == 2
        assert len(errors) > 0

    def test_empty_input(self) -> None:
        records, errors = validate_batch([])
        assert records == []
        assert errors == []

    def test_error_contains_field_and_reason(self) -> None:
        data = [
            {
                "issue_id": "test#1",
                "repo": "r",
                "issue_body": "  ",
                "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
            }
        ]
        valid, errors = validate_batch(data)
        assert len(valid) == 0
        assert any("body" in e.field or "issue_body" in e.field for e in errors)

    def test_malformed_hunks_skipped(self) -> None:
        """Malformed hunk dicts should be silently skipped, valid ones kept."""
        data = [
            {
                "issue_id": "test#1",
                "repo": "r",
                "issue_body": "body",
                "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
                "files_changed": ["f.py"],
                "parsed_hunks": [
                    {
                        "file": "good.py",
                        "old_start": 1,
                        "old_lines": 5,
                        "new_start": 1,
                        "new_lines": 6,
                        "diff_lines": ["+x"],
                    },
                    {"file": "bad.py"},  # missing required fields — should be skipped
                ],
            }
        ]
        valid, errors = validate_batch(data)
        assert len(valid) == 1
        assert len(valid[0].parsed_hunks) == 1  # only the valid hunk kept
        assert valid[0].parsed_hunks[0].file == "good.py"

    def test_invalid_test_results_handled(self) -> None:
        """When test_results is not a dict, the code uses TestResults() default
        and the record validates without a test_results error."""
        data = [
            {
                "issue_id": "test#1",
                "repo": "r",
                "issue_body": "body",
                "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
                "files_changed": ["f.py"],
                "test_results": "not-a-dict",  # string instead of dict
            }
        ]
        valid, errors = validate_batch(data)
        # Silently falls back to TestResults() default, no error for test_results
        assert len(valid) == 1
        assert not any("test_results" in e.field for e in errors)

    def test_empty_files_changed_warns(self) -> None:
        """Empty files_changed should produce a validation error
        and mark the record as invalid."""
        data = [
            {
                "issue_id": "test#1",
                "repo": "r",
                "issue_body": "body",
                "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
                "files_changed": [],
            }
        ]
        valid, errors = validate_batch(data)
        # files_changed empty should produce validation error
        assert len(valid) == 0
        assert any("files_changed" in e.field for e in errors)

    def test_invalid_test_results_type_raises_error(self) -> None:
        """TestResults with invalid field types should produce validation error
        but record still validates with default TestResults()."""
        data = [
            {
                "issue_id": "test#1",
                "repo": "r",
                "issue_body": "body",
                "patch_diff": "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
                "files_changed": ["f.py"],
                "test_results": {"passed": "not-a-list"},  # list expected
            }
        ]
        valid, errors = validate_batch(data)
        # Record invalid because test_results error is collected
        assert len(valid) == 0
        assert any("test_results" in e.field for e in errors)
