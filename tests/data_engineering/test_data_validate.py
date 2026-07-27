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
