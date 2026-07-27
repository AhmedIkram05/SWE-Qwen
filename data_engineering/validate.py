"""Schema validation module.

Validates raw record dicts against ``IssueRecord`` Pydantic schema.
Collects ALL errors per record (not just the first), and continues on
failure — no fail-fast behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from data_engineering.schema import (
    IssueRecord,
    ParsedHunk,
    TestResults,
    ValidationError,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def _validate_hunks(hunks_data: list[dict[str, Any]]) -> list[ParsedHunk]:
    """Validate a list of parsed hunk dicts."""
    hunks: list[ParsedHunk] = []
    for h in hunks_data:
        try:
            hunks.append(ParsedHunk(**h))
        except Exception:
            continue  # skip malformed hunks
    return hunks


def validate_record(record: dict[str, Any]) -> ValidationResult:
    """Validate a single raw record dict against the IssueRecord schema.

    Returns a ``ValidationResult`` with *valid=True* and a populated
    *record* field, or *valid=False* with a list of ``ValidationError``.
    """
    errors: list[ValidationError] = []

    # ── Validate hunks separately (optional enrichment) ──────────────────
    hunks_data = record.get("parsed_hunks", [])
    if isinstance(hunks_data, list) and hunks_data and isinstance(hunks_data[0], dict):
        parsed = _validate_hunks(hunks_data)
    else:
        parsed = []

    # ── Validate test_results separately ─────────────────────────────────
    tr_data = record.get("test_results", {})
    try:
        test_results = TestResults(**tr_data) if isinstance(tr_data, dict) else TestResults()
    except Exception as exc:
        errors.append(
            ValidationError(
                record_id=record.get("issue_id", "unknown"),
                field="test_results",
                error=str(exc),
                raw_value=tr_data,
            )
        )
        test_results = TestResults()

    # ── Build a clean dict for IssueRecord ───────────────────────────────
    clean_data = {
        "issue_id": record.get("issue_id", ""),
        "repo": record.get("repo", ""),
        "pr_number": record.get("pr_number", ""),
        "issue_body": record.get("issue_body", ""),
        "patch_diff": record.get("patch_diff", ""),
        "parsed_hunks": parsed,
        "test_results": test_results,
        "pr_title": record.get("pr_title", ""),
        "pr_description": record.get("pr_description", ""),
        "commit_messages": record.get("commit_messages", []),
        "files_changed": record.get("files_changed", []),
        "test_files_changed": record.get("test_files_changed", []),
        "issue_labels": record.get("issue_labels", []),
        "repo_domain": record.get("repo_domain", ""),
        "metadata": record.get("metadata", {}),
    }

    # ── Validate with Pydantic ──────────────────────────────────────────
    try:
        validated = IssueRecord(**clean_data)
    except PydanticValidationError as exc:
        for e in exc.errors():
            errors.append(
                ValidationError(
                    record_id=clean_data.get("issue_id", "unknown"),
                    field=".".join(str(p) for p in e["loc"]),
                    error=e["msg"],
                    raw_value=e.get("input"),
                )
            )
        return ValidationResult(valid=False, errors=errors)

    # ── Run additional field-level non-Pydantic checks ───────────────────
    if not validated.files_changed:
        errors.append(
            ValidationError(
                record_id=validated.issue_id,
                field="files_changed",
                error="files_changed is empty",
            )
        )

    if errors:
        return ValidationResult(valid=False, errors=errors)

    return ValidationResult(valid=True, record=validated)


def validate_batch(
    records: list[dict[str, Any]],
) -> tuple[list[IssueRecord], list[ValidationError]]:
    """Validate a batch of records.

    Returns a tuple of ``(valid_records, all_validation_errors)``.
    Invalid records are silently skipped (errors are collected).
    """
    valid: list[IssueRecord] = []
    all_errors: list[ValidationError] = []

    for record in records:
        result = validate_record(record)
        if result.valid and result.record is not None:
            valid.append(result.record)
        else:
            all_errors.extend(result.errors)

    if all_errors:
        logger.warning(
            "Validation: %d valid, %d errors across %d records",
            len(valid),
            len(all_errors),
            len(records),
        )

    return valid, all_errors
