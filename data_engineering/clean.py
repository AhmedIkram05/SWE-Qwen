"""Deduplication and cleaning/filtering module.

Two-stage pipeline:
1. Deduplicate (exact match on (repo, issue_id, pr_number); content match on
   patch SHA256).
2. Apply quality filters (test file changes, patch size, binary, non-Python,
   empty body, F2P signal).
"""

from __future__ import annotations

import hashlib
import logging
import re

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import (
    CleanStats,
    DedupStats,
    IssueRecord,
)

logger = logging.getLogger(__name__)

# ── F2P keyword pattern ───────────────────────────────────────────────────

F2P_KEYWORD_PATTERN = re.compile(
    r"\b(fix(?:es|ed|ing)?|close[ds]?|resolve[ds]?)\b",
    re.IGNORECASE,
)

# Minimum ratio of Python files in a PR's changed files to keep the record.
PYTHON_RATIO_THRESHOLD = 0.5


def _has_f2p_keywords(record: IssueRecord) -> bool:
    """Check if commit messages or PR description contain F2P keywords.

    For SWE-bench records, also check metadata.has_test_patch (indicates
    FAIL_TO_PASS tests exist, which is the ground-truth F2P signal).
    """
    text = " ".join(record.commit_messages) + " " + record.pr_description
    if F2P_KEYWORD_PATTERN.search(text):
        return True
    # SWE-bench: metadata.has_test_patch indicates FAIL_TO_PASS tests present
    return bool(record.metadata.get("has_test_patch"))


def _is_binary_diff(diff: str) -> bool:
    """Check if the unified diff contains binary file markers."""
    return "Binary files" in diff or "Binary file" in diff


def _count_py_files(files: list[str]) -> int:
    """Count how many files in the list end with ``.py``."""
    return sum(1 for f in files if f.endswith(".py"))


def _python_ratio(files_changed: list[str]) -> tuple[float, int]:
    """Return (py_ratio, total). Returns (1.0, 0) for empty file list."""
    total = len(files_changed)
    if total == 0:
        return 1.0, 0
    return _count_py_files(files_changed) / total, total


# ── Exported functions ────────────────────────────────────────────────────


def deduplicate(records: list[IssueRecord]) -> tuple[list[IssueRecord], DedupStats]:
    """Remove duplicate records.

    Primary dedup: ``(repo, issue_id, pr_number)`` — same fix for same issue+PR.
    Secondary dedup: SHA256 of ``patch_diff`` — same patch elsewhere.
    """
    seen_ids: set[tuple[str, str, str]] = set()  # (repo, issue_id, pr_number)
    seen_patches: set[str] = set()
    unique: list[IssueRecord] = []
    exact_removed = 0
    content_removed = 0

    for rec in records:
        exact_key = (rec.repo, rec.issue_id, rec.pr_number)

        if exact_key in seen_ids:
            exact_removed += 1
            continue

        patch_hash = hashlib.sha256(rec.patch_diff.encode("utf-8")).hexdigest()
        if patch_hash in seen_patches:
            # Same patch but different issue — rare but possible
            content_removed += 1
            continue

        seen_ids.add(exact_key)
        seen_patches.add(patch_hash)
        unique.append(rec)

    stats = DedupStats(
        total_input=len(records),
        exact_duplicates_removed=exact_removed,
        content_duplicates_removed=content_removed,
        unique_output=len(unique),
    )
    return unique, stats


# ── Clean filter helpers ──────────────────────────────────────────────────


_FILTER_ATTRS = {
    "no_test_files": "removed_no_test_files",
    "patch_too_large": "removed_patch_too_large",
    "binary": "removed_binary",
    "non_python": "removed_non_python",
    "empty_body": "removed_empty_body",
    "no_f2p_signal": "removed_no_f2p_signal",
}


def _check_no_test_files(record: IssueRecord) -> str | None:
    return "no_test_files" if not record.test_files_changed else None


def _check_patch_size(record: IssueRecord, max_lines: int) -> str | None:
    if len(record.patch_diff.splitlines()) > max_lines:
        return "patch_too_large"
    return None


def _check_binary(record: IssueRecord) -> str | None:
    return "binary" if _is_binary_diff(record.patch_diff) else None


def _check_non_python(record: IssueRecord) -> tuple[str | None, str | None]:
    """Return (reason, warning). reason=remove if non-py ratio below threshold."""
    ratio, total = _python_ratio(record.files_changed)
    if total > 0 and ratio < PYTHON_RATIO_THRESHOLD:
        return "non_python", None
    if total > 0 and _count_py_files(record.files_changed) < total:
        return None, f"non_python_files: {record.files_changed}"
    return None, None


def _check_empty_body(record: IssueRecord) -> str | None:
    if not record.issue_body or not record.issue_body.strip():
        return "empty_body"
    return None


def _check_f2p(record: IssueRecord) -> str | None:
    return "no_f2p_signal" if not _has_f2p_keywords(record) else None


def _apply_filter(stats: CleanStats, reason: str) -> None:
    attr = _FILTER_ATTRS.get(reason)
    if attr:
        setattr(stats, attr, getattr(stats, attr, 0) + 1)


def clean_records(
    records: list[IssueRecord],
    config: DataPipelineConfig,
) -> tuple[list[IssueRecord], CleanStats]:
    """Apply quality filters to a list of records.

    Each filter can independently remove a record. **WARN** filters log
    warnings but keep the record.

    For records with ``metadata.is_verified=False`` (e.g. SWE-bench Train
    split — no FAIL_TO_PASS), the ``no_test_files`` and ``no_f2p_signal``
    filters are demoted to warnings instead of removal reasons. This preserves
    training data while keeping verified eval data high-quality.
    """
    stats = CleanStats(total_input=len(records))
    cleaned: list[IssueRecord] = []

    for rec in records:
        reasons: list[str] = []
        warnings: list[str] = []
        is_verified = rec.metadata.get("is_verified", True)

        checks: list = [
            _check_patch_size(rec, config.max_patch_lines),
            _check_binary(rec),
        ]
        non_python_reason, non_python_warn = _check_non_python(rec)
        checks.extend(
            [
                non_python_reason,
                _check_empty_body(rec),
            ]
        )

        # Hard gates for verified records; soft warnings for training-only records
        if is_verified:
            checks.append(_check_no_test_files(rec))
            checks.append(_check_f2p(rec))
        else:
            nt = _check_no_test_files(rec)
            if nt:
                warnings.append(nt)
            f2p = _check_f2p(rec)
            if f2p:
                warnings.append(f2p)

        for reason in checks:
            if reason:
                reasons.append(reason)

        if non_python_warn:
            warnings.append(non_python_warn)

        for reason in reasons:
            _apply_filter(stats, reason)

        if warnings:
            stats.warnings_non_python.append(f"{rec.repo}#{rec.issue_id}: {warnings}")

        if reasons:
            continue  # record removed

        cleaned.append(rec)

    stats.total_removed = stats.total_input - len(cleaned)
    stats.total_output = len(cleaned)

    if stats.total_output == 0:
        logger.warning(
            "Repo yielded 0 records after cleaning (input=%s). Filter breakdown: %s",
            stats.total_input,
            stats.model_dump(),
        )

    return cleaned, stats
