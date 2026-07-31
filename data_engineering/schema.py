"""Pydantic models for the data pipeline.

All record types, stats containers, and result types live here so every
module shares a single source of truth for the data schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

# ── Core data models ──────────────────────────────────────────────────────


class ParsedHunk(BaseModel):
    """A single hunk from a unified diff."""

    file: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    diff_lines: list[str]


class TestResults(BaseModel):
    """Test results in the *final* state (post-fix).

    Note: ``failed > 0`` means tests are **still failing** after the fix —
    this is NOT a valid "was failing before" signal. Phase 5 will compare
    base-SHA vs head-SHA for ground-truth before/after.
    """

    passed: list[str] = []
    failed: list[str] = []
    errored: list[str] = []


class IssueRecord(BaseModel):
    """A single issue-PR pair ready for training or evaluation."""

    issue_id: str
    repo: str
    pr_number: str = ""
    issue_body: str
    patch_diff: str
    parsed_hunks: list[ParsedHunk] = []
    test_results: TestResults = TestResults()
    pr_title: str = ""
    pr_description: str = ""
    commit_messages: list[str] = []
    files_changed: list[str] = []
    test_files_changed: list[str] = []
    issue_labels: list[str] = []
    repo_domain: str = ""
    metadata: dict[str, Any] = {}

    @field_validator("patch_diff")
    @classmethod
    def validate_patch(cls, v: str) -> str:
        """Validate that *v* looks like a unified diff.

        Tries ``unidiff.PatchSet`` first (per plan spec); falls back to
        regex check (``---`` / ``+++`` / ``@@``) because API diffs
        can be truncated or have formatting quirks that unidiff rejects.
        """
        if not v or not v.strip():
            raise ValueError("patch_diff is empty")
        try:
            import unidiff

            ps = unidiff.PatchSet(v)
            if len(ps) > 0:
                return v
        except Exception:
            pass

        line_list = v.strip().splitlines()
        has_old = any(line.startswith("--- ") for line in line_list)
        has_new = any(line.startswith("+++ ") for line in line_list)
        has_hunk = any(line.startswith("@@") and " @@" in line for line in line_list)
        if not (has_old and has_new and has_hunk):
            raise ValueError(
                "patch_diff does not look like a unified diff (missing ---/+++/@@ headers)"
            )
        return v

    @field_validator("issue_body")
    @classmethod
    def validate_issue_body(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("issue_body is empty or whitespace-only")
        return v.strip()


# ── Validation models ─────────────────────────────────────────────────────


class ValidationError(BaseModel):
    """A single validation error for a specific record and field."""

    record_id: str
    field: str
    error: str
    raw_value: Any = None


class ValidationResult(BaseModel):
    """Result of validating a single record."""

    valid: bool
    record: IssueRecord | None = None
    errors: list[ValidationError] = []


# ── Stats models ──────────────────────────────────────────────────────────


class DedupStats(BaseModel):
    total_input: int = 0
    exact_duplicates_removed: int = 0
    content_duplicates_removed: int = 0
    unique_output: int = 0


class CleanStats(BaseModel):
    total_input: int = 0
    removed_no_test_files: int = 0
    removed_patch_too_large: int = 0
    removed_binary: int = 0
    removed_non_python: int = 0
    removed_empty_body: int = 0
    removed_no_f2p_signal: int = 0
    total_removed: int = 0
    total_output: int = 0
    warnings_non_python: list[str] = []


class SplitRatios(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class Splits(BaseModel):
    train: list[IssueRecord] = []
    val: list[IssueRecord] = []
    test: list[IssueRecord] = []
    golden: list[IssueRecord] = []


# ── Result models ─────────────────────────────────────────────────────────


class GoldenSet(BaseModel):
    records: list[IssueRecord] = []
    f2p_verified_count: int = 0
    source_split: str = "test"


class RepoResult(BaseModel):
    repo_id: str
    raw_count: int = 0
    validated_count: int = 0
    cleaned_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    golden_count: int = 0
    error: str | None = None


class PipelineStats(BaseModel):
    total_raw: int = 0
    total_validated: int = 0
    total_validation_errors: int = 0
    total_cleaned: int = 0
    dedup_stats: DedupStats = DedupStats()
    clean_stats: CleanStats = CleanStats()
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    golden_count: int = 0
    total_examples: int = 0
    repo_count: int = 0
    repo_results: list[RepoResult] = []
    gcs_paths: dict[str, str] = {}
    wandb_artifacts: dict[str, str] = {}


class PipelineResult(BaseModel):
    run_id: str
    manifest_hash: str
    splits: Splits
    stats: PipelineStats
    gcs_paths: dict[str, str] = {}
    wandb_artifacts: dict[str, str] = {}
    tokenized_paths: dict[str, str] = {}

    @property
    def total_examples(self) -> int:
        return len(self.splits.train) + len(self.splits.val) + len(self.splits.test)
