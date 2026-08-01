"""Pydantic models for the evaluation harness I/O.

All evaluation record types live here so every module shares a single
source of truth for the eval schema (mirrors ``data_engineering.schema``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from evaluation.config import EvalConfig

Status = Literal["passed", "failed", "errored", "skipped", "flaky"]
Method = Literal["git_apply", "unidiff_fallback", "failed", "none"]


class TestResult(BaseModel):
    """Outcome of a single test run (before or after patch application)."""

    name: str
    status: Status
    duration: float
    output: str = ""
    retry_count: int = 0  # number of retries taken = len(attempts) - 1


class PatchApplicationResult(BaseModel):
    """Result of applying a generated patch to a repo checkout."""

    success: bool
    method_used: Method
    error: str | None = None
    files_modified: list[str] = []


class EvalInput(BaseModel):
    """A single evaluation example reconstructed from the golden set."""

    instance_id: str
    repo: str
    issue_body: str
    base_sha: str
    head_sha: str
    test_patch: str  # ground truth test changes
    fail_to_pass: list[str]  # test names that should fail → pass
    pass_to_pass: list[str]  # test names that should pass → pass
    repo_domain: str
    metadata: dict[str, Any] = {}

    @classmethod
    def from_swebench_record(cls, record: dict[str, Any]) -> EvalInput:
        """Reconstruct an ``EvalInput`` from a data_engineering schema record.

        Golden.jsonl records are ``IssueRecord`` dumps (see
        ``data_engineering.schema``): top-level ``issue_id``, ``issue_body``,
        ``patch_diff``, ``test_results``, ``repo_domain`` plus a ``metadata``
        dict holding ``base_sha``, ``head_sha``, ``test_patch`` and
        ``instance_id``. Raw SWE-bench records (``instance_id``,
        ``problem_statement``, ``FAIL_TO_PASS``, ...) are also accepted.

        Args:
            record: An ``IssueRecord.model_dump()`` or raw SWE-bench dict.

        Returns:
            The reconstructed EvalInput.
        """
        metadata = dict(record.get("metadata") or {})
        test_results = record.get("test_results") or {}

        instance_id = (
            record.get("instance_id") or metadata.get("instance_id") or record.get("issue_id") or ""
        )
        issue_body = record.get("issue_body") or record.get("problem_statement") or ""
        base_sha = metadata.get("base_sha") or record.get("base_commit") or ""
        head_sha = metadata.get("head_sha") or record.get("environment_setup_commit") or ""
        test_patch = metadata.get("test_patch") or record.get("test_patch") or ""
        fail_to_pass = (
            test_results.get("failed")
            or record.get("FAIL_TO_PASS")
            or record.get("fail_to_pass")
            or []
        )
        pass_to_pass = (
            test_results.get("passed")
            or record.get("PASS_TO_PASS")
            or record.get("pass_to_pass")
            or []
        )

        return cls(
            instance_id=str(instance_id),
            repo=str(record.get("repo") or ""),
            issue_body=str(issue_body),
            base_sha=str(base_sha),
            head_sha=str(head_sha),
            test_patch=str(test_patch),
            fail_to_pass=_to_test_list(fail_to_pass),
            pass_to_pass=_to_test_list(pass_to_pass),
            repo_domain=str(record.get("repo_domain") or "unknown"),
            metadata=metadata,
        )


class EvalResult(BaseModel):
    """Full per-example evaluation outcome."""

    instance_id: str
    repo: str
    model_name: str
    variant: str
    prompt_template: str
    generated_patch: str
    patch_application: PatchApplicationResult
    tests_before: list[TestResult]  # at base_sha
    tests_after: list[TestResult]  # at head_sha + generated patch
    f2p: float  # 0.0-1.0
    p2p: float  # 0.0-1.0
    latency_seconds: float
    timestamp: datetime
    error: str | None = None


class F2PMetrics(BaseModel):
    """Aggregate metrics for one model/variant/prompt group."""

    model_name: str
    variant: str
    prompt_template: str
    total_examples: int
    successful_patches: int
    f2p_rate: float
    f2p_count: int
    p2p_rate: float
    p2p_count: int
    avg_latency: float
    flaky_test_rate: float
    per_repo_breakdown: dict[str, dict]


class EvalRun(BaseModel):
    """A complete evaluation run, persisted for W&B artifact logging."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    config: EvalConfig
    models_evaluated: list[str]
    results: list[EvalResult]
    aggregate: list[F2PMetrics]
    status: Literal["running", "completed", "failed", "partial"]
    cost_usd: float = 0.0  # estimated run cost (inference + test execution)


def _to_test_list(value: Any) -> list[str]:
    """Normalize a test list from either a comma-joined string or a list.

    Handles JSON-encoded lists stored as strings (e.g. ``'["test_a","test_b"]'``)
    which appear in golden.jsonl when ``test_results`` was serialized via
    ``json.dumps``. Also strips stray quotes/brackets from split fragments.
    """
    import json as _json

    if isinstance(value, str):
        # Try JSON-decode first (handles ``'["test_a","test_b"]'``)
        try:
            decoded = _json.loads(value)
            if isinstance(decoded, list):
                return [str(t).strip() for t in decoded if str(t).strip()]
        except ValueError:
            pass
        # Fall back to comma-split
        return [t.strip() for t in value.split(",") if t.strip()]

    # golden.jsonl splits JSON arrays across list elements; join, JSON-decode, then strip
    joined = "".join(str(item) for item in value)
    try:
        decoded = _json.loads(joined)
        if isinstance(decoded, list):
            return [str(t).strip() for t in decoded if str(t).strip()]
    except ValueError:
        pass
    # Fall back: strip stray brackets/quotes from each item
    out = []
    for item in value:
        s = str(item).strip().strip('",[]').strip()
        if s:
            out.append(s)
    return out
