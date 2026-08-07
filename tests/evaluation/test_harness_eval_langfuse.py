"""Per-example Langfuse tracing call-site tests for ``evaluation.harness``.

Phase 8 (§5.5 dual-write): ``run_batch`` must emit one ``trace_generation``
per *completed* example, cross-linked to W&B by ``run_id``/``instance_id``.
``trace_generation`` is a silent no-op without Langfuse keys, so every test
patches it and asserts on the call itself — fully offline (no Modal, no
network, no keys).

Decisions under test:
- Every completed example is traced, even when the generated patch is empty
  or the test result errored — eval traces are never sampled.
- Resumed (checkpoint-loaded) results are NOT re-traced: they were traced in
  the run that completed them.
- The trace is containment-guarded: a failing trace never breaks the batch.
"""

from __future__ import annotations

from typing import Any

import pytest

import evaluation.harness as harness_mod
from evaluation.config import EvalConfig
from evaluation.harness import EvaluationHarness
from evaluation.schema import EvalInput

PASSING_PAYLOAD = {
    "tests_before": [{"name": "tests.test_fix", "status": "failed", "duration": 0.1}],
    "tests_after": [{"name": "tests.test_fix", "status": "passed", "duration": 0.1}],
    "patch_application": {"success": True, "method_used": "git_apply"},
}


def _cfg(tmp_path: Any) -> EvalConfig:
    return EvalConfig(
        checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out",
        golden_data_path=str(tmp_path / "golden.jsonl"),
    )


def _input(instance_id: str = "inst-1", repo: str = "owner/repo") -> EvalInput:
    return EvalInput(
        instance_id=instance_id,
        repo=repo,
        issue_body=f"fix bug in {instance_id}",
        base_sha="abc123",
        head_sha="def456",
        test_patch="",
        fail_to_pass=["tests.test_fix"],
        pass_to_pass=[],
        repo_domain="python",
    )


def _mock_eval_path(monkeypatch: pytest.MonkeyPatch, patch: str = "+p\n") -> None:
    """Replace the Modal/network eval path with local stubs."""
    monkeypatch.setattr(
        harness_mod,
        "_generate_patches",
        lambda m, v, t, ex, **kw: [patch] * len(ex),  # noqa: E501
    )
    monkeypatch.setattr(
        harness_mod,
        "_run_tests_swebench",
        lambda instances, patches, cfg: {ex.instance_id: PASSING_PAYLOAD for ex in instances},
    )


def test_run_batch_traces_each_completed_example(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "observability.langfuse.trace_generation",
        lambda **kw: calls.append(kw),
    )
    harness = EvaluationHarness(_cfg(tmp_path))
    _mock_eval_path(monkeypatch)

    results = harness.run_batch(
        [_input("a"), _input("b")], "qwen3-14b", "baseline_14b", "chat", "run-1"
    )

    assert len(results) == 2
    assert len(calls) == 2
    assert {c["metadata"]["instance_id"] for c in calls} == {"a", "b"}
    for call in calls:
        assert call["name"] == "eval/qwen3-14b/baseline_14b/chat"
        assert call["name"].startswith("eval/")
        assert call["model"] == "qwen3-14b"
        assert call["metadata"]["run_id"] == "run-1"
        assert call["metadata"]["prompt_template"] == "chat"
        assert call["metadata"]["variant"] == "baseline_14b"
        assert call["prompt"] == f"fix bug in {call['metadata']['instance_id']}"
        assert call["completion"] == "+p\n"
        # f2p/p2p are rates (0.0-1.0 floats), not counts — keep the raw float.
        assert call["scores"] == {"f2p": 1.0, "p2p": 1.0}


def test_run_batch_traces_empty_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A missing/empty generation is still a completed example — trace it."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "observability.langfuse.trace_generation",
        lambda **kw: calls.append(kw),
    )
    harness = EvaluationHarness(_cfg(tmp_path))
    _mock_eval_path(monkeypatch, patch="")

    results = harness.run_batch([_input("a")], "qwen3-14b", "baseline_14b", "chat", "run-1")

    assert len(results) == 1
    assert len(calls) == 1
    assert calls[0]["completion"] == ""


def test_run_batch_resume_does_not_retrace_loaded_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Checkpoint-resumed results were traced in their original run — no re-trace."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "observability.langfuse.trace_generation",
        lambda **kw: calls.append(kw),
    )
    harness = EvaluationHarness(_cfg(tmp_path))
    _mock_eval_path(monkeypatch)

    first = harness.run_batch([_input("a")], "qwen3-14b", "baseline_14b", "chat", "run-1")
    assert len(first) == 1
    assert len(calls) == 1

    # Same batch again: repo checkpoint is complete → results loaded, not re-run.
    second = harness.run_batch([_input("a")], "qwen3-14b", "baseline_14b", "chat", "run-1")
    assert len(second) == 1
    assert len(calls) == 1  # only the first run's trace — no re-trace of resumed results


def test_run_batch_trace_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A failing trace must never break the batch (observability is non-goal)."""

    def _boom(**kw: Any) -> None:
        raise RuntimeError("langfuse exploded")

    monkeypatch.setattr("observability.langfuse.trace_generation", _boom)
    harness = EvaluationHarness(_cfg(tmp_path))
    _mock_eval_path(monkeypatch)

    results = harness.run_batch(
        [_input("a"), _input("b")], "qwen3-14b", "baseline_14b", "chat", "run-1"
    )

    assert len(results) == 2
    assert all(r.error is None for r in results)
