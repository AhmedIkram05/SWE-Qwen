"""P0-5 metric fairness: apply-rate vs conditional-resolve split, strict
instance-resolve (F2P == 1.0), and Wilson CI on the binary statistic.

Legacy ``f2p_rate`` (mean partial credit) conflated "patch failed to apply"
with "patch applied but wrong".  The split metrics isolate each failure
mode, and the reported CI must bound the binary resolve count — a binomial
CI on a mean is invalid.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from evaluation.comparison import compare_and_report
from evaluation.metrics import (
    aggregate_metrics,
    apply_rate,
    conditional_resolve_rate,
    is_instance_resolved,
)
from evaluation.schema import EvalResult, F2PMetrics, PatchApplicationResult
from evaluation.stats import wilson_ci


def _result(
    instance_id: str,
    f2p: float,
    applied: bool = True,
    error: str | None = None,
    p2p: float = 1.0,
) -> EvalResult:
    return EvalResult(
        instance_id=instance_id,
        repo="django/django",
        model_name="qwen3-14b",
        variant="baseline_14b",
        prompt_template="chat",
        generated_patch="",
        patch_application=PatchApplicationResult(
            success=applied, method_used="git_apply" if applied else "failed"
        ),
        tests_before=[],
        tests_after=[],
        f2p=f2p,
        p2p=p2p,
        latency_seconds=1.0,
        timestamp=datetime.now(UTC),
        error=error,
    )


class TestInstanceResolved:
    def test_full_pass_no_error_is_resolved(self):
        assert is_instance_resolved(_result("i1", 1.0)) is True

    def test_partial_credit_not_resolved(self):
        # Half the F2P tests fixed is still a failure for binary statistics.
        assert is_instance_resolved(_result("i1", 0.5)) is False

    def test_errored_result_not_resolved(self):
        assert is_instance_resolved(_result("i1", 1.0, error="boom")) is False


class TestRateHelpers:
    def test_apply_rate(self):
        assert apply_rate(3, 4) == 0.75
        assert apply_rate(0, 0) == 0.0

    def test_conditional_resolve_rate(self):
        assert conditional_resolve_rate(1, 2) == 0.5
        assert conditional_resolve_rate(0, 0) == 0.0


class TestAggregateSplit:
    def test_apply_vs_conditional_split(self):
        results = [
            _result("i1", 1.0, applied=True),  # resolved + applied
            _result("i2", 0.5, applied=True),  # applied, incomplete patch
            _result("i3", 0.0, applied=False),  # apply-fail
            _result("i4", 0.0, applied=False),  # apply-fail
        ]
        m = aggregate_metrics(results)
        assert m.total_examples == 4
        assert m.successful_patches == 2
        assert m.apply_rate == 0.5
        assert m.resolve_count == 1
        assert m.resolve_rate == 0.25
        assert m.conditional_resolve_rate == 0.5
        # Legacy mean conflates both failure modes: (1.0+0.5+0+0)/4.
        assert m.f2p_rate == pytest.approx(0.375)

    def test_errored_result_not_resolved_but_applied(self):
        results = [
            _result("i1", 1.0, applied=True, error="boom"),
            _result("i2", 1.0, applied=True),
        ]
        m = aggregate_metrics(results)
        assert m.resolve_count == 1
        assert m.resolve_rate == 0.5
        assert m.apply_rate == 1.0
        assert m.conditional_resolve_rate == 0.5

    def test_zero_applied_conditional_is_zero(self):
        m = aggregate_metrics([_result("i1", 0.0, applied=False)])
        assert m.conditional_resolve_rate == 0.0
        assert m.resolve_count == 0


def _metrics(**overrides: object) -> F2PMetrics:
    base: dict = {
        "model_name": "qwen3-14b",
        "variant": "baseline_14b",
        "prompt_template": "chat",
        "total_examples": 100,
        "successful_patches": 60,
        "f2p_rate": 0.30,
        "f2p_count": 40,
        "p2p_rate": 0.95,
        "p2p_count": 95,
        "avg_latency": 2.5,
        "flaky_test_rate": 0.0,
        "per_repo_breakdown": {},
        "apply_rate": 0.6,
        "resolve_count": 12,
        "resolve_rate": 0.12,
        "conditional_resolve_rate": 0.20,
    }
    base.update(overrides)
    return F2PMetrics(**base)  # type: ignore[arg-type]


class TestWilsonOnInstanceResolve:
    def test_ci_bounds_resolve_count_not_mean_f2p(self):
        # f2p_rate 0.30 (mean partial credit) vs resolve_count 12: the
        # legacy misuse wilson_ci(round(0.30*100), 100) = wilson(30, 100)
        # must NOT appear; the table bounds wilson(12, 100).
        md = compare_and_report({"qwen3-14b:baseline_14b": _metrics()})
        lo, hi = wilson_ci(12, 100)
        assert f"{lo:.1%}-{hi:.1%}" in md
        lo_bad, hi_bad = wilson_ci(30, 100)
        assert f"{lo_bad:.1%}-{hi_bad:.1%}" not in md

    def test_table_reports_split_columns(self):
        md = compare_and_report({"qwen3-14b:baseline_14b": _metrics()})
        header = md.splitlines()[0]
        assert "apply_rate" in header
        assert "resolve_rate" in header
        assert "cond_resolve" in header
        row = md.splitlines()[-1]
        assert "60.00%" in row  # apply_rate
        assert "12.00%" in row  # resolve_rate
        assert "20.00%" in row  # conditional_resolve_rate

    def test_zero_resolve_count_ci_starts_at_zero(self):
        md = compare_and_report({"qwen3-14b:baseline_14b": _metrics(resolve_count=0)})
        lo, hi = wilson_ci(0, 100)
        assert f"{lo:.1%}-{hi:.1%}" in md
        assert lo == 0.0
