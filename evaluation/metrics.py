"""F2P/P2P metric computation for evaluation results.

F2P (fail-to-pass): ground-truth failing tests that now pass after the
generated patch. P2P (pass-to-pass): previously passing tests that still
pass (regression ceiling).
"""

from __future__ import annotations

import logging

from evaluation.schema import EvalResult, F2PMetrics, TestResult

logger = logging.getLogger(__name__)

_FAILED_STATUSES = ("failed", "errored")
_VALID_STATUSES = ("passed", *(_FAILED_STATUSES))  # flaky/skipped excluded from denominators


def compute_f2p(
    tests_before: list[TestResult],
    tests_after: list[TestResult],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> tuple[float, float, int, int]:
    """Compute F2P and P2P rates between two test runs.

    ``F2P = |{t ∈ fail_to_pass : t failed before ∧ t passed after}| /
    |fail_to_pass|`` and ``P2P = |{t ∈ pass_to_pass : t passed before ∧
    t passed after}| / |pass_to_pass|``.

    Statuses ``failed``/``errored`` count as failed-before; ``passed``
    counts as passed. Tests whose before or after status is flaky or
    skipped (or that were not run) are excluded from the denominator.

    Args:
        tests_before: Test results at base_sha.
        tests_after: Test results after applying the patch.
        fail_to_pass: Test names expected to flip to passing.
        pass_to_pass: Test names expected to stay passing.

    Returns:
        ``(f2p_rate, p2p_rate, f2p_count, p2p_count)``. An empty
        ``fail_to_pass`` yields F2P 0.0; an empty ``pass_to_pass``
        yields P2P 1.0 (nothing regressed by definition).
    """
    before_map = {t.name: t.status for t in tests_before}
    after_map = {t.name: t.status for t in tests_after}

    f2p_count = sum(
        1
        for t in fail_to_pass
        if before_map.get(t) in _FAILED_STATUSES and after_map.get(t) == "passed"
    )
    f2p_total = sum(
        1
        for t in fail_to_pass
        if before_map.get(t) in _VALID_STATUSES and after_map.get(t) in _VALID_STATUSES
    )
    p2p_count = sum(
        1 for t in pass_to_pass if before_map.get(t) == "passed" and after_map.get(t) == "passed"
    )
    p2p_total = sum(
        1
        for t in pass_to_pass
        if before_map.get(t) in _VALID_STATUSES and after_map.get(t) in _VALID_STATUSES
    )

    f2p_rate = f2p_count / f2p_total if f2p_total else 0.0
    p2p_rate = p2p_count / p2p_total if p2p_total else 1.0
    return f2p_rate, p2p_rate, f2p_count, p2p_count


def is_instance_resolved(result: EvalResult) -> bool:
    """Instance-resolve predicate for fair comparison (P0-5).

    Strict ``F2P==1.0`` on error-free results. Partial credit (e.g. 0.5)
    counts for the legacy mean ``f2p_rate`` but NOT for resolve: a patch
    that fixes half the F2P tests is still a failure for Wilson/McNemar,
    which require binary outcomes.
    """
    return result.error is None and result.f2p == 1.0


def apply_rate(successful_patches: int, total_examples: int) -> float:
    """Patch apply-rate: applied / total."""
    return successful_patches / total_examples if total_examples else 0.0


def conditional_resolve_rate(resolve_count: int, applied_count: int) -> float:
    """Resolved | applied: full fixes among successfully applied patches."""
    return resolve_count / applied_count if applied_count else 0.0


def aggregate_metrics(results: list[EvalResult]) -> F2PMetrics:
    """Aggregate per-example results into a single ``F2PMetrics``.

    Callers pass a homogeneous group (one model/variant/prompt template);
    the model fields are taken from the first result. Per-repo breakdown
    is computed across all repos present.

    Args:
        results: Per-example evaluation results for one model group.

    Returns:
        Aggregated F2PMetrics.

    Raises:
        ValueError: if *results* is empty.
    """
    if not results:
        raise ValueError("cannot aggregate metrics from an empty results list")

    total_tests = 0
    flaky_tests = 0
    per_repo: dict[str, list[tuple[float, float]]] = {}
    # ponytail: errored results (error is not None) carried p2p=1.0 from the
    # "nothing regressed by definition" rule in compute_f2p, so an all-errored
    # run showed p2p_rate 100%. Rates are computed over error-free results only.
    valid = [r for r in results if r.error is None]
    for result in valid:
        for test in (*result.tests_before, *result.tests_after):
            total_tests += 1
            if test.status == "flaky":
                flaky_tests += 1
        per_repo.setdefault(result.repo, []).append((result.f2p, result.p2p))

    first = results[0]
    example_count = len(results)
    f2p_examples = sum(1 for r in valid if r.f2p > 0.0)
    p2p_examples = sum(1 for r in valid if r.p2p > 0.0)
    successful = sum(1 for r in results if r.patch_application.success)
    resolved = sum(1 for r in valid if is_instance_resolved(r))
    resolved_applied = sum(
        1 for r in results if r.patch_application.success and is_instance_resolved(r)
    )

    breakdown = {
        repo: {
            "f2p_rate": sum(v[0] for v in values) / len(values),
            "p2p_rate": sum(v[1] for v in values) / len(values),
            "count": len(values),
        }
        for repo, values in per_repo.items()
    }

    return F2PMetrics(
        model_name=first.model_name,
        variant=first.variant,
        prompt_template=first.prompt_template,
        total_examples=example_count,
        successful_patches=successful,
        f2p_rate=sum(r.f2p for r in valid) / len(valid) if valid else 0.0,
        f2p_count=f2p_examples,
        p2p_rate=sum(r.p2p for r in valid) / len(valid) if valid else 0.0,
        p2p_count=p2p_examples,
        avg_latency=sum(r.latency_seconds for r in results) / example_count,
        flaky_test_rate=flaky_tests / total_tests if total_tests else 0.0,
        per_repo_breakdown=breakdown,
        apply_rate=apply_rate(successful, example_count),
        resolve_rate=resolved / example_count if example_count else 0.0,
        resolve_count=resolved,
        conditional_resolve_rate=conditional_resolve_rate(resolved_applied, successful),
    )
