"""Tests for ``evaluation.stats`` — Wilson CI, McNemar, paired bootstrap."""

from evaluation.stats import mcnamar_p, paired_bootstrap_ci, wilson_ci


def test_wilson_ci_zero_total() -> None:
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 0.0


def test_wilson_ci_perfect() -> None:
    lo, hi = wilson_ci(100, 100)
    assert lo > 0.95  # narrow interval around 1.0
    assert hi == 1.0


def test_wilson_ci_50pct() -> None:
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.39 < lo < 0.41  # approx 0.40 for 95% Wilson at 50/100
    assert 0.59 < hi < 0.61


def test_wilson_ci_zero_successes() -> None:
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0
    assert hi > 0.0
    assert hi < 0.05  # upper bound ~0.036 for 0/100


def test_wilson_ci_small_sample() -> None:
    lo, hi = wilson_ci(3, 10)
    assert 0.0 < lo < 0.2
    assert 0.5 < hi < 0.7


# ── McNemar ────────────────────────────────────────────────────────────────────


def test_mcnamar_no_discordant() -> None:
    # No flips → p = 1.0 (no evidence of difference)
    assert mcnamar_p(0, 0) == 1.0


def test_mcnamar_all_one_way() -> None:
    # 10 flips A→B, 0 flips B→A → strong evidence of improvement
    p = mcnamar_p(10, 0)
    assert 0.001 < p < 0.01


def test_mcnamar_symmetric() -> None:
    # 5 flips each way → no evidence
    p = mcnamar_p(5, 5)
    assert p > 0.9  # should be ~1.0


def test_mcnamar_asymmetric() -> None:
    # 8 vs 2 → moderate evidence
    p = mcnamar_p(8, 2)
    assert 0.05 < p < 0.2


def test_mcnamar_large() -> None:
    # 100 vs 30 → very strong evidence
    p = mcnamar_p(100, 30)
    assert p < 1e-8


# ── Paired Bootstrap CI ────────────────────────────────────────────────────────


def test_bootstrap_identical() -> None:
    a = [1.0] * 50
    lo, hi, obs = paired_bootstrap_ci(a, a)
    assert obs == 0.0
    assert lo <= 0.0 <= hi


def test_bootstrap_shift() -> None:
    a = [0.9] * 50
    b = [0.5] * 50
    lo, hi, obs = paired_bootstrap_ci(a, b)
    assert obs == 0.4
    assert lo > 0.0  # should detect the shift
    assert hi > 0.0


def test_bootstrap_empty() -> None:
    lo, hi, obs = paired_bootstrap_ci([], [])
    assert lo == 0.0
    assert hi == 0.0
    assert obs == 0.0


def test_bootstrap_reproducible() -> None:
    a = [0.8, 0.7, 0.6, 0.5, 0.4]
    b = [0.5, 0.4, 0.3, 0.2, 0.1]
    lo1, hi1, obs1 = paired_bootstrap_ci(a, b, seed=42)
    lo2, hi2, obs2 = paired_bootstrap_ci(a, b, seed=42)
    assert obs1 == obs2
    assert lo1 == lo2
    assert hi1 == hi2
