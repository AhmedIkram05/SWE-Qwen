"""Tests for ``evaluation.stats`` — Wilson CI, McNemar, paired bootstrap."""

import pytest

from evaluation.stats import mcnamar_p, paired_bootstrap_ci, wilson_ci


def test_wilson_ci_zero_total() -> None:
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 0.0


def test_wilson_ci_perfect() -> None:
    lo, hi = wilson_ci(100, 100)
    assert lo > 0.95  # narrow interval around 1.0
    assert hi > 0.999  # floating-point precision near 1.0


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
    assert obs == pytest.approx(0.4)
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


# ── Bulk-sampling contract and edges ──────────────────────────────────────────


def test_bootstrap_single_pair() -> None:
    lo, hi, obs = paired_bootstrap_ci([0.75], [0.25], seed=1)
    assert obs == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)  # one sample resamples to itself


def test_bootstrap_constant_difference_is_point_mass() -> None:
    a = [0.4, 0.6, 0.8, 0.2, 1.0]
    b = [0.1, 0.3, 0.5, -0.1, 0.7]
    lo, hi, obs = paired_bootstrap_ci(a, b, seed=42)
    assert obs == pytest.approx(0.3)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_bootstrap_rejects_empty_resample_count() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_ci([0.0, 1.0], [1.0, 0.0], n_boot=0)


def test_bootstrap_rejects_negative_resample_count() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_ci([0.0, 1.0], [1.0, 0.0], n_boot=-5)


def test_bootstrap_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.0, 1.0], [1.0])


def test_bootstrap_single_resample() -> None:
    lo, hi, obs = paired_bootstrap_ci([0.2, 0.9], [0.1, 0.4], n_boot=1, seed=3)
    assert lo <= hi
    assert lo <= obs <= hi or obs == pytest.approx(0.3)


def test_bootstrap_does_not_draw_one_sample_at_a_time(monkeypatch) -> None:
    """Per-sample ``rng.choice`` in a 10k x n loop is the slow path.

    Guard against reintroducing it: bulk ``choices`` keeps the same
    percentile-bootstrap semantics at a fraction of the cost.
    """
    import random

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("paired_bootstrap_ci must sample in bulk, not per draw")

    monkeypatch.setattr(random.Random, "choice", _explode)
    lo, hi, obs = paired_bootstrap_ci([0.2, 0.9, 0.4], [0.1, 0.4, 0.3], n_boot=50, seed=2)
    assert lo <= hi
    assert obs == pytest.approx(0.2333333333)


def test_bootstrap_ci_width_shrinks_with_more_pairs() -> None:
    import random as _random

    rng = _random.Random(0)
    a = [rng.random() for _ in range(200)]
    b = [rng.random() for _ in range(200)]
    lo_small, hi_small, _ = paired_bootstrap_ci(a[:20], b[:20], seed=42)
    lo_large, hi_large, _ = paired_bootstrap_ci(a, b, seed=42)
    assert (hi_large - lo_large) < (hi_small - lo_small)
