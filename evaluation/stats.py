"""Statistical helpers for evaluation comparisons.

Small, dependency-free (stdlib only) so it runs anywhere — driver process,
Modal CPU container, or CI.  All functions are pure.

- ``wilson_ci`` — 95% Wilson score interval for a proportion (F2P/P2P rates).
- ``mcnamar_p`` — exact two-sided McNemar p-value for paired binary outcomes
  (did the same instances flip F2P outcome between two variants?).
- ``paired_bootstrap_ci`` — percentile 95% CI for the mean paired difference.

These back the per-variant confidence intervals and paired significance
tests in ``compare``/smoke-gate reporting (ADR-005: F2P primary).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

__all__ = ["wilson_ci", "mcnamar_p", "paired_bootstrap_ci"]

_Z_95 = 1.96  # two-sided 95% normal quantile


def wilson_ci(successes: int, total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion, ``(lower, upper)`` in [0, 1].

    Correct on the edges (0/N and N/N do not collapse to point mass, unlike
    the Wald interval).  ``total <= 0`` -> ``(0.0, 0.0)``.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnamar_p(b01: int, b10: int) -> float:
    """Exact two-sided McNemar p-value for discordant pair counts.

    ``b01`` = pairs where variant A failed, variant B passed;
    ``b10`` = pairs where A passed, B failed.  Under the null, discordant
    pairs split 50/50 (binomial(n, 0.5)); the two-sided p-value is
    ``2 * P(Bin(n, 0.5) <= min(b01, b10))`` capped at 1.0.
    """
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = 10_000,
) -> tuple[float, float, float]:
    """Percentile 95% CI for ``mean(a) - mean(b)`` via paired resampling.

    ``a``/``b`` are per-instance scores for two variants on *identical*
    instances (seed-42 subsets guarantee pairing).  Returns
    ``(lower, upper, observed_mean_diff)``.
    """
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += rng.choice(diffs)
        boots.append(total / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi, observed)
