"""Paired champion-vs-candidate evaluation gate (Phase 9, task 9.1).

``promotion/`` composes ``evaluation/`` (spec decision 1): this module calls
``evaluation.comparison`` and ``evaluation.stats`` directly instead of
reimplementing aggregation or significance.  It bridges two single-variant
``EvalRun`` objects from the same evaluation window — same ``dataset_run_id``
and ``tier_seed``, asserted from the embedded ``EvalConfig`` — into one
auditable :class:`PairEval`:

- per-instance 0/1 F2P vectors built from ``run.results``, filtered by the
  run's variant and intersected on ``instance_id`` so both sides are paired
  over the same instance set (spec §4.2);
- significance from :func:`evaluation.stats.paired_bootstrap_ci` with the
  candidate as argument ``a`` (``ci_lower > 0`` means the candidate beats the
  champion) plus the exact two-sided McNemar p-value;
- aggregate rates via :func:`evaluation.comparison.extract_model_metrics`.

``comparison.paired_significance`` is deliberately NOT used for the gate: it
pairs the same variant across two runs and silently reports "no overlap" for
different variants — display only (spec decision 1).

Pure computation over in-memory runs: no I/O, no cloud dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation import comparison
from evaluation.comparison import extract_model_metrics
from evaluation.config import EvalConfig
from evaluation.schema import EvalRun, F2PMetrics
from evaluation.stats import mcnamar_p, paired_bootstrap_ci

__all__ = ["PairEval", "evaluate_pair", "revalidate_champion"]


@dataclass(frozen=True)
class PairEval:
    """Immutable, auditable snapshot of one paired champion-vs-candidate eval.

    ``ci_lower``/``ci_high`` bound the paired-bootstrap CI of the F2P gain
    (candidate minus champion); ``ci_lower > 0`` means the candidate beats
    the champion.  ``mcnemar_p`` is the exact two-sided McNemar p-value over
    the discordant instance pairs.
    """

    champion_metrics: F2PMetrics | None
    candidate_metrics: F2PMetrics | None
    f2p_gain: float
    p2p_delta: float
    ci_lower: float
    ci_high: float
    mcnemar_p: float


def _run_key(run: EvalRun) -> str:
    """Return the single ``"model:variant"`` key a run evaluates.

    The canonical key comes from ``run.models_evaluated`` (the same
    ``"model:variant"`` convention ``extract_model_metrics`` produces); when
    that list is empty it is derived from the results.  Paired promotion
    requires exactly one evaluated variant per run — anything else makes the
    pairing ambiguous.

    Raises:
        ValueError: if the run evaluates zero-or-many distinct variants.
    """
    keys = list(run.models_evaluated) or sorted(
        {f"{r.model_name}:{r.variant}" for r in run.results}
    )
    variants = {key.partition(":")[2] for key in keys}
    if len(variants) != 1:
        raise ValueError(
            f"run {run.run_id} evaluates {len(variants)} distinct variants "
            f"({sorted(variants)}) — paired promotion needs exactly one"
        )
    return keys[0]


def _f2p_vector(run: EvalRun, variant: str) -> dict[str, float]:
    """Per-instance 0/1 F2P vector for one variant (spec §4.2 formula)."""
    return {r.instance_id: 1.0 if r.f2p > 0 else 0.0 for r in run.results if r.variant == variant}


def _gains(champion: F2PMetrics | None, candidate: F2PMetrics | None) -> tuple[float, float]:
    """F2P gain / P2P delta from aggregate rates, ``(0.0, 0.0)`` when missing."""
    if champion is None or candidate is None:
        return (0.0, 0.0)
    return (candidate.f2p_rate - champion.f2p_rate, candidate.p2p_rate - champion.p2p_rate)


def evaluate_pair(
    champion_run: EvalRun,
    candidate_run: EvalRun,
    config: EvalConfig,
) -> PairEval:
    """Evaluate a candidate head-to-head against the champion on paired instances.

    Both runs must come from the same evaluation window: equal
    ``dataset_run_id`` and ``tier_seed`` are asserted from the embedded run
    configs (spec decision 4) so the seed-42 instance subsets coincide.
    Per-instance 0/1 F2P vectors are built per variant and intersected on
    ``instance_id``; significance is computed with the candidate as argument
    ``a`` of ``paired_bootstrap_ci`` so ``ci_lower > 0`` is a candidate win.

    *config* is accepted for pipeline parity (``promotion.run`` passes the
    launch config); pairing is enforced from the runs' embedded configs, so
    it is not consulted by this function.

    Args:
        champion_run: Completed eval run of the incumbent (one variant).
        candidate_run: Completed eval run of the challenger (one variant).
        config: Eval config of the pairing (unused; API parity).

    Returns:
        The frozen :class:`PairEval` snapshot.

    Raises:
        AssertionError: if the runs' ``dataset_run_id`` or ``tier_seed``
            differ — they are not from the same evaluation window.
        ValueError: if either run has no results, evaluates zero-or-many
            variants, or the instance intersection is empty.
    """
    assert champion_run.config.dataset_run_id == candidate_run.config.dataset_run_id, (
        f"unpaired runs: champion={champion_run.run_id} dataset_run_id="
        f"{champion_run.config.dataset_run_id!r} != candidate={candidate_run.run_id} "
        f"dataset_run_id={candidate_run.config.dataset_run_id!r}"
    )
    assert champion_run.config.tier_seed == candidate_run.config.tier_seed, (
        f"unpaired runs: champion={champion_run.run_id} tier_seed="
        f"{champion_run.config.tier_seed} != candidate={candidate_run.run_id} "
        f"tier_seed={candidate_run.config.tier_seed}"
    )
    if not champion_run.results or not candidate_run.results:
        raise ValueError(
            f"cannot pair runs with no results: champion={champion_run.run_id} "
            f"({len(champion_run.results)} results), candidate={candidate_run.run_id} "
            f"({len(candidate_run.results)} results)"
        )

    champion_key = _run_key(champion_run)
    candidate_key = _run_key(candidate_run)
    champion_vec = _f2p_vector(champion_run, champion_key.partition(":")[2])
    candidate_vec = _f2p_vector(candidate_run, candidate_key.partition(":")[2])

    # Deterministic, pairwise-aligned order over the instance intersection.
    shared = sorted(set(champion_vec) & set(candidate_vec))
    if not shared:
        raise ValueError(
            f"no shared instances between champion={champion_run.run_id} and "
            f"candidate={candidate_run.run_id} — cannot pair"
        )
    champ_scores = [champion_vec[instance] for instance in shared]
    cand_scores = [candidate_vec[instance] for instance in shared]

    # Candidate is arg `a`: ci_lower > 0 means the candidate beats the
    # champion.  Defaults seed=42 / n_boot=10_000 per spec §4.2.
    ci_lower, ci_high, _ = paired_bootstrap_ci(cand_scores, champ_scores)

    # b01: champion solved, candidate did not.  b10: candidate solved, champion did not.
    b01 = sum(1 for c, k in zip(champ_scores, cand_scores, strict=True) if c > k)
    b10 = sum(1 for c, k in zip(champ_scores, cand_scores, strict=True) if c < k)
    mcnemar_p_value = mcnamar_p(b01, b10)

    merged = extract_model_metrics([champion_run, candidate_run])
    champion_metrics = merged.get(champion_key)
    candidate_metrics = merged.get(candidate_key)
    f2p_gain, p2p_delta = _gains(champion_metrics, candidate_metrics)

    return PairEval(
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
        f2p_gain=f2p_gain,
        p2p_delta=p2p_delta,
        ci_lower=ci_lower,
        ci_high=ci_high,
        mcnemar_p=mcnemar_p_value,
    )


def revalidate_champion(
    metrics: dict[str, F2PMetrics],
    proxy_champion: str,
    min_f2p: float,
    min_p2p: float,
) -> tuple[str, F2PMetrics] | None:
    """Thin passthrough to ``evaluation.comparison.revalidate_champion``.

    Keeps the floor semantics (``p2p_rate >= min_p2p`` and
    ``f2p_rate >= min_f2p``, ranked by ``f2p_rate`` descending) identical to
    the evaluation module so promotion callers never reach into ``evaluation``
    internals (spec §4.2, decision 1).
    """
    return comparison.revalidate_champion(metrics, proxy_champion, min_f2p, min_p2p)
