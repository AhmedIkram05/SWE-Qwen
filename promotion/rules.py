"""Pure Champion/Challenger promotion decision rules (Phase 9, ADR-007/ADR-014).

No I/O, no cloud dependencies, no side effects: ``decide`` is a pure function
of five scalars so it is offline-testable and auditable.  The gain/regression
thresholds are env-overridable at import time; the absolute quality floors are
hardcoded (ADR-014): a candidate below either floor can never be promoted.

Decision gates (checked in order — first failing gate wins, one reason):

1. **Floors** — candidate must meet absolute quality floors
   (``candidate_f2p >= 0.15`` and ``candidate_p2p >= 0.90``).
2. **Margin** — candidate must beat the champion by at least
   ``PROMOTE_MIN_F2P_GAIN`` in F2P.
3. **P2P ceiling** — candidate may regress at most ``PROMOTE_MAX_P2P_REGRESSION``
   in P2P relative to the champion.
4. **Significance** — the paired-bootstrap CI lower bound of the F2P gain must
   be strictly positive (``ci_lower > 0``); equality fails.

``==`` is not a pass anywhere: margin/ceiling pass at exact equality (strict
``<``), the significance gate fails at ``ci_lower == 0.0`` (``<=``).
"""

from __future__ import annotations

import os

# Gain the candidate must achieve over the champion to be promotable.
PROMOTE_MIN_F2P_GAIN: float = float(os.getenv("PROMOTE_MIN_F2P_GAIN", "0.05"))
# Maximum P2P drop the candidate may show relative to the champion.
PROMOTE_MAX_P2P_REGRESSION: float = float(os.getenv("PROMOTE_MAX_P2P_REGRESSION", "0.02"))

# Absolute quality floors (ADR-014) — hardcoded literals, not env-overridable.
MIN_F2P_FLOOR: float = 0.15
MIN_P2P_FLOOR: float = 0.90

OUTCOME_PROMOTE: str = "promote"
OUTCOME_REJECT: str = "reject"

REASON_FATAL_FLAW: str = "fatal-flaw"  # floor or margin failure
REASON_REGRESSION: str = "regression"  # P2P drop beyond the ceiling
REASON_MICRO_GAIN: str = "micro-gain"  # margin passes but CI includes zero

__all__ = [
    "PROMOTE_MIN_F2P_GAIN",
    "PROMOTE_MAX_P2P_REGRESSION",
    "MIN_F2P_FLOOR",
    "MIN_P2P_FLOOR",
    "OUTCOME_PROMOTE",
    "OUTCOME_REJECT",
    "REASON_FATAL_FLAW",
    "REASON_REGRESSION",
    "REASON_MICRO_GAIN",
    "decide",
]


def decide(
    champion_f2p: float,
    candidate_f2p: float,
    champion_p2p: float,
    candidate_p2p: float,
    ci_lower: float,
) -> tuple[str, list[str]]:
    """Decide whether the candidate may be promoted over the champion.

    Four-gate semantics, evaluated in order; the first failing gate wins and
    the reject carries exactly one reason:

    1. Floors: ``candidate_f2p < MIN_F2P_FLOOR`` or
       ``candidate_p2p < MIN_P2P_FLOOR`` -> ``("reject", ["fatal-flaw"])``.
    2. Margin: ``candidate_f2p < champion_f2p + PROMOTE_MIN_F2P_GAIN``
       (strict ``<``; exact equality passes) -> ``("reject", ["fatal-flaw"])``.
    3. P2P ceiling: ``candidate_p2p < champion_p2p - PROMOTE_MAX_P2P_REGRESSION``
       (strict ``<``; exact equality passes) -> ``("reject", ["regression"])``.
    4. Significance: ``ci_lower <= 0`` (equality fails) ->
       ``("reject", ["micro-gain"])``.

    Otherwise -> ``("promote", [])``.

    Args:
        champion_f2p: Champion F2P rate (proportion in [0, 1]).
        candidate_f2p: Candidate F2P rate (proportion in [0, 1]).
        champion_p2p: Champion P2P rate (proportion in [0, 1]).
        candidate_p2p: Candidate P2P rate (proportion in [0, 1]).
        ci_lower: Lower bound of the paired-bootstrap CI of the F2P gain
            (candidate minus champion); > 0 required for promotion.

    Returns:
        ``(outcome, reasons)`` where ``outcome`` is ``"promote"`` or
        ``"reject"`` and ``reasons`` is an empty list on promote or a
        single-element list with the first failing gate's reason on reject.
    """
    if candidate_f2p < MIN_F2P_FLOOR or candidate_p2p < MIN_P2P_FLOOR:
        return OUTCOME_REJECT, [REASON_FATAL_FLAW]
    if candidate_f2p < champion_f2p + PROMOTE_MIN_F2P_GAIN:
        return OUTCOME_REJECT, [REASON_FATAL_FLAW]
    if candidate_p2p < champion_p2p - PROMOTE_MAX_P2P_REGRESSION:
        return OUTCOME_REJECT, [REASON_REGRESSION]
    if ci_lower <= 0:
        return OUTCOME_REJECT, [REASON_MICRO_GAIN]
    return OUTCOME_PROMOTE, []
