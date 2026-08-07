"""SLO layer over the master plan S3/S9 serving targets (decision 9).

Derives attainment + error-budget burn from the existing ``serve/*`` collector
summary (duck-typed on dict keys — never imports inference). Emits no metric
keys itself; alerts on fast burn via ``wandb.alert``.
"""

SLO_TARGETS: dict[str, float] = {
    "ttfb_p50_ms": 500.0,  # master plan S3
    "cold_start_s": 10.0,  # master plan S9
}

_ERROR_BUDGET = 0.01  # 1 - 99% SLO: budget fraction consumed per unit burn
_WARN_BURN_RATE = 1.0  # 100% of the window's budget consumed
_ERROR_BURN_RATE = 5.0  # fast burn: budget would drain in ~6 days at this pace


def attainment(summary: dict[str, float]) -> dict[str, float]:
    """Per-SLO attainment: ``1.0`` if the summary meets the target, else ``0.0``.

    Only SLO keys present in *summary* are included, so an absent
    ``cold_start_s`` is skipped rather than scored.
    """
    return {
        key: 1.0 if summary[key] <= target else 0.0
        for key, target in SLO_TARGETS.items()
        if key in summary
    }


def burn_rate(
    attainment_history: list[float], flush_interval_s: float, window_s: float = 3600.0
) -> float:
    """Error-budget burn rate: budget consumed across the window divided by budget.

    Consumed per sample = ``(1 - attainment) * (flush_interval_s / window_s)``.
    Empty history -> ``0.0``.
    """
    if not attainment_history:
        return 0.0
    consumed = sum((1.0 - a) * (flush_interval_s / window_s) for a in attainment_history)
    return consumed / _ERROR_BUDGET


def burn_level(burn_rate: float, n_samples: int, min_samples: int = 10) -> str | None:
    """Map a burn rate to a level; ``None`` when healthy or under-sampled.

    ``n_samples < min_samples`` guards against low-traffic noise. ``ERROR``
    wins over ``WARN``.
    """
    if n_samples < min_samples:
        return None
    if burn_rate >= _ERROR_BURN_RATE:
        return "ERROR"
    if burn_rate >= _WARN_BURN_RATE:
        return "WARN"
    return None


def maybe_alert_burn(burn_rate: float, n_samples: int) -> bool:
    """Alert on fast error-budget burn; ``True`` if ``wandb.alert`` was attempted.

    Fires only while a W&B run is active (lazy import, same pattern as the
    telemetry flush loop).
    """
    level = burn_level(burn_rate, n_samples)
    if level is None:
        return False
    import wandb

    if wandb.run is None:
        return False
    wandb.alert(  # type: ignore[attr-defined]  # present at runtime (0.28.1), absent from stubs
        title=f"SLO error budget burn rate {level}",
        text=(
            f"serve/* error budget burn rate {burn_rate:.2f} ({level}); "
            f"{n_samples} flush samples in window."
        ),
    )
    return True
