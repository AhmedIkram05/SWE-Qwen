"""Cost estimation helpers (decision 3: estimate-first, rate logged for auditability).

Rates come from ``config/observability.yaml``; ``OBSERVABILITY_RATE_PER_HOUR``
overrides them. No wandb import — *wandb_run* is duck-typed.
"""

import os
from pathlib import Path
from typing import Any, Protocol

import yaml


class _Loggable(Protocol):
    """Minimal duck type: anything with a ``.log(dict)`` method."""

    def log(self, data: dict[str, Any]) -> Any: ...


DEFAULT_RATE_PER_HOUR = 1.0  # fallback when config/observability.yaml is missing/unreadable
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "observability.yaml"
_RATE_ENV_VAR = "OBSERVABILITY_RATE_PER_HOUR"


def estimate_cost_usd(gpu_seconds: float, rate_per_hour: float) -> float:
    """Estimated USD cost: ``gpu_seconds / 3600 * rate_per_hour``."""
    return gpu_seconds / 3600.0 * rate_per_hour


def rate_per_hour_from_config(gpu_type: str | None = None) -> float:
    """USD/hour for *gpu_type* from ``config/observability.yaml`` (never raises).

    ``OBSERVABILITY_RATE_PER_HOUR`` takes precedence when set and parseable.
    Missing yaml, unknown key, or unparseable values fall back to
    ``DEFAULT_RATE_PER_HOUR``.
    """
    raw = os.environ.get(_RATE_ENV_VAR)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass  # unparseable override -> fall through to config
    try:
        rates = yaml.safe_load(_CONFIG_PATH.read_text())["rates"]
        return float(rates.get(gpu_type, rates["default"]))
    except (OSError, KeyError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return DEFAULT_RATE_PER_HOUR


def cost_per_fix(total_cost_usd: float, f2p_passes: int) -> float:
    """Eval cost per successful fix: total cost / F2P passes (0-risk guard).

    ``f2p_passes`` counts fixes that pass; 0 passes → the full cost lands on
    cost_per_fix instead of dividing by zero (code-review N2: hoisted from the
    harness so the guard is unit-tested, not source-grepped).
    """
    return total_cost_usd / max(f2p_passes, 1)


def log_run_cost(wandb_run: _Loggable | None, gpu_seconds: float, rate_per_hour: float) -> None:
    """Log the estimated run cost (``cost/*`` keys) to *wandb_run*; no-op if ``None``.

    Duck-typed: any object with a ``.log(dict)`` method works.
    """
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "cost/cost_usd": estimate_cost_usd(gpu_seconds, rate_per_hour),
            "cost/gpu_seconds": gpu_seconds,
            "cost/rate_per_hour": rate_per_hour,
        }
    )
