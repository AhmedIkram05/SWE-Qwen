"""Observability package — structured logging, telemetry contract, cost, SLOs, Langfuse traces.

No side effects on import: wandb and langfuse SDKs are imported lazily inside
functions (same pattern as inference/telemetry.py), so local dev and CI run
without credentials.
"""

from observability.cost import (
    estimate_cost_usd,
    log_run_cost,
    rate_per_hour_from_config,
)
from observability.langfuse import trace_generation, trace_request
from observability.logging import JsonFormatter, configure_logging
from observability.metrics import METRIC_REGISTRY, assert_registered, log_metrics
from observability.slo import (
    SLO_TARGETS,
    attainment,
    burn_level,
    burn_rate,
    maybe_alert_burn,
)

__all__ = [
    "METRIC_REGISTRY",
    "SLO_TARGETS",
    "JsonFormatter",
    "assert_registered",
    "attainment",
    "burn_level",
    "burn_rate",
    "configure_logging",
    "estimate_cost_usd",
    "log_metrics",
    "log_run_cost",
    "maybe_alert_burn",
    "rate_per_hour_from_config",
    "trace_generation",
    "trace_request",
]
