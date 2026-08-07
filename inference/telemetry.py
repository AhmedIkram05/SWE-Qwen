"""Serving telemetry: request records, W&B flush loop, GPU/cost helpers.

Pure stdlib only (no fastapi/vllm/modal); ``wandb`` and the ``observability.*``
helpers are imported lazily inside the functions that need them so local dev
never requires a W&B install or credentials.  Metric keys follow the single
Phase 6 convention (``serve/*``) shared by the flush loop and the benchmark.

Percentiles are nearest-rank over a sorted list — no numpy dependency.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

from inference.config import ServeConfig

logger = logging.getLogger(__name__)

# Phase 8 decision 1: bounded queue of sampled serving requests awaiting the
# Langfuse drain. ``deque.append`` is atomic under the GIL — the request hot
# path never blocks on the flush-loop consumer, and maxlen bounds memory.
_trace_queue: deque[TraceDatum] = deque(maxlen=500)

# Phase 8 decision 9: trailing per-flush TTFB-p50 attainment samples consumed
# by ``observability.slo.burn_rate``; capped to one SLO window of flushes.
_attainment_history: list[float] = []

_SLO_WINDOW_S = 3600.0  # matches observability.slo.burn_rate's window default


@dataclass
class TraceDatum:
    """One sampled serving request queued for the Langfuse drain."""

    model: str
    template_name: str | None  # prompt-builder template; None when not applicable
    ttfbs_ms: float | None
    latency_ms: float
    output_tokens: int


@dataclass
class RequestRecord:
    """One completed inference request."""

    ts: float  # time.time() at request completion
    model: str  # request model string as sent by the client
    stream: bool
    ttfbs_ms: float | None  # None only when timing unavailable
    latency_ms: float
    output_tokens: int
    error: bool
    error_type: str | None
    status: int


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of a sorted list."""
    idx = min(len(sorted_values) - 1, int(math.ceil(p * len(sorted_values))) - 1)
    return sorted_values[max(idx, 0)]


class MetricsCollector:
    """Thread-safe rolling store of request records with aggregate summaries."""

    def __init__(self) -> None:
        self._records: list[RequestRecord] = []
        self._lock = threading.Lock()

    def record(self, rec: RequestRecord) -> None:
        """Append one record (safe from any thread)."""
        with self._lock:
            self._records.append(rec)

    def summary(self) -> dict[str, float | int]:
        """Aggregate over all recorded requests.

        Keys: ``count``, ``error_rate``, ``ttfb_p50_ms``, ``ttfb_p95_ms``
        (over non-None TTFBs), ``latency_p50_ms``, ``latency_p95_ms``, and
        ``tokens_per_sec`` = total output tokens / total generation seconds.
        Generation seconds approximate engine time: full latency for
        non-streaming requests; ``(latency - ttfbs)`` for streaming requests
        with a recorded TTFB (TTFB is measured before engine output starts).
        """
        with self._lock:
            records = list(self._records)
        count = len(records)
        if count == 0:
            return {
                "count": 0,
                "error_rate": 0.0,
                "ttfb_p50_ms": 0.0,
                "ttfb_p95_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "tokens_per_sec": 0.0,
            }
        ttfbs = sorted(r.ttfbs_ms for r in records if r.ttfbs_ms is not None)
        latencies = sorted(r.latency_ms for r in records)
        gen_seconds = 0.0
        for rec in records:
            if rec.stream and rec.ttfbs_ms is not None:
                gen_seconds += max((rec.latency_ms - rec.ttfbs_ms) / 1000.0, 0.0)
            else:
                gen_seconds += rec.latency_ms / 1000.0
        total_tokens = sum(r.output_tokens for r in records)
        return {
            "count": count,
            "error_rate": sum(1 for r in records if r.error) / count,
            "ttfb_p50_ms": _percentile(ttfbs, 0.50) if ttfbs else 0.0,
            "ttfb_p95_ms": _percentile(ttfbs, 0.95) if ttfbs else 0.0,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "tokens_per_sec": total_tokens / gen_seconds if gen_seconds > 0 else 0.0,
        }


def _metrics_dict(collector: MetricsCollector) -> dict[str, float | int]:
    """The ``serve/*`` W&B metric dict for one collector snapshot."""
    s = collector.summary()
    return {
        "serve/request_count": s["count"],
        "serve/error_rate": s["error_rate"],
        "serve/ttfb_p50_ms": s["ttfb_p50_ms"],
        "serve/ttfb_p95_ms": s["ttfb_p95_ms"],
        "serve/latency_p50_ms": s["latency_p50_ms"],
        "serve/latency_p95_ms": s["latency_p95_ms"],
        "serve/tokens_per_sec": s["tokens_per_sec"],
    }


def log_serving_metrics(collector: MetricsCollector, config: ServeConfig) -> None:
    """Log rolling aggregates to the active W&B run (no-op locally)."""
    import wandb

    if wandb.run is None:
        logger.warning("wandb run not active — skipping serving metrics flush")
        return
    wandb.log(_metrics_dict(collector))


def log_gpu_util() -> float | None:
    """Sample GPU utilization via ``nvidia-smi``; None when unavailable.

    Any failure (no nvidia-smi, command error, parse error) returns None —
    on macOS/local dev this is always None.
    """
    try:
        out = subprocess.run(  # noqa: PLW1510 — checked below
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return float(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def log_cold_start(seconds: float) -> None:
    """Log a cold-start measurement to W&B when a run is active."""
    import wandb

    if wandb.run is not None:
        wandb.log({"serve/cold_start_s": seconds})


def cost_per_inference(gpu_seconds: float, requests: int, rate_per_hour: float) -> float:
    """Cost per inference = GPU-hours × hourly rate ÷ request count."""
    return gpu_seconds / 3600.0 * rate_per_hour / max(requests, 1)


def add_trace_record(
    model: str,
    template_name: str | None,
    ttfbs_ms: float | None,
    latency_ms: float,
    output_tokens: int,
) -> None:
    """Queue one sampled serving request for the Langfuse drain (decision 1).

    Hot-path safe: a single bounded deque append — O(1), never blocks, never
    raises.  The flush loop drains the queue; nothing here touches Langfuse.
    """
    _trace_queue.append(
        TraceDatum(
            model=model,
            template_name=template_name,
            ttfbs_ms=ttfbs_ms,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
        )
    )


def _drain_trace_queue() -> None:
    """Flush every queued sampled trace to Langfuse; never raises, never drops.

    ``trace_request`` itself is a silent no-op without Langfuse keys, so a
    keyless environment drains the queue for free.  A failing import or a
    per-datum SDK error is logged and skipped — the drain always completes.
    """
    if not _trace_queue:
        return
    try:
        from observability.langfuse import trace_request
    except Exception as exc:  # noqa: BLE001 — observability must never break serving
        logger.warning(
            "langfuse unavailable, dropping %d queued traces: %s", len(_trace_queue), exc
        )
        _trace_queue.clear()
        return
    while True:
        try:
            datum = _trace_queue.popleft()
        except IndexError:
            return
        try:
            trace_request(
                model=datum.model,
                template_name=datum.template_name,
                ttfbs_ms=datum.ttfbs_ms,
                latency_ms=datum.latency_ms,
                output_tokens=datum.output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — one bad trace must not stall the drain
            logger.warning("langfuse trace dropped (model=%s): %s", datum.model, exc)


def run_flush_loop(
    collector: MetricsCollector, config: ServeConfig, stop_event: threading.Event
) -> None:
    """Background flush loop: every ``telemetry_flush_interval_seconds`` log
    the rolling ``serve/*`` metrics plus ``serve/gpu_util`` (when the sample
    succeeds), ``serve/cost_usd`` + ``serve/cost_per_inference_usd`` (uptime ×
    rate), alert on error-rate / TTFB-p95 threshold breaches (decision 8), and
    update the SLO error-budget burn (decision 9).  The Langfuse trace drain
    runs every tick independent of W&B.  Exits promptly on stop."""
    loop_start = time.time()
    while not stop_event.wait(config.telemetry_flush_interval_seconds):
        _drain_trace_queue()
        try:
            _flush_tick(collector, config, loop_start)
        except Exception:
            # A single bad tick must never kill the loop (code-review C1/B1):
            # a raising alert/log call would otherwise silently stop all
            # serving telemetry for the rest of the process.
            logger.exception("telemetry flush tick failed; continuing")


def _flush_tick(collector: MetricsCollector, config: ServeConfig, loop_start: float) -> None:
    """One flush tick: W&B metrics/cost/alerts + SLO burn (raises = caller retries)."""
    import wandb

    from observability import slo
    from observability.cost import estimate_cost_usd, rate_per_hour_from_config

    if wandb.run is None:
        return
    summary = collector.summary()
    metrics = _metrics_dict(collector)
    # Decision 8a: serving cost from container uptime (rate from config).
    uptime_s = time.time() - loop_start
    rate_per_hour = rate_per_hour_from_config(config.gpu_type)
    metrics["serve/cost_usd"] = estimate_cost_usd(uptime_s, rate_per_hour)
    metrics["serve/cost_per_inference_usd"] = cost_per_inference(
        uptime_s, int(summary["count"]), rate_per_hour
    )
    gpu_util = log_gpu_util()
    if gpu_util is not None:
        metrics["serve/gpu_util"] = gpu_util
    wandb.log(metrics)
    if summary["count"] == 0:
        return
    # Decision 8b: threshold alerts (only when a run is active).
    if summary["error_rate"] > config.alert_error_rate_threshold:
        wandb.alert(  # type: ignore[attr-defined]  # present at runtime (0.28.1), absent from stubs
            title="serve/error_rate above threshold",
            text=(f"error_rate={summary['error_rate']:.3f} > {config.alert_error_rate_threshold}"),
            level="ERROR",
        )
    if summary["ttfb_p95_ms"] > config.alert_ttfb_p95_threshold_ms:
        wandb.alert(  # type: ignore[attr-defined]  # present at runtime (0.28.1), absent from stubs
            title="serve/ttfb_p95 above threshold",
            text=(
                f"ttfb_p95_ms={summary['ttfb_p95_ms']:.1f} > {config.alert_ttfb_p95_threshold_ms}ms"
            ),
            level="WARN",
        )
    # Decision 9: SLO attainment history + burn rate (no new metric keys).
    attainment = slo.attainment({k: float(v) for k, v in summary.items()})
    if "ttfb_p50_ms" in attainment:
        _attainment_history.append(attainment["ttfb_p50_ms"])
        window_samples = max(1, int(_SLO_WINDOW_S / config.telemetry_flush_interval_seconds))
        _attainment_history[:] = _attainment_history[-window_samples:]
        burn = slo.burn_rate(
            _attainment_history, config.telemetry_flush_interval_seconds, _SLO_WINDOW_S
        )
        slo.maybe_alert_burn(burn, len(_attainment_history))


def finish_wandb() -> None:
    """Explicit ``wandb.finish()`` teardown.

    Modal kills the container before ``atexit`` runs (Phase 4 lesson), so
    call sites (Wave 3 modal_serve) invoke this explicitly on shutdown.
    """
    import wandb

    if wandb.run is not None:
        wandb.finish()
