"""Serving telemetry: request records, W&B flush loop, GPU/cost helpers.

Pure stdlib only (no fastapi/vllm/modal); ``wandb`` is imported lazily inside
the functions that need it so local dev never requires a W&B install or
credentials.  Metric keys follow the single Phase 6 convention
(``serve/*``) shared by the flush loop and the benchmark.

Percentiles are nearest-rank over a sorted list — no numpy dependency.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from dataclasses import dataclass

from inference.config import ServeConfig

logger = logging.getLogger(__name__)


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


def run_flush_loop(
    collector: MetricsCollector, config: ServeConfig, stop_event: threading.Event
) -> None:
    """Background flush loop: every ``telemetry_flush_interval_seconds`` log
    the rolling ``serve/*`` metrics plus ``serve/gpu_util`` (when the sample
    succeeds) in a single ``wandb.log`` call.  Exits promptly on stop."""
    while not stop_event.wait(config.telemetry_flush_interval_seconds):
        import wandb

        if wandb.run is None:
            continue
        metrics = _metrics_dict(collector)
        gpu_util = log_gpu_util()
        if gpu_util is not None:
            metrics["serve/gpu_util"] = gpu_util
        wandb.log(metrics)


def finish_wandb() -> None:
    """Explicit ``wandb.finish()`` teardown.

    Modal kills the container before ``atexit`` runs (Phase 4 lesson), so
    call sites (Wave 3 modal_serve) invoke this explicitly on shutdown.
    """
    import wandb

    if wandb.run is not None:
        wandb.finish()
