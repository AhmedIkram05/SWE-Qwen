"""Telemetry contract: the authoritative registry of allowed ``{domain}/{metric}`` keys.

Single source of truth for every key emitted to W&B (plan §4). One namespace per
domain; flat ``{domain}/{metric}`` keys, except eval's hierarchical suffix pattern
``eval/{model}/{variant}/{template}/{metric}`` (harness L793).
"""

from collections.abc import Mapping

METRIC_REGISTRY: dict[str, dict[str, str]] = {
    "serve": {
        "request_count": "Total requests served in the flush window.",
        "error_rate": "Fraction of requests that errored (0.0-1.0).",
        "ttfb_p50_ms": "Median time to first byte, ms.",
        "ttfb_p95_ms": "95th percentile time to first byte, ms.",
        "latency_p50_ms": "Median end-to-end latency, ms.",
        "latency_p95_ms": "95th percentile end-to-end latency, ms.",
        "tokens_per_sec": "Generation throughput, tokens/second.",
        "gpu_util": "GPU utilization (0.0-1.0).",
        "cold_start_s": "Container cold-start duration, seconds.",
        "cost_usd": "Cumulative serving cost, USD.",
        "cost_per_inference_usd": "Cost per single inference, USD.",
    },
    "sweep": {
        "gpu_memory_utilization": "vLLM GPU memory utilization in the sweep run.",
        "max_num_seqs": "vLLM max concurrent sequences in the sweep run.",
        "quantization": "vLLM quantization scheme in the sweep run.",
        "max_model_len": "vLLM max model length in the sweep run.",
        "ttfb_p50_ms": "Sweep p50 time-to-first-byte, ms.",
        "tokens_per_sec": "Sweep generation throughput, tokens/second.",
        "total_seconds": "Sweep wall-clock duration, seconds.",
        "concurrent_ok": "Sweep concurrency gate: 1 pass, 0 fail.",
        "error": 'Sweep error string ("" when clean).',
    },
    "train": {
        "loss": "Training loss at the current step.",
        "lr": "Learning rate at the current step.",
        "grad_norm": "Gradient norm at the current step.",
        "gpu_util": "GPU utilization (0.0-1.0).",
        "epoch": "Current epoch (float).",
        "step": "Training step count.",
        "cost_usd": "Cumulative training cost, USD.",
    },
    "eval": {
        "f2p_rate": "Fail-to-pass rate (0.0-1.0).",
        "p2p_rate": "Pass-to-pass rate (0.0-1.0).",
        "num_examples": "Number of golden examples evaluated.",
        "total_cost_usd": "Total evaluation cost, USD.",
        "cost_per_fix": "Eval cost divided by F2P-passing examples, USD.",
        "{key}/latency_p50": "Median latency for a model/variant/template segment, ms.",
        "{key}/latency_p95": "95th percentile latency for a segment, ms.",
    },
    "cost": {
        "cost_usd": "Estimated cost of a run, USD.",
        "gpu_seconds": "GPU-seconds consumed by a run.",
        "rate_per_hour": "Assumed GPU rate, USD/hour.",
    },
    "data": {
        "records_ingested": "Records ingested by the pipeline.",
        "records_validated": "Records passing validation.",
        "records_cleaned": "Records after cleaning.",
        "pipeline_seconds": "Pipeline wall time, seconds.",
    },
    "deploy": {
        "status": "Deploy outcome: 1 success, 0 failure.",
        "duration_s": "Deploy duration, seconds.",
    },
}

_EVAL_HIERARCHICAL_SUFFIXES = ("latency_p50", "latency_p95")
_MIN_EVAL_SEGMENTS = 3  # eval/<model>/<variant>/<template>/<metric>


def _is_eval_hierarchical(key: str) -> bool:
    """True for ``eval/{model}/{variant}/{template}/{metric}`` keys (harness L793)."""
    if not key.startswith("eval/") or len(key.split("/")) < _MIN_EVAL_SEGMENTS:
        return False
    return key.rsplit("/", 1)[-1] in _EVAL_HIERARCHICAL_SUFFIXES


def _is_registered(key: str) -> bool:
    domain, sep, metric = key.partition("/")
    if not sep or domain not in METRIC_REGISTRY:
        return False
    return metric in METRIC_REGISTRY[domain] or _is_eval_hierarchical(key)


def assert_registered(metrics: Mapping[str, float | int | str | None]) -> None:
    """Raise ``KeyError`` naming the first key not covered by the registry.

    Also accepts eval's hierarchical ``eval/{key}/latency_p50|p95`` pattern.
    Returns ``None`` on success.
    """
    for key in metrics:
        if not _is_registered(key):
            raise KeyError(f"unregistered metric key: {key!r}")


def log_metrics(metrics: dict[str, float | int | str | None]) -> None:
    """Validate *metrics* against the registry, then log to the active W&B run.

    ``wandb`` is imported lazily; without an active run this is a no-op.
    """
    assert_registered(metrics)
    import wandb

    if wandb.run is not None:
        wandb.log(metrics)
