"""Unit tests for inference.telemetry (collector summaries, cost, GPU probe).

Pure stdlib — no wandb run required: the wandb module is faked via
sys.modules when it must not be touched.
"""

import subprocess
import sys
import types
from typing import Any

import pytest

from inference.config import ServeConfig
from inference.telemetry import (
    MetricsCollector,
    RequestRecord,
    cost_per_inference,
    log_gpu_util,
    log_serving_metrics,
)


def _record(**overrides: Any) -> RequestRecord:
    defaults: dict[str, Any] = {
        "ts": 0.0,
        "model": "qwen3-14b",
        "stream": False,
        "ttfbs_ms": None,
        "latency_ms": 1000.0,
        "output_tokens": 10,
        "error": False,
        "error_type": None,
        "status": 200,
    }
    defaults.update(overrides)
    return RequestRecord(**defaults)


class TestSummary:
    def test_empty_collector_is_zero_safe(self):
        summary = MetricsCollector().summary()
        assert summary["count"] == 0
        assert summary["error_rate"] == 0.0
        assert summary["ttfb_p50_ms"] == 0.0
        assert summary["ttfb_p95_ms"] == 0.0
        assert summary["latency_p50_ms"] == 0.0
        assert summary["latency_p95_ms"] == 0.0
        assert summary["tokens_per_sec"] == 0.0

    def test_known_records_nearest_rank_percentiles(self):
        collector = MetricsCollector()
        # TTFBs 100..500, latencies 2000..2400: nearest-rank p50 = 300/2200,
        # p95 = 500/2400.  Non-stream gen seconds = latency/1000 each.
        for i in range(5):
            collector.record(
                _record(
                    ttfbs_ms=100.0 * (i + 1),
                    latency_ms=2000.0 + 100.0 * i,
                    output_tokens=10,
                )
            )
        summary = collector.summary()
        assert summary["count"] == 5
        assert summary["error_rate"] == 0.0
        assert summary["ttfb_p50_ms"] == 300.0
        assert summary["ttfb_p95_ms"] == 500.0
        assert summary["latency_p50_ms"] == 2200.0
        assert summary["latency_p95_ms"] == 2400.0
        # 50 tokens / (2.0 + 2.1 + 2.2 + 2.3 + 2.4) gen seconds.
        assert summary["tokens_per_sec"] == pytest.approx(50.0 / 11.0)

    def test_streaming_gen_seconds_exclude_ttfb(self):
        collector = MetricsCollector()
        collector.record(_record(stream=True, ttfbs_ms=100.0, latency_ms=1100.0, output_tokens=30))
        collector.record(_record(latency_ms=2000.0, output_tokens=40))
        summary = collector.summary()
        # (1100-100)/1000 + 2000/1000 = 3.0 gen seconds for 70 tokens.
        assert summary["tokens_per_sec"] == pytest.approx(70.0 / 3.0)
        assert summary["ttfb_p50_ms"] == 100.0

    def test_error_rate(self):
        collector = MetricsCollector()
        collector.record(_record())
        collector.record(_record(error=True, error_type="engine_error", status=500))
        summary = collector.summary()
        assert summary["count"] == 2
        assert summary["error_rate"] == 0.5


class TestCostPerInference:
    def test_known_cost(self):
        assert cost_per_inference(3600.0, 10, 1.0) == pytest.approx(0.1)

    def test_zero_requests_safe(self):
        assert cost_per_inference(7200.0, 0, 1.0) == pytest.approx(2.0)


class TestGpuUtil:
    def test_nvidia_smi_unavailable_returns_none(self, monkeypatch):
        def _no_nvidia_smi(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
        assert log_gpu_util() is None

    def test_unparseable_output_returns_none(self, monkeypatch):
        class _FakeProc:
            def __init__(self, stdout: str) -> None:
                self.stdout = stdout

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeProc("not-a-number\n"))
        assert log_gpu_util() is None


class TestWandbSafety:
    def test_log_serving_metrics_without_run_is_noop(self, monkeypatch):
        # wandb is lazy-imported; without an active run nothing crashes or logs.
        fake_wandb = types.SimpleNamespace(run=None)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        assert log_serving_metrics(MetricsCollector(), ServeConfig()) is None
