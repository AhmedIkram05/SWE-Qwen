"""Extra coverage for ``inference.telemetry``: metric dict, active-run W&B
paths, the flush loop, the success path of the GPU probe, and cold-start /
teardown hooks. ``wandb`` is installed as a ``sys.modules`` stand-in.
"""

import subprocess
import sys
import threading
import types

import pytest

from inference import telemetry
from inference.config import ServeConfig

pytestmark = pytest.mark.unit


def _record(**overrides):
    defaults = {
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
    return telemetry.RequestRecord(**defaults)


def _collector(n=1):
    collector = telemetry.MetricsCollector()
    for i in range(n):
        collector.record(
            _record(
                ttfbs_ms=100.0 * (i + 1),
                latency_ms=2000.0 + 100.0 * i,
                output_tokens=10,
            )
        )
    return collector


class TestMetricsDict:
    def test_maps_summary_to_serve_keys(self):
        out = telemetry._metrics_dict(_collector(2))
        assert out["serve/request_count"] == 2
        assert out["serve/error_rate"] == 0.0
        assert out["serve/ttfb_p50_ms"] == 100.0
        assert out["serve/ttfb_p95_ms"] == 200.0
        assert out["serve/latency_p95_ms"] == 2100.0
        assert out["serve/tokens_per_sec"] > 0.0


class TestLogServingMetrics:
    def test_active_run_logs(self, mocker):
        logged: list[dict] = []
        fake = types.SimpleNamespace(run=object(), log=logged.append)
        mocker.patch.dict(sys.modules, {"wandb": fake})
        telemetry.log_serving_metrics(_collector(3), ServeConfig())
        assert logged[0]["serve/request_count"] == 3

    def test_inactive_run_is_noop(self, mocker, caplog):
        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(run=None)})
        assert telemetry.log_serving_metrics(_collector(1), ServeConfig()) is None


class TestGpuUtil:
    def test_parses_nvidia_smi_output(self, mocker):
        mocker.patch(
            "inference.telemetry.subprocess.run",
            return_value=types.SimpleNamespace(stdout=" 42\n"),
        )
        assert telemetry.log_gpu_util() == 42.0

    def test_subprocess_error_returns_none(self, mocker):
        mocker.patch(
            "inference.telemetry.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
        )
        assert telemetry.log_gpu_util() is None


class TestLogColdStart:
    def test_active_run_logs(self, mocker):
        logged: list[dict] = []
        fake = types.SimpleNamespace(run=object(), log=logged.append)
        mocker.patch.dict(sys.modules, {"wandb": fake})
        telemetry.log_cold_start(2.5)
        assert logged == [{"serve/cold_start_s": 2.5}]

    def test_no_run_is_noop(self, mocker):
        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(run=None)})
        assert telemetry.log_cold_start(2.5) is None


class TestFlushLoop:
    class _FakeStopEvent(threading.Event):
        def __init__(self, results):
            super().__init__()
            self._results = list(results)
            self.waits: list[float | None] = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self._results.pop(0)

    def test_flushes_metrics_with_and_without_gpu_util(self, mocker):
        logged: list[dict] = []
        fake = types.SimpleNamespace(run=object(), log=logged.append)
        mocker.patch.dict(sys.modules, {"wandb": fake})
        mocker.patch("inference.telemetry.log_gpu_util", side_effect=[42.0, None])
        stop = self._FakeStopEvent([False, False, True])
        telemetry.run_flush_loop(_collector(1), ServeConfig(), stop)
        assert len(logged) == 2
        assert logged[0]["serve/gpu_util"] == 42.0
        assert "serve/gpu_util" not in logged[1]
        assert stop.waits == [ServeConfig().telemetry_flush_interval_seconds] * 3

    def test_no_run_skips_flush(self, mocker):
        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(run=None)})
        stop = self._FakeStopEvent([False, True])
        telemetry.run_flush_loop(telemetry.MetricsCollector(), ServeConfig(), stop)
        assert len(stop.waits) == 2


class TestFinishWandb:
    def test_active_run_finishes(self, mocker):
        finished: list[bool] = []
        fake = types.SimpleNamespace(run=object(), finish=lambda: finished.append(True))
        mocker.patch.dict(sys.modules, {"wandb": fake})
        telemetry.finish_wandb()
        assert finished == [True]

    def test_no_run_is_noop(self, mocker):
        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(run=None)})
        assert telemetry.finish_wandb() is None
