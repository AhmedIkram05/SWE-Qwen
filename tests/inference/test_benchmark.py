"""Coverage for ``inference.benchmark`` (6.1 sweep + 6.8 endpoint CLIs).

Everything external is mocked: the Modal ``_sweep_config.remote`` call, the
vLLM engine, ``wandb``/``openai``, and the report filesystem path (tmp_path).
No network, no cloud, no GPU.
"""

import sys
import types
from typing import Any

import pytest
from typer.testing import CliRunner

from inference import benchmark

pytestmark = pytest.mark.unit


_METRICS = {
    "serve/ttfb_p50_ms": 120.0,
    "serve/ttfb_p95_ms": 220.0,
    "serve/latency_p50_ms": 120.0,
    "serve/latency_p95_ms": 220.0,
    "serve/tokens_per_sec": 88.0,
    "serve/request_count": 25,
    "serve/error_rate": 0.04,
    "serve/cost_per_inference_usd": 0.0012,
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_wandb(mocker):
    """Installed ``sys.modules["wandb"]`` stand-in; init/log/finish recorded."""
    log_calls: list[dict] = []
    init_calls: list[dict] = []
    fake = types.SimpleNamespace(
        init=lambda **kw: (init_calls.append(kw), "fake-run")[1],
        log=log_calls.append,
        finish=lambda: None,
    )
    mocker.patch.dict(sys.modules, {"wandb": fake})
    return {"log": log_calls, "init": init_calls}


@pytest.fixture
def fake_openai(mocker):
    """Installed ``sys.modules["openai"]`` stand-in (client construction only)."""
    created: list[tuple[Any, Any]] = []

    class _FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            created.append((base_url, api_key))

    mocker.patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=_FakeOpenAI)})
    return created


def _sweep_env(monkeypatch) -> None:
    """Restore any SERVING_* keys ``_sweep_config`` mutates (env pollution)."""
    for key in (
        "SERVING_GPU_MEMORY_UTILIZATION",
        "SERVING_MAX_NUM_SEQS",
        "SERVING_QUANTIZATION",
        "SERVING_MAX_MODEL_LEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _sweep_row(gmu=0.85, mns=16, mml=4096, ttfb=123.0, ok=True, error=None):
    return {
        "config": {
            "gpu_memory_utilization": gmu,
            "max_num_seqs": mns,
            "quantization": "awq",
            "max_model_len": mml,
        },
        "ttfb_ms_p50": ttfb,
        "tokens_per_sec": 45.0,
        "total_seconds": 1.25,
        "concurrent_ok": ok,
        "error": error,
    }


class TestSweepConfig:
    def test_measures_generations_and_concurrency(self, mocker, monkeypatch):
        _sweep_env(monkeypatch)

        class FakeEngine:
            generated = 0

            def __init__(self, config):
                self.config = config

            def generate(self, prompt, **kwargs):
                type(self).generated += 1
                return types.SimpleNamespace(completion_tokens=10)

        mocker.patch("inference.serve.VLLMEngine", FakeEngine)
        row = benchmark._sweep_config.remote(0.85, 16, "awq", 4096)
        assert row["concurrent_ok"] is True
        assert row["error"] is None
        assert row["ttfb_ms_p50"] > 0.0
        assert row["tokens_per_sec"] > 0.0
        assert row["total_seconds"] >= 0.0
        assert row["config"] == {
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 16,
            "quantization": "awq",
            "max_model_len": 4096,
        }
        assert FakeEngine.generated == 10 + 16

    def test_engine_error_is_captured(self, mocker, monkeypatch):
        _sweep_env(monkeypatch)

        class BoomEngine:
            def __init__(self, config):
                pass

            def generate(self, *args, **kwargs):
                raise ValueError("out of VRAM")

        mocker.patch("inference.serve.VLLMEngine", BoomEngine)
        row = benchmark._sweep_config.remote(0.85, 16, "awq", 4096)
        assert row["concurrent_ok"] is False
        assert row["error"] == "ValueError: out of VRAM"
        assert row["ttfb_ms_p50"] == 0.0
        assert row["tokens_per_sec"] == 0.0


class TestSweepCommand:
    def test_empty_values_raise_bad_parameter(self, runner, fake_wandb):
        result = runner.invoke(
            benchmark.app, ["sweep", "--gpu-memory-utilization", " ", "--no-wandb"]
        )
        assert result.exit_code != 0
        assert "expected non-empty comma-separated values" in result.output

    def test_runs_remote_rows_and_writes_report(
        self, runner, tmp_path, fake_wandb, mocker, monkeypatch
    ):
        monkeypatch.delenv("MODAL_WEB_URL", raising=False)
        report_path = tmp_path / "docs" / "planning" / "SERVING-BENCHMARK-REPORT.md"
        mocker.patch.object(benchmark, "_REPORT_PATH", report_path)
        state = {"calls": 0}

        def remote(gmu, mns, quant, mml):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("remote boot failed")
            return _sweep_row(gmu=gmu, mns=mns, mml=mml)

        mocker.patch("inference.benchmark._sweep_config", types.SimpleNamespace(remote=remote))

        result = runner.invoke(
            benchmark.app,
            ["sweep", "--gpu-memory-utilization", "0.85,0.90", "--max-num-seqs", "8,16"],
        )
        assert result.exit_code == 0
        assert state["calls"] == 4
        assert "ttfb_p50=123.0ms concurrent_ok=True" in result.output
        assert "ttfb_p50=0.0ms concurrent_ok=False" in result.output
        assert "Serving Benchmark Report" in result.output
        assert report_path.exists()
        report = report_path.read_text(encoding="utf-8")
        assert "## 6.1 Engine-config sweep" in report
        assert "| 0.85 | 16 | awq | 4096 | 123.0 | 45.0 | 1.2 | yes | — |" in report
        assert (
            "| 0.85 | 8 | awq | 4096 | 0.0 | 0.0 | 0.0 | no | RuntimeError: remote boot failed |"
            in report
        )
        assert "lowest TTFB p50, concurrent_ok" in report
        assert len(fake_wandb["log"]) == 4
        assert fake_wandb["log"][0]["sweep/gpu_memory_utilization"] in (0.85, 0.90)

    def test_no_wandb_skips_logging(self, runner, tmp_path, fake_wandb, mocker):
        report_path = tmp_path / "SERVING-BENCHMARK-REPORT.md"
        mocker.patch.object(benchmark, "_REPORT_PATH", report_path)
        mocker.patch(
            "inference.benchmark._sweep_config",
            types.SimpleNamespace(remote=lambda *a, **k: _sweep_row()),
        )
        result = runner.invoke(
            benchmark.app, ["sweep", "--gpu-memory-utilization", "0.85", "--no-wandb"]
        )
        assert result.exit_code == 0
        assert fake_wandb["log"] == []
        assert fake_wandb["init"] == []
        assert report_path.exists()


class TestBenchmarkCommand:
    def test_missing_url_token_is_bad_parameter(self, runner, monkeypatch, fake_wandb, fake_openai):
        monkeypatch.delenv("MODAL_WEB_URL", raising=False)
        monkeypatch.delenv("MODAL_WEB_TOKEN", raising=False)
        result = runner.invoke(benchmark.app, ["benchmark"])
        assert result.exit_code != 0
        assert "MODAL_WEB_URL and MODAL_WEB_TOKEN must be set" in result.output

    def _invoke_benchmark(self, runner, mocker, *args, metrics=None):
        mocker.patch("inference.benchmark._chat", return_value="warm")
        mocker.patch(
            "inference.benchmark._measure_endpoint",
            return_value=(dict(metrics or _METRICS), 12.5),
        )
        mocker.patch("inference.benchmark._append_report")
        return runner.invoke(benchmark.app, ["benchmark", *args])

    def test_success_gate_pass(self, runner, mocker, fake_wandb, tmp_path):
        mocker.patch.object(benchmark, "_REPORT_PATH", tmp_path / "report.md")
        result = self._invoke_benchmark(
            runner, mocker, "--url", "https://example.com", "--token", "tok"
        )
        assert result.exit_code == 0
        assert "S3 gate (TTFB p50 < 500ms): PASS" in result.output
        assert "requests=25 errors=4.0% ttfb_p50=120.0ms" in result.output
        assert "tokens/s=88.0 req/s=12.5" in result.output
        assert "cost/inference=$0.00120" in result.output
        assert "report appended to" in result.output

    def test_success_gate_fail_without_wandb_logged(self, runner, mocker, tmp_path):
        mocker.patch.object(benchmark, "_REPORT_PATH", tmp_path / "report.md")
        mocker.patch.dict(
            sys.modules, {"wandb": types.SimpleNamespace(run=None, log=None, finish=None)}
        )
        result = self._invoke_benchmark(
            runner,
            mocker,
            "--no-wandb",
            "--url",
            "u",
            "--token",
            "t",
            metrics={**_METRICS, "serve/ttfb_p50_ms": 600.0},
        )
        assert result.exit_code == 0
        assert "S3 gate (TTFB p50 < 500ms): FAIL" in result.output

    def test_cold_start_logs_to_wandb(self, runner, mocker, fake_wandb, tmp_path):
        mocker.patch.object(benchmark, "_REPORT_PATH", tmp_path / "report.md")
        countdown = mocker.patch("inference.benchmark._countdown")
        log_cold = mocker.patch("inference.benchmark.telemetry.log_cold_start")
        result = self._invoke_benchmark(
            runner,
            mocker,
            "--url",
            "u",
            "--token",
            "t",
            "--cold-start",
            "--idle-wait",
            "1",
        )
        assert result.exit_code == 0
        assert "cold-start latency=" in result.output
        countdown.assert_called_once_with(1)
        assert log_cold.called
        assert len(fake_wandb["log"]) == 1  # metrics logged, run finished

    def test_endpoint_created_with_v1_suffix(
        self, runner, mocker, fake_wandb, fake_openai, tmp_path
    ):
        mocker.patch.object(benchmark, "_REPORT_PATH", tmp_path / "report.md")
        self._invoke_benchmark(
            runner, mocker, "--url", "https://example.com", "--token", "tok", "--no-wandb"
        )
        assert fake_openai == [("https://example.com/v1", "tok")]


class TestChat:
    def test_forwards_openai_args(self):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return "resp"

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        assert benchmark._chat(client, "qwen3-14b") == "resp"
        assert seen["model"] == "qwen3-14b"
        assert seen["max_tokens"] == 256
        assert seen["stream"] is False


class TestWorker:
    def test_aggregates_latencies_errors_tokens(self, mocker):
        calls = {"n": 0}

        def fake_chat(client, model):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("connection reset")
            return types.SimpleNamespace(usage=types.SimpleNamespace(completion_tokens=7))

        mocker.patch("inference.benchmark._chat", side_effect=fake_chat)
        latencies, errors, tokens = benchmark._worker(None, "m", 3)
        assert errors == 1
        assert tokens == 14
        assert len(latencies) == 2
        assert all(l >= 0.0 for l in latencies)


class TestMeasureEndpoint:
    def test_ramps_and_aggregates(self, mocker):
        mocker.patch("inference.benchmark._worker", return_value=([10.0, 20.0], 1, 30))
        mocker.patch("inference.benchmark.time.perf_counter", side_effect=[0.0, 0.5])
        metrics, req_per_sec = benchmark._measure_endpoint(None, "m", 2, 1.0)
        workers = 1 + 8 + 16
        assert metrics["serve/request_count"] == workers * 2
        assert metrics["serve/error_rate"] == pytest.approx(workers / (workers * 3))
        assert metrics["serve/tokens_per_sec"] == pytest.approx(workers * 30 / 0.5)
        assert metrics["serve/ttfb_p50_ms"] == 10.0
        assert metrics["serve/ttfb_p95_ms"] == 20.0
        assert metrics["serve/cost_per_inference_usd"] == pytest.approx(
            0.5 / 3600.0 / (workers * 2)
        )
        assert req_per_sec == pytest.approx(workers * 2 / 0.5)

    def test_no_latencies_is_safe(self, mocker):
        mocker.patch("inference.benchmark._worker", return_value=([], 0, 0))
        metrics, req_per_sec = benchmark._measure_endpoint(None, "m", 1, 1.0)
        assert metrics["serve/request_count"] == 0
        assert metrics["serve/ttfb_p50_ms"] == 0.0
        assert req_per_sec == 0.0


class TestCountdown:
    def test_steps_by_ten_then_remainder(self, monkeypatch):
        slept: list[int] = []
        monkeypatch.setattr(benchmark.time, "sleep", slept.append)
        benchmark._countdown(15)
        assert slept == [10, 5]


class TestSelectConfig:
    def test_default_when_none_pass(self):
        results = [_sweep_row(ok=False), _sweep_row(ttfb=9.0, error="RuntimeError: x")]
        assert (
            benchmark._select_config(results)
            == "0.85/16/4096 (default — all configs failed; investigate container logs)"
        )

    def test_best_ttfb_when_some_pass(self):
        results = [
            _sweep_row(ttfb=200.0, error="x"),
            _sweep_row(ttfb=100.0),
            _sweep_row(ttfb=300.0),
        ]
        assert benchmark._select_config(results) == "0.85/16/4096 (lowest TTFB p50, concurrent_ok)"


class TestSweepReport:
    def test_renders_rows(self):
        results = [_sweep_row(), _sweep_row(ttfb=0.0, ok=False, error="ValueError: boom")]
        text = benchmark._sweep_report(results)
        assert text.startswith("# Serving Benchmark Report")
        assert "| gpu_mem | max_num_seqs | quant | ctx_len | ttfb_p50_ms | tok/s |" in text
        assert "| 0.85 | 16 | awq | 4096 | 123.0 | 45.0 | 1.2 | yes | — |" in text
        assert "| 0.85 | 16 | awq | 4096 | 0.0 | 45.0 | 1.2 | no | ValueError: boom |" in text
        assert "**Selected config:**" in text


class TestEndpointReport:
    def test_with_cold_start(self):
        text = benchmark._endpoint_report(dict(_METRICS), 12.5, "PASS", 3.5)
        assert "## 6.8 Endpoint benchmark" in text
        assert f"- Date: {benchmark.datetime.date.today().isoformat()}" in text
        assert "- Requests: 25, errors: 4.0%" in text
        assert "- TTFB p50: 120.0 ms, p95: 220.0 ms" in text
        assert "- Tokens/s: 88.0, throughput: 12.5 req/s" in text
        assert "- Cost per inference: $0.00120" in text
        assert "**S3 gate (TTFB p50 < 500 ms): PASS**" in text
        assert "- Cold start: 3.5 s" in text

    def test_without_cold_start(self):
        text = benchmark._endpoint_report(dict(_METRICS), 12.5, "FAIL", None)
        assert "Cold start" not in text


class TestAppendReport:
    def test_appends_after_existing(self, tmp_path, mocker):
        report = tmp_path / "SERVING-BENCHMARK-REPORT.md"
        report.write_text("existing\n", encoding="utf-8")
        mocker.patch.object(benchmark, "_REPORT_PATH", report)
        benchmark._append_report("# NEW\n")
        assert report.read_text(encoding="utf-8") == "existing\n# NEW\n"

    def test_creates_when_missing(self, tmp_path, mocker):
        report = tmp_path / "sub" / "SERVING-BENCHMARK-REPORT.md"
        mocker.patch.object(benchmark, "_REPORT_PATH", report)
        benchmark._append_report("# NEW")
        assert report.read_text(encoding="utf-8") == "# NEW"
