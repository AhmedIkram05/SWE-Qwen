"""Phase 6 inference benchmark CLI: engine-config sweep (6.1) + endpoint
benchmark (6.8) — mirrors evaluation/cli.py's typer structure.

Usage::

    python -m inference.benchmark sweep [--gpu-memory-utilization 0.85,0.90] \\
        [--max-num-seqs 8,16,32] [--quantization fp8] [--no-wandb]
    python -m inference.benchmark benchmark [--requests 10] [--model qwen3-14b] \\
        [--gpu-rate 1.0] [--cold-start] [--no-wandb]

Import-safe locally: the Modal app/function are only *registered* at import
time (lazy, no cloud calls); ``wandb``/``openai``/vLLM are imported inside the
commands (mirrors inference/telemetry.py's lazy-wandb convention).

Reports land in ``docs/planning/SERVING-BENCHMARK-REPORT.md``: ``sweep``
overwrites the file, ``benchmark`` appends its section.
"""

# ruff: noqa: B008  # typer.Option(...) defaults (evaluation/cli.py gets this via pyproject per-file-ignores)

from __future__ import annotations

import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal
import typer

# PLC2701-safe private reuse: module-attr access, not a from-import.
from inference import (
    modal_serve,  # noqa: E402
    telemetry,
)
from inference.config import ServeConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPORT_PATH = _REPO_ROOT / "docs" / "planning" / "SERVING-BENCHMARK-REPORT.md"
_CONCURRENCY_RAMP = (1, 8, 16)
_TTFB_P50_GATE_MS = 500.0  # acceptance criterion S3
_SWEEP_PROMPTS = ["def add(a, b):\n    return"] * 10

app = typer.Typer(
    name="benchmark",
    help="Phase 6 serving benchmarks: 6.1 config sweep + 6.8 endpoint acceptance",
    no_args_is_help=True,
)

# The sweep boots its own function; the serving app (modal_serve) is NOT deployed.
benchmark_app = modal.App("swe-qwen-benchmark")


@benchmark_app.function(
    gpu="A10G:1",
    image=modal_serve.image,
    volumes={"/models": modal_serve.serve_volume},
    secrets=modal_serve._secrets,
    timeout=1800,
)
def _sweep_config(
    gpu_memory_utilization: float, max_num_seqs: int, quantization: str, max_model_len: int
) -> dict[str, Any]:
    """Boot one engine; measure N=10 synthetic generations + 16-thread concurrency."""
    from inference.serve import VLLMEngine

    # ServeConfig is pydantic-settings with the SERVING_ env prefix, so these
    # vars flow into config.quantization / gpu_memory_utilization /
    # max_num_seqs / max_model_len and from there into vLLM's LLM(...)
    # (VLLMEngine._ensure_engine).
    os.environ["SERVING_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
    os.environ["SERVING_MAX_NUM_SEQS"] = str(max_num_seqs)
    os.environ["SERVING_QUANTIZATION"] = quantization
    os.environ["SERVING_MAX_MODEL_LEN"] = str(max_model_len)

    engine = VLLMEngine(ServeConfig())
    latencies_ms: list[float] = []
    total_tokens = 0
    gen_seconds = 0.0
    concurrent_ok = False
    error: str | None = None
    t0 = time.perf_counter()
    try:
        for prompt in _SWEEP_PROMPTS:
            g0 = time.perf_counter()
            result = engine.generate(
                prompt,
                lora=None,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                stop=None,
                repetition_penalty=1.0,
            )
            elapsed_ms = (time.perf_counter() - g0) * 1000.0
            latencies_ms.append(elapsed_ms)
            total_tokens += result.completion_tokens
            gen_seconds += elapsed_ms / 1000.0
        # generate() is synchronous, so per-generation wall time is the TTFB
        # approximation (6.1 spec); tokens/sec come from the same windows.
        # Concurrency check: 16 threads, one generate each — verifies
        # VLLMEngine.generate thread-safety under vLLM's sync engine.
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [
                pool.submit(
                    engine.generate,
                    "def f():\n    return",
                    lora=None,
                    max_tokens=32,
                    temperature=0.0,
                    top_p=1.0,
                    stop=None,
                    repetition_penalty=1.0,
                )
                for _ in range(16)
            ]
            for future in futures:
                future.result(timeout=120)
        concurrent_ok = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    total_seconds = time.perf_counter() - t0
    return {
        "config": {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "quantization": quantization,
            "max_model_len": max_model_len,
        },
        "ttfb_ms_p50": (telemetry._percentile(sorted(latencies_ms), 0.50) if latencies_ms else 0.0),
        "tokens_per_sec": total_tokens / gen_seconds if gen_seconds > 0 else 0.0,
        "total_seconds": total_seconds,
        "concurrent_ok": concurrent_ok,
        "error": error,
    }


@app.command()
def sweep(
    gpu_memory_utilization: str = typer.Option("0.85,0.90", help="comma-separated sweep values"),
    max_num_seqs: str = typer.Option("8,16,32", help="comma-separated sweep values"),
    max_model_len: str = typer.Option("4096", help="comma-separated context lengths (tokens)"),
    quantization: str = typer.Option("awq", help="quantization to probe (fp8|awq)"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="skip W&B logging"),
) -> None:
    """Run the 6.1 engine-config sweep: sequential Modal boots on A10G."""
    config = ServeConfig()
    gpu_values = [float(v) for v in gpu_memory_utilization.split(",") if v.strip()]
    seq_values = [int(v) for v in max_num_seqs.split(",") if v.strip()]
    max_len_values = [int(v) for v in max_model_len.split(",") if v.strip()]
    if not gpu_values or not seq_values or not max_len_values:
        raise typer.BadParameter("expected non-empty comma-separated values")
    modal.enable_output()  # stream remote stdout so `modal run` shows progress
    results: list[dict[str, Any]] = []
    for gmu in gpu_values:
        for mns in seq_values:
            for mml in max_len_values:
                typer.echo(
                    f"[sweep] {quantization} gpu_memory_utilization={gmu} "
                    f"max_num_seqs={mns} max_model_len={mml} ..."
                )
                try:
                    row = _sweep_config.remote(gmu, mns, quantization, mml)
                except Exception as exc:
                    row = {
                        "config": {
                            "gpu_memory_utilization": gmu,
                            "max_num_seqs": mns,
                            "quantization": quantization,
                            "max_model_len": mml,
                        },
                        "ttfb_ms_p50": 0.0,
                        "tokens_per_sec": 0.0,
                        "total_seconds": 0.0,
                        "concurrent_ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(row)
                cfg = row["config"]
                typer.echo(
                    f"[sweep] {cfg['gpu_memory_utilization']}/{cfg['max_num_seqs']}/"
                    f"{cfg['max_model_len']} -> "
                    f"ttfb_p50={row['ttfb_ms_p50']:.1f}ms concurrent_ok={row['concurrent_ok']}"
                )

    report = _sweep_report(results)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")  # sweep overwrites; benchmark appends
    typer.echo(report)

    if not no_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={"benchmark": "sweep"},
        )
        try:
            for row in results:
                cfg = row["config"]
                wandb.log(
                    {
                        "sweep/gpu_memory_utilization": cfg["gpu_memory_utilization"],
                        "sweep/max_num_seqs": cfg["max_num_seqs"],
                        "sweep/quantization": cfg["quantization"],
                        "sweep/max_model_len": cfg["max_model_len"],
                        "sweep/ttfb_p50_ms": row["ttfb_ms_p50"],
                        "sweep/tokens_per_sec": row["tokens_per_sec"],
                        "sweep/total_seconds": row["total_seconds"],
                        "sweep/concurrent_ok": row["concurrent_ok"],
                        "sweep/error": row["error"] or "",
                    }
                )
        finally:
            wandb.finish()


@app.command()
def benchmark(  # noqa: PLR0913, PLR0917
    requests: int = typer.Option(10, help="requests per worker per concurrency level"),
    model: str = typer.Option("qwen3-14b", help="model id to benchmark"),
    gpu_rate: float = typer.Option(
        1.0, "--gpu-rate", help="A10G hourly rate ($) for the cost estimate"
    ),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="skip W&B logging"),
    cold_start: bool = typer.Option(
        False, "--cold-start", help="wait out the idle window, then measure one request"
    ),
    idle_wait: int | None = typer.Option(
        None,
        "--idle-wait",
        help="override idle wait seconds (default: idle_timeout_seconds + 60)",
    ),
    url: str = typer.Option(
        "", envvar="MODAL_WEB_URL", help="deployed endpoint base URL (or MODAL_WEB_URL env)"
    ),
    token: str = typer.Option(
        "", envvar="MODAL_WEB_TOKEN", help="deployed endpoint token (or MODAL_WEB_TOKEN env)"
    ),
) -> None:
    """Run the 6.8 HTTP-level acceptance benchmark against the DEPLOYED endpoint."""
    import openai
    import wandb

    if not url or not token:
        raise typer.BadParameter(
            "MODAL_WEB_URL and MODAL_WEB_TOKEN must be set (env vars or flags)"
        )
    config = ServeConfig()
    # OpenAI SDK does NOT append /v1 — MODAL_WEB_URL is the bare root, and the
    # API routes live under /v1 (health lives at the root, fetched via httpx).
    client = openai.OpenAI(base_url=url.rstrip("/") + "/v1", api_key=token)

    typer.echo("[benchmark] warm-up request ...")
    _chat(client, model)

    metrics, req_per_sec = _measure_endpoint(client, model, requests, gpu_rate)
    gate = "PASS" if metrics["serve/ttfb_p50_ms"] < _TTFB_P50_GATE_MS else "FAIL"
    typer.echo(f"[benchmark] S3 gate (TTFB p50 < {_TTFB_P50_GATE_MS:.0f}ms): {gate}")
    typer.echo(
        f"[benchmark] requests={metrics['serve/request_count']} "
        f"errors={metrics['serve/error_rate'] * 100:.1f}% "
        f"ttfb_p50={metrics['serve/ttfb_p50_ms']:.1f}ms "
        f"p95={metrics['serve/ttfb_p95_ms']:.1f}ms "
        f"tokens/s={metrics['serve/tokens_per_sec']:.1f} req/s={req_per_sec:.1f} "
        f"cost/inference=${metrics['serve/cost_per_inference']:.5f}"
    )

    cold_start_s: float | None = None
    if cold_start:
        idle_seconds = idle_wait if idle_wait is not None else config.idle_timeout_seconds + 60
        typer.echo(f"[benchmark] cold-start probe: sleeping {idle_seconds}s ...")
        _countdown(idle_seconds)
        t0 = time.perf_counter()
        _chat(client, model)
        latency_s = time.perf_counter() - t0
        cold_start_s = max(latency_s - metrics["serve/ttfb_p50_ms"] / 1000.0, 0.0)
        typer.echo(
            f"[benchmark] cold-start latency={latency_s:.1f}s -> cold_start_s={cold_start_s:.1f}s"
        )

    run = None
    if not no_wandb:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={"benchmark": "endpoint"},
        )
    try:
        if run is not None:
            wandb.log(metrics)
            if cold_start_s is not None:
                telemetry.log_cold_start(cold_start_s)  # logs serve/cold_start_s
    finally:
        if run is not None:
            wandb.finish()

    _append_report(_endpoint_report(metrics, req_per_sec, gate, cold_start_s))
    typer.echo(f"[benchmark] report appended to {_REPORT_PATH}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _measure_endpoint(
    client: Any, model: str, requests: int, gpu_rate: float
) -> tuple[dict[str, float | int], float]:
    """Ramp 1 -> 8 -> 16 workers x *requests* requests; return (serve/* metrics, req/s).

    TTFB == latency for non-stream requests.
    """
    all_latencies: list[float] = []
    errors = 0
    total_tokens = 0
    wall_t0 = time.perf_counter()
    for workers in _CONCURRENCY_RAMP:
        typer.echo(f"[benchmark] concurrency ramp: {workers} workers x {requests} requests ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, client, model, requests) for _ in range(workers)]
            for future in futures:
                latencies, err, tokens = future.result()
                all_latencies.extend(latencies)
                errors += err
                total_tokens += tokens
    wall_seconds = time.perf_counter() - wall_t0

    sorted_latencies = sorted(all_latencies)
    ttfb_p50 = telemetry._percentile(sorted_latencies, 0.50) if sorted_latencies else 0.0
    ttfb_p95 = telemetry._percentile(sorted_latencies, 0.95) if sorted_latencies else 0.0
    request_count = len(all_latencies)
    error_rate = errors / (request_count + errors) if (request_count + errors) else 0.0
    tokens_per_sec = total_tokens / wall_seconds if wall_seconds > 0 else 0.0
    req_per_sec = request_count / wall_seconds if wall_seconds > 0 else 0.0
    metrics: dict[str, float | int] = {
        "serve/ttfb_p50_ms": ttfb_p50,
        "serve/ttfb_p95_ms": ttfb_p95,
        "serve/latency_p50_ms": ttfb_p50,
        "serve/latency_p95_ms": ttfb_p95,
        "serve/tokens_per_sec": tokens_per_sec,
        "serve/request_count": request_count,
        "serve/error_rate": error_rate,
        "serve/cost_per_inference": telemetry.cost_per_inference(
            wall_seconds, request_count, gpu_rate
        ),
    }
    return metrics, req_per_sec


def _worker(client: Any, model: str, n: int) -> tuple[list[float], int, int]:
    """Send *n* non-stream chat requests; return (latencies_ms, errors, completion_tokens)."""
    latencies: list[float] = []
    errors = 0
    tokens = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            resp = _chat(client, model)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            tokens += int(resp.usage.completion_tokens)
        except Exception:
            errors += 1
    return latencies, errors, tokens


def _chat(client: Any, model: str) -> Any:
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Write a one-line Python function."}],
        max_tokens=256,
        stream=False,
    )


def _countdown(seconds: int) -> None:
    remaining = seconds
    while remaining > 0:
        step = min(remaining, 10)
        typer.echo(f"[benchmark] idle countdown: {remaining}s ...")
        time.sleep(step)
        remaining -= step


def _sweep_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Serving Benchmark Report",
        "",
        "## 6.1 Engine-config sweep",
        "",
        "| gpu_mem | max_num_seqs | quant | ctx_len | ttfb_p50_ms | tok/s |"
        " total_s | concurrent_ok | error |",
        "|---------|--------------|-------|---------|-------------|--------|---------|---------------|-------|",
    ]
    for row in results:
        cfg = row["config"]
        lines.append(
            f"| {cfg['gpu_memory_utilization']} | {cfg['max_num_seqs']} | {cfg['quantization']} "
            f"| {cfg['max_model_len']} | {row['ttfb_ms_p50']:.1f} | {row['tokens_per_sec']:.1f} "
            f"| {row['total_seconds']:.1f} | {'yes' if row['concurrent_ok'] else 'no'} "
            f"| {row['error'] or '—'} |"
        )
    lines.append("")
    lines.append(f"**Selected config:** {_select_config(results)}")
    return "\n".join(lines) + "\n"


def _select_config(results: list[dict[str, Any]]) -> str:
    ok = [r for r in results if r["concurrent_ok"] and r["error"] is None]
    if not ok:
        return "0.85/16/4096 (default — all configs failed; investigate container logs)"
    best = min(ok, key=lambda r: r["ttfb_ms_p50"])
    cfg = best["config"]
    return (
        f"{cfg['gpu_memory_utilization']}/{cfg['max_num_seqs']}/{cfg['max_model_len']} "
        "(lowest TTFB p50, concurrent_ok)"
    )


def _endpoint_report(
    metrics: dict[str, float | int], req_per_sec: float, gate: str, cold_start_s: float | None
) -> str:
    lines = [
        "",
        "## 6.8 Endpoint benchmark",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Requests: {metrics['serve/request_count']}, "
        f"errors: {metrics['serve/error_rate'] * 100:.1f}%",
        f"- TTFB p50: {metrics['serve/ttfb_p50_ms']:.1f} ms, "
        f"p95: {metrics['serve/ttfb_p95_ms']:.1f} ms",
        f"- Latency p50: {metrics['serve/latency_p50_ms']:.1f} ms, "
        f"p95: {metrics['serve/latency_p95_ms']:.1f} ms",
        f"- Tokens/s: {metrics['serve/tokens_per_sec']:.1f}, throughput: {req_per_sec:.1f} req/s",
        f"- Cost per inference: ${metrics['serve/cost_per_inference']:.5f}",
        f"- **S3 gate (TTFB p50 < 500 ms): {gate}**",
    ]
    if cold_start_s is not None:
        lines.append(f"- Cold start: {cold_start_s:.1f} s")
    return "\n".join(lines) + "\n"


def _append_report(section: str) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _REPORT_PATH.read_text(encoding="utf-8") if _REPORT_PATH.exists() else ""
    _REPORT_PATH.write_text(
        (existing.rstrip() + "\n" + section) if existing else section, encoding="utf-8"
    )


if __name__ == "__main__":
    app()
