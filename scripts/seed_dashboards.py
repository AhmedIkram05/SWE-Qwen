#!/usr/bin/env python3
"""Seed a W&B project with one synthetic run emitting every registered key.

Plan §5.6 / decision 5: the dashboards are built as code, but panels need
real-shaped curves to be useful — this script logs a single run
(``dashboard-seed``, tagged ``seed``) that emits *every* key in
``observability.metrics.METRIC_REGISTRY`` over ``--steps`` synthetic steps.

Run::

    uv run python scripts/seed_dashboards.py --project swe-qwen [--entity ...] [--steps 60]

Requires ``wandb login`` (or ``WANDB_API_KEY``); wandb is imported lazily so
the script fails with a clear message instead of an import traceback. The run
is finished after seeding and its URL is printed — delete/archive it later if
desired (it is tagged ``seed``).

Hierarchical eval keys are emitted under their real shape, e.g.
``eval/qwen3-14b/baseline_14b/template_v1/latency_p50``.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

from observability.metrics import METRIC_REGISTRY

_RNG_SEED = 42  # deterministic seed run: same curves on every invocation.
_SEGMENT = "qwen3-14b/baseline_14b/template_v1"  # eval/{model}/{variant}/{template}
_EVAL_OFFSET = 7  # eval checkpoints fire at step % 15 == this
_DEPLOY_FAIL_STEP = 30  # one deploy failure per 45-step cycle (red dot)


def expected_keys() -> set[str]:
    """Every concrete key METRIC_REGISTRY allows, incl. resolved eval segments."""
    expected: set[str] = set()
    for domain, metrics in METRIC_REGISTRY.items():
        for metric in metrics:
            if metric.startswith("{key}/"):  # eval hierarchical pattern
                suffix = metric.rsplit("/", 1)[-1]
                expected.add(f"eval/{_SEGMENT}/{suffix}")
            else:
                expected.add(f"{domain}/{metric}")
    return expected


def build_step(step: int, total: int, rng: random.Random) -> dict[str, float | int | str]:
    """One synthetic step: continuous telemetry plus cadence-driven events.

    Continuous keys (serve/*, train/*, cost/*) are logged every step; data/*,
    deploy/* and eval/* fire on cadences (with the final step as a catch-all)
    so every registered key is emitted at least once per run.
    """
    progress = step / max(total - 1, 1)
    final = step == total - 1

    ttfb_p50 = min(max(300 + 120 * math.sin(step / 8) + rng.uniform(-30, 30), 150), 450)
    latency_p50 = ttfb_p50 + rng.uniform(80, 200)
    gpu_util = min(max(0.82 + 0.10 * math.sin(step / 7) + rng.uniform(-0.05, 0.05), 0.70), 0.95)
    loss = max(2.5 - 1.7 * progress + rng.uniform(-0.05, 0.05), 0.05)
    lr = 1.5e-4 * min((step + 1) / 10, 1.0) * (1.0 - 0.8 * progress)  # warmup + decay
    error_rate = 0.005 + 0.045 * (0.5 + 0.5 * math.sin(step / 6)) * rng.uniform(0.5, 1.0)
    error_rate = min(max(error_rate, 0.001), 0.10)
    tokens_per_sec = 100 + 45 * math.sin(step / 10) + rng.uniform(-10, 10)
    tokens_per_sec = min(max(tokens_per_sec, 50), 150)
    grad_norm = 1.5 + 1.0 * math.sin(step / 4) + rng.uniform(-0.5, 0.5)
    grad_norm = min(max(grad_norm, 0.1), 3.5)
    cost_per_inference = 0.0006 + 0.0003 * math.sin(step / 9) + rng.uniform(-0.0001, 0.0001)

    metrics: dict[str, float | int | str] = {
        # serving telemetry
        "serve/request_count": round(100 + 6 * step + rng.uniform(-20, 20)),
        "serve/error_rate": round(error_rate, 4),
        "serve/ttfb_p50_ms": round(ttfb_p50, 1),
        "serve/ttfb_p95_ms": round(ttfb_p50 * rng.uniform(1.6, 2.1), 1),
        "serve/latency_p50_ms": round(latency_p50, 1),
        "serve/latency_p95_ms": round(latency_p50 * rng.uniform(1.4, 2.0), 1),
        "serve/tokens_per_sec": round(tokens_per_sec, 1),
        "serve/gpu_util": round(gpu_util, 3),
        "serve/cold_start_s": round(8.5 + 2.5 * math.sin(step / 5) + rng.uniform(-1.0, 1.0), 2),
        "serve/cost_usd": round(0.05 * (step + 1) * (1 + 0.05 * math.sin(step / 3)), 4),
        "serve/cost_per_inference_usd": round(cost_per_inference, 6),
        # training telemetry
        "train/loss": round(loss, 4),
        "train/lr": round(lr, 9),
        "train/grad_norm": round(grad_norm, 3),
        "train/gpu_util": round(gpu_util, 3),
        "train/epoch": round(3.0 * progress, 3),
        "train/step": step,
        "train/cost_usd": round(0.02 * (step + 1) * (1 + 0.05 * math.sin(step / 3)), 4),
        # cost telemetry (cumulative across the seeded run)
        "cost/cost_usd": round(0.08 * (step + 1) * (1 + 0.05 * math.sin(step / 3)), 4),
        "cost/gpu_seconds": round(600 * (step + 1) * (1 + 0.03 * math.sin(step / 5)), 1),
        "cost/rate_per_hour": 2.0,
        # benchmark sweep row (mirrors inference/benchmark.py _sweep_config)
        "sweep/gpu_memory_utilization": round(gpu_util, 3),
        "sweep/max_num_seqs": 16,
        "sweep/quantization": "awq",
        "sweep/max_model_len": 4096,
        "sweep/ttfb_p50_ms": round(ttfb_p50, 1),
        "sweep/tokens_per_sec": round(tokens_per_sec, 1),
        "sweep/total_seconds": round(30 * (step + 1) + rng.uniform(-5, 5), 1),
        "sweep/concurrent_ok": 1,
        "sweep/error": "",
    }

    if step % 20 == 0 or final:  # data pipeline runs
        ingested = 10_000 + 500 * step
        metrics.update(
            {
                "data/records_ingested": ingested,
                "data/records_validated": int(ingested * 0.95),
                "data/records_cleaned": int(ingested * 0.95 * 0.92),
                "data/pipeline_seconds": round(420 + rng.uniform(-60, 60), 1),
            }
        )

    if step % 15 == 0 or final:  # deploys; occasional failure (red dot)
        metrics.update(
            {
                "deploy/status": 0 if step % 45 == _DEPLOY_FAIL_STEP else 1,
                "deploy/duration_s": round(240 + rng.uniform(-60, 90), 1),
            }
        )

    if step % 15 == _EVAL_OFFSET or final:  # evaluation checkpoints
        f2p_rate = min(max(0.30 + 0.40 * progress + rng.uniform(-0.03, 0.03), 0.05), 0.95)
        p2p_rate = min(max(0.92 + 0.05 * math.sin(step / 9) + rng.uniform(-0.02, 0.02), 0.80), 0.99)
        num_examples = 300
        total_cost_usd = 45.0 + 10 * progress + rng.uniform(-2, 2)
        seg_p50 = 1150 + 200 * math.sin(step / 11) + rng.uniform(-80, 80)
        metrics.update(
            {
                "eval/f2p_rate": round(f2p_rate, 4),
                "eval/p2p_rate": round(p2p_rate, 4),
                "eval/num_examples": num_examples,
                "eval/total_cost_usd": round(total_cost_usd, 2),
                "eval/cost_per_fix": round(total_cost_usd / max(f2p_rate * num_examples, 1), 3),
                f"eval/{_SEGMENT}/latency_p50": round(seg_p50, 1),
                f"eval/{_SEGMENT}/latency_p95": round(seg_p50 * rng.uniform(1.4, 1.8), 1),
            }
        )

    return metrics


def seed(project: str, entity: str | None, steps: int) -> int:
    """Log one synthetic run covering the whole registry. Returns exit code."""
    try:
        import wandb
    except ImportError:
        print(
            "ERROR: wandb is not installed — run `uv sync` (wandb is a core dependency).",
            file=sys.stderr,
        )
        return 1

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name="dashboard-seed",
            job_type="seed",
            tags=["seed"],
        )
    except Exception as exc:  # not logged in / no network
        print(
            f"ERROR: could not start the W&B run — run `wandb login` or set WANDB_API_KEY: {exc}",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(_RNG_SEED)
    emitted: set[str] = set()
    try:
        for step in range(steps):
            metrics = build_step(step, steps, rng)
            emitted.update(metrics)
            run.log(metrics)
    finally:
        run.finish()

    missing = expected_keys() - emitted
    if missing:
        print(
            f"WARNING: seed run did not emit {len(missing)} registered keys: {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    print(f"Seeded {steps} steps with {len(emitted)} registered keys (run tagged 'seed').")
    print(f"Run URL: {run.url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed W&B with a synthetic run covering the metric registry"
    )
    parser.add_argument("--project", default="swe-qwen", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity (username or team; defaults to your account)")
    parser.add_argument("--steps", type=int, default=60, help="Synthetic steps")
    args = parser.parse_args()
    return seed(args.project, args.entity, args.steps)


if __name__ == "__main__":
    sys.exit(main())
