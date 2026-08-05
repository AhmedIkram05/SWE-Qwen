"""Centralized configuration for the evaluation harness."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalConfig(BaseSettings):
    """Configuration for the evaluation harness.

    Loaded from CLI args > env vars (``EVAL_`` prefix) > ``.env`` > defaults.
    All eval parameters are tunable via environment variables, e.g.
    ``EVAL_CI_SAMPLE_SIZE=100``.
    """

    # Data
    # GCS source; {run_id} placeholder
    golden_data_path: str = "gs://swe-qwen-datasets/datasets/{run_id}/golden.jsonl"
    swebench_verified_filter: str = "metadata.is_verified==true"  # filter from golden

    # Models
    baseline_model: str = "Qwen/Qwen3-14B"
    wandb_entity: str = "2571642-university-of-dundee"  # override via EVAL_WANDB_ENTITY env var
    wandb_project: str = "swe-qwen"
    lora_artifact_pattern: str = "model-qwen3-14b-{variant}"  # W&B artifact naming

    # Modal
    modal_volumes: dict[str, str] = {
        "repo_cache": "eval-repo-cache",
        "test_cache": "eval-test-cache",
    }
    docker_image_base: str = "python:3.11-slim"
    gpu_type: str = "a10g-24gb"  # for inference

    # Test execution
    test_timeout_seconds: int = 30
    repo_timeout_seconds: int = 300
    max_retries: int = 2
    flaky_threshold: float = 0.5  # if pass rate < 0.5 across retries → flaky
    # ponytail: 64-way concurrent .remote() broke the modal 1.5.3 aiohttp
    # client ("'Connection' object has no attribute '_transport'" on every
    # result). 16 keeps wall time similar — test containers idle on pip install.
    max_parallel: int = 16  # parallel test jobs (swebench + fallback paths)
    use_swebench_images: bool = (
        True  # official per-repo swebench images; clone/install fallback  # noqa: E501
    )
    # Ground-truth verification mode: "all" = every instance,
    # "once_per_repo" = first instance per repo verifies, rest skip
    # "none" = skip entirely (dev iteration only, risk of silent env drift).
    verify_mode: str = "once_per_repo"

    # Quality gates (ADR-005, Master Plan S2)
    min_f2p_threshold: float = 0.15  # Quality floor: minimum F2P to pass
    min_p2p_threshold: float = 0.90  # Regression ceiling: P2P >= 90% (no regressions)

    # Sampling
    ci_sample_size: int = 50  # lightweight PR eval
    ci_random_seed: int = 42

    # Resume
    checkpoint_dir: Path = Path("data/eval_checkpoints")
    resume_from: str | None = None  # run_id to resume

    # Output
    output_dir: Path = Path("data/eval_results")
    wandb_log_per_example: bool = True
    wandb_log_aggregate: bool = True

    # Comparison
    comparison_run_ids: str = ""  # comma-separated run IDs for champion selection
    proxy_champion_fallback: str = "baseline_14b"  # fallback when P4 inputs unavailable

    # Eval tiers (EVAL-V5-REDESIGN §2).
    # full is capped at 50: the whole 2820-example golden set is never run
    # (user decision — bare `run` and `--mode full` must stay cheap).
    tier_sizes: dict[str, int] = {"smoke": 20, "dev": 100, "final": 500, "full": 50}
    tier_seed: int = 42  # deterministic subsets → paired significance
    # Per-tier max_new_tokens.  Smoke = probe-size runs (≤20): a real diff
    # appears within ~2K tokens or never, so an 8192 budget only makes probes
    # 4× slower to fail at 4× the cost.  Dev/final/full keep 8192 — room for
    # Qwen3's out-loud reasoning before the diff; extract_patch keeps only
    # the diff block.
    tier_max_new_tokens: dict[str, int] = {
        "smoke": 2048,
        "dev": 2048,
        "final": 2048,
        "full": 2048,
    }
    # Inference GPU: A100-80GB is required for 14B bf16; a10g-24gb works for ≤7B
    inference_gpu: str = "a100-80gb"

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )
