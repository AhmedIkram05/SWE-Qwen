"""Serving configuration for the Phase 6 inference API.

Mirrors ``evaluation.config.EvalConfig`` in shape (pydantic-settings
``BaseSettings``, env-file + defaults, frozen) but is fully independent: the
serving API uses the ``SERVING_`` env prefix and its own knobs so the eval
harness and the inference server never cross-talk through env vars.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ServeConfig(BaseSettings):
    """Serving configuration for the Phase 6 inference API (SERVING_ env prefix)."""

    # Model / adapter registry
    base_model: str = "qwen3-14b"  # registry key in config/models.yaml
    # 4-bit AWQ base: FP8 weight-only on A10G (Marlin) leaves 16.07 GiB of
    # weights and starves the KV cache (Path-A fallback per 6.1)
    serving_hf_id: str = "Qwen/Qwen3-14B-AWQ"
    quantization: str = "awq"  # "fp8" | "awq"
    variants: tuple[str, ...] = ("baseline_14b", "higher_rank_14b", "higher_lr_14b")
    lora_artifact_pattern: str = "model-qwen3-14b-{variant}"
    # champion per Phase 5 golden eval (F2P 16.9% / P2P 91.2%)
    default_variant: str = "higher_lr_14b"

    # Engine knobs (6.1 sweep parameters)
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 16
    max_lora_rank: int = 64  # required: higher_rank_14b adapter is rank 32
    # training used max_seq_length=4096 (config/qlora_variants.yaml); 16384
    # cannot fit ANY concurrent request on A10G with 14B weights
    max_model_len: int = 4096
    # serving keeps CUDA graphs (eval used eager for boot speed)
    enforce_eager: bool = False

    # Sampling defaults (request params override)
    temperature: float = 0.1
    top_p: float = 0.95
    # prevents "```" fence degeneracy
    repetition_penalty: float = 1.15
    default_max_tokens: int = 4096
    # must be <= max_model_len: vLLM rejects max_tokens > max_model_len at request time
    max_tokens_cap: int = 4096

    # Modal / W&B
    gpu_type: str = "a10g-24gb"
    modal_volume: str = "serve-model-cache"
    wandb_entity: str = "2571642-university-of-dundee"
    wandb_project: str = "swe-qwen"
    idle_timeout_seconds: int = 600
    # 64-way broke modal 1.5.3 aiohttp (Phase 5 lesson)
    max_concurrent_requests: int = 16

    # Telemetry
    telemetry_flush_interval_seconds: int = 60
    gpu_util_sample_interval_seconds: int = 5

    def registry_serving_hf_id(self) -> str:
        """Resolve serving_hf_id from config/models.yaml (single source of
        truth); falls back to the field."""
        try:
            registry = (
                yaml.safe_load((_REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
                or {}
            )
            entry = (registry.get("models", {}) or {}).get(self.base_model, {}) or {}
            return str(entry.get("serving_hf_id") or self.serving_hf_id)
        except (OSError, yaml.YAMLError):
            return self.serving_hf_id

    model_config = SettingsConfigDict(
        env_prefix="SERVING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )
