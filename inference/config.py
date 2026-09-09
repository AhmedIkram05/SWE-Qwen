"""Serving configuration for the Phase 6 inference API.

Mirrors ``evaluation.config.EvalConfig`` in shape (pydantic-settings
``BaseSettings``, env-file + defaults, frozen) but is fully independent: the
serving API uses the ``SERVING_`` env prefix and its own knobs so the eval
harness and the inference server never cross-talk through env vars.

Model-coupled fields resolve registry-first (PHASE-10 Steps 2/3/9-3):

* env (``SERVING_*``) > merged model registry (``registry.loader``) > the
  legacy Phase-6 Qwen literals — used ONLY when the registry files are
  missing/unreadable, never as a silent family fallback: a readable registry
  without ``base_model`` raises.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from registry.loader import default_model_key, load_models

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Legacy Phase-6 literals: last-resort defaults when the registry is missing
# or unreadable. ``_LEGACY_VARIANTS``/``_LEGACY_DEFAULT_VARIANT`` also apply
# while the legacy entry carries no explicit ``variants``/``default_variant``
# keys (any other key without those keys stays empty — Step 2: empty-safe).
_LEGACY_KEY = "qwen3-14b"
_LEGACY_VARIANTS = ("baseline_14b", "higher_rank_14b", "higher_lr_14b")
_LEGACY_DEFAULT_VARIANT = "higher_rank_14b"
_LEGACY_MODEL_LEN = 4096


class ServeConfig(BaseSettings):
    """Serving configuration for the Phase 6 inference API (SERVING_ env prefix)."""

    # Model / adapter registry
    base_model: str = _LEGACY_KEY  # registry key in config/models.yaml
    # 4-bit AWQ base: FP8 weight-only on A10G (Marlin) leaves 16.07 GiB of
    # weights and starves the KV cache (Path-A fallback per 6.1)
    serving_hf_id: str = "Qwen/Qwen3-14B-AWQ"  # "" = plain base, adapters on top
    quantization: str = "awq"  # registry `serving_quantization`: "fp8" | "awq" | ""
    variants: tuple[str, ...] = _LEGACY_VARIANTS
    lora_artifact_pattern: str = "model-qwen3-14b-{variant}"
    # champion per final results (assets/results.txt + promotion/MODEL_CARD.md):
    # F2P 17.20% / P2P 90.10% on the 100-instance golden set
    default_variant: str = _LEGACY_DEFAULT_VARIANT

    # Engine knobs (6.1 sweep parameters)
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 16
    max_lora_rank: int = 64  # required: higher_rank_14b adapter is rank 32
    # registry `context_window` (Step 9-3, env-overridable); 16384 cannot fit
    # ANY concurrent request on A10G with 14B weights — deployments pin via
    # SERVING_MAX_MODEL_LEN
    max_model_len: int = _LEGACY_MODEL_LEN
    # serving keeps CUDA graphs (eval used eager for boot speed)
    enforce_eager: bool = False

    # Sampling defaults (request params override)
    temperature: float = 0.1
    top_p: float = 0.95
    # prevents "```" fence degeneracy
    repetition_penalty: float = 1.15
    default_max_tokens: int = _LEGACY_MODEL_LEN  # registry context_window
    # must be <= max_model_len: vLLM rejects max_tokens > max_model_len at request time
    max_tokens_cap: int = _LEGACY_MODEL_LEN  # registry context_window

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
    # Phase 8 decision 8: flush-loop alert thresholds (wandb.alert; fires only
    # while a run is active).
    alert_error_rate_threshold: float = 0.10
    alert_ttfb_p95_threshold_ms: float = 2000.0
    # Phase 8 decision 1: fraction of successful requests traced to Langfuse
    # (0.1 = 10%; eval traces are never sampled).
    telemetry_trace_sample_rate: float = 0.1

    @model_validator(mode="after")
    def _resolve_registry(self) -> ServeConfig:
        """Fill model-coupled fields from the merged registry.

        Env/init values win (already in ``model_fields_set``). Literals apply
        only when the registry files are missing or unreadable; a readable
        registry whose ``base_model`` is unknown raises instead of silently
        falling back to Qwen (Step 9-2). Also enforces the Step 3 invariant:
        ``serving_quantization`` without a ``serving_hf_id`` fails at config
        load, not at vLLM boot.
        """
        try:
            models = load_models()
        except (OSError, yaml.YAMLError):
            return self
        if "base_model" not in self.model_fields_set:
            # no default flagged → keep the legacy literal key
            with contextlib.suppress(KeyError):
                object.__setattr__(self, "base_model", default_model_key())
        spec = models.get(self.base_model)
        if spec is None:
            raise ValueError(
                f"unknown model {self.base_model!r} in the registry; available: {sorted(models)}"
            )

        def set_if_unset(name: str, value: Any) -> None:
            # ponytail: frozen config; pydantic bans plain assignment after
            # validation, object.__setattr__ bypasses the freeze check once.
            if name not in self.model_fields_set:
                object.__setattr__(self, name, value)

        variants = (spec.model_extra or {}).get("variants")
        if variants is None:
            variants = _LEGACY_VARIANTS if self.base_model == _LEGACY_KEY else ()
        elif not isinstance(variants, (list, tuple)):
            raise ValueError(
                f"registry entry {self.base_model!r}: 'variants' must be a list, "
                f"got {type(variants).__name__}"
            )
        set_if_unset("variants", tuple(variants))

        default_variant = (spec.model_extra or {}).get("default_variant")
        if default_variant is None:
            default_variant = _LEGACY_DEFAULT_VARIANT if self.base_model == _LEGACY_KEY else ""
        elif not isinstance(default_variant, str):
            raise ValueError(
                f"registry entry {self.base_model!r}: 'default_variant' must be a string"
            )
        set_if_unset("default_variant", default_variant)

        set_if_unset("serving_hf_id", spec.serving_hf_id or "")
        set_if_unset("quantization", spec.serving_quantization or "")
        set_if_unset("lora_artifact_pattern", f"model-{self.base_model}-{{variant}}")
        set_if_unset("max_model_len", spec.context_window)
        set_if_unset("default_max_tokens", spec.context_window)
        set_if_unset("max_tokens_cap", spec.context_window)

        if not self.serving_hf_id and self.quantization:
            raise ValueError(
                f"model {self.base_model!r}: serving_quantization {self.quantization!r} "
                "set but no serving_hf_id — a quantization backend needs a "
                "pre-quantized base; set serving_hf_id (or SERVING_SERVING_HF_ID) "
                "or clear serving_quantization"
            )
        return self

    def registry_serving_hf_id(self) -> str:
        """Resolve the serving engine's model id from the merged registry.

        The pre-quantized base when the entry declares one, else the plain
        ``hf_id`` (fresh models serve LoRA adapters on the unquantized base).
        Registry missing/unreadable → the (registry-resolved) field value.
        """
        try:
            spec = load_models().get(self.base_model)
        except (OSError, yaml.YAMLError, ValidationError):
            return self.serving_hf_id
        if spec is None:
            return self.serving_hf_id
        return spec.serving_hf_id or spec.hf_id

    model_config = SettingsConfigDict(
        env_prefix="SERVING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )
