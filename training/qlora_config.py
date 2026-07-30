"""QLoRA configuration factory.

Loads ``models.yaml`` and ``qlora_variants.yaml``, merges variant-level overrides
with model-level defaults, and returns instantiated ``LoraConfig`` /
``TrainingArguments`` tuples ready for ``QLoRATrainer``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig
from trl import SFTConfig

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"
_MODELS_PATH = _CONFIG_DIR / "models.yaml"
_VARIANTS_PATH = _CONFIG_DIR / "qlora_variants.yaml"

# ── Model → Modal GPU mapping ────────────────────────────────────────────────

GPU_MAP: dict[str, str] = {
    "a100-40gb": "A100:1",
    "a10g-24gb": "A10G:1",
    "a100-80gb": "A100:1,size=80GB",
    "h100-80gb": "H100:1",
}


# ── Load helpers ──────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _get_model_config(model_name: str) -> dict[str, Any]:
    """Load ``models.yaml`` and return the entry for *model_name*."""
    data = _load_yaml(_MODELS_PATH)
    models: dict = data.get("models", {})
    if model_name not in models:
        raise KeyError(f"Unknown model {model_name!r}. Available: {list(models.keys())}")
    return dict(models[model_name])


def _get_variant_config(variant: str) -> dict[str, Any]:
    """Load ``qlora_variants.yaml`` and return the entry for *variant*.

    Merges with ``baseline_14b`` so variants only override what they specify.
    """
    data = _load_yaml(_VARIANTS_PATH)
    variants: dict = data.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Unknown variant {variant!r}. Available: {list(variants.keys())}")

    # Start with a deep copy of baseline_14b
    baseline = copy.deepcopy(variants.get("baseline_14b", {}))
    overrides = copy.deepcopy(variants[variant])

    # Recursively merge
    def _deep_merge(base: dict, overrides: dict) -> dict:
        result = copy.deepcopy(base)
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    return _deep_merge(baseline, overrides)


# ── Factory ───────────────────────────────────────────────────────────────────


# GPU-specific memory settings (from Modal debugging)
# ── GPU Memory Overrides ──────────────────────────────────────────────────────
# These override variant config values to fit within GPU VRAM limits.
# GIGAGPU benchmarking (RTX 4090 24GB, same VRAM as A10G):
#   Qwen 14B QLoRA, seq=2048, batch=8, rank=32 → 17.4 GB peak
# A10G uses same 24GB VRAM but ~60% of 4090 memory bandwidth (~600 vs 1008 GB/s).
GPU_MEMORY_OVERRIDES: dict[str, dict[str, Any]] = {
    "A10G:1": {
        "packing": True,  # Sequence packing for GPU efficiency
        "max_seq_length": 2048,  # A10G 24GB max safe context
        "per_device_train_batch_size": 6,  # Batch fits: 17.4 GB < 24 GB ceiling
        "gradient_accumulation_steps": 3,  # Effective batch = 6 × 3 = 18
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
    },
    "A100:1": {
        "packing": True,
        "max_seq_length": 8192,
        "per_device_train_batch_size": 8,  # A100 has 40/80 GB, can fit larger batch
        "gradient_accumulation_steps": 2,  # Effective batch = 8 × 2 = 16
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
    },
    "H100:1": {
        "packing": True,
        "max_seq_length": 32768,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
    },
}


def build_qlora_config(
    variant: str,
    model_name: str = "qwen3-14b",
    output_dir: str | None = None,
    run_name: str | None = None,
    gpu_type: str | None = None,
) -> tuple[LoraConfig, SFTConfig]:
    """Build ``(LoraConfig, SFTConfig)`` for a given variant + model.

    Args:
        variant: Key from ``qlora_variants.yaml`` (e.g. ``"baseline"``).
        model_name: Key from ``models.yaml`` (e.g. ``"qwen3-14b"``).
        output_dir: Override output directory (default ``/tmp/qlora-{variant}``).
        run_name: W&B run name (auto-generated if ``None``).
        gpu_type: Modal GPU spec (e.g. ``"A10G:1"``, ``"A100:1"``, ``"H100:1"``).
                  If provided, applies GPU-specific memory overrides.

    Returns:
        Tuple of ``(LoraConfig, SFTConfig)`` ready for ``QLoRATrainer``
        or ``SFTTrainer``.
    """
    model_cfg = _get_model_config(model_name)
    var_cfg = _get_variant_config(variant)

    # ── Resolve target_modules ────────────────────────────────────────────────
    lora_params = dict(var_cfg.get("lora", {}))
    if lora_params.get("target_modules") is None:
        lora_params["target_modules"] = list(model_cfg.get("target_modules", []))

    lora_config = LoraConfig(**lora_params)

    # ── Build SFTConfig ────────────────────────────────────────────────────────
    train_params = dict(var_cfg.get("training", {}))
    train_params["output_dir"] = output_dir or f"/tmp/qlora-{variant}"
    if run_name:
        train_params["run_name"] = run_name

    # Apply GPU-specific memory overrides if provided
    if gpu_type and gpu_type in GPU_MEMORY_OVERRIDES:
        gpu_overrides = GPU_MEMORY_OVERRIDES[gpu_type]
        train_params.update(gpu_overrides)

    # Remove params that aren't SFTConfig kwargs
    packing = train_params.pop("packing", True)
    max_seq_length = train_params.pop("max_seq_length", 32768)

    # Pop bf16/fp16 — transformers v5 validates these at init time against
    # CUDA availability, which may not be ready at config-build time on Modal.
    # Restored directly on the SFTConfig instance after init.
    _bf16 = train_params.pop("bf16", False)
    _fp16 = train_params.pop("fp16", False)

    training_args = SFTConfig(
        **train_params,
        bf16=False,
        fp16=False,
        packing=packing,
        max_length=max_seq_length,
    )

    # Restore precision flags
    training_args.bf16 = _bf16
    training_args.fp16 = _fp16

    return lora_config, training_args


def resolve_gpu_type(model_name: str, prefer_fallback: bool = False) -> str:
    """Resolve a Modal GPU spec string for the given model.

    Args:
        model_name: Key from ``models.yaml``.
        prefer_fallback: If ``True``, use the fallback GPU (e.g. A10G for 30B).

    Returns:
        Modal GPU spec string e.g. ``"A100:1"``.
    """
    model_cfg = _get_model_config(model_name)
    mapping: dict[str, str] = model_cfg.get("gpu_mapping", {})
    key = "fallback" if prefer_fallback else "primary"
    gpu_label = mapping.get(key, mapping.get("primary", "A10G:1")) or "A10G:1"
    return GPU_MAP.get(gpu_label, gpu_label)


def list_models() -> list[str]:
    """Return all registered model names."""
    data = _load_yaml(_MODELS_PATH)
    return list(data.get("models", {}).keys())


def list_variants() -> list[str]:
    """Return all registered variant names."""
    data = _load_yaml(_VARIANTS_PATH)
    return list(data.get("variants", {}).keys())


def build_model_and_peft(
    variant: str,
    model_name: str = "qwen3-14b",
    max_seq_length: int = 8192,
    use_flash_attn: bool = True,
    gpu_type: str | None = None,
) -> tuple:
    """Build (model, tokenizer) using Unsloth or fallback.

    Delegates to `training.unsloth_factory.build_model_and_peft`.
    Uses model config from models.yaml and variant config from qlora_variants.yaml.

    Args:
        variant: Key from qlora_variants.yaml
        model_name: Key from models.yaml
        max_seq_length: Max sequence length (overrides variant config)
        use_flash_attn: Whether to use flash attention
        gpu_type: Modal GPU spec (e.g. ``"A10G:1"``). When provided, applies
                  GPU-specific memory overrides to max_seq_length.

    Returns:
        Tuple of (model, tokenizer) ready for SFTTrainer
    """
    from training.unsloth_factory import build_model_and_peft as _build

    model_cfg = _get_model_config(model_name)
    var_cfg = _get_variant_config(variant)

    # Override max_seq_length from variant if provided
    if "max_seq_length" in var_cfg.get("training", {}):
        max_seq_length = var_cfg["training"]["max_seq_length"]

    # Apply GPU-specific overrides (same as build_qlora_config does for SFTConfig)
    if gpu_type and gpu_type in GPU_MEMORY_OVERRIDES:
        gpu_max_seq = GPU_MEMORY_OVERRIDES[gpu_type].get("max_seq_length")
        if gpu_max_seq is not None:
            max_seq_length = gpu_max_seq
            logger.info(
                "Overriding max_seq_length to %d for GPU %s",
                max_seq_length,
                gpu_type,
            )

    return _build(
        model_cfg=model_cfg,
        variant_cfg=var_cfg,
        max_seq_length=max_seq_length,
        use_flash_attn=use_flash_attn,
    )
