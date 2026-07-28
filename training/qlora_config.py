"""QLoRA configuration factory.

Loads ``models.yaml`` and ``qlora_variants.yaml``, merges variant-level overrides
with model-level defaults, and returns instantiated ``LoraConfig`` /
``TrainingArguments`` tuples ready for ``QLoRATrainer``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig
from transformers import TrainingArguments

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

    Merges with ``baseline`` so variants only override what they specify.
    """
    data = _load_yaml(_VARIANTS_PATH)
    variants: dict = data.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Unknown variant {variant!r}. Available: {list(variants.keys())}")

    # Start with a deep copy of baseline
    baseline = copy.deepcopy(variants.get("baseline", {}))
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


def build_qlora_config(
    variant: str,
    model_name: str = "qwen3-30b-a3b",
    output_dir: str | None = None,
    run_name: str | None = None,
) -> tuple[LoraConfig, TrainingArguments]:
    """Build ``(LoraConfig, TrainingArguments)`` for a given variant + model.

    Args:
        variant: Key from ``qlora_variants.yaml`` (e.g. ``"baseline"``).
        model_name: Key from ``models.yaml`` (e.g. ``"qwen3-30b-a3b"``).
        output_dir: Override output directory (default ``/tmp/qlora-{variant}``).
        run_name: W&B run name (auto-generated if ``None``).

    Returns:
        Tuple of ``(LoraConfig, TrainingArguments)`` ready for ``QLoRATrainer``
        or ``SFTTrainer``.
    """
    model_cfg = _get_model_config(model_name)
    var_cfg = _get_variant_config(variant)

    # ── Resolve target_modules ────────────────────────────────────────────────
    lora_params = dict(var_cfg.get("lora", {}))
    if lora_params.get("target_modules") is None:
        lora_params["target_modules"] = list(model_cfg.get("target_modules", []))

    lora_config = LoraConfig(**lora_params)

    # ── Build TrainingArguments ────────────────────────────────────────────────
    train_params = dict(var_cfg.get("training", {}))
    train_params["output_dir"] = output_dir or f"/tmp/qlora-{variant}"
    if run_name:
        train_params["run_name"] = run_name
    # Remove params that aren't TrainingArguments kwargs
    packing = train_params.pop("packing", True)
    max_seq_length = train_params.pop("max_seq_length", 32768)

    training_args = TrainingArguments(**train_params)

    # Attach packing and max_seq_length for downstream use
    training_args._packing = packing  # type: ignore[attr-defined]
    training_args._max_seq_length = max_seq_length  # type: ignore[attr-defined]

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
