"""Unsloth integration factory with mechanical fallback.

Provides `build_model_and_peft()` that tries Unsloth first, falls back to
standard TRL + PEFT + bitsandbytes on any exception. Controlled by
`UNSLOTH_ENABLED` env var (default: 1 for H100/A100, 0 to force fallback).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Env var control
# ──────────────────────────────────────────────────────────────────────────────

_UNSLOTH_ENABLED = os.environ.get("UNSLOTH_ENABLED", "1") == "1"
_UNSLOTH_AVAILABLE = importlib.util.find_spec("unsloth") is not None
if not _UNSLOTH_AVAILABLE:
    logger.warning("Unsloth not installed; will use standard TRL + PEFT + bitsandbytes")


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

ModelAndPEFT = tuple[Any, Any]  # (model, peft_model)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def build_model_and_peft(
    model_cfg: dict[str, Any],
    variant_cfg: dict[str, Any],
    max_seq_length: int = 8192,
    use_flash_attn: bool = True,
) -> ModelAndPEFT:
    """Build (model, peft_model) using Unsloth or fallback.

    Args:
        model_cfg: Model config from models.yaml (hf_id, quantization, etc.)
        variant_cfg: Variant config from qlora_variants.yaml (lora, training)
        max_seq_length: Max sequence length for Unsloth
        use_flash_attn: Whether to use flash attention

    Returns:
        Tuple of (base_model, peft_model) ready for SFTTrainer
    """
    if _UNSLOTH_ENABLED and _UNSLOTH_AVAILABLE:
        try:
            return _build_with_unsloth(model_cfg, variant_cfg, max_seq_length, use_flash_attn)
        except Exception as e:
            logger.warning(f"Unsloth failed ({e}); falling back to standard TRL+PEFT+BnB")
            return _build_fallback(model_cfg, variant_cfg)
    else:
        logger.info("Unsloth disabled; using standard TRL + PEFT + bitsandbytes")
        return _build_fallback(model_cfg, variant_cfg)


def _build_with_unsloth(
    model_cfg: dict[str, Any],
    variant_cfg: dict[str, Any],
    max_seq_length: int,
    use_flash_attn: bool,
) -> ModelAndPEFT:
    """Build model + PEFT using Unsloth FastLanguageModel."""
    from unsloth import FastLanguageModel

    hf_id = model_cfg["hf_id"]
    compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16

    # Determine quantization from model config
    load_in_4bit = model_cfg.get("quantization", "nf4") in ("nf4", "fp4", "4bit")
    load_in_8bit = model_cfg.get("quantization") == "8bit"

    # Attention implementation
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"

    logger.info(
        "Loading %s with Unsloth (4bit=%s, dtype=%s, attn=%s)",
        hf_id,
        load_in_4bit,
        compute_dtype,
        attn_impl,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_id,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Extract LoRA params from variant config
    lora_params = variant_cfg.get("lora", {})
    r = lora_params.get("r", 16)
    lora_alpha = lora_params.get("lora_alpha", 32)
    lora_dropout = lora_params.get("lora_dropout", 0.05)
    target_modules = lora_params.get("target_modules")
    bias = lora_params.get("bias", "none")
    task_type = lora_params.get("task_type", "CAUSAL_LM")

    # If target_modules not specified, use model config
    if target_modules is None:
        target_modules = model_cfg.get("target_modules", [])

    logger.info(f"Applying Unsloth LoRA: r={r}, alpha={lora_alpha}, targets={target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        use_gradient_checkpointing="unsloth",  # saves 30% VRAM
        padding_free=False,  # avoids collision with assistant_only_loss
    )

    # Attach tokenizer for downstream use
    model.tokenizer = tokenizer

    return model, model


def _build_fallback(
    model_cfg: dict[str, Any],
    variant_cfg: dict[str, Any],
) -> ModelAndPEFT:
    """Build model + PEFT using standard TRL + PEFT + bitsandbytes."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    hf_id = model_cfg["hf_id"]
    compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16

    # Quantization config (4-bit FP4 — NF4 causes CUDA illegal memory access
    # on A10G with Qwen3-14B, a known bitsandbytes issue with MoE-like
    # architectures. FP4 trades ~1% perplexity for full stability.)
    # FORCE FP4 in fallback — ignore model config's nf4 setting
    load_in_4bit = True
    quant_type = "fp4"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload for A10G 24GB
    )

    logger.info(
        "Loading %s with standard bitsandbytes (4bit=%s, cpu_offload=True)",
        hf_id,
        load_in_4bit,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native SDPA
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply PEFT LoRA
    lora_params = variant_cfg.get("lora", {})
    lora_config = LoraConfig(**lora_params)
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Fallback mapping (for documentation/debugging)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_MAP = {
    "FastLanguageModel.from_pretrained": "AutoModelForCausalLM.from_pretrained + prepare_model_for_kbit_training",  # noqa: E501
    "FastLanguageModel.get_peft_model": "get_peft_model(model, LoraConfig(...))",
    "FastLanguageModel.for_inference": "model.eval() + model.merge_and_unload() if needed",
}
