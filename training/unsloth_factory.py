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
            # Clean up GPU memory from the failed Unsloth load so the fallback
            # doesn't pile two models on the same GPU (guaranteed OOM on A10G 24GB).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            return _build_fallback(model_cfg, variant_cfg)
    else:
        logger.info("Unsloth disabled; using standard TRL + PEFT + bitsandbytes")
        return _build_fallback(model_cfg, variant_cfg)


def _fix_eos_token(tokenizer: Any) -> None:
    """Fix tokenizer eos_token if it's not in the vocabulary.

    Unsloth's FastLanguageModel wrapper may set eos_token to a placeholder
    (e.g. '<EOS_TOKEN>') that doesn't exist in the actual tokenizer vocab.
    SFTTrainer validates eos_token is in vocab at init time, so we need
    to correct it here.

    Falls back to a known-good token for the architecture if needed.
    """
    if tokenizer.eos_token is None:
        return

    # Fast path: already valid
    vocab = tokenizer.get_vocab()
    if tokenizer.eos_token in vocab:
        return

    logger.warning(
        "eos_token %r not in vocabulary. Attempting to fix...",
        tokenizer.eos_token,
    )

    # Try common candidates in priority order
    # Qwen3 uses <|endoftext|> (token ID 151643)
    for candidate in ("<|endoftext|>", "<|im_end|>", "</s>", "<EOS>"):
        if candidate in vocab:
            tokenizer.eos_token = candidate
            tokenizer.eos_token_id = vocab[candidate]
            logger.info("Fixed eos_token: %s -> %s (id=%d)", candidate, candidate, vocab[candidate])
            return

    # Last resort: set eos_token_id from the model config (it's usually correct)
    # and update the string representation
    if tokenizer.eos_token_id is not None:
        # Try to decode the token ID to find its string representation
        try:
            decoded = tokenizer.decode([tokenizer.eos_token_id])
            decoded = decoded.strip()
            if decoded:
                tokenizer.eos_token = decoded
                logger.info("Restored eos_token from id=%d: %s", tokenizer.eos_token_id, decoded)
                return
        except Exception:
            pass

    logger.error(
        "Could not fix eos_token %r — not in vocabulary and no candidate found. "
        "SFTTrainer will likely fail.",
        tokenizer.eos_token,
    )


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

    # A10G has 24GB VRAM but only ~22GB usable. Reserve 4GB for training + activations.
    # Use explicit device_map to force GPU memory limit - Unsloth may not pass max_memory through.
    max_memory = {0: "18GiB", "cpu": "32GiB"}
    device_map = "auto"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_id,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        max_memory=max_memory,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Fix eos_token: Unsloth wrapper may set a placeholder (<EOS_TOKEN>)
    # that doesn't exist in the actual vocabulary. Qwen3 uses <|endoftext|>.
    # SFTTrainer validates eos_token is in vocab before training.
    _fix_eos_token(tokenizer)

    # Ensure tokenizer has proper pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Extract LoRA params from variant config
    lora_params = variant_cfg.get("lora", {})
    r = lora_params.get("r", 16)
    lora_alpha = lora_params.get("lora_alpha", 32)
    lora_dropout = lora_params.get("lora_dropout", 0.05)
    target_modules = lora_params.get("target_modules")
    bias = lora_params.get("bias", "none")
    # If target_modules not specified, use model config
    if target_modules is None:
        target_modules = model_cfg.get("target_modules", [])

    logger.info(f"Applying Unsloth LoRA: r={r}, alpha={lora_alpha}, targets={target_modules}")

    # Remove keys that we pass explicitly or that Unsloth handles internally.
    # task_type is routed by FastLanguageModel.get_peft_model to LoraConfig internally;
    # keeping it in peft_kwargs causes "multiple values for keyword argument 'task_type'".
    peft_kwargs = dict(lora_params)
    for key in ["r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type"]:
        peft_kwargs.pop(key, None)

    # Pass everything via peft_kwargs - DO NOT include task_type or padding_free
    # Unsloth wrapper passes to LoraConfig which doesn't support these
    peft_kwargs.update(
        {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": bias,
            "use_gradient_checkpointing": "unsloth",  # saves 30% VRAM
        }
    )

    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    # Attach tokenizer for downstream use
    model.tokenizer = tokenizer

    return model, tokenizer


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

    # A10G has 24GB VRAM but only ~22GB usable. Reserve 4GB for training.
    max_memory = {0: "18GiB", "cpu": "32GiB"}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload for A10G 24GB
    )

    logger.info(
        "Loading %s with standard bitsandbytes (4bit=%s, cpu_offload=True, max_memory=20GiB)",
        hf_id,
        load_in_4bit,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native SDPA
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply PEFT LoRA
    lora_params = variant_cfg.get("lora", {})
    lora_config = LoraConfig(**lora_params)
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]

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
