"""Modal training entrypoint for SWE-Qwen QLoRA fine-tuning.

Supersedes ``src/swe_qwen/modal_app.py::train_swe_qwen``.

Usage:
    modal run training/modal_train.py --model-name qwen3-14b --variant baseline_14b

Env vars:
    UNSLOTH_ENABLED=1  # Enable Unsloth acceleration (default: 1 on H100/A100)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import modal

from training.qlora_config import resolve_gpu_type

logger = logging.getLogger(__name__)

# ── Repo-relative mounts (via Image.add_local_dir) ──────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _REPO_ROOT / "training"
_CONFIG_DIR = _REPO_ROOT / "config"

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("swe-qwen-training")

# ── Modal image with all training dependencies ───────────────────────────────

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
    )
    .pip_install("packaging>=24.0")  # required by flash-attn setup.py
    # Install flash-attn from a pre-built wheel (avoids nvcc requirement at build time)
    # Uses torch 2.6+cu12 wheel — ABI-compatible with our torch 2.11+cu126 at runtime
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
        "flash_attn-2.8.3.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        "torch==2.11.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "transformers>=5.5.0,<5.6.0",
        "accelerate>=1.14.0",
        "peft>=0.19.0",
        "bitsandbytes>=0.49.0",
        "trl>=1.9.0",
        "datasets>=5.0.0",
        "wandb>=0.28.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.28.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    )
    # Unsloth for 2-5x speedup, 60-74% VRAM reduction (installed in Modal image only)
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    # Critical: reduce fragmentation for large models on limited VRAM (A10G)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Enable Unsloth by default on H100/A100 ( Modal GPU types)
    .env({"UNSLOTH_ENABLED": "1"})
    # Copy local source into the image — must be LAST
    .add_local_dir(str(_TRAINING_DIR), remote_path="/root/training", copy=True)
    .add_local_dir(str(_CONFIG_DIR), remote_path="/root/config", copy=True)
)

# ── Modal volumes ─────────────────────────────────────────────────────────────

# Persistent volume for training data (tokenized .arrow shards)
data_volume = modal.Volume.from_name("swe-qwen-datasets", create_if_missing=True)

# Persistent volume for model checkpoints
models_volume = modal.Volume.from_name("swe-qwen-models", create_if_missing=True)

# ── Training function ─────────────────────────────────────────────────────────


@app.function(
    image=training_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={
        "/data": data_volume,
        "/models": models_volume,
    },
    gpu="A10G:1",  # 14B model fits on A10G 24GB
    memory=32000,  # 32 GB for 14B on A10G
    timeout=7200,  # 2 hours max
    retries=modal.Retries(
        max_retries=1,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    ),
)
def train_qlora(  # noqa: PLR0913, PLR0917
    model_name: str = "qwen3-14b",
    variant: str = "baseline_14b",
    data_dir: str = "/data/tokenized",
    output_dir: str = "/models/qlora-output",
    run_name: str | None = None,
    resume: str | None = None,
    wandb_project: str = "swe-qwen",
    wandb_entity: str | None = None,
    gpu_type: str | None = None,
    use_unsloth: bool | None = None,
):
    """Run QLoRA training on Modal.

    Args:
        model_name: Key from ``models.yaml`` (e.g. ``"qwen3-14b"``).
        variant: Key from ``qlora_variants.yaml`` (e.g. ``"baseline_14b"``).
        data_dir: Path within volume to tokenized data.
        output_dir: Path within volume for checkpoints.
        run_name: W&B run name (auto-generated if ``None``).
        resume: Checkpoint path or W&B artifact ref for resume.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (optional).
        gpu_type: Modal GPU spec (e.g. ``"A10G:1"``, ``"A100:1"``, ``"H100:1"``).
                  If None, auto-resolved from model config.
        use_unsloth: Enable Unsloth acceleration (default: from UNSLOTH_ENABLED env var).

    Returns:
        Dict with training results.
    """
    # Validate W&B API key is available (injected via Modal secret)
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY environment variable not set. "
            "Ensure 'wandb-secret' Modal secret is mounted with WANDB_API_KEY."
        )

    from training.qlora_config import build_model_and_peft, resolve_gpu_type
    from training.qlora_trainer import QLoRATrainer

    # Auto-resolve GPU type if not provided
    if gpu_type is None:
        gpu_type = resolve_gpu_type(model_name)

    # Build model + tokenizer (uses Unsloth if UNSLOTH_ENABLED=1)
    unsloth_enabled = (
        use_unsloth if use_unsloth is not None else os.environ.get("UNSLOTH_ENABLED", "1") == "1"
    )

    if unsloth_enabled:
        logger.info("Building model with Unsloth acceleration...")
    else:
        logger.info("Building model with standard TRL + PEFT + bitsandbytes (UNSLOTH_ENABLED=0)")

    model, tokenizer = build_model_and_peft(
        variant=variant,
        model_name=model_name,
        max_seq_length=8192,  # will be overridden by variant config
        use_flash_attn=True,
    )

    trainer = QLoRATrainer(
        model_name=model_name,
        variant=variant,
        data_dir=data_dir,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        resume_from_checkpoint=resume,
        gpu_type=gpu_type,
        model=model,
        tokenizer=tokenizer,
    )

    metrics = trainer.train()

    return {
        "status": "completed",
        "model_name": model_name,
        "variant": variant,
        "output_dir": output_dir,
        "metrics": metrics,
    }


# ── GPU resolution helper ────────────────────────────────────────────────────


@app.function(
    image=training_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=60,
)
def get_gpu_for_model(model_name: str = "qwen3-14b") -> str:
    """Return the Modal GPU spec for a given model.

    Used by orchestration scripts to determine GPU allocation before launching.
    """
    return resolve_gpu_type(model_name)


# ── Alias for orchestration scripts ───────────────────────────────────────────

# Export train_qlora as modal_entrypoint for compatibility with orchestration
modal_entrypoint = train_qlora

# ── CLI entrypoint ────────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(  # noqa: PLR0913, PLR0917
    model_name: str = "qwen3-14b",
    variant: str = "baseline_14b",
    data_dir: str = "/data/tokenized",
    output_dir: str = "/models/qlora-output",
    run_name: str | None = None,
    resume: str | None = None,
    wandb_project: str = "swe-qwen",
    wandb_entity: str | None = None,
):
    """Launch QLoRA training locally (for testing with small models)."""
    print(f"Starting training: model={model_name}, variant={variant}")
    print(f"Data: {data_dir}, Output: {output_dir}")

    # For local testing, use tiny model
    local_model = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    from training.qlora_trainer import QLoRATrainer

    trainer = QLoRATrainer(
        model_name=model_name,
        variant=variant,
        hf_id=local_model,  # override hf_id for local testing
        data_dir=data_dir,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        resume_from_checkpoint=resume,
        use_flash_attn=False,  # no flash-attn on CPU
    )

    result = trainer.train()
    print(f"Training result: {result}")
