"""Modal training entrypoint for SWE-Qwen QLoRA fine-tuning.

Supersedes ``src/swe_qwen/modal_app.py::train_swe_qwen``.

Usage:
    # Production training (A100-80GB):
    modal run training/modal_train.py::train_qlora --model-name qwen3-14b --variant baseline_14b

    # Local testing with tiny model:
    modal run training/local_cli.py --model-name qwen3-14b --variant baseline_14b

Env vars:
    UNSLOTH_ENABLED=1  # Enable Unsloth acceleration (default: 1 on H100/A100)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import modal
import wandb

# ── Import Unsloth BEFORE trl, transformers, peft ─────────────────────────────
# Unsloth must be imported before these libraries to apply its performance
# patches and avoid a warning + sub-optimal performance.
# The `_ = None` silences lint about an unused import.
try:
    import unsloth  # noqa: F401 — side-effect import (patching)

    _unsloth_patched = True
except ImportError:
    _unsloth_patched = False

from training.qlora_config import resolve_gpu_type

logger = logging.getLogger(__name__)

# ── GCS constants ─────────────────────────────────────────────────────────────
# The bucket is publicly readable (allUsers with objectViewer).
# We download data at runtime via the public JSON API instead of CloudBucketMount,
# because GCS CloudBucketMount requires HMAC keys which are blocked by GCP org
# policy (iam.disableServiceAccountKeyCreation).

_GCS_BUCKET = "swe-qwen-datasets"


def _tokenized_prefix(run_id: str) -> str:
    """Construct GCS prefix for tokenized data given a pipeline run ID."""
    return f"tokenized/{run_id}/"


def _download_gcs_public(prefix: str, dst_dir: str) -> str:
    """Download all objects under ``prefix`` from the public GCS bucket to ``dst_dir``.

    Uses the public JSON API — no auth required because the bucket has
    ``allUsers`` with ``roles/storage.objectViewer``.
    """
    # List objects via public JSON API
    list_url = f"https://www.googleapis.com/storage/v1/b/{_GCS_BUCKET}/o?prefix={prefix}"
    logger.info("Downloading tokenized data from GCS: %s", list_url)

    with urllib.request.urlopen(list_url) as resp:
        payload = json.loads(resp.read().decode())

    items = payload.get("items", [])
    if not items:
        raise RuntimeError(
            f"No objects found at gs://{_GCS_BUCKET}/{prefix} — "
            "check that the bucket and prefix are correct."
        )

    dst_root = Path(dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    for obj in items:
        name: str = obj["name"]  # e.g. "tokenized/18e63eac42bb/train/data-00000-of-00001.arrow"
        media_link: str = obj["mediaLink"]  # public download URL

        # Strip the prefix to get the relative path under dst_dir
        rel_path = name[len(prefix) :].lstrip("/")
        if not rel_path:
            continue
        dst_file = dst_root / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("Downloading %s -> %s", name, dst_file)
        with urllib.request.urlopen(media_link) as src, dst_file.open("wb") as f:
            shutil.copyfileobj(src, f)

    actual_data_dir = str(dst_root)
    logger.info("Downloaded %d files from GCS to %s", len(items), actual_data_dir)
    return actual_data_dir


# ── Repo-relative mounts (via Image.add_local_dir) ──────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _REPO_ROOT / "training"
_CONFIG_DIR = _REPO_ROOT / "config"

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("swe-qwen-training-v2")

# ── Modal image with all training dependencies ───────────────────────────────

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
    )
    # Install torch FIRST — must be before xformers/FA so they build/link against it
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
        "packaging>=24.0",
        "xformers>=0.0.29",  # for Unsloth fallback attention speedup
    )
    # FlashAttention-2 for 3-5x attention speedup (cu126+torch2.11 pre-built wheel)
    # Community wheel — matches torch+cu+py version exactly
    .pip_install(
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/"
        "flash_attn-2.8.3+cu126torch2.11-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
    )
    # Unsloth for 2-5x speedup, 60-74% VRAM reduction (installed in Modal image only)
    # Must be after torch so its deps resolve against the correct CUDA runtime
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    # Critical: reduce fragmentation for large models on limited VRAM (A10G)
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Enable Unsloth by default
    .env({"UNSLOTH_ENABLED": "1"})
    # Force rebuild v8 — add flash-attn 2.8.3 wheel for torch 2.11+cu126+cp311
    .run_commands("echo 'cache-bust-v8'")
    # Copy local source into the image — must be LAST
    .add_local_dir(str(_TRAINING_DIR), remote_path="/root/training", copy=True)
    .add_local_dir(str(_CONFIG_DIR), remote_path="/root/config", copy=True)
)

# ── Modal volumes ─────────────────────────────────────────────────────────────

# Tokenized data is downloaded from the public GCS bucket at function start
# (see _download_gcs_public) — no CloudBucketMount needed.

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
        "/models": models_volume,
    },
    gpu="A100-80GB",
    memory=32000,
    timeout=18000,  # 5 hours max (full 19K run ~3-4h)
    retries=modal.Retries(
        max_retries=1,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    ),
)
def train_qlora(  # noqa: PLR0913, PLR0917
    model_name: str = "qwen3-14b",
    variant: str = "baseline_14b",
    run_id: str = "expanded-repos",
    data_dir: str = "/data/tokenized",
    output_dir: str = "/models/qlora-output",
    run_name: str | None = None,
    resume: str | None = None,
    wandb_project: str = "swe-qwen",
    wandb_entity: str | None = None,
    gpu_type: str | None = None,
    use_unsloth: bool | None = None,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    """Run QLoRA training on Modal.

    Tokenized data is downloaded from the public GCS bucket to
    ``/tmp/data/tokenized`` at function start — no GCS credentials needed
    since the bucket is publicly readable.

    Args:
        model_name: Key from ``models.yaml`` (e.g. ``"qwen3-14b"``).
        variant: Key from ``qlora_variants.yaml`` (e.g. ``"efficient_14b"``).
        run_id: Phase 3 pipeline run ID, Determines which tokenized dataset to use.
        data_dir: Ignored — tokenized data is always downloaded from GCS.
        output_dir: Path within volume for checkpoints.
        run_name: W&B run name (auto-generated if ``None``).
        resume: Checkpoint path or W&B artifact ref for resume.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (optional).
        gpu_type: Modal GPU spec (e.g. ``"A10G:1"``, ``"A100:1"``, ``"H100:1"``).
                  If None, auto-resolved from model config.
        use_unsloth: Enable Unsloth acceleration (default: from UNSLOTH_ENABLED env var).
        max_train_samples: Subsample train dataset to this many rows (for quick debugging).
                           If None, uses all data.

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

    # ── Download tokenized data from public GCS ──────────────────────
    # The bucket is publicly readable so no credentials needed.
    # We use stdlib urllib + shutil to avoid adding google-cloud-storage dep.
    _download_dir = "/tmp/data"
    data_dir = _download_gcs_public(_tokenized_prefix(run_id), _download_dir)

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
        use_flash_attn=True,
        gpu_type=gpu_type,
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
        max_train_samples=max_train_samples,
    )

    metrics = trainer.train()

    # Capture W&B run info before trainer cleanup
    wandb_run_id = wandb.run.id if wandb and wandb.run else None
    artifact_name = f"model-{model_name}-{variant}"

    # Gracefully shut down torch compile workers before Modal kills the container.
    # Prevents harmless-but-ugly "Exception ignored in atexit callback" traceback.
    try:
        import torch._inductor.async_compile  # noqa: F811

        torch._inductor.async_compile.shutdown_compile_workers()
    except Exception:
        pass

    return {
        "status": "completed",
        "wandb_run_id": wandb_run_id,
        "artifact_name": artifact_name,
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
