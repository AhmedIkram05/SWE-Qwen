"""QLoRA fine-tuning trainer class.

``QLoRATrainer`` encapsulates model loading (4-bit NF4), PEFT wrapping,
data loading, callback setup, training loop, checkpoint saving, and
experiment resumption.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from training.callbacks import WandbCheckpointCallback, WandbLoggingCallback
from training.prompt_loader import PromptLoader
from training.qlora_config import _get_model_config, build_qlora_config

logger = logging.getLogger(__name__)


class QLoRATrainer:
    """End-to-end QLoRA trainer for SWE-Qwen.

    Args:
        model_name: Key from ``models.yaml`` (e.g. ``"qwen3-14b"``).
        variant: Key from ``qlora_variants.yaml`` (e.g. ``"baseline"``).
        data_dir: Path to tokenized ``.arrow`` shards (from ``tokenize.py``).
        output_dir: Local directory for checkpoints (Modal volume mount point).
        wandb_project: W&B project name.
        wandb_entity: W&B entity (optional).
        run_name: Optional W&B run name (auto-generated if ``None``).
        resume_from_checkpoint: Path or W&B artifact ref for resume.
        use_flash_attn: Enable Flash Attention 2.
        prompt_template_dir: Override prompt template directory.
        gpu_type: GPU identifier for training (e.g. ``"A10G:1"``).
        hf_id: Direct HuggingFace model ID override. When set, skips models.yaml
               lookup and uses this as the model source. Useful for local testing
               with tiny models.
        model: Pre-built model (e.g., from Unsloth factory). If provided,
            skips internal model loading.
        tokenizer: Pre-built tokenizer. If provided, skips internal tokenizer loading.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        model_name: str = "qwen3-14b",
        variant: str = "baseline_14b",
        data_dir: str = "data/tokenized",
        output_dir: str = "/tmp/qlora-output",
        wandb_project: str = "swe-qwen",
        wandb_entity: str | None = None,
        run_name: str | None = None,
        resume_from_checkpoint: str | None = None,
        use_flash_attn: bool = True,
        prompt_template_dir: str | None = None,
        gpu_type: str | None = None,
        hf_id: str | None = None,
        model: Any | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        max_train_samples: int | None = None,
    ):
        self.model_name = model_name
        self.variant = variant
        self.hf_id = hf_id
        self.data_dir = Path(data_dir)
        self.max_train_samples = max_train_samples
        self.output_dir = Path(output_dir)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.run_name = run_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.use_flash_attn = use_flash_attn
        self.prompt_template_dir = prompt_template_dir
        self.gpu_type = gpu_type

        # Pre-built model/tokenizer (e.g., from Unsloth factory)
        self._prebuilt_model = model
        self._prebuilt_tokenizer = tokenizer

        # Set early — resolved below
        self.model_cfg: dict[str, Any] = {}
        self.lora_config: LoraConfig | None = None
        self.training_args: TrainingArguments | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.train_dataset: Any = None
        self.eval_dataset: Any = None
        self.trainer: SFTTrainer | None = None
        self.prompt_loader: PromptLoader | None = None

    # ── Setup methods ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Run the full setup: config → model → data → callbacks."""
        self._setup_config()
        self._setup_wandb()
        self._setup_model_and_tokenizer()
        self._setup_data()
        self._setup_callbacks()

    def _setup_config(self) -> None:
        """Load model config and build QLoRA config."""
        # Skip if config already set externally (e.g., by modal_train.py with GPU overrides)
        if self.lora_config is not None and self.training_args is not None:
            logger.info("Config already initialized externally, skipping rebuild")
            # Still need model_cfg for other setup methods
            self.model_cfg = _get_model_config(self.model_name)
            return

        # Build model_cfg — use hf_id override if provided (local testing with tiny models)
        if self.hf_id is not None:
            self.model_cfg = {"hf_id": self.hf_id}
        else:
            self.model_cfg = _get_model_config(self.model_name)

        self.lora_config, self.training_args = build_qlora_config(
            variant=self.variant,
            model_name=self.model_name,
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            gpu_type=self.gpu_type,
        )
        logger.info(
            "Config loaded: model=%s, variant=%s, lora_r=%d",
            self.model_name,
            self.variant,
            self.lora_config.r,
        )

    def _setup_wandb(self) -> None:
        """Initialize W&B run."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise RuntimeError(
                "WANDB_API_KEY environment variable not set. "
                "Ensure 'wandb-secret' Modal secret is mounted with WANDB_API_KEY."
            )
        wandb.login(key=api_key)
        config = {
            "model_name": self.model_name,
            "model_hf_id": self.model_cfg.get("hf_id"),
            "variant": self.variant,
            "lora_r": self.lora_config.r if self.lora_config else None,
            "lora_alpha": self.lora_config.lora_alpha if self.lora_config else None,
            "learning_rate": self.training_args.learning_rate if self.training_args else None,
            "num_epochs": self.training_args.num_train_epochs if self.training_args else None,
            "gradient_accumulation_steps": (
                self.training_args.gradient_accumulation_steps if self.training_args else None
            ),
        }
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.run_name,
            config=config,
        )

        # Log prompt templates as artifact
        self.prompt_loader = PromptLoader(
            template_dir=self.prompt_template_dir,
        )
        if wandb.run is not None:
            self.prompt_loader.log_to_wandb_artifact(version="1.0")

        logger.info("W&B run initialized: %s", wandb.run.name if wandb.run else "N/A")

    def _setup_model_and_tokenizer(self) -> None:
        """Load model, apply PEFT LoRA. Uses 4-bit quantization on CUDA, full precision on CPU.

        If pre-built model/tokenizer provided (e.g., from Unsloth factory), uses those instead.
        """
        # Use pre-built model/tokenizer if provided (Unsloth path)
        if self._prebuilt_model is not None and self._prebuilt_tokenizer is not None:
            self.model = self._prebuilt_model
            self.tokenizer = self._prebuilt_tokenizer
            logger.info("Using pre-built model and tokenizer (Unsloth)")
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.model.print_trainable_parameters()
            logger.info("Model and tokenizer loaded successfully")
            return

        # Standard path (bitsandbytes + PEFT)
        model_cfg = self.model_cfg
        hf_id: str = model_cfg["hf_id"]
        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        torch_dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16

        # Detect CUDA availability — quantization only works on NVIDIA GPUs
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            # Quantization config (4-bit FP4 — NF4 causes CUDA illegal memory access
            # on A10G with Qwen3-14B, a known bitsandbytes issue with MoE-like
            # architectures. FP4 trades ~1% perplexity for full stability.)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=False,
            )
            device_map = "auto"
            logger.info("Loading model %s with 4-bit FP4 (CUDA)...", hf_id)
        else:
            # CPU/MPS: no quantization, full precision
            bnb_config = None
            device_map = None
            logger.info("Loading model %s in full precision (CPU/MPS)...", hf_id)

        # SDPA — PyTorch native. flash-attn 2.8.3 wheel for torch 2.11+cu126+cp311
        # is installed in Modal image (see modal_train.py). SDPA auto-dispatches to
        # flash-attn CUDA kernels when they're installed.
        attn_impl = "sdpa" if self.use_flash_attn else "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply PEFT LoRA
        if self.lora_config is not None:
            self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        logger.info("Model and tokenizer loaded successfully")

    def _setup_data(self) -> None:
        """Load tokenized ``DatasetDict`` from disk."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Tokenized data directory not found: {self.data_dir}. Run tokenize.py first."
            )

        dataset = load_from_disk(str(self.data_dir))
        dataset_dict: DatasetDict = dataset  # type: ignore[assignment]

        self.train_dataset = dataset_dict.get("train")
        self.eval_dataset = dataset_dict.get("val") or dataset_dict.get("test")

        # Subsample for quick debugging
        if self.max_train_samples is not None and self.train_dataset is not None:
            n = min(self.max_train_samples, len(self.train_dataset))
            logger.info(
                "Subsampling train dataset: %d -> %d (max_train_samples=%d)",
                len(self.train_dataset),
                n,
                self.max_train_samples,
            )
            self.train_dataset = self.train_dataset.select(range(n))

        if self.train_dataset is None:
            raise ValueError(f"No 'train' split found in {self.data_dir}")

        # Adjust eval_strategy if no eval dataset available
        if self.eval_dataset is None and self.training_args is not None:
            logger.warning(
                "No 'val' or 'test' split found in %s; disabling evaluation",
                self.data_dir,
            )
            self.training_args.eval_strategy = "no"

        logger.info(
            "Data loaded: train=%d, eval=%d",
            len(self.train_dataset),
            len(self.eval_dataset) if self.eval_dataset else 0,
        )

    def _setup_callbacks(self) -> None:
        """Attach W&B callbacks to the SFTTrainer."""
        if self.training_args is None:
            raise RuntimeError("TrainingArguments not initialized. Call _setup_config() first.")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model/tokenizer not initialized. Call _setup_model_and_tokenizer() first."
            )

        # SFTConfig now carries packing/max_length as proper fields.
        # SFTTrainer skips re-construction since args is already an SFTConfig.
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            callbacks=[
                WandbCheckpointCallback(),
                WandbLoggingCallback(),
            ],
        )

        logger.info("SFTTrainer initialized with callbacks")

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """Run training (with optional resume from checkpoint)."""
        if self.trainer is None:
            self.setup()

        resume_path = self.resume_from_checkpoint
        if resume_path:
            from training.resume import resolve_checkpoint_path

            # Pass wandb run_id if available for "latest" resolution
            run_id = wandb.run.id if wandb.run else None
            resume_path = resolve_checkpoint_path(
                resume_path,
                local_volume_path=self.output_dir,
                run_id=run_id,
            )
            if resume_path:
                logger.info("Resuming from checkpoint: %s", resume_path)
            else:
                logger.warning("Resume path not found, starting from scratch")

        logger.info("Starting training (variant=%s, model=%s)...", self.variant, self.model_name)
        assert self.trainer is not None
        train_result = self.trainer.train(resume_from_checkpoint=resume_path)

        # Save final model
        self.save_model()

        # Log final metrics
        metrics = {
            "train_loss": train_result.training_loss
            if hasattr(train_result, "training_loss")
            else 0.0,
            "train_runtime": train_result.metrics.get("train_runtime", 0)
            if hasattr(train_result, "metrics")
            else 0,
            "train_samples_per_second": (
                train_result.metrics.get("train_samples_per_second", 0)
                if hasattr(train_result, "metrics")
                else 0
            ),
        }
        wandb.log(metrics)
        logger.info("Training complete. Metrics: %s", metrics)

        return metrics

    def save_model(self) -> None:
        """Save the LoRA adapter and tokenizer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer is None:
            logger.warning("No trainer to save")
            return

        # Save adapter
        self.trainer.save_model(str(self.output_dir))
        logger.info("Model saved to %s", self.output_dir)

        # Log as W&B artifact
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"model-{self.model_name}-{self.variant}",
                type="model_checkpoint",
                metadata={
                    "model_name": self.model_name,
                    "variant": self.variant,
                    "lora_r": self.lora_config.r if self.lora_config else None,
                    "lora_alpha": self.lora_config.lora_alpha if self.lora_config else None,
                },
            )
            artifact.add_dir(str(self.output_dir))
            wandb.log_artifact(artifact)
            artifact.wait()
            time.sleep(1)
            logger.info("Model artifact logged to W&B")

    def resume(self, checkpoint_path: str) -> dict[str, Any]:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Local path or W&B artifact ref.

        Returns:
            Training metrics.
        """
        self.resume_from_checkpoint = checkpoint_path
        return self.train()
