"""Training callbacks for W&B logging and checkpointing.

- ``WandbCheckpointCallback``: Uploads checkpoints as W&B Artifacts on save.
- ``WandbLoggingCallback``: Logs training metrics to W&B on each log step.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import wandb
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class WandbCheckpointCallback(TrainerCallback):
    """Upload checkpoints to W&B Artifacts when saved.

    Fires on ``on_save``: uploads the checkpoint directory as a
    ``model_checkpoint`` artifact with step/epoch/eval_loss metadata.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Any:
        """Called after a checkpoint is saved."""
        if wandb.run is None:
            return

        # Determine the latest checkpoint directory
        assert args.output_dir is not None
        checkpoint_dir = Path(args.output_dir)
        if not checkpoint_dir.exists():
            return

        # Find the latest checkpoint folder (global_step-XXXX)
        checkpoints = sorted(
            [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            return

        latest_ckpt = checkpoints[-1]
        step = int(latest_ckpt.name.split("-")[-1])

        artifact_name = (
            f"checkpoint-{wandb.run.name}-step-{step}"
            if wandb.run.name
            else f"checkpoint-step-{step}"
        )

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model_checkpoint",
            metadata={
                "step": step,
                "epoch": state.epoch if state.epoch else 0,
                "eval_loss": state.log_history[-1].get("eval_loss", None)
                if state.log_history
                else None,
                "global_step": state.global_step,
                "max_steps": state.max_steps,
            },
        )

        # Add the checkpoint directory (all files except optimizer state for size)
        artifact.add_dir(str(latest_ckpt), name="checkpoint")
        wandb.log_artifact(artifact)
        artifact.wait()
        time.sleep(1)

        logger.info("Checkpoint artifact logged: %s (step %d)", artifact_name, step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Any:
        """Final cleanup when training ends."""
        pass


class WandbLoggingCallback(TrainerCallback):
    """Log all training metrics to W&B on each logging step.

    Also logs experiment configuration at the start of training.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Any:
        """Log metrics to W&B."""
        logs: dict[str, float] | None = kwargs.get("logs")
        if wandb.run is None or not logs:
            return

        # Add step info
        metrics = dict(logs)
        metrics["global_step"] = state.global_step
        metrics["epoch"] = state.epoch

        wandb.log(metrics)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Any:
        """Log training configuration at the start."""
        if wandb.run is None:
            return

        # Log the full TrainingArguments as a config summary
        config_summary = {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler_type,
            "optim": args.optim,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
        }
        wandb.config.update(config_summary, allow_val_change=True)
        logger.debug("Training config logged to W&B")
