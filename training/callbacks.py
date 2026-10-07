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

    P0-4: uploads the checkpoint *this run* just saved (``checkpoint-{step}``
    where step == ``state.global_step``), never the global ``checkpoint-*``
    max — a stale checkpoint from a previous run sharing the output dir can
    no longer be uploaded for the wrong run. ``run_id`` (the W&B run ID at
    construction) additionally guards against a callback instance firing
    under a different W&B run.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id

    @staticmethod
    def _step_of(path: Path) -> int:
        """Numeric step of a ``checkpoint-{step}`` dir; -1 when unparsable."""
        step = path.name.rsplit("-", 1)[-1]
        return int(step) if step.isdigit() else -1

    def _select_checkpoint(self, output_dir: Path, global_step: int) -> Path | None:
        """Return the checkpoint this run just saved, or None.

        Primary: ``checkpoint-{global_step}`` — HF saves exactly this
        directory before ``on_save`` fires. Fallback (resume edge cases):
        the newest checkpoint at or below *global_step*.
        """
        current = output_dir / f"checkpoint-{global_step}"
        if current.is_dir():
            return current
        candidates = [
            p
            for p in output_dir.glob("checkpoint-*")
            if p.is_dir() and self._step_of(p) <= global_step
        ]
        if not candidates:
            return None
        return max(candidates, key=self._step_of)

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

        if self.run_id is not None and wandb.run.id != self.run_id:
            logger.warning(
                "WandbCheckpointCallback bound to run %s but active W&B run is %s — "
                "skipping checkpoint upload",
                self.run_id,
                wandb.run.id,
            )
            return

        # Determine the latest checkpoint directory
        assert args.output_dir is not None
        checkpoint_dir = Path(args.output_dir)
        if not checkpoint_dir.exists():
            return

        latest_ckpt = self._select_checkpoint(checkpoint_dir, state.global_step)
        if latest_ckpt is None:
            return

        step = self._step_of(latest_ckpt)

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
        try:
            artifact.wait(timeout=120)
        except Exception:  # WANDB service busy, network blip — don't crash training
            logger.warning(
                "W&B artifact wait timed out for %s — checkpoint saved locally, "
                "artifact may appear later",
                artifact_name,
            )
        else:
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
        """Log registry-normalized metrics to W&B.

        Only ``train/*`` registry keys are emitted (decision 7, plan §5.7):
        HF raw keys are renamed into their namespace and everything else is
        dropped. ``train/gpu_util`` is sampled on each log step (None on
        machines without nvidia-smi — key skipped).
        """
        logs: dict[str, float] | None = kwargs.get("logs")
        if wandb.run is None or not logs:
            return

        normalized: dict[str, float | int] = {}
        if "loss" in logs:
            normalized["train/loss"] = logs["loss"]
        if "learning_rate" in logs:
            normalized["train/lr"] = logs["learning_rate"]
        if "grad_norm" in logs:
            normalized["train/grad_norm"] = logs["grad_norm"]
        normalized["train/epoch"] = state.epoch if state.epoch is not None else 0.0
        normalized["train/step"] = state.global_step

        try:
            from inference.telemetry import log_gpu_util
        except ImportError:
            gpu_util = None
        else:
            gpu_util = log_gpu_util()
        if gpu_util is not None:
            normalized["train/gpu_util"] = gpu_util

        if normalized:
            wandb.log(normalized)

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
