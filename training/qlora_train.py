#!/usr/bin/env python
"""Thin CLI wrapper for QLoRA training.

Enables ``python -m training.qlora_train`` per Master Plan acceptance criteria.

Usage:
    python -m training.qlora_train --model-name qwen3-30b-a3b \\
        --variant baseline --data-dir data/tokenized
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SWE-Qwen QLoRA Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="qwen3-30b-a3b",
        help="Model key from models.yaml",
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        help="QLoRA variant from qlora_variants.yaml",
    )
    parser.add_argument(
        "--data-dir",
        default="data/tokenized",
        help="Path to tokenized .arrow shards",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/qlora-output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path or W&B artifact ref for resume",
    )
    parser.add_argument(
        "--wandb-project",
        default="swe-qwen",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity name",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention 2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from training.qlora_trainer import QLoRATrainer

    trainer = QLoRATrainer(
        model_name=args.model_name,
        variant=args.variant,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        resume_from_checkpoint=args.resume,
        use_flash_attn=not args.no_flash_attn,
    )

    metrics = trainer.train()
    print(f"\nTraining complete. Metrics: {metrics}")


def app() -> None:
    """Entry point for ``pyproject.toml [project.scripts]``."""
    main()


if __name__ == "__main__":
    main(sys.argv[1:])
