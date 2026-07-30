"""Local CLI entrypoint for SWE-Qwen training (for testing with tiny models).

Separated from ``modal_train.py`` so the orchestrator can import
``app`` and ``train_qlora`` without triggering the local entrypoint
when doing ``with app.run(): train_qlora.spawn()``.

Usage:
    modal run training/local_cli.py --model-name qwen3-14b --variant baseline_14b
"""

from __future__ import annotations

from training.modal_train import app  # noqa: PLC0414 — re-export for local entrypoint


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
    max_train_samples: int | None = None,
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
        max_train_samples=max_train_samples,
    )

    result = trainer.train()
    print(f"Training result: {result}")
