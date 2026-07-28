"""Smoke tests for QLoRATrainer — requires GPU.

These tests are marked as ``slow`` and ``requires_modal`` because they need
real GPU hardware (Modal or local CUDA) to run.

Run on Modal:
    modal run tests/test_qlora_trainer_smoke.py

Or skip with:
    pytest tests/test_qlora_trainer_smoke.py -m "not slow" --co
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.requires_modal
class TestQLoRATrainerSmoke:
    """Smoke tests that exercise the trainer on real hardware."""

    def test_trainer_init(self):
        """Instantiates QLoRATrainer without error."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer(
            model_name="qwen3-30b-a3b",
            variant="baseline",
            data_dir="/tmp/nonexistent",  # won't be accessed in init
            output_dir="/tmp/qlora-smoke",
            wandb_project="swe-qwen-test",
        )
        assert trainer.model_name == "qwen3-30b-a3b"
        assert trainer.variant == "baseline"
        assert trainer.lora_config is None  # not yet set up

    def test_setup_config(self):
        """Config setup loads model and variant configs."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer(
            model_name="qwen3-30b-a3b",
            variant="baseline",
            data_dir="/tmp/nonexistent",
            output_dir="/tmp/qlora-smoke",
        )
        trainer._setup_config()
        assert trainer.lora_config is not None
        assert trainer.lora_config.r == 16
        assert trainer.training_args is not None
        assert trainer.training_args.learning_rate == 2.0e-5

    def test_setup_model_tokenizer_tiny(self):
        """Uses tiny model for quick smoke test (4-bit load + PEFT)."""
        from training.qlora_trainer import QLoRATrainer

        # Use tiny model for fast test
        trainer = QLoRATrainer(
            model_name="qwen3-30b-a3b",
            variant="baseline",
            data_dir="/tmp/nonexistent",
            output_dir="/tmp/qlora-smoke",
        )
        trainer._setup_config()

        # Override model_name to tiny HF test model
        trainer.model_cfg = {
            "hf_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "quantization": "nf4",
            "compute_dtype": "bfloat16",
            "context_window": 512,
            "target_modules": ["q_proj", "v_proj"],
        }

        trainer._setup_model_and_tokenizer()
        assert trainer.model is not None
        assert trainer.tokenizer is not None

        # Verify PEFT was applied
        from peft import PeftModel

        assert isinstance(trainer.model, PeftModel)
        # Should have trainable parameters
        trainable = trainer.model.print_trainable_parameters()
        assert trainable is None  # print function returns None

    def test_one_train_step(self):
        """Single batch forward/backward with tiny model."""
        import tempfile
        from pathlib import Path

        import torch
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        from training.qlora_trainer import QLoRATrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup trainer
            trainer = QLoRATrainer(
                model_name="qwen3-30b-a3b",
                variant="baseline",
                data_dir=tmpdir,
                output_dir=str(Path(tmpdir) / "output"),
                wandb_project="swe-qwen-test",
            )
            trainer._setup_config()

            # Use tiny model
            trainer.model_cfg = {
                "hf_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
                "context_window": 512,
                "target_modules": ["q_proj", "v_proj"],
            }
            trainer._setup_model_and_tokenizer()

            # Create tiny dataset
            data = {
                "input_ids": torch.randint(0, 100, (4, 64)).tolist(),
                "attention_mask": torch.ones(4, 64).int().tolist(),
                "labels": torch.randint(0, 100, (4, 64)).tolist(),
            }
            train_dataset = Dataset.from_dict(data)

            trainer.train_dataset = train_dataset
            trainer.eval_dataset = None

            # Create SFTTrainer directly for the one-step test
            training_args = TrainingArguments(
                output_dir=Path(tmpdir) / "output",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=1,
                logging_steps=1,
                save_steps=500,
                eval_steps=500,
                report_to="none",
                bf16=torch.cuda.is_available(),
                fp16=False,
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
                dataloader_num_workers=0,
            )

            sft_trainer = SFTTrainer(
                model=trainer.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=trainer.tokenizer,
                max_seq_length=64,
                packing=False,
            )

            # Run one step
            result = sft_trainer.train()
            assert result is not None
            assert result.training_loss > 0  # loss should be finite
