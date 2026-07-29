"""Shared pytest fixtures for training tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for checkpoint output."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_model_config() -> dict[str, Any]:
    """Pre-loaded models.yaml data."""
    return {
        "models": {
            "qwen3-14b": {
                "hf_id": "Qwen/Qwen3-14B",
                "active_params": 14_000_000_000,
                "total_params": 14_000_000_000,
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
                "gpu_mapping": {
                    "primary": "a10g-24gb",
                    "fallback": "a100-40gb",
                },
                "context_window": 32768,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "qwen3-30b-a3b": {
                "hf_id": "Qwen/Qwen3-30B-A3B",
                "active_params": 3_000_000_000,
                "total_params": 30_000_000_000,
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
                "gpu_mapping": {
                    "primary": "a100-40gb",
                    "fallback": "a10g-24gb",
                },
                "context_window": 32768,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "phase4_excluded": True,
            },
        },
    }


@pytest.fixture
def mock_variant_config() -> dict[str, Any]:
    """Pre-loaded qlora_variants.yaml data."""
    return {
        "variants": {
            "baseline_14b": {
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "training": {
                    "learning_rate": 2.0e-5,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "warmup_ratio": 0.03,
                    "lr_scheduler_type": "cosine",
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "bf16": True,
                    "fp16": False,
                    "optim": "paged_adamw_8bit",
                    "logging_steps": 10,
                    "save_steps": 500,
                    "eval_steps": 500,
                    "save_total_limit": 3,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                    "report_to": "wandb",
                    "ddp_find_unused_parameters": False,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 2,
                    "remove_unused_columns": False,
                    "packing": True,
                    "max_seq_length": 8192,
                },
            },
            "higher_rank_14b": {
                "lora": {
                    "r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.05,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "training": {
                    "learning_rate": 2.0e-5,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "warmup_ratio": 0.03,
                    "lr_scheduler_type": "cosine",
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "bf16": True,
                    "fp16": False,
                    "optim": "paged_adamw_8bit",
                    "logging_steps": 10,
                    "save_steps": 500,
                    "eval_steps": 500,
                    "save_total_limit": 3,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                    "report_to": "wandb",
                    "ddp_find_unused_parameters": False,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 2,
                    "remove_unused_columns": False,
                    "packing": True,
                    "max_seq_length": 8192,
                },
            },
            "higher_lr_14b": {
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "training": {
                    "learning_rate": 5.0e-5,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "warmup_ratio": 0.03,
                    "lr_scheduler_type": "cosine",
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "bf16": True,
                    "fp16": False,
                    "optim": "paged_adamw_8bit",
                    "logging_steps": 10,
                    "save_steps": 500,
                    "eval_steps": 500,
                    "save_total_limit": 3,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                    "report_to": "wandb",
                    "ddp_find_unused_parameters": False,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 2,
                    "remove_unused_columns": False,
                    "packing": True,
                    "max_seq_length": 8192,
                },
            },
            "efficient_14b": {
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": None,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "training": {
                    "learning_rate": 5.0e-5,
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "warmup_ratio": 0.03,
                    "lr_scheduler_type": "cosine",
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "bf16": True,
                    "fp16": False,
                    "optim": "paged_adamw_8bit",
                    "logging_steps": 10,
                    "save_steps": 500,
                    "eval_steps": 500,
                    "save_total_limit": 3,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                    "report_to": "wandb",
                    "ddp_find_unused_parameters": False,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 2,
                    "remove_unused_columns": False,
                    "packing": True,
                    "max_seq_length": 8192,
                },
            },
        },
    }


@pytest.fixture
def sample_tokenized_data_dict() -> dict:
    """Small sample tokenized dataset dict for smoke tests."""
    return {
        "train": {
            "input_ids": [[101, 102, 103], [201, 202, 203]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "labels": [[101, 102, 103], [201, 202, 203]],
        },
        "val": {
            "input_ids": [[301, 302, 303]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[301, 302, 303]],
        },
    }


@pytest.fixture
def mock_wandb_run(mocker):
    """Mock W&B run for callback tests."""
    mock_run = mocker.MagicMock()
    mock_run.name = "test-run-1234"
    mock_run.entity = "test-entity"
    mock_run.project = "test-project"
    mock_run.id = "run-abc123"
    mocker.patch("wandb.run", mock_run)
    mocker.patch("wandb.log")
    mocker.patch("wandb.log_artifact")
    mocker.patch("wandb.Artifact")
    return mock_run
