"""Tests for ``training/qlora_config.py``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.qlora_config import (
    GPU_MAP,
    _get_model_config,
    _get_variant_config,
    build_qlora_config,
    list_models,
    list_variants,
    resolve_gpu_type,
)


class TestModelConfig:
    """Tests for model config loading."""

    def test_get_model_config_14b_primary(self):
        """Returns correct config for primary 14B model."""
        cfg = _get_model_config("qwen3-14b")
        assert cfg["hf_id"] == "Qwen/Qwen3-14B"
        assert cfg["context_window"] == 32768
        assert "target_modules" in cfg
        assert len(cfg["target_modules"]) == 7
        assert cfg["gpu_mapping"]["primary"] == "a10g-24gb"

    def test_get_model_config_14b(self):
        """Returns correct config for fallback model."""
        cfg = _get_model_config("qwen3-14b")
        assert cfg["hf_id"] == "Qwen/Qwen3-14B"
        assert cfg["gpu_mapping"]["primary"] == "a10g-24gb"

    def test_get_model_config_invalid_raises(self):
        """Raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Unknown model"):
            _get_model_config("nonexistent-model")


class TestVariantConfig:
    """Tests for variant config loading and merging."""

    def test_baseline_14b_config(self):
        """Returns baseline_14b variant with all fields."""
        cfg = _get_variant_config("baseline_14b")
        assert cfg["lora"]["r"] == 16
        assert cfg["lora"]["lora_alpha"] == 32
        assert cfg["lora"]["lora_dropout"] == 0.05
        assert cfg["training"]["learning_rate"] == 2.0e-5
        assert cfg["training"]["num_train_epochs"] == 3
        assert cfg["training"]["bf16"] is True
        assert cfg["training"]["gradient_checkpointing"] is True

    def test_higher_rank_14b_inherits_rest(self):
        """Higher rank variant overrides r/alpha but inherits rest."""
        cfg = _get_variant_config("higher_rank_14b")
        assert cfg["lora"]["r"] == 32
        assert cfg["lora"]["lora_alpha"] == 64
        # Inherited from baseline_14b
        assert cfg["lora"]["lora_dropout"] == 0.05
        assert cfg["lora"]["bias"] == "none"
        assert cfg["training"]["learning_rate"] == 2.0e-5
        assert cfg["training"]["num_train_epochs"] == 3
        assert cfg["training"]["per_device_train_batch_size"] == 1
        assert cfg["training"]["gradient_accumulation_steps"] == 16

    def test_higher_lr_14b_overrides_lr(self):
        """Higher LR variant overrides learning rate."""
        cfg = _get_variant_config("higher_lr_14b")
        assert cfg["lora"]["r"] == 16  # same as baseline_14b
        assert cfg["training"]["learning_rate"] == 5.0e-5  # overridden
        assert cfg["training"]["num_train_epochs"] == 3  # inherited

    def test_efficient_14b_config(self):
        """Efficient variant for cheap iteration."""
        cfg = _get_variant_config("efficient_14b")
        assert cfg["lora"]["r"] == 8
        assert cfg["lora"]["lora_alpha"] == 16
        assert cfg["training"]["learning_rate"] == 5.0e-5
        assert cfg["training"]["num_train_epochs"] == 2

    def test_invalid_variant_raises(self):
        """Raises KeyError for unknown variant."""
        with pytest.raises(KeyError, match="Unknown variant"):
            _get_variant_config("nonexistent")

    def test_all_variants_have_required_fields(self):
        """All variants have lora and training sections."""
        variants = list_variants()
        for v in variants:
            cfg = _get_variant_config(v)
            assert "lora" in cfg
            assert "training" in cfg
            assert cfg["lora"].get("r") is not None


class TestBuildQLoraConfig:
    """Tests for the factory function."""

    def test_build_baseline_14b_config(self):
        """Returns LoraConfig and TrainingArguments with correct params for baseline_14b."""
        lora_config, training_args = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
        )
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        assert lora_config.task_type == "CAUSAL_LM"
        # target_modules resolved from model config
        assert len(lora_config.target_modules or []) == 7

        assert training_args.learning_rate == 2.0e-5
        assert training_args.num_train_epochs == 3
        assert training_args.bf16 is True
        assert training_args.fp16 is False
        assert training_args.gradient_checkpointing is True
        assert training_args.max_length == 4096

    def test_build_higher_rank_14b_config(self):
        """Higher rank: r=32, alpha=64, batch=1, grad_accum=16."""
        lora_config, training_args = build_qlora_config(
            variant="higher_rank_14b",
            model_name="qwen3-14b",
        )
        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert training_args.per_device_train_batch_size == 1
        assert training_args.gradient_accumulation_steps == 16

    def test_build_higher_lr_14b_config(self):
        """Higher LR: lr=5e-5."""
        _, training_args = build_qlora_config(
            variant="higher_lr_14b",
            model_name="qwen3-14b",
        )
        assert training_args.learning_rate == 5.0e-5

    def test_build_efficient_14b_config(self):
        """Efficient variant: r=8, lr=5e-5, epochs=2."""
        lora_config, training_args = build_qlora_config(
            variant="efficient_14b",
            model_name="qwen3-14b",
        )
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert training_args.learning_rate == 5.0e-5
        assert training_args.num_train_epochs == 2

    def test_target_modules_merged_from_model_config(self):
        """target_modules resolved from model config when variant has null."""
        lora_config, _ = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
        )
        expected = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert sorted(lora_config.target_modules or []) == sorted(expected)

    def test_build_with_output_dir(self):
        """Custom output_dir is respected."""
        _, training_args = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
            output_dir="/custom/path",
        )
        assert training_args.output_dir == "/custom/path"

    def test_build_with_run_name(self):
        """Custom run_name is set."""
        _, training_args = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
            run_name="my-test-run",
        )
        assert training_args.run_name == "my-test-run"

    def test_gpu_memory_overrides_applied(self):
        """GPU-specific overrides (packing, max_seq_length) are applied."""
        lora_config, training_args = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
            gpu_type="A10G:1",
        )
        # A10G overrides from GPU_MEMORY_OVERRIDES
        assert training_args.packing is False
        assert training_args.max_length == 2048


class TestResolveGPU:
    """Tests for GPU type resolution."""

    def test_primary_gpu_14b(self):
        """14B model resolves to A10G:1 (primary for Phase 4)."""
        gpu = resolve_gpu_type("qwen3-14b")
        assert gpu == "A10G:1"

    def test_fallback_gpu_14b(self):
        """14B model with fallback resolves to A100:1."""
        gpu = resolve_gpu_type("qwen3-14b", prefer_fallback=True)
        assert gpu == "A100:1"

    def test_30b_excluded_from_phase4(self):
        """30B model still resolves but is phase4_excluded in config."""
        gpu = resolve_gpu_type("qwen3-30b-a3b")
        assert gpu == "H100:1"

    def test_gpu_map_complete(self):
        """All expected GPU mappings exist."""
        assert "A100:1" in GPU_MAP.values()
        assert "A10G:1" in GPU_MAP.values()
        assert "H100:1" in GPU_MAP.values()


class TestListFunctions:
    """Tests for list_models and list_variants."""

    def test_list_models(self):
        models = list_models()
        assert "qwen3-30b-a3b" in models
        assert "qwen3-14b" in models

    def test_list_variants(self):
        variants = list_variants()
        assert "baseline_14b" in variants
        assert "higher_rank_14b" in variants
        assert "higher_lr_14b" in variants
        assert "efficient_14b" in variants


class TestConfigYamlFiles:
    """Verify the actual YAML files are valid."""

    def test_models_yaml_is_valid(self):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        models_path = config_dir / "models.yaml"
        assert models_path.exists(), "models.yaml not found"
        with models_path.open() as f:
            data = yaml.safe_load(f)
        assert "models" in data
        assert "qwen3-30b-a3b" in data["models"]
        assert "qwen3-14b" in data["models"]

    def test_qlora_variants_yaml_is_valid(self):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        variants_path = config_dir / "qlora_variants.yaml"
        assert variants_path.exists(), "qlora_variants.yaml not found"
        with variants_path.open() as f:
            data = yaml.safe_load(f)
        assert "variants" in data
        assert "baseline_14b" in data["variants"]
        assert "higher_rank_14b" in data["variants"]
        assert "higher_lr_14b" in data["variants"]
        assert "efficient_14b" in data["variants"]
