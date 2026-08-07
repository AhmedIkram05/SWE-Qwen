"""Unit tests for inference.config.ServeConfig (SERVING_ env-prefixed settings)."""

from inference.config import ServeConfig


class TestServeConfigDefaults:
    def test_defaults(self):
        cfg = ServeConfig()
        assert cfg.model_config.get("env_prefix") == "SERVING_"
        assert cfg.model_config.get("frozen") is True
        assert cfg.base_model == "qwen3-14b"
        assert cfg.serving_hf_id == "Qwen/Qwen3-14B-AWQ"
        assert cfg.quantization == "awq"
        assert cfg.variants == ("baseline_14b", "higher_rank_14b", "higher_lr_14b")
        assert cfg.default_variant == "higher_lr_14b"
        assert cfg.max_model_len == 4096
        assert cfg.gpu_memory_utilization == 0.85
        assert cfg.max_num_seqs == 16
        assert cfg.max_lora_rank == 64
        assert cfg.repetition_penalty == 1.15

    def test_env_override(self, monkeypatch):
        # Frozen BaseSettings still reads the environment at construction time.
        monkeypatch.setenv("SERVING_GPU_MEMORY_UTILIZATION", "0.90")
        assert ServeConfig().gpu_memory_utilization == 0.90

    def test_registry_serving_hf_id(self):
        # Single source of truth: config/models.yaml wins over the field.
        assert ServeConfig().registry_serving_hf_id() == "Qwen/Qwen3-14B-AWQ"
