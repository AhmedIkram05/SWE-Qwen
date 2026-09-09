"""Unit tests for inference.config.ServeConfig (SERVING_ env-prefixed settings)."""

import pytest

from inference.config import ServeConfig
from registry.loader import ModelSpec

# tests/inference/test_benchmark.py leaks SERVING_* into os.environ (it drives
# inference/benchmark.py which exports its sweep knobs as env vars). ServeConfig
# must honor the env, so scrub it here instead of fighting the leak.
_SERVING_ENV_VARS = (
    "SERVING_BASE_MODEL",
    "SERVING_SERVING_HF_ID",
    "SERVING_QUANTIZATION",
    "SERVING_VARIANTS",
    "SERVING_DEFAULT_VARIANT",
    "SERVING_LORA_ARTIFACT_PATTERN",
    "SERVING_MAX_MODEL_LEN",
    "SERVING_DEFAULT_MAX_TOKENS",
    "SERVING_MAX_TOKENS_CAP",
)


@pytest.fixture(autouse=True)
def _clean_serving_env(monkeypatch):
    for var in _SERVING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _fresh_spec(**overrides) -> ModelSpec:
    """A fresh (non-Qwen) registry entry: plain base, no pre-quantized id."""
    base = {
        "hf_id": "meta-llama/Fresh-8B",
        "context_window": 16384,
        "target_modules": ["q_proj"],
        "serving_hf_id": None,
        "serving_quantization": None,
    }
    base.update(overrides)
    return ModelSpec.model_validate(base)


def _patch_registry(monkeypatch, models: dict[str, ModelSpec] | None) -> str:
    """Point the config's loader at fake *models* and return the default key."""
    key = next(iter(models)) if models else "qwen3-14b"
    monkeypatch.setattr("inference.config.load_models", lambda: models or {})
    monkeypatch.setattr("inference.config.default_model_key", lambda: key)
    return key


class TestServeConfigDefaults:
    def test_defaults(self):
        cfg = ServeConfig()
        assert cfg.model_config.get("env_prefix") == "SERVING_"
        assert cfg.model_config.get("frozen") is True
        assert cfg.base_model == "qwen3-14b"  # registry default: true
        assert cfg.serving_hf_id == "Qwen/Qwen3-14B-AWQ"  # registry serving_hf_id
        assert cfg.quantization == "awq"  # registry serving_quantization
        assert cfg.variants == ("baseline_14b", "higher_rank_14b", "higher_lr_14b")
        assert cfg.default_variant == "higher_rank_14b"
        assert cfg.lora_artifact_pattern == "model-qwen3-14b-{variant}"
        # PHASE-10 Step 9-3: context-coupled knobs default to context_window
        assert cfg.max_model_len == 32768
        assert cfg.default_max_tokens == 32768
        assert cfg.max_tokens_cap == 32768
        assert cfg.gpu_memory_utilization == 0.85
        assert cfg.max_num_seqs == 16
        assert cfg.max_lora_rank == 64
        assert cfg.repetition_penalty == 1.15

    def test_env_override(self, monkeypatch):
        # Frozen BaseSettings still reads the environment at construction time.
        monkeypatch.setenv("SERVING_GPU_MEMORY_UTILIZATION", "0.90")
        assert ServeConfig().gpu_memory_utilization == 0.90

    def test_env_wins_over_registry(self, monkeypatch):
        # PHASE-10: env priority is above the registry, not below it.
        monkeypatch.setenv("SERVING_SERVING_HF_ID", "env/quantized-base")
        monkeypatch.setenv("SERVING_MAX_MODEL_LEN", "8192")
        monkeypatch.setenv("SERVING_VARIANTS", '["only_env_variant"]')
        cfg = ServeConfig()
        assert cfg.serving_hf_id == "env/quantized-base"
        assert cfg.max_model_len == 8192
        assert cfg.variants == ("only_env_variant",)
        # untouched fields still come from the registry
        assert cfg.default_max_tokens == 32768

    def test_registry_serving_hf_id(self):
        # Single source of truth: config/models.yaml wins over the field.
        assert ServeConfig().registry_serving_hf_id() == "Qwen/Qwen3-14B-AWQ"


class TestRegistryResolution:
    """PHASE-10 Steps 2/3: registry-first, env above, literals last resort."""

    def test_fresh_model_empty_safe(self, monkeypatch):
        key = _patch_registry(monkeypatch, {"fresh": _fresh_spec()})
        cfg = ServeConfig()
        assert cfg.base_model == "fresh" == key
        assert cfg.serving_hf_id == ""
        assert cfg.quantization == ""
        assert cfg.variants == ()
        assert cfg.default_variant == ""
        assert cfg.lora_artifact_pattern == "model-fresh-{variant}"
        assert cfg.max_model_len == 16384
        assert cfg.default_max_tokens == 16384
        assert cfg.max_tokens_cap == 16384
        # LoRA path resolves the plain base
        assert cfg.registry_serving_hf_id() == "meta-llama/Fresh-8B"

    def test_fresh_model_explicit_variants(self, monkeypatch):
        _patch_registry(
            monkeypatch, {"fresh": _fresh_spec(variants=["v1", "v2"], default_variant="v1")}
        )
        cfg = ServeConfig()
        assert cfg.variants == ("v1", "v2")
        assert cfg.default_variant == "v1"

    def test_env_overrides_fresh_model(self, monkeypatch):
        _patch_registry(monkeypatch, {"fresh": _fresh_spec()})
        monkeypatch.setenv("SERVING_DEFAULT_VARIANT", "my-champion")
        monkeypatch.setenv("SERVING_LORA_ARTIFACT_PATTERN", "custom-{variant}")
        cfg = ServeConfig()
        assert cfg.default_variant == "my-champion"
        assert cfg.lora_artifact_pattern == "custom-{variant}"

    def test_serving_quantization_without_serving_hf_id_raises(self, monkeypatch):
        # PHASE-10 Step 3: fail at config load, not at vLLM boot.
        _patch_registry(monkeypatch, {"fresh": _fresh_spec(serving_quantization="awq")})
        with pytest.raises(ValueError, match="serving_quantization.*no serving_hf_id"):
            ServeConfig()

    def test_unknown_base_model_raises(self, monkeypatch):
        # No silent Qwen fallback when the registry is readable (Step 9-2).
        monkeypatch.setenv("SERVING_BASE_MODEL", "does-not-exist")
        with pytest.raises(ValueError, match="unknown model 'does-not-exist'"):
            ServeConfig()

    def test_registry_missing_uses_literals(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise OSError("no registry")

        monkeypatch.setattr("inference.config.load_models", fail)
        monkeypatch.setattr("inference.config.default_model_key", fail)
        cfg = ServeConfig()
        assert cfg.base_model == "qwen3-14b"
        assert cfg.serving_hf_id == "Qwen/Qwen3-14B-AWQ"
        assert cfg.quantization == "awq"
        assert cfg.variants == ("baseline_14b", "higher_rank_14b", "higher_lr_14b")
        assert cfg.default_variant == "higher_rank_14b"
        assert cfg.lora_artifact_pattern == "model-qwen3-14b-{variant}"
        assert cfg.max_model_len == 4096
