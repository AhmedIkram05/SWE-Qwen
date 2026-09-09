"""Unit tests for the standard (non-Unsloth) model build path.

No network, no GPU: transformers/peft loading is monkeypatched (same pattern
as ``test_qlora_trainer_unit.py``).

Step-5 note: the real offline run of ``build_model_and_peft`` on
SmolLM2-135M (llama arch, UNSLOTH_ENABLED=0, HF_HUB_OFFLINE=1) was
impossible — the model is absent from the local HF cache and network access
is forbidden — so the fallback path is verified here against a SmolLM2-135M
config with mocked loading.
"""

from __future__ import annotations

from typing import Any

import pytest

from training import unsloth_factory
from training.qlora_config import default_model_name, resolve_device_memory

LORA_PARAMS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj"],
}


def _patch_transformers(mocker) -> Any:
    """Monkeypatch transformers/peft loading; return the model loader mock."""
    mock_model = mocker.MagicMock()
    from_pretrained = mocker.patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
        autospec=False,
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock(),
        autospec=False,
    )
    mocker.patch(
        "peft.prepare_model_for_kbit_training",
        return_value=mock_model,
        autospec=False,
    )
    mocker.patch("peft.get_peft_model", return_value=mock_model, autospec=False)
    return from_pretrained


def _run_fallback(mocker, model_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Call the fallback builder and return (model, loader kwargs)."""
    from_pretrained = _patch_transformers(mocker)
    model, tokenizer = unsloth_factory._build_fallback(model_cfg, {"lora": dict(LORA_PARAMS)})
    assert model is not None and tokenizer is not None
    kwargs = from_pretrained.call_args.kwargs
    # hf_id is passed positionally
    assert from_pretrained.call_args.args[0] == model_cfg["hf_id"]
    return model, kwargs


@pytest.mark.unit
class TestFallbackQuantization:
    def test_smolm2_dispatch_nf4_offline(self, mocker):
        """build_model_and_peft dispatches to the fallback (unsloth absent)."""
        from_pretrained = _patch_transformers(mocker)
        model_cfg = {
            "hf_id": "HuggingFaceTB/SmolLM2-135M",
            "quantization": "nf4",
            "compute_dtype": "bfloat16",
            "gpu_mapping": {"primary": "a100-80gb", "fallback": "a10g-24gb"},
        }
        model, tokenizer = unsloth_factory.build_model_and_peft(
            model_cfg, {"lora": dict(LORA_PARAMS)}
        )
        assert model is not None and tokenizer is not None
        assert from_pretrained.call_args.args[0] == "HuggingFaceTB/SmolLM2-135M"
        qcfg = from_pretrained.call_args.kwargs["quantization_config"]
        assert qcfg.load_in_4bit is True
        assert qcfg.bnb_4bit_quant_type == "nf4"
        assert from_pretrained.call_args.kwargs["max_memory"] == {
            0: "70GiB",
            "cpu": "64GiB",
        }

    def test_nf4_quantization(self, mocker):
        _, kwargs = _run_fallback(mocker, {"hf_id": "h/nf4", "quantization": "nf4"})
        qcfg = kwargs["quantization_config"]
        assert qcfg.load_in_4bit is True
        assert qcfg.bnb_4bit_quant_type == "nf4"

    def test_fp4_fallback_quantization_wins(self, mocker):
        """Per-model fallback_quantization (registry workaround) overrides."""
        _, kwargs = _run_fallback(
            mocker,
            {
                "hf_id": "h/fp4",
                "quantization": "nf4",
                "fallback_quantization": "fp4",
            },
        )
        qcfg = kwargs["quantization_config"]
        assert qcfg.load_in_4bit is True
        assert qcfg.bnb_4bit_quant_type == "fp4"

    def test_int8_quantization(self, mocker):
        _, kwargs = _run_fallback(mocker, {"hf_id": "h/int8", "quantization": "int8"})
        qcfg = kwargs["quantization_config"]
        assert qcfg.load_in_8bit is True

    def test_none_quantization_plain_bf16(self, mocker):
        _, kwargs = _run_fallback(mocker, {"hf_id": "h/none", "quantization": "none"})
        assert kwargs["quantization_config"] is None

    def test_absent_quantization_defaults_nf4(self, mocker):
        _, kwargs = _run_fallback(mocker, {"hf_id": "h/noquant"})
        qcfg = kwargs["quantization_config"]
        assert qcfg.load_in_4bit is True
        assert qcfg.bnb_4bit_quant_type == "nf4"


@pytest.mark.unit
class TestResolveDeviceMemory:
    def test_a10g(self):
        assert resolve_device_memory({"gpu_mapping": {"primary": "a10g-24gb"}}) == {
            0: "18GiB",
            "cpu": "32GiB",
        }

    def test_a100_80gb(self):
        assert resolve_device_memory({"gpu_mapping": {"primary": "a100-80gb"}}) == {
            0: "70GiB",
            "cpu": "64GiB",
        }

    def test_h100_80gb(self):
        assert resolve_device_memory({"gpu_mapping": {"primary": "h100-80gb"}}) == {
            0: "70GiB",
            "cpu": "64GiB",
        }

    def test_a100_40gb(self):
        assert resolve_device_memory({"gpu_mapping": {"primary": "a100-40gb"}}) == {
            0: "34GiB",
            "cpu": "48GiB",
        }

    def test_unknown_tier_falls_back_to_a10g(self):
        assert resolve_device_memory({"gpu_mapping": {"primary": "t4-16gb"}}) == {
            0: "18GiB",
            "cpu": "32GiB",
        }

    def test_missing_gpu_mapping_falls_back_to_a10g(self):
        assert resolve_device_memory({"hf_id": "h/x"}) == {0: "18GiB", "cpu": "32GiB"}


@pytest.mark.unit
class TestDefaultModelName:
    def test_registry_default(self):
        assert default_model_name() == "qwen3-14b"

    def test_literal_last_resort_when_registry_absent(self, mocker):
        mocker.patch(
            "training.qlora_config.default_model_key",
            side_effect=OSError("registry missing"),
        )
        assert default_model_name() == "qwen3-14b"
