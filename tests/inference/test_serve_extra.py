"""Extra coverage for ``inference.serve``: the real VLLMEngine paths, stream
error branches, ``_build_prompt`` variants, and the stub-selection switch.

The ``vllm`` package is faked via ``sys.modules`` inside the tests; the
tokenizer cache is stubbed so nothing ever touches HuggingFace.
"""

import sys
import types

import pytest
from fastapi.testclient import TestClient

from inference import prompt_builder, serve
from inference.config import ServeConfig

pytestmark = pytest.mark.unit


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"FAKE_CHAT:{len(messages)}"


@pytest.fixture
def stub_tokenizer(monkeypatch):
    monkeypatch.setitem(prompt_builder._TOKENIZER_CACHE, "Qwen/Qwen3-14B", _FakeTokenizer())


@pytest.fixture
def fake_adapter(monkeypatch):
    monkeypatch.setattr(
        prompt_builder, "resolve_adapter_path", lambda variant, config=None: "/tmp/fake-adapter"
    )


def _install_fake_vllm(mocker) -> dict:
    """Install a vLLM stand-in capturing constructor/generate arguments."""
    vllm_mod = types.ModuleType("vllm")
    state: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            state["llm_kwargs"] = kwargs

        def generate(self, prompts, sampling_params, lora_request=None):
            state["sampling_params"] = sampling_params
            state["lora_request"] = lora_request
            token = types.SimpleNamespace(text="generated text", token_ids=[1, 2, 3])
            output = types.SimpleNamespace(outputs=[token], prompt_token_ids=[4, 5, 6])
            return [output]

    vllm_mod.LLM = FakeLLM
    vllm_mod.SamplingParams = types.SimpleNamespace

    lora_mod = types.ModuleType("vllm.lora")
    request_mod = types.ModuleType("vllm.lora.request")
    request_mod.LoRARequest = lambda lora_name=None, lora_int_id=None, lora_path=None: (
        types.SimpleNamespace(lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path)
    )
    lora_mod.request = request_mod
    vllm_mod.lora = lora_mod

    mocker.patch.dict(
        sys.modules,
        {"vllm": vllm_mod, "vllm.lora": lora_mod, "vllm.lora.request": request_mod},
    )
    return state


class TestVLLMEngine:
    def test_generate_with_lora_then_cache_hit(self, mocker):
        state = _install_fake_vllm(mocker)
        mocker.patch.object(serve, "_LLM_CACHE", {})
        engine = serve.VLLMEngine(ServeConfig())

        result = engine.generate(
            "prompt",
            lora=("adapter_name", "/path"),
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            repetition_penalty=1.0,
        )
        assert result.text == "generated text"
        assert result.completion_tokens == 3
        assert state["lora_request"].lora_name == "adapter_name"
        assert state["sampling_params"].max_tokens == 16
        assert state["llm_kwargs"]["enable_lora"] is True
        assert state["llm_kwargs"]["quantization"] == "awq"

        # second engine reuses the process-singleton LLM (cache-hit branch)
        second = serve.VLLMEngine(ServeConfig())
        result2 = second.generate(
            "p2",
            lora=None,
            max_tokens=8,
            temperature=0.5,
            top_p=0.9,
            stop=["\n\n"],
            repetition_penalty=1.2,
        )
        assert result2.prompt_tokens == 3
        assert state["lora_request"] is None
        assert len(serve._LLM_CACHE) == 1

    def test_generate_respects_stop_list(self, mocker, monkeypatch):
        state = _install_fake_vllm(mocker)
        mocker.patch.object(serve, "_LLM_CACHE", {})
        engine = serve.VLLMEngine(ServeConfig())
        engine.generate(
            "p",
            lora=("n", "p"),
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            stop=[],
            repetition_penalty=1.1,
        )
        assert state["sampling_params"].stop == []


class TestDefaultEngine:
    def test_serving_stub_zero_picks_vllm(self, monkeypatch):
        monkeypatch.setenv("SERVING_STUB", "0")
        assert isinstance(serve._default_engine(), serve.VLLMEngine)

    def test_default_is_stub(self, monkeypatch):
        monkeypatch.delenv("SERVING_STUB", raising=False)
        assert isinstance(serve._default_engine(), serve.StubEngine)


class TestCreateApp:
    def test_create_app_without_config(self):
        app = serve.create_app(serve.StubEngine())
        assert app.title == "swe-qwen-inference"
        assert app.version == "0.1.0"


class TestBuildPrompt:
    def test_lora_inserts_no_think_gate(self, fake_adapter):
        prompt = serve._build_prompt(
            "Qwen/Qwen3-14B", "SYS", "BODY\n### Response\nREST", ("lora", "/p")
        )
        assert prompt == "SYS\n\nBODY\n/no_think\n### Response\nREST"

    def test_lora_without_system_prompt(self, fake_adapter):
        prompt = serve._build_prompt(
            "Qwen/Qwen3-14B", "", "BODY\n### Response\nREST", ("lora", "/p")
        )
        assert prompt == "BODY\n/no_think\n### Response\nREST"

    def test_base_with_system_prompt(self, stub_tokenizer):
        prompt = serve._build_prompt("Qwen/Qwen3-14B", "SYS", "HI", None)
        assert prompt == "FAKE_CHAT:2"

    def test_base_tokenizer_miss_imports_transformers(self, mocker):
        hf_id = "Other/Model"
        tok = types.SimpleNamespace()
        tok.apply_chat_template = lambda messages, **kwargs: "FROM_HF"
        fake_tr = types.ModuleType("transformers")
        fake_tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=staticmethod(lambda hf: tok))
        mocker.patch.dict(sys.modules, {"transformers": fake_tr})
        mocker.patch.object(prompt_builder, "_TOKENIZER_CACHE", {})
        prompt = serve._build_prompt(hf_id, "SYS", "HI", None)
        assert prompt == "FROM_HF"
        assert hf_id in prompt_builder._TOKENIZER_CACHE


class _ThrowingEngine:
    def __init__(self, exc):
        self._exc = exc

    def generate(  # noqa: PLR0913
        self,
        prompt,
        *,
        lora,
        max_tokens,
        temperature,
        top_p,
        stop,
        repetition_penalty,
    ):
        raise self._exc


def _stream_payload():
    return {
        "model": "baseline_14b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


class TestStreamingErrors:
    def test_runtime_error_swallowed_as_cancelled(self, fake_adapter):
        client = TestClient(
            serve.create_app(_ThrowingEngine(RuntimeError("cancelled")), ServeConfig())
        )
        response = client.post("/v1/chat/completions", json=_stream_payload())
        assert response.status_code == 200
        assert "[DONE]" not in response.text

    def test_engine_error_streams_error_frame(self, fake_adapter):
        client = TestClient(
            serve.create_app(_ThrowingEngine(ValueError("gen failed")), ServeConfig())
        )
        response = client.post("/v1/chat/completions", json=_stream_payload())
        assert response.status_code == 200
        assert "internal generation failure" in response.text
        assert "[DONE]" in response.text


class TestStubEngine:
    def test_generates_tagged_text(self):
        result = serve.StubEngine().generate(
            "ppp",
            lora=("qwen3-14b-v", "/p"),
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            repetition_penalty=1.0,
        )
        assert result.text.startswith("stub[qwen3-14b-v]:")
        assert result.prompt_tokens == 1

    def test_generates_base_tag_without_lora(self):
        result = serve.StubEngine().generate(
            "ppp",
            lora=None,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            repetition_penalty=1.0,
        )
        assert result.text.startswith("stub[base]:")
