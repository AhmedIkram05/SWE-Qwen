"""Unit tests for the Phase 6 OpenAI wire format (inference.openai_compat).

Pure formatting logic — no fastapi/vllm/GPU needed.  resolve_engine_model
tests monkeypatch ``resolve_adapter_path`` so a WANDB_API_KEY in the
environment can never trigger a real artifact download.
"""

import json

import pytest

from evaluation.schema import EvalInput
from inference import prompt_builder
from inference.config import ServeConfig
from inference.openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    ModelNotFoundError,
    assemble_response,
    build_messages,
    error_response,
    iter_chunks,
    resolve_engine_model,
)

_BASE_HF_ID = "Qwen/Qwen3-14B"  # resolve_hf_id("qwen3-14b") from config/models.yaml
_FAKE_ADAPTER = "/tmp/fake-adapter"


def _patch_adapter(monkeypatch: pytest.MonkeyPatch, path: str | None = _FAKE_ADAPTER) -> None:
    monkeypatch.setattr(
        prompt_builder,
        "resolve_adapter_path",
        lambda variant, config=None: path,
    )


class TestResolveEngineModel:
    def test_base_model(self, monkeypatch):
        _patch_adapter(monkeypatch)
        assert resolve_engine_model("qwen3-14b", ServeConfig()) == (_BASE_HF_ID, None, None)

    def test_variant_suffix(self, monkeypatch):
        _patch_adapter(monkeypatch)
        assert resolve_engine_model("qwen3-14b:baseline_14b", ServeConfig()) == (
            _BASE_HF_ID,
            "qwen3-14b-baseline_14b",
            _FAKE_ADAPTER,
        )

    def test_bare_variant(self, monkeypatch):
        _patch_adapter(monkeypatch)
        assert resolve_engine_model("baseline_14b", ServeConfig()) == (
            _BASE_HF_ID,
            "qwen3-14b-baseline_14b",
            _FAKE_ADAPTER,
        )

    def test_wandb_artifact_name(self, monkeypatch):
        _patch_adapter(monkeypatch)
        assert resolve_engine_model("model-qwen3-14b-baseline_14b", ServeConfig()) == (
            _BASE_HF_ID,
            "qwen3-14b-baseline_14b",
            _FAKE_ADAPTER,
        )

    def test_all_other_variants(self, monkeypatch):
        _patch_adapter(monkeypatch)
        cfg = ServeConfig()
        for request_model in (
            "qwen3-14b:higher_rank_14b",
            "higher_lr_14b",
            "model-qwen3-14b-higher_lr_14b",
        ):
            hf_id, lora_name, lora_path = resolve_engine_model(request_model, cfg)
            assert hf_id == _BASE_HF_ID
            assert lora_path == _FAKE_ADAPTER
            assert lora_name in ("qwen3-14b-higher_rank_14b", "qwen3-14b-higher_lr_14b")

    def test_unknown_model_raises(self, monkeypatch):
        _patch_adapter(monkeypatch)
        with pytest.raises(ModelNotFoundError):
            resolve_engine_model("gpt-4", ServeConfig())

    def test_unknown_variant_raises(self, monkeypatch):
        _patch_adapter(monkeypatch)
        with pytest.raises(ModelNotFoundError):
            resolve_engine_model("qwen3-14b:does_not_exist", ServeConfig())
        with pytest.raises(ModelNotFoundError):
            resolve_engine_model("model-qwen3-14b-does_not_exist", ServeConfig())

    def test_adapter_unavailable_falls_back_to_base(self, monkeypatch):
        _patch_adapter(monkeypatch, path=None)
        assert resolve_engine_model("baseline_14b", ServeConfig()) == (
            _BASE_HF_ID,
            "qwen3-14b-baseline_14b",
            None,
        )


class TestBuildMessages:
    def test_system_and_user_split(self):
        request = ChatCompletionRequest(
            model="qwen3-14b",
            messages=[
                ChatMessage(role="system", content="You are a patch expert."),
                ChatMessage(role="user", content="Fix the bug."),
                ChatMessage(role="user", content="Also update the tests."),
            ],
        )
        assert build_messages(request) == (
            "You are a patch expert.",
            "Fix the bug.\n\nAlso update the tests.",
        )

    def test_no_system_message(self):
        request = ChatCompletionRequest(
            model="qwen3-14b",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert build_messages(request) == ("", "Hello")

    def test_swe_bench_uses_server_side_assembly(self, monkeypatch):
        monkeypatch.setattr(
            prompt_builder, "render_patch_prompt", lambda example: "RENDERED_PATCH_PROMPT"
        )
        record = {"instance_id": "django__django-1000", "repo": "django/django"}
        seen: dict[str, object] = {}

        def fake_from_swebench_record(cls, rec):
            seen["record"] = rec
            return "fake-eval-input"

        monkeypatch.setattr(
            EvalInput, "from_swebench_record", classmethod(fake_from_swebench_record)
        )
        request = ChatCompletionRequest(
            model="qwen3-14b",
            messages=[ChatMessage(role="user", content="ignored")],
            swe_bench=record,
        )
        assert build_messages(request) == ("", "RENDERED_PATCH_PROMPT")
        assert seen["record"] == record


class TestAssembleResponse:
    def test_shape(self):
        response = assemble_response(
            "chatcmpl-abc123",
            1234,
            "qwen3-14b",
            "hello world",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        assert isinstance(response.created, int)
        assert response.model == "qwen3-14b"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "hello world"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class TestIterChunks:
    def test_frame_sequence(self):
        frames = list(iter_chunks("chatcmpl-abc123", 1234, "qwen3-14b", iter(["one", "two"])))
        assert frames[-1] == "data: [DONE]\n\n"
        for frame in frames:
            assert frame.startswith("data: ")
            assert frame.endswith("\n\n")
        parsed = [json.loads(frame.removeprefix("data: ").rstrip("\n")) for frame in frames[:-1]]
        assert len(parsed) == 4  # role chunk + 2 content chunks + finish chunk
        assert parsed[0]["object"] == "chat.completion.chunk"
        assert parsed[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert parsed[0]["choices"][0]["finish_reason"] is None
        assert parsed[1]["choices"][0]["delta"] == {"content": "one"}
        assert parsed[2]["choices"][0]["delta"] == {"content": "two"}
        assert parsed[3]["choices"][0]["delta"] == {}
        assert parsed[3]["choices"][0]["finish_reason"] == "stop"


class TestErrorResponse:
    def test_shape(self):
        status, body = error_response(
            404, "model 'gpt-4' not found", "invalid_request_error", code="model_not_found"
        )
        assert status == 404
        assert body == {
            "error": {
                "message": "model 'gpt-4' not found",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        }
