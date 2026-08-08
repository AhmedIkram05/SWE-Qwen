"""Serving-layer tests: FastAPI TestClient over the deterministic StubEngine.

No GPU, no vLLM, no HF downloads: the tokenizer cache is stubbed with a fake
``apply_chat_template`` and ``resolve_adapter_path`` is monkeypatched so a
WANDB_API_KEY in the environment can never trigger a real artifact download.
"""

import json

import pytest
from fastapi.testclient import TestClient

from inference import prompt_builder
from inference.config import ServeConfig
from inference.serve import StubEngine, create_app


class _FakeTokenizer:
    """Minimal tokenizer stand-in: deterministic chat template, no HF fetch."""

    def apply_chat_template(self, messages, **kwargs):
        return "FAKE_CHAT:" + repr(messages)


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch):
    """Serve the base-model path from the tokenizer cache (no network)."""
    monkeypatch.setitem(prompt_builder._TOKENIZER_CACHE, "Qwen/Qwen3-14B", _FakeTokenizer())


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("MODAL_SERVE_TOKEN", "test-token-123")
    monkeypatch.setattr(
        prompt_builder,
        "resolve_adapter_path",
        lambda variant, config=None: "/tmp/fake-adapter",
    )
    return ServeConfig()


@pytest.fixture
def client(config):
    return TestClient(create_app(StubEngine(), config))


class TestHealth:
    def test_ok(self, client, config):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model"] == config.serving_hf_id
        assert body["engine"] == "StubEngine"


class TestChatCompletions:
    @staticmethod
    def _payload(model="qwen3-14b", **overrides):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Write a patch."}],
        }
        payload.update(overrides)
        return payload

    def test_non_stream_base_model(self, client):
        response = client.post(
            "/v1/chat/completions",
            json=self._payload(),
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "qwen3-14b"
        choice = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"].startswith("stub[base]:")
        assert choice["finish_reason"] == "stop"
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert isinstance(body["usage"][key], int)

    def test_non_stream_variant(self, client):
        response = client.post(
            "/v1/chat/completions",
            json=self._payload(model="baseline_14b"),
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "baseline_14b"
        assert body["choices"][0]["message"]["content"].startswith("stub[qwen3-14b-baseline_14b]:")

    def test_stream(self, client):
        response = client.post(
            "/v1/chat/completions",
            json=self._payload(stream=True),
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 200
        text = response.text
        assert "data: [DONE]" in text
        frames = [
            json.loads(line.removeprefix("data: ").rstrip("\n"))
            for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert frames, "expected at least one SSE frame"
        assert frames[0]["object"] == "chat.completion.chunk"
        deltas = [frame["choices"][0]["delta"] for frame in frames]
        assert deltas[0] == {"role": "assistant"}
        assert any("content" in delta for delta in deltas)
        assert deltas[-1] == {}
        assert frames[-1]["choices"][0]["finish_reason"] == "stop"

    def test_unknown_model_404(self, client):
        response = client.post(
            "/v1/chat/completions",
            json=self._payload(model="gpt-4"),
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "model_not_found"

    def test_invalid_payload_422(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 422
        body = response.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"

    def test_engine_error_500(self, config):
        class FailingEngine:
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
                raise RuntimeError("boom")

        client = TestClient(create_app(FailingEngine(), config))
        response = client.post(
            "/v1/chat/completions",
            json=self._payload(),
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["type"] == "server_error"
