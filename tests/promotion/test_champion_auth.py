"""Phase 9 Step 0: Bearer-token auth on POST /v1/chat/completions.

Offline unit tests: StubEngine, fake tokenizer (no HF fetch), no Modal.
``MODAL_SERVE_TOKEN`` is set per-test via monkeypatch so the developer's shell
env can never leak in (or out) of a test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference import prompt_builder
from inference.config import ServeConfig
from inference.serve import StubEngine, create_app

pytestmark = pytest.mark.unit

TOKEN = "test-serve-token"


class _FakeTokenizer:
    """Deterministic chat template — keeps the base-model path offline."""

    def apply_chat_template(self, messages, **kwargs):
        return "FAKE_CHAT:" + repr(messages)


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch):
    """Serve the base-model prompt path from the tokenizer cache (no network)."""
    monkeypatch.setitem(prompt_builder._TOKENIZER_CACHE, "Qwen/Qwen3-14B", _FakeTokenizer())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODAL_SERVE_TOKEN", TOKEN)
    return TestClient(create_app(StubEngine(), ServeConfig()))


def _payload():
    return {
        "model": "qwen3-14b",
        "messages": [{"role": "user", "content": "Write a patch."}],
        "stream": False,
    }


def test_no_authorization_header_401(client):
    response = client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid or missing bearer token"}


def test_wrong_token_401(client):
    response = client.post(
        "/v1/chat/completions", json=_payload(), headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_correct_token_200(client):
    response = client.post(
        "/v1/chat/completions", json=_payload(), headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.text  # StubEngine output — assert presence, not shape


def test_unset_env_fails_closed(monkeypatch):
    monkeypatch.delenv("MODAL_SERVE_TOKEN", raising=False)
    client = TestClient(create_app(StubEngine(), ServeConfig()))
    response = client.post(
        "/v1/chat/completions", json=_payload(), headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 401


def test_health_open_without_token(monkeypatch):
    monkeypatch.delenv("MODAL_SERVE_TOKEN", raising=False)
    client = TestClient(create_app(StubEngine(), ServeConfig()))
    response = client.get("/health")
    assert response.status_code == 200
