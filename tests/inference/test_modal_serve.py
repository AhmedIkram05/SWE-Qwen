"""Coverage for ``inference.modal_serve`` (serverless serving app definition).

Module import exercises the fake-Modal Image chain / App registration; the
decorated class bodies are exercised directly with a stubbed ``VLLMEngine``.
No cloud, no GPU, no network.
"""

import pytest

from inference import modal_serve

pytestmark = pytest.mark.unit


class TestBuildSmoke:
    def test_generates_ping_on_a_fake_engine(self, mocker):
        engine = mocker.MagicMock()
        mocker.patch("inference.modal_serve.VLLMEngine", return_value=engine)
        modal_serve._build_smoke()
        engine.generate.assert_called_once_with(
            "ping",
            lora=None,
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            repetition_penalty=1.0,
        )


class TestQwenServer:
    def test_enter_constructs_engine(self, mocker):
        mocker.patch("inference.modal_serve.VLLMEngine", return_value="ENGINE")
        server = modal_serve.QwenServer()
        server.enter()  # type: ignore[attr-defined]
        assert server._engine == "ENGINE"

    def test_web_returns_create_app_result(self, mocker):
        mocker.patch("inference.modal_serve.create_app", return_value="APP")
        server = modal_serve.QwenServer()
        engine = object()
        server._engine = engine
        assert server.web() == "APP"  # type: ignore[attr-defined]

    def test_enter_then_web_wire_engine_into_app(self, mocker):
        app = mocker.MagicMock()
        mocker.patch("inference.modal_serve.create_app", return_value=app)
        server = modal_serve.QwenServer()
        engine = object()
        server._engine = engine
        assert server.web() is app  # type: ignore[attr-defined]
