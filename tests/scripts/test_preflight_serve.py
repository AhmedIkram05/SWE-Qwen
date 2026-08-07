"""Unit tests for ``scripts/preflight_serve.py``.

The serving endpoint, httpx and the OpenAI SDK are all mocked — no network.
"""

from __future__ import annotations

import sys

import pytest


def _load():
    from scripts import preflight_serve

    return preflight_serve


@pytest.fixture
def mod():
    return _load()


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Delta:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.content = content
        self.finish_reason = finish_reason


class _Choice:
    def __init__(self, message=None, delta=None, finish_reason=None) -> None:
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, completion_tokens: int) -> None:
        self.completion_tokens = completion_tokens


class _Completion:
    object = "chat.completion"

    def __init__(self, content: str) -> None:
        self.choices = [_Choice(message=_Message(content))]
        self.usage = _Usage(len(content))


class _Chunk:
    def __init__(self, choices) -> None:
        self.choices = choices


def _stream_chunks() -> list[_Chunk]:
    """One content delta + a terminating finish_reason frame."""
    return [
        _Chunk([_Choice(delta=_Delta(content="def f():\n    return 1"))]),
        _Chunk([_Choice(delta=_Delta(content=None), finish_reason="stop")]),
    ]


def _fake_create(stream: bool = False, **kwargs):  # noqa: ARG001
    if stream:
        return _stream_chunks()
    return _Completion(content="def f():\n    return 1")


class TestEnv:
    def test_missing(self, mod, mocker, capsys):
        mocker.patch.dict(mod.os.environ, {}, clear=True)
        with pytest.raises(SystemExit) as e:
            mod._env()
        assert e.value.code == 2
        assert "both be set" in capsys.readouterr().err

    def test_partial(self, mod, mocker):
        mocker.patch.dict(mod.os.environ, {"MODAL_WEB_URL": "http://x"}, clear=True)
        with pytest.raises(SystemExit) as e:
            mod._env()
        assert e.value.code == 2

    def test_ok(self, mod, mocker):
        mocker.patch.dict(
            mod.os.environ,
            {"MODAL_WEB_URL": "http://x.modal.run", "MODAL_WEB_TOKEN": "t0k"},
            clear=True,
        )
        assert mod._env() == ("http://x.modal.run", "t0k")


class TestCheck:
    def test_pass(self, mod, capsys):
        assert mod._check("health", True, "detail") is True
        assert "PASS: health (detail)" in capsys.readouterr().out

    def test_fail_no_detail(self, mod, capsys):
        assert mod._check("step", False) is False
        assert "FAIL: step" in capsys.readouterr().out


class TestNonStream:
    def test_success(self, mod, mocker):
        client = mocker.MagicMock()
        client.chat.completions.create.return_value = _Completion(content="def f():\n    pass")
        assert mod._non_stream(client, mod._MODEL_BASE) is True

    def test_empty_content(self, mod, mocker, capsys):
        client = mocker.MagicMock()
        client.chat.completions.create.return_value = _Completion(content="")
        assert mod._non_stream(client, mod._MODEL_BASE) is False
        assert "detail:" in capsys.readouterr().out

    def test_exception(self, mod, mocker, capsys):
        client = mocker.MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        assert mod._non_stream(client, mod._MODEL_BASE) is False
        assert "error: RuntimeError" in capsys.readouterr().out


class TestStream:
    def test_success(self, mod, mocker):
        client = mocker.MagicMock()
        client.chat.completions.create.return_value = _stream_chunks()
        assert mod._stream(client) is True

    def test_missing_delta(self, mod, mocker, capsys):
        client = mocker.MagicMock()
        client.chat.completions.create.return_value = [
            _Chunk([_Choice(delta=_Delta(content=None, finish_reason="stop"))])
        ]
        assert mod._stream(client) is False
        assert "content_deltas=0" in capsys.readouterr().out

    def test_empty_stream(self, mod, mocker):
        client = mocker.MagicMock()
        client.chat.completions.create.return_value = []
        assert mod._stream(client) is False

    def test_exception(self, mod, mocker, capsys):
        client = mocker.MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("stream err")
        assert mod._stream(client) is False
        assert "error: RuntimeError" in capsys.readouterr().out


class TestMain:
    @pytest.fixture
    def env(self, mod, mocker):
        mocker.patch.dict(
            mod.os.environ,
            {"MODAL_WEB_URL": "http://x.modal.run", "MODAL_WEB_TOKEN": "t0k"},
            clear=True,
        )

    @pytest.fixture
    def fake_sdks(self, mod, mocker):
        """Replace httpx/openai in sys.modules so the in-main imports hit fakes."""
        openai_fake = mocker.MagicMock()
        openai_fake.OpenAI.return_value.chat.completions.create.side_effect = _fake_create
        httpx_fake = mocker.MagicMock()
        httpx_fake.codes.OK = 200
        health = httpx_fake.Client.return_value.__enter__.return_value.get.return_value
        health.status_code = 200
        health.json.return_value = {"status": "ok"}
        mocker.patch.dict(sys.modules, {"openai": openai_fake, "httpx": httpx_fake})
        return httpx_fake

    def test_main_success(self, mod, env, fake_sdks, capsys):
        assert mod.main() == 0
        assert "preflight passed" in capsys.readouterr().out

    def test_main_health_raises(self, mod, env, fake_sdks):
        fake_sdks.Client.return_value.__enter__.return_value.get.side_effect = RuntimeError(
            "conn refused"
        )
        assert mod.main() == 1

    def test_main_health_bad_status(self, mod, env, fake_sdks):
        fake_sdks.Client.return_value.__enter__.return_value.get.return_value.status_code = 500
        assert mod.main() == 1

    def test_main_health_bad_body(self, mod, env, fake_sdks):
        fake_sdks.Client.return_value.__enter__.return_value.get.return_value.json.return_value = {
            "status": "down"
        }
        assert mod.main() == 1

    def test_main_non_stream_fails(self, mod, env, fake_sdks, mocker):
        # base model check passes, LoRA check fails → return 1
        mocker.patch.object(mod, "_non_stream", side_effect=[True, False])
        assert mod.main() == 1

    def test_main_non_stream_base_fails(self, mod, env, fake_sdks, mocker):
        mocker.patch.object(mod, "_non_stream", return_value=False)
        assert mod.main() == 1

    def test_main_stream_fails(self, mod, env, fake_sdks, mocker):
        mocker.patch.object(mod, "_non_stream", return_value=True)
        mocker.patch.object(mod, "_stream", return_value=False)
        assert mod.main() == 1

    def test_import_inserts_repo_root(self, mocker):
        """Module-level bootstrap re-inserts the repo root on sys.path."""
        import importlib
        from pathlib import Path

        root = str(Path(__file__).resolve().parents[2])
        import scripts.preflight_serve as preflight

        without_root = [p for p in sys.path if p != root]
        mocker.patch.object(sys, "path", without_root)
        importlib.reload(preflight)
        assert root in sys.path
