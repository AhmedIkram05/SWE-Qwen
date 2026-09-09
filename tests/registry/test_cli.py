"""Tests for ``registry/cli.py`` — `model add` / `model list`.

Fully offline: the HTTP layer is mocked by monkeypatching ``registry.cli._client``
(the client the CLI uses), so the real validation/write logic runs. The repo root
is monkeypatched to a tmp_path seeded with the tracked ``config/models.yaml``
(same approach as ``tests/registry/test_loader.py``).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

import registry.cli as registry_cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MODELS = _REPO_ROOT / "config" / "models.yaml"


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """tmp_path seeded with a copy of the tracked ``config/models.yaml``."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(_SOURCE_MODELS, config_dir / "models.yaml")
    return tmp_path


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:  # noqa: PLR2004
            raise httpx.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no payload on fake response")
        return self._payload


class _FakeClient:
    """Stands in for ``registry.cli._client``: HEAD + two resolve-file GETs."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        tokenizer: dict[str, Any] | None = None,
        head_error: Exception | None = None,
        head_status: int = 200,
    ) -> None:
        self.config = config or {}
        self.tokenizer = tokenizer or {}
        self.head_error = head_error
        self.head_status = head_status

    def head(self, url: str) -> _FakeResponse:
        if self.head_error is not None:
            raise self.head_error
        return _FakeResponse(self.head_status)

    def get(self, url: str) -> _FakeResponse:
        if url.endswith("/config.json"):
            return _FakeResponse(200, self.config)
        if url.endswith("/tokenizer_config.json"):
            return _FakeResponse(200, self.tokenizer)
        return _FakeResponse(404)


@pytest.fixture
def run(monkeypatch, registry_dir) -> Callable[..., Any]:
    """CliRunner with repo root pointed at tmp_path and a fake HF client."""
    monkeypatch.setattr(registry_cli, "_REPO_ROOT", registry_dir)
    runner = CliRunner()

    def invoke(client: _FakeClient | None = None, *args: str) -> Any:
        monkeypatch.setattr(registry_cli, "_client", client or _FakeClient())
        return runner.invoke(registry_cli.app, list(args))

    return invoke


def _read_user(registry_dir: Path) -> dict[str, Any]:
    loaded = yaml.safe_load((registry_dir / "config" / "models.user.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def _good_client(**overrides: Any) -> _FakeClient:
    config = {"max_position_embeddings": 4096, "model_type": "llama"}
    tokenizer = {"chat_template": "{{ messages }}"}
    if "config" in overrides:
        config.update(overrides.pop("config"))
    if "tokenizer" in overrides:
        tokenizer.update(overrides.pop("tokenizer"))
    return _FakeClient(config=config, tokenizer=tokenizer, **overrides)


class TestAddValidation:
    def test_reachability_failure_writes_nothing(self, run, registry_dir):
        result = run(
            _FakeClient(head_error=httpx.ConnectError("network down")),
            "add",
            "org/m",
            "--name",
            "m1",
        )
        assert result.exit_code == 1
        assert not (registry_dir / "config" / "models.user.yaml").exists()

    def test_reachability_404_writes_nothing(self, run, registry_dir):
        result = run(_FakeClient(head_status=404), "add", "org/m", "--name", "m1")
        assert result.exit_code == 1
        assert not (registry_dir / "config" / "models.user.yaml").exists()

    def test_context_probe_uses_max_position_embeddings(self, run, registry_dir):
        result = run(_good_client(), "add", "org/llama-1b", "--name", "m1")
        assert result.exit_code == 0, result.output
        entry = _read_user(registry_dir)["models"]["m1"]
        assert entry["context_window"] == 4096
        assert entry["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_explicit_context_wins_over_probe(self, run, registry_dir):
        result = run(_good_client(), "add", "org/m", "--name", "m1", "--context", "2048")
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["m1"]["context_window"] == 2048

    def test_chat_template_missing_requires_flag(self, run, registry_dir):
        result = run(
            _FakeClient(config={"max_position_embeddings": 4096, "model_type": "llama"}),
            "add",
            "org/m",
            "--name",
            "m1",
        )
        assert result.exit_code == 1
        assert "chat_template" in result.output
        assert not (registry_dir / "config" / "models.user.yaml").exists()

    def test_allow_no_template_proceeds(self, run, registry_dir):
        result = run(
            _FakeClient(config={"max_position_embeddings": 4096, "model_type": "llama"}),
            "add",
            "org/m",
            "--name",
            "m1",
            "--allow-no-template",
        )
        assert result.exit_code == 0
        assert "m1" in _read_user(registry_dir)["models"]

    def test_thinking_markers_set_no_think_false(self, run, registry_dir):
        result = run(
            _good_client(tokenizer={"chat_template": "t", "enable_thinking": True}),
            "add",
            "org/m",
            "--name",
            "m1",
        )
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["m1"]["prompt_behavior"]["no_think"] is False

    def test_default_no_think_true(self, run, registry_dir):
        result = run(_good_client(), "add", "org/m", "--name", "m1")
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["m1"]["prompt_behavior"]["no_think"] is True

    def test_clobber_refused_without_force(self, run, registry_dir):
        result = run(_good_client(), "add", "org/m", "--name", "qwen3-14b")
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert not (registry_dir / "config" / "models.user.yaml").exists()

    def test_force_overlays_existing_key(self, run, registry_dir):
        result = run(_good_client(), "add", "org/m2", "--name", "qwen3-14b", "--force")
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["qwen3-14b"]["hf_id"] == "org/m2"

    def test_explicit_target_modules_list(self, run, registry_dir):
        result = run(
            _good_client(), "add", "org/m", "--name", "m1", "--target-modules", "q_proj, v_proj"
        )
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["m1"]["target_modules"] == ["q_proj", "v_proj"]

    def test_auto_target_modules_unknown_arch_defaults(self, run, registry_dir):
        result = run(
            _good_client(config={"model_type": "weird-arch"}), "add", "org/m", "--name", "m1"
        )
        assert result.exit_code == 0
        assert _read_user(registry_dir)["models"]["m1"]["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_reports_stages_that_pick_it_up(self, run):
        result = run(_good_client(), "add", "org/m", "--name", "m1")
        assert result.exit_code == 0
        assert "stages picking this up" in result.output


class TestWritesOnlyUserFile:
    def test_base_models_yaml_untouched(self, run, registry_dir):
        base_before = (registry_dir / "config" / "models.yaml").read_text()
        result = run(_good_client(), "add", "org/m", "--name", "m1")
        assert result.exit_code == 0
        assert (registry_dir / "config" / "models.yaml").read_text() == base_before
        assert (registry_dir / "config" / "models.user.yaml").exists()

    def test_user_file_appends_alongside_existing_entries(self, run, registry_dir):
        (registry_dir / "config" / "models.user.yaml").write_text(
            "models:\n  first:\n    hf_id: org/one\n    context_window: 512\n"
            "    target_modules: [q_proj]\n",
            encoding="utf-8",
        )
        result = run(_good_client(), "add", "org/two", "--name", "second")
        assert result.exit_code == 0
        models = _read_user(registry_dir)["models"]
        assert set(models) == {"first", "second"}
        assert models["first"]["hf_id"] == "org/one"


class TestModelList:
    def test_list_shows_source_column(self, run, registry_dir):
        (registry_dir / "config" / "models.user.yaml").write_text(
            "models:\n  tiny-llm:\n    hf_id: org/tiny\n    context_window: 1024\n"
            "    target_modules: [q_proj]\n",
            encoding="utf-8",
        )
        result = run(_FakeClient(), "list")
        assert result.exit_code == 0, result.output
        lines = result.output.strip().splitlines()
        assert lines[0].split() == ["key", "hf_id", "context_window", "source"]
        columns = [line.split() for line in lines[1:]]
        tracked = next(c for c in columns if c[0] == "qwen3-14b")
        assert tracked[1] == "Qwen/Qwen3-14B"
        assert tracked[-1] == "tracked"
        user = next(c for c in columns if c[0] == "tiny-llm")
        assert user[-1] == "user"
