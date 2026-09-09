"""Coverage for ``inference.modal_serve`` (serverless serving app definition).

Module import exercises the fake-Modal Image chain / App registration; the
decorated class bodies are exercised directly with a stubbed ``VLLMEngine``.
No cloud, no GPU, no network.
"""

import shutil
from pathlib import Path

import pytest

from inference import modal_serve

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MODELS = _REPO_ROOT / "config" / "models.yaml"


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """tmp_path seeded with a copy of the tracked ``config/models.yaml``."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(_SOURCE_MODELS, config_dir / "models.yaml")
    return tmp_path


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


class TestModelServer:
    def test_enter_constructs_engine(self, mocker):
        mocker.patch("inference.modal_serve.VLLMEngine", return_value="ENGINE")
        server = modal_serve.ModelServer()
        server.enter()  # type: ignore[attr-defined]
        assert server._engine == "ENGINE"

    def test_web_returns_create_app_result(self, mocker):
        mocker.patch("inference.modal_serve.create_app", return_value="APP")
        server = modal_serve.ModelServer()
        engine = object()
        server._engine = engine
        assert server.web() == "APP"  # type: ignore[attr-defined]

    def test_enter_then_web_wire_engine_into_app(self, mocker):
        app = mocker.MagicMock()
        mocker.patch("inference.modal_serve.create_app", return_value=app)
        server = modal_serve.ModelServer()
        engine = object()
        server._engine = engine
        assert server.web() is app  # type: ignore[attr-defined]


class TestGpuResolution:
    def test_known_tiers_map_to_modal_specs(self):
        assert modal_serve.gpu_spec_for_tier("a10g-24gb") == "A10G:1"
        assert modal_serve.gpu_spec_for_tier("a100-40gb") == "A100:1"
        assert modal_serve.gpu_spec_for_tier("a100-80gb") == "A100-80GB"
        assert modal_serve.gpu_spec_for_tier("h100-80gb") == "H100:1"

    def test_missing_tier_defaults_to_a10g(self):
        assert modal_serve.gpu_spec_for_tier(None) == "A10G:1"
        assert modal_serve.gpu_spec_for_tier("") == "A10G:1"

    def test_unknown_tier_passes_through_verbatim(self):
        assert modal_serve.gpu_spec_for_tier("tpu-v3") == "tpu-v3"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("SERVING_GPU", "H100:1")
        assert modal_serve.resolve_serving_gpu() == "H100:1"

    def test_base_model_env_selects_non_default_tier(self, registry_dir, monkeypatch):
        # SERVING_BASE_MODEL picks a NON-default model: its tier, not the
        # default key's, drives the GPU resolution.
        monkeypatch.delenv("SERVING_GPU", raising=False)
        monkeypatch.setenv("SERVING_BASE_MODEL", "qwen3-30b-a3b")
        assert modal_serve.resolve_serving_gpu(repo_root=registry_dir) == "H100:1"

    def test_registry_primary_tier(self, registry_dir, monkeypatch):
        monkeypatch.delenv("SERVING_GPU", raising=False)
        monkeypatch.delenv("SERVING_BASE_MODEL", raising=False)
        # qwen3-14b is flagged default; its primary tier is a100-80gb.
        assert modal_serve.resolve_serving_gpu(repo_root=registry_dir) == "A100-80GB"

    def test_missing_registry_falls_back_to_a10g(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SERVING_GPU", raising=False)
        monkeypatch.delenv("SERVING_BASE_MODEL", raising=False)
        assert modal_serve.resolve_serving_gpu(repo_root=tmp_path) == "A10G:1"
