"""Tests for ``registry/loader.py`` — the merged registry loading path.

No network: ``load_models`` is exercised against copies of the tracked
``config/models.yaml`` inside ``tmp_path`` fixtures.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from registry.loader import default_model_key, get_model_config, load_models

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MODELS = _REPO_ROOT / "config" / "models.yaml"


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """tmp_path seeded with a copy of the tracked ``config/models.yaml``."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(_SOURCE_MODELS, config_dir / "models.yaml")
    return tmp_path


def _write_user(root: Path, content: str) -> None:
    (root / "config" / "models.user.yaml").write_text(content, encoding="utf-8")


class TestLoadModels:
    def test_missing_user_file_is_fine(self, registry_dir):
        specs = load_models(repo_root=registry_dir)
        assert specs["qwen3-14b"].hf_id == "Qwen/Qwen3-14B"
        assert specs["qwen3-14b"].context_window == 32768
        assert len(specs["qwen3-14b"].target_modules) == 7

    def test_overlay_wins_on_conflict(self, registry_dir):
        _write_user(registry_dir, "models:\n  qwen3-14b:\n    context_window: 4096\n")
        spec = load_models(repo_root=registry_dir)["qwen3-14b"]
        assert spec.context_window == 4096  # overridden
        assert spec.hf_id == "Qwen/Qwen3-14B"  # untouched sibling field

    def test_overlay_deep_merges_per_key(self, registry_dir):
        _write_user(
            registry_dir,
            ("models:\n  qwen3-14b:\n    gpu_mapping:\n      primary: h100-80gb\n"),
        )
        spec = load_models(repo_root=registry_dir)["qwen3-14b"]
        assert spec.gpu_mapping["primary"] == "h100-80gb"  # overridden
        assert spec.gpu_mapping["fallback"] == "a10g-24gb"  # sibling kept
        assert spec.context_window == 32768  # sibling kept

    def test_overlay_adds_new_model_with_defaults(self, registry_dir):
        _write_user(
            registry_dir,
            (
                "models:\n"
                "  tiny-llm:\n"
                "    hf_id: HuggingFaceTB/SmolLM2-135M\n"
                "    context_window: 2048\n"
                "    target_modules: [q_proj]\n"
            ),
        )
        specs = load_models(repo_root=registry_dir)
        assert "tiny-llm" in specs
        assert "qwen3-14b" in specs  # base models kept
        tiny = specs["tiny-llm"]
        assert tiny.context_window == 2048
        assert tiny.gpu_mapping == {"primary": "a10g-24gb", "fallback": "a10g-24gb"}
        assert tiny.serving_hf_id is None
        assert tiny.serving_quantization is None

    def test_invalid_entry_raises_naming_model(self, registry_dir):
        _write_user(registry_dir, "models:\n  broken-model:\n    context_window: 1024\n")
        with pytest.raises(ValidationError, match="broken-model"):
            load_models(repo_root=registry_dir)

    def test_get_model_config_unknown_key_raises(self, registry_dir):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent", repo_root=registry_dir)


class TestCacheInvalidation:
    def test_rewrite_of_user_file_is_picked_up(self, registry_dir):
        assert load_models(repo_root=registry_dir)["qwen3-14b"].context_window == 32768
        _write_user(registry_dir, "models:\n  qwen3-14b:\n    context_window: 4096\n")
        assert load_models(repo_root=registry_dir)["qwen3-14b"].context_window == 4096
        time.sleep(0.02)  # ensure a distinct mtime (mtime-ns is the cache key)
        _write_user(registry_dir, "models:\n  qwen3-14b:\n    context_window: 8192\n")
        assert load_models(repo_root=registry_dir)["qwen3-14b"].context_window == 8192


class TestDefaultModelKey:
    def test_returns_qwen3_14b(self):
        assert default_model_key() == "qwen3-14b"

    def test_respects_user_overlay(self, registry_dir):
        _write_user(
            registry_dir,
            (
                "models:\n"
                "  qwen3-14b:\n"
                "    default: false\n"
                "  tiny-llm:\n"
                "    default: true\n"
                "    hf_id: HuggingFaceTB/SmolLM2-135M\n"
                "    context_window: 2048\n"
                "    target_modules: [q_proj]\n"
            ),
        )
        assert default_model_key(repo_root=registry_dir) == "tiny-llm"
