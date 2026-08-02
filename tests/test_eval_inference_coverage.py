"""Coverage tests for ``evaluation.inference``.

Covers the pure helpers exhaustively (``extract_patch``, ``_files_from_diff``,
``render_patch_prompt``, ``resolve_hf_id``, ``resolve_adapter_path``),
``_get_llm`` caching, and the Modal-decorated ``generate_patches_batch`` via
``Function.local()`` with a fake vLLM (no model downloads, no containers).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from evaluation.config import EvalConfig
from evaluation.inference import (
    _DEFAULT_HF_ID,
    _files_from_diff,
    _get_llm,
    extract_patch,
    generate_patches_batch,
    render_patch_prompt,
    resolve_adapter_path,
    resolve_hf_id,
)
from evaluation.schema import EvalInput

# ── helpers / fixtures ─────────────────────────────────────────────────────


def _example(**overrides: Any) -> EvalInput:
    base: dict[str, Any] = {
        "instance_id": "django__django-10554",
        "repo": "django/django",
        "issue_body": "BooleanField crashes on None.",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "test_patch": ("diff --git a/tests/test_models.py b/tests/test_models.py\n@@ -1 +1 @@\n"),
        "fail_to_pass": ["tests/test_models.py::test_x"],
        "pass_to_pass": ["tests/test_models.py::test_y"],
        "repo_domain": "python",
    }
    base.update(overrides)
    return EvalInput(**base)


@pytest.fixture(autouse=True)
def _clear_llm_cache() -> Iterator[None]:
    import evaluation.inference as inf

    inf._LLM_CACHE.clear()
    yield
    inf._LLM_CACHE.clear()


@pytest.fixture
def fake_vllm(monkeypatch) -> Iterator[dict]:
    """Install a fake ``vllm`` package (SamplingParams/LoRARequest/LLM)."""
    recorded: dict = {}

    vllm = types.ModuleType("vllm")
    vllm.__spec__ = importlib.util.spec_from_loader("vllm", loader=None)

    class LLM:
        def __init__(self, model, enable_lora, gpu_memory_utilization):
            recorded["llm"] = {
                "model": model,
                "enable_lora": enable_lora,
                "gpu_memory_utilization": gpu_memory_utilization,
            }
            recorded["llm_instances"] = recorded.get("llm_instances", 0) + 1

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    lora = types.ModuleType("vllm.lora")
    lora.__spec__ = importlib.util.spec_from_loader("vllm.lora", loader=None)
    req = types.ModuleType("vllm.lora.request")
    req.__spec__ = importlib.util.spec_from_loader("vllm.lora.request", loader=None)

    class LoRARequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    req.LoRARequest = LoRARequest

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req)
    yield recorded


def _fake_wandb(api_cls: type) -> types.ModuleType:
    """Build a fake ``wandb`` module exposing only ``Api``."""
    mod = types.ModuleType("wandb")
    mod.Api = api_cls
    return mod


class _GenOutput:
    def __init__(self, text: str):
        self.outputs = [types.SimpleNamespace(text=text)]


class _FakeLLM:
    def __init__(self):
        self.calls: list[tuple] = []

    def generate(self, prompts, sampling_params, lora_request=None):
        self.calls.append((list(prompts), sampling_params, lora_request))
        suffix = "\n```patch\n" + "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n" + "\n```"
        return [_GenOutput(p + suffix) for p in prompts]


# ── extract_patch ──────────────────────────────────────────────────────────


class TestExtractPatch:
    def test_diff_fenced_block(self):
        text = "reasoning...\n```diff\ndiff --git a/x.py b/x.py\n@@ -1 +1 @@\n```\nafter"
        assert extract_patch(text) == "diff --git a/x.py b/x.py\n@@ -1 +1 @@"

    def test_diff_fenced_takes_last(self):
        text = "```diff\ndiff --git a/one b/one\n```\n```diff\ndiff --git a/two b/two\n```"
        assert extract_patch(text) == "diff --git a/two b/two"

    def test_diff_fenced_starting_with_plusplus(self):
        text = "```diff\n+++ b/x.py\n@@ -1 +1 @@\n```"
        assert extract_patch(text) == "+++ b/x.py\n@@ -1 +1 @@"

    def test_generic_fence_with_language_tag(self):
        text = "```python\nprint('x')\n```\n```\ndiff --git a/a b/b\n```"
        assert extract_patch(text) == "diff --git a/a b/b"

    def test_xml_fence_fallback_to_generic(self):
        text = "```xml\n<changes>\n</changes>\n```\n```\ndiff --git a/a.py b/a.py\n@@ -1 +1 @@\n```"
        assert extract_patch(text) == "diff --git a/a.py b/a.py\n@@ -1 +1 @@"

    def test_no_fences_diff_at_end(self):
        text = "some reasoning text\ndiff --git a/x b/y\n@@ -1 +1 @@\n"
        assert extract_patch(text) == "diff --git a/x b/y\n@@ -1 +1 @@"

    def test_no_fences_rfind_takes_last(self):
        text = "diff --git a/one b/one\nnoise\ndiff --git a/two b/two\n@@ -1 +1 @@\n"
        assert extract_patch(text) == "diff --git a/two b/two\n@@ -1 +1 @@"

    def test_diff_fence_with_non_diff_body_returns_whole(self):
        # ```diff fence whose body is not diff-like: no fallback match, no bare
        # "diff --git" → the entire (stripped) text is returned as-is.
        text = "hello\n```diff\nnot a real diff\n```\nworld"
        assert extract_patch(text) == text.strip()

    def test_no_diff_anywhere_returns_stripped(self):
        assert extract_patch("  just some words  ") == "just some words"

    def test_empty(self):
        assert extract_patch("") == ""

    def test_whitespace(self):
        assert extract_patch("   \n  ") == ""


# ── _files_from_diff ───────────────────────────────────────────────────────


class TestFilesFromDiff:
    def test_empty(self):
        assert _files_from_diff("") == []

    def test_non_diff(self):
        assert _files_from_diff("no diff markers here") == []

    def test_single_file(self):
        assert _files_from_diff("diff --git a/src/app.py b/src/app.py\n@@ -1 +1 @@\n") == [
            "src/app.py"
        ]

    def test_multi_file(self):
        patch = (
            "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n"
            "diff --git a/b.py b/b.py\n@@ -1 +1 @@\n"
            "diff --git a/c.py b/d.py\n@@ -1 +1 @@\n"
        )
        assert _files_from_diff(patch) == ["a.py", "b.py", "d.py"]


# ── render_patch_prompt ────────────────────────────────────────────────────


class TestRenderPatchPrompt:
    def test_chat_default_includes_all_context(self):
        ex = _example(
            metadata={
                "language": "Python",
                "task_description": "fix the crash",
                "style_guide": "keep it simple",
                "context_files": ["app/models.py", "app/views.py"],
            }
        )
        prompt = render_patch_prompt(ex)
        assert "django__django-10554" in prompt
        assert "BooleanField crashes on None." in prompt
        assert "django/django" in prompt
        assert "python" in prompt.lower()
        assert "fix the crash" in prompt
        assert "keep it simple" in prompt
        assert "`app/models.py`" in prompt
        assert "tests/test_models.py" in prompt
        assert "### Response" in prompt

    def test_metadata_defaults(self):
        ex = _example()
        prompt = render_patch_prompt(ex, template_name="system")
        assert "You are an expert Python developer" in prompt
        assert "Fix the bug described in the issue below." in prompt
        assert "existing code style and test conventions" in prompt

    def test_user_template(self):
        prompt = render_patch_prompt(_example(), template_name="user")
        assert "### Relevant Files" in prompt
        assert "### Test Files" in prompt
        assert "Generate a patch" in prompt

    def test_user_template_empty_context_files(self):
        prompt = render_patch_prompt(_example(), template_name="user")
        assert "app/models.py" not in prompt  # no context_files
        assert "tests/test_models.py" in prompt  # test_files still rendered

    def test_system_template(self):
        prompt = render_patch_prompt(_example(), template_name="system")
        assert "You are an expert Python developer" in prompt
        assert "Style Guide" in prompt

    def test_assistant_template_warns(self, caplog):
        with caplog.at_level("WARNING", logger="evaluation.inference"):
            prompt = render_patch_prompt(_example(), template_name="assistant")
        assert "assistant template is a training-target template" in caplog.text
        assert "### Code Changes" in prompt

    def test_custom_template_dir(self, tmp_path):
        (tmp_path / "system.j2").write_text("SYS {{ language }}")
        (tmp_path / "user.j2").write_text("USR {{ issue_title }}")
        (tmp_path / "chat.j2").write_text("CUSTOM {{ system_prompt }} / {{ user_prompt }}")
        prompt = render_patch_prompt(_example(), template_name="chat", template_dir=tmp_path)
        # chat.j2 uses system_prompt + messages; user_prompt is only a fallback.
        assert prompt == "CUSTOM SYS Python / " and "USR django__django-10554" not in prompt

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="unknown prompt template"):
            render_patch_prompt(_example(), template_name="nope")


# ── resolve_hf_id ──────────────────────────────────────────────────────────


class TestResolveHfId:
    def test_full_hf_id_returned_as_is(self):
        assert resolve_hf_id("Qwen/Qwen3-14B") == "Qwen/Qwen3-14B"

    def test_registry_key(self):
        assert resolve_hf_id("qwen3-14b") == "Qwen/Qwen3-14B"

    def test_unknown_registry_key_falls_back(self):
        assert resolve_hf_id("not-a-model") == _DEFAULT_HF_ID

    def test_missing_registry_file(self, monkeypatch):
        import evaluation.inference as inf

        monkeypatch.setattr(inf, "_REPO_ROOT", pytest.importorskip("pathlib").Path("/nonexistent"))
        assert resolve_hf_id("qwen3-14b") == _DEFAULT_HF_ID

    def test_empty_registry(self, tmp_path, monkeypatch):
        import evaluation.inference as inf

        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "models.yaml").write_text("models: {}\n")
        monkeypatch.setattr(inf, "_REPO_ROOT", tmp_path)
        assert resolve_hf_id("qwen3-14b") == _DEFAULT_HF_ID

    def test_null_registry(self, tmp_path, monkeypatch):
        import evaluation.inference as inf

        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "models.yaml").write_text("models: null\n")
        monkeypatch.setattr(inf, "_REPO_ROOT", tmp_path)
        assert resolve_hf_id("qwen3-14b") == _DEFAULT_HF_ID

    def test_missing_pyyaml(self, monkeypatch):

        monkeypatch.setitem(sys.modules, "yaml", None)
        assert resolve_hf_id("qwen3-14b") == _DEFAULT_HF_ID


# ── resolve_adapter_path ───────────────────────────────────────────────────


def _cfg(**overrides) -> EvalConfig:
    return EvalConfig().model_copy(
        update={
            "lora_artifact_pattern": "model-qwen3-14b-{variant}",
            "wandb_entity": "ent",
            "wandb_project": "proj",
            **overrides,
        }
    )


class TestResolveAdapterPath:
    def test_local_adapter_found(self, tmp_path, monkeypatch):
        variant = "v-test-local"
        candidate = tmp_path / "models" / "checkpoints" / variant
        candidate.mkdir(parents=True)
        (candidate / "adapter_config.json").write_text("{}")
        monkeypatch.chdir(tmp_path)
        out = resolve_adapter_path(variant, _cfg())
        # Second local candidate is cwd-relative: Path("models/checkpoints")/variant.
        assert out == str(Path("models/checkpoints") / variant)

    def test_wandb_download(self, tmp_path, monkeypatch):
        dl_dir = tmp_path / "artifact_dl"

        class FakeArtifact:
            def download(self):
                dl_dir.mkdir(parents=True, exist_ok=True)
                return str(dl_dir)

        class FakeApi:
            def __init__(self):
                self.refs = []

            def artifact(self, ref):
                self.refs.append(ref)
                return FakeArtifact()

        fake_wandb = _fake_wandb(FakeApi)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        out = resolve_adapter_path("v-remote", _cfg())
        assert out == str(dl_dir)

    def test_wandb_failure_returns_none(self, monkeypatch, caplog):
        class BadApi:
            def __init__(self):
                raise RuntimeError("no auth")

        monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(BadApi))
        with caplog.at_level("WARNING", logger="evaluation.inference"):
            assert resolve_adapter_path("v-bad", _cfg()) is None
        assert "no LoRA adapter found" in caplog.text

    def test_config_none_builds_default(self, tmp_path, monkeypatch, caplog):
        import evaluation.inference as inf

        class BadApi:
            def __init__(self):
                raise RuntimeError("no auth")

        monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(BadApi))
        # No local adapter in tmp repo root or cwd → wandb path → exception → None.
        monkeypatch.setattr(inf, "_REPO_ROOT", tmp_path)
        monkeypatch.chdir(tmp_path)
        with caplog.at_level("WARNING", logger="evaluation.inference"):
            assert resolve_adapter_path("v-none-cfg") is None
        assert "no LoRA adapter found" in caplog.text

    def test_wandb_artifact_ref_format(self, tmp_path, monkeypatch):
        captured = {}

        class FakeArtifact:
            def download(self):
                return str(tmp_path)

        class FakeApi:
            def artifact(self, ref):
                captured["ref"] = ref
                return FakeArtifact()

        monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(FakeApi))
        assert resolve_adapter_path("mychamp", _cfg()) == str(tmp_path)
        assert captured["ref"] == "ent/proj/model-qwen3-14b-mychamp:latest"


# ── _get_llm ───────────────────────────────────────────────────────────────


class TestGetLLM:
    def test_creates_llm(self, fake_vllm):
        llm = _get_llm("Qwen/Qwen3-14B")
        assert fake_vllm["llm"] == {
            "model": "Qwen/Qwen3-14B",
            "enable_lora": True,  # C1: always True to support both baseline and LoRA variants
            "gpu_memory_utilization": 0.85,
        }
        assert fake_vllm["llm_instances"] == 1
        assert llm is not None

    def test_cache_hit(self, fake_vllm):
        first = _get_llm("m1")
        second = _get_llm("m1")
        assert first is second
        assert fake_vllm["llm_instances"] == 1

    def test_cache_keyed_by_model_only(self, fake_vllm):
        _get_llm("m1")
        _get_llm("m2")
        assert fake_vllm["llm_instances"] == 2  # C1: same model = same LLM regardless of variant

    def test_lora_enabled_with_adapter(self, fake_vllm):
        _get_llm("m1")
        assert fake_vllm["llm"]["enable_lora"] is True  # C1: always True


# ── generate_patches_batch (Modal body via .local) ────────────────────────


class TestGeneratePatchesBatch:
    def test_baseline_batch(self, fake_vllm, monkeypatch):
        import evaluation.inference as inf

        seen = {}
        llm = _FakeLLM()

        def fake_resolve(variant, config):
            seen["resolve"] = (variant, config is not None)

        def fake_get_llm(model_name, adapter_path=None):
            return llm

        monkeypatch.setattr(inf, "resolve_adapter_path", fake_resolve)
        monkeypatch.setattr(inf, "_get_llm", fake_get_llm)

        examples = [_example(), _example(instance_id="inst-2")]
        out = generate_patches_batch.local(
            "qwen3-14b",
            "baseline_14b",
            "chat",
            examples,
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.9,
        )

        assert len(out) == 2
        assert all(p.startswith("diff --git a/x.py b/x.py") for p in out)
        assert seen["resolve"] == ("baseline_14b", True)
        prompts, params, lora_req = llm.calls[0]
        assert len(prompts) == 2
        assert "django__django-10554" in prompts[0]
        assert "inst-2" in prompts[1]
        assert params.kwargs == {"max_tokens": 64, "temperature": 0.2, "top_p": 0.9}
        assert lora_req is None

    def test_with_lora_adapter(self, fake_vllm, monkeypatch):
        import evaluation.inference as inf

        llm = _FakeLLM()

        def fake_resolve(variant, config):
            return "/tmp/lora-adapter"

        def fake_get_llm(model_name, adapter_path=None):
            return llm

        monkeypatch.setattr(inf, "resolve_adapter_path", fake_resolve)
        monkeypatch.setattr(inf, "_get_llm", fake_get_llm)

        out = generate_patches_batch.local("qwen3-14b", "baseline_14b", "chat", [_example()])
        assert len(out) == 1
        prompts, params, lora_req = llm.calls[0]
        assert lora_req is not None
        assert lora_req.kwargs == {
            "lora_name": "qwen3-14b-baseline_14b",
            "lora_int_id": 1,
            "lora_path": "/tmp/lora-adapter",
        }

    def test_extract_applied_to_raw_generation(self, fake_vllm, monkeypatch):
        import evaluation.inference as inf

        class RawLLM:
            def generate(self, prompts, sampling_params, lora_request=None):
                gen = "thinking...\n```patch\ndiff --git a/fix.py b/fix.py\n@@ -1 +1 @@\n```"
                return [_GenOutput(gen)]

        def fake_resolve(variant, config):
            return None

        def fake_get_llm(model_name, adapter_path=None):
            return RawLLM()

        monkeypatch.setattr(inf, "resolve_adapter_path", fake_resolve)
        monkeypatch.setattr(inf, "_get_llm", fake_get_llm)

        out = generate_patches_batch.local("qwen3-14b", "baseline_14b", "chat", [_example()])
        assert out == ["diff --git a/fix.py b/fix.py\n@@ -1 +1 @@"]


# ── C1: shared LLM across variants ─────────────────────────────────────────


class TestSharedLLM:
    def test_same_model_different_variants_return_same_llm(self, fake_vllm):
        """C1: _get_llm caches by model_name only, so different variants share the same LLM."""
        import evaluation.inference as inf

        inf._LLM_CACHE.clear()
        instance_ids: list[int] = []

        class _TrackingLLM:
            def __init__(self, **kw):
                instance_ids.append(id(self))

        class _TrackingSamplingParams:
            def __init__(self, **kw):
                self.kwargs = kw

        fake_vllm_v2 = types.ModuleType("vllm")
        fake_vllm_v2.LLM = _TrackingLLM
        fake_vllm_v2.SamplingParams = _TrackingSamplingParams
        lora2 = types.ModuleType("vllm.lora")
        lora_req2 = types.ModuleType("vllm.lora.request")
        lora_req2.LoRARequest = type(
            "LoRARequest", (), {"__init__": lambda s, **kw: setattr(s, "kwargs", kw)}
        )
        lora2.request = lora_req2

        monkeypatch2 = pytest.MonkeyPatch()
        monkeypatch2.setattr(inf, "resolve_hf_id", lambda _: "fake/model")
        monkeypatch2.setitem(sys.modules, "vllm", fake_vllm_v2)
        monkeypatch2.setitem(sys.modules, "vllm.lora", lora2)
        monkeypatch2.setitem(sys.modules, "vllm.lora.request", lora_req2)

        from evaluation.inference import _get_llm

        # Both calls: same model_name → same LLM instance
        llm_a = _get_llm("qwen3-14b")
        llm_b = _get_llm("qwen3-14b")
        assert llm_a is llm_b
        assert len(instance_ids) == 1  # one LLM constructed

        # Different model_name → different LLM instance
        llm_c = _get_llm("qwen3-30b")
        assert len(instance_ids) == 2

        monkeypatch2.undo()

    def test_generate_patches_batch_has_local_for_testing(self):
        """C3: generate_patches_batch.local delegates to _generate_patches_batch_body."""
        from evaluation.inference import generate_patches_batch

        assert hasattr(generate_patches_batch, "local")
        assert callable(generate_patches_batch.local)
