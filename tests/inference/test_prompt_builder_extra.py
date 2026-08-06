"""Coverage for ``inference.prompt_builder``: prompt wrapping, diff/file
helpers, golden-fetch, template rendering, and model/adapter resolution.

Every network call (GitHub raw, GCS golden) is mocked; ``_eval`` is pinned to
this module so patched state resolves deterministically regardless of whether
``evaluation.inference`` is imported elsewhere.
"""

import json
import sys
import types
import urllib.error

import pytest

import training.prompt_loader  # noqa: E402, F401  # heavy (wandb): absorb at collection
from evaluation.schema import EvalInput
from inference import prompt_builder
from inference.config import ServeConfig

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _force_local_state(monkeypatch):
    monkeypatch.setattr(prompt_builder, "_eval", lambda: prompt_builder)


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    prompt_builder.fetch_raw_file.cache_clear()
    yield
    prompt_builder.fetch_raw_file.cache_clear()


@pytest.fixture
def example() -> EvalInput:
    return EvalInput(
        instance_id="django__django-1000",
        repo="django/django",
        issue_body="The bug is in django/db/models/base.py; fix it.",
        base_sha="base123",
        head_sha="head123",
        test_patch="diff --git a/tests/test_models.py b/tests/test_models.py\n@@ -1 +1 @@",
        gold_patch="",
        fail_to_pass=["tests.test_models::test_x"],
        pass_to_pass=[],
        repo_domain="github.com",
    )


class TestNoThinkWrap:
    def test_cached_tokenizer(self, monkeypatch):
        tok = types.SimpleNamespace()
        tok.apply_chat_template = lambda *a, **k: "WRAPPED"
        monkeypatch.setitem(prompt_builder._TOKENIZER_CACHE, "my/hf", tok)
        assert prompt_builder.no_think_wrap("my/hf", "hello") == "WRAPPED"

    def test_loads_tokenizer_on_miss(self, mocker):
        tok = types.SimpleNamespace()
        tok.apply_chat_template = lambda *a, **k: "FROM_HF"
        fake_tr = types.ModuleType("transformers")
        fake_tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=staticmethod(lambda hf: tok))
        mocker.patch.dict(sys.modules, {"transformers": fake_tr})
        mocker.patch.dict(prompt_builder._TOKENIZER_CACHE, {}, clear=True)
        assert prompt_builder.no_think_wrap("my/hf2", "hello") == "FROM_HF"
        assert "my/hf2" in prompt_builder._TOKENIZER_CACHE


class TestFilesFromDiff:
    def test_empty_patch(self):
        assert prompt_builder._files_from_diff("") == []

    def test_extracts_b_sides(self):
        patch = "diff --git a/x/a.py b/x/a.py\n---\ndiff --git a/src/main.py b/src/main.py\n"
        assert prompt_builder._files_from_diff(patch) == ["x/a.py", "src/main.py"]


class TestFetchRawFile:
    def test_fetches_and_decodes(self, mocker):
        class _Resp:
            def read(self):
                return b"line1\nline2"

        class _Ctx:
            def __enter__(self):
                return _Resp()

            def __exit__(self, *args):
                return None

        mocker.patch("inference.prompt_builder.urllib.request.urlopen", return_value=_Ctx())
        assert prompt_builder.fetch_raw_file("owner/repo", "sha", "src/a.py") == "line1\nline2"

    def test_network_error_returns_none(self, mocker):
        mocker.patch(
            "inference.prompt_builder.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        )
        assert prompt_builder.fetch_raw_file("owner/repo", "sha", "a.py") is None


class TestFileSnippets:
    def test_dedup_skips_and_truncates(self, mocker):
        def fake_fetch(repo, base_sha, path):
            if path == "missing.py":
                return None
            if path == "b.py":
                return "short\nfile"  # below max_lines -> no truncation
            return "\n".join(f"line{i}" for i in range(6))

        mocker.patch.object(prompt_builder, "_fetch_raw_file", side_effect=fake_fetch)
        out = prompt_builder.file_snippets(
            "r",
            "s",
            ["missing.py", "a.py", "a.py", "", "b.py", "c.py", "d.py"],
            max_files=3,
            max_lines=3,
        )
        assert [o["path"] for o in out] == ["a.py", "b.py", "c.py"]
        assert out[0]["content"] == "line0\nline1\nline2\n# ... (file truncated)"
        assert out[1]["content"] == "short\nfile"
        assert out[2]["content"].endswith("(file truncated)")


class TestEnsureGolden:
    def test_existing_file_returned(self, monkeypatch, tmp_path):
        dst = tmp_path / "data" / "d1" / "swebench" / "golden.jsonl"
        dst.parent.mkdir(parents=True)
        dst.write_text("x\n")
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert prompt_builder._ensure_golden("d1") == dst

    def test_downloads_when_missing(self, mocker, monkeypatch, tmp_path):
        grabbed: dict = {}
        mocker.patch(
            "inference.prompt_builder.urllib.request.urlretrieve",
            side_effect=lambda url, dst: grabbed.update(url=url, dst=dst),
        )
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        dst = prompt_builder._ensure_golden("d2")
        assert dst == tmp_path / "data" / "d2" / "swebench" / "golden.jsonl"
        assert "swe-qwen-datasets/datasets/d2/swebench/golden.jsonl" in grabbed["url"]

    def test_fetch_failure_falls_back(self, mocker, monkeypatch, tmp_path):
        mocker.patch(
            "inference.prompt_builder.urllib.request.urlretrieve",
            side_effect=urllib.error.URLError("offline"),
        )
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert prompt_builder._ensure_golden("d3") == prompt_builder._GOLDEN_PATH


def _golden_lines():
    return [
        json.dumps({"repo": "r/a", "patch_diff": "D\nE\nF", "instance_id": "i1"}),
        "not json {",
        json.dumps({"repo": "r/a", "instance_id": "i2"}),
        json.dumps({"repo": "r/b", "patch_diff": "X\nY\nZ\nW", "instance_id": "i3"}),
    ]


class TestGoldenPatches:
    def test_builds_index_and_reuses_it(self, monkeypatch, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text("\n".join(_golden_lines()), encoding="utf-8")
        monkeypatch.setattr(prompt_builder, "_GOLDEN_SOURCE", golden)
        monkeypatch.setattr(prompt_builder, "_GOLDEN_INDEX", None)
        patches = prompt_builder.golden_patches("r/a", exclude_instance_id="i1")
        assert patches == []  # the only r/a example is the excluded instance
        # second call skips the file (index already built)
        assert prompt_builder.golden_patches("r/b") == ["X\nY\nZ\nW"]

    def test_truncates_long_diffs(self, monkeypatch, tmp_path):
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            json.dumps({"repo": "r/x", "patch_diff": "LINE1\nLINE2\nLINE3", "instance_id": "i1"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(prompt_builder, "_GOLDEN_SOURCE", golden)
        monkeypatch.setattr(prompt_builder, "_GOLDEN_INDEX", None)
        assert prompt_builder.golden_patches("r/x", max_lines=2) == [
            "LINE1\nLINE2\n# ... (diff truncated)"
        ]

    def test_max_examples_bounds(self, monkeypatch, tmp_path):
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            "\n".join(
                json.dumps({"repo": "r/9", "patch_diff": "P-1", "instance_id": f"e{i}"})
                for i in range(5)
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(prompt_builder, "_GOLDEN_SOURCE", golden)
        monkeypatch.setattr(prompt_builder, "_GOLDEN_INDEX", None)
        assert len(prompt_builder.golden_patches("r/9", max_examples=2)) == 2

    def test_missing_file_resets_index(self, monkeypatch, tmp_path):
        monkeypatch.setattr(prompt_builder, "_GOLDEN_SOURCE", tmp_path / "none.jsonl")
        monkeypatch.setattr(prompt_builder, "_GOLDEN_INDEX", None)
        assert prompt_builder.golden_patches("r/z") == []
        assert prompt_builder._GOLDEN_INDEX == {}


class TestRenderPatchPrompt:
    def test_chat_uses_golden_patches(self, monkeypatch, example):
        monkeypatch.setattr(prompt_builder, "_golden_patches", lambda *a, **k: ["GOLD1", "GOLD2"])
        out = prompt_builder.render_patch_prompt(example)
        assert example.issue_body in out
        assert "GOLD1" in out
        assert "### Test Files" in out
        assert "- `tests/test_models.py`" in out

    def test_chat_with_explicit_patches_skips_golden(self, monkeypatch, example):
        def boom(*a, **k):
            raise AssertionError("golden must not be touched")

        monkeypatch.setattr(prompt_builder, "_golden_patches", boom)
        out = prompt_builder.render_patch_prompt(example, example_patches=["RAW"])
        assert "RAW" in out

    def test_include_file_contents_seeds_candidates_and_snippets(self, monkeypatch, example):
        seen: dict = {}

        def fake_snippets(repo, base_sha, paths, **kwargs):
            seen["paths"] = paths
            return [{"path": paths[0], "content": "fn main() {}\n"}]

        def fake_golden(*a, **k):
            return ["G"]

        monkeypatch.setattr(prompt_builder, "_file_snippets", fake_snippets)
        monkeypatch.setattr(prompt_builder, "_golden_patches", fake_golden)
        rich = example.model_copy(
            update={"metadata": {"context_files": ["src/lib.rs"], "language": "Rust"}}
        )
        out = prompt_builder.render_patch_prompt(
            rich, template_name="chat", include_file_contents=True
        )
        assert "### File Contents" in out
        assert "src/lib.rs" in out
        assert seen["paths"] == ["src/lib.rs", "tests/test_models.py"]

    def test_issue_body_paths_seed_candidates(self, monkeypatch, example):
        bare = example.model_copy(update={"metadata": {}})
        seen: dict = {}
        monkeypatch.setattr(
            prompt_builder,
            "_file_snippets",
            lambda repo, base_sha, paths, **kwargs: seen.update(paths=paths) or [],
        )
        monkeypatch.setattr(prompt_builder, "_golden_patches", lambda *a, **k: [])
        prompt_builder.render_patch_prompt(bare, template_name="user", include_file_contents=True)
        assert seen["paths"] == ["django/db/models/base.py", "tests/test_models.py"]

    def test_system_template(self, example):
        out = prompt_builder.render_patch_prompt(example, template_name="system")
        assert "You are an expert Python developer" in out
        assert "Follow the repository's existing code style" in out

    def test_user_template(self, example):
        out = prompt_builder.render_patch_prompt(example, template_name="user")
        assert f"## Issue: {example.instance_id}" in out
        assert "- **Repository:** django/django" in out

    def test_assistant_template_renders_empty_sections(self, example, caplog):
        out = prompt_builder.render_patch_prompt(example, template_name="assistant")
        assert "### Analysis" in out
        assert "### Code Changes" in out

    def test_unknown_template_raises(self, example):
        with pytest.raises(ValueError, match="unknown prompt template"):
            prompt_builder.render_patch_prompt(example, template_name="bogus")


class TestResolveHfId:
    def test_full_hf_id_passthrough(self):
        assert prompt_builder.resolve_hf_id("Qwen/Qwen3-14B") == "Qwen/Qwen3-14B"

    def test_missing_pyyaml_returns_default(self, mocker):
        mocker.patch.dict(sys.modules, {"yaml": None})
        assert prompt_builder.resolve_hf_id("qwen3-99b") == prompt_builder._DEFAULT_HF_ID

    def test_missing_registry_file_returns_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert prompt_builder.resolve_hf_id("qwen3-14b") == prompt_builder._DEFAULT_HF_ID

    def test_registry_entry(self, monkeypatch, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "models.yaml").write_text(
            "models:\n  qwen3-14b:\n    hf_id: OWNER/MODEL\n", encoding="utf-8"
        )
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert prompt_builder.resolve_hf_id("qwen3-14b") == "OWNER/MODEL"


class TestResolveAdapterPath:
    def test_local_checkpoint_first(self, monkeypatch, tmp_path):
        local = tmp_path / "models" / "checkpoints" / "baseline_14b"
        local.mkdir(parents=True)
        (local / "adapter_config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        got = prompt_builder.resolve_adapter_path("baseline_14b", config=types.SimpleNamespace())  # type: ignore[arg-type]
        assert got == str(local)

    def test_relative_checkpoint_when_absolute_missing(self, monkeypatch, tmp_path):
        from pathlib import Path

        rel = tmp_path / "models" / "checkpoints" / "v2"
        rel.mkdir(parents=True)
        (rel / "adapter_config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path / "no-models")
        got = prompt_builder.resolve_adapter_path("v2", config=types.SimpleNamespace())  # type: ignore[arg-type]
        assert got == str(Path("models/checkpoints") / "v2")
        assert got is not None and (Path(got) / "adapter_config.json").is_file()

    def test_wandb_artifact_download(self, mocker, monkeypatch, tmp_path):
        downloads: list[str] = []

        class _Api:
            def artifact(self, ref):
                return types.SimpleNamespace(
                    download=lambda: downloads.append(ref) or "/adapters/v"
                )

        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=_Api)})
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        cfg = ServeConfig()
        expected = (
            f"{cfg.wandb_entity}/{cfg.wandb_project}/"
            f"{cfg.lora_artifact_pattern.format(variant='baseline_14b')}:latest"
        )
        got = prompt_builder.resolve_adapter_path("baseline_14b", config=cfg)  # type: ignore[arg-type]
        assert got == "/adapters/v"
        assert downloads == [expected]

    def test_failure_falls_back_to_base(self, mocker, monkeypatch, tmp_path):
        class _BadApi:
            def artifact(self, ref):
                raise RuntimeError("no artifact")

        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=_BadApi)})
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert (
            prompt_builder.resolve_adapter_path(
                "v9",
                config=types.SimpleNamespace(
                    lora_artifact_pattern="{v}", wandb_entity="e", wandb_project="p"
                ),  # type: ignore[arg-type]
            )
            is None
        )

    def test_no_config_instantiates_eval_config(self, mocker, monkeypatch, tmp_path):
        constructed: list = []
        mocker.patch(
            "evaluation.config.EvalConfig",
            side_effect=lambda: constructed.append(True) or types.SimpleNamespace(),
        )

        class _BadApi:
            def artifact(self, ref):
                raise RuntimeError("nope")

        mocker.patch.dict(sys.modules, {"wandb": types.SimpleNamespace(Api=_BadApi)})
        monkeypatch.setattr(prompt_builder, "_REPO_ROOT", tmp_path)
        assert prompt_builder.resolve_adapter_path("anything") is None
        assert constructed == [True]
