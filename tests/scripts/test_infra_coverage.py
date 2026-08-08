"""Unit tests for infra/gpu/modal files with 0% coverage.

Tests pure functions and mock-testable wrappers in files that require
GPU/Modal/W&B infrastructure to run end-to-end.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_scripts_module(name: str) -> Any:
    """Load a module from scripts/ (not a package, so use importlib)."""
    path = _PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Pre-mock globally ─────────────────────────────────────────────────────────
# training/modal_train.py and training/local_cli.py import modal/wandb at module
# level and use @app.function() decorators that crash without Modal context.
# Inject mocks into sys.modules BEFORE any test class is loaded.
_MOCK_MODAL = mock.MagicMock()
_MOCK_APP = mock.MagicMock()
_MOCK_APP.function = lambda **_kw: lambda f: f
_MOCK_APP.local_entrypoint = lambda **_kw: lambda f: f
# modal.App("name") must return our mock app, not a fresh MagicMock
_MOCK_MODAL.App = lambda *_a, **_kw: _MOCK_APP
# modal.Image.debian_slim().pip_install() etc. — keep MagicMock auto-chaining
_MOCK_MODAL.Image.debian_slim.return_value.pip_install.return_value = mock.MagicMock()

_MOCK_WANDB = mock.MagicMock()
_MOCK_WANDB.__spec__ = mock.MagicMock()  # so importlib.util.find_spec works
# Force-set mocks into sys.modules — always replace real modules, not just when absent.
# Many test files import modal/wandb at module level, overwriting the mock.
# These force-sets ensure infra tests always see the mock regardless of import order.
if "modal" not in sys.modules:
    sys.modules["modal"] = _MOCK_MODAL
if "wandb" not in sys.modules:
    sys.modules["wandb"] = _MOCK_WANDB


def _import_training(name: str) -> Any:
    """Import from a training module with modal/wandb mocked at sys.modules level."""
    with mock.patch.dict("sys.modules", {"modal": _MOCK_MODAL, "wandb": _MOCK_WANDB}):
        return __import__(f"training.{name}", fromlist=["_"])


# ── training/unsloth_factory.py ────────────────────────────────────────────────


class TestFixEosToken:
    """_fix_eos_token is pure tokenizer-manipulation logic — no GPU needed."""

    def test_none_eos_token(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = None
        _fix_eos_token(tok)
        tok.get_vocab.assert_not_called()

    def test_already_in_vocab(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "<|endoftext|>"
        tok.get_vocab.return_value = {"<|endoftext|>": 151643}
        _fix_eos_token(tok)
        assert tok.eos_token == "<|endoftext|>"

    def test_fixes_placeholder_token(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "<EOS_TOKEN>"
        tok.get_vocab.return_value = {
            "real_token": 42,
            "<|endoftext|>": 151643,
        }
        _fix_eos_token(tok)
        assert tok.eos_token == "<|endoftext|>"
        assert tok.eos_token_id == 151643

    def test_tries_candidates_in_order(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "bogus"
        tok.get_vocab.return_value = {"</s>": 2, "keep": 1}
        _fix_eos_token(tok)
        assert tok.eos_token == "</s>"
        assert tok.eos_token_id == 2

    def test_fallback_decode(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "bogus"
        tok.eos_token_id = 151643
        tok.get_vocab.return_value = {"keep": 1}
        tok.decode.return_value = "<|endoftext|>"
        _fix_eos_token(tok)
        assert tok.eos_token == "<|endoftext|>"

    def test_decode_fails(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "bogus"
        tok.eos_token_id = 151643
        tok.get_vocab.return_value = {"keep": 1}
        tok.decode.side_effect = RuntimeError("decode failed")
        _fix_eos_token(tok)
        assert tok.eos_token == "bogus"

    def test_no_candidate_and_no_id(self, mocker):
        from training.unsloth_factory import _fix_eos_token

        tok = mocker.MagicMock()
        tok.eos_token = "bogus"
        tok.eos_token_id = None
        tok.get_vocab.return_value = {"keep": 1}
        _fix_eos_token(tok)
        assert tok.eos_token == "bogus"


class TestBuildModelAndPeft:
    """Test the orchestration logic with all internals mocked."""

    def test_unsloth_path(self, mocker):
        mocker.patch("training.unsloth_factory._UNSLOTH_AVAILABLE", True)
        mocker.patch("training.unsloth_factory._UNSLOTH_ENABLED", True)
        mock_build = mocker.patch("training.unsloth_factory._build_with_unsloth")
        mock_build.return_value = ("model", "tokenizer")

        from training.unsloth_factory import build_model_and_peft

        result = build_model_and_peft({"hf_id": "test"}, {"lora": {"r": 8}})
        assert result == ("model", "tokenizer")
        mock_build.assert_called_once()

    def test_fallback_path_when_unsloth_disabled(self, mocker):
        mocker.patch("training.unsloth_factory._UNSLOTH_AVAILABLE", True)
        mocker.patch("training.unsloth_factory._UNSLOTH_ENABLED", False)
        mock_fallback = mocker.patch("training.unsloth_factory._build_fallback")
        mock_fallback.return_value = ("model_fb", "tok_fb")

        from training.unsloth_factory import build_model_and_peft

        result = build_model_and_peft({"hf_id": "test"}, {"lora": {"r": 8}})
        assert result == ("model_fb", "tok_fb")
        mock_fallback.assert_called_once()

    def test_fallback_path_when_unsloth_not_available(self, mocker):
        mocker.patch("training.unsloth_factory._UNSLOTH_AVAILABLE", False)
        mocker.patch("training.unsloth_factory._UNSLOTH_ENABLED", True)
        mock_fallback = mocker.patch("training.unsloth_factory._build_fallback")
        mock_fallback.return_value = ("model_fb", "tok_fb")

        from training.unsloth_factory import build_model_and_peft

        result = build_model_and_peft({"hf_id": "test"}, {"lora": {"r": 8}})
        assert result == ("model_fb", "tok_fb")

    def test_fallback_on_exception(self, mocker):
        mocker.patch("training.unsloth_factory._UNSLOTH_AVAILABLE", True)
        mocker.patch("training.unsloth_factory._UNSLOTH_ENABLED", True)
        mocker.patch("training.unsloth_factory._build_with_unsloth", side_effect=ValueError("OOM"))
        mock_fallback = mocker.patch("training.unsloth_factory._build_fallback")
        mock_fallback.return_value = ("model_fb", "tok_fb")
        mocker.patch("torch.cuda.is_available", return_value=False)

        from training.unsloth_factory import build_model_and_peft

        result = build_model_and_peft({"hf_id": "test"}, {"lora": {"r": 8}})
        assert result == ("model_fb", "tok_fb")


class TestBuildWithUnsloth:
    """Test _build_with_unsloth config-logic with unsloth mocked."""

    def _mock_unsloth(self, mocker):
        """Patch unsloth.FastLanguageModel for _build_with_unsloth tests."""
        mock_unsloth_mod = mocker.MagicMock()
        mock_unsloth_mod.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"unsloth": mock_unsloth_mod})
        mock_flm = mocker.MagicMock()
        mock_unsloth_mod.FastLanguageModel = mock_flm
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.get_vocab.return_value = {"<|endoftext|>": 151643}
        mock_tokenizer.pad_token = None
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model
        return mock_flm

    def test_quantization_4bit(self, mocker):
        mock_flm = self._mock_unsloth(mocker)

        from training.unsloth_factory import _build_with_unsloth

        model_cfg = {"hf_id": "Qwen/Qwen3-14B", "quantization": "nf4"}
        variant_cfg = {"lora": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}}
        model, _ = _build_with_unsloth(model_cfg, variant_cfg, 8192, True)

        call_kwargs = mock_flm.from_pretrained.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["load_in_8bit"] is False
        assert call_kwargs["attn_implementation"] == "flash_attention_2"
        assert call_kwargs["model_name"] == "Qwen/Qwen3-14B"

    def test_quantization_8bit(self, mocker):
        mock_flm = self._mock_unsloth(mocker)

        from training.unsloth_factory import _build_with_unsloth

        model_cfg = {"hf_id": "Qwen/Qwen3-14B", "quantization": "8bit"}
        variant_cfg = {"lora": {"r": 16, "lora_alpha": 32}}
        model, _ = _build_with_unsloth(model_cfg, variant_cfg, 4096, False)

        call_kwargs = mock_flm.from_pretrained.call_args[1]
        assert call_kwargs["load_in_4bit"] is False
        assert call_kwargs["load_in_8bit"] is True
        assert call_kwargs["attn_implementation"] == "eager"

    def test_peft_kwargs_construction(self, mocker):
        mock_flm = self._mock_unsloth(mocker)

        from training.unsloth_factory import _build_with_unsloth

        model_cfg = {"hf_id": "test", "compute_dtype": "float16"}
        variant_cfg = {
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "lora_only",
                "task_type": "CAUSAL_LM",
                "extra_param": "should_pass",
            }
        }
        _build_with_unsloth(model_cfg, variant_cfg, 4096, True)

        peft_call_kwargs = mock_flm.get_peft_model.call_args[1]
        assert peft_call_kwargs["r"] == 32
        assert peft_call_kwargs["lora_alpha"] == 64
        assert peft_call_kwargs["lora_dropout"] == 0.1
        assert peft_call_kwargs["bias"] == "lora_only"
        assert peft_call_kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert "task_type" not in peft_call_kwargs
        assert peft_call_kwargs["use_gradient_checkpointing"] == "unsloth"
        assert peft_call_kwargs["extra_param"] == "should_pass"


class TestBuildFallback:
    """Test _build_fallback config with transformers/peft mocked."""

    def _mock_fallback_deps(self, mocker):
        """Mock peft/transformers in sys.modules to avoid slow imports."""
        mock_peft = mocker.MagicMock()
        mock_peft.__spec__ = mocker.MagicMock()
        mock_tf = mocker.MagicMock()
        mock_tf.__spec__ = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"peft": mock_peft, "transformers": mock_tf})

        mock_bnb = mocker.MagicMock()
        mock_tok_cls = mocker.MagicMock()
        mock_prepare = mocker.MagicMock()
        mock_lora_config_cls = mocker.MagicMock()
        mock_get_peft = mocker.MagicMock()

        mock_tf.BitsAndBytesConfig = mock_bnb
        mock_tf.AutoModelForCausalLM = mocker.MagicMock()
        mock_tf.AutoTokenizer = mock_tok_cls
        mock_peft.prepare_model_for_kbit_training = mock_prepare
        mock_peft.LoraConfig = mock_lora_config_cls
        mock_peft.get_peft_model = mock_get_peft

        mock_model = mocker.MagicMock()
        mock_prepare.return_value = mock_model
        mock_get_peft.return_value = mock_model
        tok = mocker.MagicMock()
        mock_tok_cls.from_pretrained.return_value = tok
        return {"bnb": mock_bnb, "lora_config": mock_lora_config_cls}

    def test_bnb_config_fp4(self, mocker):
        deps = self._mock_fallback_deps(mocker)

        from training.unsloth_factory import _build_fallback

        model_cfg = {"hf_id": "Qwen/Qwen3-14B"}
        variant_cfg = {"lora": {"r": 8}}
        _build_fallback(model_cfg, variant_cfg)

        bnb_call = deps["bnb"].call_args[1]
        assert bnb_call["load_in_4bit"] is True
        assert bnb_call["bnb_4bit_quant_type"] == "fp4"
        assert bnb_call["bnb_4bit_use_double_quant"] is False

    def test_fallback_lora_config(self, mocker):
        deps = self._mock_fallback_deps(mocker)

        from training.unsloth_factory import _build_fallback

        model_cfg = {"hf_id": "Qwen/Qwen3-14B"}
        variant_cfg = {"lora": {"r": 16, "lora_alpha": 32}}
        _build_fallback(model_cfg, variant_cfg)

        lora_call = deps["lora_config"].call_args[1]
        assert lora_call["r"] == 16
        assert lora_call["lora_alpha"] == 32


# ── training/local_cli.py ──────────────────────────────────────────────────────


class TestLocalCliMain:
    """Test local_cli.main() constructs QLoRATrainer correctly."""

    def test_main_constructs_trainer(self, mocker):
        mock_trainer_cls = mocker.patch("training.qlora_trainer.QLoRATrainer")
        mock_instance = mocker.MagicMock()
        mock_instance.train.return_value = {"loss": 0.5}
        mock_trainer_cls.return_value = mock_instance

        mod = _import_training("local_cli")

        mod.main(model_name="qwen3-14b", variant="test_variant", data_dir="/tmp/data")

        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs["model_name"] == "qwen3-14b"
        assert call_kwargs["variant"] == "test_variant"
        assert call_kwargs["data_dir"] == "/tmp/data"
        assert call_kwargs["hf_id"] == "hf-internal-testing/tiny-random-LlamaForCausalLM"
        assert call_kwargs["use_flash_attn"] is False

    def test_main_returns_result(self, mocker):
        mock_trainer_cls = mocker.patch("training.qlora_trainer.QLoRATrainer")
        mock_instance = mocker.MagicMock()
        expected = {"loss": 0.25, "eval_loss": 0.30}
        mock_instance.train.return_value = expected
        mock_trainer_cls.return_value = mock_instance

        mod = _import_training("local_cli")
        mod.main()
        mock_instance.train.assert_called_once()


# ── training/modal_train.py ──────────────────────────────────────────────────


class TestModalTrainHelpers:
    """Test standalone helper functions in modal_train.py."""

    def test_tokenized_prefix(self):
        mod = _import_training("modal_train")

        assert mod._tokenized_prefix("abc123") == "tokenized/abc123/"
        assert mod._tokenized_prefix("") == "tokenized//"

    def test_module_constants(self):
        mod = _import_training("modal_train")

        assert mod._GCS_BUCKET == "swe-qwen-datasets"


# ── scripts/f2p_proxy.py ─────────────────────────────────────────────────────


class TestF2PProxy:
    """Test standalone pure functions in f2p_proxy.py."""

    def _load(self) -> Any:
        return _load_scripts_module("f2p_proxy")

    def test_select_champion_basic(self):
        mod = self._load()
        scores = {
            "baseline_14b": {"mean_f2p": 0.5},
            "higher_rank_14b": {"mean_f2p": 0.8},
        }
        assert mod.select_champion(scores) == "higher_rank_14b"

    def test_select_champion_tie(self):
        mod = self._load()
        scores = {
            "a": {"mean_f2p": 0.9},
            "b": {"mean_f2p": 0.9},
        }
        result = mod.select_champion(scores)
        assert result in ("a", "b")

    def test_select_champion_empty_scores(self):
        mod = self._load()
        with pytest.raises(ValueError):
            mod.select_champion({})

    def test_select_champion_missing_key(self):
        mod = self._load()
        scores = {
            "a": {},
            "b": {"mean_f2p": 0.7},
        }
        assert mod.select_champion(scores) == "b"


# ── scripts/prepare_training_data.py ────────────────────────────────────────


class TestLoadJsonl:
    """Test JSONL loading helper."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_load_jsonl(self, tmp_path):
        mod = self._load()
        f = tmp_path / "test.jsonl"
        f.write_text('{"a": 1}\n{"a": 2}\n')
        records = mod.load_jsonl(f)
        assert records == [{"a": 1}, {"a": 2}]

    def test_load_jsonl_skips_empty_lines(self, tmp_path):
        mod = self._load()
        f = tmp_path / "empty.jsonl"
        f.write_text('{"a": 1}\n\n{"a": 2}\n\n')
        records = mod.load_jsonl(f)
        assert records == [{"a": 1}, {"a": 2}]


class TestBasicQualityFilters:
    """Test quality filter logic — no disk I/O needed."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_keeps_valid_record(self):
        mod = self._load()
        records = [{"issue_id": "1", "repo": "a", "patch_diff": "diff", "issue_body": "body"}]
        assert mod.basic_quality_filters(records) == records

    def test_skips_missing_fields(self):
        mod = self._load()
        records = [
            {"issue_id": "1", "repo": "a", "patch_diff": "diff"},
            {"issue_id": "2", "repo": "b", "patch_diff": "diff", "issue_body": "body"},
        ]
        result = mod.basic_quality_filters(records)
        assert len(result) == 1
        assert result[0]["issue_id"] == "2"

    def test_skips_empty_patch(self):
        mod = self._load()
        records = [
            {"issue_id": "1", "repo": "a", "patch_diff": "", "issue_body": "body"},
            {"issue_id": "2", "repo": "b", "patch_diff": "  ", "issue_body": "body"},
            {"issue_id": "3", "repo": "c", "patch_diff": "diff", "issue_body": "body"},
        ]
        result = mod.basic_quality_filters(records)
        assert len(result) == 1
        assert result[0]["issue_id"] == "3"

    def test_keeps_non_python_patches(self):
        mod = self._load()
        records = [
            {
                "issue_id": "1",
                "repo": "a",
                "patch_diff": "diff",
                "issue_body": "body",
                "files_changed": ["README.md"],
            }
        ]
        assert mod.basic_quality_filters(records) == records


class TestReproStratifiedSplit:
    """Test repo-stratified split logic."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_basic_split(self):
        mod = self._load()
        records = [
            {"issue_id": "1", "repo": "repo_a"},
            {"issue_id": "2", "repo": "repo_a"},
            {"issue_id": "3", "repo": "repo_a"},
            {"issue_id": "4", "repo": "repo_b"},
            {"issue_id": "5", "repo": "repo_b"},
            {"issue_id": "6", "repo": "repo_c"},
        ]
        train, val, test = mod.repo_stratified_split(records, seed=42)
        assert len(train) + len(val) + len(test) == len(records)
        train_repos = {r["repo"] for r in train}
        val_repos = {r["repo"] for r in val}
        test_repos = {r["repo"] for r in test}
        assert train_repos & val_repos == set()
        assert train_repos & test_repos == set()
        assert val_repos & test_repos == set()

    def test_deterministic_with_seed(self):
        mod = self._load()
        records = [{"issue_id": str(i), "repo": f"repo_{i % 5}"} for i in range(100)]
        t1, v1, te1 = mod.repo_stratified_split(records, seed=42)
        t2, v2, te2 = mod.repo_stratified_split(records, seed=42)
        assert [r["issue_id"] for r in t1] == [r["issue_id"] for r in t2]
        assert [r["issue_id"] for r in v1] == [r["issue_id"] for r in v2]
        assert len(te1) == len(te2)

    def test_different_seed_gives_different_split(self):
        mod = self._load()
        records = [{"issue_id": str(i), "repo": f"repo_{i % 5}"} for i in range(100)]
        t1, _, _ = mod.repo_stratified_split(records, seed=42)
        t2, _, _ = mod.repo_stratified_split(records, seed=99)
        t1_ids = [r["issue_id"] for r in t1]
        t2_ids = [r["issue_id"] for r in t2]
        assert t1_ids != t2_ids

    def test_single_repo(self):
        mod = self._load()
        records = [{"issue_id": str(i), "repo": "mono_repo"} for i in range(10)]
        train, val, test = mod.repo_stratified_split(records)
        # Smallest target split gets the first repo to avoid empty splits
        assert len(train) + len(val) + len(test) == 10
        assert sorted(r["issue_id"] for r in train + val + test) == [str(i) for i in range(10)]

    def test_ratio_validation(self):
        mod = self._load()
        records = [{"issue_id": "1", "repo": "r"}]
        with pytest.raises(AssertionError):
            mod.repo_stratified_split(records, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_each_repo_in_one_split(self):
        mod = self._load()
        records = [{"issue_id": str(i), "repo": f"repo_{i}"} for i in range(20)]
        train, val, test = mod.repo_stratified_split(records, seed=7)
        assert len(train) + len(val) + len(test) == 20
        all_ids = [r["issue_id"] for r in train + val + test]
        assert sorted(all_ids, key=int) == [str(i) for i in range(20)]


class TestExtractGolden:
    """Test golden extraction logic."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_basic_extraction(self):
        mod = self._load()
        records = [{"issue_id": str(i), "metadata": {"has_test_patch": True}} for i in range(10)]
        golden, remaining = mod.extract_golden(records, max_golden=3, seed=42)
        assert len(golden) == 3
        assert len(remaining) == 7

    def test_no_eligible(self):
        mod = self._load()
        records = [{"issue_id": str(i), "metadata": {"has_test_patch": False}} for i in range(5)]
        golden, remaining = mod.extract_golden(records, max_golden=3)
        assert len(golden) == 0
        assert len(remaining) == 5

    def test_respects_max_golden(self):
        mod = self._load()
        records = [{"issue_id": str(i), "test_files_changed": ["test_a.py"]} for i in range(100)]
        golden, remaining = mod.extract_golden(records, max_golden=10, seed=1)
        assert len(golden) == 10
        assert len(remaining) == 90

    def test_deterministic(self):
        mod = self._load()
        records = [{"issue_id": str(i), "metadata": {"has_test_patch": True}} for i in range(20)]
        g1, r1 = mod.extract_golden(records, max_golden=5, seed=42)
        g2, r2 = mod.extract_golden(records, max_golden=5, seed=42)
        assert [r["issue_id"] for r in g1] == [r["issue_id"] for r in g2]
        assert len(r1) == len(r2)


class TestSaveJsonl:
    """Test JSONL save helper."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_save_jsonl(self, tmp_path):
        mod = self._load()
        records = [{"a": 1}, {"b": 2}]
        mod.save_jsonl(records, tmp_path / "out.jsonl")
        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_creates_parent_dirs(self, tmp_path):
        mod = self._load()
        mod.save_jsonl([{"x": 1}], tmp_path / "nested/dir/out.jsonl")
        assert (tmp_path / "nested/dir/out.jsonl").exists()
        assert json.loads((tmp_path / "nested/dir/out.jsonl").read_text()) == {"x": 1}


class TestCreateParser:
    """Test argument parser creation."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_create_parser(self):
        mod = self._load()
        parser = mod.create_parser()
        args = parser.parse_args(["--input", "x.jsonl", "--model-name", "test"])
        assert args.input == "x.jsonl"
        assert args.model_name == "test"
        assert args.seed == 42
        assert args.skip_tokenize is False


class TestLogSummary:
    """Test log_summary helper."""

    def _load(self) -> Any:
        return _load_scripts_module("prepare_training_data")

    def test_log_summary(self, tmp_path):
        mod = self._load()
        shards_dir = tmp_path / "shards"
        mod.log_summary((800, 50, 50, 100), shards_dir, skip_tokenize=False)

    def test_log_summary_skip_tokenize(self, tmp_path):
        mod = self._load()
        mod.log_summary((100, 20, 20, 10), tmp_path, skip_tokenize=True)


# ── scripts/run_3config_comparison.py ─────────────────────────────────────────


class Test3ConfigHelpers:
    """Test pure helper functions in run_3config_comparison.py."""

    def _load(self) -> Any:
        return _load_scripts_module("run_3config_comparison")

    def test_artifact_name(self):
        mod = self._load()
        assert mod._artifact_name("baseline_14b") == "model-qwen3-14b-baseline_14b"

    def test_gcs_golden_prefix(self):
        mod = self._load()
        assert mod._gcs_golden_prefix("abc123") == "datasets/abc123/swebench/"

    def test_golden_path(self):
        mod = self._load()
        path = mod._golden_path("run123")
        assert str(path).endswith("data/run123/swebench/golden.jsonl")

    def test_new_state(self):
        mod = self._load()
        state = mod._new_state("run_abc")
        assert state["run_id"] == "run_abc"
        assert state["completed_variants"] == []
        assert state["variants"] == {}

    def test_variant_run_name_format(self):
        mod = self._load()
        name = mod._variant_run_name("baseline_14b")
        assert name.startswith("3config-baseline_14b-")

    def test_tail_log(self, tmp_path):
        mod = self._load()
        f = tmp_path / "test.log"
        f.write_text("\n".join(f"line_{i}" for i in range(100)))
        tail = mod._tail_log(f, n=3)
        assert "line_97" in tail
        assert "line_99" in tail

    def test_tail_log_empty_file(self, tmp_path):
        mod = self._load()
        f = tmp_path / "empty.log"
        f.write_text("")
        assert mod._tail_log(f) == ""

    def test_tail_log_missing_file(self):
        mod = self._load()
        assert mod._tail_log(Path("/nonexistent/test.log")) == "(unable to read log)"

    def test_build_modal_cmd(self):
        mod = self._load()
        cmd = mod._build_modal_cmd("baseline_14b", "run-name-1", {"model_name": "qwen3-14b"})
        assert "modal" in cmd
        assert "run" in cmd
        assert "baseline_14b" in cmd
        assert "run-name-1" in cmd
        assert "--model-name" in cmd
        assert "qwen3-14b" in cmd

    def test_build_modal_cmd_skips_none(self):
        mod = self._load()
        cmd = mod._build_modal_cmd("test", "rn", {"a": None, "b": "val"})
        assert "--a" not in cmd
        assert "--b" in cmd

    def test_build_modal_cmd_bool_flag(self):
        mod = self._load()
        cmd = mod._build_modal_cmd("test", "rn", {"flag": True, "noflag": False})
        assert "--flag" in cmd
        assert "--noflag" not in cmd

    def test_parse_args_defaults(self):
        mod = self._load()
        args = mod.parse_args(["--run-id", "abc"])
        assert args.run_id == "abc"
        assert args.retrain_variants is None

    def test_parse_args_retrain_variants(self):
        mod = self._load()
        args = mod.parse_args(
            ["--run-id", "abc", "--retrain-variants", "higher_rank_14b", "higher_lr_14b"]
        )
        assert args.retrain_variants == ["higher_rank_14b", "higher_lr_14b"]
        assert args.dry_run is False
        assert args.variants == ["baseline_14b", "higher_rank_14b", "higher_lr_14b"]
        assert args.skip_eval is False
        assert args.force_retrain is False

    def test_parse_args_dry_run(self):
        mod = self._load()
        args = mod.parse_args(["--run-id", "abc", "--dry-run"])
        assert args.dry_run is True

    def test_parse_args_custom_variants(self):
        mod = self._load()
        args = mod.parse_args(["--run-id", "abc", "--variants", "a", "b"])
        assert args.variants == ["a", "b"]

    def test_parse_args_skip_eval(self):
        mod = self._load()
        args = mod.parse_args(["--run-id", "abc", "--skip-eval"])
        assert args.skip_eval is True


class TestMarkCompleted:
    """Test _mark_completed state mutation."""

    def _load(self) -> Any:
        return _load_scripts_module("run_3config_comparison")

    def test_mark_completed(self, tmp_path, mocker):
        mod = self._load()
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / ".pipeline-state.json")

        state = {
            "run_id": "test",
            "completed_variants": [],
            "variants": {"v1": {"status": "running", "run_name": "test-run"}},
        }
        mod._mark_completed(state, "v1", "wandb_123", "artifact_v1")
        assert state["variants"]["v1"]["status"] == "completed"
        assert state["variants"]["v1"]["result"]["wandb_run_id"] == "wandb_123"
        assert "v1" in state["completed_variants"]

    def test_mark_completed_dedup(self, tmp_path, mocker):
        mod = self._load()
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / ".pipeline-state.json")

        state = {
            "run_id": "test",
            "completed_variants": ["v1"],
            "variants": {"v1": {"status": "running", "run_name": "r"}},
        }
        mod._mark_completed(state, "v1", "wb", "art")
        assert state["completed_variants"] == ["v1"]


class TestLoadState:
    """Test _load_state logic."""

    def _load(self) -> Any:
        return _load_scripts_module("run_3config_comparison")

    def test_load_state_fresh(self, tmp_path, mocker):
        mod = self._load()
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "nonexistent.json")

        state = mod._load_state("run_new")
        expected = mod._new_state("run_new")
        assert state["run_id"] == expected["run_id"]
        assert state["completed_variants"] == expected["completed_variants"]

    def test_load_state_matching_run(self, tmp_path, mocker):
        mod = self._load()
        state_path = tmp_path / ".pipeline-state.json"
        state_path.write_text(
            json.dumps({"run_id": "run_match", "completed_variants": ["v1"], "variants": {}})
        )
        mocker.patch.object(mod, "_STATE_PATH", state_path)

        state = mod._load_state("run_match")
        assert state["run_id"] == "run_match"
        assert state["completed_variants"] == ["v1"]

    def test_load_state_different_run(self, tmp_path, mocker):
        mod = self._load()
        state_path = tmp_path / ".pipeline-state.json"
        state_path.write_text(
            json.dumps({"run_id": "old_run", "completed_variants": ["v1"], "variants": {}})
        )
        mocker.patch.object(mod, "_STATE_PATH", state_path)

        state = mod._load_state("new_run")
        assert state["run_id"] == "new_run"
        assert state["completed_variants"] == []


# ── scripts/init_wandb.py ─────────────────────────────────────────────────────


class TestInitWandb:
    """Test W&B init functions with mocked wandb."""

    def _load(self) -> Any:
        return _load_scripts_module("init_wandb")

    def test_init_wandb_project_no_api_key(self, mocker):
        mod = self._load()
        mocker.patch.dict(mod.os.environ, {"WANDB_API_KEY": ""}, clear=True)

        result = mod.init_wandb_project("test-project")
        assert result is False

    def test_init_wandb_project_login_fails(self, mocker):
        mod = self._load()
        mocker.patch.dict(mod.os.environ, {"WANDB_API_KEY": "fake-key"}, clear=True)
        mocker.patch.object(mod, "wandb")
        mod.wandb.login.side_effect = Exception("auth failed")

        result = mod.init_wandb_project("test-project")
        assert result is False

    def test_init_wandb_project_success(self, mocker):
        mod = self._load()
        mocker.patch.dict(mod.os.environ, {"WANDB_API_KEY": "fake-key"}, clear=True)
        mocker.patch.object(mod, "wandb")
        mod.wandb.login.return_value = True
        mock_run = mocker.MagicMock()
        mod.wandb.init.return_value = mock_run

        result = mod.init_wandb_project("swe-qwen", entity="myteam", tags=["test"])
        assert result is True
        mock_run.log_artifact.assert_called_once()
        mock_run.finish.assert_called_once()

    def test_init_wandb_with_description(self, mocker):
        mod = self._load()
        mocker.patch.dict(mod.os.environ, {"WANDB_API_KEY": "fake-key"}, clear=True)
        mocker.patch.object(mod, "wandb")
        mod.wandb.login.return_value = True
        mock_run = mocker.MagicMock()
        mod.wandb.init.return_value = mock_run

        result = mod.init_wandb_project("swe-qwen", description="Test project")
        assert result is True
        assert mock_run.notes == "Test project"

    def test_create_sweep_config(self, mocker):
        mod = self._load()
        mocker.patch.object(mod, "wandb")
        mod.wandb.sweep.return_value = "sweep_abc"

        sweep_id = mod.create_sweep_config("swe-qwen", "myteam")
        assert sweep_id == "sweep_abc"

    def test_setup_artifact_registries(self, mocker):
        mod = self._load()
        mock_api = mocker.MagicMock()
        mocker.patch.object(mod, "wandb")
        mod.wandb.Api.return_value = mock_api

        mod.setup_artifact_registries("swe-qwen", "myteam")
        # wandb's artifact_type() has no entity kwarg; entity goes in the project path
        mock_api.artifact_type.assert_any_call("model", project="myteam/swe-qwen")
        mock_api.artifact_type.assert_any_call("dataset", project="myteam/swe-qwen")

    def test_setup_artifact_registries_exception(self, mocker):
        mod = self._load()
        mock_api = mocker.MagicMock()
        mock_api.artifact_type.side_effect = Exception("API error")
        mocker.patch.object(mod, "wandb")
        mod.wandb.Api.return_value = mock_api

        mod.setup_artifact_registries("swe-qwen", "myteam")
        assert mock_api.artifact_type.call_count == 2


# ── scripts/run_3config_comparison.py advanced tests ──────────────────────────


class Test3ConfigAdvanced:
    """Test additional pure functions in run_3config_comparison.py."""

    def _load(self):
        return _load_scripts_module("run_3config_comparison")

    @pytest.fixture(autouse=True)
    def _mock_wandb_global(self):
        """Ensure wandb is mocked globally for these tests.
        run_3config_comparison imports wandb inside functions via sys.modules.
        """
        old = sys.modules.get("wandb")
        sys.modules["wandb"] = mock.MagicMock()
        sys.modules["wandb"].__spec__ = mock.MagicMock()
        yield
        if old is not None:
            sys.modules["wandb"] = old
        else:
            sys.modules.pop("wandb", None)

    def test_cleanup_state(self, tmp_path):
        mod = self._load()
        state_file = tmp_path / ".pipeline-state.json"
        state_file.touch()
        mod._STATE_PATH = state_file

        mod._cleanup_state()
        assert not state_file.exists()

    def test_cleanup_state_no_file(self, tmp_path):
        mod = self._load()
        mod._STATE_PATH = tmp_path / ".pipeline-state.json"

        mod._cleanup_state()

    def test_resolve_wandb_entity(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"

        result = mod._resolve_wandb_entity()
        assert result == "test-entity"

    def test_resolve_wandb_entity_no_entity(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = None

        with pytest.raises(RuntimeError, match="W&B entity not found"):
            mod._resolve_wandb_entity()

    def test_wandb_run_finished(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"
        mock_run = mock.MagicMock()
        mock_run.state = "finished"
        mock_run.id = "run-123"
        mock_run.config.get.return_value = "baseline_14b"
        sys.modules["wandb"].Api.return_value.runs.return_value = [mock_run]

        result = mod._wandb_run_finished("test-run")
        assert result == {
            "wandb_run_id": "run-123",
            "artifact_name": "model-qwen3-14b-baseline_14b",
        }

    def test_wandb_run_finished_running_with_loss(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"
        mock_run = mock.MagicMock()
        mock_run.state = "running"
        mock_run.id = "run-456"
        mock_run.config.get.return_value = "higher_rank_14b"
        mock_run.summary = {"train/loss": 0.5}
        sys.modules["wandb"].Api.return_value.runs.return_value = [mock_run]

        result = mod._wandb_run_finished("test-run")
        assert result == {
            "wandb_run_id": "run-456",
            "artifact_name": "model-qwen3-14b-higher_rank_14b",
        }

    def test_wandb_run_finished_crashed(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"
        mock_run = mock.MagicMock()
        mock_run.state = "crashed"
        sys.modules["wandb"].Api.return_value.runs.return_value = [mock_run]

        with pytest.raises(RuntimeError, match="crashed"):
            mod._wandb_run_finished("test-run")

    def test_wandb_run_finished_not_found(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"
        sys.modules["wandb"].Api.return_value.runs.return_value = []

        result = mod._wandb_run_finished("test-run")
        assert result is None

    def test_wandb_run_finished_exception(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"
        sys.modules["wandb"].Api.return_value.runs.side_effect = Exception("API down")

        result = mod._wandb_run_finished("test-run")
        assert result is None

    def test_wandb_run_finished_defers_when_api_down(self):
        mod = self._load()
        mod._WANDB_API_RETRIES = 2
        mod._WANDB_RETRY_DELAY = 0.01
        # Api() constructor failing (wedged service) must defer, not raise
        sys.modules["wandb"].Api.side_effect = RuntimeError("service process is busy")

        assert mod._wandb_run_finished("test-run") is None

    def test_wandb_run_finished_recovers_after_api_down(self):
        mod = self._load()
        mod._WANDB_API_RETRIES = 3
        mod._WANDB_RETRY_DELAY = 0.01
        ok_api = mock.MagicMock()
        ok_api.default_entity = "test-entity"
        mock_run = mock.MagicMock()
        mock_run.state = "finished"
        mock_run.id = "run-789"
        mock_run.config.get.return_value = "baseline_14b"
        ok_api.runs.return_value = [mock_run]
        sys.modules["wandb"].Api.side_effect = [
            RuntimeError("service process is busy"),
            ok_api,
        ]

        result = mod._wandb_run_finished("test-run")
        assert result == {
            "wandb_run_id": "run-789",
            "artifact_name": "model-qwen3-14b-baseline_14b",
        }

    def test_retrain_variants_clears_state_and_skips_reconcile(self):
        mod = self._load()
        sys.modules["wandb"].Api.return_value.default_entity = "test-entity"

        def mk_run(name: str, variant: str) -> mock.MagicMock:
            r = mock.MagicMock()
            r.name = name
            r.id = f"id-{variant}"
            r.state = "finished"
            r.created_at = datetime.datetime(2026, 8, 6)
            r.config.get.return_value = variant
            return r

        sys.modules["wandb"].Api.return_value.runs.return_value = [
            mk_run("3config-higher_rank_14b-20260730-200513", "higher_rank_14b"),
            mk_run("3config-higher_lr_14b-20260730-202714", "higher_lr_14b"),
            mk_run("3config-baseline_14b-20260806-013838", "baseline_14b"),
        ]

        state = {
            "run_id": "expanded-repos",
            "completed_variants": ["baseline_14b", "higher_rank_14b", "higher_lr_14b"],
            "variants": {
                "baseline_14b": {
                    "status": "completed",
                    "run_name": "3config-baseline_14b-20260806-013838",
                },
                "higher_rank_14b": {
                    "status": "completed",
                    "run_name": "3config-higher_rank_14b-20260730-200513",
                },
                "higher_lr_14b": {
                    "status": "completed",
                    "run_name": "3config-higher_lr_14b-20260730-202714",
                },
            },
        }

        out = mod._reconcile_state_with_wandb(
            state,
            requested_variants=["baseline_14b", "higher_rank_14b", "higher_lr_14b"],
            retrain_variants=["higher_rank_14b", "higher_lr_14b"],
        )
        # Retrained variants are cleared and never re-completed from stale runs
        assert out["completed_variants"] == ["baseline_14b"]
        assert "higher_rank_14b" not in out["variants"]
        assert "higher_lr_14b" not in out["variants"]
        # Untouched variant still reconciles against today's run
        assert out["variants"]["baseline_14b"]["status"] == "completed"
        assert out["variants"]["baseline_14b"]["result"]["wandb_run_id"] == "id-baseline_14b"

    def test_download_adapter_dry_run(self, tmp_path):
        mod = self._load()
        result = mod.download_adapter("model-qwen3-14b-baseline_14b", tmp_path, dry_run=True)
        assert result == str(tmp_path / "model-qwen3-14b-baseline_14b")

    def test_promote_champion_dry_run(self, capsys):
        mod = self._load()
        mod.promote_champion("baseline_14b", "model-qwen3-14b-baseline_14b", dry_run=True)
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "champion" in captured.out.lower()

    def test_evaluate_proxy_f2p_dry_run(self, tmp_path):
        mod = self._load()
        result = mod.evaluate_proxy_f2p(
            "baseline_14b", tmp_path / "golden.jsonl", "/tmp/adapter", dry_run=True
        )
        assert result == {"variant": "baseline_14b", "mean_f2p": 0.0}

    def test_launch_modal_training_dry_run(self):
        mod = self._load()
        state = {"variants": {}, "completed_variants": []}
        result = mod.launch_modal_training(
            "baseline_14b", "run123", "3config-baseline-20240101-000000", state, dry_run=True
        )
        assert result == {
            "wandb_run_id": "dry-run-baseline_14b",
            "artifact_name": "model-qwen3-14b-baseline_14b",
        }

    def test_ensure_golden_local(self, tmp_path):
        mod = self._load()
        golden_dir = tmp_path / "data" / "run123" / "swebench"
        golden_dir.mkdir(parents=True)
        golden_file = golden_dir / "golden.jsonl"
        golden_file.write_text("{}")
        mod._REPO_ROOT = tmp_path

        result = mod._ensure_golden("run123")
        assert result == golden_file

    def test_ensure_golden_download(self, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        list_data = json.dumps(
            {
                "items": [
                    {
                        "name": "datasets/run123/swebench/golden.jsonl",
                        "mediaLink": "https://storage/golden",
                    },
                ]
            }
        ).encode()
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            list_resp = mock.MagicMock()
            list_resp.read.return_value = list_data
            list_resp.__enter__.return_value = list_resp
            file_resp = mock.MagicMock()
            file_resp.read.return_value = b'{"test": "data"}'
            file_resp.__enter__.return_value = file_resp
            mock_urlopen.side_effect = [list_resp, file_resp]

            result = mod._ensure_golden("run123")
        assert result.exists()
        assert result.read_text() == '{"test": "data"}'

    def test_ensure_golden_empty_gcs(self):
        mod = self._load()
        list_data = json.dumps({"items": []}).encode()
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            resp = mock.MagicMock()
            resp.read.return_value = list_data
            resp.__enter__.return_value = resp
            mock_urlopen.return_value = resp

            with pytest.raises(SystemExit):
                mod._ensure_golden("run123")

    def test_ensure_golden_missing_golden(self):
        mod = self._load()
        list_data = json.dumps(
            {
                "items": [
                    {
                        "name": "datasets/run123/swebench/other.jsonl",
                        "mediaLink": "https://storage/other",
                    },
                ]
            }
        ).encode()
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            resp = mock.MagicMock()
            resp.read.return_value = list_data
            resp.__enter__.return_value = resp
            mock_urlopen.return_value = resp

            with pytest.raises(SystemExit):
                mod._ensure_golden("run123")

    def test_ensure_golden_skips_directory_item_and_downloads_binary(self, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        list_data = json.dumps(
            {
                "items": [
                    {"name": "datasets/run123/swebench/", "mediaLink": "https://storage/dir"},
                    {
                        "name": "datasets/run123/swebench/golden.jsonl",
                        "mediaLink": "https://storage/golden",
                    },
                ]
            }
        ).encode()
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            list_resp = mock.MagicMock()
            list_resp.read.return_value = list_data
            list_resp.__enter__.return_value = list_resp
            file_resp = mock.MagicMock()
            file_resp.read.return_value = b'{"test": "data"}'
            file_resp.__enter__.return_value = file_resp
            mock_urlopen.side_effect = [list_resp, file_resp]

            result = mod._ensure_golden("run123")
        assert result.exists()

    def test_reconcile_state(self):
        mod = self._load()
        w = sys.modules["wandb"]
        w.Api.return_value.default_entity = "test-entity"
        w.Api.return_value.runs.side_effect = None

        def _run(name, state, rid, variant, summary=None):
            r = mock.MagicMock()
            r.name = name
            r.state = state
            r.id = rid
            r.config = {"variant": variant}
            r.created_at = "2024-01-01"
            r.summary = summary or {}
            return r

        runs = [
            _run("3config-baseline_14b", "finished", "r1", "baseline_14b"),
            _run("3config-qlora_32b", "crashed", "r2", "qlora_32b"),
            _run("3config-qlora_8b", "running", "r3", "qlora_8b", {"train/loss": 0.5}),
            _run("3config-qlora_16b", "running", "r4", "qlora_16b"),
        ]
        w.Api.return_value.runs.return_value = runs

        state = mod._new_state("run123")
        req = ["baseline_14b", "qlora_32b", "qlora_8b", "qlora_16b"]
        result = mod._reconcile_state_with_wandb(state, req)

        assert "baseline_14b" in result["completed_variants"]
        assert result["variants"]["baseline_14b"]["status"] == "completed"
        assert result["variants"]["baseline_14b"]["result"]["wandb_run_id"] == "r1"
        assert result["variants"]["qlora_32b"]["status"] == "failed"
        assert "qlora_32b" not in result["completed_variants"]
        assert "qlora_8b" in result["completed_variants"]
        assert result["variants"]["qlora_8b"]["status"] == "completed"
        assert result["variants"]["qlora_16b"]["status"] == "running"
        assert "qlora_16b" not in result["completed_variants"]

    def test_download_adapter_live(self, tmp_path):
        mod = self._load()
        w = sys.modules["wandb"]
        w.Api.return_value.default_entity = "test-entity"
        mock_artifact = mock.MagicMock()
        w.Api.return_value.artifact.return_value = mock_artifact

        result = mod.download_adapter("model-qwen3-14b-baseline_14b", tmp_path, dry_run=False)
        assert result == str(tmp_path / "model-qwen3-14b-baseline_14b")
        mock_artifact.download.assert_called_once()

    def test_promote_champion_live(self, capsys):
        mod = self._load()
        w = sys.modules["wandb"]
        w.Api.return_value.default_entity = "test-entity"
        mock_artifact = mock.MagicMock()
        w.Api.return_value.artifact.return_value = mock_artifact

        mod.promote_champion("baseline_14b", "model-qwen3-14b-baseline_14b", dry_run=False)
        captured = capsys.readouterr()
        assert "Promoted" in captured.out
        assert "champion" in captured.out
        mock_artifact.save.assert_called_once()


# ── scripts/f2p_proxy.py advanced tests ─────────────────────────────────────────


class TestF2PProxyAdvanced:
    """Test compute_proxy_f2p_scores with mocked W&B."""

    def _setup(self):
        w = sys.modules["wandb"]
        # Reset Api mock without calling w.reset_mock() (real wandb module doesn't have it)
        fresh_api = mock.MagicMock()
        fresh_api.runs.side_effect = None
        fresh_api.default_entity = "test-entity"
        w.Api = mock.PropertyMock(return_value=fresh_api)
        return w

    def test_compute_proxy_f2p_scores(self, tmp_path):
        from scripts.f2p_proxy import compute_proxy_f2p_scores

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n{"id": 2}\n')
        w = self._setup()
        mock_run = mock.MagicMock()
        mock_run.created_at = "2024-01-02"
        mock_run.state = "finished"
        mock_run.summary = {"train/loss": 0.5}
        w.Api.return_value.runs.return_value = [mock_run]

        result = compute_proxy_f2p_scores(golden, {"baseline": "adapter-path"})
        assert "baseline" in result
        assert result["baseline"]["mean_f2p"] == 1.0
        assert result["baseline"]["count"] == 2

    def test_compute_proxy_f2p_scores_no_runs(self, tmp_path):
        from scripts.f2p_proxy import compute_proxy_f2p_scores

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        w = self._setup()
        w.Api.return_value.runs.return_value = []

        result = compute_proxy_f2p_scores(golden, {"baseline": "adapter"})
        assert result["baseline"]["mean_f2p"] == 0.0
        assert "warning" in result["baseline"]

    def test_compute_proxy_f2p_scores_no_loss(self, tmp_path):
        from scripts.f2p_proxy import compute_proxy_f2p_scores

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        w = self._setup()
        mock_run = mock.MagicMock()
        mock_run.created_at = "2024-01-02"
        mock_run.summary = {}
        w.Api.return_value.runs.return_value = [mock_run]

        result = compute_proxy_f2p_scores(golden, {"baseline": "adapter"})
        assert result["baseline"]["mean_f2p"] == 0.0
        assert "warning" in result["baseline"]

    def test_compute_proxy_f2p_scores_multi_variant(self, tmp_path):
        from scripts.f2p_proxy import compute_proxy_f2p_scores

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        w = self._setup()
        run_a = mock.MagicMock()
        run_a.created_at = "2024-01-03"
        run_a.summary = {"train/loss": 0.5}
        w.Api.return_value.runs.return_value = [run_a]

        result = compute_proxy_f2p_scores(golden, {"variant_a": "path_a", "variant_b": "path_b"})
        assert "variant_a" in result
        assert "variant_b" in result


class TestModalTrainDownload:
    """Test GCS download helper in modal_train.py."""

    def test_download_gcs_public(self, tmp_path, mocker):
        mod = _import_training("modal_train")

        prefix = "tokenized/run123/"
        list_data = json.dumps(
            {
                "items": [
                    {
                        "name": "tokenized/run123/train/data-00000.arrow",
                        "mediaLink": "https://storage/file1",
                    },
                ]
            }
        ).encode()

        list_resp = mocker.MagicMock()
        list_resp.read.return_value = list_data
        list_resp.__enter__.return_value = list_resp

        file_resp = mocker.MagicMock()
        file_resp.read.side_effect = [b"data", b""]
        file_resp.__enter__.return_value = file_resp

        mocker.patch.object(mod.urllib.request, "urlopen", side_effect=[list_resp, file_resp])

        result = mod._download_gcs_public(prefix, str(tmp_path))
        assert (tmp_path / "train" / "data-00000.arrow").exists()
        assert result == str(tmp_path)


# ── training/__init__.py ───────────────────────────────────────────────────────


class TestTrainingInit:
    """Test training package exports."""

    def test_imports_are_resolvable(self):
        from training.prompt_loader import PromptLoader
        from training.qlora_config import (
            build_qlora_config,
            list_models,
            list_variants,
            resolve_gpu_type,
        )
        from training.qlora_trainer import QLoRATrainer
        from training.resume import resolve_checkpoint_path

        assert QLoRATrainer is not None
        assert build_qlora_config is not None
        assert list_models is not None
        assert list_variants is not None
        assert resolve_gpu_type is not None
        assert PromptLoader is not None
        assert resolve_checkpoint_path is not None
