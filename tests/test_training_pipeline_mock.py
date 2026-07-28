"""Mocked tests for training pipeline components.

These tests do NOT require a GPU — they mock model loading and training
to verify callback behavior, resume logic, and artifact logging.
"""

from __future__ import annotations

import json

import pytest


class TestPromptLoader:
    """Tests for prompt_loader (no GPU needed)."""

    def test_loader_init(self):
        """PromptLoader initializes with default template dir."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        assert "chat" in loader.available_templates
        assert "system" in loader.available_templates
        assert "user" in loader.available_templates
        assert "assistant" in loader.available_templates

    def test_render_chat_template(self):
        """Chat template renders without error."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.render_chat(
            system_prompt="You are a coding assistant.",
            messages=[
                {"role": "user", "content": "Fix the bug."},
                {"role": "assistant", "content": "Here's the patch."},
            ],
        )
        assert "You are a coding assistant" in result
        assert "Fix the bug" in result
        assert "Here's the patch" in result

    def test_render_user_template(self):
        """User template renders with context."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.render(
            "user",
            issue_title="test-issue-1",
            issue_body="Something broke",
            repo_name="test/repo",
            repo_domain="utils",
            context_files=["src/main.py"],
            test_files=["tests/test_main.py"],
        )
        assert "test-issue-1" in result
        assert "Something broke" in result
        assert "test/repo" in result

    def test_render_system_template(self):
        """System template renders."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.render(
            "system",
            task_description="Fix bugs.",
            language="Python",
            style_guide="Follow PEP 8.",
        )
        assert "Fix bugs" in result
        assert "Python" in result

    def test_get_template_source(self):
        """Returns raw template source."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        source = loader.get_template_source("system")
        assert "{{ task_description }}" in source
        assert "{{ style_guide }}" in source

    def test_loader_invalid_template_raises(self):
        """Asking for nonexistent template raises FileNotFoundError."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        with pytest.raises(FileNotFoundError):
            loader.get_template_source("nonexistent")


class TestResume:
    """Tests for resume.py (mocked W&B)."""

    def test_resume_local_path_exists(self, tmp_path):
        """Local path returns itself if it exists."""
        from training.resume import resolve_checkpoint_path

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        result = resolve_checkpoint_path(str(ckpt_dir))
        assert result == str(ckpt_dir.resolve())

    def test_resume_local_path_not_found(self, tmp_path):
        """Non-existent local path returns None."""
        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("/nonexistent/path/checkpoint-100")
        assert result is None

    def test_resume_empty_spec(self):
        """Empty or None spec returns None."""
        from training.resume import resolve_checkpoint_path

        assert resolve_checkpoint_path("") is None

    def test_resume_local_path_under_volume(self, tmp_path):
        """Path found under volume path."""
        from training.resume import resolve_checkpoint_path

        vol_path = tmp_path / "models"
        vol_path.mkdir()
        ckpt = vol_path / "checkpoint-500"
        ckpt.mkdir()

        result = resolve_checkpoint_path(
            "checkpoint-500",  # relative path
            local_volume_path=str(vol_path),
        )
        assert result is not None
        assert "checkpoint-500" in result

    def test_resume_latest_no_wandb_run(self, mocker):
        """'latest' without active W&B run returns None."""
        mocker.patch("wandb.run", None)
        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("latest")
        assert result is None  # No active run, no artifacts to query


class TestCallbacks:
    """Tests for callback classes (mocked W&B)."""

    def test_wandb_checkpoint_callback_init(self):
        """WandbCheckpointCallback initializes."""
        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        assert cb is not None

    def test_wandb_logging_callback_init(self):
        """WandbLoggingCallback initializes."""
        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        assert cb is not None

    def test_on_log_with_metrics(self, mocker):
        """on_log calls wandb.log with metrics."""
        mock_wandb = mocker.patch("wandb.log")
        mocker.patch("wandb.run", mocker.MagicMock())

        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(output_dir="/tmp/test-cb")
        state = TrainerState()
        control = TrainerControl()

        cb.on_log(
            args,
            state,
            control,
            logs={"loss": 0.5, "eval_loss": 0.4},
        )
        mock_wandb.assert_called_once()
        call_args = mock_wandb.call_args[0][0]
        assert call_args["loss"] == 0.5
        assert call_args["eval_loss"] == 0.4

    def test_on_train_begin_logs_config(self, mocker):
        """on_train_begin logs training config to W&B."""
        mock_config = mocker.patch("wandb.config")
        mocker.patch("wandb.run", mocker.MagicMock())

        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(
            output_dir="/tmp/test-cb",
            learning_rate=1e-4,
            num_train_epochs=2,
        )
        state = TrainerState()
        control = TrainerControl()

        cb.on_train_begin(args, state, control)
        mock_config.update.assert_called_once()


class TestQLoRAConfig:
    """Additional config tests.

    Note: Core tests are in test_qlora_config.py (separate file).
    This class adds edge cases for the CLI wrapper.
    """

    def test_cli_parser_help(self):
        """CLI parser handles --help."""
        from training.qlora_train import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--help"])

    def test_cli_parser_defaults(self):
        """CLI parser returns correct defaults."""
        from training.qlora_train import parse_args

        args = parse_args([])
        assert args.model_name == "qwen3-30b-a3b"
        assert args.variant == "baseline"
        assert args.data_dir == "data/tokenized"
        assert args.output_dir == "/tmp/qlora-output"
        assert args.wandb_project == "swe-qwen"
        assert args.no_flash_attn is False

    def test_cli_parser_custom_values(self):
        """CLI parser accepts custom values."""
        from training.qlora_train import parse_args

        args = parse_args(
            [
                "--model-name",
                "qwen3-14b",
                "--variant",
                "higher_lr",
                "--data-dir",
                "/custom/data",
                "--output-dir",
                "/custom/output",
                "--run-name",
                "my-run",
                "--no-flash-attn",
            ]
        )
        assert args.model_name == "qwen3-14b"
        assert args.variant == "higher_lr"
        assert args.data_dir == "/custom/data"
        assert args.output_dir == "/custom/output"
        assert args.run_name == "my-run"
        assert args.no_flash_attn is True


class TestTokenize:
    """Tests for tokenize.py (mocked transformers)."""

    def test_format_training_prompt(self):
        """Training prompt formatting works with sample record."""
        from data_engineering.tokenize import format_training_prompt
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        record = {
            "issue_id": "test-issue-123",
            "issue_body": "The bug is in the parser.",
            "repo": "test/repo",
            "repo_domain": "utils",
            "patch_diff": "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1,2 @@\n-fix\n+fixed\n",
            "files_changed": ["src/main.py", "src/parser.py"],
            "test_files_changed": ["tests/test_parser.py"],
        }

        prompt = format_training_prompt(record, loader)
        assert "test-issue-123" in prompt
        assert "The bug is in the parser" in prompt
        assert "+fixed" in prompt
        assert "-fix" in prompt

    def test_load_jsonl(self, tmp_path):
        """Load JSONL file correctly."""
        from data_engineering.tokenize import load_jsonl_split

        jsonl_file = tmp_path / "test.jsonl"
        records = [
            {"issue_id": "1", "repo": "test/repo"},
            {"issue_id": "2", "repo": "test/repo"},
        ]
        with jsonl_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        loaded = load_jsonl_split(jsonl_file)
        assert len(loaded) == 2
        assert loaded[0]["issue_id"] == "1"

    def test_load_jsonl_empty(self, tmp_path):
        """Empty JSONL returns empty list."""
        from data_engineering.tokenize import load_jsonl_split

        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        loaded = load_jsonl_split(jsonl_file)
        assert loaded == []
