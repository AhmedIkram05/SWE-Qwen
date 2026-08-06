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

    def test_resume_local_path_not_found(self, mocker, tmp_path):
        """Non-existent local path returns None."""
        mocker.patch("wandb.Api", side_effect=ImportError("no wandb"))
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
        mocker.patch("wandb.Api", side_effect=ImportError("no wandb"))
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
        assert args.model_name == "qwen3-14b"
        assert args.variant == "baseline_14b"
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


class TestResolveCheckpoint:
    """Checkpoint resolution edge cases (W&B artifact, latest, fallback)."""

    def test_resolve_warn_none(self):
        """Non-existent local path with volume returns None."""
        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path(
            "nonexistent_name",
            local_volume_path="/tmp/vol",
        )
        assert result is None

    def test_resolve_wandb_artifact_success(self, mocker):
        """W&B artifact ref downloads and returns path."""
        mock_artifact = mocker.MagicMock()
        mock_artifact.download.return_value = "/tmp/artifact/checkpoint"

        mock_api = mocker.MagicMock()
        mock_api.artifact.return_value = mock_artifact
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("entity/project/artifact:v1")
        assert result == "/tmp/artifact/checkpoint"
        mock_api.artifact.assert_called_once_with("entity/project/artifact:v1")

    def test_resolve_wandb_artifact_fail(self, mocker):
        """W&B artifact ref that fails returns None."""
        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not found")
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("entity/project/artifact:v99")
        assert result is None

    def test_resolve_latest_with_full_run_id(self, mocker):
        """ "latest" with full entity/project/id run_id."""
        mock_artifact = mocker.MagicMock()
        mock_artifact.type = "model_checkpoint"
        mock_artifact.created_at = "2024-06-01T00:00:00"
        mock_artifact.download.return_value = "/tmp/latest/ckpt"

        mock_run = mocker.MagicMock()
        mock_run.logged_artifacts.return_value = [mock_artifact]

        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not artifact")
        mock_api.run.return_value = mock_run
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path(
            "latest",
            run_id="entity/project/run123",
        )
        assert result == "/tmp/latest/ckpt"
        mock_api.run.assert_called_once_with("entity/project/run123")

    def test_resolve_latest_with_partial_run_id(self, mocker):
        """ "latest" with bare run_id constructs path from wandb.run."""
        mock_wandb_run = mocker.MagicMock()
        mock_wandb_run.entity = "myentity"
        mock_wandb_run.project = "myproject"
        mocker.patch("wandb.run", mock_wandb_run)

        mock_artifact = mocker.MagicMock()
        mock_artifact.type = "model_checkpoint"
        mock_artifact.created_at = "2024-06-01T00:00:00"
        mock_artifact.download.return_value = "/tmp/latest/ckpt"

        mock_run = mocker.MagicMock()
        mock_run.logged_artifacts.return_value = [mock_artifact]

        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not artifact")
        mock_api.run.return_value = mock_run
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("latest", run_id="run123")
        assert result == "/tmp/latest/ckpt"
        mock_api.run.assert_called_once_with("myentity/myproject/run123")

    def test_resolve_latest_no_artifacts(self, mocker):
        """ "latest" returns None when no checkpoint artifacts exist."""
        mock_wandb_run = mocker.MagicMock()
        mock_wandb_run.entity = "e"
        mock_wandb_run.project = "p"
        mocker.patch("wandb.run", mock_wandb_run)

        mock_run = mocker.MagicMock()
        mock_run.logged_artifacts.return_value = []

        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not artifact")
        mock_api.run.return_value = mock_run
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("latest", run_id="r1")
        assert result is None

    def test_resolve_latest_exception(self, mocker):
        """ "latest" handles API exception."""
        mocker.patch("wandb.run", mocker.MagicMock())

        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not artifact")
        mock_api.run.side_effect = Exception("API error")
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("latest", run_id="r1")
        assert result is None

    def test_resolve_warn_none_no_vol(self):
        """Non-existent name without volume returns None (vol_path=None)."""
        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("nonexistent_name_no_vol")
        assert result is None

    def test_resolve_latest_active_run_no_run_id(self, mocker):
        """ "latest" with active wandb.run and no run_id."""
        mock_wandb_run = mocker.MagicMock()
        mock_wandb_run.entity = "e"
        mock_wandb_run.project = "p"
        mock_wandb_run.id = "run123"
        mocker.patch("wandb.run", mock_wandb_run)

        mock_artifact = mocker.MagicMock()
        mock_artifact.type = "model_checkpoint"
        mock_artifact.created_at = "2024-06-01T00:00:00"
        mock_artifact.download.return_value = "/tmp/latest/ckpt"

        mock_run = mocker.MagicMock()
        mock_run.logged_artifacts.return_value = [mock_artifact]

        mock_api = mocker.MagicMock()
        mock_api.artifact.side_effect = Exception("not artifact")
        mock_api.run.return_value = mock_run
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.resume import resolve_checkpoint_path

        result = resolve_checkpoint_path("latest")
        assert result == "/tmp/latest/ckpt"
        mock_api.run.assert_called_once_with("e/p/run123")


class TestWandbCheckpointCallback:
    """WandbCheckpointCallback edge cases."""

    def test_on_save_no_wandb_run(self, mocker):
        """on_save returns early when wandb.run is None."""
        mocker.patch("wandb.run", None)
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        args = TrainingArguments(output_dir="/tmp/test")
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_save(args, state, control)
        assert result is None

    def test_on_save_checkpoint_dir_not_exists(self, mocker):
        """on_save returns early when checkpoint dir missing."""
        mocker.patch("wandb.run", mocker.MagicMock())
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        args = TrainingArguments(output_dir="/tmp/nonexistent-dir-xyz")
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_save(args, state, control)
        assert result is None

    def test_on_save_no_checkpoints(self, mocker, tmp_path):
        """on_save returns early when no checkpoint-* dirs found."""
        mocker.patch("wandb.run", mocker.MagicMock())
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_save(args, state, control)
        assert result is None

    def test_on_save_artifact_wait_success(self, mocker, tmp_path):
        """on_save: artifact.wait() succeeds."""
        mock_run = mocker.MagicMock()
        mock_run.name = "test-run"
        mocker.patch("wandb.run", mock_run)

        mock_artifact = mocker.MagicMock()
        mocker.patch("wandb.Artifact", return_value=mock_artifact)
        mocker.patch("wandb.log_artifact")
        mocker.patch("time.sleep")

        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()

        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        state.epoch = 1.0
        state.global_step = 100
        state.max_steps = 1000
        state.log_history = [{"eval_loss": 0.5}]
        control = TrainerControl()

        cb.on_save(args, state, control)

        mock_artifact.wait.assert_called_once_with(timeout=120)

    def test_on_save_artifact_wait_exception(self, mocker, tmp_path):
        """on_save: artifact.wait() raises exception."""
        mock_run = mocker.MagicMock()
        mock_run.name = "test-run"
        mocker.patch("wandb.run", mock_run)

        mock_artifact = mocker.MagicMock()
        mock_artifact.wait.side_effect = Exception("timeout")
        mocker.patch("wandb.Artifact", return_value=mock_artifact)
        mocker.patch("wandb.log_artifact")

        ckpt_dir = tmp_path / "checkpoint-50"
        ckpt_dir.mkdir()

        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbCheckpointCallback

        cb = WandbCheckpointCallback()
        args = TrainingArguments(output_dir=str(tmp_path))
        state = TrainerState()
        state.epoch = 0.5
        state.global_step = 50
        state.max_steps = 500
        state.log_history = []
        control = TrainerControl()

        cb.on_save(args, state, control)

        mock_artifact.wait.assert_called_once_with(timeout=120)


class TestWandbLoggingCallback:
    """WandbLoggingCallback guard branches."""

    def test_on_log_no_wandb_run(self, mocker):
        """on_log returns early when wandb.run is None."""
        mocker.patch("wandb.run", None)
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(output_dir="/tmp/test")
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_log(args, state, control, logs={"loss": 0.5})
        assert result is None

    def test_on_log_no_logs(self, mocker):
        """on_log returns early when logs is empty/None."""
        mocker.patch("wandb.run", mocker.MagicMock())
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(output_dir="/tmp/test")
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_log(args, state, control)
        assert result is None

    def test_on_log_active_log(self, mocker):
        """on_log calls wandb.log with metrics."""
        mock_wandb_log = mocker.patch("wandb.log")
        mocker.patch("wandb.run", mocker.MagicMock())

        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(output_dir="/tmp/test")
        state = TrainerState()
        state.global_step = 10
        state.epoch = 0.5
        control = TrainerControl()

        cb.on_log(args, state, control, logs={"loss": 0.3})
        mock_wandb_log.assert_called_once()

    def test_on_train_begin_no_wandb_run(self, mocker):
        """on_train_begin returns early when wandb.run is None."""
        mocker.patch("wandb.run", None)
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerControl, TrainerState

        from training.callbacks import WandbLoggingCallback

        cb = WandbLoggingCallback()
        args = TrainingArguments(output_dir="/tmp/test")
        state = TrainerState()
        control = TrainerControl()

        result = cb.on_train_begin(args, state, control)
        assert result is None


class TestPromptLoaderEdgeCases:
    """Prompt loader edge cases and W&B artifact paths."""

    def test_init_missing_template_dir(self, tmp_path):
        """PromptLoader raises on missing template dir."""
        from training.prompt_loader import PromptLoader

        bogus_dir = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            PromptLoader(template_dir=bogus_dir)

    def test_get_template_source_missing_file(self):
        """get_template_source raises on missing template name."""
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        with pytest.raises(FileNotFoundError):
            loader.get_template_source("definitely_not_a_template_name")

    def test_log_to_wandb_artifact_no_run(self, mocker):
        """log_to_wandb_artifact returns None when no active run."""
        mocker.patch("wandb.run", None)
        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.log_to_wandb_artifact(version="1.0")
        assert result is None

    def test_log_to_wandb_artifact_success(self, mocker):
        """log_to_wandb_artifact uploads and returns artifact."""
        mocker.patch("wandb.run", mocker.MagicMock())
        mock_artifact = mocker.MagicMock()
        mocker.patch("wandb.Artifact", return_value=mock_artifact)
        mocker.patch("wandb.log_artifact")
        mocker.patch("time.sleep")

        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.log_to_wandb_artifact(version="2.0")

        assert result is mock_artifact
        mock_artifact.wait.assert_called_once_with(timeout=120)

    def test_log_to_wandb_artifact_exception(self, mocker):
        """log_to_wandb_artifact handles wait exception."""
        mocker.patch("wandb.run", mocker.MagicMock())
        mock_artifact = mocker.MagicMock()
        mock_artifact.wait.side_effect = Exception("timeout")
        mocker.patch("wandb.Artifact", return_value=mock_artifact)
        mocker.patch("wandb.log_artifact")

        from training.prompt_loader import PromptLoader

        loader = PromptLoader()
        result = loader.log_to_wandb_artifact(version="3.0")

        assert result is mock_artifact
        mock_artifact.wait.assert_called_once_with(timeout=120)

    def test_load_from_wandb_artifact(self, mocker, tmp_path):
        """load_from_wandb_artifact downloads and constructs loader."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "chat.j2").write_text("chat template")
        (templates_dir / "system.j2").write_text("system template")

        mock_artifact = mocker.MagicMock()

        mock_api = mocker.MagicMock()
        mock_api.artifact.return_value = mock_artifact
        mocker.patch("wandb.Api", return_value=mock_api)

        from training.prompt_loader import PromptLoader

        loader = PromptLoader.load_from_wandb_artifact(
            "entity/project/prompt-templates:v1",
            download_dir=str(templates_dir),
        )

        assert "chat" in loader.available_templates
        assert "system" in loader.available_templates
        mock_api.artifact.assert_called_once_with(
            "entity/project/prompt-templates:v1",
        )


class TestQloraConfigBuild:
    """QLoRA config building edge cases."""

    def test_target_modules_from_model_cfg(self, mocker):
        """target_modules falls back to model cfg when variant has null."""
        mock_model_cfg = {
            "model_id": "Qwen/Qwen3-14B",
            "target_modules": ["q_proj", "v_proj", "o_proj"],
        }
        mock_var_cfg = {
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": None,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training": {
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
            },
        }
        mocker.patch(
            "training.qlora_config._get_model_config",
            return_value=mock_model_cfg,
        )
        mocker.patch(
            "training.qlora_config._get_variant_config",
            return_value=mock_var_cfg,
        )

        from training.qlora_config import build_qlora_config

        lora_config, _ = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
        )
        assert set(lora_config.target_modules) == {"q_proj", "v_proj", "o_proj"}

    def test_target_modules_from_variant(self, mocker):
        """target_modules from variant when present."""
        mock_model_cfg = {
            "model_id": "Qwen/Qwen3-14B",
            "target_modules": ["q_proj", "v_proj", "o_proj"],
        }
        mock_var_cfg = {
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training": {
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
            },
        }
        mocker.patch(
            "training.qlora_config._get_model_config",
            return_value=mock_model_cfg,
        )
        mocker.patch(
            "training.qlora_config._get_variant_config",
            return_value=mock_var_cfg,
        )

        from training.qlora_config import build_qlora_config

        lora_config, _ = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
        )
        assert set(lora_config.target_modules) == {"q_proj", "v_proj"}

    def test_build_with_gpu_override(self, mocker):
        """build_qlora_config applies GPU memory overrides."""
        mock_model_cfg = {
            "model_id": "Qwen/Qwen3-14B",
            "target_modules": ["q_proj", "v_proj", "o_proj"],
        }
        mock_var_cfg = {
            "lora": {
                "r": 16,
                "target_modules": None,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training": {
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
            },
        }
        mocker.patch(
            "training.qlora_config._get_model_config",
            return_value=mock_model_cfg,
        )
        mocker.patch(
            "training.qlora_config._get_variant_config",
            return_value=mock_var_cfg,
        )

        from training.qlora_config import build_qlora_config

        _, training_args = build_qlora_config(
            variant="baseline_14b",
            model_name="qwen3-14b",
            gpu_type="A10G:1",
        )
        assert training_args.max_length == 2048
        assert training_args.per_device_train_batch_size == 6

    def test_build_model_and_peft_gpu_override(self, mocker):
        """build_model_and_peft applies GPU max_seq_length override."""
        mock_model_cfg = {
            "model_id": "Qwen/Qwen3-14B",
            "target_modules": ["q_proj", "v_proj"],
        }
        mock_var_cfg = {
            "lora": {"r": 16, "target_modules": None},
            "training": {},
        }
        mocker.patch(
            "training.qlora_config._get_model_config",
            return_value=mock_model_cfg,
        )
        mocker.patch(
            "training.qlora_config._get_variant_config",
            return_value=mock_var_cfg,
        )
        mock_build = mocker.patch(
            "training.unsloth_factory.build_model_and_peft",
        )

        from training.qlora_config import build_model_and_peft

        build_model_and_peft(
            variant="baseline_14b",
            model_name="qwen3-14b",
            gpu_type="A10G:1",
        )

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["max_seq_length"] == 2048


class TestQloraTrain:
    """QLoRA training CLI entry points."""

    def test_main(self, mocker):
        """main() parses args and calls trainer.train()."""
        mock_trainer_cls = mocker.patch("training.qlora_trainer.QLoRATrainer")
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.train.return_value = {"eval_loss": 0.42}

        from training.qlora_train import main

        main(["--model-name", "qwen3-14b", "--variant", "baseline_14b"])

        mock_trainer_cls.assert_called_once_with(
            model_name="qwen3-14b",
            variant="baseline_14b",
            data_dir="data/tokenized",
            output_dir="/tmp/qlora-output",
            wandb_project="swe-qwen",
            wandb_entity=None,
            run_name=None,
            resume_from_checkpoint=None,
            use_flash_attn=True,
        )
        mock_trainer.train.assert_called_once()

    def test_main_with_resume(self, mocker):
        """main() passes resume flag to trainer."""
        mock_trainer_cls = mocker.patch("training.qlora_trainer.QLoRATrainer")
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.train.return_value = {"eval_loss": 0.33}

        from training.qlora_train import main

        main(["--resume", "entity/project/artifact:v1", "--no-flash-attn"])

        mock_trainer_cls.assert_called_once_with(
            model_name="qwen3-14b",
            variant="baseline_14b",
            data_dir="data/tokenized",
            output_dir="/tmp/qlora-output",
            wandb_project="swe-qwen",
            wandb_entity=None,
            run_name=None,
            resume_from_checkpoint="entity/project/artifact:v1",
            use_flash_attn=False,
        )

    def test_app(self, mocker):
        """app() calls main()."""
        mock_main = mocker.patch("training.qlora_train.main")
        from training.qlora_train import app

        app()
        mock_main.assert_called_once_with()
