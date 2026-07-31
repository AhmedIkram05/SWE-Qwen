"""Unit tests for QLoRATrainer — no GPU required. All model loading,
training, and W&B calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestQLoRATrainerUnit:
    """Unit tests for QLoRATrainer — mocks all external dependencies."""

    # ── __init__ ──────────────────────────────────────────────────────────────

    def test_init_defaults(self):
        """Defaults match expected values."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer()
        assert trainer.model_name == "qwen3-14b"
        assert trainer.variant == "baseline_14b"
        assert trainer.data_dir == Path("data/tokenized")
        assert trainer.output_dir == Path("/tmp/qlora-output")
        assert trainer.lora_config is None
        assert trainer.training_args is None

    def test_init_with_hf_id(self):
        """hf_id override is stored."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer(hf_id="test/model")
        assert trainer.hf_id == "test/model"

    def test_init_with_prebuilt(self):
        """Pre-built model/tokenizer are stored as _prebuilt_*."""
        from training.qlora_trainer import QLoRATrainer

        model = MagicMock()
        tokenizer = MagicMock()
        trainer = QLoRATrainer(model=model, tokenizer=tokenizer)
        assert trainer._prebuilt_model is model
        assert trainer._prebuilt_tokenizer is tokenizer

    # ── _setup_config ─────────────────────────────────────────────────────────

    def test_setup_config_with_hf_id(self, mocker):
        """When hf_id is set, model_cfg uses it directly (skips _get_model_config)."""
        from training.qlora_trainer import QLoRATrainer

        mock_lora = MagicMock()
        mock_lora.r = 8
        mock_args = MagicMock()
        mocker.patch(
            "training.qlora_trainer.build_qlora_config",
            return_value=(mock_lora, mock_args),
        )

        trainer = QLoRATrainer(hf_id="test/model")
        trainer._setup_config()

        assert trainer.model_cfg == {"hf_id": "test/model"}

    def test_setup_config_without_hf_id(self, mocker):
        """When hf_id is None, _get_model_config is called."""
        from training.qlora_trainer import QLoRATrainer

        mock_lora = MagicMock()
        mock_lora.r = 16
        mock_args = MagicMock()
        mocker.patch(
            "training.qlora_trainer._get_model_config",
            return_value={"hf_id": "real/model"},
        )
        mocker.patch(
            "training.qlora_trainer.build_qlora_config",
            return_value=(mock_lora, mock_args),
        )

        trainer = QLoRATrainer()
        trainer._setup_config()

        assert trainer.model_cfg == {"hf_id": "real/model"}

    def test_setup_config_pre_initialized(self, mocker):
        """When lora_config AND training_args are already set, skip rebuild."""
        from training.qlora_trainer import QLoRATrainer

        mock_build = mocker.patch("training.qlora_trainer.build_qlora_config")
        mocker.patch(
            "training.qlora_trainer._get_model_config",
            return_value={"hf_id": "some/model"},
        )

        trainer = QLoRATrainer()
        trainer.lora_config = MagicMock()
        trainer.training_args = MagicMock()
        trainer._setup_config()

        mock_build.assert_not_called()

    # ── _setup_wandb ──────────────────────────────────────────────────────────

    def test_setup_wandb_missing_api_key(self, monkeypatch):
        """Raises RuntimeError when WANDB_API_KEY env var is missing."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        trainer = QLoRATrainer()
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            trainer._setup_wandb()

    def test_setup_wandb_success(self, mocker, monkeypatch):
        """Happy path: logs in, inits run, creates PromptLoader, logs artifact."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.name = "test-run"

        mock_prompt_loader_cls = mocker.patch("training.qlora_trainer.PromptLoader", autospec=False)
        mock_prompt_loader = MagicMock()
        mock_prompt_loader_cls.return_value = mock_prompt_loader

        trainer = QLoRATrainer()
        trainer.model_name = "test-model"
        trainer.model_cfg = {"hf_id": "test/model"}
        trainer.lora_config = MagicMock()
        trainer.lora_config.r = 8
        trainer.lora_config.lora_alpha = 16
        trainer.training_args = MagicMock()
        trainer.training_args.learning_rate = 1e-4
        trainer.training_args.num_train_epochs = 3
        trainer.training_args.gradient_accumulation_steps = 2

        trainer._setup_wandb()

        mock_wandb.login.assert_called_once_with(key="test-key")
        mock_wandb.init.assert_called_once()
        assert trainer.prompt_loader is mock_prompt_loader
        mock_prompt_loader.log_to_wandb_artifact.assert_called_once_with(version="1.0")

    # ── _setup_model_and_tokenizer ────────────────────────────────────────────

    def test_setup_model_tokenizer_prebuilt(self, mocker):
        """Uses pre-built model/tokenizer, sets pad_token if missing."""
        from training.qlora_trainer import QLoRATrainer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        trainer = QLoRATrainer(model=mock_model, tokenizer=mock_tokenizer)
        trainer._setup_model_and_tokenizer()

        assert trainer.model is mock_model
        assert trainer.tokenizer is mock_tokenizer
        assert trainer.tokenizer.pad_token == "<eos>"
        assert trainer.tokenizer.padding_side == "right"
        mock_model.print_trainable_parameters.assert_called_once()

    def test_setup_model_tokenizer_prebuilt_with_pad(self, mocker):
        """Pre-built tokenizer with pad_token already set keeps it."""
        from training.qlora_trainer import QLoRATrainer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"

        trainer = QLoRATrainer(model=mock_model, tokenizer=mock_tokenizer)
        trainer._setup_model_and_tokenizer()

        assert trainer.tokenizer.pad_token == "<pad>"

    def test_setup_model_tokenizer_cpu_path(self, mocker):
        """CPU path: no quantization, no device_map, full PEFT wrapping."""
        from training.qlora_trainer import QLoRATrainer

        mocker.patch(
            "training.qlora_trainer.torch.cuda.is_available",
            return_value=False,
        )
        mock_model = MagicMock()
        mocker.patch(
            "training.qlora_trainer.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.prepare_model_for_kbit_training",
            return_value=mock_model,
            autospec=False,
        )
        mock_get_peft = mocker.patch(
            "training.qlora_trainer.get_peft_model",
            return_value=mock_model,
            autospec=False,
        )
        mock_tok = MagicMock()
        mock_tok.eos_token = "<eos>"
        mocker.patch(
            "training.qlora_trainer.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
            autospec=False,
        )

        trainer = QLoRATrainer()
        trainer.model_cfg = {"hf_id": "test/model", "compute_dtype": "bfloat16"}
        trainer.lora_config = MagicMock()

        trainer._setup_model_and_tokenizer()

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        mock_get_peft.assert_called_once()

    def test_setup_model_tokenizer_no_lora_config(self, mocker):
        """When lora_config is None, get_peft_model is not called."""
        from training.qlora_trainer import QLoRATrainer

        mocker.patch(
            "training.qlora_trainer.torch.cuda.is_available",
            return_value=False,
        )
        mock_model = MagicMock()
        mocker.patch(
            "training.qlora_trainer.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.prepare_model_for_kbit_training",
            return_value=mock_model,
            autospec=False,
        )
        mock_get_peft = mocker.patch(
            "training.qlora_trainer.get_peft_model",
            autospec=False,
        )
        mock_tok = MagicMock()
        mock_tok.eos_token = "<eos>"
        mocker.patch(
            "training.qlora_trainer.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
            autospec=False,
        )

        trainer = QLoRATrainer()
        trainer.model_cfg = {"hf_id": "test/model"}
        trainer.lora_config = None

        trainer._setup_model_and_tokenizer()

        mock_get_peft.assert_not_called()

    # ── _setup_data ───────────────────────────────────────────────────────────

    def test_setup_data_dir_not_found(self, tmp_path):
        """Raises FileNotFoundError when data_dir does not exist."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer(data_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="Tokenized data directory not found"):
            trainer._setup_data()

    def test_setup_data_no_train_split(self, mocker, tmp_path):
        """Raises ValueError when 'train' split is missing."""
        from training.qlora_trainer import QLoRATrainer

        mock_dataset = MagicMock()
        mock_dataset.get.return_value = None
        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        trainer = QLoRATrainer(data_dir=str(tmp_path))
        trainer.data_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="No 'train' split"):
            trainer._setup_data()

    def test_setup_data_no_eval_dataset(self, mocker, tmp_path):
        """When no val/test split, eval_strategy is set to 'no'."""
        from training.qlora_trainer import QLoRATrainer

        mock_train = MagicMock()
        mock_train.__len__.return_value = 100

        mock_dataset = MagicMock()
        mock_dataset.get.side_effect = lambda key, default=None: {"train": mock_train}.get(
            key, default
        )

        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        trainer = QLoRATrainer(data_dir=str(tmp_path))
        trainer.data_dir.mkdir(parents=True, exist_ok=True)
        trainer.training_args = MagicMock()

        trainer._setup_data()

        assert trainer.eval_dataset is None
        assert trainer.training_args.eval_strategy == "no"

    def test_setup_data_subsample(self, mocker, tmp_path):
        """max_train_samples subsamples the train dataset."""
        from training.qlora_trainer import QLoRATrainer

        mock_sub = MagicMock()
        mock_sub.__len__.return_value = 10
        mock_train = MagicMock()
        mock_train.__len__.return_value = 100
        mock_train.select.return_value = mock_sub

        mock_dataset = MagicMock()
        mock_dataset.get.side_effect = lambda key, default=None: {"train": mock_train}.get(
            key, default
        )

        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        trainer = QLoRATrainer(data_dir=str(tmp_path), max_train_samples=10)
        trainer.data_dir.mkdir(parents=True, exist_ok=True)

        trainer._setup_data()

        mock_train.select.assert_called_once_with(range(10))
        assert len(trainer.train_dataset) == 10

    def test_setup_data_happy_path(self, mocker, tmp_path):
        """Loads both train and eval datasets."""
        from training.qlora_trainer import QLoRATrainer

        mock_train = MagicMock()
        mock_train.__len__.return_value = 100
        mock_eval = MagicMock()
        mock_eval.__len__.return_value = 20

        mock_dataset = MagicMock()
        mock_dataset.get.side_effect = lambda key, default=None: {
            "train": mock_train,
            "val": mock_eval,
        }.get(key, default)

        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        trainer = QLoRATrainer(data_dir=str(tmp_path))
        trainer.data_dir.mkdir(parents=True, exist_ok=True)

        trainer._setup_data()

        assert trainer.train_dataset is not None
        assert trainer.eval_dataset is not None

    # ── _setup_callbacks ──────────────────────────────────────────────────────

    def test_setup_callbacks_args_not_initialized(self):
        """Raises RuntimeError when training_args is None."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer()
        with pytest.raises(RuntimeError, match="TrainingArguments not initialized"):
            trainer._setup_callbacks()

    def test_setup_callbacks_model_not_initialized(self, mocker):
        """Raises RuntimeError when model or tokenizer is None."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer()
        trainer.training_args = MagicMock()
        trainer.model = None
        trainer.tokenizer = None

        with pytest.raises(RuntimeError, match="Model/tokenizer not initialized"):
            trainer._setup_callbacks()

    def test_setup_callbacks_success(self, mocker):
        """Creates SFTTrainer with callbacks."""
        from training.qlora_trainer import QLoRATrainer

        mock_sft = mocker.patch(
            "training.qlora_trainer.SFTTrainer",
            autospec=False,
        )

        trainer = QLoRATrainer()
        trainer.training_args = MagicMock()
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = MagicMock()

        trainer._setup_callbacks()

        mock_sft.assert_called_once()
        assert trainer.trainer is not None

        # Verify callbacks were passed
        call_kwargs = mock_sft.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert len(call_kwargs["callbacks"]) == 2

    # ── setup() integration ───────────────────────────────────────────────────

    def test_setup_full_sequence(self, mocker, monkeypatch, tmp_path):
        """setup() calls all _setup_* methods in order."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")

        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.name = "test-run"

        mocker.patch(
            "training.qlora_trainer.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.prepare_model_for_kbit_training",
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.get_peft_model",
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.build_qlora_config",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("training.qlora_trainer.PromptLoader", autospec=False)
        mocker.patch("training.qlora_trainer.SFTTrainer", autospec=False)

        mock_train = MagicMock()
        mock_train.__len__.return_value = 50
        mock_dataset = MagicMock()
        mock_dataset.get.side_effect = lambda key, default=None: {"train": mock_train}.get(
            key, default
        )
        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        data_dir = Path(str(tmp_path))
        data_dir.mkdir(parents=True, exist_ok=True)

        trainer = QLoRATrainer(data_dir=str(data_dir))
        trainer.setup()

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.train_dataset is not None
        assert trainer.trainer is not None

    # ── train() ───────────────────────────────────────────────────────────────

    def test_train_without_resume(self, mocker, monkeypatch):
        """train() runs training, saves model, returns metrics."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "run-id"

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.5
        mock_train_result.metrics = {
            "train_runtime": 120.0,
            "train_samples_per_second": 10.0,
        }

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_result

        trainer = QLoRATrainer(model=MagicMock(), tokenizer=MagicMock())
        trainer.trainer = mock_sft
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        result = trainer.train()

        mock_sft.train.assert_called_once_with(resume_from_checkpoint=None)
        mock_sft.save_model.assert_called_once()
        assert result["train_loss"] == 0.5
        assert result["train_runtime"] == 120.0
        assert result["train_samples_per_second"] == 10.0
        mock_wandb.log.assert_called_once()

    def test_train_with_resume(self, mocker, monkeypatch, tmp_path):
        """train() resolves checkpoint and passes it to trainer.train()."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "run-id"

        ckpt_path = str(tmp_path / "checkpoint-100")
        mocker.patch(
            "training.resume.resolve_checkpoint_path",
            return_value=ckpt_path,
        )

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.3
        mock_train_result.metrics = {}

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_result

        trainer = QLoRATrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            resume_from_checkpoint="checkpoint-100",
        )
        trainer.trainer = mock_sft
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        result = trainer.train()

        mock_sft.train.assert_called_once_with(resume_from_checkpoint=ckpt_path)
        assert result["train_loss"] == 0.3

    def test_train_with_resume_not_found(self, mocker, monkeypatch):
        """When resolve_checkpoint_path returns None, train starts from scratch."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "run-id"

        mocker.patch(
            "training.resume.resolve_checkpoint_path",
            return_value=None,
        )

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.4
        mock_train_result.metrics = {}

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_result

        trainer = QLoRATrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            resume_from_checkpoint="missing-checkpoint",
        )
        trainer.trainer = mock_sft
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        result = trainer.train()

        mock_sft.train.assert_called_once_with(resume_from_checkpoint=None)
        assert result["train_loss"] == 0.4

    def test_train_triggers_setup_when_no_trainer(self, mocker, monkeypatch, tmp_path):
        """When trainer is None, train() calls setup() first."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.name = "test-run"

        mocker.patch(
            "training.qlora_trainer.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.prepare_model_for_kbit_training",
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.get_peft_model",
            autospec=False,
        )
        mocker.patch(
            "training.qlora_trainer.build_qlora_config",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("training.qlora_trainer.PromptLoader", autospec=False)

        mock_sft = mocker.patch("training.qlora_trainer.SFTTrainer", autospec=False)
        mock_sft_instance = MagicMock()
        mock_sft.return_value = mock_sft_instance
        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.2
        mock_train_result.metrics = {}
        mock_sft_instance.train.return_value = mock_train_result

        mock_train = MagicMock()
        mock_train.__len__.return_value = 50
        mock_dataset = MagicMock()
        mock_dataset.get.side_effect = lambda key, default=None: {"train": mock_train}.get(
            key, default
        )
        mocker.patch(
            "training.qlora_trainer.load_from_disk",
            return_value=mock_dataset,
        )

        data_dir = Path(str(tmp_path))
        data_dir.mkdir(parents=True, exist_ok=True)

        trainer = QLoRATrainer(data_dir=str(data_dir))
        assert trainer.trainer is None

        result = trainer.train()

        assert trainer.trainer is not None
        assert result["train_loss"] == 0.2

    # ── save_model() ──────────────────────────────────────────────────────────

    def test_save_model_no_trainer(self, tmp_path):
        """When trainer is None, logs warning and returns."""
        from training.qlora_trainer import QLoRATrainer

        trainer = QLoRATrainer(output_dir=str(tmp_path / "no-trainer"))
        # Should not raise
        trainer.save_model()

    def test_save_model_no_wandb_run(self, mocker):
        """When wandb.run is None, saves locally, skips artifact logging."""
        from training.qlora_trainer import QLoRATrainer

        mock_trainer = MagicMock()
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = None

        trainer = QLoRATrainer(output_dir="/tmp/test-save-no-wandb")
        trainer.model_name = "test"
        trainer.variant = "test"
        trainer.trainer = mock_trainer
        trainer.lora_config = MagicMock()

        trainer.save_model()

        mock_trainer.save_model.assert_called_once()

    def test_save_model_with_wandb_artifact(self, mocker):
        """Logs artifact to W&B, waits for it."""
        from training.qlora_trainer import QLoRATrainer

        mock_trainer = MagicMock()
        mock_artifact_instance = MagicMock()

        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact_instance

        trainer = QLoRATrainer(output_dir="/tmp/test-wandb-artifact")
        trainer.model_name = "test-model"
        trainer.variant = "test-variant"
        trainer.lora_config = MagicMock()
        trainer.lora_config.r = 8
        trainer.lora_config.lora_alpha = 16
        trainer.trainer = mock_trainer

        trainer.save_model()

        mock_wandb.Artifact.assert_called_once()
        mock_artifact_instance.add_dir.assert_called_once()
        mock_wandb.log_artifact.assert_called_once_with(mock_artifact_instance)
        mock_artifact_instance.wait.assert_called_once_with(timeout=120)

    def test_save_model_artifact_timeout(self, mocker):
        """Handles artifact.wait() timeout gracefully."""
        from training.qlora_trainer import QLoRATrainer

        mock_artifact_instance = MagicMock()
        mock_artifact_instance.wait.side_effect = TimeoutError("timed out")

        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact_instance

        trainer = QLoRATrainer(output_dir="/tmp/test-timeout")
        trainer.model_name = "test-model"
        trainer.variant = "test-variant"
        trainer.lora_config = MagicMock()
        trainer.trainer = MagicMock()

        # Should not raise
        trainer.save_model()

        mock_artifact_instance.wait.assert_called_once_with(timeout=120)

    # ── resume() ──────────────────────────────────────────────────────────────

    def test_resume_delegates_to_train(self, mocker, monkeypatch):
        """resume() sets checkpoint and delegates to train()."""
        from training.qlora_trainer import QLoRATrainer

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        mock_wandb = mocker.patch("training.qlora_trainer.wandb", autospec=False)
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "run-id"

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.4
        mock_train_result.metrics = {}

        mock_sft = MagicMock()
        mock_sft.train.return_value = mock_train_result

        trainer = QLoRATrainer(model=MagicMock(), tokenizer=MagicMock())
        trainer.trainer = mock_sft
        trainer.output_dir.mkdir(parents=True, exist_ok=True)

        result = trainer.resume("checkpoint-200")

        assert trainer.resume_from_checkpoint == "checkpoint-200"
        mock_sft.train.assert_called_once()
        assert result["train_loss"] == 0.4
