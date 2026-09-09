"""Tests for data_engineering.config."""

from __future__ import annotations

from unittest import mock

import pytest

from data_engineering.config import DataPipelineConfig
from registry.loader import ModelSpec


class TestDataPipelineConfig:
    """Tests for DataPipelineConfig model and its validate_auth method."""

    # ── Default values ───────────────────────────────────────────────────

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All fields should have sensible defaults (env clean)."""
        monkeypatch.setenv("DATA_PIPELINE_GCS_BUCKET", "")
        monkeypatch.setenv("DATA_PIPELINE_BIGQUERY_ENABLED", "false")
        monkeypatch.setenv("DATA_PIPELINE_BIGQUERY_PROJECT", "")
        cfg = DataPipelineConfig()
        assert cfg.batch_size == 50
        assert cfg.max_patch_lines == 500
        assert cfg.min_golden_examples == 100
        assert cfg.parallel_workers == 1
        assert cfg.max_issues_per_repo == 2000
        assert cfg.max_events_per_issue == 100
        assert cfg.gcs_bucket == ""
        assert str(cfg.output_dir) == "data"
        assert cfg.wandb_project == "swe-qwen-data"
        assert cfg.wandb_entity is None
        assert cfg.golden_source_split == "verified+test+dev"
        assert cfg.train_ratio == 0.8
        assert cfg.val_ratio == 0.1
        assert cfg.test_ratio == 0.1
        assert str(cfg.swe_bench_dir) == "data/swe_bench"
        assert cfg.swe_bench_version == "2025-04-29"
        assert cfg.bigquery_enabled is False
        assert cfg.bigquery_project == ""
        assert cfg.resume_from is None
        assert cfg.enabled_stages is None
        assert cfg.run_id_override is None
        assert cfg.run_name is None

    # ── Env var overrides ────────────────────────────────────────────────

    def test_env_var_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATA_PIPELINE_* env vars should override defaults."""
        monkeypatch.setenv("DATA_PIPELINE_BATCH_SIZE", "100")
        monkeypatch.setenv("DATA_PIPELINE_WANDB_PROJECT", "custom-project")
        monkeypatch.setenv("DATA_PIPELINE_PARALLEL_WORKERS", "8")
        monkeypatch.setenv("DATA_PIPELINE_TRAIN_RATIO", "0.7")
        monkeypatch.setenv("DATA_PIPELINE_BIGQUERY_ENABLED", "true")
        monkeypatch.setenv("DATA_PIPELINE_RUN_NAME", "test-run")

        cfg = DataPipelineConfig()
        assert cfg.batch_size == 100
        assert cfg.wandb_project == "custom-project"
        assert cfg.parallel_workers == 8
        assert cfg.train_ratio == 0.7
        assert cfg.bigquery_enabled is True
        assert cfg.run_name == "test-run"
        # Unset field stays default
        assert cfg.max_patch_lines == 500

    def test_env_var_boolean_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATA_PIPELINE_BIGQUERY_ENABLED=false should parse to False."""
        monkeypatch.setenv("DATA_PIPELINE_BIGQUERY_ENABLED", "false")
        cfg = DataPipelineConfig()
        assert cfg.bigquery_enabled is False

    def test_env_var_path_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Path-typed fields should accept env var overrides."""
        monkeypatch.setenv("DATA_PIPELINE_OUTPUT_DIR", "/tmp/custom-out")
        monkeypatch.setenv("DATA_PIPELINE_SWE_BENCH_DIR", "/tmp/custom-swe")
        cfg = DataPipelineConfig()
        assert str(cfg.output_dir) == "/tmp/custom-out"
        assert str(cfg.swe_bench_dir) == "/tmp/custom-swe"

    # ── validate_auth: all present ───────────────────────────────────────

    def test_validate_auth_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty list when WANDB_API_KEY set and GCP ADC works."""
        monkeypatch.setenv("WANDB_API_KEY", "test-key-123")
        with mock.patch("google.auth.default", return_value=(None, None)):
            cfg = DataPipelineConfig()
            missing = cfg.validate_auth()
            assert missing == []

    # ── validate_auth: missing WANDB_API_KEY ─────────────────────────────

    def test_validate_auth_missing_wandb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WANDB_API_KEY reported missing when env var absent."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        with mock.patch("google.auth.default", return_value=(None, None)):
            with mock.patch("dotenv.load_dotenv"):
                cfg = DataPipelineConfig()
                missing = cfg.validate_auth()
                assert missing == ["WANDB_API_KEY"]

    # ── validate_auth: GCP auth failure ──────────────────────────────────

    def test_validate_auth_gcp_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GCP_ADC reported missing when google.auth.default raises."""
        monkeypatch.setenv("WANDB_API_KEY", "test-key-123")
        with mock.patch(
            "google.auth.default",
            side_effect=Exception("Application Default Credentials not available"),
        ):
            cfg = DataPipelineConfig()
            missing = cfg.validate_auth()
            assert missing == ["GCP_ADC"]

    # ── validate_auth: both missing ──────────────────────────────────────

    def test_validate_auth_both_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both WANDB_API_KEY and GCP_ADC missing."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        with mock.patch(
            "google.auth.default",
            side_effect=Exception("no ADC"),
        ):
            with mock.patch("dotenv.load_dotenv"):
                cfg = DataPipelineConfig()
                missing = cfg.validate_auth()
                assert missing == ["WANDB_API_KEY", "GCP_ADC"]

    # ── validate_auth: load_dotenv is called ─────────────────────────────

    def test_validate_auth_load_dotenv_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_auth should call load_dotenv before checking env."""
        monkeypatch.setenv("WANDB_API_KEY", "test-key-123")
        with mock.patch("google.auth.default", return_value=(None, None)):
            with mock.patch("dotenv.load_dotenv") as mock_load:
                cfg = DataPipelineConfig()
                cfg.validate_auth()
                mock_load.assert_called_once_with(".env")

    # ── Integration marker (CI only) ─────────────────────────────────────

    @pytest.mark.requires_credentials
    def test_validate_auth_integration(self) -> None:
        """Live check: real env vars (CI-only)."""
        cfg = DataPipelineConfig()
        missing = cfg.validate_auth()
        assert isinstance(missing, list)


class TestTokenizeRegistryResolution:
    """PHASE-10 Steps 2/9-3: registry-first, env above, literals last resort."""

    def test_defaults_follow_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATA_PIPELINE_TOKENIZE_MODEL", raising=False)
        monkeypatch.delenv("DATA_PIPELINE_TOKENIZE_MAX_LENGTH", raising=False)
        cfg = DataPipelineConfig()
        assert cfg.tokenize_model == "qwen3-14b"  # registry default: true
        assert cfg.tokenize_max_length == 32768  # registry context_window

    def test_env_wins_over_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATA_PIPELINE_TOKENIZE_MODEL", "qwen3-30b-a3b")
        monkeypatch.setenv("DATA_PIPELINE_TOKENIZE_MAX_LENGTH", "1024")
        cfg = DataPipelineConfig()
        assert cfg.tokenize_model == "qwen3-30b-a3b"
        assert cfg.tokenize_max_length == 1024

    def test_fresh_model_context_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = ModelSpec.model_validate(
            {
                "hf_id": "meta-llama/Fresh-8B",
                "context_window": 8192,
                "target_modules": ["q_proj"],
            }
        )
        monkeypatch.setattr("data_engineering.config.load_models", lambda: {"fresh": spec})
        monkeypatch.setattr("data_engineering.config.default_model_key", lambda: "fresh")
        cfg = DataPipelineConfig()
        assert cfg.tokenize_model == "fresh"
        assert cfg.tokenize_max_length == 8192

    def test_unknown_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATA_PIPELINE_TOKENIZE_MODEL", "does-not-exist")
        with pytest.raises(ValueError, match="unknown model 'does-not-exist'"):
            DataPipelineConfig()

    def test_registry_missing_uses_literals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_args, **_kwargs):
            raise OSError("no registry")

        monkeypatch.setattr("data_engineering.config.load_models", fail)
        monkeypatch.setattr("data_engineering.config.default_model_key", fail)
        monkeypatch.delenv("DATA_PIPELINE_TOKENIZE_MODEL", raising=False)
        monkeypatch.delenv("DATA_PIPELINE_TOKENIZE_MAX_LENGTH", raising=False)
        cfg = DataPipelineConfig()
        assert cfg.tokenize_model == "qwen3-14b"
        assert cfg.tokenize_max_length == 4096
