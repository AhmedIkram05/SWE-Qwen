"""Tests for data_engineering.config."""

from __future__ import annotations

from unittest import mock

import pytest

from data_engineering.config import DataPipelineConfig


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
