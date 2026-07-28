"""Tests for data_engineering.cli."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from typer.testing import CliRunner

from data_engineering.cli import app
from data_engineering.schema import PipelineResult, PipelineStats, Splits

runner = CliRunner()


class TestCli:
    """CLI command tests via typer.testing.CliRunner."""

    def test_help(self) -> None:
        """Running with --help should succeed."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SWE-Qwen pipeline" in result.stdout

    def test_config_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running config subcommand should show effective config."""
        monkeypatch.setenv("DATA_PIPELINE_BIGQUERY_ENABLED", "false")
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["batch_size"] == 50
        assert data["max_issues_per_repo"] == 2000
        assert data["wandb_project"] == "swe-qwen-data"
        assert data["train_ratio"] == 0.8
        assert data["val_ratio"] == 0.1
        assert data["test_ratio"] == 0.1
        assert data["parallel_workers"] == 1
        assert data["bigquery_enabled"] is False
        assert data["resume_from"] is None
        assert data["enabled_stages"] is None

    # ── run command: successful flow ─────────────────────────────────────

    def test_run_command_valid(self) -> None:
        """run should print JSON summary when pipeline succeeds."""
        mock_result = PipelineResult(
            run_id="test-run-001",
            manifest_hash="abc123def456",
            splits=Splits(),
            stats=PipelineStats(
                total_raw=100,
                total_validated=80,
                total_cleaned=60,
                total_examples=55,
                train_count=30,
                val_count=10,
                test_count=15,
            ),
            gcs_paths={"raw": "gs://bucket/raw.jsonl"},
            wandb_artifacts={"golden": "wandb-artifact://golden-v1"},
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result):
            result = runner.invoke(app, ["run"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["run_id"] == "test-run-001"
        assert data["manifest_hash"] == "abc123def456"
        assert data["stats"]["total_raw"] == 100
        assert data["stats"]["train_count"] == 30
        assert data["gcs_paths"]["raw"] == "gs://bucket/raw.jsonl"
        assert data["wandb_artifacts"]["golden"] == "wandb-artifact://golden-v1"

    # ── run command: RuntimeError ────────────────────────────────────────

    def test_run_command_runtime_error(self) -> None:
        """run should exit code 1 and print error on RuntimeError."""
        with mock.patch(
            "data_engineering.cli.run_pipeline",
            side_effect=RuntimeError("GCP credentials not found"),
        ):
            result = runner.invoke(app, ["run"])

        assert result.exit_code == 1
        assert "GCP credentials not found" in result.stderr

    # ── run command: stages parameter split ──────────────────────────────

    def test_run_command_stages_splits_correctly(self) -> None:
        """--stages splits CSV into list, verified via mock call_args."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        mock_run = mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result)
        with mock_run as m:
            runner.invoke(app, ["run", "--stages", "ingest,validate,clean"])
            cfg = m.call_args[0][0]
            assert cfg.enabled_stages == ["ingest", "validate", "clean"]

    def test_run_command_default_stages_is_none(self) -> None:
        """Omitting --stages leaves enabled_stages as None (= all stages)."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result) as m:
            runner.invoke(app, ["run"])
            cfg = m.call_args[0][0]
            assert cfg.enabled_stages is None

    def test_run_command_single_stage(self) -> None:
        """--stages with one value produces single-element list."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result) as m:
            runner.invoke(app, ["run", "--stages", "ingest"])
            cfg = m.call_args[0][0]
            assert cfg.enabled_stages == ["ingest"]

    # ── run command: CLI option forwarding ───────────────────────────────

    def test_run_command_forwards_all_options(self) -> None:
        """CLI options should be forwarded into DataPipelineConfig."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result) as m:
            runner.invoke(
                app,
                [
                    "run",
                    "--swe-bench-dir",
                    "/tmp/swe",
                    "--output",
                    "/tmp/out",
                    "--max-issues",
                    "500",
                    "--parallel-workers",
                    "4",
                    "--batch-size",
                    "25",
                    "--max-patch-lines",
                    "300",
                    "--min-golden",
                    "100",
                    "--resume-from",
                    "validated",
                    "--run-id",
                    "abc-123",
                    "--train-ratio",
                    "0.7",
                    "--val-ratio",
                    "0.15",
                    "--test-ratio",
                    "0.15",
                    "--bigquery",
                ],
            )
            cfg = m.call_args[0][0]
            assert str(cfg.swe_bench_dir) == "/tmp/swe"
            assert str(cfg.output_dir) == "/tmp/out"
            assert cfg.max_issues_per_repo == 500
            assert cfg.parallel_workers == 4
            assert cfg.batch_size == 25
            assert cfg.max_patch_lines == 300
            assert cfg.min_golden_examples == 100
            assert cfg.resume_from == "validated"
            assert cfg.run_id_override == "abc-123"
            assert cfg.train_ratio == 0.7
            assert cfg.val_ratio == 0.15
            assert cfg.test_ratio == 0.15
            assert cfg.bigquery_enabled is True

    # ── Verbose mode ─────────────────────────────────────────────────────

    def test_run_verbose_passes_to_setup_logging(self) -> None:
        """--verbose should call _setup_logging(True)."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with (
            mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result),
            mock.patch("data_engineering.cli._setup_logging") as mock_setup,
        ):
            runner.invoke(app, ["run", "--verbose"])
        mock_setup.assert_called_once_with(True)

    def test_run_non_verbose_passes_to_setup_logging(self) -> None:
        """Omitting --verbose should call _setup_logging(False)."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with (
            mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result),
            mock.patch("data_engineering.cli._setup_logging") as mock_setup,
        ):
            runner.invoke(app, ["run"])
        mock_setup.assert_called_once_with(False)

    def test_run_short_verbose_flag(self) -> None:
        """-v should work like --verbose."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result):
            result = runner.invoke(app, ["run", "-v"])
        assert result.exit_code == 0

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_run_command_no_args_uses_defaults(self) -> None:
        """run with no options should use DataPipelineConfig defaults."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result) as m:
            runner.invoke(app, ["run"])
            cfg = m.call_args[0][0]
            assert cfg.batch_size == 50
            assert cfg.max_issues_per_repo == 2000
            assert cfg.parallel_workers == 1
            assert cfg.train_ratio == 0.8
            assert cfg.val_ratio == 0.1
            assert cfg.test_ratio == 0.1

    def test_config_shows_env_overrides(self, monkeypatch) -> None:
        """config subcommand should reflect DATA_PIPELINE_ env vars."""
        monkeypatch.setenv("DATA_PIPELINE_BATCH_SIZE", "999")
        monkeypatch.setenv("DATA_PIPELINE_WANDB_PROJECT", "env-override")
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["batch_size"] == 999
        assert data["wandb_project"] == "env-override"

    def test_run_command_zero_values(self) -> None:
        """run accepts edge values like min-golden 0, batch-size 1."""
        mock_result = PipelineResult(
            run_id="s",
            manifest_hash="h",
            splits=Splits(),
            stats=PipelineStats(),
        )
        with mock.patch("data_engineering.cli.run_pipeline", return_value=mock_result):
            result = runner.invoke(app, ["run", "--min-golden", "0", "--batch-size", "1"])
        assert result.exit_code == 0
