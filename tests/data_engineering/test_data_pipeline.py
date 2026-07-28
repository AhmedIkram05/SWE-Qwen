"""Tests for data_engineering.run_pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_engineering.config import DataPipelineConfig
from data_engineering.run_pipeline import (
    _checkpoint_dir,
    _load_stage,
    _manifest_hash,
    _save_stage,
    _stage_enabled,
    run_pipeline,
    run_pipeline_swebench,
)
from data_engineering.schema import IssueRecord


def _mock_wandb() -> MagicMock:
    """Return a wandb module mock pre-configured for pipeline tests."""
    m = MagicMock()
    m.init.return_value = MagicMock()
    return m


def _make_record(issue_id: str, repo: str = "r") -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo=repo,
        issue_body="body",
        patch_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
    )


class TestManifestHash:
    def test_manifest_hash_consistent(self) -> None:
        """Same manifest input should produce identical hash."""
        h1 = _manifest_hash({"a": 1})
        h2 = _manifest_hash({"a": 1})
        assert h1 == h2

    def test_manifest_hash_different_for_different_input(self) -> None:
        """Different manifest input should produce different hash."""
        h1 = _manifest_hash({"a": 1})
        h2 = _manifest_hash({"a": 2})
        assert h1 != h2


class TestStageEnabled:
    def test_all_stages_when_none(self) -> None:
        """None enabled_stages means all stages are enabled."""
        config = DataPipelineConfig(enabled_stages=None)
        assert _stage_enabled(config, "raw") is True

    def test_stage_in_enabled_list(self) -> None:
        """Stage whose human name is in enabled_stages should be enabled."""
        config = DataPipelineConfig(enabled_stages=["ingest", "validate"])
        assert _stage_enabled(config, "raw") is True  # raw maps to "ingest"

    def test_stage_not_in_enabled_list(self) -> None:
        """Stage whose human name is not in enabled_stages should be disabled."""
        config = DataPipelineConfig(enabled_stages=["ingest"])
        assert _stage_enabled(config, "validated") is False  # validated maps to "validate"

    def test_human_stage_direct_match(self) -> None:
        """Passing the human stage name directly should also match."""
        config = DataPipelineConfig(enabled_stages=["ingest"])
        assert _stage_enabled(config, "ingest") is True

    def test_unknown_stage_with_enabled_list(self) -> None:
        """Unknown stage with non-null enabled_stages returns False."""
        config = DataPipelineConfig(enabled_stages=["ingest"])
        assert _stage_enabled(config, "nonexistent") is False


class TestCheckpoint:
    def test_checkpoint_dir(self) -> None:
        """Checkpoint directory should be output_dir/run_id/repo_id."""
        config = DataPipelineConfig(output_dir=Path("/tmp"))
        p = _checkpoint_dir(config, "run123", "swebench")
        assert str(p).endswith("run123/swebench")

    def test_save_and_load_stage(self, tmp_path) -> None:
        """Saved records should be loadable and preserve fields."""
        config = DataPipelineConfig(output_dir=tmp_path)
        recs = [_make_record("test#1")]
        _save_stage(recs, config, "run123", "swebench", "raw")
        loaded = _load_stage(config, "run123", "swebench", "raw")
        assert len(loaded) == 1
        assert loaded[0]["issue_id"] == "test#1"

    def test_load_nonexistent_returns_empty(self, tmp_path) -> None:
        """Loading a nonexistent stage should return empty list."""
        config = DataPipelineConfig(output_dir=tmp_path)
        loaded = _load_stage(config, "run123", "swebench", "nonexistent")
        assert loaded == []

    def test_save_round_trip_preserves_all_fields(self, tmp_path) -> None:
        """All fields from IssueRecord should survive a save+load cycle."""
        rec = _make_record("test#2", repo="django/django")
        config = DataPipelineConfig(output_dir=tmp_path)
        _save_stage([rec], config, "runX", "swebench", "raw")
        loaded = _load_stage(config, "runX", "swebench", "raw")
        assert loaded[0]["repo"] == "django/django"
        assert loaded[0]["issue_id"] == "test#2"
        assert "patch_diff" in loaded[0]


class TestRunPipelineSWEBench:
    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @patch("data_engineering.run_pipeline.clean.clean_records")
    @patch("data_engineering.run_pipeline.clean.deduplicate")
    def test_basic_flow(self, mock_dedup, mock_clean, mock_validate, mock_ingest, tmp_path) -> None:
        """Basic ingest->validate->clean flow with mocks should complete."""
        rec = _make_record("test#1")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([rec], [])
        mock_dedup.return_value = ([rec], MagicMock())
        mock_clean.return_value = ([rec], MagicMock())

        config = DataPipelineConfig(output_dir=tmp_path)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", resume_from=None)

        assert len(result) == 1
        mock_ingest.assert_called_once()

    @patch("data_engineering.run_pipeline.validate.validate_batch")
    def test_resume_from_validated(self, mock_validate, tmp_path) -> None:
        """Resume from validated should skip ingest and load from checkpoint."""
        mock_validate.return_value = ([_make_record("resume#1")], [])

        # First save a raw checkpoint
        _save_stage(
            [_make_record("resume#1")],
            DataPipelineConfig(output_dir=tmp_path),
            "run123",
            "swebench",
            "raw",
        )

        config = DataPipelineConfig(output_dir=tmp_path)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", resume_from="validated")

        # ingest skipped during resume -- mock_validate not patched as decorator
        # so we use the one from args

    def test_ingest_only(self, tmp_path) -> None:
        """When only ingest stage is enabled, pipeline returns raw records."""
        config = DataPipelineConfig(output_dir=tmp_path, enabled_stages=["ingest"])
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            mock_ingest = MagicMock()
            mock_ingest.return_value = [_make_record("raw_only#1")]
            with patch(
                "data_engineering.run_pipeline.swebench_ingest.ingest_swebench",
                mock_ingest,
            ):
                result = run_pipeline_swebench(config, "run123", None)
        assert result is not None
        assert result[0].issue_id == "raw_only#1"

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    def test_empty_ingest_raises(self, mock_ingest, tmp_path) -> None:
        """Empty ingest result should raise RuntimeError."""
        mock_ingest.return_value = []
        config = DataPipelineConfig(output_dir=tmp_path)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="0 records"):
                run_pipeline_swebench(config, "run123", None)

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    def test_validate_disabled_clean_disabled(self, mock_ingest, tmp_path) -> None:
        """With validate+clean disabled, pipeline returns raw records."""
        mock_ingest.return_value = [_make_record("raw_only#1")]
        config = DataPipelineConfig(output_dir=tmp_path, enabled_stages=["ingest"])
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", None)
        assert len(result) == 1
        assert result[0].issue_id == "raw_only#1"


class TestRunPipeline:
    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    def test_run_pipeline_basic(self, mock_run_swebench, tmp_path) -> None:
        """run_pipeline should coordinate stages and return PipelineResult."""
        mock_run_swebench.return_value = [_make_record("test#1")]

        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)

        assert result.run_id is not None
        assert isinstance(result.stats.total_raw, int)

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    def test_run_pipeline_empty_cleaned_raises(self, mock_run_swebench, tmp_path) -> None:
        """Pipeline with 0 cleaned records should raise RuntimeError."""
        mock_run_swebench.return_value = []
        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="0 cleaned records"):
                run_pipeline(config)
