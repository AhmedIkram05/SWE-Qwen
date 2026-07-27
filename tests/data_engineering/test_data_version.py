"""Tests for data_engineering.version."""

from __future__ import annotations

from unittest.mock import patch

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord
from data_engineering.version import log_dataset_artifacts


def _rec(issue_id: str) -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo="r",
        issue_body="body",
        patch_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
    )


class TestLogDatasetArtifacts:
    def test_logs_artifacts(self) -> None:
        """Verify function calls wandb APIs correctly."""
        stages = {
            "train": [_rec("r#1")],
            "val": [],
            "test": [_rec("r#2")],
            "golden": [],
        }
        config = DataPipelineConfig(wandb_project="test-proj")

        with patch("data_engineering.version.wandb") as mock_wandb:
            mock_run = mock_wandb.init.return_value
            mock_artifact_cls = mock_wandb.Artifact
            mock_artifact = mock_artifact_cls.return_value
            mock_artifact.name = "test-artifact"

            result = log_dataset_artifacts("run_id_123", stages, config, "abc123")

            mock_wandb.init.assert_called_once()
            assert mock_artifact.add_file.called
            assert mock_run.log_artifact.called
            assert isinstance(result, dict)

    def test_empty_stages(self) -> None:
        stages: dict = {}
        config = DataPipelineConfig()
        with patch("data_engineering.version.wandb") as mock_wandb:
            result = log_dataset_artifacts("run_id", stages, config, "hash")
        assert isinstance(result, dict)
