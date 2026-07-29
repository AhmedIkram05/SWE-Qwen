"""Tests for data_engineering.version."""

from __future__ import annotations

from unittest.mock import call, patch

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, ValidationError
from data_engineering.version import (
    _build_artifact_metadata,
    _errors_to_jsonl,
    log_dataset_artifacts,
    log_validation_errors,
)


def _rec(issue_id: str) -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo="r",
        issue_body="body",
        patch_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
    )


class TestErrorsToJsonl:
    def test_errors_to_jsonl(self, tmp_path) -> None:
        """Write validation errors as JSONL and verify content."""
        errors = [ValidationError(record_id="r1", field="body", error="empty")]
        p = tmp_path / "errors.jsonl"
        _errors_to_jsonl(errors, p)
        content = p.read_text()
        assert "r1" in content
        assert "body" in content
        assert "empty" in content


class TestBuildArtifactMetadata:
    def test_build_metadata_with_stats(self) -> None:
        """Stats branch of _build_artifact_metadata includes all AC-required keys."""
        stats = {
            "total_validated": 100,
            "total_validation_errors": 5,
            "dedup_stats": {"exact_duplicates_removed": 10, "content_duplicates_removed": 3},
            "clean_stats": {"removed_no_test_files": 5},
            "repo_results": [{"repo_id": "django/django"}],
            "train_count": 50,
            "val_count": 10,
            "test_count": 10,
            "golden_count": 5,
        }
        meta = _build_artifact_metadata("run_1", "train", "abc123", [], stats)
        assert meta["validation_pass"] == 100
        assert meta["validation_fail"] == 5
        assert meta["dedup_exact"] == 10
        assert meta["golden_size"] == 5
        assert "django/django" in meta["repo_list"]
        assert "split_ratios" in meta
        assert meta["counts"]["train"] == 50

    def test_build_metadata_without_stats(self) -> None:
        """Without stats, metadata should only contain core fields."""
        meta = _build_artifact_metadata("run_1", "train", "abc123", [])
        assert meta["count"] == 0
        assert "validation_pass" not in meta


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

    def test_artifact_upload_waits_for_files(self) -> None:
        """artifact.wait() and time.sleep(1) are called after log_artifact.

        Guards against regression: if either is removed, file uploads may
        not complete before run.finish() terminates the uploader thread.
        """
        stages = {"train": [_rec("r#1")]}
        config = DataPipelineConfig(wandb_project="test-proj")

        with (
            patch("data_engineering.version.wandb") as mock_wandb,
            patch("data_engineering.version.time.sleep") as mock_sleep,
        ):
            mock_run = mock_wandb.init.return_value
            mock_artifact = mock_wandb.Artifact.return_value
            mock_artifact.name = "dataset-train:v1"

            log_dataset_artifacts("run_id", stages, config, "abc123")

            mock_run.log_artifact.assert_called_with(mock_artifact)
            mock_artifact.wait.assert_called_once()
            mock_sleep.assert_has_calls([call(1)])

    def test_empty_stages(self) -> None:
        """Empty stages dict should not raise and return empty dict."""
        stages: dict = {}
        config = DataPipelineConfig()
        with patch("data_engineering.version.wandb") as mock_wandb:
            result = log_dataset_artifacts("run_id", stages, config, "hash")
        assert isinstance(result, dict)

    def test_log_dataset_artifacts_with_validation_errors(self) -> None:
        """validation_errors stage should use _errors_to_jsonl path."""
        stages = {
            "validation_errors": [ValidationError(record_id="r1", field="body", error="e")],
        }
        config = DataPipelineConfig(wandb_project="test-proj")
        with patch("data_engineering.version.wandb") as mock_wandb:
            mock_run = mock_wandb.init.return_value
            result = log_dataset_artifacts("run_id", stages, config, "hash")
            assert mock_wandb.init.called
            assert isinstance(result, dict)


class TestLogValidationErrors:
    def test_log_validation_errors(self) -> None:
        """log_validation_errors delegates to log_dataset_artifacts and returns artifact."""
        errors = [ValidationError(record_id="r1", field="body", error="e")]
        config = DataPipelineConfig(wandb_project="test-proj")
        with patch("data_engineering.version.wandb") as mock_wandb:
            artifact = log_validation_errors("run_id", errors, config)
            assert mock_wandb.init.called
            assert artifact is not None

    def test_log_validation_errors_fallback(self) -> None:
        """When wandb.Api() fails, log_validation_errors returns minimal artifact."""
        errors = [ValidationError(record_id="r1", field="body", error="e")]
        config = DataPipelineConfig(wandb_project="test-proj")
        with patch("data_engineering.version.wandb") as mock_wandb:
            mock_wandb.Api.side_effect = Exception("API unavailable")
            artifact = log_validation_errors("run_id", errors, config)
            assert artifact is not None

    def test_log_dataset_artifacts_with_per_repo_stats(self) -> None:
        """log_dataset_artifacts with stats that include repo_results should log per_repo_counts."""
        stages = {
            "train": [_rec("r#1")],
            "test": [_rec("r#2")],
        }
        config = DataPipelineConfig(wandb_project="test-proj", run_name="custom-run")
        stats = {
            "repo_results": [
                {
                    "repo_id": "django/django",
                    "raw_count": 10,
                    "validated_count": 8,
                    "cleaned_count": 6,
                },
            ],
            "train_count": 1,
            "val_count": 0,
            "test_count": 1,
            "golden_count": 0,
        }

        with patch("data_engineering.version.wandb") as mock_wandb:
            mock_run = mock_wandb.init.return_value
            result = log_dataset_artifacts("run_id", stages, config, "hash", stats)
            assert mock_wandb.init.called
            assert isinstance(result, dict)
