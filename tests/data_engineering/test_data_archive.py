"""Tests for data_engineering.archive.

``google.cloud.storage`` is imported lazily inside ``_ensure_gcs_bucket``,
so we patch it at the source to avoid real GCS calls.
"""

from __future__ import annotations

from unittest.mock import patch

from data_engineering.archive import upload_to_gcs
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord


def _rec(issue_id: str) -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo="r",
        issue_body="body",
        patch_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n",
    )


class TestUploadToGcs:
    def test_uploads_stages(self) -> None:
        stages = {
            "train": [_rec("r#1")],
            "test": [_rec("r#2")],
        }
        manifest: dict = {"version": "1", "repositories": []}
        config = DataPipelineConfig(gcs_bucket="test-bucket")

        # Patch google.cloud.storage (lazy-imported inside _ensure_gcs_bucket)
        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.blob.return_value

            result = upload_to_gcs(
                "run_123",
                stages,
                manifest,
                "# Dataset Card\nContent",
                config,
            )

            mock_client_cls.assert_called_once()
            assert mock_bucket.blob.called
            assert isinstance(result, dict)

    def test_no_bucket_skips(self) -> None:
        """Empty bucket name should skip quietly, returning {}. Covers log+return path."""
        config = DataPipelineConfig(gcs_bucket="")
        result = upload_to_gcs("r", {}, {}, "card", config)
        assert result == {}

    def test_uploads_dataset_card(self) -> None:
        stages: dict = {}
        config = DataPipelineConfig(gcs_bucket="bucket")

        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value

            result = upload_to_gcs(
                "run_123",
                stages,
                {},
                "# Card",
                config,
            )
            mock_bucket.blob.assert_called()
