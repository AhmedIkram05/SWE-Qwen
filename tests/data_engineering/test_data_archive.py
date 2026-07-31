"""Tests for data_engineering.archive.

``google.cloud.storage`` is imported lazily inside ``_ensure_gcs_bucket``,
so we patch it at the source to avoid real GCS calls.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import patch

from data_engineering.archive import upload_text_to_gcs, upload_to_gcs
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

    def test_upload_jsonl_logs_hash(self) -> None:
        """Verify _upload_jsonl logs debug message with md5 hash and record count."""
        stages = {
            "train": [_rec("r#1")],
        }
        config = DataPipelineConfig(gcs_bucket="bucket")

        with (
            patch("google.cloud.storage.Client") as mock_client_cls,
            patch("data_engineering.archive.logger.debug") as mock_debug,
            patch("hashlib.md5") as mock_md5,
        ):
            mock_md5.return_value.hexdigest.return_value = "abc123"
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "bucket"

            upload_to_gcs("run_1", stages, {}, "card", config)

            mock_debug.assert_any_call(
                "Uploaded %s (%d records, md5=%s)",
                mock.ANY,
                mock.ANY,
                mock.ANY,
            )

    def test_upload_jsonl_with_dict_fallback(self) -> None:
        """Record with .dict() method (pydantic v1) should use dict() fallback."""

        class _OldModel:
            def dict(self):
                return {"issue_id": "old#1"}

        stages = {"train": [_OldModel()]}
        config = DataPipelineConfig(gcs_bucket="bucket")
        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "bucket"
            result = upload_to_gcs("run_1", stages, {}, "card", config)
            assert "train" in result

    def test_upload_jsonl_with_plain_dict(self) -> None:
        """Plain dict records should be handled (neither model_dump nor dict)."""
        stages = {"train": [{"issue_id": "dict#1"}]}
        config = DataPipelineConfig(gcs_bucket="bucket")
        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "bucket"
            result = upload_to_gcs("run_1", stages, {}, "card", config)
            assert "train" in result

    def test_skips_empty_stages(self) -> None:
        """Empty stage lists should not call _upload_jsonl (no blob for that stage)."""
        stages = {
            "train": [],  # empty — should be skipped
            "test": [_rec("r#1")],  # non-empty — should upload
        }
        config = DataPipelineConfig(gcs_bucket="bucket")

        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "bucket"

            result = upload_to_gcs("run_1", stages, {}, "card", config)

            # train blob should NOT be created; test blob should
            train_calls = [c for c in mock_bucket.blob.call_args_list if "train" in c[0][0]]
            test_calls = [c for c in mock_bucket.blob.call_args_list if "test" in c[0][0]]
            assert len(train_calls) == 0, "empty stage should not create a blob"
            assert len(test_calls) == 1, "non-empty stage should create a blob"
            # Manifest + dataset card blobs should always be created
            assert "dataset_card.md" in str(result.get("dataset_card", ""))
            assert "manifest.json" in str(result.get("manifest", ""))


class TestUploadTextToGcs:
    def test_upload_text_to_gcs(self) -> None:
        """upload_text_to_gcs uploads text and returns gs:// path."""
        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "bucket"
            mock_blob = mock_bucket.blob.return_value

            result = upload_text_to_gcs("bucket", "prefix", "test.md", "# content")

            mock_bucket.blob.assert_called_once_with("prefix/test.md")
            mock_blob.upload_from_string.assert_called_once_with(
                "# content", content_type="text/markdown; charset=utf-8"
            )
            assert result == "gs://bucket/prefix/test.md"

    def test_creates_bucket_when_not_exists(self) -> None:
        """When GCS bucket doesn't exist, _ensure_gcs_bucket should create it."""
        stages = {"train": [_rec("r#1")]}
        config = DataPipelineConfig(gcs_bucket="new-bucket")

        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_bucket = mock_client.bucket.return_value
            mock_bucket.name = "new-bucket"
            mock_bucket.exists.return_value = False  # trigger create path

            result = upload_to_gcs("run_1", stages, {}, "card", config)

            mock_bucket.exists.assert_called_once()
            mock_client.create_bucket.assert_called_once_with("new-bucket", location="US-CENTRAL1")
            assert isinstance(result, dict)
            assert "train" in result
