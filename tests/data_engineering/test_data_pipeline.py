"""Tests for data_engineering.run_pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_engineering.config import DataPipelineConfig
from data_engineering.run_pipeline import (
    _checkpoint_dir,
    _load_stage,
    _load_stage_gcs,
    _manifest_hash,
    _save_splits_jsonl,
    _save_stage,
    _save_stage_gcs,
    _stage_enabled,
    run_pipeline,
    run_pipeline_swebench,
)
from data_engineering.schema import (
    GoldenSet,
    IssueRecord,
    Splits,
    ValidationError,
)


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
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    @pytest.mark.slow
    def test_run_pipeline_basic(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """run_pipeline should coordinate stages and return PipelineResult."""
        mock_run_swebench.return_value = [_make_record("test#1")]

        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
            augment_codecontests=False,
            augment_codealpaca=False,
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)

        assert result.run_id is not None
        assert isinstance(result.stats.total_raw, int)

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_run_pipeline_empty_cleaned_raises(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """Pipeline with 0 cleaned records should raise RuntimeError."""
        mock_run_swebench.return_value = []
        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
            augment_codecontests=False,
            augment_codealpaca=False,
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="0 cleaned records"):
                run_pipeline(config)

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    @pytest.mark.slow
    def test_run_pipeline_with_augmentation(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """When augmentation is enabled, synthetic records are added to train split."""
        mock_run_swebench.return_value = [_make_record("swe#1")]

        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
            augment_codecontests=True,
            augment_codealpaca=True,
            max_train_examples=30000,
        )

        synthetic_records = [
            _make_record("synth_cc_001", repo="synthetic/codecontests"),
            _make_record("synth_ca_001", repo="synthetic/codealpaca"),
        ]

        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch(
                "data_engineering.run_pipeline.synthetic_augment.augment_training_data",
                return_value=synthetic_records,
            ) as mock_augment,
        ):
            result = run_pipeline(config)

        mock_augment.assert_called_once()
        args, kwargs = mock_augment.call_args
        assert len(args) >= 2  # records, config
        # Verify augmentation was called with the cleaned records and config
        assert args[1] is config
        assert result.stats.train_count == 2  # synthetic records
        assert result.stats.total_examples == 2  # only train has records (no val/test)

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_missing_credentials_raises(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """run_pipeline should raise RuntimeError when auth is missing."""
        mock_validate_auth.return_value = ["WANDB_API_KEY"]
        config = DataPipelineConfig(
            output_dir=tmp_path,
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="Missing credentials"):
                run_pipeline(config)

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    @pytest.mark.slow
    def test_run_id_override(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """run_id_override should be used as the run ID."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            run_id_override="custom-id",
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)
        assert result.run_id == "custom-id"

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_split_disabled(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """Disabled split stage uses empty Splits()."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate", "clean"],
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)
        assert result.stats.train_count == 0
        assert result.stats.val_count == 0
        assert result.stats.test_count == 0

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_golden_disabled(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """Disabled golden stage uses empty GoldenSet()."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate", "clean", "split"],
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)
        assert result.stats.golden_count == 0

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    @pytest.mark.slow
    def test_validation_errors_load_failure(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """Corrupted validation_errors JSON does not crash pipeline."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            run_id_override="test-run",
        )
        out_dir = _checkpoint_dir(config, "test-run", "swebench")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation_errors.jsonl").write_text('{"invalid": true}\n')
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline(config)
        assert result.stats.total_validation_errors == 0

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_version_failure_non_fatal(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """W&B artifact logging failure is caught and logged."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate", "clean", "split", "golden", "version"],
        )
        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch(
                "data_engineering.run_pipeline.version.log_dataset_artifacts",
                side_effect=Exception("version boom"),
            ),
        ):
            result = run_pipeline(config)
        assert result.run_id is not None

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_archive_failure_non_fatal(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """GCS upload failure is caught and logged."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate", "clean", "split", "archive"],
        )
        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch(
                "data_engineering.run_pipeline.archive.upload_to_gcs",
                side_effect=Exception("archive boom"),
            ),
        ):
            result = run_pipeline(config)
        assert result.run_id is not None

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_gcs_card_upload(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """Dataset card is uploaded to GCS when gcs_paths and gcs_bucket set."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            gcs_bucket="my-bucket",
            enabled_stages=["ingest", "validate", "clean", "split", "archive"],
        )
        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch("data_engineering.run_pipeline.archive.upload_to_gcs") as mock_upload,
            patch("data_engineering.run_pipeline.archive.upload_text_to_gcs") as mock_text,
        ):
            mock_upload.return_value = {"raw": "gs://path"}
            result = run_pipeline(config)
        mock_text.assert_called_once()
        assert result.run_id is not None

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_gcs_card_upload_failure_non_fatal(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """GCS card upload failure is caught and logged."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            gcs_bucket="my-bucket",
            enabled_stages=["ingest", "validate", "clean", "split", "archive"],
        )
        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch("data_engineering.run_pipeline.archive.upload_to_gcs") as mock_upload,
            patch(
                "data_engineering.run_pipeline.archive.upload_text_to_gcs",
                side_effect=Exception("card boom"),
            ),
        ):
            mock_upload.return_value = {"raw": "gs://path"}
            result = run_pipeline(config)
        assert result.run_id is not None

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_tokenize_failure_non_fatal(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """Tokenization failure is caught and logged."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate", "clean", "split", "tokenize"],
        )
        mock_tokenize_mod = MagicMock()
        mock_tokenize_mod.tokenize_pipeline = MagicMock(side_effect=Exception("tokenize boom"))
        with patch.dict(
            "sys.modules",
            {
                "wandb": _mock_wandb(),
                "data_engineering.tokenize": mock_tokenize_mod,
            },
        ):
            result = run_pipeline(config)
        assert result.run_id is not None
        assert result.tokenized_paths == {}

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    def test_tokenize_success(self, mock_validate_auth, mock_run_swebench, tmp_path) -> None:
        """Successful tokenization produces tokenized_paths."""
        mock_run_swebench.return_value = [_make_record("test#1")]
        config = DataPipelineConfig(
            output_dir=tmp_path,
            run_id_override="tid",
            enabled_stages=["ingest", "validate", "clean", "split", "tokenize"],
        )
        mock_tokenize_mod = MagicMock()
        mock_tokenize_mod.tokenize_pipeline = MagicMock(return_value=MagicMock())
        with patch.dict(
            "sys.modules",
            {
                "wandb": _mock_wandb(),
                "data_engineering.tokenize": mock_tokenize_mod,
            },
        ):
            result = run_pipeline(config)
        assert result.run_id is not None
        assert "train" in result.tokenized_paths

    @patch("data_engineering.run_pipeline.run_pipeline_swebench")
    @patch.object(DataPipelineConfig, "validate_auth", return_value=[])
    @pytest.mark.slow
    def test_run_pipeline_augmentation_disabled(
        self, mock_validate_auth, mock_run_swebench, tmp_path
    ) -> None:
        """When augmentation is disabled, augment_training_data is NOT called."""
        mock_run_swebench.return_value = [_make_record("swe#1")]

        config = DataPipelineConfig(
            output_dir=tmp_path,
            bigquery_enabled=False,
            augment_codecontests=False,
            augment_codealpaca=False,
        )

        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch(
                "data_engineering.run_pipeline.synthetic_augment.augment_training_data",
            ) as mock_augment,
        ):
            result = run_pipeline(config)

        mock_augment.assert_not_called()
        assert result.stats.train_count == 1  # Only the SWE-bench record


class TestGCSHelpers:
    """Tests for _save_stage_gcs and _load_stage_gcs."""

    def test_save_stage_gcs_no_bucket(self, tmp_path) -> None:
        """Empty gcs_bucket skips GCS save."""
        config = DataPipelineConfig(gcs_bucket="", output_dir=tmp_path)
        _save_stage_gcs([], config, "run123", "swebench", "raw")

    def test_save_stage_gcs_bucket_not_exists(self, tmp_path) -> None:
        """Bucket not found skips GCS save."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = False
            _save_stage_gcs([_make_record("t1")], config, "run123", "swebench", "raw")
            mock_bucket.blob.assert_not_called()

    def test_save_stage_gcs_upload(self, tmp_path) -> None:
        """Successful GCS save calls blob.upload_from_string."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = True
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            _save_stage_gcs([_make_record("t1")], config, "run123", "swebench", "raw")
            mock_bucket.blob.assert_called_once()
            mock_blob.upload_from_string.assert_called_once()

    def test_save_stage_gcs_exception(self, tmp_path) -> None:
        """GCS exception is caught and does not propagate."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client", side_effect=Exception("boom")):
            _save_stage_gcs([_make_record("t1")], config, "run123", "swebench", "raw")

    def test_load_stage_gcs_no_bucket(self, tmp_path) -> None:
        """Empty gcs_bucket returns [] from GCS load."""
        config = DataPipelineConfig(gcs_bucket="", output_dir=tmp_path)
        result = _load_stage_gcs(config, "run123", "swebench", "validated")
        assert result == []

    def test_load_stage_gcs_bucket_not_exists(self, tmp_path) -> None:
        """Bucket not found returns [] from GCS load."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = False
            result = _load_stage_gcs(config, "run123", "swebench", "validated")
            assert result == []

    def test_load_stage_gcs_blob_not_exists(self, tmp_path) -> None:
        """Blob not found returns [] from GCS load."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = True
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = False
            result = _load_stage_gcs(config, "run123", "swebench", "validated")
            assert result == []

    def test_load_stage_gcs_success(self, tmp_path) -> None:
        """Successful GCS load returns parsed records."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = True
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = True
            rec_data = _make_record("t1").model_dump()
            mock_blob.download_as_text.return_value = json.dumps(rec_data)
            result = _load_stage_gcs(config, "run123", "swebench", "validated")
            assert len(result) == 1
            assert result[0]["issue_id"] == "t1"

    def test_load_stage_gcs_exception(self, tmp_path) -> None:
        """GCS load exception returns []."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        with patch("google.cloud.storage.Client", side_effect=Exception("boom")):
            result = _load_stage_gcs(config, "run123", "swebench", "validated")
            assert result == []


class TestCheckpointEdgeCases:
    """Edge cases for _save_stage and _load_stage."""

    def test_save_stage_skip_gcs_when_bucket_empty(self, tmp_path) -> None:
        """_save_stage skips GCS when gcs_bucket is empty (overrides .env)."""
        config = DataPipelineConfig(output_dir=tmp_path, gcs_bucket="")
        recs = [_make_record("t1")]
        with patch("data_engineering.run_pipeline._save_stage_gcs") as mock_gcs:
            _save_stage(recs, config, "run123", "swebench", "raw")
            mock_gcs.assert_not_called()

    def test_save_stage_calls_gcs_when_bucket_set(self, tmp_path) -> None:
        """_save_stage calls _save_stage_gcs when gcs_bucket is set."""
        config = DataPipelineConfig(gcs_bucket="my-bucket", output_dir=tmp_path)
        recs = [_make_record("t1")]
        with patch("data_engineering.run_pipeline._save_stage_gcs") as mock_gcs:
            _save_stage(recs, config, "run123", "swebench", "raw")
            mock_gcs.assert_called_once_with(recs, config, "run123", "swebench", "raw")

    def test_load_stage_blank_line(self, tmp_path) -> None:
        """_load_stage handles blank lines in JSONL gracefully."""
        config = DataPipelineConfig(output_dir=tmp_path)
        out_dir = _checkpoint_dir(config, "run123", "swebench")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "raw.jsonl"
        path.write_text(
            json.dumps({"issue_id": "a"}) + "\n\n" + json.dumps({"issue_id": "b"}) + "\n"
        )
        loaded = _load_stage(config, "run123", "swebench", "raw")
        assert len(loaded) == 2
        assert loaded[0]["issue_id"] == "a"
        assert loaded[1]["issue_id"] == "b"


class TestRunPipelineSWEBenchBranchCoverage:
    """Branch coverage for run_pipeline_swebench."""

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @patch("data_engineering.run_pipeline.clean.deduplicate")
    @patch("data_engineering.run_pipeline.clean.clean_records")
    @pytest.mark.slow
    def test_custom_run_name(
        self, mock_clean, mock_dedup, mock_validate, mock_ingest, tmp_path
    ) -> None:
        """When run_name is set, it should be used for the W&B run."""
        rec = _make_record("test#1")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([rec], [])
        mock_dedup.return_value = ([rec], MagicMock())
        mock_clean.return_value = ([rec], MagicMock())

        config = DataPipelineConfig(output_dir=tmp_path, run_name="my-custom-run")
        mock_wandb = _mock_wandb()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = run_pipeline_swebench(config, "run123", None)
        assert len(result) == 1
        _, kwargs = mock_wandb.init.call_args
        assert kwargs["name"] == "my-custom-run"

    def test_ingest_disabled_raises(self, tmp_path) -> None:
        """Disabled ingest with no resume raises RuntimeError."""
        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["validate", "clean"],
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="0 records"):
                run_pipeline_swebench(config, "run123", None)

    @pytest.mark.slow
    def test_resume_from_cleaned(self, tmp_path) -> None:
        """Resume from cleaned loads validated from checkpoint."""
        rec = _make_record("resume#1")
        config = DataPipelineConfig(output_dir=tmp_path)
        _save_stage([rec], config, "run123", "swebench", "raw")
        _save_stage([rec], config, "run123", "swebench", "validated")

        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch("data_engineering.run_pipeline.clean.deduplicate") as mock_dedup,
            patch("data_engineering.run_pipeline.clean.clean_records") as mock_clean,
        ):
            mock_dedup.return_value = ([rec], MagicMock())
            mock_clean.return_value = ([rec], MagicMock())
            result = run_pipeline_swebench(config, "run123", resume_from="cleaned")
        assert len(result) == 1

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @pytest.mark.slow
    def test_save_validation_errors(self, mock_validate, mock_ingest, tmp_path) -> None:
        """Validation errors should be saved to disk."""
        rec = _make_record("test#1")
        err = ValidationError(record_id="test#1", field="patch_diff", error="bad patch")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([rec], [err])

        config = DataPipelineConfig(output_dir=tmp_path)
        with (
            patch.dict("sys.modules", {"wandb": _mock_wandb()}),
            patch("data_engineering.run_pipeline.clean.deduplicate") as mock_dedup,
            patch("data_engineering.run_pipeline.clean.clean_records") as mock_clean,
        ):
            mock_dedup.return_value = ([rec], MagicMock())
            mock_clean.return_value = ([rec], MagicMock())
            result = run_pipeline_swebench(config, "run123", None)

        path = _checkpoint_dir(config, "run123", "swebench") / "validation_errors.jsonl"
        assert path.exists()
        loaded = [json.loads(l) for l in path.read_text().strip().split("\n") if l.strip()]
        assert len(loaded) == 1
        assert loaded[0]["field"] == "patch_diff"

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @pytest.mark.slow
    def test_validate_disabled_zero_validated_raises(
        self, mock_validate, mock_ingest, tmp_path
    ) -> None:
        """With validate enabled but 0 valid records, should raise."""
        rec = _make_record("test#1")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([], [MagicMock()])

        config = DataPipelineConfig(output_dir=tmp_path)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            with pytest.raises(RuntimeError, match="0 valid records"):
                run_pipeline_swebench(config, "run123", None)

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @patch("data_engineering.run_pipeline.clean.deduplicate")
    @patch("data_engineering.run_pipeline.clean.clean_records")
    def test_bigquery_disabled_skips_bq(
        self, mock_clean, mock_dedup, mock_validate, mock_ingest, tmp_path
    ) -> None:
        """BigQuery is skipped when bigquery_enabled=False (overrides .env)."""
        rec = _make_record("test#1")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([rec], [])
        mock_dedup.return_value = ([rec], MagicMock())
        mock_clean.return_value = ([rec], MagicMock())

        config = DataPipelineConfig(output_dir=tmp_path, bigquery_enabled=False)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", None)
        assert len(result) == 1

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.swebench_ingest.augment_with_bigquery")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    @patch("data_engineering.run_pipeline.clean.deduplicate")
    @patch("data_engineering.run_pipeline.clean.clean_records")
    @pytest.mark.slow
    def test_bigquery_augmentation_enabled(
        self, mock_clean, mock_dedup, mock_validate, mock_bq, mock_ingest, tmp_path
    ) -> None:
        """BigQuery augmentation runs when bigquery_enabled=True."""
        recs = [_make_record("test#1"), _make_record("test#2")]
        bq_rec = _make_record("bq#1")
        mock_ingest.return_value = recs
        mock_bq.return_value = recs + [bq_rec]
        mock_validate.return_value = (recs + [bq_rec], [])
        mock_dedup.return_value = (recs + [bq_rec], MagicMock())
        mock_clean.return_value = (recs + [bq_rec], MagicMock())

        config = DataPipelineConfig(output_dir=tmp_path, bigquery_enabled=True)
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", None)
        assert len(result) == 3
        mock_bq.assert_called_once()

    @patch("data_engineering.run_pipeline.swebench_ingest.ingest_swebench")
    @patch("data_engineering.run_pipeline.validate.validate_batch")
    def test_clean_disabled(self, mock_validate, mock_ingest, tmp_path) -> None:
        """Disabled clean stage returns empty cleaned_records."""
        rec = _make_record("test#1")
        mock_ingest.return_value = [rec]
        mock_validate.return_value = ([rec], [])

        config = DataPipelineConfig(
            output_dir=tmp_path,
            enabled_stages=["ingest", "validate"],
        )
        with patch.dict("sys.modules", {"wandb": _mock_wandb()}):
            result = run_pipeline_swebench(config, "run123", None)
        assert result == []


class TestSaveSplits:
    """Tests for _save_splits_jsonl."""

    def test_save_splits_hasattr_not_model_dump(self, tmp_path) -> None:
        """_save_splits_jsonl handles records without model_dump."""
        config = DataPipelineConfig(output_dir=tmp_path)
        rec = {
            "issue_id": "t1",
            "repo": "r",
            "issue_body": "b",
            "patch_diff": "---\n+++\n@@ -1 +1,2 @@\n-x\n+y\n",
        }
        splits = Splits.model_construct(train=[rec], val=[], test=[])
        golden_set = GoldenSet.model_construct(records=[])
        _save_splits_jsonl(splits, golden_set, config, "run123")
        path = config.output_dir / "run123" / "swebench" / "train.jsonl"
        assert path.exists()
        content = path.read_text().strip()
        assert "t1" in content
