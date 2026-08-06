"""Tests for data_engineering.tokenize."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from data_engineering.config import DataPipelineConfig
from data_engineering.tokenize import (
    _save_dataset_to_gcs,
    format_training_prompt,
    load_jsonl_split,
    load_tokenized_shards,
    tokenize_dataset,
    tokenize_pipeline,
    tokenize_split,
)
from training.prompt_loader import PromptLoader

TINY_MODEL = "hf-internal-testing/tiny-random-GPT2"


@pytest.fixture(scope="session")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def prompt_loader() -> PromptLoader:
    return PromptLoader()


@pytest.fixture
def sample_record() -> dict[str, Any]:
    return {
        "issue_id": "test#123",
        "issue_body": "This is a test issue body describing a bug.",
        "repo": "test/repo",
        "repo_domain": "github.com",
        "patch_diff": "--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-x=1\n+y=2\n",
        "files_changed": ["main.py", "utils.py"],
        "test_files_changed": ["test_main.py"],
    }


@pytest.fixture
def sample_records(sample_record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sample_record,
        {**sample_record, "issue_id": "test#456", "issue_body": "Second issue."},
    ]


# ── load_jsonl_split ─────────────────────────────────────────────────────────


class TestLoadJsonlSplit:
    def test_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        records = [{"a": 1}, {"b": 2}]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        assert load_jsonl_split(path) == records

    def test_blank_lines_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        assert load_jsonl_split(path) == [{"a": 1}, {"b": 2}]

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text("")
        assert load_jsonl_split(path) == []

    def test_trailing_newlines(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n\n\n')
        assert load_jsonl_split(path) == [{"a": 1}, {"b": 2}]


# ── format_training_prompt ───────────────────────────────────────────────────


class TestFormatTrainingPrompt:
    def test_all_fields(self, prompt_loader: PromptLoader, sample_record: dict[str, Any]) -> None:
        result = format_training_prompt(sample_record, prompt_loader)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "### Input" in result
        assert "### Response" in result
        assert "test#123" in result
        assert sample_record["patch_diff"] in result

    def test_minimal_fields(self, prompt_loader: PromptLoader) -> None:
        record: dict[str, Any] = {
            "issue_body": "A bug.",
            "patch_diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
        }
        result = format_training_prompt(record, prompt_loader)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "unknown" in result
        assert "A bug." in result


# ── tokenize_split ───────────────────────────────────────────────────────────


class TestTokenizeSplit:
    def test_basic(
        self, tokenizer: Any, prompt_loader: PromptLoader, sample_record: dict[str, Any]
    ) -> None:
        ds = tokenize_split([sample_record], tokenizer, prompt_loader, max_length=512)
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        row = ds[0]
        assert "input_ids" in row
        assert "attention_mask" in row
        assert "labels" in row
        assert len(row["input_ids"]) > 0
        assert len(row["input_ids"]) == len(row["attention_mask"])
        assert len(row["input_ids"]) == len(row["labels"])
        assert -100 in row["labels"]

    def test_multiple_records(
        self, tokenizer: Any, prompt_loader: PromptLoader, sample_records: list[dict[str, Any]]
    ) -> None:
        ds = tokenize_split(sample_records, tokenizer, prompt_loader, max_length=512)
        assert len(ds) == 2
        for row in ds:
            assert -100 in row["labels"]

    def test_empty_records_list(self, tokenizer: Any, prompt_loader: PromptLoader) -> None:
        ds = tokenize_split([], tokenizer, prompt_loader, max_length=512)
        assert isinstance(ds, Dataset)
        assert len(ds) == 0

    def test_format_errors_logged(
        self, tokenizer: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        pdir = tmp_path / "missing_templates"
        pdir.mkdir()
        # Only provide chat.j2 — user.j2 and system.j2 are MISSING
        (pdir / "chat.j2").write_text(
            "{{ system_prompt }}\n{% for m in messages %}{{ m.content }}{% endfor %}"
        )
        bad_loader = PromptLoader(template_dir=pdir)
        recs = [
            {
                "issue_id": "err1",
                "issue_body": "b1",
                "repo": "r",
                "repo_domain": "d",
                "patch_diff": "d",
                "files_changed": [],
                "test_files_changed": [],
            },
            {
                "issue_id": "err2",
                "issue_body": "b2",
                "repo": "r",
                "repo_domain": "d",
                "patch_diff": "d",
                "files_changed": [],
                "test_files_changed": [],
            },
        ]
        ds = tokenize_split(recs, tokenizer, bad_loader, max_length=512)
        assert len(ds) == 0
        assert "Failed to format record" in caplog.text
        assert "Tokenization:" in caplog.text

    def test_exceeds_max_log_errors(
        self, tokenizer: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        pdir = tmp_path / "no_user"
        pdir.mkdir()
        (pdir / "chat.j2").write_text(
            "{{ system_prompt }}\n{% for m in messages %}{{ m.content }}{% endfor %}"
        )
        bad_loader = PromptLoader(template_dir=pdir)
        recs = [
            {
                "issue_id": f"err{i}",
                "issue_body": "b",
                "repo": "r",
                "repo_domain": "d",
                "patch_diff": "d",
                "files_changed": [],
                "test_files_changed": [],
            }
            for i in range(7)
        ]
        ds = tokenize_split(recs, tokenizer, bad_loader, max_length=512)
        assert len(ds) == 0
        assert "Tokenization:" in caplog.text

    def test_label_length_correction(
        self, tokenizer: Any, prompt_loader: PromptLoader, sample_record: dict[str, Any]
    ) -> None:
        # max_length=2 is smaller than the prompt alone: the record is dropped
        # rather than emitted with an all -100 (empty-target) label.
        ds = tokenize_split([sample_record], tokenizer, prompt_loader, max_length=2)
        assert len(ds) == 0


# ── tokenize_pipeline ────────────────────────────────────────────────────────


class TestTokenizeSplitPromptOverflow:
    """Records whose prompt alone exceeds max_length must be dropped, not
    emitted with all -100 labels.

    Regression: the prompt-only tokenization call had no truncation, so a
    file-contents prompt blew up to 145K tokens and every truncated example
    lost its gold-patch target.
    """

    def _rec(self, issue_id: str, body: str) -> dict[str, Any]:
        return {
            "issue_id": issue_id,
            "issue_body": body,
            "repo": "test/repo",
            "repo_domain": "github.com",
            "patch_diff": "--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-x=1\n+y=2\n",
            "files_changed": [],
            "test_files_changed": [],
        }

    def test_oversized_prompt_dropped(self, tokenizer: Any, prompt_loader: PromptLoader) -> None:
        big = self._rec("big#1", "bug " * 3000)  # ~12K chars, far past max_length
        small = self._rec("small#1", "Fix it.")
        ds = tokenize_split([big, small], tokenizer, prompt_loader, max_length=2048)
        assert len(ds) == 1
        # The surviving example must keep a non-masked gold-patch tail.
        row = ds[0]
        assert any(t != -100 for t in row["labels"])

    def test_all_oversized_yields_empty(self, tokenizer: Any, prompt_loader: PromptLoader) -> None:
        ds = tokenize_split(
            [self._rec("big#1", "bug " * 3000)],
            tokenizer,
            prompt_loader,
            max_length=2048,
        )
        assert len(ds) == 0


class TestTokenizePipeline:
    def _write_split(self, data_dir: Path, filename: str, records: list[dict[str, Any]]) -> Path:
        path = data_dir / filename
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    @pytest.fixture
    def mock_model_cfg(self):
        with patch("training.qlora_config._get_model_config") as m:
            m.return_value = {"hf_id": TINY_MODEL, "context_window": 1024}
            yield m

    def test_basic_pipeline(self, tmp_path: Path, mock_model_cfg: MagicMock) -> None:
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"
        rec = {
            "issue_id": "t#1",
            "issue_body": "body",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "diff",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(data_dir, "train.jsonl", [rec])
        self._write_split(data_dir, "val.jsonl", [rec])
        self._write_split(data_dir, "test.jsonl", [rec])
        self._write_split(data_dir, "golden.jsonl", [rec])

        ds = tokenize_pipeline(data_dir=data_dir, output_dir=out_dir, max_length=512)

        assert isinstance(ds, DatasetDict)
        for split in ("train", "val", "test", "golden"):
            assert split in ds, f"Missing split: {split}"
            assert len(ds[split]) == 1
        assert out_dir.exists()
        assert (out_dir / "dataset_dict.json").exists()

    def test_missing_split_logs_warning(
        self, tmp_path: Path, mock_model_cfg: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"
        rec = {
            "issue_id": "t#1",
            "issue_body": "b",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "d",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(data_dir, "train.jsonl", [rec])
        self._write_split(data_dir, "val.jsonl", [rec])

        ds = tokenize_pipeline(data_dir=data_dir, output_dir=out_dir, max_length=512)

        assert "train" in ds
        assert "val" in ds
        assert "test" not in ds
        assert "golden" not in ds
        assert "not found, skipping" in caplog.text

    def test_all_splits_missing_raises(self, tmp_path: Path, mock_model_cfg: MagicMock) -> None:
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"

        with pytest.raises(ValueError, match="No tokenized datasets produced"):
            tokenize_pipeline(data_dir=data_dir, output_dir=out_dir, max_length=512)

    def test_empty_dataset_skipped(self, tmp_path: Path, mock_model_cfg: MagicMock) -> None:
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"
        rec = {
            "issue_id": "t#1",
            "issue_body": "body",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "diff",
            "files_changed": [],
            "test_files_changed": [],
        }
        (data_dir / "train.jsonl").write_text(json.dumps(rec) + "\n")
        (data_dir / "golden.jsonl").write_text("")  # empty → 0 records → empty dataset → skipped

        ds = tokenize_pipeline(data_dir=data_dir, output_dir=out_dir, max_length=512)
        assert "train" in ds
        assert "golden" not in ds

    def test_gcs_upload_path(self, tmp_path: Path, mock_model_cfg: MagicMock) -> None:
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"
        rec = {
            "issue_id": "t#1",
            "issue_body": "b",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "d",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(data_dir, "train.jsonl", [rec])

        config = DataPipelineConfig(gcs_bucket="test-bucket")

        with patch("data_engineering.tokenize._save_dataset_to_gcs") as mock_save:
            mock_save.return_value = {"dataset": "gs://test-bucket/tokenized/run1"}
            ds = tokenize_pipeline(
                data_dir=data_dir,
                output_dir=out_dir,
                max_length=512,
                config=config,
                run_id="run1",
            )
            mock_save.assert_called_once()
            args, _ = mock_save.call_args
            assert args[1] == "test-bucket"
            assert args[2] == "run1"

        assert "train" in ds


# ── _save_dataset_to_gcs ─────────────────────────────────────────────────────


class TestSaveDatasetToGcs:
    def test_bucket_exists_uploads(self, tmp_path: Path) -> None:
        ds = DatasetDict({"train": Dataset.from_dict({"a": [1]})})
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client):
            result = _save_dataset_to_gcs(ds, "test-bucket", "myprefix")

        assert "dataset" in result
        assert "gs://" in result["dataset"]
        assert "test-bucket" in result["dataset"]
        mock_bucket.blob.assert_called()
        mock_blob.upload_from_filename.assert_called()

    def test_bucket_not_exists_returns_empty(self, tmp_path: Path) -> None:
        ds = DatasetDict({"train": Dataset.from_dict({"a": [1]})})
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client):
            result = _save_dataset_to_gcs(ds, "test-bucket", "myprefix")

        assert result == {}
        mock_bucket.blob.assert_not_called()

    def test_exception_returns_empty(self, tmp_path: Path) -> None:
        ds = DatasetDict({"train": Dataset.from_dict({"a": [1]})})
        with patch("google.cloud.storage.Client", side_effect=Exception("GCS error")):
            result = _save_dataset_to_gcs(ds, "test-bucket", "myprefix")
        assert result == {}


# ── load_tokenized_shards ─────────────────────────────────────────────────────


class TestLoadTokenizedShards:
    def test_round_trip(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "splits"
        data_dir.mkdir()
        out_dir = tmp_path / "tokenized"
        rec = {
            "issue_id": "t#1",
            "issue_body": "body",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "diff",
            "files_changed": [],
            "test_files_changed": [],
        }
        (data_dir / "train.jsonl").write_text(json.dumps(rec) + "\n")

        with patch("training.qlora_config._get_model_config") as mock_get:
            mock_get.return_value = {"hf_id": TINY_MODEL, "context_window": 1024}
            original = tokenize_pipeline(data_dir=data_dir, output_dir=out_dir, max_length=512)

        loaded = load_tokenized_shards(out_dir)
        assert isinstance(loaded, DatasetDict)
        assert list(loaded.keys()) == list(original.keys())
        assert len(loaded["train"]) == len(original["train"])
        assert loaded["train"][0]["input_ids"] == original["train"][0]["input_ids"]


# ── tokenize_dataset ─────────────────────────────────────────────────────────


class TestTokenizeDataset:
    def _write_split(self, data_dir: Path, filename: str, records: list[dict[str, Any]]) -> Path:
        path = data_dir / filename
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    def test_with_explicit_run_id(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "data"
        run_id = "test-run-001"
        data_dir = output_dir / run_id / "swebench"
        data_dir.mkdir(parents=True)
        rec = {
            "issue_id": "t#1",
            "issue_body": "b",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "d",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(data_dir, "train.jsonl", [rec])

        config = DataPipelineConfig(output_dir=output_dir)

        with patch("training.qlora_config._get_model_config") as mock_get:
            mock_get.return_value = {"hf_id": TINY_MODEL, "context_window": 1024}
            result = tokenize_dataset(
                run_id=run_id, model_name="tiny", max_seq_length=512, config=config
            )

        assert result["run_id"] == run_id
        assert result["total_examples"] > 0
        assert (output_dir / f"{run_id}_tokenized").exists()

    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "data"
        output_dir.mkdir()
        config = DataPipelineConfig(output_dir=output_dir)

        with pytest.raises(ValueError, match="Run directory not found"):
            tokenize_dataset(run_id="nonexistent", config=config)

    def test_run_id_none_uses_latest(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "data"
        run_dir = output_dir / "run-001" / "swebench"
        run_dir.mkdir(parents=True)
        rec = {
            "issue_id": "t#1",
            "issue_body": "b",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "d",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(run_dir, "train.jsonl", [rec])

        config = DataPipelineConfig(output_dir=output_dir)

        with patch("training.qlora_config._get_model_config") as mock_get:
            mock_get.return_value = {"hf_id": TINY_MODEL, "context_window": 1024}
            result = tokenize_dataset(
                run_id=None, model_name="tiny", max_seq_length=512, config=config
            )

        assert result["run_id"] == "run-001"
        assert result["total_examples"] > 0

    def test_no_run_dirs_raises(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "empty"
        output_dir.mkdir()
        config = DataPipelineConfig(output_dir=output_dir)

        with pytest.raises(ValueError, match="No run directories found"):
            tokenize_dataset(run_id=None, config=config)

    def test_without_config_default_constructor(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        run_id = "test-no-cfg"
        data_dir = Path("data") / run_id / "swebench"
        data_dir.mkdir(parents=True)
        rec = {
            "issue_id": "t#1",
            "issue_body": "b",
            "repo": "r",
            "repo_domain": "d",
            "patch_diff": "d",
            "files_changed": [],
            "test_files_changed": [],
        }
        self._write_split(data_dir, "train.jsonl", [rec])

        with patch("training.qlora_config._get_model_config") as mock_get:
            mock_get.return_value = {"hf_id": TINY_MODEL, "context_window": 1024}
            result = tokenize_dataset(run_id=run_id, model_name="tiny", max_seq_length=512)

        assert result["run_id"] == run_id
        assert result["total_examples"] > 0


# ── file contents in training prompts ────────────────────────────────────────


class TestTrainingPromptFileContents:
    """Training prompts must embed changed-file contents (### File Contents),
    mirroring the eval prompt, so the LoRA learns to diff real code."""

    def _record_with_base_sha(self, sample_record: dict[str, Any]) -> dict[str, Any]:
        return {
            **sample_record,
            "metadata": {"base_sha": "a" * 40},
            "files_changed": ["src/app.py"],
            "test_files_changed": ["tests/test_app.py"],
        }

    def test_file_contents_embedded(
        self, sample_record: dict[str, Any], prompt_loader: PromptLoader, monkeypatch: Any
    ) -> None:
        from evaluation import inference as inf

        monkeypatch.setattr(
            inf,
            "_fetch_raw_file",
            lambda repo, base_sha, path: (
                "def foo():\n    return 1\n" if path == "src/app.py" else None
            ),
        )
        text = format_training_prompt(self._record_with_base_sha(sample_record), prompt_loader)
        assert "### File Contents" in text
        assert "#### `src/app.py`" in text
        assert "def foo():" in text

    def test_no_base_sha_skips_fetch(
        self, sample_record: dict[str, Any], prompt_loader: PromptLoader, monkeypatch: Any
    ) -> None:
        from evaluation import inference as inf

        calls: list = []
        monkeypatch.setattr(inf, "_fetch_raw_file", lambda *a, **k: calls.append(a) or None)
        text = format_training_prompt(sample_record, prompt_loader)
        assert calls == []
        assert "### File Contents" not in text
