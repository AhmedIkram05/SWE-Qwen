"""Integration test: full pipeline end-to-end with mock data.

Tests the entire ingest → validate → clean → split → golden → card flow
using local fixture data, skipping GCS and W&B (which require credentials).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data_engineering.card import generate_dataset_card

pytestmark = pytest.mark.integration
from data_engineering.clean import clean_records, deduplicate
from data_engineering.config import DataPipelineConfig
from data_engineering.golden import build_golden_set_from_config
from data_engineering.schema import IssueRecord
from data_engineering.split import stratified_split
from data_engineering.validate import validate_batch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config() -> DataPipelineConfig:
    return DataPipelineConfig(
        max_patch_lines=500,
        test_directories=["tests/", "test/"],
        min_golden_examples=1,
    )


class TestFullPipeline:
    """End-to-end pipeline test with synthetic data."""

    def test_validate_clean_split_cycle(self, config: DataPipelineConfig) -> None:
        """Run validate → clean → split with sample data."""
        with (FIXTURES / "sample_issues.json").open() as f:
            raw = json.load(f)

        # Validate
        records, errors = validate_batch(raw)
        assert len(errors) == 0, f"Validation failed: {errors}"
        assert len(records) == len(raw)
        assert all(isinstance(r, IssueRecord) for r in records)

        # Dedup + clean
        deduped, dedup_stats = deduplicate(records)
        cleaned, clean_stats = clean_records(deduped, config)

        # At least some records should survive
        assert len(cleaned) > 0, "All records removed by cleaning"

        # Split
        splits = stratified_split(cleaned, config, seed=42)
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == len(cleaned)
        assert len(splits.train) > 0

        # Golden
        golden_set = build_golden_set_from_config(splits, config)
        train_repos = {r.repo for r in splits.train}
        test_repos = {r.repo for r in splits.test}
        assert train_repos.isdisjoint(test_repos), "Data leakage detected"

    def test_validate_rejects_invalid(self, config: DataPipelineConfig) -> None:
        """Invalid records should be filtered out."""
        with (FIXTURES / "sample_invalid_issues.json").open() as f:
            invalid = json.load(f)

        records, errors = validate_batch(invalid)
        assert len(records) == 0
        assert len(errors) >= 3

    def test_clean_removes_no_test_files(self, config: DataPipelineConfig) -> None:
        """Records without test files should be removed."""
        with (FIXTURES / "sample_issues.json").open() as f:
            raw = json.load(f)

        records, _ = validate_batch(raw)
        cleaned, stats = clean_records(records, config)

        # Our fixture has 2 records without test_files
        assert stats.removed_no_test_files >= 2

    def test_card_generation(self, config: DataPipelineConfig) -> None:
        """Dataset card should render without error."""
        from data_engineering.schema import (
            CleanStats,
            DedupStats,
            PipelineStats,
        )

        manifest = {
            "version": "1",
            "repositories": [
                {"id": "owner/repo1"},
                {"id": "owner/repo2"},
            ],
        }
        stats = PipelineStats(
            total_raw=50,
            total_validated=45,
            total_cleaned=40,
            train_count=30,
            val_count=5,
            test_count=5,
            golden_count=2,
            total_examples=40,
            repo_count=2,
            dedup_stats=DedupStats(
                total_input=45,
                exact_duplicates_removed=3,
                content_duplicates_removed=2,
                unique_output=40,
            ),
            clean_stats=CleanStats(
                total_input=40,
                removed_no_test_files=5,
                total_removed=5,
                total_output=35,
            ),
        )

        card = generate_dataset_card(manifest, stats, "run_test")
        assert card
        assert "SWE-Qwen Fine-Tuning Dataset" in card

    def test_output_dir_created(self) -> None:
        """Pipeline output directory should be created."""
        with tempfile.TemporaryDirectory() as tmp:
            config = DataPipelineConfig(output_dir=Path(tmp))
            assert config.output_dir.exists()

    def test_invalid_then_valid_mixed(self) -> None:
        """Mixed valid/invalid input should keep valid records."""
        with (FIXTURES / "sample_issues.json").open() as f:
            valid_data = json.load(f)
        with (FIXTURES / "sample_invalid_issues.json").open() as f:
            invalid_data = json.load(f)

        mixed = valid_data[:2] + invalid_data[:2]
        records, errors = validate_batch(mixed)
        assert len(records) == 2
        assert len(errors) >= 2
