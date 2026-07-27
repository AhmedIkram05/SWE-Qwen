"""Tests for data_engineering.card."""

from __future__ import annotations

from data_engineering.card import generate_dataset_card
from data_engineering.schema import (
    CleanStats,
    DedupStats,
    PipelineStats,
)


class TestGenerateDatasetCard:
    def test_generates_markdown(self) -> None:
        manifest = {
            "version": "1.0",
            "repositories": [
                {"id": "owner/repo1"},
                {"id": "owner/repo2"},
            ],
        }
        stats = PipelineStats(
            total_raw=200,
            total_validated=180,
            total_cleaned=150,
            train_count=100,
            val_count=25,
            test_count=25,
            golden_count=10,
            total_examples=150,
            repo_count=2,
            dedup_stats=DedupStats(
                total_input=180,
                exact_duplicates_removed=20,
                content_duplicates_removed=10,
                unique_output=150,
            ),
            clean_stats=CleanStats(
                total_input=150,
                removed_no_test_files=20,
                total_removed=20,
                total_output=130,
            ),
        )

        card = generate_dataset_card(manifest, stats, "run_abc", git_sha="abc123")
        assert "SWE-Qwen Fine-Tuning Dataset" in card
        assert "owner/repo1" in card
        assert "owner/repo2" in card
        assert "200" in card  # total_raw
        assert "run_abc" in card
        assert "abc123" in card

    def test_single_repo(self) -> None:
        manifest = {"version": "1", "repositories": [{"id": "only/repo"}]}
        stats = PipelineStats(
            total_raw=10,
            total_validated=9,
            total_cleaned=8,
            train_count=6,
            val_count=1,
            test_count=1,
            golden_count=1,
            total_examples=8,
            repo_count=1,
        )
        card = generate_dataset_card(manifest, stats, "r1")
        assert "only/repo" in card
        assert "8" in card

    def test_percentages_derived_from_stats(self) -> None:
        """Card should show actual computed percentages, not hardcoded 80/10/10."""
        manifest = {"version": "1", "repositories": [{"id": "o/r"}]}
        stats = PipelineStats(
            total_raw=100,
            total_validated=90,
            total_cleaned=80,
            train_count=56,
            val_count=14,
            test_count=10,
            golden_count=3,
            total_examples=80,
            repo_count=1,
        )
        card = generate_dataset_card(manifest, stats, "r1")
        # 56/80 = 70%, 14/80 = 17.5% -> rounds to 18%, 10/80 = 12.5% -> rounds to 12%
        assert "70%" in card
        assert "18%" in card or "17%" in card
        assert "12%" in card or "13%" in card
