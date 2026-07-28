"""Tests for data_engineering.card (SWE-bench source)."""

from __future__ import annotations

from data_engineering.card import generate_dataset_card
from data_engineering.schema import (
    CleanStats,
    DedupStats,
    PipelineStats,
    RepoResult,
)


class TestGenerateDatasetCard:
    def test_generates_markdown(self) -> None:
        manifest = {}  # SWE-bench uses empty manifest
        stats = PipelineStats(
            total_raw=200,
            total_validated=180,
            total_cleaned=150,
            train_count=100,
            val_count=25,
            test_count=25,
            golden_count=10,
            total_examples=150,
            repo_count=1,
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
            repo_results=[],
        )

        card = generate_dataset_card(
            manifest, stats, "run_abc", git_sha="abc123", source="swebench"
        )
        assert "SWE-Qwen Fine-Tuning Dataset" in card
        assert "run_abc" in card
        assert "abc123" in card
        assert "SWE-bench/SWE-bench" in card  # SWE-bench dataset
        assert "django/django" in card  # One of the 18 repos

    def test_single_repo(self) -> None:
        manifest = {}
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
            repo_results=[],
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        assert "r1" in card
        assert "8" in card
        assert "SWE-bench Dataset" in card
        assert "SWE-bench/SWE-bench" in card

    def test_percentages_derived_from_stats(self) -> None:
        """Card should show actual computed percentages, not hardcoded 80/10/10."""
        manifest = {}
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
            repo_results=[],
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        # 56/80 = 70%, 14/80 = 17.5% -> rounds to 18%, 10/80 = 12.5% -> rounds to 12%
        assert "70%" in card
        assert "18%" in card or "17%" in card
        assert "12%" in card or "13%" in card

    def test_source_repos_table_with_repo_counts(self) -> None:
        """Repo counts from repo_results should appear in the source repos table."""
        manifest = {}
        stats = PipelineStats(
            total_raw=100,
            total_validated=80,
            total_cleaned=60,
            train_count=40,
            val_count=10,
            test_count=10,
            golden_count=5,
            total_examples=60,
            repo_count=2,
            repo_results=[
                RepoResult(repo_id="django/django", cleaned_count=42),
                RepoResult(repo_id="pytest-dev/pytest", cleaned_count=18),
            ],
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        assert "42" in card
        assert "18" in card
        assert "django/django" in card
        assert "pytest-dev/pytest" in card

    def test_percentages_with_zero_total(self) -> None:
        """Zero total_examples should show em dash instead of crashing on division."""
        manifest = {}
        stats = PipelineStats(
            total_raw=0,
            total_validated=0,
            total_cleaned=0,
            train_count=0,
            val_count=0,
            test_count=0,
            golden_count=0,
            total_examples=0,
            repo_count=0,
            repo_results=[],
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        # When total_examples is 0, the ratio column shows "—" for each split
        assert "—" in card or "–" in card
        assert "r1" in card

    def test_empty_manifest_path(self) -> None:
        """Card should generate cleanly even with minimal/default stats and empty manifest."""
        manifest = {}
        stats = PipelineStats()
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        assert "r1" in card
        assert "SWE-Qwen Fine-Tuning Dataset" in card

    def test_wandb_artifact_links_in_card(self) -> None:
        """W&B artifact links should appear in card when stats has wandb_artifacts."""
        manifest = {}
        stats = PipelineStats(
            total_raw=50,
            total_validated=40,
            total_cleaned=30,
            train_count=20,
            val_count=5,
            test_count=5,
            golden_count=2,
            total_examples=30,
            repo_count=1,
            wandb_artifacts={
                "train": "dataset-train-v1",
                "test": "dataset-test-v1",
            },
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        assert "W&B Artifact Links" in card
        assert "dataset-train-v1" in card
        assert "dataset-test-v1" in card

    def test_gcs_paths_in_card(self) -> None:
        """GCS archive paths should appear in card when stats has gcs_paths."""
        manifest = {}
        stats = PipelineStats(
            total_raw=50,
            total_validated=40,
            total_cleaned=30,
            train_count=20,
            val_count=5,
            test_count=5,
            golden_count=2,
            total_examples=30,
            repo_count=1,
            gcs_paths={
                "train": "gs://bucket/datasets/r1/train.jsonl",
            },
        )
        card = generate_dataset_card(manifest, stats, "r1", source="swebench")
        assert "GCS Archive Paths" in card
        assert "gs://bucket/datasets/r1/train.jsonl" in card
