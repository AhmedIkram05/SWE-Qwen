"""Centralized configuration for the data engineering pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataPipelineConfig(BaseSettings):
    """Configuration for the data pipeline, loaded from CLI args > env vars > .env > defaults.

    All filter thresholds and pipeline parameters are tunable via environment
    variables prefixed with ``DATA_PIPELINE_`` or a ``.env`` file.
    """

    # Processing
    batch_size: int = 50
    max_patch_lines: int = 500
    min_golden_examples: int = 200
    parallel_workers: int = 1  # >1 triggers GitHub rate limit; sequential is safer
    max_issues_per_repo: int = 2000
    max_events_per_issue: int = 100

    # Paths
    gcs_bucket: str = ""
    manifest_path: Path = Path("repos/manifest.json")
    output_dir: Path = Path("data/")

    # W&B
    wandb_project: str = "swe-qwen-data"
    wandb_entity: str | None = None  # optional, defaults to user default

    # Splits
    golden_source_split: str = "test"  # "test" default; "all" requires opt-in
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Ingestion
    issue_labels_to_include: list[str] = []
    test_directories: list[str] = ["tests/", "test/"]

    # Stage control
    resume_from: str | None = None  # stage name to resume from (None = full run)
    enabled_stages: list[str] | None = None  # None = all stages; else whitelist
    run_id_override: str | None = None  # optional UUID override
    run_name: str | None = None  # optional W&B run name (auto-generated if None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DATA_PIPELINE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def validate_auth(self) -> list[str]:
        """Return list of missing credential names; empty means all ok."""
        import os

        from dotenv import load_dotenv

        load_dotenv(".env")

        missing: list[str] = []
        if not os.environ.get("GITHUB_TOKEN"):
            missing.append("GITHUB_TOKEN")
        if not os.environ.get("WANDB_API_KEY"):
            missing.append("WANDB_API_KEY")
        # Check GCP Application Default Credentials (ADC) are available
        try:
            import google.auth

            google.auth.default()
        except Exception:
            missing.append("GCP_ADC")
        return missing
