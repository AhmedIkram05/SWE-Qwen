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
    parallel_workers: int = 1
    max_issues_per_repo: int = 2000
    max_events_per_issue: int = 100

    # Paths
    gcs_bucket: str = ""
    output_dir: Path = Path("data/")

    # W&B
    wandb_project: str = "swe-qwen-data"
    wandb_entity: str | None = None  # optional, defaults to user default

    # Splits
    golden_source_split: str = "test"  # SWE-bench has F2P in verified+test+dev
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # SWE-bench source config
    swe_bench_dir: Path = Path("data/swe_bench")
    swe_bench_version: str = "2025-04-29"  # dataset version pin
    bigquery_enabled: bool = False
    bigquery_project: str = ""  # GCP project for BigQuery

    # Synthetic augmentation
    augment_codecontests: bool = False  # +13k Python solutions from CodeContests
    augment_codealpaca: bool = False  # +20k instruction-following (filtered to ~8k Python)
    max_train_examples: int = 30000  # Cap total training size after augmentation

    # Tokenization (integrated at end of pipeline)
    tokenize_model: str = "qwen3-14b"
    tokenize_max_length: int = 4096

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
        if not os.environ.get("WANDB_API_KEY"):
            missing.append("WANDB_API_KEY")
        # Check GCP Application Default Credentials (ADC) are available
        try:
            import google.auth

            google.auth.default()
        except Exception:
            missing.append("GCP_ADC")
        return missing
