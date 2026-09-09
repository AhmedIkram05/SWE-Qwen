"""Centralized configuration for the data engineering pipeline."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import yaml
from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from registry.loader import default_model_key, load_models

# Legacy Phase-4 literal: last-resort default only when the registry is
# missing or unreadable (PHASE-10 Steps 2/9-3).
_LEGACY_MODEL = "qwen3-14b"
_LEGACY_MAX_LENGTH = 4096


def default_tokenize_model() -> str:
    """Registry default key (env-overridable), legacy literal last resort."""
    if env_value := os.environ.get("DATA_PIPELINE_TOKENIZE_MODEL"):
        return env_value
    try:
        return default_model_key()
    except (KeyError, OSError, yaml.YAMLError):
        return _LEGACY_MODEL


def default_tokenize_max_length() -> int:
    """Registry ``context_window`` of the default model, literal last resort."""
    try:
        return load_models()[default_tokenize_model()].context_window
    except (KeyError, OSError, yaml.YAMLError, ValidationError):
        return _LEGACY_MAX_LENGTH


class DataPipelineConfig(BaseSettings):
    """Configuration for the data pipeline, loaded from CLI args > env vars > .env > defaults.

    All filter thresholds and pipeline parameters are tunable via environment
    variables prefixed with ``DATA_PIPELINE_`` or a ``.env`` file.
    """

    # Processing
    batch_size: int = 50
    max_patch_lines: int = 500
    min_golden_examples: int = 100
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
    golden_source_split: str = "verified+test+dev"  # official SWE-bench F2P splits
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
    # Registry-first (default key / context_window); env (DATA_PIPELINE_*) and
    # init kwargs win; see _resolve_registry.
    tokenize_model: str = _LEGACY_MODEL
    tokenize_max_length: int = _LEGACY_MAX_LENGTH

    @model_validator(mode="after")
    def _resolve_registry(self) -> DataPipelineConfig:
        """Resolve ``tokenize_model``/``tokenize_max_length`` registry-first.

        Env/init values win (already in ``model_fields_set``). A readable
        registry without ``tokenize_model`` raises — literals apply only when
        the registry is missing or unreadable (no silent family fallback).
        """
        try:
            models = load_models()
        except (OSError, yaml.YAMLError):
            return self
        if "tokenize_model" not in self.model_fields_set:
            # no default flagged → keep the legacy literal key
            with contextlib.suppress(KeyError):
                self.tokenize_model = default_model_key()
        spec = models.get(self.tokenize_model)
        if spec is None:
            raise ValueError(
                f"unknown model {self.tokenize_model!r} in the registry; "
                f"available: {sorted(models)}"
            )
        if "tokenize_max_length" not in self.model_fields_set:
            self.tokenize_max_length = spec.context_window
        return self

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
