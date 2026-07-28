"""W&B dataset versioning and artifact management.

Creates versioned W&B artifacts per pipeline stage (raw, validated, cleaned,
train, val, test, golden, validation_errors) with rich metadata.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import wandb

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, ValidationError

logger = logging.getLogger(__name__)


def _records_to_jsonl(records: list[IssueRecord], path: Path) -> None:
    """Write a list of records as JSONL."""
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.model_dump(), default=str) + "\n")


def _errors_to_jsonl(errors: list[ValidationError], path: Path) -> None:
    """Write validation errors as JSONL."""
    with path.open("w") as f:
        for err in errors:
            f.write(json.dumps(err.model_dump(), default=str) + "\n")


def _build_artifact_metadata(
    run_id: str,
    stage_name: str,
    manifest_hash: str,
    records: list[Any],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build flat metadata dict per AC spec."""
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage_name,
        "manifest_hash": manifest_hash,
        "count": len(records),
        "pipeline_version": os.environ.get("GIT_SHA", "unknown"),
    }

    if stats:
        # AC-required flat keys
        metadata["validation_pass"] = stats.get("total_validated", 0)
        metadata["validation_fail"] = stats.get("total_validation_errors", 0)

        dedup = stats.get("dedup_stats") or {}
        metadata["dedup_exact"] = (
            dedup.get("exact_duplicates_removed", 0) if isinstance(dedup, dict) else 0
        )
        metadata["dedup_content"] = (
            dedup.get("content_duplicates_removed", 0) if isinstance(dedup, dict) else 0
        )

        clean = stats.get("clean_stats") or {}
        metadata["filter_counts"] = clean if isinstance(clean, dict) else {}

        repo_results = stats.get("repo_results") or []
        metadata["repo_list"] = [r.get("repo_id", "") for r in repo_results if isinstance(r, dict)]

        train_c = stats.get("train_count", 0)
        val_c = stats.get("val_count", 0)
        test_c = stats.get("test_count", 0)
        total_s = train_c + val_c + test_c
        metadata["split_ratios"] = {
            k: round(v / total_s, 4) if total_s > 0 else 0.0
            for k, v in [("train", train_c), ("val", val_c), ("test", test_c)]
        }
        metadata["counts"] = {
            "train": train_c,
            "val": val_c,
            "test": test_c,
        }
        metadata["golden_size"] = stats.get("golden_count", 0)

        # Also keep original stats for debugging
        metadata["stats"] = stats

    return metadata


def _log_single_artifact(
    run: Any,
    stage_name: str,
    records: list[Any],
    metadata: dict[str, Any],
    artifact_names: dict[str, str],
) -> None:
    """Create and log a single W&B artifact."""
    is_validation = stage_name == "validation_errors"
    artifact = wandb.Artifact(
        name=f"dataset-{stage_name}",
        type="dataset",
        description=f"Pipeline {stage_name} stage",
        metadata=metadata,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
        if is_validation:
            _errors_to_jsonl(records, Path(tmp_path))  # type: ignore[arg-type]
        else:
            _records_to_jsonl(records, Path(tmp_path))  # type: ignore[arg-type]
        artifact.add_file(tmp_path, name=f"{stage_name}.jsonl")

    run.log_artifact(artifact)
    artifact.wait()
    artifact_names[stage_name] = artifact.name

    logger.info(
        "W&B artifact '%s' (%s): %d records",
        artifact.name,
        stage_name,
        len(records),
    )


def log_dataset_artifacts(
    run_id: str,
    stages: dict[str, Any],
    config: DataPipelineConfig,
    manifest_hash: str,
    stats: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Create versioned W&B artifacts for each pipeline stage.

    Args:
        run_id: Unique pipeline run ID.
        stages: Mapping of stage name → records.
        config: Pipeline config.
        manifest_hash: SHA of the input manifest for lineage.
        stats: Optional stats dict embedded in artifact metadata.

    Returns:
        Mapping of stage name → W&B artifact name (versioned).
    """
    artifact_names: dict[str, str] = {}

    wandb_entity = config.wandb_entity
    wandb_project = config.wandb_project

    # Generate descriptive run name if not provided
    from datetime import datetime

    if config.run_name:
        run_display_name = config.run_name
    else:
        now = datetime.now()
        run_display_name = f"run-{now.strftime('%Y%m%d-%H%M')}-{run_id[:6]}"

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type="data_pipeline",
        name=run_display_name,
        config={"run_id": run_id, "manifest_hash": manifest_hash, "run_name": run_display_name},
        reinit=True,
    )

    try:
        # Standard stages (everything except validation_errors)
        for stage_name, records in stages.items():
            if stage_name == "validation_errors":
                continue
            metadata = _build_artifact_metadata(
                run_id,
                stage_name,
                manifest_hash,
                records,
                stats,
            )
            _log_single_artifact(run, stage_name, records, metadata, artifact_names)

        # Validation errors logged explicitly (handles empty list edge case)
        if "validation_errors" in stages:
            ve_records = stages["validation_errors"] or []
            ve_metadata = _build_artifact_metadata(
                run_id,
                "validation_errors",
                manifest_hash,
                ve_records,
                stats,
            )
            _log_single_artifact(run, "validation_errors", ve_records, ve_metadata, artifact_names)

        # AC #11: Log per-repo progress as W&B summary metrics
        if stats:
            repo_results = stats.get("repo_results") or []
            per_repo: dict[str, dict[str, int]] = {}
            for r in repo_results:
                if isinstance(r, dict) and "repo_id" in r:
                    per_repo[r["repo_id"]] = {
                        "raw": r.get("raw_count", 0),
                        "validated": r.get("validated_count", 0),
                        "cleaned": r.get("cleaned_count", 0),
                    }
            if per_repo:
                run.summary["per_repo_counts"] = per_repo

    finally:
        run.finish()

    return artifact_names


def log_validation_errors(
    run_id: str,
    errors: list[ValidationError],
    config: DataPipelineConfig,
) -> wandb.Artifact:
    """Log validation errors as a separate W&B artifact.

    Returns the created ``wandb.Artifact``.
    """
    artifacts = log_dataset_artifacts(
        run_id,
        {"validation_errors": errors},
        config,
        manifest_hash="",
    )
    name = artifacts.get("validation_errors", "")
    # Retrieve the artifact object from wandb
    try:
        api = wandb.Api()
        return cast(wandb.Artifact, api.artifact(name))
    except Exception:
        # Fallback: return a minimal artifact with the name
        return wandb.Artifact(
            name=name or f"validation_errors-{run_id}",
            type="dataset",
        )
