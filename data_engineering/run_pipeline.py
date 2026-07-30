"""Pipeline orchestrator.

Coordinates the full data pipeline: ingest -> validate -> clean -> split ->
golden -> version -> archive -> card generation.

Source: SWE-bench dataset from Hugging Face
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from data_engineering import (
    archive,
    card,
    clean,
    golden,
    split,
    swebench_ingest,
    synthetic_augment,
    validate,
    version,
)
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import (
    GoldenSet,
    IssueRecord,
    PipelineResult,
    PipelineStats,
    RepoResult,
    Splits,
)

# ─── GCS Streaming Helpers ──────────────────────────────────────────────────


def _save_stage_gcs(
    records: list[Any],
    config: DataPipelineConfig,
    run_id: str,
    repo_id: str,
    stage: str,
) -> None:
    """Save records as JSONL to GCS checkpoint."""
    if not config.gcs_bucket:
        return
    try:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client()
        bucket = client.bucket(config.gcs_bucket)
        if not bucket.exists():
            return
        prefix = f"datasets/{run_id}/{repo_id}"
        key = f"{prefix}/{stage}.jsonl"
        blob = bucket.blob(key)
        lines = []
        for r in records:
            if hasattr(r, "model_dump"):
                line = json.dumps(r.model_dump(), default=str)
            else:
                line = json.dumps(r, default=str)
            lines.append(line + "\n")
        content = "".join(lines)
        blob.upload_from_string(content, content_type="application/jsonl")
        logging.info(
            f"Uploaded {len(records)} records to gs://{config.gcs_bucket}/datasets/{run_id}/{repo_id}/{stage}.jsonl"
        )
    except Exception as e:
        logging.warning(f"GCS save failed for stage {stage} (non-fatal): {e}")


def _load_stage_gcs(
    config: DataPipelineConfig,
    run_id: str,
    repo_id: str,
    stage: str,
) -> list[dict[str, Any]]:
    """Load records from GCS checkpoint."""
    if not config.gcs_bucket:
        return []
    try:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client()
        bucket = client.bucket(config.gcs_bucket)
        if not bucket.exists():
            return []
        prefix = f"datasets/{run_id}/{repo_id}"
        key = f"{prefix}/{stage}.jsonl"
        blob = bucket.blob(key)
        if not blob.exists():
            return []
        content = blob.download_as_text()
        records = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
    except Exception as e:
        logging.warning(f"GCS load failed for stage {stage} (non-fatal): {e}")
        return []
    else:
        logging.info(
            f"Loaded {len(records)} records from gs://{config.gcs_bucket}/datasets/{run_id}/{stage}.jsonl"
        )
        return records


logger = logging.getLogger(__name__)
console = Console()


def _manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute SHA256 of the serialised manifest."""
    raw = json.dumps(manifest, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# Checkpoint helpers

_STAGE_MAP = {
    "raw": "ingest",
    "validated": "validate",
    "cleaned": "clean",
    "train": "split",
    "val": "split",
    "test": "split",
    "golden": "golden",
    "version": "version",
    "archive": "archive",
    "card": "card",
    "tokenize": "tokenize",
}

# Reverse map: human-readable -> file-stage name
_HUMAN_TO_FILE = {v: k for k, v in _STAGE_MAP.items()}


def _stage_enabled(config: DataPipelineConfig, stage: str) -> bool:
    """Check if stage is in the enabled-stages whitelist.

    Args:
        stage: File-stage name (e.g., "raw", "validated", "cleaned")
    """
    if config.enabled_stages is None:
        return True
    # Map file stage to human stage
    human_stage = _STAGE_MAP.get(stage)
    if human_stage and human_stage in config.enabled_stages:
        return True
    # Also check direct match (in case human stage was passed)
    if stage in config.enabled_stages:
        return True
    return False


def _checkpoint_dir(config: DataPipelineConfig, run_id: str, repo_id: str) -> Path:
    return config.output_dir / run_id / repo_id


def _save_stage(
    records: list[Any],
    config: DataPipelineConfig,
    run_id: str,
    repo_id: str,
    stage: str,
) -> None:
    """Save records as JSONL checkpoint locally (for resume) and to GCS (if configured)."""
    # Local save (for fast resume)
    out_dir = _checkpoint_dir(config, run_id, repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stage}.jsonl"
    with path.open("w") as f:
        for rec in records:
            if isinstance(rec, IssueRecord):
                line = json.dumps(rec.model_dump(), default=str)
            else:
                line = json.dumps(rec, default=str)
            f.write(line + "\n")

    # GCS save (async-style, non-blocking)
    if config.gcs_bucket:
        _save_stage_gcs(records, config, run_id, "swebench", stage)


def _load_stage(
    config: DataPipelineConfig,
    run_id: str,
    repo_id: str,
    stage: str,
) -> list[dict[str, Any]]:
    """Load records from JSONL checkpoint."""
    path = _checkpoint_dir(config, run_id, repo_id) / f"{stage}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# SWE-bench pipeline


def run_pipeline_swebench(
    config: DataPipelineConfig,
    run_id: str,
    resume_from: str | None,
) -> list[IssueRecord]:
    """Run the complete pipeline for SWE-bench source with progress tracking."""

    # Initialize W&B run for progress tracking
    import wandb

    wandb_entity = config.wandb_entity
    wandb_project = config.wandb_project
    from datetime import datetime

    if config.run_name:
        run_display_name = config.run_name
    else:
        now = datetime.now()
        run_display_name = f"run-{now.strftime('%Y%m%d-%H%M')}-{run_id[:6]}"

    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type="data_pipeline",
        name=run_display_name,
        config={"run_id": run_id, "source": "swebench", "run_name": run_display_name},
        reinit=True,
    )

    try:
        # Initialize stats for W&B logging
        dedup_stats = clean.DedupStats()
        clean_stats = clean.CleanStats()
        validation_errors: list = []

        # Progress bar for SWE-bench stages
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("SWE-bench pipeline", total=4)

            # Stage 1: Ingest
            ingest_enabled = _stage_enabled(config, "ingest")
            raw_records: list[IssueRecord] = []

            if resume_from and resume_from in ("validated", "cleaned"):
                logger.info("Resuming from '%s' -- skipping SWE-bench ingest", resume_from)
                raw_dicts = _load_stage(config, run_id, "swebench", "raw")
                raw_records = [IssueRecord(**r) for r in raw_dicts]
            elif not ingest_enabled:
                raw_records = []
            else:
                progress.update(task, description="Ingesting SWE-bench dataset...")
                logger.info("Ingesting SWE-bench dataset...")
                raw_records = swebench_ingest.ingest_swebench(config)
                _save_stage(
                    [r.model_dump() for r in raw_records],
                    config,
                    run_id,
                    "swebench",
                    "raw",
                )

            # Convert to dicts for validation (validate_batch expects dicts)
            raw_dicts = [r.model_dump() for r in raw_records]

            logger.info("SWE-bench raw records: %d", len(raw_records))
            progress.update(task, advance=1, description=f"Raw: {len(raw_records)} records")
            # Log to W&B
            wandb_run.log({"stage_raw_count": len(raw_records)})

            if not raw_records:
                raise RuntimeError("SWE-bench ingest produced 0 records")

            # BigQuery Augmentation (optional)
            if config.bigquery_enabled:
                progress.update(task, description="Augmenting with BigQuery...")
                raw_records = swebench_ingest.augment_with_bigquery(raw_records, config)
                _save_stage(
                    [r.model_dump() for r in raw_records],
                    config,
                    run_id,
                    "swebench",
                    "bigquery_augmented",
                )
                wandb_run.log({"stage_bigquery_augmented": len(raw_records)})

            # Stage 2: Validate
            validate_enabled = _stage_enabled(config, "validated")
            if resume_from and resume_from == "cleaned":
                validated_records = _load_stage(config, run_id, "swebench", "validated")
                validated = [IssueRecord(**r) for r in validated_records]
            elif not validate_enabled:
                validated = []
            else:
                progress.update(task, description="Validating records...")
                # Use augmented records if BigQuery enabled, otherwise use raw
                validation_dicts = (
                    [r.model_dump() for r in raw_records] if config.bigquery_enabled else raw_dicts
                )
                validated, validation_errors = validate.validate_batch(validation_dicts)
                _save_stage(
                    [r.model_dump() for r in validated],
                    config,
                    run_id,
                    "swebench",
                    "validated",
                )
                if validation_errors:
                    _save_stage(
                        [e.model_dump() for e in validation_errors],
                        config,
                        run_id,
                        "swebench",
                        "validation_errors",
                    )

            logger.info("SWE-bench validated records: %d", len(validated))
            progress.update(task, advance=1, description=f"Validated: {len(validated)} records")
            wandb_run.log(
                {
                    "stage_validated_count": len(validated),
                    "stage_validation_errors": len(validation_errors),
                }
            )

            if not validated and validate_enabled:
                raise RuntimeError("SWE-bench validation produced 0 valid records")

            # Stage 3: Clean
            clean_enabled = _stage_enabled(config, "cleaned")
            if not validate_enabled and not clean_enabled:
                # Validation disabled and clean disabled: stop pipeline after ingest
                logger.info("Pipeline stopped after ingest (no further stages enabled)")
                return raw_records

            if clean_enabled:
                progress.update(task, description="Cleaning records...")
                deduped, dedup_stats = clean.deduplicate(validated)
                cleaned_records, clean_stats = clean.clean_records(deduped, config)
                _save_stage(
                    [r.model_dump() for r in cleaned_records],
                    config,
                    run_id,
                    "swebench",
                    "cleaned",
                )
            else:
                cleaned_records = []

            logger.info("SWE-bench cleaned records: %d", len(cleaned_records))
            progress.update(task, advance=1, description=f"Cleaned: {len(cleaned_records)} records")
            wandb_run.log(
                {
                    "stage_cleaned_count": len(cleaned_records),
                    "dedup_exact_removed": dedup_stats.exact_duplicates_removed,
                    "dedup_content_removed": dedup_stats.content_duplicates_removed,
                    "clean_removed_no_test_files": clean_stats.removed_no_test_files,
                    "clean_removed_patch_too_large": clean_stats.removed_patch_too_large,
                    "clean_removed_binary": clean_stats.removed_binary,
                    "clean_removed_non_python": clean_stats.removed_non_python,
                    "clean_removed_empty_body": clean_stats.removed_empty_body,
                    "clean_removed_no_f2p_signal": clean_stats.removed_no_f2p_signal,
                }
            )

            # Stage 4: Complete
            progress.update(task, advance=1, description="Complete")

    finally:
        wandb_run.finish()

    return cleaned_records


# Full pipeline entry point


def _save_splits_jsonl(
    splits: Splits,
    golden_set: GoldenSet,
    config: DataPipelineConfig,
    run_id: str,
) -> None:
    """Save train/val/test/golden splits as JSONL files for tokenization."""
    out_dir = config.output_dir / run_id / "swebench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Helper to save records
    def save_records(records, filename):
        path = out_dir / filename
        with path.open("w") as f:
            for rec in records:
                if hasattr(rec, "model_dump"):
                    f.write(json.dumps(rec.model_dump(), default=str) + "\n")
                else:
                    f.write(json.dumps(rec, default=str) + "\n")
        logger.info("Saved %d records to %s", len(records), path)

    save_records(splits.train, "train.jsonl")
    save_records(splits.val, "val.jsonl")
    save_records(splits.test, "test.jsonl")
    save_records(golden_set.records, "golden.jsonl")


def run_pipeline(config: DataPipelineConfig) -> PipelineResult:
    """Run the full multi-repo pipeline end to end."""
    # Setup
    missing = config.validate_auth()
    if missing:
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. "
            "Set them as environment variables or in .env"
        )

    run_id = config.run_id_override or uuid.uuid4().hex[:12]
    resume_from = config.resume_from

    logger.info(
        "Pipeline run %s: source=swebench, resume_from=%s",
        run_id,
        resume_from,
    )

    all_cleaned: list[IssueRecord] = []
    repo_results: list[RepoResult] = []

    # SWE-bench flow (only source)
    console.print(f"[bold]Pipeline Run:[/bold] {run_id} (SWE-bench)")

    all_cleaned = run_pipeline_swebench(config, run_id, resume_from)

    repo_results = [
        RepoResult(
            repo_id="swebench",
            raw_count=0,
            validated_count=0,
            cleaned_count=len(all_cleaned),
        )
    ]

    raw_dicts = _load_stage(config, run_id, "swebench", "raw")
    validated_dicts = _load_stage(config, run_id, "swebench", "validated")
    repo_results[0].raw_count = len(raw_dicts)
    repo_results[0].validated_count = len(validated_dicts)

    if not all_cleaned:
        raise RuntimeError("Pipeline produced 0 cleaned records. Check per-repo logs.")

    # Split
    if _stage_enabled(config, "split"):
        seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "little")
        splits = split.stratified_split(all_cleaned, config, seed=seed)
    else:
        splits = split.Splits()

    # Synthetic augmentation (Phase 3 extension)
    if config.augment_codecontests or config.augment_codealpaca:
        logger.info("Augmenting training set with synthetic data...")
        orig_count = len(splits.train)
        splits.train = synthetic_augment.augment_training_data(splits.train, config)
        logger.info(
            "Training set augmented: %d records (was %d)",
            len(splits.train),
            orig_count,
        )

    # Golden
    if _stage_enabled(config, "golden"):
        golden_set = golden.build_golden_set_from_config(splits, config)
    else:
        golden_set = golden.GoldenSet()

    # Save splits as JSONL for tokenization
    _save_splits_jsonl(splits, golden_set, config, run_id)

    # Collect validation errors
    from data_engineering.schema import ValidationError

    all_validation_errors: list[ValidationError] = []
    for d in _load_stage(config, run_id, "swebench", "validation_errors"):
        try:
            all_validation_errors.append(ValidationError(**d))
        except Exception:
            pass
    total_validation_errors = len(all_validation_errors)

    # Stats
    total_raw = sum(r.raw_count for r in repo_results)
    total_validated = sum(r.validated_count for r in repo_results)

    dedup_stats_total = clean.DedupStats()
    clean_stats_total = clean.CleanStats()

    stats = PipelineStats(
        total_raw=total_raw,
        total_validated=total_validated,
        total_validation_errors=total_validation_errors,
        total_cleaned=len(all_cleaned),
        dedup_stats=dedup_stats_total,
        clean_stats=clean_stats_total,
        train_count=len(splits.train),
        val_count=len(splits.val),
        test_count=len(splits.test),
        golden_count=len(golden_set.records),
        total_examples=len(splits.train) + len(splits.val) + len(splits.test),
        repo_count=len(repo_results),
        repo_results=repo_results,
    )

    stages: dict[str, Any] = {
        "raw": [],
        "validated": [],
        "cleaned": all_cleaned,
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
        "golden": golden_set.records,
        "validation_errors": all_validation_errors,
    }

    raw_dicts = _load_stage(config, run_id, "swebench", "raw")
    validated_dicts = _load_stage(config, run_id, "swebench", "validated")
    stages["raw"] = [IssueRecord(**d) for d in raw_dicts]
    stages["validated"] = [IssueRecord(**d) for d in validated_dicts]

    wandb_artifacts: dict[str, str] = {}
    if _stage_enabled(config, "version"):
        try:
            mft_hash = _manifest_hash({})
            wandb_artifacts = version.log_dataset_artifacts(
                run_id,
                stages,
                config,
                mft_hash,
                stats=stats.model_dump(),
            )
        except Exception as exc:
            logger.warning("W&B artifact logging failed (non-fatal): %s", exc)

    stats.wandb_artifacts = {k: str(v) for k, v in wandb_artifacts.items()}

    gcs_paths: dict[str, str] = {}
    if _stage_enabled(config, "archive"):
        try:
            gcs_paths = archive.upload_to_gcs(
                run_id,
                stages,
                {},
                "",
                config,
            )
        except Exception as exc:
            logger.warning("GCS upload failed (non-fatal): %s", exc)

    stats.gcs_paths = gcs_paths

    # Dataset Card
    dataset_card_md = card.generate_dataset_card(
        manifest={},
        stats=stats,
        run_id=run_id,
        git_sha=os.environ.get("GIT_SHA", ""),
        source="swebench",
    )

    card_path = config.output_dir / run_id / "dataset_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(dataset_card_md)

    if gcs_paths and config.gcs_bucket:
        try:
            archive.upload_text_to_gcs(
                config.gcs_bucket,
                f"datasets/{run_id}",
                "dataset_card.md",
                dataset_card_md,
            )
        except Exception as exc:
            logger.warning("GCS card re-upload failed (non-fatal): %s", exc)

    # Tokenize (optional, runs if model_name provided)
    tokenized_paths: dict[str, str] = {}
    if _stage_enabled(config, "tokenize"):
        try:
            from data_engineering.tokenize import tokenize_pipeline

            model_name = getattr(config, "tokenize_model", "qwen3-14b")
            max_seq_length = getattr(config, "tokenize_max_length", 4096)

            tokenized_dir = config.output_dir / f"{run_id}_tokenized"
            ds = tokenize_pipeline(
                data_dir=config.output_dir / run_id / "swebench",
                output_dir=tokenized_dir,
                model_name=model_name,
                max_length=max_seq_length,
                config=config,
                run_id=run_id,
            )
            tokenized_paths = {
                "train": str(tokenized_dir / "train"),
                "val": str(tokenized_dir / "val"),
                "test": str(tokenized_dir / "test"),
                "golden": str(tokenized_dir / "golden"),
                "dataset_dict": str(tokenized_dir / "dataset_dict.json"),
            }
            logger.info(f"Tokenized dataset saved to {tokenized_dir}")
        except Exception as exc:
            logger.warning("Tokenization failed (non-fatal): %s", exc)

    # Print summary
    console.print("\n[bold green]Pipeline Complete![/bold green]")
    table = Table(title=f"Run {run_id[:8]} Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Train", str(len(splits.train)))
    table.add_row("Val", str(len(splits.val)))
    table.add_row("Test", str(len(splits.test)))
    table.add_row("Golden", str(len(golden_set.records)))
    table.add_row("[bold]Total[/bold]", str(stats.total_examples))
    console.print(table)

    return PipelineResult(
        run_id=run_id,
        manifest_hash=_manifest_hash({}),
        splits=splits,
        stats=stats,
        gcs_paths=gcs_paths,
        wandb_artifacts=wandb_artifacts,
        tokenized_paths=tokenized_paths,
    )


if __name__ == "__main__":
    from data_engineering.cli import app as cli_app

    cli_app()  # type: ignore[has-type]
