"""Pipeline orchestrator.

Coordinates the full data pipeline: ingest → validate → clean → split →
golden → version → archive → card generation.

Supports checkpoint resume (``--resume-from``) and per-repo parallelism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from data_engineering import archive, card, clean, golden, ingest, split, validate, version
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import (
    IssueRecord,
    PipelineResult,
    PipelineStats,
    RepoResult,
)

logger = logging.getLogger(__name__)
console = Console()


# ── Checkpoint helpers ────────────────────────────────────────────────────


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
}

# Reverse map: human-readable → file-stage name
_HUMAN_TO_FILE = {v: k for k, v in _STAGE_MAP.items()}


def _stage_enabled(config: DataPipelineConfig, stage: str) -> bool:
    """Check if *stage* is in the enabled-stages whitelist.

    Accepts both file-stage names (``raw``, ``validated``) and human names
    (``ingest``, ``validate``). ``enabled_stages=None`` means all stages are
    enabled.
    """
    if config.enabled_stages is None:
        return True
    if stage in config.enabled_stages:
        return True
    # Translate human name to file-stage name (e.g. "ingest" → "raw")
    file_stage = _HUMAN_TO_FILE.get(stage)
    if file_stage and file_stage in config.enabled_stages:
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
    """Save records as JSONL checkpoint."""
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


def _manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute SHA256 of the serialised manifest."""
    raw = json.dumps(manifest, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ── Per-repo pipeline ─────────────────────────────────────────────────────


def run_pipeline_for_repo(
    repo_cfg: dict[str, Any],
    config: DataPipelineConfig,
    run_id: str,
    resume_from: str | None,
    gh: Any = None,
) -> RepoResult:
    """Run the complete pipeline for a single repo.

    Stages: ingest → validate → clean
    (split/golden are global operations across all repos).
    """
    repo_id: str = repo_cfg["id"]
    result = RepoResult(repo_id=repo_id)

    try:
        # ── Stage 1: Ingest ─────────────────────────────────────────────
        ingest_enabled = _stage_enabled(config, "raw")
        if resume_from and resume_from in ("validated", "cleaned"):
            logger.info("Resuming %s from '%s' — skipping ingest", repo_id, resume_from)
            raw_records = _load_stage(config, run_id, repo_id, "raw")
        elif not ingest_enabled:
            raw_records = []
        else:
            _gh = gh or ingest.get_github_client()  # shared client when available
            raw_records = ingest.ingest_repo(repo_cfg, _gh, config)
            _save_stage(raw_records, config, run_id, repo_id, "raw")

        result.raw_count = len(raw_records)
        if not raw_records:
            logger.warning("Repo %s: 0 raw records (zero-yield), skipping further stages", repo_id)
            return result

        # ── Stage 2: Validate ───────────────────────────────────────────
        validate_enabled = _stage_enabled(config, "validated")
        if resume_from and resume_from == "cleaned":
            validated_records = _load_stage(config, run_id, repo_id, "validated")
            validated = [IssueRecord(**r) for r in validated_records]
        elif not validate_enabled:
            validated = []
        else:
            validated, validation_errors = validate.validate_batch(raw_records)
            _save_stage(
                [r.model_dump() for r in validated],
                config,
                run_id,
                repo_id,
                "validated",
            )
            if validation_errors:
                _save_stage(
                    [e.model_dump() for e in validation_errors],
                    config,
                    run_id,
                    repo_id,
                    "validation_errors",
                )

        result.validated_count = len(validated)
        if not validated:
            logger.warning("Repo %s: 0 valid records after validation (zero-yield)", repo_id)
            return result

        # ── Stage 3: Clean ──────────────────────────────────────────────
        if _stage_enabled(config, "cleaned"):
            deduped, dedup_stats = clean.deduplicate(validated)
            cleaned_records, clean_stats = clean.clean_records(deduped, config)
            _save_stage(
                [r.model_dump() for r in cleaned_records],
                config,
                run_id,
                repo_id,
                "cleaned",
            )
        else:
            cleaned_records = []

        result.cleaned_count = len(cleaned_records)

    except Exception as exc:
        logger.exception("Pipeline failed for repo %s: %s", repo_id, exc)
        result.error = str(exc)

    return result


# ── Full pipeline ─────────────────────────────────────────────────────────


def run_pipeline(config: DataPipelineConfig) -> PipelineResult:
    """Run the full multi-repo pipeline end to end.

    Returns a ``PipelineResult`` with all splits, stats, and paths.
    """
    # ── Setup ───────────────────────────────────────────────────────────
    missing = config.validate_auth()
    if missing:
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. "
            "Set them as environment variables or in .env"
        )

    run_id = config.run_id_override or uuid.uuid4().hex[:12]
    manifest = ingest.load_manifest(str(config.manifest_path))
    mft_hash = _manifest_hash(manifest)
    repos: list[dict[str, Any]] = manifest.get("repositories", [])

    resume_from = config.resume_from

    # Create ONE shared Github client for all repos (shares connection pool + rate-limit tracking)
    gh = ingest.get_github_client()

    logger.info(
        "Pipeline run %s: %d repos, manifest hash=%s",
        run_id,
        len(repos),
        mft_hash,
    )

    all_validated: list[IssueRecord] = []
    all_raw: list[dict[str, Any]] = []

    # ── Per-repo processing (parallel) ──────────────────────────────────
    repo_results: list[RepoResult] = []

    console.print(f"[bold]Pipeline Run:[/bold] {run_id}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(repos)} repos...", total=len(repos))

        with ThreadPoolExecutor(max_workers=config.parallel_workers) as pool:
            futures = {
                pool.submit(run_pipeline_for_repo, r, config, run_id, resume_from, gh): r["id"]
                for r in repos
            }
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    repo_res = future.result()
                    repo_results.append(repo_res)
                    if repo_res.error:
                        console.print(f"  [red]✗[/red] {rid}: {repo_res.error}")
                    else:
                        progress.update(
                            task,
                            advance=1,
                            description=f"{rid}: {repo_res.cleaned_count} cleaned",
                        )
                except Exception as exc:
                    logger.exception("Unhandled exception for repo %s", rid)
                    repo_results.append(RepoResult(repo_id=rid, error=str(exc)))
                    progress.update(task, advance=1)

    # ── Zero-yield warning summary ──────────────────────────────────────
    zero_yield_repos = [r for r in repo_results if r.raw_count == 0 and not r.error]
    if zero_yield_repos:
        logger.warning(
            "Zero-yield repos (%d): %s",
            len(zero_yield_repos),
            ", ".join(r.repo_id for r in zero_yield_repos),
        )
        console.print(
            f"[yellow]⚠ {len(zero_yield_repos)} repo(s) yielded 0 records: "
            f"{', '.join(r.repo_id for r in zero_yield_repos)}[/yellow]"
        )

    # ── Aggregate all cleaned records ───────────────────────────────────
    all_cleaned: list[IssueRecord] = []
    total_raw = 0
    total_validated = 0
    total_validation_errors = 0
    dedup_stats_total = clean.DedupStats()
    clean_stats_total = clean.CleanStats()

    for r in repo_results:
        total_raw += r.raw_count
        total_validated += r.validated_count

        # Load cleaned records from checkpoint
        cleaned_dicts = _load_stage(config, run_id, r.repo_id, "cleaned")
        all_cleaned.extend(IssueRecord(**d) for d in cleaned_dicts)

    if not all_cleaned:
        raise RuntimeError("Pipeline produced 0 cleaned records. Check per-repo logs.")

    # Re-run dedup + clean globally to get accurate total stats
    # (per-repo dedup is approximate; global dedup is authoritative)
    all_cleaned, dedup_stats_total = clean.deduplicate(all_cleaned)
    all_cleaned, clean_stats_total = clean.clean_records(all_cleaned, config)

    # ── Split ───────────────────────────────────────────────────────────
    if _stage_enabled(config, "split"):
        seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "little")
        splits = split.stratified_split(all_cleaned, config, seed=seed)
    else:
        splits = split.Splits()

    # ── Golden ──────────────────────────────────────────────────────────
    if _stage_enabled(config, "golden"):
        golden_set = golden.build_golden_set_from_config(splits, config)
    else:
        golden_set = golden.GoldenSet()

    # ── Collect validation errors ───────────────────────────────────────
    from data_engineering.schema import ValidationError

    all_validation_errors: list[ValidationError] = []
    for r in repo_results:
        for d in _load_stage(config, run_id, r.repo_id, "validation_errors"):
            try:
                all_validation_errors.append(ValidationError(**d))
            except Exception:
                pass  # skip malformed error records
    total_validation_errors = len(all_validation_errors)

    # ── Stats ───────────────────────────────────────────────────────────
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
        repo_count=len(repos),
        repo_results=repo_results,
    )

    # ── Version (W&B) ───────────────────────────────────────────────────
    stages: dict[str, Any] = {
        "raw": [
            IssueRecord(**d)
            for r in repo_results
            for d in _load_stage(config, run_id, r.repo_id, "raw")
        ],
        "validated": [
            IssueRecord(**d)
            for r in repo_results
            for d in _load_stage(config, run_id, r.repo_id, "validated")
        ],
        "cleaned": all_cleaned,
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
        "golden": golden_set.records,
        "validation_errors": all_validation_errors,
    }

    wandb_artifacts: dict[str, str] = {}
    if _stage_enabled(config, "version"):
        try:
            wandb_artifacts = version.log_dataset_artifacts(
                run_id,
                stages,
                config,
                mft_hash,
                stats=stats.model_dump(),
            )
        except Exception as exc:
            logger.warning("W&B artifact logging failed (non-fatal): %s", exc)

    # ── Update stats with W&B paths BEFORE card ─────────────────────────
    stats.wandb_artifacts = {k: str(v) for k, v in wandb_artifacts.items()}

    # ── Archive (GCS) ───────────────────────────────────────────────────
    gcs_paths: dict[str, str] = {}
    if _stage_enabled(config, "archive"):
        try:
            # Upload stages + manifest first; card gets placeholder,
            # overwritten after generation below
            gcs_paths = archive.upload_to_gcs(
                run_id,
                stages,
                manifest,
                "",  # placeholder card — overwritten below
                config,
            )
        except Exception as exc:
            logger.warning("GCS upload failed (non-fatal): %s", exc)

    # ── Update stats with GCS paths BEFORE card ─────────────────────────
    stats.gcs_paths = gcs_paths

    # ── Dataset Card (now with full W&B + GCS paths) ────────────────────
    dataset_card_md = card.generate_dataset_card(
        manifest,
        stats,
        run_id,
        git_sha=os.environ.get("GIT_SHA", ""),
    )

    # Write dataset card locally
    card_path = config.output_dir / run_id / "dataset_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(dataset_card_md)

    # Overwrite card on GCS with full paths
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
        manifest_hash=mft_hash,
        splits=splits,
        stats=stats,
        gcs_paths=gcs_paths,
        wandb_artifacts=wandb_artifacts,
    )


if __name__ == "__main__":
    """Entry point: ``python -m data_engineering.run_pipeline``.

    Delegates to the Typer CLI so both entry points behave identically.
    """
    from data_engineering.cli import app as cli_app

    cli_app()
