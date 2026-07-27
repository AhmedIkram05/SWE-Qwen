"""Typer CLI for the data pipeline.

Usage:

    # Full pipeline run
    python -m data_engineering.cli run

    # Run with custom config
    python -m data_engineering.cli run \\
        --manifest repos/manifest.json \\
        --output data/ \\
        --max-issues 1000 \\
        --parallel-workers 8

    # Resume from checkpoint
    python -m data_engineering.cli run --resume-from validated

    # Validate a single manifest
    python -m data_engineering.cli validate-manifest manifest.json

    # Show pipeline config
    python -m data_engineering.cli config
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from data_engineering.config import DataPipelineConfig
from data_engineering.run_pipeline import run_pipeline
from data_engineering.schema import PipelineResult

app = typer.Typer(
    name="data-engineering",
    help="SWE-Qwen data pipeline: ingest → validate → clean → split → golden → version → archive",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


@app.command()
def run(  # noqa: PLR0913,B008 — typer CLI dispatcher; Option() calls required by typer
    manifest: Path = typer.Option(
        "repos/manifest.json",
        "--manifest",
        "-m",
        help="Path to manifest JSON",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        "data/",
        "--output",
        "-o",
        help="Output directory for pipeline artifacts",
        file_okay=False,
    ),
    max_issues: int = typer.Option(
        2000,
        "--max-issues",
        help="Max issues per repo",
        min=1,
    ),
    parallel_workers: int = typer.Option(
        1,
        "--parallel-workers",
        "-p",
        help="Number of parallel repo workers (default 1 = sequential, avoids GitHub rate limits)",
        min=1,
        max=32,
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        "-b",
        help="Records per batch",
        min=1,
    ),
    max_patch_lines: int = typer.Option(
        500,
        "--max-patch-lines",
        help="Max lines in a patch diff",
        min=1,
    ),
    min_golden: int = typer.Option(
        200,
        "--min-golden",
        help="Minimum golden examples target",
        min=0,
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Stage to resume from (validated|cleaned)",
    ),
    stages: str | None = typer.Option(
        None,
        "--stages",
        help="Comma-separated stages to run (e.g. 'ingest,validate,clean'). Default: all",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional UUID override for the run ID",
    ),
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="Descriptive W&B run name (auto-generated if not set)",
    ),
    train_ratio: float = typer.Option(
        0.8,
        "--train-ratio",
        help="Train split ratio",
        min=0.0,
        max=1.0,
    ),
    val_ratio: float = typer.Option(
        0.1,
        "--val-ratio",
        help="Validation split ratio",
        min=0.0,
        max=1.0,
    ),
    test_ratio: float = typer.Option(
        0.1,
        "--test-ratio",
        help="Test split ratio",
        min=0.0,
        max=1.0,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
) -> None:
    """Run the full data pipeline end to end."""
    _setup_logging(verbose)

    config = DataPipelineConfig(
        manifest_path=manifest,
        output_dir=output,
        max_issues_per_repo=max_issues,
        parallel_workers=parallel_workers,
        batch_size=batch_size,
        max_patch_lines=max_patch_lines,
        min_golden_examples=min_golden,
        resume_from=resume_from,
        enabled_stages=stages.split(",") if stages else None,
        run_id_override=run_id,
        run_name=run_name,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    try:
        result: PipelineResult = run_pipeline(config)
    except RuntimeError as exc:
        typer.secho(f"Pipeline failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # Print JSON summary to stdout
    summary = {
        "run_id": result.run_id,
        "manifest_hash": result.manifest_hash,
        "stats": result.stats.model_dump(),
        "gcs_paths": result.gcs_paths,
        "wandb_artifacts": result.wandb_artifacts,
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def validate_manifest(  # noqa: B008 — typer Argument() call required
    path: Path = typer.Argument(
        ...,
        help="Path to manifest JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Validate a manifest JSON file for correctness."""
    _setup_logging(False)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        typer.secho(f"Invalid JSON: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    repos = data.get("repositories", [])
    errors: list[str] = []

    for i, repo in enumerate(repos):
        repo_id = repo.get("id", f"index {i}")
        if "owner" not in repo:
            errors.append(f"  [{repo_id}] missing 'owner'")
        if "name" not in repo:
            errors.append(f"  [{repo_id}] missing 'name'")

    if errors:
        typer.secho(
            f"Manifest validation FAILED ({len(errors)} issue(s)):",
            fg=typer.colors.RED,
            err=True,
        )
        for err in errors:
            typer.secho(err, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Manifest valid: {len(repos)} repositories", fg=typer.colors.GREEN)


@app.command()
def config() -> None:
    """Show the effective pipeline configuration."""
    _setup_logging(False)
    cfg = DataPipelineConfig()
    typer.echo(cfg.model_dump_json(indent=2))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
