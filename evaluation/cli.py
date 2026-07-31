"""Typer CLI for the SWE-Qwen evaluation harness.

Usage::

    # Full golden-set evaluation of the three P4 variants
    python -m evaluation.cli run

    # SWE-bench verified split, light sample
    python -m evaluation.cli run --split swebench_verified --sample 50

    # Compare existing runs (local JSONL, or W&B artifacts)
    python -m evaluation.cli compare --run_ids run_a,run_b
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NoReturn

import typer

from evaluation.comparison import (
    compare_and_report,
    extract_model_metrics,
    load_all_eval_runs,
    proxy_champion_from_f2p_proxy,
)
from evaluation.config import EvalConfig
from evaluation.schema import EvalRun

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = "qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b"

app = typer.Typer(
    name="eval",
    help="SWE-Qwen evaluation harness CLI",
    no_args_is_help=True,
)


@app.command()
def run(
    models: str = typer.Option(_DEFAULT_MODELS, help="comma-separated model:variant pairs"),
    split: str = typer.Option("golden", help="golden|swebench_verified"),
    prompts: str = typer.Option("chat", help="comma-separated prompt templates"),
    sample: int = typer.Option(0, help="0 = all"),
    resume: str | None = typer.Option(None, help="run_id to resume"),
    ci_mode: bool = typer.Option(False, help="sample=50, seed=42"),
) -> None:
    """Main evaluation entry point."""
    config = EvalConfig()
    pairs = _parse_model_pairs(models)
    templates = _parse_prompts(prompts)
    if not templates:
        raise typer.BadParameter("expected at least one prompt template")
    if ci_mode:
        sample = config.ci_sample_size
    try:
        eval_run = _dispatch(split, pairs, templates, sample, resume, config)
    except KeyboardInterrupt:
        _echo_interrupt(resume)
    _report_run(eval_run)


@app.command()
def run_golden(
    models: str = typer.Option(_DEFAULT_MODELS, help="comma-separated model:variant pairs"),
    prompts: str = typer.Option("chat", help="comma-separated prompt templates"),
    sample: int = typer.Option(0, help="0 = all"),
    resume: str | None = typer.Option(None, help="run_id to resume"),
) -> None:
    """Run golden-set evaluation for model:variant pairs."""
    config = EvalConfig()
    pairs = _parse_model_pairs(models)
    templates = _parse_prompts(prompts)
    if not templates:
        raise typer.BadParameter("expected at least one prompt template")
    try:
        eval_run = _dispatch("golden", pairs, templates, sample, resume, config)
    except KeyboardInterrupt:
        _echo_interrupt(resume)
    _report_run(eval_run)


@app.command()
def run_swebench(
    models: str = typer.Option(_DEFAULT_MODELS, help="comma-separated model:variant pairs"),
    sample: int = typer.Option(0, help="0 = all"),
    resume: str | None = typer.Option(None, help="run_id to resume"),
) -> None:
    """Run evaluation on the SWE-bench verified subset."""
    config = EvalConfig()
    pairs = _parse_model_pairs(models)
    try:
        eval_run = _dispatch("swebench_verified", pairs, ["chat"], sample, resume, config)
    except KeyboardInterrupt:
        _echo_interrupt(resume)
    _report_run(eval_run)


@app.command()
def run_prompt_ab(
    model: str = typer.Option("qwen3-14b"),
    variant: str = typer.Option("baseline_14b"),
    templates: str = typer.Option("", help="comma-separated templates; empty = all available"),
    sample: int = typer.Option(200, help="examples per template"),
) -> None:
    """Run prompt-template A/B evaluation."""
    from evaluation.prompt_ab_test import run_prompt_ab_test

    config = EvalConfig()
    template_list = _parse_prompts(templates)
    if not template_list:
        from training.prompt_loader import PromptLoader

        template_list = PromptLoader().available_templates
    try:
        eval_run = run_prompt_ab_test(
            config, model=model, variant=variant, templates=template_list, sample=sample
        )
    except KeyboardInterrupt:
        _echo_interrupt(None)
    _report_run(eval_run)


@app.command()
def run_baseline(
    model: str = typer.Option("Qwen/Qwen3-14B", help="baseline model id"),
    sample: int = typer.Option(0, help="0 = all"),
) -> None:
    """Run baseline (untuned) model evaluation."""
    from evaluation.harness import EvaluationHarness

    config = EvalConfig()
    try:
        eval_run = EvaluationHarness(config).run_baseline(model=model, sample=sample)
    except KeyboardInterrupt:
        _echo_interrupt(None)
    _report_run(eval_run)


@app.command()
def compare(
    run_ids: str = typer.Option(..., "--run_ids", help="comma-separated run_ids"),
    proxy: bool = typer.Option(True, help="annotate proxy champion"),
    golden_path: str | None = typer.Option(None, help="golden.jsonl for P4 proxy champion scoring"),
    variant_adapter_map: str | None = typer.Option(
        None, help="comma-separated variant=adapter pairs (requires --golden-path)"
    ),
) -> None:
    """Compare multiple eval runs, output markdown table."""
    config = EvalConfig()
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    runs = load_all_eval_runs(ids, config)
    metrics = extract_model_metrics(runs)
    if not metrics:
        typer.echo("no runs loaded; nothing to compare", err=True)
        raise typer.Exit(code=1)
    proxy_champion = _resolve_proxy_champion(proxy, golden_path, variant_adapter_map)
    typer.echo(compare_and_report(metrics, proxy_champion=proxy_champion))


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_model_pairs(value: str) -> list[tuple[str, str]]:
    """Parse comma-separated ``model:variant`` pairs."""
    pairs: list[tuple[str, str]] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        model, _, variant = part.partition(":")
        if not model or not variant:
            raise typer.BadParameter(f"expected 'model:variant', got {part!r}")
        pairs.append((model, variant))
    return pairs


def _parse_prompts(value: str) -> list[str]:
    """Parse comma-separated prompt template names."""
    return [template.strip() for template in value.split(",") if template.strip()]


def _parse_adapter_map(value: str) -> dict[str, str]:
    """Parse comma-separated ``variant=adapter`` pairs."""
    mapping: dict[str, str] = {}
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        variant, _, adapter = part.partition("=")
        if not variant or not adapter:
            raise typer.BadParameter(f"expected 'variant=adapter', got {part!r}")
        mapping[variant] = adapter
    return mapping


def _dispatch(
    split: str,
    pairs: list[tuple[str, str]],
    templates: list[str],
    sample: int,
    resume: str | None,
    config: EvalConfig,
) -> EvalRun:
    """Dispatch a run to the harness entry point for *split*."""
    from evaluation.harness import EvaluationHarness  # lazy: heavy deps

    harness = EvaluationHarness(config)
    if split == "swebench_verified":
        return harness.run_swebench_verified(pairs, sample=sample, run_id=resume)
    if split == "golden":
        return harness.run_golden(pairs, prompt_templates=templates, sample=sample, run_id=resume)
    raise typer.BadParameter(f"unknown split {split!r}; expected 'golden' or 'swebench_verified'")


def _resolve_proxy_champion(
    proxy: bool, golden_path: str | None, variant_adapter_map: str | None
) -> str | None:
    """Resolve the proxy champion name for comparison annotations."""
    if not proxy:
        return None
    if (golden_path is None) != (variant_adapter_map is None):
        raise typer.BadParameter(
            "--golden-path and --variant-adapter-map must be provided together"
        )
    if golden_path is not None and variant_adapter_map is not None:
        return proxy_champion_from_f2p_proxy(
            Path(golden_path), _parse_adapter_map(variant_adapter_map)
        )
    return "baseline_14b"


def _report_run(eval_run: EvalRun) -> None:
    """Print the aggregate summary table and run_id for *eval_run*."""
    metrics = {f"{m.model_name}:{m.variant}": m for m in eval_run.aggregate}
    typer.echo(compare_and_report(metrics))
    typer.echo(f"run_id: {eval_run.run_id}")


def _echo_interrupt(resume: str | None) -> NoReturn:
    """Echo the resume hint after a KeyboardInterrupt, then exit 130."""
    typer.echo(f"interrupted — use --resume {resume or '<run_id>'} to continue")
    raise typer.Exit(code=130)


if __name__ == "__main__":
    app()
