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

import json
import logging
from pathlib import Path
from typing import NoReturn

import typer

from evaluation.comparison import (
    compare_and_report,
    extract_model_metrics,
    load_all_eval_runs,
    paired_significance,
    promote_champion_to_registry,
    proxy_champion_from_f2p_proxy,
    revalidate_champion,
)
from evaluation.config import EvalConfig
from evaluation.schema import EvalRun

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = "qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b"

# --mode presets: deterministic seed-42 subsets (see config.tier_seed).
# smoke/dev/final evaluate the SWE-bench Verified split; full = whole golden set.
_TIER_SPLITS = {
    "smoke": "swebench_verified",
    "dev": "swebench_verified",
    "final": "swebench_verified",
    "full": "golden",
}

# CI gate: fail if a variant's F2P drops more than this (absolute) vs stored baseline.
_SMOKE_TOLERANCE = 0.05

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
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="smoke|dev|final|full preset (seed-42 subsets); smoke runs the CI F2P gate",
    ),
    backend: str = typer.Option("modal", help="modal|local"),
    ollama_model: str = typer.Option("qwen2.5-coder:7b", help="Ollama model tag (local backend)"),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL"),
) -> None:
    """Main evaluation entry point."""
    config = EvalConfig()
    if mode is not None:
        if mode not in config.tier_sizes:
            raise typer.BadParameter(
                f"unknown mode {mode!r}; expected one of {', '.join(config.tier_sizes)}"
            )
        split = _TIER_SPLITS[mode]
        sample = config.tier_sizes[mode]
    pairs = _parse_model_pairs(models)
    templates = _parse_prompts(prompts)
    if not templates:
        raise typer.BadParameter("expected at least one prompt template")
    if ci_mode:
        sample = config.ci_sample_size
    try:
        eval_run = _dispatch(
            split,
            pairs,
            templates,
            sample,
            resume,
            config,
            backend=backend,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )
    except KeyboardInterrupt:
        _echo_interrupt(resume)
    _report_run(eval_run)
    if mode == "smoke":
        _smoke_gate(eval_run, config)


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
    champion = revalidate_champion(
        metrics, proxy_champion or "", config.min_f2p_threshold, config.min_p2p_threshold
    )
    if champion:
        promoted = promote_champion_to_registry(champion[0], config)
        if promoted:
            typer.echo(promoted)
    if len(runs) >= 2:  # noqa: PLR2004
        ranked_runs = sorted(runs, key=_run_best_f2p, reverse=True)
        typer.echo(paired_significance(ranked_runs[0], ranked_runs[1]))
    total_cost = sum(r.cost_usd for r in runs)
    if total_cost > 0:
        typer.echo(f"est. total cost across {len(runs)} run(s): ${total_cost:.2f}")


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
    *,
    backend: str = "modal",
    ollama_model: str = "qwen2.5-coder:7b",
    ollama_url: str = "http://localhost:11434",
) -> EvalRun:
    """Dispatch a run to the harness entry point for *split*."""
    from evaluation.harness import EvaluationHarness  # lazy: heavy deps

    if backend == "local":
        _patch_harness_backend(ollama_model=ollama_model, ollama_url=ollama_url)

    harness = EvaluationHarness(config)
    if split == "swebench_verified":
        return harness.run_swebench_verified(pairs, sample=sample, run_id=resume)
    if split == "golden":
        return harness.run_golden(pairs, prompt_templates=templates, sample=sample, run_id=resume)
    raise typer.BadParameter(f"unknown split {split!r}; expected 'golden' or 'swebench_verified'")


def _patch_harness_backend(*, ollama_model: str, ollama_url: str) -> None:
    """Monkeypatch harness indirection functions with local backends."""
    import evaluation.harness as harness_mod
    from evaluation.local_backend import generate_patches_local, run_tests_local

    def _gen_patches(model_name, variant, prompt_template, examples, **kwargs):
        return generate_patches_local(
            model_name,
            variant,
            prompt_template,
            examples,
            ollama_model=ollama_model,
            ollama_base_url=ollama_url,
        )

    def _run_tests(example, generated_patch, config):
        return run_tests_local(example, generated_patch, config)

    harness_mod._generate_patches = _gen_patches  # type: ignore[attr-defined]
    harness_mod._run_tests = _run_tests  # type: ignore[attr-defined]
    harness_mod._run_tests_batch_modal = _raise_modal_disabled  # type: ignore[attr-defined]


def _raise_modal_disabled(*args, **kwargs):
    """Stub for local runs — skip Modal container, go straight to fallback."""
    raise RuntimeError("Modal batch disabled for local backend")


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


def _run_best_f2p(run: EvalRun) -> float:
    """Best f2p_rate across a run's aggregate groups (for ranking runs)."""
    return max((m.f2p_rate for m in run.aggregate), default=0.0)


def _report_run(eval_run: EvalRun) -> None:
    """Print the aggregate summary table and run_id for *eval_run*."""
    metrics = {f"{m.model_name}:{m.variant}": m for m in eval_run.aggregate}
    typer.echo(compare_and_report(metrics))
    typer.echo(f"run_id: {eval_run.run_id}")


def _smoke_gate(eval_run: EvalRun, config: EvalConfig) -> None:
    """CI regression gate: exit 1 if any model:variant F2P dropped > tolerance.

    Baseline lives at ``{output_dir}/smoke_baseline.json`` as
    ``{"model:variant:prompt": f2p_rate}``.  The first run writes the baseline
    and passes; later runs fail when F2P < baseline - ``_SMOKE_TOLERANCE``.
    """
    baseline_path = config.output_dir / "smoke_baseline.json"
    baseline: dict[str, float] = {}
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text())
        except json.JSONDecodeError:
            typer.echo(f"SMOKE GATE FAIL: corrupt baseline {baseline_path}", err=True)
            raise typer.Exit(code=1) from None
    # key by (model, variant, prompt) — a multi-prompt run must not let one
    # template's rate silently mask another's
    current = {
        f"{m.model_name}:{m.variant}:{m.prompt_template}": m.f2p_rate for m in eval_run.aggregate
    }
    if not current:
        typer.echo(
            "SMOKE GATE FAIL: run produced no aggregate metrics (all repos checkpointed?)", err=True
        )
        raise typer.Exit(code=1)
    if not baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        typer.echo(f"smoke baseline written: {baseline_path}")
        return
    failures = [
        (key, rate, baseline[key])
        for key, rate in current.items()
        if key in baseline and rate < baseline[key] - _SMOKE_TOLERANCE
    ]
    # a variant previously gated that vanished from this run is also a failure
    missing = [key for key in baseline if key not in current]
    if failures or missing:
        for key, rate, base in failures:
            typer.echo(
                f"SMOKE GATE FAIL: {key} f2p {rate:.2%} < baseline {base:.2%}"
                f" (drop > {_SMOKE_TOLERANCE:.0%})"
            )
        for key in missing:
            typer.echo(f"SMOKE GATE FAIL: {key} missing from this run (was in baseline)")
        raise typer.Exit(code=1)
    baseline.update(current)
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    typer.echo("smoke gate passed; baseline updated")


def _echo_interrupt(resume: str | None) -> NoReturn:
    """Echo the resume hint after a KeyboardInterrupt, then exit 130."""
    typer.echo(f"interrupted — use --resume {resume or '<run_id>'} to continue")
    raise typer.Exit(code=130)


if __name__ == "__main__":
    app()
