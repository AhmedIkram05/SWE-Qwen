"""Prompt template A/B testing for the evaluation harness.

Compares prompt templates (``chat``, ``system``, ``user``, ``assistant``) on a
sampled golden set: the same model+variant runs each template and the results
are logged as one W&B artifact so templates can be compared side by side.

Plain Python (no Modal decorator) — the CLI calls this directly, and the
harness (not this module) owns the Modal indirection.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime

from evaluation.config import EvalConfig
from evaluation.harness import EvaluationHarness, _persist_run, make_run_id
from evaluation.metrics import aggregate_metrics
from evaluation.schema import EvalResult, EvalRun, F2PMetrics

logger = logging.getLogger(__name__)

_FALLBACK_TEMPLATES = ["system", "user", "assistant", "chat"]


def _default_templates() -> list[str]:
    """Return all templates shipped with PromptLoader (lazy import), or fallback."""
    try:
        from training.prompt_loader import PromptLoader

        templates = PromptLoader().available_templates
    except Exception:  # noqa: BLE001 — template discovery must never block the run
        logger.warning("PromptLoader unavailable — using fallback templates", exc_info=True)
        return list(_FALLBACK_TEMPLATES)
    return templates or list(_FALLBACK_TEMPLATES)


def run_prompt_ab_test(  # noqa: PLR0913, PLR0917 — spec-mandated public signature
    config: EvalConfig,
    model: str = "qwen3-14b",
    variant: str = "baseline_14b",
    templates: list[str] | None = None,
    sample: int = 200,
    run_id: str | None = None,
) -> EvalRun:
    """Run the harness on a golden sample with each prompt template.

    Args:
        config: Eval config (sampling seed, checkpoint/output dirs, W&B).
        model: Model key (``models.yaml`` entry, e.g. ``"qwen3-14b"``).
        variant: LoRA variant key (adapter resolution; baseline if unresolved).
        templates: Template names to compare; defaults to all templates from
            ``PromptLoader.available_templates`` (or the four known ones:
            ``system``, ``user``, ``assistant``, ``chat``).
        sample: Number of golden examples per template (0 = all), sampled
            with ``config.tier_seed``.
        run_id: Eval run id; generated if not provided.

    Returns:
        Assembled EvalRun whose results span every template; per-example
        results are logged to the ``eval-prompt-ab-{run_id}`` W&B artifact.
    """
    harness = EvaluationHarness(config)
    run_id = run_id or make_run_id()
    started_at = datetime.now(UTC)
    templates = templates or _default_templates()

    examples = harness.load_examples("golden", run_id=run_id)
    if sample > 0:
        examples = random.Random(config.tier_seed).sample(examples, min(sample, len(examples)))
        logger.info(
            "A/B test sampled %d of %d examples (seed=%d)",
            len(examples),
            sample,
            config.tier_seed,
        )

    results: list[EvalResult] = []
    for template in templates:
        logger.info("A/B template %r: %d examples", template, len(examples))
        results.extend(harness.run_batch(examples, model, variant, template, run_id))

    aggregate: list[F2PMetrics] = []
    for template in templates:
        group = [r for r in results if r.prompt_template == template]
        if group:
            aggregate.append(aggregate_metrics(group))

    run = EvalRun(
        run_id=run_id,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=[f"{model}:{variant}"],
        results=results,
        aggregate=aggregate,
        status="partial" if any(r.error for r in results) else "completed",
    )
    _persist_run(run, config)
    harness.wandb_logger.log_per_example(results, run_id, artifact_name=f"eval-prompt-ab-{run_id}")
    return run
