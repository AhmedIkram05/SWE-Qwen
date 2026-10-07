"""Re-validate the Phase 4 proxy champion with real F2P evaluation.

Phase 4 selected a champion variant using a training-loss proxy
(``scripts/f2p_proxy.py``). This module loads completed evaluation runs
(local JSONL or W&B artifacts), aggregates real F2P/P2P metrics per
model:variant, and re-runs the champion selection against the quality
gates (P2P >= 90% regression ceiling, F2P >= 15% quality floor — ADR-005).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics, is_instance_resolved
from evaluation.schema import EvalResult, EvalRun, F2PMetrics
from evaluation.stats import mcnamar_p, paired_bootstrap_ci, wilson_ci
from scripts.f2p_proxy import compute_proxy_f2p_scores, select_champion

logger = logging.getLogger(__name__)

# Mirrors EvalConfig.min_f2p_threshold / min_p2p_threshold defaults —
# compare_and_report() has no config access and the rejection labels are
# fixed by the CLI spec.
_MIN_F2P_RATE = 0.15
_MIN_P2P_RATE = 0.90

# W&B artifact name/type for persisted eval runs (JSONL of EvalResult).
_ARTIFACT_NAME = "eval-results-{run_id}"
_ARTIFACT_TYPE = "eval_results"

# Fallback proxy champion when P4 scoring inputs are unavailable.
# Now configurable via EvalConfig.proxy_champion_fallback (default: "baseline_14b")


def load_all_eval_runs(run_ids: list[str], config: EvalConfig) -> list[EvalRun]:
    """Load evaluation runs from local files or W&B artifacts.

    Each run is read from ``{config.output_dir}/{run_id}.json`` (JSONL of
    ``EvalResult``, or a single ``EvalRun`` JSON dump) if present — this is
    the local-first path used by tests. Otherwise it is downloaded from the
    W&B artifact ``eval-results-{run_id}`` (type ``eval_results``). Runs
    that are missing or unparseable are warned about and skipped; this
    function never raises.

    Args:
        run_ids: Run IDs to load.
        config: Eval config providing ``output_dir`` and W&B coordinates.

    Returns:
        The loaded runs in ``run_ids`` order (missing runs omitted).
    """
    runs: list[EvalRun] = []
    for run_id in run_ids:
        run = _load_local_run(run_id, config) or _load_wandb_run(run_id, config)
        if run is None:
            logger.warning("run %s not found locally or in W&B — skipping", run_id)
        else:
            runs.append(run)
    return runs


def extract_model_metrics(runs: list[EvalRun]) -> dict[str, F2PMetrics]:
    """Aggregate per-model metrics across runs.

    Results are merged by ``model_name:variant`` key and re-aggregated with
    ``aggregate_metrics`` so that one run per (model, variant) pair is
    produced regardless of how the runs were split.

    Shared instances across runs (e.g. a smoke-20 run then a dev-100 run of
    the same tier subset) are deduped by ``instance_id`` — last occurrence
    wins — so overlapping evaluations are not double-counted.

    Args:
        runs: Evaluation runs to aggregate.

    Returns:
        Mapping of ``"model_name:variant"`` to merged F2PMetrics.
    """
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for run in runs:
        for result in run.results:
            grouped[f"{result.model_name}:{result.variant}"].append(result)
    merged: dict[str, F2PMetrics] = {}
    for key, results in grouped.items():
        seen: set[str] = set()
        deduped: list[EvalResult] = []
        for result in results:
            if result.instance_id:
                if result.instance_id in seen:
                    continue
                seen.add(result.instance_id)
            deduped.append(result)
        merged[key] = aggregate_metrics(deduped)
    return merged


def revalidate_champion(
    metrics: dict[str, F2PMetrics],
    proxy_champion: str,
    min_f2p: float,
    min_p2p: float,
) -> tuple[str, F2PMetrics] | None:
    """Re-validate the P4 proxy champion against real F2P metrics.

    1. Filter candidates: ``p2p_rate >= min_p2p`` (regression ceiling,
       ADR-005) and ``f2p_rate >= min_f2p`` (quality floor).
    2. Rank survivors by ``f2p_rate`` descending.
    3. Return the rank-1 ``(model, metrics)`` — the champion. If no model
       passes both gates, return ``None`` (no promotion).

    NOTE: this ranks REAL F2P; it deliberately does not call
    ``select_champion`` (which is for the P4 proxy path only).

    Args:
        metrics: Per-model metrics keyed by ``"model_name:variant"``.
        proxy_champion: The P4 proxy champion variant name (informational;
            selection here is purely F2P-based).
        min_f2p: Quality floor on ``f2p_rate``.
        min_p2p: Regression ceiling on ``p2p_rate``.

    Returns:
        ``(model_key, metrics)`` of the top candidate, or ``None`` when no
        candidate clears both gates.
    """
    candidates = [
        (model, m)
        for model, m in metrics.items()
        if m.p2p_rate >= min_p2p and m.f2p_rate >= min_f2p
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1].f2p_rate, reverse=True)
    return candidates[0]


def proxy_champion_from_f2p_proxy(golden_path: Path, variant_adapter_map: dict[str, str]) -> str:
    """Select the P4 proxy champion using the Phase 4 proxy scorer.

    DRY-RUN ONLY (P0-4): the train-loss proxy can invert the true F2P
    ranking, so its result annotates comparisons only. Champion promotion
    requires real F2P via ``python -m evaluation.cli compare`` (gated by
    :func:`revalidate_champion` below).

    Thin wrapper around ``scripts.f2p_proxy`` (P4 artifact — reused, not
    reimplemented): scores each variant from W&B training loss and returns
    the winning variant name.

    Args:
        golden_path: Path to the golden.jsonl set (validated by the proxy).
        variant_adapter_map: Mapping of variant name to adapter reference.

    Returns:
        The proxy champion variant name.
    """
    scores = compute_proxy_f2p_scores(golden_path, variant_adapter_map)
    return select_champion(scores)


def compare_and_report(metrics: dict[str, F2PMetrics], proxy_champion: str | None = None) -> str:
    """Render a markdown comparison table of per-model metrics.

    Rows are ranked by ``f2p_rate`` descending. When *proxy_champion* is
    given, rows are annotated:

    - ``[proxy-champion]`` — the P4 proxy pick,
    - ``[champion]`` — current rank-1 candidate clearing both gates,
    - ``[rejected: p2p<90%]`` / ``[rejected: f2p<15%]`` — gate failures.

    Fairness (P0-5): ``f2p_95ci`` bounds instance-resolve
    (``resolve_count / total``, strict ``F2P==1.0``) via Wilson — NOT the
    mean partial-credit ``f2p_rate``. Binomial CI on a mean is invalid;
    partial 0.5s are not Bernoulli trials. ``apply_rate``
    (applied/total) vs ``resolve_rate`` vs ``cond_resolve``
    (resolved|applied) splits apply-fail from wrong-patch.

    Args:
        metrics: Per-model metrics keyed by ``"model_name:variant"``.
        proxy_champion: Variant name selected by the P4 proxy (matched
            against the variant part of each key), or ``None``.

    Returns:
        The markdown table as a string.
    """
    ranked = sorted(metrics.items(), key=lambda item: item[1].f2p_rate, reverse=True)
    passing = [
        key
        for key, m in metrics.items()
        if m.p2p_rate >= _MIN_P2P_RATE and m.f2p_rate >= _MIN_F2P_RATE
    ]
    champion = max(passing, key=lambda key: metrics[key].f2p_rate) if passing else None

    lines = [
        "| model | variant | total | f2p_rate | f2p_95ci | p2p_rate | avg_latency | flaky_rate | apply_rate | resolve_rate | cond_resolve | note |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in ranked:
        model, _, variant = key.partition(":")
        # P0-5: the CI bounds instance-resolve (strict F2P==1.0), NOT the
        # mean partial-credit f2p_rate — a binomial CI on a mean is invalid
        # (partial 0.5s are not Bernoulli trials).
        lo, hi = wilson_ci(m.resolve_count, m.total_examples)
        notes = []
        if proxy_champion is not None and (
            key == proxy_champion or key.endswith(f":{proxy_champion}")
        ):
            notes.append("[proxy-champion]")
        if champion == key:
            notes.append("[champion]")
        if m.p2p_rate < _MIN_P2P_RATE:
            notes.append("[rejected: p2p<90%]")
        if m.f2p_rate < _MIN_F2P_RATE:
            notes.append("[rejected: f2p<15%]")
        lines.append(
            f"| {model} | {variant} | {m.total_examples} | {m.f2p_rate:.2%} | {lo:.1%}-{hi:.1%} "
            f"| {m.p2p_rate:.2%} | {m.avg_latency:.2f} | {m.flaky_test_rate:.2%} "
            f"| {m.apply_rate:.2%} | {m.resolve_rate:.2%} | {m.conditional_resolve_rate:.2%} "
            f"| {' '.join(notes)} |"
        )
    return "\n".join(lines)


def _mcnemar_line(
    label: str,
    a_scores: dict[str, float],
    b_scores: dict[str, float],
) -> str | None:
    """One McNemar + paired-bootstrap line for shared instances, or None when
    the two sides share no instance (nothing to pair)."""
    shared = [i for i in a_scores if i in b_scores]
    if not shared:
        return None
    b01 = sum(1 for i in shared if a_scores[i] < b_scores[i])  # a lost, b won
    b10 = sum(1 for i in shared if a_scores[i] > b_scores[i])  # a won, b lost
    p = mcnamar_p(b01, b10)
    lo, hi, diff = paired_bootstrap_ci([a_scores[i] for i in shared], [b_scores[i] for i in shared])
    return (
        f"- {label}: resolve diff {diff:+.2%} "
        f"(bootstrap 95% CI {lo:+.2%} to {hi:+.2%}), "
        f"McNemar p={p:.4f}{'' if p < 0.05 else ' (n.s.)'} (n={len(shared)})"  # noqa: PLR2004
    )


def paired_significance(a: EvalRun, b: EvalRun) -> str:
    """McNemar + paired-bootstrap significance between two runs.

    Per-instance outcomes are the STRICT instance-resolve predicate
    (``is_instance_resolved``: error-free and ``F2P == 1.0`` — partial credit
    is not a binary outcome for McNemar/bootstrap), keyed by
    ``model:variant`` first, then instance ID. Blank instance IDs are never
    paired. Two pairings are reported:

    - **same-variant** — the same ``model:variant`` evaluated in both runs
      (e.g. prompt A/B runs of one variant on the seed-42 subset):
      repeat-run variance.
    - **cross-variant** — the same model under two DIFFERENT variants,
      paired on identical instances (e.g. ``baseline_14b`` vs
      ``higher_rank_14b`` in separate runs of the same tier): the
      variant-vs-variant comparison needed to call a fine-tune a win.

    Fairness preconditions: sampling params (temperature/top_p/max_tokens)
    and prompt wrapping must be identical across the two runs, and batch
    sizes should match — eager (probe/smoke <32 prompts) vs compiled vLLM
    engines can add small decode variance (see
    ``evaluation.inference._generate_patches_batch_body``).

    Returns a short markdown block.
    """
    a_vars: dict[str, dict[str, float]] = defaultdict(dict)
    b_vars: dict[str, dict[str, float]] = defaultdict(dict)
    for r in a.results:
        if r.instance_id:
            a_vars[f"{r.model_name}:{r.variant}"][r.instance_id] = (
                1.0 if is_instance_resolved(r) else 0.0
            )
    for r in b.results:
        if r.instance_id:
            b_vars[f"{r.model_name}:{r.variant}"][r.instance_id] = (
                1.0 if is_instance_resolved(r) else 0.0
            )

    lines: list[str] = []
    same_variants = sorted(set(a_vars) & set(b_vars))
    if same_variants:
        same_lines = [
            line
            for line in (
                _mcnemar_line(variant, a_vars[variant], b_vars[variant])
                for variant in same_variants
            )
            if line
        ]
        lines.extend(
            same_lines
            or ["_same-variant runs have no overlapping instances — same-variant pairing skipped_"]
        )
    else:
        lines.append(
            "_no variant evaluated in both runs — same-variant pairing skipped "
            "(compare runs that share a variant name, e.g. prompt A/B on one variant)_"
        )

    cross_pairs = sorted(
        (va, vb)
        for va in a_vars
        for vb in b_vars
        if va != vb and va.partition(":")[0] == vb.partition(":")[0]
    )
    if cross_pairs:
        cross_lines = [
            line
            for line in (
                _mcnemar_line(f"{va} vs {vb}", a_vars[va], b_vars[vb]) for va, vb in cross_pairs
            )
            if line
        ]
        lines.extend(
            cross_lines
            or ["_cross-variant pairs share no instances — cross-variant pairing skipped_"]
        )

    return f"paired significance ({a.run_id} vs {b.run_id}):\n" + "\n".join(lines)


def revalidate_proxy_champion(
    config: EvalConfig,
    run_ids: list[str],
    golden_path: Path | None = None,
    variant_adapter_map: dict[str, str] | None = None,
) -> tuple[str, F2PMetrics] | None:
    """Full re-validation flow: load runs, extract metrics, rank champions.

    The proxy champion is derived from P4 proxy scoring when *golden_path*
    and *variant_adapter_map* are both provided, falling back to
    ``baseline_14b`` otherwise. Real F2P ranking and gate filtering are
    always applied via :func:`revalidate_champion`.

    Args:
        config: Eval config (output dir, W&B coordinates, gate thresholds).
        run_ids: Run IDs to load.
        golden_path: Optional golden.jsonl for P4 proxy champion scoring.
        variant_adapter_map: Optional variant→adapter map for P4 scoring.

    Returns:
        ``(model_key, metrics)`` of the re-validated champion, or ``None``
        when no model clears the quality gates.
    """
    runs = load_all_eval_runs(run_ids, config)
    metrics = extract_model_metrics(runs)
    if not metrics:
        logger.warning("no metrics loaded for %s — nothing to revalidate", run_ids)
        return None
    if golden_path is not None and variant_adapter_map is not None:
        proxy_champion = proxy_champion_from_f2p_proxy(golden_path, variant_adapter_map)
    else:
        proxy_champion = config.proxy_champion_fallback
    return revalidate_champion(
        metrics, proxy_champion, config.min_f2p_threshold, config.min_p2p_threshold
    )


def _clear_champion_alias(api: Any, config: EvalConfig) -> None:
    """Remove the ``champion`` alias from any artifact currently holding it."""
    try:
        collection = api.artifact_collection("eval-champion")
        for member in collection.artifacts:
            if "champion" in member.aliases:
                member.aliases.remove("champion")
                member.save()
                logger.info("cleared 'champion' alias from %s", member.name)
    except Exception:  # noqa: BLE001 — W&B must never break the harness
        logger.warning("failed to clear previous W&B champion alias", exc_info=True)


def promote_champion_to_registry(champion_key: str, config: EvalConfig) -> str | None:
    """Link the champion's Phase 4 model artifact to W&B Registry.

    Creates/updates the ``eval-champion`` registry portfolio with the
    ``champion`` alias on the champion variant's model checkpoint artifact
    (``config.lora_artifact_pattern``).  Phase 6 consumes the champion from
    the W&B Registry ``champion`` alias.

    Best-effort: returns None (warn + skip) without wandb, without the
    artifact, or on any failure — never raises.

    Args:
        champion_key: ``"model_name:variant"`` key (as used in metrics).
        config: Eval config (wandb entity/project + artifact pattern).

    Returns:
        Human-readable summary, or None when skipped.
    """
    if not champion_key:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping champion registry promotion")
        return None
    variant = champion_key.partition(":")[2]
    artifact_name = config.lora_artifact_pattern.format(variant=variant)
    qualified = f"{config.wandb_entity}/{config.wandb_project}/{artifact_name}:latest"
    try:
        api = wandb.Api(timeout=30)
        artifact = api.artifact(qualified)
    except Exception:  # noqa: BLE001
        logger.warning("champion artifact %s not found in W&B — skipping promotion", qualified)
        return None
    try:
        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            job_type="eval-promote",
            name=f"promote-{champion_key}",
            reinit="finish_previous",
        )
        try:
            _clear_champion_alias(api, config)
            run.link_artifact(artifact, "eval-champion", aliases=["champion"])
        finally:
            run.finish()
        logger.info("W&B Registry: champion alias -> %s", artifact_name)
        return f"W&B Registry: champion alias -> {artifact_name}"  # noqa: TRY300
    except Exception:  # noqa: BLE001
        logger.warning("W&B champion promotion failed", exc_info=True)
        return None


# ── Loading helpers ────────────────────────────────────────────────────────


def _load_local_run(run_id: str, config: EvalConfig) -> EvalRun | None:
    """Load a run from ``{config.output_dir}/{run_id}.json`` if present."""
    path = config.output_dir / f"{run_id}.json"
    if not path.is_file():
        return None
    parsed = _parse_run_file(path)
    if not isinstance(parsed, EvalRun) and not parsed:
        logger.warning("run %s: no evaluation results in %s", run_id, path)
        return None
    return _as_run(run_id, config, parsed)


def _load_wandb_run(run_id: str, config: EvalConfig) -> EvalRun | None:
    """Load a run from the W&B artifact ``eval-results-{run_id}``."""
    import wandb  # lazy: never imported at module import time

    artifact_name = _ARTIFACT_NAME.format(run_id=run_id)
    try:
        api = wandb.Api(timeout=30)
        artifact = api.artifact(
            f"{config.wandb_entity}/{config.wandb_project}/{artifact_name}:latest"
        )
    except Exception as exc:  # no API key, missing artifact, auth failure, no network
        logger.warning("run %s: W&B artifact %s unavailable: %s", run_id, artifact_name, exc)
        return None
    download_dir = Path(artifact.download())
    results: list[EvalResult] = []
    for path in sorted(download_dir.iterdir()):
        parsed = _parse_run_file(path)
        if isinstance(parsed, EvalRun):
            return parsed
        if parsed:
            results.extend(parsed)
    if not results:
        logger.warning("run %s: W&B artifact %s contained no results", run_id, artifact_name)
        return None
    return _as_run(run_id, config, results)


def _parse_run_file(path: Path) -> EvalRun | list[EvalResult] | None:
    """Parse a run file: a single JSON document or JSONL of ``EvalResult``.

    The whole file is tried as one JSON document first — the format
    ``_persist_run`` writes (``EvalRun`` dump, indented multi-line; a single
    ``EvalResult`` dict is also accepted). If that fails (multiple top-level
    values, i.e. JSONL), fall back to parsing one ``EvalResult`` per line.

    Returns ``None`` when the file is unreadable, empty, or fails schema
    validation (never raises).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("run file %s unreadable: %s", path, exc)
        return None
    if not text.strip():
        return None
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None

    parsed: EvalRun | list[EvalResult] | None = None
    if isinstance(document, dict):
        try:
            if "results" in document:
                parsed = EvalRun.model_validate(document)
            else:
                parsed = [EvalResult.model_validate(document)]
        except ValidationError as exc:
            logger.warning("run file %s failed schema validation: %s", path, exc)
            return None
    if parsed is None:
        try:
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            if lines:
                if isinstance(lines[0], dict) and "results" in lines[0]:
                    parsed = EvalRun.model_validate(lines[0])
                else:
                    parsed = [EvalResult.model_validate(line) for line in lines]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("run file %s unparseable: %s", path, exc)
            return None
    return parsed


def _as_run(run_id: str, config: EvalConfig, parsed: EvalRun | list[EvalResult]) -> EvalRun:
    """Normalize parsed run content into a completed ``EvalRun``.

    A raw ``EvalRun`` dump is returned as-is; a list of ``EvalResult`` is
    grouped by (model, variant, prompt template) and aggregated via
    :func:`evaluation.metrics.aggregate_metrics`.
    """
    if isinstance(parsed, EvalRun):
        return parsed
    by_group: dict[tuple[str, str, str], list[EvalResult]] = defaultdict(list)
    for result in parsed:
        by_group[(result.model_name, result.variant, result.prompt_template)].append(result)
    timestamps = [result.timestamp for result in parsed]
    return EvalRun(
        run_id=run_id,
        started_at=min(timestamps),
        completed_at=max(timestamps),
        config=config,
        models_evaluated=sorted({f"{result.model_name}:{result.variant}" for result in parsed}),
        results=parsed,
        aggregate=[aggregate_metrics(results) for results in by_group.values()],
        status="completed",
    )
