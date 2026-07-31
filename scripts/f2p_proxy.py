"""Proxy F2P (Fail-to-Pass) scorer for Phase 4 champion selection.

True F2P evaluation runs a generated patch against the real test suite
(Phase 5).  This proxy uses W&B training loss as a heuristic — lower loss
indicates better learning.  Not a substitute for the real harness.

The proxy DOES NOT need a GPU or a model adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _wandb_project_entity() -> str:
    """Return 'entity/project' for W&B API calls."""
    import wandb

    api = wandb.Api(timeout=30)
    entity = api.default_entity
    if not entity:
        raise RuntimeError("W&B entity not found. Run `wandb login`.")
    return f"{entity}/swe-qwen"


def compute_proxy_f2p_scores(
    golden_path: Path,
    variant_adapter_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Compute mean proxy F2P for each variant using W&B training loss.

    In Phase 4 we don't have model-generated patches, so the proxy scores
    variants by their W&B training loss (lower loss → higher score).
    The golden_path is read to confirm it's valid but not used for scoring.

    Returns:
        ``{variant: {"mean_f2p": float, "count": int}}``
        Higher mean_f2p = better (inverted from loss).
    """
    # Confirm golden set is valid
    with golden_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    import wandb

    api = wandb.Api(timeout=30)
    project = _wandb_project_entity()
    results: dict[str, dict[str, Any]] = {}

    # First pass: collect all train_losses
    losses: dict[str, float] = {}
    for variant in variant_adapter_map:
        runs = api.runs(project, {"config.variant": variant})
        if not runs:
            results[variant] = {
                "mean_f2p": 0.0,
                "count": len(records),
                "warning": "no W&B run found",
            }
            continue

        finished = [r for r in runs if r.state == "finished"]
        if not finished:
            results[variant] = {
                "mean_f2p": 0.0,
                "count": len(records),
                "warning": "no finished W&B run found",
            }
            continue

        run = sorted(finished, key=lambda r: r.created_at, reverse=True)[0]
        train_loss = run.summary.get("train_loss", None)
        if train_loss is None:
            results[variant] = {
                "mean_f2p": 0.0,
                "count": len(records),
                "warning": "no train_loss in summary",
            }
            continue

        losses[variant] = train_loss

    # Second pass: normalize relative to min/max loss (lower loss → higher score)
    if not losses:
        return results

    min_loss = min(losses.values())
    max_loss = max(losses.values())
    loss_range = max_loss - min_loss

    for variant, train_loss in losses.items():
        score = 1.0 - (train_loss - min_loss) / loss_range if loss_range > 0 else 1.0

        results[variant] = {
            "mean_f2p": round(score, 4),
            "count": len(records),
            "train_loss": round(train_loss, 4),
        }

    return results


def select_champion(
    scores: dict[str, dict[str, Any]],
) -> str:
    """Return the variant name with the highest mean F2P."""
    best = max(scores, key=lambda v: scores[v].get("mean_f2p", 0.0))
    return best
