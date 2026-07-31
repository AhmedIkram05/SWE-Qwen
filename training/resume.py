"""Hybrid checkpoint resume logic.

Supports three resolution strategies:
1. **Local path** — direct filesystem path to a checkpoint directory.
2. **W&B artifact ref** — ``entity/project/artifact-name:version`` downloads to local
   volume and returns the local path.
3. **``"latest"``** — queries W&B for the latest ``model_checkpoint`` artifact for the
   current run and downloads it.

Falls back through the chain: local → W&B artifact → raise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

# Expected number of path components in a full W&B run path (entity/project/run_id)
_WANDB_RUN_PATH_PARTS = 2


def resolve_checkpoint_path(  # noqa: PLR0911, PLR0912
    resume_spec: str,
    local_volume_path: str | Path | None = None,
    run_id: str | None = None,
) -> str | None:
    """Resolve a resume spec to a local checkpoint path.

    Args:
        resume_spec: One of:
            - A local filesystem path (str or Path).
            - A W&B artifact reference e.g. ``"entity/project/artifact:v1"``.
            - ``"latest"`` — auto-detect latest checkpoint for current run.
        local_volume_path: Base path for local checkpoints. Only used for
            ``"latest"`` resolution when checking for local checkpoints first.
        run_id: Optional W&B run ID for "latest" resolution. If provided,
            used instead of requiring an active wandb.run.

    Returns:
        Absolute path to a checkpoint directory, or ``None`` if not found.
    """
    if not resume_spec:
        return None

    # ── Strategy 1: Local path ────────────────────────────────────────────────
    local_path = Path(resume_spec)
    if local_path.exists():
        logger.info("Resolved checkpoint via local path: %s", local_path.resolve())
        return str(local_path.resolve())

    # If it looks like a W&B artifact ref (contains `/` and `:` or starts with `entity/`)
    is_wandb_ref = "/" in resume_spec and ":" in resume_spec or resume_spec.count("/") >= 1

    if not is_wandb_ref and resume_spec.lower() != "latest":
        # It's a local path that doesn't exist — try under local_volume_path
        vol_path = Path(local_volume_path) if local_volume_path else None
        if vol_path:
            alt_path = vol_path / resume_spec
            if alt_path.exists():
                logger.info("Resolved checkpoint under volume path: %s", alt_path)
                return str(alt_path.resolve())

        logger.warning(
            "Resume path does not exist: %s (checked absolute and under %s)",
            resume_spec,
            vol_path,
        )
        return None

    # ── Strategy 2: W&B artifact ref ──────────────────────────────────────────
    try:
        api = wandb.Api()
        artifact = api.artifact(resume_spec)
        dl_path = artifact.download()
        logger.info("Resolved checkpoint via W&B artifact: %s → %s", resume_spec, dl_path)
    except Exception as exc:
        logger.warning("Failed to resolve W&B artifact %s: %s", resume_spec, exc)
    else:
        return dl_path  # type: ignore[no-any-return]

    # ── Strategy 3: Latest from current run ────────────────────────────────────
    if resume_spec.lower() == "latest":
        try:
            api = wandb.Api()
            # Use provided run_id or fall back to active wandb.run
            if run_id:
                entity = wandb.run.entity if wandb.run else ""
                project = wandb.run.project if wandb.run else ""
                run_path = f"{entity}/{project}/{run_id}"
                # If run_id already contains entity/project, use it directly
                if run_id.count("/") == _WANDB_RUN_PATH_PARTS:
                    run_path = run_id
            elif wandb.run is not None:
                run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
            else:
                logger.warning(
                    "Cannot resolve 'latest' checkpoint: no active wandb.run and no run_id provided"
                )
                return None

            run = api.run(run_path)
            artifacts = run.logged_artifacts()

            # Filter to model_checkpoint type
            checkpoint_artifacts = [a for a in artifacts if a.type == "model_checkpoint"]

            if checkpoint_artifacts:
                # Latest = most recently created
                latest = max(checkpoint_artifacts, key=lambda a: a.created_at)
                dl_path = latest.download()
                logger.info("Resolved latest checkpoint from W&B: %s", dl_path)
                return dl_path  # type: ignore[no-any-return]

            logger.warning("No model_checkpoint artifacts found for run %s", run_path)
        except Exception as exc:
            logger.warning("Failed to resolve latest checkpoint: %s", exc)

    logger.error(
        "Could not resolve checkpoint from spec: %s (tried local, W&B artifact, latest)",
        resume_spec,
    )
    return None
