"""Champion/Challenger decision audit trail (Phase 9, spec §4.4, decision 9).

Pure record builders plus a thin, failure-soft W&B uploader: every decision
lands as an immutable ``promotion-decision-{id}`` artifact (JSON + markdown)
with the six registered ``promote/*`` scalars logged as one literal
``log_metrics`` dict.  The telemetry AST contract test resolves exact literal
keys (spec 4.11), so templated or helper-built payloads are forbidden here.

``wandb`` is imported lazily inside the upload functions (repo convention:
offline tests pass without cloud); any failure degrades to ``None`` and never
raises, mirroring ``comparison.promote_champion_to_registry``.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from observability.metrics import log_metrics
from promotion.rules import (
    MIN_F2P_FLOOR,
    MIN_P2P_FLOOR,
    OUTCOME_PROMOTE,
    OUTCOME_REJECT,
    PROMOTE_MAX_P2P_REGRESSION,
    PROMOTE_MIN_F2P_GAIN,
)

__all__ = [
    "build_decision_record",
    "log_decision_metrics",
    "note_gating_off",
    "render_markdown",
    "write_decision_record",
]


def build_decision_record(  # noqa: PLR0913 — spec §4.4 fixed field set, keyword-only
    *,
    decision_id: str,
    pipeline_version: str,
    candidate_run_id: str,
    candidate_variant: str,
    candidate_model_ref: str,
    champion_run_id: str,
    champion_variant: str,
    champion_model_ref: str,
    candidate_f2p: float,
    champion_f2p: float,
    candidate_p2p: float,
    champion_p2p: float,
    f2p_gain: float,
    p2p_delta: float,
    ci_lower: float,
    ci_high: float,
    mcnemar_p: float,
    outcome: str,
    reasons: list[str],
    git_sha: str | None = None,
) -> dict:
    """Build the spec §4.4 decision record with a frozen field set.

    ``incumbent`` is the champion (the incumbent baseline of the paired eval).
    Threshold values come from ``promotion.rules`` constants — never
    re-hardcoded — so threshold tuning stays auditable in the record.
    ``deployed`` starts ``False``; the deploy job flips it (and the workflow
    re-uploads the artifact metadata).
    """
    return {
        "decision_id": decision_id,
        "pipeline_version": pipeline_version,
        "candidate": {
            "run_id": candidate_run_id,
            "variant": candidate_variant,
            "model_ref": candidate_model_ref,
        },
        "incumbent": {
            "run_id": champion_run_id,
            "variant": champion_variant,
            "model_ref": champion_model_ref,
        },
        "metrics": {
            "candidate": {"f2p_rate": candidate_f2p, "p2p_rate": candidate_p2p},
            "incumbent": {"f2p_rate": champion_f2p, "p2p_rate": champion_p2p},
            "f2p_gain": f2p_gain,
            "p2p_delta": p2p_delta,
            "ci_lower": ci_lower,
            "ci_high": ci_high,
            "mcnemar_p": mcnemar_p,
        },
        "thresholds": {
            "min_f2p_gain": PROMOTE_MIN_F2P_GAIN,
            "max_p2p_regression": PROMOTE_MAX_P2P_REGRESSION,
            "floors": {"f2p": MIN_F2P_FLOOR, "p2p": MIN_P2P_FLOOR},
        },
        "outcome": outcome,
        "reasons": reasons,
        "deployed": False,
        "timestamps": {"created_utc": datetime.now(UTC).isoformat()},
        "git_sha": git_sha,
    }


def render_markdown(record: dict) -> str:
    """Human-readable markdown summary of a decision record (artifact + step summary)."""
    metrics = record["metrics"]
    candidate = record["candidate"]
    incumbent = record["incumbent"]
    thresholds = record["thresholds"]
    floors = thresholds["floors"]
    reasons = ", ".join(record["reasons"]) if record["reasons"] else "none"
    return "\n".join(
        [
            f"# Promotion decision: {record['decision_id']}",
            "",
            f"- **Outcome**: `{record['outcome']}`",
            f"- **Pipeline version**: {record['pipeline_version']}",
            (
                f"- **Candidate**: {candidate['variant']} ({candidate['model_ref']}, "
                f"run {candidate['run_id']})"
            ),
            (
                f"- **Incumbent (champion)**: {incumbent['variant']} "
                f"({incumbent['model_ref']}, run {incumbent['run_id']})"
            ),
            "",
            "## Metrics",
            f"- F2P rate: candidate {metrics['candidate']['f2p_rate']:.3f} vs "
            f"incumbent {metrics['incumbent']['f2p_rate']:.3f}",
            f"- P2P rate: candidate {metrics['candidate']['p2p_rate']:.3f} vs "
            f"incumbent {metrics['incumbent']['p2p_rate']:.3f}",
            f"- **F2P gain**: {metrics['f2p_gain']:+.3f}",
            f"- **P2P delta**: {metrics['p2p_delta']:+.3f}",
            f"- **CI (lower, upper)**: {metrics['ci_lower']:.3f}, {metrics['ci_high']:.3f}",
            f"- **McNemar p**: {metrics['mcnemar_p']:.4f}",
            "",
            "## Thresholds",
            f"- Min F2P gain: {thresholds['min_f2p_gain']:.3f}",
            f"- Max P2P regression: {thresholds['max_p2p_regression']:.3f}",
            f"- Floors: F2P {floors['f2p']:.2f}, P2P {floors['p2p']:.2f}",
            "",
            "## Reasons",
            reasons,
        ]
    )


def write_decision_record(record: dict, *, entity: str, project: str) -> str | None:
    """Upload one decision record to W&B as an immutable ``decision`` artifact.

    Writes ``decision.json`` + ``decision.md`` to a temp dir, attaches both to
    ``wandb.Artifact("promotion-decision-{id}", "decision")``, logs the six
    ``promote/*`` scalars on the run, and returns the artifact name.

    Failure-soft (spec decision 9): returns ``None`` on ImportError or any
    exception — never raises, matching ``promote_champion_to_registry``.
    """
    artifact_name = f"promotion-decision-{record['decision_id']}"
    try:
        import wandb
    except ImportError:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "decision.json"
            md_path = Path(tmpdir) / "decision.md"
            json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            md_path.write_text(render_markdown(record), encoding="utf-8")
            run = wandb.init(
                project=project,
                entity=entity,
                job_type="promotion-decision",
                name=f"decision-{record['decision_id']}",
                reinit="finish_previous",
            )
            try:
                artifact = wandb.Artifact(name=artifact_name, type="decision")
                artifact.add_file(str(json_path), name="decision.json")
                artifact.add_file(str(md_path), name="decision.md")
                run.log_artifact(artifact)
                log_decision_metrics(record)
            finally:
                run.finish()
        return artifact_name  # noqa: TRY300 — matches promote_champion_to_registry
    except Exception:  # noqa: BLE001 — W&B must never break the pipeline
        return None


def log_decision_metrics(record: dict) -> None:
    """Log the six registered ``promote/*`` scalars for one decision.

    Literal dict on purpose (spec decision 9): the telemetry AST walker
    resolves these exact keys statically, and a templated/helper-built dict
    would fail ``test_telemetry_contract``.
    """
    log_metrics(
        {
            "promote/outcome": 1.0 if record["outcome"] == OUTCOME_PROMOTE else 0.0,
            "promote/f2p_gain": float(record["metrics"]["f2p_gain"]),
            "promote/p2p_delta": float(record["metrics"]["p2p_delta"]),
            "promote/ci_lower": float(record["metrics"]["ci_lower"]),
            "promote/mcnemar_p": float(record["metrics"]["mcnemar_p"]),
            "promote/deploy_status": 1.0 if record.get("deployed") else 0.0,
        }
    )


def note_gating_off(
    candidate_variant: str, reason: str, *, entity: str, project: str
) -> str | None:
    """Record the ``RUN_MODAL_EVAL=false`` skip path as a reject decision.

    No paired eval ran, so every ``promote/*`` scalar is 0.0 and the outcome
    is ``reject``; the reason and candidate variant are stored in the run
    config (they are not registry keys, so they never go through
    ``wandb.log``).  Lazy wandb, failure-soft: returns the run name on
    success, ``None`` on ImportError or any exception.
    """
    try:
        import wandb
    except ImportError:
        return None
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="promotion-decision",
            name=f"gating-off-{candidate_variant}",
            reinit="finish_previous",
        )
        try:
            run.config.update({"candidate_variant": candidate_variant, "gating_off_reason": reason})
            log_decision_metrics(
                {
                    "outcome": OUTCOME_REJECT,
                    "metrics": {
                        "f2p_gain": 0.0,
                        "p2p_delta": 0.0,
                        "ci_lower": 0.0,
                        "mcnemar_p": 0.0,
                    },
                    "deployed": False,
                }
            )
        finally:
            run.finish()
        return f"gating-off-{candidate_variant}"  # noqa: TRY300 — never-raise contract
    except Exception:  # noqa: BLE001 — W&B must never break the pipeline
        return None
