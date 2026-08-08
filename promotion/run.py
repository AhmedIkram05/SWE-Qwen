"""Champion/challenger pipeline entrypoints (Phase 9, spec §4.6, decision 11):
the ``decide`` job and the ``deploy`` job.

Two mutually exclusive entrypoints behind one module:

- ``python -m promotion.run --candidate-variant <v> [--no-eval]`` — the
  ``decide`` job: read champion.json (GCS fallback), validate the challenger
  (trained variant + W&B ``challenger`` tag), launch a paired dev-tier eval of
  incumbent vs candidate as two parallel ``uv run eval run`` subprocesses,
  poll until both runs land, then ``promotion.gate.evaluate_pair`` →
  ``promotion.rules.decide`` and write the audit decision record.  On PROMOTE
  the machine lines for the workflow's ``$GITHUB_OUTPUT`` are appended
  (``promote``/``candidate``/``champion``/``decision_id``); champion.json is
  NOT written here (single writer: the deploy job).  Rejections exit 0 —
  rejection is a successful outcome.  ``--no-eval`` is the
  ``RUN_MODAL_EVAL=false`` kill switch: no eval, no W&B artifact check, zero
  Modal spend.
- ``python -m promotion.run --deploy --variant <v> [--decision-id <id>]`` —
  the ``deploy`` job (``environment: production``): pin the variant, deploy to
  Modal, probe it, and only after the probe is green write champion.json and
  sync the W&B ``champion`` alias.  Any failure (deploy non-zero, probe
  failure, alias failure) rolls the displaced incumbent back through the
  same deploy path and exits 1.

Exec/network lines (subprocess exec, GCS fallback, W&B artifact reads) are
``# pragma: no cover`` per plan §4.11; everything else is unit-tested.
All ``promote/*`` scalars are emitted by ``audit.log_decision_metrics`` — this
module never calls ``log_metrics``/``wandb.log`` itself (telemetry contract).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evaluation.comparison import load_all_eval_runs
from evaluation.config import EvalConfig
from evaluation.harness import make_run_id
from evaluation.schema import EvalRun
from inference.config import ServeConfig
from promotion import audit
from promotion import deploy as deploy_mod
from promotion.audit import render_markdown
from promotion.deploy import ConfigGapError, ProbeError
from promotion.gate import PairEval, evaluate_pair
from promotion.registry import ChampionRecord, read_champion, write_champion
from promotion.rules import OUTCOME_PROMOTE, OUTCOME_REJECT, REASON_FATAL_FLAW, decide

DEFAULT_CHAMPION_PATH = Path("data/eval_results/champion.json")
# Spec §4.6: poll every 30 s, overall ceiling ~180 min.
POLL_INTERVAL_S = 30.0
EVAL_TIMEOUT_S = 180 * 60.0
DEFAULT_PIPELINE_VERSION = "9"
GATING_OFF_REASON = "RUN_MODAL_EVAL=false"
GCS_BUCKET = "swe-qwen-datasets"
GCS_CHAMPION_BLOB = "ci/champion.json"

# Abort reasons carried in the decision record (reasons is a free-form list;
# the rules-level reasons live in promotion.rules).
REASON_CONFIG_GAP = "config-gap"
REASON_NO_CHALLENGER = "no-challenger"
REASON_EVAL_TIMEOUT = "eval-timeout"
REASON_EVAL_FAILED = "eval-failed"


def _decision_id() -> str:
    """Timestamped decision id (spec §4.6 example): ``promote-YYYYMMDD-HHMMSS``."""
    return f"promote-{datetime.now():%Y%m%d-%H%M%S}"


def _unique_run_ids() -> list[str]:
    """Two distinct eval run ids even when ``make_run_id`` (second granularity) collides.

    ``evaluation.harness.make_run_id`` has one-second resolution, and the
    paired launch calls it twice back-to-back; a colliding second id gets a
    ``-2`` suffix so the two subprocess results can never clobber each other.
    """
    ids: list[str] = []
    for _ in range(2):
        run_id = make_run_id()
        if run_id in ids:
            run_id = f"{run_id}-{len(ids) + 1}"
        ids.append(run_id)
    return ids


def _load_champion(path: str | Path) -> ChampionRecord | None:
    """Read the champion record, falling back to the GCS copy on a local miss.

    Returns ``None`` on any failure (missing locally AND on GCS, or malformed
    JSON): the decide job then exits 1 with a clear stderr message.  The GCS
    fallback is best-effort — it exists for local/dev runs (spec §4.6); the
    workflow pre-downloads champion.json for CI.
    """
    try:
        return read_champion(path)
    except FileNotFoundError:
        pass
    except ValueError as exc:
        print(f"invalid champion record at {path}: {exc}", file=sys.stderr)
        return None
    if _download_champion_from_gcs(Path(path)):
        try:
            return read_champion(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"invalid champion record downloaded to {path}: {exc}", file=sys.stderr)
    return None


def _download_champion_from_gcs(destination: Path) -> bool:
    """Download ``gs://swe-qwen-datasets/ci/champion.json`` to *destination*.

    Anonymous public read (reads are public, writes need WIF auth — spec §4.3);
    any failure — SDK missing, no network, blob absent — returns ``False``.
    """
    try:
        from google.cloud import storage  # type: ignore[attr-defined]

        client = storage.Client.create_anonymous_client()  # pragma: no cover — GCS network
        bucket = client.bucket(GCS_BUCKET)  # pragma: no cover — GCS network
        blob = bucket.blob(GCS_CHAMPION_BLOB)  # pragma: no cover — GCS network
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))  # pragma: no cover — GCS network
    except Exception:  # noqa: BLE001 — GCS is a best-effort fallback
        return False
    return True


def _has_challenger_alias(variant: str, config: ServeConfig) -> bool:
    """True when the W&B ``model-qwen3-14b-{variant}:latest`` artifact carries
    the ``challenger`` alias.

    Lazy ``wandb`` (repo convention): wandb unavailable, artifact absent, or
    artifact untagged all read as "no challenger" — the decide job aborts
    rather than guessing (spec §4.6 step 2b).
    """
    qualified = f"{config.lora_artifact_pattern.format(variant=variant)}:latest"
    try:
        import wandb

        artifact = wandb.Api(timeout=30).artifact(qualified)  # pragma: no cover — W&B network
    except Exception:  # noqa: BLE001 — wandb unavailable -> no challenger
        return False
    return "challenger" in artifact.aliases  # pragma: no cover — W&B network


def _abort_record(  # noqa: PLR0913 — spec §4.4 record fields, keyword-only (audit.py parity)
    *,
    decision_id: str,
    pipeline_version: str,
    candidate_variant: str,
    candidate_model_ref: str,
    champion: ChampionRecord | None,
    reason: str,
) -> dict:
    """Decision record for an abort path (config-gap / no-challenger / eval failure).

    No paired eval ran, so every metric is 0.0 and the run ids are empty; the
    incumbent fields come from champion.json when it was loadable (config-gap
    can abort before the registry exists — first cycle).
    """
    return audit.build_decision_record(
        decision_id=decision_id,
        pipeline_version=pipeline_version,
        candidate_run_id="",
        candidate_variant=candidate_variant,
        candidate_model_ref=candidate_model_ref,
        champion_run_id="",
        champion_variant=champion.variant if champion else "",
        champion_model_ref=champion.model_ref if champion else "",
        candidate_f2p=0.0,
        champion_f2p=0.0,
        candidate_p2p=0.0,
        champion_p2p=0.0,
        f2p_gain=0.0,
        p2p_delta=0.0,
        ci_lower=0.0,
        ci_high=0.0,
        mcnemar_p=0.0,
        outcome=OUTCOME_REJECT,
        reasons=[reason],
    )


def _build_decided_record(  # noqa: PLR0913 — spec §4.4 record fields, keyword-only (audit.py parity)
    *,
    decision_id: str,
    pipeline_version: str,
    champion: ChampionRecord,
    champion_run_id: str,
    candidate_run_id: str,
    candidate_variant: str,
    candidate_model_ref: str,
    pair: PairEval,
    outcome: str,
    reasons: list[str],
) -> dict:
    """Decision record for a completed paired eval (spec §4.4 frozen field set)."""
    champion_metrics, candidate_metrics = pair.champion_metrics, pair.candidate_metrics
    return audit.build_decision_record(
        decision_id=decision_id,
        pipeline_version=pipeline_version,
        candidate_run_id=candidate_run_id,
        candidate_variant=candidate_variant,
        candidate_model_ref=candidate_model_ref,
        champion_run_id=champion_run_id,
        champion_variant=champion.variant,
        champion_model_ref=champion.model_ref,
        candidate_f2p=candidate_metrics.f2p_rate if candidate_metrics else 0.0,
        champion_f2p=champion_metrics.f2p_rate if champion_metrics else 0.0,
        candidate_p2p=candidate_metrics.p2p_rate if candidate_metrics else 0.0,
        champion_p2p=champion_metrics.p2p_rate if champion_metrics else 0.0,
        f2p_gain=pair.f2p_gain,
        p2p_delta=pair.p2p_delta,
        ci_lower=pair.ci_lower,
        ci_high=pair.ci_high,
        mcnemar_p=pair.mcnemar_p,
        outcome=outcome,
        reasons=reasons,
    )


def _finalize(record: dict, config: EvalConfig) -> None:
    """Persist the decision record, log the ``promote/*`` scalars, print the summary.

    The ``promote=true|false <reason>`` line goes to stdout ALWAYS (the E2E
    tests assert it); scalar emission happens only inside
    ``audit.log_decision_metrics`` — never here.  Spec §4.10 L179 (approver
    clarity): the full decision markdown is appended to ``$GITHUB_STEP_SUMMARY``
    in the decide job, and decision.md/decision.json are persisted to
    ``data/promotion_decisions/`` for the workflow's ``actions/upload-artifact``
    step (``promotion-decision-<id>``, retention 7).  The files are written
    unconditionally — the Actions artifact is the durable local copy even when
    the W&B upload degrades.  ``--no-eval`` never reaches here.
    """
    audit.write_decision_record(record, entity=config.wandb_entity, project=config.wandb_project)
    audit.log_decision_metrics(record)
    decision_md = render_markdown(record)
    _write_step_summary(decision_md)
    decision_dir = Path("data/promotion_decisions")
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "decision.md").write_text(decision_md, encoding="utf-8")
    (decision_dir / "decision.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    reasons = " ".join(record["reasons"]) if record["reasons"] else ""
    print(f"promote={record['outcome'] == OUTCOME_PROMOTE}" + (f" {reasons}" if reasons else ""))


def _launch_evals(
    pairs: list[tuple[str, str]], mode: str, base_model: str
) -> list[subprocess.Popen[str]]:
    """Spawn one ``eval run`` subprocess per ``(variant, run_id)`` pair, in parallel.

    Each subprocess evaluates a single ``model:variant`` on the deterministic
    seed-42 subset of the ``--mode`` tier (spec §4.6 step 3); ``--resume``
    pins the run id so the poll loop knows what to load.
    """
    procs: list[subprocess.Popen[str]] = []
    for variant, run_id in pairs:
        cmd = [
            "uv",
            "run",
            "eval",
            "run",
            "--mode",
            mode,
            "--models",
            f"{base_model}:{variant}",
            "--resume",
            run_id,
        ]
        procs.append(  # pragma: no cover — real subprocess exec (plan §4.11)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        )
    return procs


def _wait_for_pair(
    champion_run_id: str, candidate_run_id: str, config: EvalConfig
) -> tuple[tuple[EvalRun, EvalRun] | None, str]:
    """Poll until both paired runs are terminal.

    Returns ``((champion_run, candidate_run), "ok")`` once both report status
    ``completed``, ``(None, "eval-failed")`` as soon as either reports
    ``failed``/``partial``, or ``(None, "eval-timeout")`` at the deadline.
    Absent or still-``running`` runs keep the poll going (spec §4.6 step 3).
    """
    deadline = time.monotonic() + EVAL_TIMEOUT_S
    while time.monotonic() < deadline:
        runs = load_all_eval_runs([champion_run_id, candidate_run_id], config)
        by_id = {run.run_id: run for run in runs}
        champion_run = by_id.get(champion_run_id)
        candidate_run = by_id.get(candidate_run_id)
        if champion_run is not None and candidate_run is not None:
            statuses = (champion_run.status, candidate_run.status)
            if statuses == ("completed", "completed"):
                return (champion_run, candidate_run), "ok"
            if "failed" in statuses or "partial" in statuses:
                return None, REASON_EVAL_FAILED
        time.sleep(POLL_INTERVAL_S)
    return None, REASON_EVAL_TIMEOUT


def _reap(procs: list[subprocess.Popen[str]]) -> None:
    """Best-effort cleanup of the eval subprocesses after the poll."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
        # the eval may still be finishing its final W&B upload
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)


def _write_github_output(lines: list[str]) -> None:
    """Append machine ``key=value`` lines to ``$GITHUB_OUTPUT``.

    Job outputs are read from this file, not stdout (spec §4.6 step 6); when
    the env var is unset (local runs) this is a silent no-op — the workflow is
    the only consumer.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_step_summary(text: str) -> bool:
    """Append raw markdown to ``$GITHUB_STEP_SUMMARY``; no-op when the env is unset.

    Same env-file append pattern as ``_write_github_output``. The decide job
    surfaces the full decision record to the human approver here (spec §4.10
    L179 — terraform-plan parity); returns whether anything was written.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return False
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")
    return True


def _decide_main(args: argparse.Namespace) -> int:
    """The decide job (spec §4.6): validate challenger, paired eval, rules, record."""
    serve = ServeConfig()
    config = EvalConfig()
    candidate = args.candidate_variant
    assert candidate is not None  # enforced by _parse_args
    base_model = serve.base_model

    # Config gap (spec decision 6: v1 promotes only among trained variants)
    # aborts before the champion read: the variant error is the actionable one
    # even when the registry is empty (first cycle).
    if candidate not in serve.variants:
        champion = _load_champion(args.champion_path)  # best-effort for the record
        _finalize(
            _abort_record(
                decision_id=_decision_id(),
                pipeline_version=args.pipeline_version,
                candidate_variant=candidate,
                candidate_model_ref=f"{base_model}:{candidate}",
                champion=champion,
                reason=REASON_CONFIG_GAP,
            ),
            config,
        )
        return 1

    # RUN_MODAL_EVAL=false kill switch (spec §4.6 step 8, decision 11): no
    # eval, no W&B artifact check, no GITHUB_OUTPUT — zero Modal spend, works
    # even when the bucket is empty (first cycle).
    if args.no_eval:
        audit.note_gating_off(
            candidate, GATING_OFF_REASON, entity=config.wandb_entity, project=config.wandb_project
        )
        print(f"promote=false gating-off ({GATING_OFF_REASON}) — no eval launched")
        return 0

    champion = _load_champion(args.champion_path)
    if champion is None:
        print(
            f"promote aborted: no champion record at {args.champion_path} (local or GCS)",
            file=sys.stderr,
        )
        return 1

    if not _has_challenger_alias(candidate, serve):
        _finalize(
            _abort_record(
                decision_id=_decision_id(),
                pipeline_version=args.pipeline_version,
                candidate_variant=candidate,
                candidate_model_ref=f"{base_model}:{candidate}",
                champion=champion,
                reason=REASON_NO_CHALLENGER,
            ),
            config,
        )
        return 1

    champion_run_id, candidate_run_id = _unique_run_ids()
    procs = _launch_evals(
        [(champion.variant, champion_run_id), (candidate, candidate_run_id)],
        args.mode,
        base_model,
    )
    try:
        pair_runs, wait_reason = _wait_for_pair(champion_run_id, candidate_run_id, config)
    finally:
        _reap(procs)
    if pair_runs is None:
        _finalize(
            _abort_record(
                decision_id=_decision_id(),
                pipeline_version=args.pipeline_version,
                candidate_variant=candidate,
                candidate_model_ref=f"{base_model}:{candidate}",
                champion=champion,
                reason=wait_reason,
            ),
            config,
        )
        return 1
    champion_run, candidate_run = pair_runs

    pair = evaluate_pair(champion_run, candidate_run, config)
    if pair.champion_metrics is None or pair.candidate_metrics is None:
        # Spec §4.6 step 4: missing aggregate metrics are a fatal-flaw reject —
        # never fabricate a gain from an unpaired/empty run.
        outcome, reasons = OUTCOME_REJECT, [REASON_FATAL_FLAW]
    else:
        outcome, reasons = decide(
            pair.champion_metrics.f2p_rate,
            pair.candidate_metrics.f2p_rate,
            pair.champion_metrics.p2p_rate,
            pair.candidate_metrics.p2p_rate,
            pair.ci_lower,
        )
    record = _build_decided_record(
        decision_id=_decision_id(),
        pipeline_version=args.pipeline_version,
        champion=champion,
        champion_run_id=champion_run_id,
        candidate_run_id=candidate_run_id,
        candidate_variant=candidate,
        candidate_model_ref=f"{base_model}:{candidate}",
        pair=pair,
        outcome=outcome,
        reasons=reasons,
    )
    _finalize(record, config)
    if outcome == OUTCOME_PROMOTE:
        # Spec §4.10 approver summary: the deploy job shows the F2P/P2P
        # before-after and passes the candidate rates into champion.json via
        # --f2p-rate/--p2p-rate, so the exact rates decide() saw must also
        # reach $GITHUB_OUTPUT.  Promote implies both metrics are present
        # (decide() is only reached when they are); the fallbacks mirror the
        # abort-record zeros for the impossible None case.
        champion_metrics, candidate_metrics = pair.champion_metrics, pair.candidate_metrics
        champion_f2p = champion_metrics.f2p_rate if champion_metrics else champion.f2p_rate
        champion_p2p = champion_metrics.p2p_rate if champion_metrics else champion.p2p_rate
        candidate_f2p = candidate_metrics.f2p_rate if candidate_metrics else 0.0
        candidate_p2p = candidate_metrics.p2p_rate if candidate_metrics else 0.0
        _write_github_output(
            [
                "promote=true",
                f"candidate={candidate}",
                # champion output = the PROMOTED candidate, not the incumbent
                # record's variant: the deploy job runs `--variant ${{ needs
                # .decide.outputs.champion }}` and the new ChampionRecord must
                # reference the gate winner (finding: self-referential record).
                f"champion={candidate}",
                f"decision_id={record['decision_id']}",
                f"champion_f2p={round(champion_f2p, 4)}",
                f"candidate_f2p={round(candidate_f2p, 4)}",
                f"champion_p2p={round(champion_p2p, 4)}",
                f"candidate_p2p={round(candidate_p2p, 4)}",
            ]
        )
    return 0


def _append_deploy_status(decision_id: str | None, status: str) -> None:
    """Best-effort ``deploy_status``/timestamp append to the decision artifact metadata.

    The workflow re-reads this metadata to render the deploy outcome (spec
    §4.10 step 3); a failure here never fails the deploy — ``artifact.save()``
    is required for metadata edits to persist on the W&B API.
    """
    if not decision_id:
        return
    try:
        import wandb

        artifact = wandb.Api(timeout=30).artifact(  # pragma: no cover — W&B network
            f"promotion-decision-{decision_id}"
        )
        artifact.metadata.update(  # pragma: no cover — W&B network
            {"deploy_status": status, "deployed_at": datetime.now(UTC).isoformat()}
        )
        artifact.save()  # pragma: no cover — W&B network
    except Exception:  # noqa: BLE001 — metadata append is best-effort, never raises
        return


def _rollback_and_report(old: ChampionRecord, decision_id: str | None, champion_path: Path) -> None:
    """Re-promote the displaced incumbent *old* itself (spec decision 7).

    Rollback target is the champion the failed deploy displaced — ``old``,
    never ``old.previous`` (which is the pre-incumbent chain).  On rollback
    success champion.json is rewritten back to ``old``: the alias-sync
    failure path has already overwritten it with the failed candidate, the
    probe-failure path rewrite is a harmless no-op.  Never raises: the
    deploy failure must surface regardless of rollback state; the deploy
    job exits 1 either way.
    """
    try:
        outcome = deploy_mod.rollback(old, ServeConfig(), env=dict(os.environ))
        print(f"rollback to {old.variant} ({outcome['outcome']})")
        write_champion(champion_path, old)
    except Exception as exc:  # noqa: BLE001 — rollback is best-effort, never raises
        print(f"rollback failed: {exc}", file=sys.stderr)
    _append_deploy_status(decision_id, "rollback")


def _deploy_main(args: argparse.Namespace) -> int:
    """The deploy job (spec §4.10): pin, deploy, probe, record, rollback on failure."""
    serve = ServeConfig()
    variant = args.variant
    assert variant is not None  # enforced by _parse_args
    try:
        deploy_mod.assert_variant_known(variant, serve)
    except ConfigGapError as exc:
        print(f"deploy aborted: {exc}", file=sys.stderr)
        return 1
    try:
        old = read_champion(args.champion_path)
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"deploy aborted: cannot read champion record at {args.champion_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    champion_key = f"{serve.base_model}:{variant}"
    result = deploy_mod.deploy(champion_key, serve, env=dict(os.environ))
    failed = result.returncode != 0
    if not failed:
        # Probe credentials are provided by the workflow runner (spec §4.10).
        try:
            deploy_mod.health_check(
                os.environ.get("MODAL_WEB_URL", ""), os.environ.get("MODAL_SERVE_TOKEN", "")
            )
        except ProbeError as exc:
            print(f"deploy probe failed: {exc}", file=sys.stderr)
            failed = True
    if not failed:
        if args.f2p_rate is None or args.p2p_rate is None:
            print(
                "warning: --f2p-rate/--p2p-rate not provided; champion.json carries forward "
                "incumbent rates (pass the decision record's candidate rates)",
                file=sys.stderr,
            )
        try:
            # Single writer of champion.json, and only after the probe is
            # green (spec §4.10 step 2: a variant is recorded as champion only
            # if it is deployed and green).
            write_champion(
                args.champion_path,
                ChampionRecord(
                    variant=variant,
                    model_ref=champion_key,
                    f2p_rate=old.f2p_rate if args.f2p_rate is None else args.f2p_rate,
                    p2p_rate=old.p2p_rate if args.p2p_rate is None else args.p2p_rate,
                    dataset_run_id=old.dataset_run_id,
                    tier="dev",
                    seed=old.seed,
                    promoted_at=datetime.now(UTC).isoformat(),
                    previous=old,
                ),
            )
            deploy_mod.sync_alias_or_abort(champion_key, EvalConfig())
        except RuntimeError as exc:
            print(f"deploy alias sync failed: {exc}", file=sys.stderr)
            failed = True
    if failed:
        _rollback_and_report(old, args.decision_id, args.champion_path)
        return 1
    _append_deploy_status(args.decision_id, "success")
    print(f"deployed champion={variant} probe=ok alias=ok champion.json written")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Argparse surface; import-safe (configs are instantiated in the callers)."""
    parser = argparse.ArgumentParser(
        prog="promotion.run",
        description="Champion/challenger decide and deploy entrypoints (Phase 9, spec §4.6).",
    )
    parser.add_argument(
        "--candidate-variant",
        type=str,
        default=None,
        help=("challenger variant to evaluate against the champion (decide path)"),
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=("RUN_MODAL_EVAL=false kill switch: record the gating-off note and exit 0"),
    )
    parser.add_argument(
        "--deploy", action="store_true", help="run the deploy job instead of decide"
    )
    parser.add_argument("--variant", type=str, default=None, help="variant to deploy (deploy path)")
    parser.add_argument(
        "--decision-id",
        type=str,
        default=None,
        help=("decision record id to append deploy_status to (deploy path)"),
    )
    parser.add_argument(
        "--champion-path",
        type=Path,
        default=DEFAULT_CHAMPION_PATH,
        help=(f"champion.json location (default: {DEFAULT_CHAMPION_PATH})"),
    )
    parser.add_argument("--mode", type=str, default="dev", help="eval tier (smoke|dev|final|full)")
    parser.add_argument(
        "--pipeline-version",
        type=str,
        default=DEFAULT_PIPELINE_VERSION,
        help=("pipeline version stamped into the decision record"),
    )
    parser.add_argument(
        "--f2p-rate",
        type=float,
        default=None,
        help=("candidate F2P rate for the new champion record (deploy path)"),
    )
    parser.add_argument(
        "--p2p-rate",
        type=float,
        default=None,
        help=("candidate P2P rate for the new champion record (deploy path)"),
    )
    args = parser.parse_args(argv)
    if args.deploy:
        if args.candidate_variant is not None or args.no_eval:
            parser.error("--deploy cannot be combined with --candidate-variant/--no-eval")
        if args.variant is None:
            parser.error("--deploy requires --variant")
    elif args.candidate_variant is None:
        parser.error("--candidate-variant is required (or pass --deploy)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the decide or deploy job; returns the process exit code."""
    args = _parse_args(argv)
    if args.deploy:
        return _deploy_main(args)
    return _decide_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
