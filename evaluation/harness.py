"""Evaluation orchestration: checkpoints, W&B logging, and entry points.

Consolidates the Phase 5 orchestrator (was ``golden_runner`` /
``swebench_runner`` / ``baseline_runner``), per-repo resume (was
``resume.py``) and W&B artifact logging (was ``wandb_logger.py``) into one
module.

Executors (patch generation, test running) are reached through thin
module-level indirection functions (``_generate_patches``, ``_run_tests``) so
tests can monkeypatch them; Modal, W&B and GCS are imported lazily inside
methods and never at import time.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import random
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics, compute_f2p
from evaluation.schema import (
    EvalInput,
    EvalResult,
    EvalRun,
    F2PMetrics,
    PatchApplicationResult,
    TestResult,
)

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    """Generate a timestamped eval run id (e.g. ``eval-20260731-143000``)."""
    return f"eval-{datetime.now():%Y%m%d-%H%M%S}"


# ── Modal app lifecycle ──────────────────────────────────────────────────────

# Long-lived ``app.run()`` contexts, keyed by Modal App instance. Entered once
# per app on first use and never closed (the Modal/synchronicity event loop
# runs on a daemon thread, so process exit is never blocked).
_APP_RUN_STACKS: dict[Any, contextlib.ExitStack] = {}
# Apps whose ``app.run()`` raised once: Modal execution stays disabled for them
# for the rest of the process. Without this, every later example re-enters
# ``app.run()``, re-triggering AppCreate + image build and eventually burning
# Modal's app-create rate limit on attempts that cannot succeed.
_APP_RUN_FAILED: set[Any] = set()
# One-shot flag: Modal output streaming enabled for the whole process (a bare
# list because ``global`` trips PLW0603; entering Modal's ``enable_output``
# context manager and never exiting matches the ``app.run()`` lifetime).
_OUTPUT_ENABLED: list[bool] = [False]


def _ensure_app_running(app: Any) -> None:
    """Open ``app.run()`` once per app and keep it open for the process.

    Modal 1.5.3: an ``@app.function()`` object only works via ``.remote()``
    while its owning ``App`` is running — functions are hydrated when
    ``app.run()`` is entered and un-hydrated again on exit, and re-entering
    ``app.run()`` while it is already running raises ``InvalidError``. One
    long-lived context per app therefore serves every ``.remote()`` call of
    the eval run (the same shape ``modal run`` uses for a whole entrypoint),
    instead of creating and tearing down an app per call.

    A failed ``app.run()`` (e.g. a remote image build error) is recorded in
    ``_APP_RUN_FAILED`` before re-raising: the first example surfaces the real
    error and every later example returns here without entering, so callers
    can fail fast instead of re-attempting the doomed build.
    """
    if app in _APP_RUN_FAILED:
        return
    if not _OUTPUT_ENABLED[0]:
        # Stream build/function logs to the console for the rest of the
        # process, so a failing image build prints its real build logs.
        import modal

        modal.enable_output().__enter__()
        _OUTPUT_ENABLED[0] = True
    if app not in _APP_RUN_STACKS:
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(app.run())
        except Exception as exc:  # noqa: BLE001 — surface once, then disable Modal
            logger.error(
                "Modal app.run() failed for %r — disabling Modal execution for this process: %s",
                app,
                exc,
                exc_info=True,
            )
            _APP_RUN_FAILED.add(app)
            raise
        _APP_RUN_STACKS[app] = stack


# ── Executor indirection (monkeypatchable in tests) ─────────────────────────


def _generate_patches(
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[EvalInput],
) -> list[str]:
    """Generate patches for *examples* via ``evaluation.inference``.

    Thin wrapper over the Modal ``generate_patches_batch`` function (imported
    lazily) so the harness never touches Modal at import time and tests can
    replace this function wholesale.
    """
    from evaluation.inference import app as _inference_app
    from evaluation.inference import generate_patches_batch

    _ensure_app_running(_inference_app)
    if _inference_app in _APP_RUN_FAILED:
        raise RuntimeError("Modal disabled for this process (app.run() failed earlier)")

    # Modal 1.x: @app.function() objects are not callable — invoke via .remote()
    return generate_patches_batch.remote(model_name, variant, prompt_template, examples)  # type: ignore[no-any-return]


def _run_tests(example: EvalInput, generated_patch: str, config: EvalConfig) -> dict[str, Any]:
    """Run the eval test suite for one example via ``evaluation.test_runner``.

    Thin wrapper over the Modal ``run_tests_in_container`` function (imported
    lazily); tests mock this layer to avoid Modal entirely.
    """
    from evaluation.test_runner import app as _test_runner_app
    from evaluation.test_runner import run_tests_in_container

    _ensure_app_running(_test_runner_app)
    if _test_runner_app in _APP_RUN_FAILED:
        raise RuntimeError("Modal disabled for this process (app.run() failed earlier)")

    return run_tests_in_container.remote(  # type: ignore[no-any-return]
        example.repo,
        example.base_sha,
        test_patch=example.test_patch,
        generated_patch=generated_patch,
        fail_to_pass=example.fail_to_pass,
        pass_to_pass=example.pass_to_pass,
        timeout=config.test_timeout_seconds,
        max_retries=config.max_retries,
    )


def _run_tests_batch(
    repo: str,
    base_sha: str,
    test_patch: str | None,
    test_jobs: list[dict[str, Any]],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """Run tests for multiple patches (Modal batch, fallback to per-job).

    Tries ``_run_tests_batch_modal`` first (one container for all jobs).
    Falls back to ``_run_tests`` per job when Modal is unavailable or fails.
    Uses explicit module lookup so tests can monkeypatch the modal function.
    """
    # Explicit module lookup: allows tests to monkeypatch the modal function
    import evaluation.harness as _harness  # noqa: PLW0406 — intentional for monkeypatch support

    try:
        return _harness._run_tests_batch_modal(  # type: ignore[attr-defined]
            repo,
            base_sha,
            test_patch,
            test_jobs,
            config,
        )
    except Exception:  # noqa: BLE001 — Modal unavailable, fall back
        logger.info(
            "batch test runner unavailable for %s, falling back to per-job",
            repo,
        )
        return _harness._run_tests_batch_fallback(  # type: ignore[attr-defined]
            repo,
            base_sha,
            test_patch,
            test_jobs,
            config,
        )


def _run_tests_batch_modal(
    repo: str,
    base_sha: str,
    test_patch: str | None,
    test_jobs: list[dict[str, Any]],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """Modal batch path: one container for all jobs."""
    from evaluation.test_runner import app as _test_runner_app
    from evaluation.test_runner import run_tests_batch as _modal_batch

    _ensure_app_running(_test_runner_app)
    if _test_runner_app in _APP_RUN_FAILED:
        raise RuntimeError("Modal disabled for this process (app.run() failed earlier)")

    return _modal_batch.remote(  # type: ignore[no-any-return]
        repo,
        base_sha,
        test_patch,
        test_jobs,
        timeout=config.test_timeout_seconds,
        max_retries=config.max_retries,
    )


def _run_tests_batch_fallback(
    repo: str,
    base_sha: str,
    test_patch: str | None,
    test_jobs: list[dict[str, Any]],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """Fallback: delegate to ``_run_tests`` per job (test-compatible).

    Uses explicit module lookup so monkeypatched ``_run_tests`` is picked up.
    """
    import evaluation.harness as _harness  # noqa: PLW0406 — intentional for monkeypatch support

    results: list[dict[str, Any]] = []
    for job in test_jobs:
        example = EvalInput(
            instance_id=job.get("instance_id", ""),
            repo=repo,
            issue_body="",
            base_sha=base_sha,
            head_sha="",
            test_patch=test_patch or "",
            fail_to_pass=job.get("fail_to_pass") or [],
            pass_to_pass=job.get("pass_to_pass") or [],
            repo_domain="",
        )
        results.append(
            _harness._run_tests(example, job.get("generated_patch") or "", config)  # type: ignore[attr-defined]
        )
    return results


# ── Data reading ─────────────────────────────────────────────────────────────


def _read_gcs(path: str) -> str:
    """Read a ``gs://`` object to text (lazy google-cloud-storage import)."""
    from google.cloud import storage  # type: ignore[attr-defined]

    client = storage.Client()
    bucket_name, _, key = path[len("gs://") :].partition("/")
    if not bucket_name or not key:
        raise ValueError(f"invalid GCS path: {path!r}")
    return str(client.bucket(bucket_name).blob(key).download_as_text())


def _read_text(path: str) -> str:
    """Read a dataset file from ``gs://`` or the local filesystem."""
    if path.startswith("gs://"):
        return _read_gcs(path)
    return Path(path).read_text(encoding="utf-8")


def _record_to_input(record: dict[str, Any]) -> EvalInput:
    """Build an ``EvalInput`` from any golden record shape.

    Always goes through ``EvalInput.from_swebench_record`` because it
    normalizes ``test_results`` fields through ``_to_test_list``.  The
    ``model_validate`` fast path was removed — golden.jsonl stores test-name
    lists as split JSON-encoded strings that only ``_to_test_list`` correctly
    reassembles.
    """
    return EvalInput.from_swebench_record(record)


# ── CheckpointManager ────────────────────────────────────────────────────────


class CheckpointManager:
    """Per-repo checkpoint resume (consolidated from ``resume.py``).

    Checkpoints are JSON files at
    ``{checkpoint_dir}/{run_id}/{repo}__{model}__{variant}__{template}.json``
    (``/`` in repo names replaced with ``_``). A repo with multiple examples
    stores a JSON list of results; a single-example repo stores the bare
    result dict. Writes are atomic (temp file + rename).
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    def get_checkpoint_key(
        self, run_id: str, repo: str, model: str, variant: str, prompt_template: str
    ) -> str:
        """Return the stable checkpoint key for a repo+model+variant+template.

        The key embeds all five identity parts (``run_id`` as the leading
        directory segment) and is stable across invocations, so a completed
        repo is skipped on resume. The prompt template is part of the key —
        without it, two templates in one run would collide and the second
        template's repos would be wrongly skipped.
        """
        repo_slug = repo.replace("/", "_")
        template_slug = prompt_template.replace("/", "_")
        return f"{run_id}/{repo_slug}__{model}__{variant}__{template_slug}"

    def _path(self, key: str) -> Path:
        return self.checkpoint_dir / f"{key}.json"

    def is_completed(self, key: str) -> bool:
        """Return True if a checkpoint file exists for *key*."""
        return self._path(key).is_file()

    def save_result(self, key: str, result: EvalResult | list[EvalResult]) -> None:
        """Persist one result (or a repo's list of results) atomically."""
        if isinstance(result, list):
            payload: dict[str, Any] | list[dict[str, Any]] = [
                r.model_dump(mode="json") for r in result
            ]
        else:
            payload = result.model_dump(mode="json")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def load_results(self, run_id: str) -> list[EvalResult]:
        """Load every checkpointed result for *run_id*, skipping corrupt files.

        Args:
            run_id: Eval run id (leading segment of each checkpoint key).

        Returns:
            All valid ``EvalResult`` records found under
            ``{checkpoint_dir}/{run_id}/``.
        """
        results: list[EvalResult] = []
        run_dir = self.checkpoint_dir / run_id
        if not run_dir.is_dir():
            return results
        for path in sorted(run_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("skipping unreadable checkpoint %s", path)
                continue
            records = raw if isinstance(raw, list) else [raw]
            for record in records:
                try:
                    results.append(EvalResult.model_validate(record))
                except ValidationError:
                    logger.warning("skipping invalid checkpoint record in %s", path)
        logger.info("loaded %d checkpointed results for run %s", len(results), run_id)
        return results


# ── WandbLogger ──────────────────────────────────────────────────────────────


def _wandb_or_none() -> Any:
    """Return the ``wandb`` module, or None when it is not installed.

    Pure availability check. The active-run guard lives in
    ``WandbLogger._ensure_run``, which lazily initializes a run instead of
    skipping logging.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
        return None
    return wandb


def _log_text_artifact(
    wandb_mod: Any,
    name: str,
    type_: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write *content* to a temp file and upload it as a W&B artifact."""
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, encoding="utf-8") as fh:
            fh.write(content)
            path = Path(fh.name)
        artifact = wandb_mod.Artifact(name=name, type=type_, metadata=metadata or {})
        artifact.add_file(str(path))
        wandb_mod.log_artifact(artifact)
        artifact.wait(timeout=120)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


class WandbLogger:
    """W&B artifact logging (consolidated from ``wandb_logger.py``).

    The first log attempt lazily initializes a W&B run (once per run_id,
    ``reinit=True``); if W&B is unavailable or init fails, logging is
    disabled for this logger and every method is a no-op. W&B failures
    never raise.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self._wandb_mod: Any = None  # lazily imported wandb module
        self._wandb_run_id: str | None = None  # run_id the active run was started for
        self._wandb_disabled = False  # permanently skip W&B after import/init failure

    def _ensure_run(self, run_id: str) -> Any:
        """Return ``wandb`` with an active run for *run_id*, or None.

        Lazily initializes a run on the first log attempt for *run_id*;
        ``reinit=True`` lets a second run_id in the same process start a
        fresh run. If W&B is unavailable or init fails, W&B logging is
        permanently disabled for this logger and None is returned — the
        harness never depends on W&B.

        Args:
            run_id: Eval run id; also used as the W&B run name.
        """
        if self._wandb_disabled:
            return None
        if self._wandb_mod is None:
            self._wandb_mod = _wandb_or_none()
            if self._wandb_mod is None:
                self._wandb_disabled = True
                return None
        if self._wandb_run_id == run_id:
            return self._wandb_mod
        try:
            self._wandb_mod.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=run_id,  # run_id already carries the "eval-" prefix (make_run_id)
                reinit=True,
            )
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B init failed — disabling W&B logging", exc_info=True)
            self._wandb_disabled = True
            return None
        self._wandb_run_id = run_id
        return self._wandb_mod

    def log_eval_run(self, run: EvalRun, config: EvalConfig) -> None:
        """Log a full eval run: per-example, aggregate and per-repo artifacts."""
        if config.wandb_log_per_example:
            self.log_per_example(run.results, run.run_id)
        if config.wandb_log_aggregate:
            self.log_aggregate(run.aggregate, run.run_id)
        self.log_per_repo(run.aggregate, run.run_id)

    def log_per_example(
        self,
        results: list[EvalResult],
        run_id: str,
        artifact_name: str | None = None,
    ) -> None:
        """Write results as a JSONL artifact ``eval-results-{run_id}``.

        Args:
            results: Per-example evaluation results.
            run_id: Eval run id used in the default artifact name.
            artifact_name: Optional artifact name override (e.g. the prompt
                A/B artifact ``eval-prompt-ab-{run_id}``).
        """
        if not results:
            return
        wandb_mod = self._ensure_run(run_id)
        if wandb_mod is None:
            return
        try:
            lines = [json.dumps(r.model_dump(mode="json")) for r in results]
            _log_text_artifact(
                wandb_mod,
                artifact_name or f"eval-results-{run_id}",
                "eval_results",
                "\n".join(lines) + "\n",
                {"run_id": run_id, "example_count": len(results)},
            )
            logger.info(
                "logged eval results artifact for run %s (%d examples)", run_id, len(results)
            )
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B per-example logging failed for run %s", run_id, exc_info=True)

    def log_aggregate(self, metrics: list[F2PMetrics], run_id: str) -> None:
        """Log summary scalars plus an ``eval-aggregate-{run_id}`` artifact.

        Scalar keys follow ``eval/{model}/{variant}/{prompt}/<metric>``; the
        artifact (type ``eval_metrics``) holds the full ``F2PMetrics`` dumps.
        """
        if not metrics:
            return
        wandb_mod = self._ensure_run(run_id)
        if wandb_mod is None:
            return
        try:
            scalars: dict[str, float | int] = {}
            for m in metrics:
                prefix = f"eval/{m.model_name}/{m.variant}/{m.prompt_template}"
                scalars[f"{prefix}/f2p_rate"] = m.f2p_rate
                scalars[f"{prefix}/p2p_rate"] = m.p2p_rate
                scalars[f"{prefix}/avg_latency"] = m.avg_latency
                scalars[f"{prefix}/flaky_test_rate"] = m.flaky_test_rate
                scalars[f"{prefix}/successful_patches"] = m.successful_patches
                scalars[f"{prefix}/total_examples"] = m.total_examples
            wandb_mod.log(scalars)
            _log_text_artifact(
                wandb_mod,
                f"eval-aggregate-{run_id}",
                "eval_metrics",
                json.dumps([m.model_dump(mode="json") for m in metrics], indent=2),
                {"run_id": run_id, "group_count": len(metrics)},
            )
            logger.info("logged eval aggregate for run %s (%d groups)", run_id, len(metrics))
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B aggregate logging failed for run %s", run_id, exc_info=True)

    def log_per_repo(self, metrics: list[F2PMetrics], run_id: str) -> None:
        """Log a CSV ``eval-per-repo-{run_id}`` artifact (type ``eval_breakdown``)."""
        if not metrics:
            return
        wandb_mod = self._ensure_run(run_id)
        if wandb_mod is None:
            return
        try:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "model_name",
                    "variant",
                    "prompt_template",
                    "repo",
                    "f2p_rate",
                    "p2p_rate",
                    "count",
                ]
            )
            for m in metrics:
                for repo, breakdown in m.per_repo_breakdown.items():
                    writer.writerow(
                        [
                            m.model_name,
                            m.variant,
                            m.prompt_template,
                            repo,
                            breakdown.get("f2p_rate", 0.0),
                            breakdown.get("p2p_rate", 0.0),
                            breakdown.get("count", 0),
                        ]
                    )
            _log_text_artifact(
                wandb_mod,
                f"eval-per-repo-{run_id}",
                "eval_breakdown",
                buf.getvalue(),
                {"run_id": run_id},
            )
            logger.info("logged per-repo breakdown for run %s", run_id)
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B per-repo logging failed for run %s", run_id, exc_info=True)


# ── EvaluationHarness ────────────────────────────────────────────────────────


def _persist_run(run: EvalRun, config: EvalConfig) -> None:
    """Persist a finalized run to ``{output_dir}/{run_id}.json``.

    The file is a single ``EvalRun`` JSON dump (``model_dump(mode="json")``),
    which is what ``evaluation.comparison.load_all_eval_runs`` reads on its
    local-first path.
    """
    path = config.output_dir / f"{run.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


class EvaluationHarness:
    """Orchestrator: load examples → generate patches → run tests → metrics.

    Entry points (``run_golden`` / ``run_swebench_verified`` /
    ``run_baseline``) share one code path: load (optionally sampled) examples,
    run per-model/variant/prompt batches with per-repo checkpoint resume,
    aggregate F2P/P2P metrics, assemble an ``EvalRun`` and log it to W&B.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.results: list[EvalResult] = []
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        self.wandb_logger = WandbLogger(config)

    def load_examples(self, split: str = "golden", run_id: str | None = None) -> list[EvalInput]:
        """Load ``EvalInput`` records from ``golden_data_path``.

        The ``{run_id}`` placeholder in the path is substituted from *run_id*
        or ``config.resume_from`` (a dataset path without the placeholder is
        used verbatim). ``split="swebench_verified"`` keeps only records whose
        ``metadata.is_verified`` is True.

        Args:
            split: ``"golden"`` (default) or ``"swebench_verified"``.
            run_id: Dataset run id for ``{run_id}`` substitution; falls back
                to ``config.resume_from``.

        Returns:
            Reconstructed EvalInput list, one per JSONL line.

        Raises:
            ValueError: if a placeholder is present but no run id is available,
                or for an unknown split.
        """
        run_id = run_id or self.config.resume_from
        path = self.config.golden_data_path
        if "{run_id}" in path:
            if not run_id:
                raise ValueError(
                    "golden_data_path contains a {run_id} placeholder but no run id "
                    "was provided; pass run_id=... or set EVAL_RESUME_FROM"
                )
            path = path.format(run_id=run_id)
        examples: list[EvalInput] = []
        for line_no, line in enumerate(_read_text(path).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSON line %d in %s", line_no, path)
                continue
            examples.append(_record_to_input(record))
        if split == "swebench_verified":
            examples = [ex for ex in examples if ex.metadata.get("is_verified") is True]
        elif split != "golden":
            raise ValueError(f"unknown split {split!r}; expected 'golden' or 'swebench_verified'")
        logger.info("loaded %d examples (%s) from %s", len(examples), split, path)
        return examples

    def run_example(
        self,
        example: EvalInput,
        model_name: str,
        variant: str,
        prompt_template: str,
        generated_patch: str | None = None,
    ) -> EvalResult:
        """Evaluate a single example: generate patch → run tests → metrics.

        Threads ``_generate_patches`` (inference) → ``_run_tests`` (Modal test
        runner) → ``compute_f2p`` (metrics). Exceptions never propagate: a
        failure is captured in the returned ``EvalResult.error`` so callers
        can mark the run partial.

        Args:
            example: Eval input instance.
            model_name: Model registry key.
            variant: Trained variant key.
            prompt_template: Template name for prompt rendering.
            generated_patch: Pre-generated patch string. If provided, skips
                ``_generate_patches`` call (batch already did it). If not
                provided (legacy path), calls ``_generate_patches`` for this
                single example.

        Returns:
            An EvalResult with f2p/p2p rates computed from the test runner's
            before/after payloads.
        """
        started = time.monotonic()
        try:
            if generated_patch is not None:
                # Batch path: patch already generated
                pass
            else:
                # Legacy path: generate on the fly
                patches = _generate_patches(model_name, variant, prompt_template, [example])
                generated_patch = patches[0] if patches else ""
            output = _run_tests(example, generated_patch, self.config)
            tests_before = [TestResult.model_validate(t) for t in output.get("tests_before") or []]
            tests_after = [TestResult.model_validate(t) for t in output.get("tests_after") or []]
            raw_patch = output.get("patch_application") or {}
            patch_application = (
                PatchApplicationResult.model_validate(raw_patch)
                if raw_patch
                else PatchApplicationResult(
                    success=False,
                    method_used="failed",
                    error=str(
                        output.get("error") or "test runner returned no patch application result"
                    ),
                )
            )
            error = str(output["error"]) if output.get("error") else None
            f2p, p2p, _f2p_count, _p2p_count = compute_f2p(
                tests_before, tests_after, example.fail_to_pass, example.pass_to_pass
            )
            return EvalResult(
                instance_id=example.instance_id,
                repo=example.repo,
                model_name=model_name,
                variant=variant,
                prompt_template=prompt_template,
                generated_patch=generated_patch,
                patch_application=patch_application,
                tests_before=tests_before,
                tests_after=tests_after,
                f2p=f2p,
                p2p=p2p,
                latency_seconds=time.monotonic() - started,
                timestamp=datetime.now(UTC),
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 — one bad example must not fail the batch
            logger.warning(
                "run_example failed for %s (%s/%s): %s",
                example.instance_id,
                model_name,
                variant,
                exc,
                exc_info=True,
            )
            return EvalResult(
                instance_id=example.instance_id,
                repo=example.repo,
                model_name=model_name,
                variant=variant,
                prompt_template=prompt_template,
                generated_patch="",
                patch_application=PatchApplicationResult(
                    success=False, method_used="failed", error=str(exc)
                ),
                tests_before=[],
                tests_after=[],
                f2p=0.0,
                p2p=0.0,
                latency_seconds=time.monotonic() - started,
                timestamp=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_example_from_output(  # noqa: PLR0913, PLR0917
        self,
        example: EvalInput,
        model_name: str,
        variant: str,
        prompt_template: str,
        generated_patch: str,
        output: dict[str, Any],
    ) -> EvalResult:
        """Build EvalResult from pre-fetched test output (batch path).

        Args:
            example: Eval input instance.
            model_name: Model registry key.
            variant: Trained variant key.
            prompt_template: Template name.
            generated_patch: Patch string already applied in container.
            output: Test result dict from ``run_tests_batch`` (same shape as
                ``run_tests_in_container`` return).

        Returns:
            EvalResult with f2p/p2p computed from the output payload.
        """
        tests_before = [TestResult.model_validate(t) for t in output.get("tests_before") or []]
        tests_after = [TestResult.model_validate(t) for t in output.get("tests_after") or []]
        raw_patch = output.get("patch_application") or {}
        patch_application = (
            PatchApplicationResult.model_validate(raw_patch)
            if raw_patch
            else PatchApplicationResult(
                success=False,
                method_used="failed",
                error="no patch application result",
            )
        )
        error = str(output["error"]) if output.get("error") else None
        f2p, p2p, _f2p_count, _p2p_count = compute_f2p(
            tests_before, tests_after, example.fail_to_pass, example.pass_to_pass
        )
        return EvalResult(
            instance_id=example.instance_id,
            repo=example.repo,
            model_name=model_name,
            variant=variant,
            prompt_template=prompt_template,
            generated_patch=generated_patch,
            patch_application=patch_application,
            tests_before=tests_before,
            tests_after=tests_after,
            f2p=f2p,
            p2p=p2p,
            latency_seconds=0.0,
            timestamp=datetime.now(UTC),
            error=error,
        )

    def run_batch(  # noqa: PLR0912
        self,
        examples: list[EvalInput],
        model_name: str,
        variant: str,
        prompt_template: str,
        run_id: str,
    ) -> list[EvalResult]:
        """Run a batch with per-repo checkpoint resume.

        Generates patches for ALL non-completed examples in a single
        ``_generate_patches`` call (one Modal container lifetime), then
        processes per-repo for tests and checkpointing.
        """
        if not examples:
            return []
        if not run_id:
            raise ValueError("run_id is required for checkpointing")
        loaded = [
            r
            for r in self.checkpoint_mgr.load_results(run_id)
            if r.model_name == model_name
            and r.variant == variant
            and r.prompt_template == prompt_template
        ]
        by_repo: dict[str, list[EvalInput]] = {}
        for example in examples:
            by_repo.setdefault(example.repo, []).append(example)

        # Phase 1: identify non-completed repos, collect all examples
        pending_repos: dict[str, list[EvalInput]] = {}
        for repo in sorted(by_repo):
            key = self.checkpoint_mgr.get_checkpoint_key(
                run_id, repo, model_name, variant, prompt_template
            )
            if self.checkpoint_mgr.is_completed(key):
                logger.info("skipping completed repo %s (checkpoint %s)", repo, key)
                continue
            pending_repos[repo] = by_repo[repo]

        # Generate patches for ALL pending examples in ONE call
        all_pending: list[EvalInput] = []
        for repo in sorted(pending_repos):
            all_pending.extend(pending_repos[repo])

        patches_map: dict[str, str] = {}
        if all_pending:
            patches = _generate_patches(model_name, variant, prompt_template, all_pending)
            for i, example in enumerate(all_pending):
                patches_map[example.instance_id] = patches[i] if i < len(patches) else ""

        # Phase 2: per-repo test execution and checkpointing
        new: list[EvalResult] = []
        for repo in sorted(pending_repos):
            repo_examples = pending_repos[repo]

            # Build test jobs list
            test_jobs: list[dict[str, Any]] = []
            for example in repo_examples:
                patch = patches_map.get(example.instance_id, "")
                test_jobs.append(
                    {
                        "generated_patch": patch,
                        "fail_to_pass": example.fail_to_pass or [],
                        "pass_to_pass": example.pass_to_pass or [],
                    }
                )

            # Run all tests for this repo in one container
            batch_results = _run_tests_batch(
                repo,
                repo_examples[0].base_sha,
                repo_examples[0].test_patch or None,
                test_jobs,
                self.config,
            )

            # Pair results back with examples
            repo_results: list[EvalResult] = []
            for i, example in enumerate(repo_examples):
                if i < len(batch_results):
                    result = self.run_example_from_output(
                        example,
                        model_name,
                        variant,
                        prompt_template,
                        patches_map.get(example.instance_id, ""),
                        batch_results[i],
                    )
                else:
                    # Fallback if batch returned fewer results
                    result = self.run_example(
                        example,
                        model_name,
                        variant,
                        prompt_template,
                        generated_patch=patches_map.get(example.instance_id, ""),
                    )
                repo_results.append(result)
            new.extend(repo_results)

            key = self.checkpoint_mgr.get_checkpoint_key(
                run_id, repo, model_name, variant, prompt_template
            )
            if repo_results:
                payload: EvalResult | list[EvalResult] = (
                    repo_results[0] if len(repo_results) == 1 else repo_results
                )
                self.checkpoint_mgr.save_result(key, payload)
                logger.info(
                    "checkpointed %d result(s) for %s under %s", len(repo_results), repo, key
                )
        return [*loaded, *new]

    def run_golden(
        self,
        models: list[tuple[str, str]],
        prompt_templates: list[str] | None = None,
        sample: int = 0,
        run_id: str | None = None,
    ) -> EvalRun:
        """Entry point: run the golden eval on all model/variant/prompt combos.

        Args:
            models: ``(model_name, variant)`` pairs.
            prompt_templates: Template names to evaluate (default ``["chat"]``).
            sample: If > 0, random sample (seeded from ``ci_random_seed``).
            run_id: Eval run id; generated if not provided.

        Returns:
            Assembled EvalRun, already logged to W&B.
        """
        return self._run_split(
            models, "golden", prompt_templates or ["chat"], sample=sample, run_id=run_id
        )

    def run_swebench_verified(
        self,
        models: list[tuple[str, str]],
        sample: int = 0,
        run_id: str | None = None,
    ) -> EvalRun:
        """Entry point: run the SWE-bench Verified subset (``is_verified``)."""
        return self._run_split(models, "swebench_verified", ["chat"], sample=sample, run_id=run_id)

    def run_baseline(
        self,
        model: str | None = None,
        sample: int = 0,
        run_id: str | None = None,
    ) -> EvalRun:
        """Entry point: evaluate the unfine-tuned base model (no LoRA).

        Uses variant ``"baseline"``; adapter resolution in ``inference.py``
        returns None for it, falling back to the base model.
        """
        return self.run_golden(
            [(model or self.config.baseline_model, "baseline")],
            ["chat"],
            sample=sample,
            run_id=run_id,
        )

    def _run_split(
        self,
        models: list[tuple[str, str]],
        split: str,
        prompt_templates: list[str],
        sample: int,
        run_id: str | None,
    ) -> EvalRun:
        """Shared runner: load → sample → per-combo batches → aggregate → log."""
        run_id = run_id or make_run_id()
        started_at = datetime.now(UTC)
        examples = self.load_examples(split, run_id=run_id)
        if sample > 0:
            examples = random.Random(self.config.ci_random_seed).sample(
                examples, min(sample, len(examples))
            )
            logger.info(
                "sampled %d of %d examples (seed=%d)",
                len(examples),
                sample,
                self.config.ci_random_seed,
            )

        results: list[EvalResult] = []
        for model_name, variant in models:
            for template in prompt_templates:
                results.extend(self.run_batch(examples, model_name, variant, template, run_id))
        self.results.extend(results)

        run = EvalRun(
            run_id=run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            config=self.config,
            models_evaluated=[f"{name}:{variant}" for name, variant in models],
            results=results,
            aggregate=self._aggregate(results),
            status="partial" if any(r.error for r in results) else "completed",
        )
        _persist_run(run, self.config)
        try:
            self.wandb_logger.log_eval_run(run, self.config)
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B logging failed for run %s", run_id, exc_info=True)
        logger.info(
            "run %s (%s): %d results, status=%s",
            run_id,
            split,
            len(results),
            run.status,
        )
        return run

    @staticmethod
    def _aggregate(results: list[EvalResult]) -> list[F2PMetrics]:
        """Group results by model/variant/prompt and aggregate each group."""
        groups: dict[tuple[str, str, str], list[EvalResult]] = {}
        for result in results:
            groups.setdefault(
                (result.model_name, result.variant, result.prompt_template), []
            ).append(result)
        return [aggregate_metrics(groups[key]) for key in sorted(groups)]
