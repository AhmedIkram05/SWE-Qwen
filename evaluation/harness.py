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

import atexit
import contextlib
import csv
import io
import json
import logging
import math
import os
import random
import statistics
import tempfile
import threading
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
# per app on first use and closed at process exit (see ``_close_modal_apps``);
# closing earlier would un-hydrate functions still needed by later calls.
_APP_RUN_STACKS: dict[Any, contextlib.ExitStack] = {}
# Serializes the check-and-enter in ``_ensure_app_running``: without it, two
# combo/swebench worker threads can both see the app absent, both call
# ``app.run()``, and the second raises InvalidError (already running).
_APP_LOCK = threading.Lock()
# Apps whose ``app.run()`` raised once: Modal execution stays disabled for them
# for the rest of the process. Without this, every later example re-enters
# ``app.run()``, re-triggering AppCreate + image build and eventually burning
# Modal's app-create rate limit on attempts that cannot succeed.
_APP_RUN_FAILED: set[Any] = set()
# One-shot flag: Modal output streaming enabled for the whole process (a bare
# list because ``global`` trips PLW0603). The context manager object is kept
# so the atexit handler can close it symmetrically.
_OUTPUT_ENABLED: list[bool] = [False]
_OUTPUT_CM: list[Any] = [None]
# Apps a stop-watchdog thread is already running for (see
# ``_start_stop_watchdog``). Guarded by ``_APP_LOCK``.
_APP_WATCHED: set[Any] = set()


def _close_modal_apps() -> None:
    """Close open ``app.run()`` contexts at interpreter exit.

    Without this, Python 3.14 shuts down the asyncio threadpool before Modal's
    deferred AppClientDisconnect runs, and teardown of the never-closed app
    context crashes with ``cannot schedule new futures after shutdown`` +
    ``ConnectionError`` + ``generator didn't stop after athrow()`` — a noisy
    traceback cascade after every successful CLI run.  atexit handlers run
    (LIFO) before ``threading._shutdown`` kills the daemon loop thread, so the
    disconnect happens while the event loop is still alive.
    """
    while _APP_RUN_STACKS:
        _app, stack = _APP_RUN_STACKS.popitem()
        try:
            stack.close()
        except Exception:  # noqa: BLE001 — best-effort at exit
            logger.warning("modal app teardown failed for %r", _app, exc_info=True)
    if _OUTPUT_CM[0] is not None:
        try:
            _OUTPUT_CM[0].__exit__(None, None, None)
        except Exception:  # noqa: BLE001 — best-effort at exit
            logger.warning("modal output streaming teardown failed", exc_info=True)
        _OUTPUT_CM[0] = None


atexit.register(_close_modal_apps)


def _app_heartbeat_alive(client: Any, request: Any, loop: Any) -> bool:
    """One heartbeat probe: True if the app is alive, False if stopped.

    Modal 1.5.x raises ``ConflictError`` on ``AppHeartbeat`` once the app
    leaves the running state (e.g. stopped from the dashboard). Any other
    error counts as alive so a transient network blip doesn't kill the eval.
    """
    try:
        loop.run_until_complete(client.stub.AppHeartbeat(request))
    except Exception as exc:  # noqa: BLE001 — deliberate: only stop on ConflictError
        from modal.exception import ConflictError

        return not isinstance(exc, ConflictError)
    return True


def _start_stop_watchdog(app: Any) -> None:
    """Exit the process if the app is stopped from the Modal dashboard.

    Modal 1.5.x's client keeps heartbeating forever after a dashboard stop
    (the ``ConflictError`` is logged and swallowed by ``_run_app``'s infinite
    loop) and the in-flight ``.remote()`` result wait never resolves, so a CI
    eval step would hang until the job timeout. This watchdog replays modal's
    own heartbeat probe from a daemon thread and aborts the process shortly
    after the app is stopped, so the GH Actions step fails fast instead of
    burning 240 minutes of runner time.

    Must be called with ``_APP_LOCK`` held (from ``_ensure_app_running``).
    """
    if app in _APP_WATCHED:
        return
    running = getattr(app, "_running_app", None)
    app_id = getattr(running, "app_id", None)
    if not app_id:
        return
    _APP_WATCHED.add(app)

    def _watch() -> None:
        import asyncio

        from modal._load_context import load_context
        from modal_proto import api_pb2

        try:
            client = load_context.client
        except Exception:  # noqa: BLE001 — no client, nothing to watch
            logger.debug("modal stop watchdog: no client available", exc_info=True)
            return
        request = api_pb2.AppHeartbeatRequest(app_id=app_id)
        loop = asyncio.new_event_loop()
        while True:
            time.sleep(15)  # mirror modal's own HEARTBEAT_INTERVAL
            if not _app_heartbeat_alive(client, request, loop):
                logger.error(
                    "Modal app %s was stopped from the dashboard — aborting "
                    "(modal 1.5.x client would heartbeat forever)",
                    app_id,
                )
                # App is gone; modal teardown is pointless. os._exit skips
                # atexit, which is exactly what we want on the abort path.
                os._exit(1)

    threading.Thread(target=_watch, name="modal-stop-watchdog", daemon=True).start()


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

    Thread-safe: the whole check-and-enter runs under ``_APP_LOCK`` so
    concurrent callers (combo threads, swebench pool workers) cannot both
    enter ``app.run()`` for the same app.
    """
    with _APP_LOCK:
        if app in _APP_RUN_FAILED:
            return
        if not _OUTPUT_ENABLED[0]:
            # Stream build/function logs to the console for the rest of the
            # process, so a failing image build prints its real build logs.
            import modal

            _OUTPUT_CM[0] = modal.enable_output()
            _OUTPUT_CM[0].__enter__()
            _OUTPUT_ENABLED[0] = True
        if app not in _APP_RUN_STACKS:
            stack = contextlib.ExitStack()
            try:
                stack.enter_context(app.run())
            except Exception as exc:  # noqa: BLE001, E501 — surface once, then disable Modal
                logger.error(
                    "Modal app.run() failed for %r — disabling Modal execution for this process: %s",  # noqa: E501
                    app,
                    exc,
                    exc_info=True,
                )
                _APP_RUN_FAILED.add(app)
                raise
            _APP_RUN_STACKS[app] = stack
        _start_stop_watchdog(app)


# ── Executor indirection (monkeypatchable in tests) ─────────────────────────


def _generate_patches(  # noqa: PLR0913, PLR0917
    model_name: str,
    variant: str,
    prompt_template: str,
    examples: list[EvalInput],
    max_new_tokens: int = 8192,
    dataset_run_id: str | None = None,
) -> list[str]:
    """Generate patches for *examples* via ``evaluation.inference``.

    Thin wrapper over the Modal ``generate_patches_batch`` function (imported
    lazily) so the harness never touches Modal at import time and tests can
    replace this function wholesale.

    Args:
        max_new_tokens: Maximum completion length (per-tier C2 config).
        dataset_run_id: Pipeline run id; forwarded so the container fetches
            the few-shot golden from that run's GCS artifacts.
    """
    from evaluation.inference import app as _inference_app
    from evaluation.inference import generate_patches_batch

    _ensure_app_running(_inference_app)
    if _inference_app in _APP_RUN_FAILED:
        raise RuntimeError("Modal disabled for this process (app.run() failed earlier)")

    # Chunked remote calls: the fn timeout is 600s and the full tier (2,056
    # instances × ~2K out tokens ≈ 4.1M tokens ≈ 17-20 min on A100-80GB) would
    # be killed in one call. Sequential chunks reuse the SAME warm container
    # (Modal keeps it ~minutes after each call) — model loaded once, no
    # per-chunk cold start. GPU-bound wall time is unchanged; the timeout is
    # the only thing this fixes.
    _INFER_CHUNK_SIZE = 100  # ~200K out tokens ≈ ~1 min per call  # noqa: N806
    patches: list[str] = []
    for i in range(0, len(examples), _INFER_CHUNK_SIZE):
        chunk = examples[i : i + _INFER_CHUNK_SIZE]
        # generate_patches_batch is a plain-function dispatcher that performs
        # the Modal .remote() itself (per inference_gpu tier); calling .remote()
        # on it was drift from the pre-dispatcher refactor (AttributeError).
        patches.extend(
            generate_patches_batch(  # type: ignore[no-any-return]
                model_name,
                variant,
                prompt_template,
                chunk,
                max_new_tokens=max_new_tokens,
                dataset_run_id=dataset_run_id,
            )
        )
    return patches


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
        gold_patch=example.gold_patch or None,
        generated_patch=generated_patch,
        fail_to_pass=example.fail_to_pass,
        pass_to_pass=example.pass_to_pass,
        timeout=config.test_timeout_seconds,
        max_retries=config.max_retries,
        instance_id=example.instance_id,
        verify_mode=config.verify_mode,
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
        verify_mode=config.verify_mode,
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
    Runs jobs in parallel via ``ThreadPoolExecutor`` (up to ``max_parallel``).
    """
    import concurrent.futures

    import evaluation.harness as _harness  # noqa: PLW0406 — intentional for monkeypatch support

    def _run_one(job: dict[str, Any]) -> dict[str, Any]:
        example = EvalInput(
            instance_id=job.get("instance_id", ""),
            repo=repo,
            issue_body="",
            base_sha=base_sha,
            head_sha="",
            test_patch=job.get("test_patch")
            or test_patch
            or "",  # per-job patch, not group-first's
            gold_patch=job.get("gold_patch") or "",
            fail_to_pass=job.get("fail_to_pass") or [],
            pass_to_pass=job.get("pass_to_pass") or [],
            repo_domain="",
        )
        # One bad job (Modal timeout, remote error) must not kill the whole
        # batch: return an error-shaped result (same shape as test_runner's
        # repo-prep failure) and let callers handle it per instance.
        try:
            return _harness._run_tests(example, job.get("generated_patch") or "", config)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — per-job isolation
            logger.error("test run failed for %s: %s", example.instance_id, exc, exc_info=True)
            return {
                "repo": repo,
                "base_sha": base_sha,
                "error": str(exc),
                "tests_before": [],
                "tests_head": [],
                "tests_after": [],
                "patch_application": {},
                "ground_truth": {},
            }

    max_workers = config.max_parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_run_one, test_jobs))


def _run_tests_swebench(
    instances: list[EvalInput],
    patches: dict[str, str],
    config: EvalConfig,
) -> dict[str, dict[str, Any]]:
    """Run instances on their OFFICIAL per-repo SWE-bench images.

    One Modal container per instance; one function (and its swebench image)
    is registered per repo via ``swebench_fn`` — the image contains the repo
    with full git history and a pre-installed conda env, so no clone/install
    happens at eval time.  All functions are registered BEFORE
    ``_ensure_app_running`` (Modal only hydrates functions registered at
    ``app.run()`` entry).

    Returns ``{instance_id: result_dict}``.  Raises on ANY failure so callers
    fall back to the clone/install batch path (coarse group-level fallback;
    the swebench path is primary for SWE-bench Verified where every instance
    has a prebuilt image).
    """
    import concurrent.futures

    from evaluation.test_runner import app as _test_runner_app
    from evaluation.test_runner import swebench_fn, swebench_image_exists

    # Pre-flight: a repo whose swebench image is NOT published on Docker Hub
    # (e.g. sqlfluff_1776_sqlfluff-3662) fails the whole Modal app at
    # image-build time, disabling Modal for the process and losing every test
    # run. Drop those instances here — run_batch's ``missing`` handling
    # routes them to the clone/install batch fallback, so coverage survives.
    missing = [ex for ex in instances if not swebench_image_exists(ex.instance_id)]
    if missing:
        logger.warning(
            "no published swebench eval image for %d instance(s) — routing to "
            "clone/install fallback: %s",
            len(missing),
            ", ".join(ex.instance_id for ex in missing),
        )
    instances = [ex for ex in instances if ex not in missing]
    if not instances:
        return {}

    # Modal 1.5.x hydrates only functions registered BEFORE app.run() entry;
    # registering on a running app leaves the handle unhydrated and the first
    # .remote() raises ExecutionError. Pre-register every per-repo function
    # (all share one module-level body; one image per repo) first, then enter.
    # If the app is already running (earlier batch path in this process), the
    # functions are already registered and the guard skips.
    if _test_runner_app not in _APP_RUN_STACKS:
        for example in instances:
            swebench_fn(example.repo, example.instance_id)

    _ensure_app_running(_test_runner_app)
    if _test_runner_app in _APP_RUN_FAILED:
        raise RuntimeError("Modal disabled for this process (app.run() failed earlier)")

    def _run_one(example: EvalInput) -> tuple[str, dict[str, Any]]:
        fn = swebench_fn(example.repo, example.instance_id)
        return example.instance_id, fn.remote(  # type: ignore[no-any-return]
            example.instance_id,
            example.base_sha,
            test_patch=example.test_patch or None,
            gold_patch=example.gold_patch or None,
            generated_patch=patches.get(example.instance_id, ""),
            fail_to_pass=example.fail_to_pass or [],
            pass_to_pass=example.pass_to_pass or [],
            timeout=config.test_timeout_seconds,
            max_retries=config.max_retries,
            verify_mode=config.verify_mode,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_parallel) as pool:
        return dict(pool.map(_run_one, instances))


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
                reinit="finish_previous",
            )
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B init failed — disabling W&B logging", exc_info=True)
            self._wandb_disabled = True
            return None
        self._wandb_run_id = run_id
        return self._wandb_mod

    def _link_model_lineage(self, run: EvalRun, config: EvalConfig) -> None:
        """Declare used model-checkpoint artifacts so logged eval artifacts get
        a lineage edge (old plan §8: eval artifact references the Phase 4
        ``model-qwen3-14b-{variant}`` checkpoint).  Must run BEFORE the eval
        artifact is logged; missing artifacts are warned and skipped."""
        wandb_mod = self._ensure_run(run.run_id)
        if wandb_mod is None:
            return
        for variant in sorted({r.variant for r in run.results if r.variant}):
            name = config.lora_artifact_pattern.format(variant=variant)
            try:
                wandb_mod.use_artifact(
                    f"{config.wandb_entity}/{config.wandb_project}/{name}:latest"
                )
                logger.info("lineage: eval run %s uses artifact %s", run.run_id, name)
            except Exception:  # noqa: BLE001 — W&B must never break the harness
                logger.warning(
                    "lineage: could not link artifact %s for run %s (missing checkpoint?)",
                    name,
                    run.run_id,
                )

    def log_eval_run(self, run: EvalRun, config: EvalConfig) -> None:
        """Log a full eval run: per-example, aggregate and per-repo artifacts."""
        self._link_model_lineage(run, config)
        if config.wandb_log_per_example:
            self.log_per_example(run.results, run.run_id)
        if config.wandb_log_aggregate:
            self.log_aggregate(run.aggregate, run.run_id)
        self.log_per_repo(run.aggregate, run.run_id)
        wandb_mod = self._ensure_run(run.run_id)
        if wandb_mod is None:
            return
        if run.cost_usd > 0:
            wandb_mod.log({"eval/cost_usd": run.cost_usd})
        try:
            scalars: dict[str, float] = {}
            for key, p in latency_percentiles(run.results).items():
                scalars[f"eval/{key}/latency_p50"] = p["p50"]
                scalars[f"eval/{key}/latency_p95"] = p["p95"]
            if scalars:
                wandb_mod.log(scalars)
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B latency percentile logging failed for run %s", run.run_id)

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


# Approximate Modal list rates (2026).  C3: Rate changed from A10G ($0.0167/min)
# to A100-80GB ($0.0417/min) because ``inference.py`` hardcodes A100-80GB.
# Update when config.inference_gpu switches to a different GPU tier.
_GPU_RATE_PER_MIN = 0.0417  # A100-80GB ≈ $2.50/hr (was 0.0167 for A10G)
_VCPU_RATE_PER_HOUR = 0.008  # Modal CPU
_TEST_MIN_PER_INSTANCE = 1.5  # est. wall min/instance on swebench images (repo-batched)
_VCPUS_PER_TEST_CONTAINER = 2


def latency_percentiles(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """p50/p95 of per-example latency, keyed ``model/variant/prompt`` (mirrors
    the aggregate scalar prefix).  Empty results -> ``{}``."""
    groups: dict[str, list[float]] = {}
    for r in results:
        if r.latency_seconds > 0:
            groups.setdefault(f"{r.model_name}/{r.variant}/{r.prompt_template}", []).append(
                r.latency_seconds
            )
    out: dict[str, dict[str, float]] = {}
    for key, latencies in groups.items():
        latencies = sorted(latencies)  # noqa: PLW2901
        p50 = statistics.median(latencies)
        # Nearest-rank 95th percentile (matches common latency SLOs).
        p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if len(latencies) > 1 else p50
        out[key] = {"p50": round(p50, 4), "p95": round(p95, 4)}
    return out


def estimate_run_cost(results: list[EvalResult]) -> dict[str, float]:
    """Estimate run cost: measured inference GPU time + estimated test CPU time.

    Inference is *measured* (sum of per-example ``latency_seconds``, which the
    vLLM container records); test time is an estimate because the runner does
    not yet report wall-clock (Phase 8 will make it measured too).
    """
    inference_min = sum(r.latency_seconds for r in results) / 60.0
    inference_usd = inference_min * _GPU_RATE_PER_MIN
    test_hours = (len(results) * _TEST_MIN_PER_INSTANCE) / 60.0
    test_usd = test_hours * _VCPUS_PER_TEST_CONTAINER * _VCPU_RATE_PER_HOUR
    return {
        "inference_usd": round(inference_usd, 4),
        "tests_usd": round(test_usd, 4),
        "total_usd": round(inference_usd + test_usd, 4),
    }


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

        The ``{run_id}`` placeholder in the path is the DATASET pipeline run
        id, not the eval resume id: it is substituted from
        ``config.dataset_run_id`` (EVAL_DATASET_RUN_ID), then *run_id*, then
        ``config.resume_from`` (a dataset path without the placeholder is used
        verbatim). ``split="swebench_verified"`` keeps only records whose
        ``metadata.is_verified`` is True.

        Args:
            split: ``"golden"`` (default) or ``"swebench_verified"``.
            run_id: Fallback dataset run id for ``{run_id}`` substitution
                (overridden by ``config.dataset_run_id``).

        Returns:
            Reconstructed EvalInput list, one per JSONL line.

        Raises:
            ValueError: if a placeholder is present but no run id is available,
                or for an unknown split.
        """
        run_id = self.config.dataset_run_id or run_id or self.config.resume_from
        path = self.config.golden_data_path
        if "{run_id}" in path:
            if not run_id:
                raise ValueError(
                    "golden_data_path contains a {run_id} placeholder but no run id "
                    "was provided; pass run_id=... or set EVAL_DATASET_RUN_ID"
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
                patches = _generate_patches(
                    model_name,
                    variant,
                    prompt_template,
                    [example],
                    dataset_run_id=self.config.dataset_run_id or None,
                )
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
        latency_seconds: float = 0.0,
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
            latency_seconds: Per-instance inference latency; batch mode has no
                per-instance timing, so callers amortize the single
                ``_generate_patches`` call duration (default 0.0).

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
            latency_seconds=latency_seconds,
            timestamp=datetime.now(UTC),
            error=error,
        )

    def run_batch(  # noqa: PLR0912, PLR0915, PLR0913, PLR0917
        self,
        examples: list[EvalInput],
        model_name: str,
        variant: str,
        prompt_template: str,
        run_id: str,
        max_new_tokens: int = 8192,
    ) -> list[EvalResult]:
        """Run a batch with per-repo checkpoint resume.

        Generates patches for ALL non-completed examples in a single
        ``_generate_patches`` call (one Modal container lifetime), then
        processes per-repo for tests and checkpointing.

        Args:
            max_new_tokens: Maximum generation length (per-tier C2 config).
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

        # Phase 1: identify non-completed repos
        pending_repos: dict[str, list[EvalInput]] = {}
        for repo in sorted(by_repo):
            key = self.checkpoint_mgr.get_checkpoint_key(
                run_id, repo, model_name, variant, prompt_template
            )
            if self.checkpoint_mgr.is_completed(key):
                logger.info("skipping completed repo %s (checkpoint %s)", repo, key)
                continue
            pending_repos[repo] = by_repo[repo]

        all_instances: list[EvalInput] = []
        for repo in sorted(pending_repos):
            all_instances.extend(pending_repos[repo])

        # C4: reuse generated_patches from checkpointed results where available.
        # Build a set of instance_ids that already have patches from loaded results.
        loaded_patches: dict[str, str] = {
            r.instance_id: r.generated_patch for r in loaded if r.generated_patch
        }

        # Only generate patches for instances WITHOUT cached patches
        patches_map: dict[str, str] = dict(loaded_patches)
        new_instances = [ex for ex in all_instances if ex.instance_id not in loaded_patches]
        per_instance_latency = 0.0
        if new_instances:
            _gen_start = time.monotonic()
            patches = _generate_patches(
                model_name,
                variant,
                prompt_template,
                new_instances,
                max_new_tokens=max_new_tokens,
                dataset_run_id=self.config.dataset_run_id or None,
            )
            gen_seconds = max(time.monotonic() - _gen_start, 0.0)
            per_instance_latency = gen_seconds / max(len(all_instances), 1)
            for i, example in enumerate(new_instances):
                patches_map[example.instance_id] = patches[i] if i < len(patches) else ""

        # Phase 2: test execution.  Swebench images first — ONE pool call over
        # ALL pending instances (both freshly-generated and cached patches).
        swebench_results: dict[str, dict[str, Any]] = {}
        if self.config.use_swebench_images:
            _id_instances = [ex for ex in all_instances if ex.instance_id]
            if _id_instances:
                try:
                    swebench_results = _run_tests_swebench(_id_instances, patches_map, self.config)
                except Exception:  # noqa: BLE001 — swebench path failed, fall back
                    logger.info(
                        "swebench image path unavailable — falling back to batch "
                        "for %d instance(s)",
                        len(_id_instances),
                    )

        new: list[EvalResult] = []
        results_by_repo: dict[str, list[EvalResult]] = {repo: [] for repo in pending_repos}
        for repo in sorted(pending_repos):
            repo_examples = pending_repos[repo]

            by_base: dict[str, list[EvalInput]] = {}
            for example in repo_examples:
                by_base.setdefault(example.base_sha, []).append(example)

            for base_sha in sorted(by_base):
                group = by_base[base_sha]

                test_jobs: list[dict[str, Any]] = []
                for example in group:
                    patch = patches_map.get(example.instance_id, "")
                    test_jobs.append(
                        {
                            "instance_id": example.instance_id,
                            "generated_patch": patch,
                            "fail_to_pass": example.fail_to_pass or [],
                            "pass_to_pass": example.pass_to_pass or [],
                            "test_patch": example.test_patch or "",
                            "gold_patch": example.gold_patch or "",
                        }
                    )

                missing = [ex for ex in group if ex.instance_id not in swebench_results]
                if missing:
                    _fallback_jobs = [
                        job
                        for job in test_jobs
                        if job["instance_id"] in {m.instance_id for m in missing}
                    ]
                    _fallback_results = _run_tests_batch(
                        repo,
                        base_sha,
                        group[0].test_patch or None,
                        _fallback_jobs,
                        self.config,
                    )
                    for example, fallback_result in zip(missing, _fallback_results):  # noqa: B905
                        swebench_results[example.instance_id] = fallback_result

                for example in group:
                    result = swebench_results.get(example.instance_id)
                    if result is not None:
                        eval_result = self.run_example_from_output(
                            example,
                            model_name,
                            variant,
                            prompt_template,
                            patches_map.get(example.instance_id, ""),
                            result,
                            latency_seconds=per_instance_latency,
                        )
                    else:
                        eval_result = self.run_example(
                            example,
                            model_name,
                            variant,
                            prompt_template,
                            generated_patch=patches_map.get(example.instance_id, ""),
                        )
                    results_by_repo[repo].append(eval_result)

            repo_results = results_by_repo[repo]
            new.extend(repo_results)
            key = self.checkpoint_mgr.get_checkpoint_key(
                run_id, repo, model_name, variant, prompt_template
            )
            errored = sum(1 for r in repo_results if r.error)
            if repo_results and errored == 0:
                payload: EvalResult | list[EvalResult] = (
                    repo_results[0] if len(repo_results) == 1 else repo_results
                )
                self.checkpoint_mgr.save_result(key, payload)
                logger.info(
                    "checkpointed %d result(s) for %s under %s", len(repo_results), repo, key
                )
            elif repo_results:
                logger.warning(
                    "NOT checkpointing %s: %d/%d result(s) have errors — resume will retry",
                    repo,
                    errored,
                    len(repo_results),
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
            sample: If > 0, random sample (seeded from ``tier_seed``).
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
        """Shared runner: load → sample → per-combo batches → aggregate → log.

        T1: Combos run in parallel via ThreadPoolExecutor (max_workers = number
        of combos) so gen+test for different (model, variant, template) combos
        overlap — wall time is roughly the slowest combo, not the sum of all.
        """
        import concurrent.futures

        run_id = run_id or make_run_id()
        started_at = datetime.now(UTC)
        examples = self.load_examples(split, run_id=run_id)
        if sample > 0:
            examples = random.Random(self.config.tier_seed).sample(
                examples, min(sample, len(examples))
            )
            logger.info(
                "sampled %d of %d examples (seed=%d)",
                len(examples),
                sample,
                self.config.tier_seed,
            )

        # Determine max_new_tokens from tier config by matching sample size.
        # Sorted by size ascending so the smallest enclosing tier wins.
        max_new_tokens = 8192
        if sample == 0 or sample >= 1500:  # noqa: PLR2004
            max_new_tokens = self.config.tier_max_new_tokens.get("full", 2048)
        else:
            for _tier, _size in sorted(self.config.tier_sizes.items(), key=lambda x: x[1]):
                if _size > 0 and sample <= _size:
                    max_new_tokens = self.config.tier_max_new_tokens.get(_tier, 1024)
                    break

        # T1: run all (model, variant, template) combos in parallel.
        combos = [
            (model_name, variant, template)
            for model_name, variant in models
            for template in prompt_templates
        ]
        all_results: list[list[EvalResult]] = [None] * len(combos)  # type: ignore[list-item]

        def _run_combo(idx: int) -> None:
            model_name, variant, template = combos[idx]
            all_results[idx] = self.run_batch(
                examples,
                model_name,
                variant,
                template,
                run_id,
                max_new_tokens=max_new_tokens,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(combos)) as pool:
            # ponytail: consume the lazy iterator so exceptions propagate
            list(pool.map(_run_combo, range(len(combos))))

        results: list[EvalResult] = []
        for batch_results in all_results:
            if batch_results:
                results.extend(batch_results)
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
            cost_usd=estimate_run_cost(results)["total_usd"],
        )
        _persist_run(run, self.config)
        try:
            self.wandb_logger.log_eval_run(run, self.config)
        except Exception:  # noqa: BLE001 — W&B must never break the harness
            logger.warning("W&B logging failed for run %s", run_id, exc_info=True)
        logger.info(
            "run %s (%s): %d results, status=%s, est. cost=$%.2f",
            run_id,
            split,
            len(results),
            run.status,
            run.cost_usd,
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
