#!/usr/bin/env python3
"""Phase 4H: 3-Config QLoRA Comparison & Champion Selection.

Orchestrates training for all 3 QLoRA variants on Modal, computes proxy
F2P on the golden set, and promotes the champion to the W&B Registry.

Sleep-resilient: launches via ``modal run`` subprocess + polls W&B for completion
so training continues on Modal servers even if laptop sleeps. Re-run the same
command to resume from where it left off.

Usage:
    # Dry-run (validate orchestration logic, no Modal)
    python scripts/run_3config_comparison.py --dry-run --run-id 25d3f8fd0ccb

    # Real run (requires Modal + W&B credentials)
    python scripts/run_3config_comparison.py --run-id 25d3f8fd0ccb
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_RELPATH = "swebench/golden.jsonl"
_STATE_PATH = _REPO_ROOT / "scripts" / ".pipeline-state.json"
_POLL_INTERVAL = 60  # seconds between polling for Modal job completion
_POLL_TIMEOUT = 6 * 3600  # 6 hours max poll time (Modal function timeout is 5h)
_MIN_POLL_WAIT = 120  # seconds before declaring a launch failed (W&B init takes time)
_WANDB_API_RETRIES = 10  # retry count for transient W&B service failures
_WANDB_RETRY_DELAY = 30  # seconds between W&B API retries


# ── W&B entity resolution ────────────────────────────────────────────────────


def _resolve_wandb_entity() -> str:
    """Resolve W&B entity from API credentials."""
    import wandb

    api = wandb.Api(timeout=30)
    entity = api.default_entity
    if not entity:
        raise RuntimeError("W&B entity not found. Run `wandb login` to set credentials.")
    return entity


def _wandb_project_entity() -> str:
    """Return 'entity/project' for W&B API calls."""
    return f"{_resolve_wandb_entity()}/swe-qwen"


# Known artifact name pattern — deterministic from variant name
def _artifact_name(variant: str) -> str:
    return f"model-qwen3-14b-{variant}"


# ── GCS buckets (same bucket/dataset as modal_train.py) ──────────────────────
# golden.jsonl is archived alongside the pipeline data; download at runtime
# so we don't need to manage a local copy.
_GCS_BUCKET = "swe-qwen-datasets"


def _gcs_golden_prefix(run_id: str) -> str:
    """Construct GCS prefix for golden data given a pipeline run ID."""
    return f"datasets/{run_id}/swebench/"


# ── State persistence (sleep-resilient) ────────────────────────────────────────


def _load_state(run_id: str) -> dict[str, Any]:
    """Load pipeline state, returning a fresh state if none exists or run_id changed."""
    if _STATE_PATH.exists():
        with _STATE_PATH.open("r") as f:
            state = json.load(f)
        if state.get("run_id") == run_id:
            return state
        print(f"  State file found for different run_id ({state.get('run_id')}), starting fresh.")
    return _new_state(run_id)


def _new_state(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "completed_variants": [],
        "variants": {},  # variant_name -> {status, run_name, handle_id, result}
        "last_poll": 0.0,
    }


def _save_state(state: dict[str, Any]) -> None:
    state["last_poll"] = time.time()
    with _STATE_PATH.open("w") as f:
        json.dump(state, f, indent=2)
    _STATE_PATH.chmod(0o644)


def _cleanup_state(*_args: Any) -> None:
    """Remove state file on clean exit (successful completion only)."""
    if _STATE_PATH.exists():
        _STATE_PATH.unlink(missing_ok=True)


def _reconcile_state_with_wandb(  # noqa: PLR0915
    state: dict[str, Any],
    requested_variants: list[str] | None = None,
    retrain_variants: list[str] | None = None,
) -> dict[str, Any]:
    """Reconcile local state with W&B to recover from interrupted sessions.

    Scans W&B for runs matching known variant names and updates state accordingly.
    Also detects crashed runs and marks them for re-launch.

    When *requested_variants* is provided, also scans for variants not yet in
    state (fresh start after state file deletion) — avoids re-training completed variants.

    *retrain_variants* are cleared from state and excluded from reconciliation,
    so stale W&B runs can't be mistaken for fresh completions.
    """
    import wandb

    api = wandb.Api(timeout=30)
    project = _wandb_project_entity()

    try:
        all_runs = api.runs(project)
    except Exception as e:
        logger.warning("Failed to fetch W&B runs for reconciliation: %s", e)
        return state

    # Variants explicitly requested for retraining: drop any prior completion
    # state so the train loop re-launches them (stale W&B runs get ignored).
    if retrain_variants:
        for variant in retrain_variants:
            state["completed_variants"] = [v for v in state["completed_variants"] if v != variant]
            state["variants"].pop(variant, None)
            logger.info("  %s: forced retrain (ignoring previous W&B state)", variant)

    # Build index: variant -> list of runs (sorted by creation time, newest first)
    variant_runs: dict[str, list[Any]] = {}
    for run in all_runs:
        variant = run.config.get("variant")
        if not variant:
            continue
        if variant not in variant_runs:
            variant_runs[variant] = []
        variant_runs[variant].append(run)

    # Sort each variant's runs by creation time (newest first)
    for runs in variant_runs.values():
        runs.sort(key=lambda r: r.created_at, reverse=True)

    # Collect all variants to reconcile: existing state + requested variants
    variants_to_check = set(state["variants"].keys())
    if requested_variants:
        variants_to_check.update(requested_variants)

    def _match_and_update(variant: str) -> None:  # noqa: PLR0912
        """Find best W&B run for variant and update state."""
        if variant not in variant_runs:
            return

        runs = variant_runs[variant]
        variant_state = state["variants"].get(variant, {})
        run_name = variant_state.get("run_name", "")

        # Try exact match first (saved run_name or saved wandb_run_id)
        matched_run = None
        for r in runs:
            if r.name == run_name or r.id == (variant_state.get("result", {}) or {}).get(
                "wandb_run_id"
            ):
                matched_run = r
                break

        if matched_run is None:
            # Fallback: latest 3config- run for this variant
            for r in runs:
                if r.name.startswith("3config-"):
                    matched_run = r
                    break
            if matched_run is None:
                matched_run = runs[0]

        if matched_run:
            if matched_run.state == "finished":
                if variant not in state["completed_variants"]:
                    state["completed_variants"].append(variant)
                state["variants"][variant] = {
                    "status": "completed",
                    "run_name": matched_run.name,
                    "result": {
                        "wandb_run_id": matched_run.id,
                        "artifact_name": _artifact_name(variant),
                    },
                }
                logger.info("  Reconciled %s: finished (W&B run %s)", variant, matched_run.id)
            elif matched_run.state == "crashed":
                state["variants"][variant] = {
                    "status": "failed",
                    "run_name": matched_run.name,
                }
                logger.warning("  %s: W&B run %s crashed — will re-launch", variant, matched_run.id)
            elif matched_run.state == "running":
                if "train_loss" in matched_run.summary:
                    if variant not in state["completed_variants"]:
                        state["completed_variants"].append(variant)
                    state["variants"][variant] = {
                        "status": "completed",
                        "run_name": matched_run.name,
                        "result": {
                            "wandb_run_id": matched_run.id,
                            "artifact_name": _artifact_name(variant),
                        },
                    }
                    logger.info(
                        "  Reconciled %s: training complete (Modal killed before wandb.finish)",
                        variant,
                    )
                else:
                    state["variants"][variant] = {
                        "status": "running",
                        "run_name": matched_run.name,
                    }
                    logger.info(
                        "  %s: still running on Modal (W&B run %s)", variant, matched_run.id
                    )

    for variant in variants_to_check:
        if retrain_variants and variant in retrain_variants:
            continue
        _match_and_update(variant)

    return state


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-Config QLoRA comparison runner (sleep-resilient)",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Phase 3 run ID (subdirectory under data/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate orchestration without launching Modal jobs",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=["baseline_14b", "higher_rank_14b", "higher_lr_14b"],
        help="Variants to compare (default: all 3 mandatory 14B variants)",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Subsample to N rows per variant (for smoke-testing the pipeline)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for summary JSON (default: comparison-report.json in repo root)",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip F2P evaluation and champion selection (training only)",
    )
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore W&B state, retrain all variants from scratch",
    )
    p.add_argument(
        "--retrain-variants",
        nargs="+",
        default=None,
        help="Force retrain ONLY these variants (e.g. --retrain-variants "
        "higher_rank_14b higher_lr_14b). Clears their completion state and "
        "skips W&B reconciliation for them, so stale runs can't be reused.",
    )
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _variant_run_name(variant: str) -> str:
    return f"3config-{variant}-{time.strftime('%Y%m%d-%H%M%S')}"


def _tail_log(path: Path, n: int = 50) -> str:
    """Return last N lines of a log file (safe for partially-written files)."""
    try:
        lines = path.read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(unable to read log)"


def _golden_path(run_id: str) -> Path:
    return _REPO_ROOT / "data" / run_id / _GOLDEN_RELPATH


def _ensure_golden(run_id: str) -> Path:
    """Ensure golden.jsonl is available at ``data/{run_id}/swebench/golden.jsonl``.

    Downloads from the public GCS bucket if not already present locally.
    The golden set is the same for all training runs — it's the benchmark eval
    set archived alongside the pipeline data on GCS.
    """
    dst = _golden_path(run_id)
    if dst.is_file():
        return dst

    golden_prefix = _gcs_golden_prefix(run_id)
    print(f"  Downloading golden.jsonl from GCS (gs://{_GCS_BUCKET}/{golden_prefix}) ...")
    import json
    import shutil
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)

    # List objects under the datasets prefix (should only be golden.jsonl)
    list_url = f"https://www.googleapis.com/storage/v1/b/{_GCS_BUCKET}/o?prefix={golden_prefix}"
    with urllib.request.urlopen(list_url) as resp:
        payload = json.loads(resp.read().decode())

    items = payload.get("items", [])
    if not items:
        print(
            f"Error: no files found at gs://{_GCS_BUCKET}/{golden_prefix}\n"
            "       Check that the bucket and dataset run ID are correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    for obj in items:
        name: str = obj["name"]
        rel_path = name[len(golden_prefix) :].lstrip("/")
        if not rel_path:
            continue
        dst_file = dst.parent / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        media_link: str = obj.get("mediaLink") or obj.get("selfLink", "")
        print(f"  Downloading {name} -> {dst_file} ...")
        with urllib.request.urlopen(media_link) as src:
            if rel_path.endswith(".jsonl"):
                content = src.read().decode()
                dst_file.write_text(content)
            else:
                with dst_file.open("wb") as f:
                    shutil.copyfileobj(src, f)

    if not dst.is_file():
        print(
            f"Error: golden.jsonl not found in GCS at prefix {golden_prefix}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Golden: {dst} ({dst.stat().st_size / 1_000_000:.0f} MB)")
    return dst


# ── Resilient training launcher (spawn + poll + state) ────────────────────────


def _build_modal_cmd(variant: str, run_name: str, train_kwargs: dict[str, Any]) -> list[str]:
    """Build ``modal run`` CLI command for a training variant."""
    cmd = [
        "modal",
        "run",
        f"{_REPO_ROOT / 'training/modal_train.py'}::train_qlora",
        "--variant",
        variant,
        "--run-name",
        run_name,
    ]
    for key, val in train_kwargs.items():
        if val is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def _wandb_run_finished(run_name: str) -> dict[str, Any] | None:
    """Check W&B for a completed run matching *run_name*.

    Returns the run summary (including artifact name and wandb run id)
    if complete, ``None`` if still running or not found.
    Raises ``RuntimeError`` if the run is found but crashed.

    Transient W&B service failures (e.g. "service process is busy" during
    ``Api()`` verification) are retried with backoff. If the service stays
    down, returns ``None`` so the caller defers to the next poll cycle
    instead of failing the variant.
    """
    import wandb

    api = None
    for attempt in range(1, _WANDB_API_RETRIES + 1):
        try:
            api = wandb.Api(timeout=30)
            break
        except Exception as e:  # transient: wedged local service, network blip
            logger.warning(
                "  W&B API down (attempt %d/%d): %s — retrying in %ds",
                attempt,
                _WANDB_API_RETRIES,
                type(e).__name__,
                _WANDB_RETRY_DELAY,
            )
            if attempt < _WANDB_API_RETRIES:
                time.sleep(_WANDB_RETRY_DELAY)
    if api is None:
        logger.warning(
            "  W&B API unavailable after %d attempts — deferring to next poll",
            _WANDB_API_RETRIES,
        )
        return None

    entity = api.default_entity
    if not entity:
        raise RuntimeError("W&B entity not found. Run `wandb login` to set credentials.")
    try:
        runs = api.runs(f"{entity}/swe-qwen", {"display_name": run_name})
    except Exception as e:
        logger.warning("  W&B query failed: %s — deferring to next poll", e)
        return None

    for run in runs:
        if run.state == "finished":
            return {
                "wandb_run_id": run.id,
                "artifact_name": _artifact_name(run.config.get("variant", "")),
            }
        # Catch runs where training completed but Modal killed the container
        # before wandb.finish() was called (state stays "running" indefinitely).
        # Presence of train_loss in the summary confirms training finished.
        if run.state == "running" and "train_loss" in run.summary:
            return {
                "wandb_run_id": run.id,
                "artifact_name": _artifact_name(run.config.get("variant", "")),
            }
        # Surface crashed runs so caller can re-launch
        if run.state == "crashed":
            raise RuntimeError(
                f"W&B run '{run_name}' ({run.id}) crashed. Will re-launch with new run name."
            )
    return None


def launch_modal_training(  # noqa: PLR0913, PLR0915, PLR0917
    variant: str,
    run_id: str,
    run_name: str,
    state: dict[str, Any],
    train_kwargs: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Launch (or resume) a Modal training job.

    Uses ``subprocess`` to run ``modal run`` — the Modal job continues on
    Modal servers even if the local process dies (laptop sleep).  Polls W&B
    for completion.  State persisted to ``.pipeline-state.json`` for resume.

    Includes timeout to detect stuck launches (Modal job failed before W&B init).
    Captures Modal stdout/stderr to a log file for debugging.

    Returns ``{"wandb_run_id": str, "artifact_name": str}``.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would launch variant={variant} run_name={run_name}")
        return {
            "wandb_run_id": f"dry-run-{variant}",
            "artifact_name": _artifact_name(variant),
        }

    variant_state = state["variants"].get(variant, {})
    saved_run_name = variant_state.get("run_name")

    # ── Check W&B for completion (resume after sleep) ────────────────────
    if saved_run_name and variant_state.get("status") in ("launched", "running"):
        try:
            finished = _wandb_run_finished(saved_run_name)
        except RuntimeError as e:
            # Previous run crashed — will re-launch below
            logger.warning("  %s: %s", variant, e)
            finished = None
        if finished:
            print(f"  {variant} already complete (verified via W&B).")
            _mark_completed(state, variant, finished["wandb_run_id"], finished["artifact_name"])
            return finished
        print(f"  {variant} W&B run not finished yet. Re-launching ...")

    # ── Check if variant was marked failed from reconciliation ────────────
    if variant_state.get("status") == "failed":
        logger.warning("  %s: marked failed from previous run, re-launching ...", variant)

    # ── Launch new job ────────────────────────────────────────────────────
    cmd = _build_modal_cmd(variant, run_name, train_kwargs or {})
    log_path = _REPO_ROOT / "logs" / f"modal-{variant}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Launching: {' '.join(cmd)}")
    print(f"  Log file:  {log_path}")
    print("  (Job runs on Modal servers — safe to close laptop)")

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    state["variants"][variant] = {
        "status": "launched",
        "run_name": run_name,
        "log_file": str(log_path),
        "pid": process.pid,
    }
    _save_state(state)

    # ── Poll W&B for completion (with timeout) ──────────────────────────
    print(f"  Polling W&B every {_POLL_INTERVAL}s for run '{run_name}' ...")
    state["variants"][variant]["status"] = "running"
    _save_state(state)

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time

        # Hard timeout — if W&B run doesn't appear after _MIN_POLL_WAIT,
        # check Modal process exit code to detect silent failures
        if elapsed > _MIN_POLL_WAIT:
            retcode = process.poll()
            if retcode is not None and retcode != 0:
                raise RuntimeError(
                    f"Modal process exited with code {retcode} for variant '{variant}'. "
                    f"Check log: {log_path}\n"
                    f"Last 50 lines:\n"
                    f"{_tail_log(log_path, 50)}"
                )
            if elapsed > _POLL_TIMEOUT:
                raise RuntimeError(
                    f"Poll timeout ({_POLL_TIMEOUT}s) exceeded for variant '{variant}'. "
                    f"W&B run '{run_name}' never appeared. "
                    f"Check log: {log_path}"
                )

        try:
            finished = _wandb_run_finished(run_name)
        except RuntimeError as e:
            # Run crashed on W&B — fail fast
            raise RuntimeError(f"Variant '{variant}': {e}") from e

        if finished:
            print(f"  {variant} complete (detected via W&B, {elapsed:.0f}s).")
            _mark_completed(state, variant, finished["wandb_run_id"], finished["artifact_name"])
            return finished

        print(f"  {variant} still training ... ({time.strftime('%H:%M:%S')})")
        state["variants"][variant]["status"] = "running"
        _save_state(state)

        try:
            time.sleep(_POLL_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n  Interrupted — {variant} continues on Modal servers.")
            print(f"  Re-run with --run-id {state.get('run_id')} to resume.")
            _save_state(state)
            sys.exit(130)


def _mark_completed(
    state: dict[str, Any],
    variant: str,
    wandb_run_id: str | None,
    artifact_name: str,
) -> None:
    """Update state to completed and persist."""
    state["variants"][variant].update(
        {
            "status": "completed",
            "result": {
                "wandb_run_id": wandb_run_id,
                "artifact_name": artifact_name,
            },
        }
    )
    if variant not in state["completed_variants"]:
        state["completed_variants"].append(variant)
    _save_state(state)
    print(f"  {variant} complete → W&B run: {wandb_run_id}")


# ── Download, evaluate, promote ────────────────────────────────────────────────


def download_adapter(
    artifact_name: str,
    output_dir: Path,
    dry_run: bool = False,
) -> str:
    """Download the trained adapter from W&B Artifacts."""
    if dry_run:
        local_path = str(output_dir / artifact_name)
        print(f"  [DRY-RUN] Would download W&B artifact {artifact_name} to {local_path}")
        return local_path

    import wandb

    api = wandb.Api()
    project = _wandb_project_entity()
    artifact = api.artifact(f"{project}/{artifact_name}:latest")
    local_dir = str(output_dir / artifact_name)
    artifact.download(root=local_dir)
    return local_dir


def evaluate_proxy_f2p(
    variant: str,
    golden_path: Path,
    adapter_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run proxy F2P evaluation for one variant."""
    if dry_run:
        return {"variant": variant, "mean_f2p": 0.0}

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "f2p_proxy",
        _REPO_ROOT / "scripts" / "f2p_proxy.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load f2p_proxy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scores = mod.compute_proxy_f2p_scores(
        golden_path=golden_path,
        variant_adapter_map={variant: adapter_path},
    )
    return scores[variant]


def promote_champion(
    champion_variant: str,
    champion_artifact_name: str,
    dry_run: bool = False,
) -> None:
    """Tag the champion adapter as ``champion`` in W&B Registry."""
    if dry_run:
        print(
            f"  [DRY-RUN] Would promote {champion_variant} ({champion_artifact_name}) "
            "to W&B Registry champion alias"
        )
        return

    import wandb

    api = wandb.Api()
    project = _wandb_project_entity()
    artifact = api.artifact(f"{project}/{champion_artifact_name}:latest")
    artifact.aliases.append("champion")
    artifact.save()
    print(f"  Promoted {champion_variant} ({champion_artifact_name}) → champion alias")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: PLR0912, PLR0915 — 63 stmts for sequential orchestration logic
    args = parse_args()

    golden_path = _ensure_golden(args.run_id)

    print("Phase 4H: 3-Config Comparison (sleep-resilient mode)")
    print(f"  Run ID:    {args.run_id}")
    print(f"  Golden:    {golden_path}")
    print(f"  Variants:  {', '.join(args.variants)}")
    print(f"  Mode:      {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"  State:     {_STATE_PATH}")
    print()

    # Register signal handlers for graceful Ctrl-C (preserve state for resume)
    signal.signal(
        signal.SIGINT,
        lambda *_: (print("\nInterrupted — state preserved for resume."), sys.exit(130)),
    )
    signal.signal(
        signal.SIGTERM,
        lambda *_: (print("\nTerminated — state preserved for resume."), sys.exit(143)),
    )

    output_dir = _REPO_ROOT / "models" / "comparisons" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or init state ────────────────────────────────────────────────
    state = _load_state(args.run_id)

    # ── Reconcile with W&B (recover from interrupted sessions) ────────────
    if not args.dry_run and not args.force_retrain:
        print("  Reconciling state with W&B ...")
        state = _reconcile_state_with_wandb(
            state,
            requested_variants=args.variants,
            retrain_variants=args.retrain_variants,
        )
        _save_state(state)
    elif args.force_retrain:
        state["completed_variants"] = []
        print("  Force mode: ignoring previous completions, will retrain all variants")
    if state.get("completed_variants"):
        print(f"  Completed: {', '.join(state['completed_variants'])}")
    print()

    # Common kwargs passed to every train_qlora call
    train_kwargs = {
        "model_name": "qwen3-14b",
        "run_id": args.run_id,
        "data_dir": "/data/tokenized",
        "gpu_type": None,  # auto-resolve from models.yaml → A100-80GB
    }
    if args.max_train_samples:
        train_kwargs["max_train_samples"] = args.max_train_samples
        print(f"  Smoke mode: {args.max_train_samples} samples per variant")
        print()

    # ── Step 1: Train all variants sequentially ───────────────────────────
    results: dict[str, dict[str, Any]] = {}
    failed_variants: dict[str, str] = {}  # variant -> error message

    for variant in args.variants:
        if variant in state.get("completed_variants", []):
            print(f"[{variant}] Already completed, skipping. ✓")
            previous_result = state["variants"].get(variant, {}).get("result", {})
            results[variant] = {
                "wandb_run_id": previous_result.get("wandb_run_id"),
                "artifact_name": previous_result.get("artifact_name", _artifact_name(variant)),
            }
            continue

        print(f"[{variant}]")
        run_name = _variant_run_name(variant)

        try:
            launch_info = launch_modal_training(
                variant, args.run_id, run_name, state, train_kwargs, args.dry_run
            )
            wandb_run_id = launch_info["wandb_run_id"]
            artifact_name = launch_info["artifact_name"]

            adapter_path = download_adapter(artifact_name, output_dir, args.dry_run)
            f2p_result = evaluate_proxy_f2p(variant, golden_path, adapter_path, args.dry_run)

            results[variant] = {
                "run_name": run_name,
                "wandb_run_id": wandb_run_id,
                "artifact_name": artifact_name,
                "adapter_path": adapter_path,
                **f2p_result,
            }
            print(f"  mean_f2p={f2p_result['mean_f2p']}")
            print(f"  ✓ {variant} complete")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ {variant} failed: {error_msg}")
            failed_variants[variant] = error_msg
            # Don't abort — try remaining variants
        print()

    # Report failures before proceeding
    if failed_variants:
        print("Failed variants:")
        for v, err in failed_variants.items():
            print(f"  {v}: {err[:200]}")
        print()

    # ── Load f2p_proxy module (used for both F2P scoring and champion selection) ──
    select_champion = None
    if not args.skip_eval and results:
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "f2p_proxy",
            _REPO_ROOT / "scripts" / "f2p_proxy.py",
        )
        if _spec is None or _spec.loader is None:
            raise RuntimeError("Failed to load f2p_proxy.py")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        select_champion = _mod.select_champion

        # ── Compute F2P for ALL completed variants in ONE batch ──
        print("Computing F2P scores for all completed variants ...")

        # f2p_proxy queries W&B directly; adapter paths not needed for scoring.
        # Pass all variant names in a single call so normalization is consistent.
        all_variants = dict.fromkeys(results, "")
        all_f2p = _mod.compute_proxy_f2p_scores(golden_path, all_variants)

        for variant, f2p in all_f2p.items():
            if variant in results:
                results[variant].update(f2p)
                loss = f2p.get("train_loss", "?")
                print(f"  {variant}: F2P={f2p['mean_f2p']} (loss={loss})")
        print()

    # ── Step 2: Select champion ────────────────────────────────────────────
    if args.skip_eval:
        print("  Skipping evaluation and champion selection (--skip-eval).")
        champion = None
        champion_artifact_name = None
    elif not results:
        print("No completed variants — cannot select champion.")
        champion = None
        champion_artifact_name = None
    else:
        assert select_champion is not None  # guaranteed by if not args.skip_eval and results
        champion = select_champion(results)
        champion_artifact_name = results[champion]["artifact_name"]
        print(f"Champion: {champion} (F2P={results[champion]['mean_f2p']})")

    # ── Step 3: Promote champion ───────────────────────────────────────────
    if champion and champion_artifact_name:
        promote_champion(champion, champion_artifact_name, args.dry_run)

    # ── Step 4: Output summary ─────────────────────────────────────────────
    summary = {
        "run_id": args.run_id,
        "dry_run": args.dry_run,
        "golden_path": str(golden_path),
        "variants": results,
        "failed_variants": failed_variants,
        "champion": champion,
        "champion_f2p": results[champion]["mean_f2p"] if champion else None,
    }
    out_path = args.output or (_REPO_ROOT / "comparison-report.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")

    # Cleanup state on successful completion
    _cleanup_state()


if __name__ == "__main__":
    main()
