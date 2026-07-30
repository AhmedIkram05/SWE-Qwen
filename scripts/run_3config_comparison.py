#!/usr/bin/env python3
"""Phase 4H: 3-Config QLoRA Comparison & Champion Selection.

Orchestrates training for all 3 QLoRA variants on Modal, computes proxy
F2P on the golden set, and promotes the champion to the W&B Registry.

Sleep-resilient: uses ``Modal.spawn()`` + persistent state file so training
continues on Modal even if your laptop goes to sleep. Re-run the same
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
import signal
import sys
import time
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_RELPATH = "swebench/golden.jsonl"
_STATE_PATH = _REPO_ROOT / "scripts" / ".pipeline-state.json"
_POLL_INTERVAL = 60  # seconds between polling for Modal job completion


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
    """Remove state file on clean exit (Ctrl+C or completion)."""
    if _STATE_PATH.exists():
        _STATE_PATH.unlink(missing_ok=True)


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
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _variant_run_name(variant: str) -> str:
    return f"3config-{variant}-{time.strftime('%Y%m%d-%H%M%S')}"


def _golden_path(run_id: str) -> Path:
    return _REPO_ROOT / "data" / run_id / _GOLDEN_RELPATH


def _resolve_golden_path(run_id: str) -> Path:
    """Resolve golden eval set for *run_id*.

    Checks three locations in order:
    1. ``data/{run_id}/swebench/golden.jsonl`` (exact match)
    2. ``data/{run_id}/swebench/cleaned.jsonl`` (fallback within exact dir)
    3. Any existing ``data/*/swebench/golden.jsonl`` (golden is same for all runs)

    Exits with an error message if none are found.
    """
    # 1. Exact match
    exact = _golden_path(run_id)
    if exact.is_file():
        return exact

    # 2. Fallback within exact dir
    exact_dir = _REPO_ROOT / "data" / run_id
    fallback = exact_dir / "swebench/cleaned.jsonl"
    if fallback.is_file():
        print(f"  (using cleaned.jsonl under {exact_dir})")
        return fallback

    # 3. Scan any existing data dir with golden.jsonl
    data_root = _REPO_ROOT / "data"
    for candidate in sorted(data_root.iterdir()):
        if not candidate.is_dir():
            continue
        candidate_golden = candidate / _GOLDEN_RELPATH
        if candidate_golden.is_file():
            print(f"  Run dir {exact_dir} not found; using golden from {candidate_golden}")
            return candidate_golden

    print(
        f"Error: no golden eval set found under {data_root}/<run_id>/swebench/.\n"
        "       Generate the dataset first with Phase 2/3 pipeline, or "
        "symlink an existing run:\n"
        f"         mkdir -p data/{run_id} && ln -s ../<existing>/swebench data/{run_id}/swebench",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Resilient training launcher (spawn + poll + state) ────────────────────────


def launch_modal_training(  # noqa: PLR0913, PLR0917 — 6 justified params for Modal spawn orchestration
    variant: str,
    run_id: str,
    run_name: str,
    state: dict[str, Any],
    train_kwargs: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Launch (or resume) a Modal training job for *variant*.

    Uses ``modal_entrypoint.spawn()`` so the Modal job continues running even
    if this script is interrupted or the laptop sleeps.  State is persisted to
    ``.pipeline-state.json`` — re-run the same command to resume.

    Returns ``{"wandb_run_id": str, "artifact_name": str}``.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would spawn variant={variant} run_name={run_name}")
        return {
            "wandb_run_id": f"dry-run-{variant}",
            "artifact_name": f"model-qwen3-14b-{variant}",
        }

    # ── Check existing state for a running/spawned job ─────────────────────
    variant_state = state["variants"].get(variant, {})
    handle_id = variant_state.get("handle_id")
    existing_status = variant_state.get("status")

    if handle_id and existing_status in ("launched", "running"):
        from modal import FunctionCall

        print(f"  Resuming {variant} from saved handle {handle_id} ...")
        call = FunctionCall.from_id(handle_id)
    else:
        # Look up the Modal function (must have been built with `modal run` at least once).
        # Spawn a new job — does NOT block.
        from modal import Function

        print(f"  Spawning {variant} on Modal (A100-80GB) ...")
        f = Function.from_name("swe-qwen-training-v2", "train_qlora")
        kwargs = dict(train_kwargs) if train_kwargs else {}
        kwargs["variant"] = variant
        kwargs["run_name"] = run_name
        call = f.spawn(**kwargs)

        # Persist handle immediately so we can resume if interrupted
        handle_id = call.object_id
        state["variants"][variant] = {
            "status": "launched",
            "run_name": run_name,
            "handle_id": handle_id,
        }
        _save_state(state)
        print(f"  Handle saved: {handle_id}")
        print(f"  To monitor:   modal app logs swe-qwen-training-{variant}")

    state["variants"][variant]["status"] = "running"
    _save_state(state)

    # ── Poll until completion (sleep-resilient) ────────────────────────────
    from modal.exception import TimeoutError

    while True:
        try:
            result: dict = call.get(timeout=_POLL_INTERVAL)
            break  # success — result is the train_qlora() return dict
        except TimeoutError:
            print(f"  {variant} still training ... (polled at {time.strftime('%H:%M:%S')})")
            state["variants"][variant]["status"] = "running"
            _save_state(state)
        except Exception as exc:
            # Likely a connection error (laptop sleep / network blip).
            # The Modal job keeps running; retry after a brief wait.
            print(f"  Connection issue: {exc}")
            print(f"  Retrying in {_POLL_INTERVAL}s (job continues on Modal)...")
            time.sleep(_POLL_INTERVAL)

    # ── Training completed ─────────────────────────────────────────────────
    wandb_run_id = result.get("wandb_run_id")
    artifact_name = result.get("artifact_name", f"model-qwen3-14b-{variant}")

    state["variants"][variant].update(
        {
            "status": "completed",
            "result": {
                "wandb_run_id": wandb_run_id,
                "artifact_name": artifact_name,
            },
        }
    )
    state["completed_variants"].append(variant)
    _save_state(state)

    print(f"  {variant} complete → W&B run: {wandb_run_id}")
    return {"wandb_run_id": wandb_run_id, "artifact_name": artifact_name}


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
    artifact = api.artifact(f"swe-qwen/{artifact_name}:latest")
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
    artifact = api.artifact(f"swe-qwen/{champion_artifact_name}:latest")
    artifact.aliases.append("champion")
    artifact.save()
    print(f"  Promoted {champion_variant} ({champion_artifact_name}) → champion alias")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: PLR0915 — 63 stmts for sequential orchestration logic
    args = parse_args()

    golden_path = _resolve_golden_path(args.run_id)

    print("Phase 4H: 3-Config Comparison (sleep-resilient mode)")
    print(f"  Run ID:    {args.run_id}")
    print(f"  Golden:    {golden_path}")
    print(f"  Variants:  {', '.join(args.variants)}")
    print(f"  Mode:      {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"  State:     {_STATE_PATH}")
    print()

    # Register signal handlers for graceful Ctrl-C
    signal.signal(signal.SIGINT, lambda *_: (_cleanup_state(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup_state(), sys.exit(143)))

    output_dir = _REPO_ROOT / "models" / "comparisons" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or init state ────────────────────────────────────────────────
    state = _load_state(args.run_id)
    if state.get("completed_variants"):
        print(f"  Resuming — already completed: {', '.join(state['completed_variants'])}")
    print()

    # Common kwargs passed to every train_qlora call
    train_kwargs = {
        "model_name": "qwen3-14b",
        "data_dir": "/data/tokenized",
        "gpu_type": None,  # auto-resolve from models.yaml → A100-80GB
    }
    if args.max_train_samples:
        train_kwargs["max_train_samples"] = args.max_train_samples
        print(f"  Smoke mode: {args.max_train_samples} samples per variant")
        print()

    # ── Step 1: Train all variants sequentially ───────────────────────────
    results: dict[str, dict[str, Any]] = {}
    for variant in args.variants:
        if variant in state.get("completed_variants", []):
            print(f"[{variant}] Already completed, skipping. ✓")
            previous_result = state["variants"].get(variant, {}).get("result", {})
            results[variant] = {"wandb_run_id": previous_result.get("wandb_run_id")}
            continue

        print(f"[{variant}]")
        run_name = _variant_run_name(variant)

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
        print()

    # ── Step 2: Select champion ────────────────────────────────────────────
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "f2p_proxy",
        _REPO_ROOT / "scripts" / "f2p_proxy.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    select_champion = _mod.select_champion

    champion = select_champion(results)
    champion_artifact_name = results[champion]["artifact_name"]
    print(f"Champion: {champion} (F2P={results[champion]['mean_f2p']})")

    # ── Step 3: Promote champion ───────────────────────────────────────────
    promote_champion(champion, champion_artifact_name, args.dry_run)

    # ── Step 4: Output summary ─────────────────────────────────────────────
    summary = {
        "run_id": args.run_id,
        "dry_run": args.dry_run,
        "golden_path": str(golden_path),
        "variants": results,
        "champion": champion,
        "champion_f2p": results[champion]["mean_f2p"],
    }
    out_path = args.output or (_REPO_ROOT / "comparison-report.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")

    # Cleanup state on successful completion
    _cleanup_state()


if __name__ == "__main__":
    main()
