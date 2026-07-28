#!/usr/bin/env python3
"""Phase 4H: 3-Config QLoRA Comparison & Champion Selection.

Orchestrates training for all 3 QLoRA variants on Modal, computes proxy
F2P on the golden set, and promotes the champion to the W&B Registry.

Usage:
    # Dry-run (validate orchestration logic, no Modal)
    python scripts/run_3config_comparison.py --dry-run --run-id 25d3f8fd0ccb

    # Real run (requires Modal + W&B credentials)
    python scripts/run_3config_comparison.py --run-id 25d3f8fd0ccb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_RELPATH = "swebench/golden.jsonl"

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-Config QLoRA comparison runner",
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
        default=["baseline", "higher_rank", "higher_lr"],
        help="Variants to compare (default: all 3)",
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


# ── Training launcher ─────────────────────────────────────────────────────────


def launch_modal_training(
    variant: str,
    run_id: str,
    run_name: str,
    dry_run: bool = False,
) -> str:
    """Launch a Modal training job for *variant*.

    Returns the W&B run ID (mocked in dry-run mode).
    """
    if dry_run:
        print(f"  [DRY-RUN] Would launch variant={variant} run_name={run_name}")
        return f"dry-run-{variant}"

    # Import only when actually running on Modal
    from training.modal_train import modal_entrypoint

    print(f"  Launching {variant} on Modal ...")
    # Modal .remote() call — blocks until completion
    wandb_run_id = modal_entrypoint.remote(
        model_name="qwen3-30b-a3b",
        variant=variant,
        data_dir=f"data/{run_id}",
        run_name=run_name,
    )
    return wandb_run_id


def download_adapter(
    wandb_run_id: str,
    output_dir: Path,
    dry_run: bool = False,
) -> str:
    """Download the trained adapter from W&B Artifacts.

    Returns the local path to the adapter directory.
    """
    if dry_run:
        local_path = str(output_dir / f"adapter-{wandb_run_id}")
        print(f"  [DRY-RUN] Would download adapter to {local_path}")
        return local_path

    import wandb

    api = wandb.Api()
    artifact = api.artifact(f"swe-qwen/adapter-{wandb_run_id}:latest")
    local_dir = str(output_dir / f"adapter-{wandb_run_id}")
    artifact.download(root=local_dir)
    return local_dir


# ── F2P evaluation ────────────────────────────────────────────────────────────


def evaluate_proxy_f2p(
    variant: str,
    golden_path: Path,
    adapter_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run proxy F2P evaluation for one variant."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "f2p_proxy",
        _REPO_ROOT / "scripts" / "f2p_proxy.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scores = mod.compute_proxy_f2p_scores(
        golden_path=golden_path,
        variant_adapter_map={variant: adapter_path if not dry_run else ""},
    )
    return scores[variant]


# ── Champion promotion ────────────────────────────────────────────────────────


def promote_champion(
    champion_variant: str,
    champion_wandb_run_id: str,
    dry_run: bool = False,
) -> None:
    """Tag the champion adapter as ``champion`` in W&B Registry."""
    if dry_run:
        print(
            f"  [DRY-RUN] Would promote {champion_variant} ({champion_wandb_run_id}) "
            "to W&B Registry champion alias"
        )
        return

    import wandb

    api = wandb.Api()
    artifact = api.artifact(f"swe-qwen/adapter-{champion_wandb_run_id}:latest")
    artifact.aliases.append("champion")
    artifact.save()
    print(f"  Promoted {champion_variant} ({champion_wandb_run_id}) → champion alias")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    data_dir = _REPO_ROOT / "data" / args.run_id
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    golden_path = _golden_path(args.run_id)
    if not golden_path.is_file():
        print(f"Warning: golden set not found at {golden_path}, proxy F2P will use cleaned.jsonl")
        # Fall back to cleaned.jsonl — all records, no train/val split
        golden_path = data_dir / "swebench/cleaned.jsonl"
        if not golden_path.is_file():
            print(f"Error: no data found under {data_dir}", file=sys.stderr)
            sys.exit(1)

    print("Phase 4H: 3-Config Comparison")
    print(f"  Run ID:    {args.run_id}")
    print(f"  Golden:    {golden_path}")
    print(f"  Variants:  {', '.join(args.variants)}")
    print(f"  Mode:      {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print()

    output_dir = _REPO_ROOT / "models" / "comparisons" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Launch all variants
    results: dict[str, dict[str, Any]] = {}
    for variant in args.variants:
        print(f"[{variant}]")
        run_name = _variant_run_name(variant)

        wandb_run_id = launch_modal_training(variant, args.run_id, run_name, args.dry_run)
        adapter_path = download_adapter(wandb_run_id, output_dir, args.dry_run)
        f2p_result = evaluate_proxy_f2p(variant, golden_path, adapter_path, args.dry_run)

        results[variant] = {
            "run_name": run_name,
            "wandb_run_id": wandb_run_id,
            "adapter_path": adapter_path,
            **f2p_result,
        }
        print(f"  mean_f2p={f2p_result['mean_f2p']}")
        print()

    # Step 2: Select champion
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "f2p_proxy",
        _REPO_ROOT / "scripts" / "f2p_proxy.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    select_champion = _mod.select_champion

    champion = select_champion(results)
    champion_wandb_run_id = results[champion]["wandb_run_id"]
    print(f"Champion: {champion} (F2P={results[champion]['mean_f2p']})")

    # Step 3: Promote champion
    promote_champion(champion, champion_wandb_run_id, args.dry_run)

    # Step 4: Output summary
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


if __name__ == "__main__":
    main()
