#!/usr/bin/env python3
"""Tag a finished training run's W&B artifact as ``challenger`` (Phase 9 §4.7).

The promotion pipeline reads the ``challenger`` alias to know a candidate is
in the lane. Idempotent and retry-able: an already-tagged artifact is a no-op
and ``latest`` is never removed.

Usage:
    uv run python scripts/tag_challenger.py --variant higher_lr_14b
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tag a training artifact as challenger in W&B (Phase 9 §4.7)."
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Training variant name, e.g. higher_lr_14b.",
    )
    parser.add_argument("--entity", default="2571642-university-of-dundee")
    parser.add_argument("--project", default="swe-qwen")
    args = parser.parse_args(argv)

    artifact_name = f"model-qwen3-14b-{args.variant}"

    # Lazy import: module stays import-safe offline (house style, log_deploy.py).
    try:
        import wandb
    except ImportError as e:
        print(f"error: wandb not installed ({e}); run `uv sync` first", file=sys.stderr)
        return 1

    try:
        api = wandb.Api(timeout=30)
        artifact = api.artifact(f"{args.entity}/{args.project}/{artifact_name}:latest")
    except Exception as e:
        print(
            f"error: could not resolve {args.entity}/{args.project}/{artifact_name}:latest: {e}",
            file=sys.stderr,
        )
        return 1

    if "latest" in artifact.aliases and "challenger" in artifact.aliases:
        print(f"{artifact_name} already tagged challenger (no-op)")
        return 0

    # Keep `latest`; append `challenger` only.
    artifact.aliases.append("challenger")
    try:
        artifact.save()
    except Exception as e:
        print(f"error: failed to tag {artifact_name} as challenger: {e}", file=sys.stderr)
        return 1

    print(f"Tagged {artifact_name} → challenger (kept latest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
