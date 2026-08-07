#!/usr/bin/env python3
"""Deploy telemetry for cd.yml — logs ``deploy/status`` + ``deploy/duration_s``.

One short-lived W&B run per deploy (ADR-018 / plan §5.6b): emits the two
registered ``deploy/*`` keys. `build_payload` is pure and registry-checked so
the contract test can import it without credentials.

Usage:
    uv run python scripts/log_deploy.py --status 1 --duration-s 342 --sha abc123
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from observability.metrics import METRIC_REGISTRY, assert_registered


def build_payload(
    status: int,
    duration_s: float | None = None,
    sha: str | None = None,
) -> dict[str, int | float | str]:
    """Return exactly the registered ``deploy/*`` keys, validated.

    ``deploy/sha`` is not a registered metric, so *sha* is never emitted here;
    the CLI stores it in the run config instead. Keys are checked against
    ``METRIC_REGISTRY`` so a future registry change fails loudly, not silently.
    """
    if status not in (0, 1):
        raise ValueError(f"status must be 0 or 1, got {status}")

    payload: dict[str, int | float | str] = {"deploy/status": status}
    if duration_s is not None:
        payload["deploy/duration_s"] = duration_s
    if sha is not None and "sha" in METRIC_REGISTRY["deploy"]:
        payload["deploy/sha"] = sha

    assert_registered(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Log deploy status and duration to W&B (plan §5.6b)."
    )
    parser.add_argument(
        "--status",
        type=int,
        required=True,
        choices=[0, 1],
        help="Deploy outcome: 1 success, 0 failure.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Deploy wall time, seconds.",
    )
    parser.add_argument(
        "--sha",
        default=None,
        help="Commit SHA, stored in the run config, not a metric.",
    )
    parser.add_argument("--entity", default="2571642-university-of-dundee")
    parser.add_argument("--project", default="swe-qwen")
    args = parser.parse_args(argv)

    payload = build_payload(args.status, args.duration_s, args.sha)

    config: dict[str, Any] = {
        "deploy_status": args.status,
        "deploy_duration_s": args.duration_s,
    }
    if args.sha:
        config["sha"] = args.sha

    import wandb

    try:
        wandb.init(project=args.project, entity=args.entity, job_type="deploy", config=config)
        wandb.log(payload)
    finally:
        if wandb.run is not None:
            wandb.run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
