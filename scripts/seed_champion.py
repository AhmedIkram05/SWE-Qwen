"""One-shot seed of the first champion record (Phase 9, spec §4.7).

The champion loop needs a baseline on first run, and ``gs://swe-qwen-datasets/
ci/champion.json`` is empty until this runs (writes there require GCP WIF
auth).  This script writes the documented 2026-08-06 Champion
(``higher_lr_14b`` — F2P 0.169, P2P 0.912, n=50 golden, promoted 2026-08-06)
to a local ``champion.json`` and prints the ``gcloud storage cp`` command to
upload it::

    uv run python scripts/seed_champion.py
    uv run python scripts/seed_champion.py --output /tmp/champion.json

One-shot, import-safe (no side effects at import), no GCS SDK needed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from promotion.registry import ChampionRecord, write_champion

DEFAULT_OUTPUT = Path("data/eval_results/champion.json")
PROD_BUCKET_PATH = "gs://swe-qwen-datasets/ci/champion.json"


def build_record(
    promoted_at: str,
    model_ref: str | None = None,
    variant: str | None = None,
) -> ChampionRecord:
    """The 2026-08-06 Champion (spec §4.7): seeded, no previous.

    Defaults reproduce the historical seeded record verbatim;
    ``--model-ref``/``--variant`` override model/variant for future cycles.

    ``tier="full"`` because n=50 matches ``EvalConfig.tier_sizes["full"]`` —
    ``EvalConfig`` itself has no ``tier`` field (spec §4.6).
    """
    model_ref = model_ref or "qwen3-14b"
    variant = variant or "higher_lr_14b"
    return ChampionRecord(
        variant=variant,
        model_ref=f"{model_ref}:{variant}",
        f2p_rate=0.169,
        p2p_rate=0.912,
        dataset_run_id="expanded-repos",
        tier="full",
        seed=42,
        promoted_at=promoted_at,
        previous=None,
    )


def main(argv: list[str] | None = None) -> int:
    """Write the baseline champion record and print the upload command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"local destination for champion.json (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model-ref",
        default=None,
        help="model half of model_ref for future cycles (default: seeded 'qwen3-14b')",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="champion variant for future cycles (default: seeded 'higher_lr_14b')",
    )
    args = parser.parse_args(argv)

    record = build_record(
        datetime.now(UTC).isoformat(),
        model_ref=args.model_ref,
        variant=args.variant,
    )
    write_champion(args.output, record)

    print(f"champion.json written to {args.output}")
    print(f"  variant={record.variant} model_ref={record.model_ref}")
    print(
        f"  f2p_rate={record.f2p_rate} p2p_rate={record.p2p_rate} "
        f"dataset_run_id={record.dataset_run_id}"
    )
    print(f"  tier={record.tier} seed={record.seed} promoted_at={record.promoted_at}")
    print(f"  previous={record.previous}")
    print(f"Upload with: gcloud storage cp {args.output} {PROD_BUCKET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
