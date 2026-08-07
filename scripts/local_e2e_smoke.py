#!/usr/bin/env python3
"""Local end-to-end eval smoke test: 1 record, local pytest, verify F2P.

Usage::

    # Ensure Ollama is running:
    #   ollama pull qwen2.5-coder:7b
    #   ollama serve
    #
    # Then:
    python scripts/local_e2e_smoke.py --sample 1

    # Dry-run: validate data path only (skip inference + tests)
    python scripts/local_e2e_smoke.py --dry-run

Skips Modal entirely. Uses Ollama for inference, local pytest for tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from observability.logging import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger("local_e2e_smoke")


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Local E2E eval smoke test")
    parser.add_argument(
        "--golden-path",
        default=str(
            Path(__file__).resolve().parent.parent / "data/expanded-repos/swebench/golden.jsonl"
        ),
    )
    parser.add_argument("--sample", type=int, default=1, help="Records to test")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model tag")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--dry-run", action="store_true", help="Data-path validation only")
    parser.add_argument(
        "--use-golden-patch",
        action="store_true",
        help="Use ground-truth test_patch (skip Ollama inference). "
        "Validates patch apply + test runner + metrics.",
    )
    args = parser.parse_args()

    # ── 1. Load and validate data ────────────────────────────────────────
    from evaluation.config import EvalConfig
    from evaluation.harness import EvaluationHarness

    config = EvalConfig(golden_data_path=args.golden_path)
    harness = EvaluationHarness(config)

    logger.info("Loading examples from %s ...", args.golden_path)
    examples = harness.load_examples(split="golden")
    logger.info("Loaded %d examples", len(examples))

    if args.sample:
        import random

        random.seed(42)
        examples = random.sample(examples, min(args.sample, len(examples)))

    example = examples[0]
    logger.info(
        "Selected: instance=%s repo=%s fail_to_pass=%d pass_to_pass=%d",
        example.instance_id,
        example.repo,
        len(example.fail_to_pass or []),
        len(example.pass_to_pass or []),
    )

    if args.dry_run:
        logger.info("DRY-RUN: data path validated, exiting")
        return

    # ── 2. Monkeypatch harness with local backends ────────────────────────
    import evaluation.harness as harness_mod
    from evaluation.local_backend import run_tests_local

    def _local_patches(model_name, variant, prompt_template, examples):
        if args.use_golden_patch:
            # Skip inference: use ground-truth test_patch to validate patch apply + tests + metrics
            return [ex.test_patch for ex in examples]  # noqa: ARG001
        from evaluation.local_backend import generate_patches_local

        return generate_patches_local(
            model_name,
            variant,
            prompt_template,
            examples,
            ollama_model=args.model,
            ollama_base_url=args.ollama_url,
        )

    harness_mod._generate_patches = _local_patches  # type: ignore[assignment]

    # Patch test runner
    def _run_tests(example, generated_patch, config):
        return run_tests_local(example, generated_patch, config)

    harness_mod._run_tests = _run_tests

    # ── 3. Run evaluation ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting local eval: %s via Ollama (%s)", example.instance_id, args.model)
    logger.info("=" * 60)

    start = time.monotonic()
    result = harness.run_example(
        example,
        model_name="qwen3-14b",
        variant="baseline_14b",
        prompt_template="chat",
    )
    elapsed = time.monotonic() - start

    # ── 4. Report ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULT (%.1fs)", elapsed)
    logger.info("  instance_id:  %s", result.instance_id)
    logger.info("  patch_applied: %s", result.patch_application.success)
    logger.info("  F2P:          %.2f", result.f2p)
    logger.info("  P2P:          %.2f", result.p2p)
    logger.info("  latency:      %.1fs", result.latency_seconds)
    logger.info("  error:        %s", result.error)
    logger.info("  tests_before: %d", len(result.tests_before))
    logger.info("  tests_after:  %d", len(result.tests_after))

    if result.patch_application.success:
        logger.info("  files:        %s", result.patch_application.files_modified)

    # Print test outcome summary
    for phase, tests in [("before", result.tests_before), ("after", result.tests_after)]:
        statuses: dict[str, int] = {}
        for t in tests:
            statuses[t.status] = statuses.get(t.status, 0) + 1
        logger.info("  tests_%s: %s", phase, statuses)

    logger.info("=" * 60)
    print("\n--- JSON summary ---")
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "f2p": result.f2p,
                "p2p": result.p2p,
                "patch_success": result.patch_application.success,
                "latency_s": round(result.latency_seconds, 1),
                "error": result.error,
                "tests_before": len(result.tests_before),
                "tests_after": len(result.tests_after),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
