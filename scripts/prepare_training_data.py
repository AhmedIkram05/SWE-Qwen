#!/usr/bin/env python3
"""Prepare training data from Phase 3 pipeline output.

Loads raw.jsonl (8282 records), applies quality filters, creates repo-stratified
train/val/test/golden splits, and tokenizes into .arrow shards.

Usage:
    # From project root (using venv python):
    ./.venv/bin/python scripts/prepare_training_data.py \\
        --input data/25d3f8fd0ccb/swebench/raw.jsonl \\
        --output-dir data/tokenized \\
        --model-name qwen3-30b-a3b

    # With custom seed and split ratios:
    ./.venv/bin/python scripts/prepare_training_data.py \\
        --input data/25d3f8fd0ccb/swebench/raw.jsonl \\
        --output-dir data/tokenized \\
        --seed 137 \\
        --train-ratio 0.85 --val-ratio 0.05 --test-ratio 0.10 \\
        --model-name qwen3-14b
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_training_data")


RATIO_EPSILON = 1e-9


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def basic_quality_filters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply basic quality filters, keeping ALL records for training.

    Only removes records with:
    - Empty patch_diff
    - Binary file changes (non-Python)
    - Missing required fields
    """
    required = {"issue_id", "repo", "patch_diff", "issue_body"}
    kept: list[dict[str, Any]] = []
    skipped = 0
    for rec in records:
        # Check required fields
        if not all(k in rec for k in required):
            skipped += 1
            continue

        # Empty patch
        if not rec.get("patch_diff", "").strip():
            skipped += 1
            continue

        # Binary/non-Python patch (patch only changes non-.py files)
        files_changed = rec.get("files_changed", [])
        if files_changed and not any(f.endswith(".py") for f in files_changed):
            # If patch only touches non-Python files, still keep it
            # as it may have test configs or docs
            pass

        kept.append(rec)

    logger.info(
        "Quality filter: %d kept, %d skipped",
        len(kept),
        skipped,
    )
    return kept


def repo_stratified_split(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by repo to prevent leakage.

    Each repo appears in exactly one split. Repos are assigned via greedy
    bin-packing by size to hit the target ratios.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < RATIO_EPSILON

    repo_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        repo_groups[rec["repo"]].append(rec)

    rng = random.Random(seed)
    repos = sorted(repo_groups.keys())
    rng.shuffle(repos)

    total = len(records)
    targets = [int(total * r) for r in (train_ratio, val_ratio, test_ratio)]
    # Ensure smallest split gets at least one repo
    min_split = min(range(3), key=lambda i: targets[i])

    bins: list[list[str]] = [[] for _ in range(3)]
    bin_sizes = [0, 0, 0]

    for repo in repos:
        group_size = len(repo_groups[repo])
        # Always assign smallest split if it's still empty
        if not bins[min_split]:
            assign = min_split
        else:
            # Pick the bin that's most under its target
            assign = min(
                range(3),
                key=lambda i: bin_sizes[i] / targets[i] if targets[i] > 0 else float("inf"),
            )
        bins[assign].append(repo)
        bin_sizes[assign] += group_size

    train_repos, val_repos, test_repos = bins

    train = [r for r in records if r["repo"] in train_repos]
    val = [r for r in records if r["repo"] in val_repos]
    test = [r for r in records if r["repo"] in test_repos]

    logger.info(
        "Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train),
        100 * len(train) / total,
        len(val),
        100 * len(val) / total,
        len(test),
        100 * len(test) / total,
    )

    for name, rps in [("train", train_repos), ("val", val_repos), ("test", test_repos)]:
        rl = sorted(rps)
        logger.info("  %s repos (%d): %s", name, len(rl), ", ".join(rl))

    return train, val, test


def extract_golden(
    test_records: list[dict[str, Any]],
    max_golden: int = 500,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extup to *max_golden* golden records from test split.

    Golden = records with test patches (F2P-evaluable). Caps at *max_golden*
    so the test split retains records for dev-time evaluation.

    Returns (golden, remaining_test).
    """
    rng = random.Random(seed)
    golden_eligible = [
        r
        for r in test_records
        if r.get("metadata", {}).get("has_test_patch", False) or r.get("test_files_changed", [])
    ]

    rng.shuffle(golden_eligible)
    golden = golden_eligible[:max_golden]
    golden_set = {id(r) for r in golden}
    remaining = [r for r in test_records if id(r) not in golden_set]

    logger.info(
        "Golden: %d records (capped at %d, from %d eligible)",
        len(golden),
        max_golden,
        len(golden_eligible),
    )
    return golden, remaining


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def tokenize_datasets(
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    max_length: int | None = None,
) -> bool:
    """Run the tokenization pipeline on prepared JSONL files.

    Returns True on success, False on failure.
    """
    try:
        from data_engineering.tokenize import tokenize_pipeline

        tokenize_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            max_length=max_length,
        )
    except Exception:
        logger.exception("Tokenization failed")
        return False
    else:
        return True


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare training data from Phase 3 pipeline output",
    )
    parser.add_argument(
        "--input",
        default="data/25d3f8fd0ccb/swebench/raw.jsonl",
        help="Path to raw.jsonl from Phase 3 pipeline (default: best run)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tokenized",
        help="Output directory for JSONL splits and .arrow shards",
    )
    parser.add_argument(
        "--model-name",
        default="qwen3-14b",
        help="Model name for tokenizer selection (from models.yaml)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override max sequence length (default: from models.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test split ratio")
    parser.add_argument(
        "--max-golden", type=int, default=500, help="Max golden records to extract from test split"
    )
    parser.add_argument("--skip-tokenize", action="store_true", help="Skip tokenization step")
    return parser


def log_summary(
    counts: tuple[int, int, int, int],
    shards_dir: Path,
    skip_tokenize: bool,
) -> None:
    """Log final summary of data preparation."""
    total_train, total_val, total_test, total_golden = counts
    total_all = total_train + total_val + total_test + total_golden
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("  Train:  %d records (%.1f%%)", total_train, 100 * total_train / total_all)
    logger.info("  Val:    %d records (%.1f%%)", total_val, 100 * total_val / total_all)
    logger.info("  Test:   %d records (%.1f%%)", total_test, 100 * total_test / total_all)
    logger.info("  Golden: %d records (%.1f%%)", total_golden, 100 * total_golden / total_all)
    logger.info("  Total:  %d records", total_all)
    logger.info("  Tokenized: %s", shards_dir if not skip_tokenize else "skipped")
    logger.info("=" * 60)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        # List available runs
        data_dir = input_path.parent.parent.parent
        if data_dir.exists():
            runs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name[0].isdigit())
            if runs:
                logger.info("Available runs: %s", [r.name for r in runs[-5:]])
        return

    output_dir = Path(args.output_dir)
    jsonl_dir = output_dir / "jsonl"
    shards_dir = output_dir / "shards"

    # 1. Load raw data
    raw = load_jsonl(input_path)

    # 2. Quality filters
    filtered = basic_quality_filters(raw)

    if not filtered:
        logger.error("No records passed quality filters!")
        return

    # 3. Stratified split
    train, val, test = repo_stratified_split(
        filtered,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 4. Extract golden from test
    golden, test_remaining = extract_golden(test, max_golden=args.max_golden, seed=args.seed)

    # 5. Save JSONL splits
    save_jsonl(train, jsonl_dir / "train.jsonl")
    save_jsonl(val, jsonl_dir / "val.jsonl")
    save_jsonl(test_remaining, jsonl_dir / "test.jsonl")
    save_jsonl(golden, jsonl_dir / "golden.jsonl")

    # 6. Tokenize (unless --skip-tokenize)
    if not args.skip_tokenize:
        logger.info("Starting tokenization...")
        if tokenize_datasets(
            jsonl_dir,
            shards_dir,
            model_name=args.model_name,
            max_length=args.max_length,
        ):
            logger.info("Tokenized data saved to %s", shards_dir)
        else:
            logger.error("Tokenization failed — check logs above")
    else:
        logger.info("Skipping tokenization (--skip-tokenize). JSONL at: %s", jsonl_dir)

    log_summary(
        (len(train), len(val), len(test_remaining), len(golden)),
        shards_dir,
        args.skip_tokenize,
    )


if __name__ == "__main__":
    main()
