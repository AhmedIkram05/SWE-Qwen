"""Repo-stratified train/val/test split and golden eval subset extraction.

Key constraint: each repo appears in EXACTLY one split (no data leakage).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from data_engineering.clean import F2P_KEYWORD_PATTERN, _has_f2p_keywords
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, Splits

logger = logging.getLogger(__name__)


def stratified_split(
    records: list[IssueRecord],
    config: DataPipelineConfig,
    seed: int = 42,
) -> Splits:
    """Split records by repo into train/val/test.

    Args:
        records: Cleaned deduplicated records.
        config: Pipeline config with ratio settings.
        seed: Random seed for reproducible splits (used hash of seed + run_id).

    Returns:
        Splits with train/val/test. All non-empty.
    """
    # Group by repo
    repo_groups: dict[str, list[IssueRecord]] = defaultdict(list)
    for rec in records:
        repo_groups[rec.repo].append(rec)

    repos = list(repo_groups.keys())
    rng = random.Random(seed)

    # Shuffle repos deterministically
    rng.shuffle(repos)

    # Calculate split indices
    total = len(repos)
    n_train = max(1, round(total * config.train_ratio))
    n_val = max(1, round(total * config.val_ratio))

    train_repos = repos[:n_train]
    val_repos = repos[n_train : n_train + n_val]
    test_repos = repos[n_train + n_val :]

    # Handle edge case: if test set would be empty, peel one from val
    if not test_repos and val_repos:
        test_repos = val_repos[-1:]
        val_repos = val_repos[:-1]

    splits = Splits(
        train=[rec for r in train_repos for rec in repo_groups[r]],
        val=[rec for r in val_repos for rec in repo_groups[r]],
        test=[rec for r in test_repos for rec in repo_groups[r]],
    )

    logger.info(
        "Split: %d train / %d val / %d test repos → %d / %d / %d records",
        len(train_repos),
        len(val_repos),
        len(test_repos),
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    return splits


def extract_golden(
    splits: Splits,
    min_size: int,
    source_split: str = "test",
) -> list[IssueRecord]:
    """Extract golden eval subset from the test split.

    Uses the V1 F2P proxy: ``test_files_changed`` non-empty AND F2P keywords
    in commit messages/PR description.

    Args:
        splits: Pipeline splits (train/val/test).
        min_size: Minimum number of golden examples required.
        source_split: Which split to source from (``"test"`` or ``"all"``).

    Returns:
        List of golden-eval-qualified records.
    """
    if source_split == "all":
        source = splits.train + splits.val + splits.test
        logger.warning(
            "Golden set sourced from ALL splits — DATA LEAKAGE RISK. Prefer source_split='test'."
        )
    else:
        source = splits.test

    golden: list[IssueRecord] = []
    for rec in source:
        if rec.test_files_changed and _has_f2p_keywords(rec):
            golden.append(rec)

    n_golden = len(golden)
    if n_golden < min_size:
        logger.warning(
            "Golden set has %d examples (min %d requested). "
            "Consider expanding repo pool or lowering min_golden_examples.",
            n_golden,
            min_size,
        )

    logger.info(
        "Golden set: %d examples from '%s' split (min_target=%d)",
        n_golden,
        source_split,
        min_size,
    )
    return golden
