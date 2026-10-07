"""Repo-stratified train/val/test split and golden eval subset extraction.

Key constraint: each repo appears in EXACTLY one split (no data leakage).
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict

from data_engineering.clean import F2P_KEYWORD_PATTERN, _has_f2p_keywords
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, Splits

logger = logging.getLogger(__name__)

# Pinned split seed. A fresh pipeline run uses a random run_id (uuid), and a
# run_id-derived seed would reshuffle the split every run — train/val/test
# would not be reproducible across pipeline re-runs. The seed stays pinned;
# a run_id-derived seed is used only when the user explicitly opts in via
# ``run_id_override`` (stable across resumes).
DEFAULT_SPLIT_SEED = 42


def resolve_split_seed(config: DataPipelineConfig, run_id: str) -> int:
    """Split seed for a pipeline run: pinned, or run_id-derived under override.

    Without ``run_id_override`` the run_id is a fresh random uuid per run, so
    deriving the seed from it makes every re-run produce a different split.
    With an explicit ``run_id_override`` the id is stable across resumes and
    a derived seed is reproducible, so it is allowed.
    """
    if config.run_id_override:
        return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "little")
    return DEFAULT_SPLIT_SEED


def split_summary(splits: Splits) -> dict[str, dict[str, int]]:
    """Per-split audit table: repo, example, and domain counts."""
    return {
        name: {
            "repos": len({r.repo for r in recs}),
            "examples": len(recs),
            "domains": len({r.repo_domain for r in recs if r.repo_domain}),
        }
        for name, recs in (("train", splits.train), ("val", splits.val), ("test", splits.test))
    }


def stratified_split(
    records: list[IssueRecord],
    config: DataPipelineConfig,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Splits:
    """Split records by repo into train/val/test.

    Args:
        records: Cleaned deduplicated records.
        config: Pipeline config with ratio settings.
        seed: Random seed for reproducible splits. Pinned via
            ``resolve_split_seed`` (``run_id`` never seeds the split unless
            ``run_id_override`` is set).

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
    logger.info("Split summary (repos/examples/domains): %s", split_summary(splits))

    return splits


def extract_golden(
    splits: Splits,
    min_size: int,
    source_split: str = "test",
) -> list[IssueRecord]:
    """Extract golden eval subset.

    Uses the V1 F2P proxy: ``test_files_changed`` non-empty AND F2P keywords
    in commit messages/PR description.

    ``source_split="verified+test+dev"`` instead sources from records whose
    ingest provenance is one of the official SWE-bench F2P splits
    (verified/test/dev — all carry FAIL_TO_PASS ground truth), skipping the
    F2P keyword heuristic.

    Args:
        splits: Pipeline splits (train/val/test).
        min_size: Minimum number of golden examples required.
        source_split: Which split to source from (``"test"``, ``"all"``, or
            ``"verified+test+dev"``).

    Returns:
        List of golden-eval-qualified records.
    """
    if source_split == "verified+test+dev":
        source = [
            rec
            for rec in splits.train + splits.val + splits.test
            if rec.metadata.get("source_split") in {"verified", "test", "dev"}
        ]
        # All three official splits have FAIL_TO_PASS by construction.
        golden = [rec for rec in source if rec.test_files_changed]
        logger.info(
            "Golden set: %d examples from official SWE-bench F2P splits "
            "(verified+test+dev, min_target=%d)",
            len(golden),
            min_size,
        )
        if len(golden) < min_size:
            logger.warning(
                "Golden set has %d examples (min %d requested).",
                len(golden),
                min_size,
            )
        return golden

    if source_split == "all":
        source = splits.train + splits.val + splits.test
        logger.warning(
            "Golden set sourced from ALL splits — DATA LEAKAGE RISK. Prefer source_split='test'."
        )
    else:
        source = splits.test

    golden = []
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
