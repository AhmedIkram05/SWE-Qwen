"""P0-7 determinism guards: pinned split seed + reproducible splits + audit table.

- ``stratified_split`` is reproducible given the same seed (and the same input
  records).
- The split seed is pinned: a random per-run ``run_id`` must never reshuffle
  the split; a run_id-derived seed is allowed only when the user sets
  ``run_id_override`` (stable across resumes).
- A per-split repos/examples/domains table is logged and available via
  ``split_summary``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord
from data_engineering.split import (
    DEFAULT_SPLIT_SEED,
    resolve_split_seed,
    split_summary,
    stratified_split,
)


def _make_record(issue_id: str, repo: str, domain: str = "web") -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo=repo,
        issue_body="body text for issue",
        patch_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n",
        test_files_changed=["test_foo.py"],
        files_changed=["foo.py"],
        commit_messages=["fix: something"],
        repo_domain=domain,
    )


def _records(n_repos: int = 20, per_repo: int = 3) -> list[IssueRecord]:
    return [
        _make_record(f"r{i}#{j}", repo=f"org/repo-{i:02d}", domain=f"domain-{i % 4}")
        for i in range(n_repos)
        for j in range(per_repo)
    ]


def _split_ids(splits: Any) -> tuple[list[str], list[str], list[str]]:
    return (
        [r.issue_id for r in splits.train],
        [r.issue_id for r in splits.val],
        [r.issue_id for r in splits.test],
    )


class TestSplitReproducibility:
    def test_same_seed_same_split(self) -> None:
        config = DataPipelineConfig()
        s1 = stratified_split(_records(), config, seed=42)
        s2 = stratified_split(_records(), config, seed=42)
        assert _split_ids(s1) == _split_ids(s2)

    def test_default_seed_is_pinned_constant(self) -> None:
        config = DataPipelineConfig()
        assert (
            stratified_split(_records(), config).train
            == stratified_split(_records(), config, seed=DEFAULT_SPLIT_SEED).train
        )

    def test_different_seed_changes_split(self) -> None:
        config = DataPipelineConfig()
        s42 = stratified_split(_records(24), config, seed=42)
        s7 = stratified_split(_records(24), config, seed=7)
        train_42 = {r.repo for r in s42.train}
        train_7 = {r.repo for r in s7.train}
        assert train_42 != train_7


class TestSplitSeedPinned:
    """run_pipeline must not seed the split from a random run_id."""

    def test_pinned_seed_ignores_random_run_id(self) -> None:
        config = DataPipelineConfig()
        assert config.run_id_override is None
        # A random per-run run_id must never change the seed.
        assert resolve_split_seed(config, "random-uuid-aaa") == DEFAULT_SPLIT_SEED
        assert resolve_split_seed(config, "random-uuid-bbb") == DEFAULT_SPLIT_SEED

    def test_pinned_seed_gives_identical_split_across_run_ids(self) -> None:
        config = DataPipelineConfig()
        records = _records()
        s_a = stratified_split(records, config, seed=resolve_split_seed(config, "run-a"))
        s_b = stratified_split(records, config, seed=resolve_split_seed(config, "run-b"))
        assert _split_ids(s_a) == _split_ids(s_b)

    def test_run_id_override_derives_stable_seed(self) -> None:
        config = DataPipelineConfig(run_id_override="pinned-run")
        seed_a = resolve_split_seed(config, "stable-run-id")
        assert seed_a == resolve_split_seed(config, "stable-run-id")
        assert seed_a == int.from_bytes(hashlib.sha256(b"stable-run-id").digest()[:4], "little")
        # Different run ids may derive different seeds under an override.
        assert resolve_split_seed(config, "other-run-id") != seed_a


class TestSplitSummary:
    def test_summary_counts(self) -> None:
        config = DataPipelineConfig()
        splits = stratified_split(_records(8, per_repo=2), config, seed=42)
        summary = split_summary(splits)
        assert set(summary) == {"train", "val", "test"}
        for name, recs in (
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        ):
            assert summary[name]["repos"] == len({r.repo for r in recs})
            assert summary[name]["examples"] == len(recs)
            assert summary[name]["domains"] == len({r.repo_domain for r in recs if r.repo_domain})

    def test_summary_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        config = DataPipelineConfig()
        with caplog.at_level("INFO", logger="data_engineering.split"):
            splits = stratified_split(_records(8, per_repo=2), config, seed=42)
        line = next((r.message for r in caplog.records if "Split summary" in r.message), None)
        assert line is not None
        summary = split_summary(splits)
        for name in ("train", "val", "test"):
            assert str(summary[name]["repos"]) in line
            assert str(summary[name]["examples"]) in line
