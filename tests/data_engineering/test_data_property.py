"""Property-based tests for the data pipeline using Hypothesis.

Validates invariants across random data:
- Validation never throws (always returns a tuple)
- Dedup never increases cardinality
- Split preserves total count and repo isolation
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from data_engineering.clean import clean_records, deduplicate
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord
from data_engineering.split import stratified_split
from data_engineering.validate import validate_batch

# ── Strategies ─────────────────────────────────────────────────────────────

valid_diff = st.builds(
    str,
    st.from_regex(
        r"--- a/.*\n\+\+\+ b/.*\n@@ -\d+,\d+ \+\d+,\d+ @@\n.*",
        fullmatch=True,
    ),
)

# Fallback: generate a known-good diff if regex doesn't match
_fallback_diff = "--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n-x\n+x\n+y\n"


def _make_patch() -> st.SearchStrategy[str]:
    """Generate a valid unified diff."""
    return st.sampled_from(
        [
            _fallback_diff,
            "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,4 @@\n foo\n-bar\n+baz\n+qux\n",
            "--- a/x.py\n+++ b/x.py\n@@ -5,7 +5,8 @@\n def f():\n-    pass\n+    return 1\n",
        ]
    )


def _non_empty_body() -> st.SearchStrategy[str]:
    """Generate a non-empty, non-whitespace issue body."""
    return st.text(
        min_size=1,
        max_size=100,
        # Exclude whitespace-only strings (IssueRecord validator strips + rejects)
    ).filter(lambda s: s.strip())


record_strategy = st.builds(
    IssueRecord,
    issue_id=st.text(min_size=1, max_size=20),
    pr_number=st.text(min_size=1, max_size=10),
    repo=st.sampled_from(["owner/repo1", "owner/repo2", "owner/repo3"]),
    issue_body=_non_empty_body(),
    patch_diff=_make_patch(),
    test_files_changed=st.lists(st.text(min_size=1, max_size=20), max_size=3),
    files_changed=st.lists(st.text(min_size=1, max_size=20), max_size=3),
    commit_messages=st.lists(st.text(min_size=1, max_size=50), max_size=3),
    pr_description=st.text(max_size=100),
    issue_labels=st.lists(st.text(max_size=20), max_size=3),
)


# ── Config ─────────────────────────────────────────────────────────────────

CONFIG = DataPipelineConfig(
    max_patch_lines=500,
    min_golden_examples=0,
    test_directories=["tests/", "test/"],
)


# ── Property tests ─────────────────────────────────────────────────────────


@pytest.mark.hypothesis
class TestValidateProperties:
    @given(st.lists(record_strategy, max_size=5))
    def test_validate_never_throws(self, records: list[IssueRecord]) -> None:
        """Validation should always return a tuple."""
        data = [r.model_dump() for r in records]
        valid, errors = validate_batch(data)
        assert isinstance(valid, list)
        assert isinstance(errors, list)

    @given(st.lists(record_strategy, max_size=5))
    def test_validated_count_leq_input(self, records: list[IssueRecord]) -> None:
        """Number of valid records should not exceed input."""
        data = [r.model_dump() for r in records]
        valid, _ = validate_batch(data)
        assert len(valid) <= len(data)


@pytest.mark.hypothesis
class TestDedupProperties:
    @given(st.lists(record_strategy, max_size=10))
    def test_dedup_never_increases(self, records: list[IssueRecord]) -> None:
        """Dedup should never produce more records than input."""
        deduped, stats = deduplicate(records)
        assert len(deduped) <= len(records)
        assert stats.total_input == len(records)

    @given(st.lists(record_strategy, max_size=10))
    def test_dedup_stats_sum(self, records: list[IssueRecord]) -> None:
        """Stats counts should be consistent."""
        deduped, stats = deduplicate(records)
        expected = stats.unique_output
        actual = len(deduped)
        assert expected == actual

    @given(st.lists(record_strategy, max_size=10))
    def test_all_duplicates_removed(self, records: list[IssueRecord]) -> None:
        """No two records in dedup output should have same ID hash."""
        deduped, _ = deduplicate(records)
        ids = [(r.issue_id, r.repo, r.pr_number) for r in deduped]
        assert len(ids) == len(set(ids))


@pytest.mark.hypothesis
class TestCleanProperties:
    @given(st.lists(record_strategy, max_size=10))
    def test_clean_never_increases(self, records: list[IssueRecord]) -> None:
        """Clean should never produce more records than input."""
        cleaned, stats = clean_records(records, CONFIG)
        assert len(cleaned) <= len(records)
        assert stats.total_input == len(records)

    @given(st.lists(record_strategy, max_size=10))
    def test_clean_stats_count(self, records: list[IssueRecord]) -> None:
        """Removed + output = input."""
        cleaned, stats = clean_records(records, CONFIG)
        assert len(cleaned) + stats.total_removed == stats.total_input


@pytest.mark.hypothesis
class TestSplitProperties:
    @given(st.lists(record_strategy, min_size=3, max_size=15))
    def test_split_preserves_total(self, records: list[IssueRecord]) -> None:
        """Total records across splits should equal input."""
        assume(len({r.repo for r in records}) > 1)
        splits = stratified_split(records, CONFIG, seed=42)
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == len(records)

    @given(st.lists(record_strategy, min_size=3, max_size=15))
    def test_no_repo_leakage(self, records: list[IssueRecord]) -> None:
        """Each repo should appear in at most one split."""
        assume(len({r.repo for r in records}) > 1)
        splits = stratified_split(records, CONFIG, seed=42)
        train_repos = {r.repo for r in splits.train}
        val_repos = {r.repo for r in splits.val}
        test_repos = {r.repo for r in splits.test}
        assert train_repos.isdisjoint(val_repos)
        assert train_repos.isdisjoint(test_repos)
        assert val_repos.isdisjoint(test_repos)

    @given(st.lists(record_strategy, min_size=3, max_size=15))
    def test_split_reproducible(self, records: list[IssueRecord]) -> None:
        """Same seed should produce identical splits."""
        assume(len({r.repo for r in records}) > 1)
        s1 = stratified_split(records, CONFIG, seed=42)
        s2 = stratified_split(records, CONFIG, seed=42)
        t1 = [r.issue_id for r in s1.train]
        t2 = [r.issue_id for r in s2.train]
        assert t1 == t2
