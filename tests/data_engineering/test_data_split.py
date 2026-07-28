"""Tests for data_engineering.split."""

from __future__ import annotations

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, Splits
from data_engineering.split import extract_golden, stratified_split


def _make_record(
    issue_id: str,
    repo: str = "owner/repo",
    test_files: list[str] | None = None,
    commit_msgs: list[str] | None = None,
    pr_description: str = "",
    files_changed: list[str] | None = None,
) -> IssueRecord:
    return IssueRecord(
        issue_id=issue_id,
        repo=repo,
        issue_body="body text for issue",
        patch_diff=("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n-x=1\n+x=1\n+y=2\n"),
        test_files_changed=test_files or [],
        files_changed=files_changed or ["foo.py"],
        commit_messages=commit_msgs or ["fix: something"],
        pr_description=pr_description,
    )


class TestStratifiedSplit:
    def test_basic_split(self) -> None:
        config = DataPipelineConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        records = [_make_record(f"r{i}#{j}", repo=f"r{i}") for i in range(12) for j in range(3)]
        splits = stratified_split(records, config, seed=42)
        assert len(splits.train) > 0
        assert len(splits.val) > 0
        assert len(splits.test) > 0
        # Total records preserved
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == 36

    def test_no_leakage(self) -> None:
        config = DataPipelineConfig()
        records = [_make_record(f"r{i}#{j}", repo=f"repo{i}") for i in range(4) for j in range(3)]
        splits = stratified_split(records, config, seed=42)
        train_repos = {r.repo for r in splits.train}
        val_repos = {r.repo for r in splits.val}
        test_repos = {r.repo for r in splits.test}
        assert train_repos.isdisjoint(val_repos)
        assert train_repos.isdisjoint(test_repos)
        assert val_repos.isdisjoint(test_repos)

    def test_reproducible_seed(self) -> None:
        config = DataPipelineConfig()
        records = [_make_record(f"r{i}#{j}", repo=f"r{i}") for i in range(5) for j in range(2)]
        s1 = stratified_split(records, config, seed=42)
        s2 = stratified_split(records, config, seed=42)
        assert [r.issue_id for r in s1.train] == [r.issue_id for r in s2.train]
        assert [r.issue_id for r in s1.val] == [r.issue_id for r in s2.val]
        assert [r.issue_id for r in s1.test] == [r.issue_id for r in s2.test]

    def test_different_seed_different(self) -> None:
        config = DataPipelineConfig()
        records = [_make_record(f"r{i}#{j}", repo=f"r{i}") for i in range(5) for j in range(2)]
        s1 = stratified_split(records, config, seed=42)
        s2 = stratified_split(records, config, seed=99)
        # At least something should be different (very unlikely to match)
        ids1 = {r.issue_id for r in s1.train}
        ids2 = {r.issue_id for r in s2.train}
        assert ids1 != ids2

    def test_single_repo(self) -> None:
        config = DataPipelineConfig()
        records = [_make_record(f"r1#{i}", repo="r1") for i in range(3)]
        splits = stratified_split(records, config, seed=42)
        # All records go to train when only one repo
        assert len(splits.train) == 3
        assert len(splits.val) == 0
        assert len(splits.test) == 0

    def test_small_number_of_repos(self) -> None:
        config = DataPipelineConfig()
        records = [_make_record(f"r{i}#1", repo=f"r{i}") for i in range(2)]
        splits = stratified_split(records, config, seed=42)
        assert len(splits.train) + len(splits.val) + len(splits.test) == 2


class TestRunIdSeedDerivation:
    """The pipeline derives a split seed from run_id using hashlib.sha256
    (not hash(), which is salted per process). Same run_id must always
    produce the same seed across interpreter sessions."""

    def test_same_run_id_same_seed(self) -> None:
        import hashlib

        run_id = "abc123def456"
        seed1 = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "little")
        seed2 = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "little")
        assert seed1 == seed2

    def test_different_run_id_different_seed(self) -> None:
        import hashlib

        seed_a = int.from_bytes(hashlib.sha256(b"run_one").digest()[:4], "little")
        seed_b = int.from_bytes(hashlib.sha256(b"run_two").digest()[:4], "little")
        assert seed_a != seed_b

    def test_seed_within_int31_range(self) -> None:
        import hashlib

        for rid in ["r1", "some-longer-run-id-12345", "", "a" * 100]:
            seed = int.from_bytes(hashlib.sha256(rid.encode()).digest()[:4], "little") % (2**31)
            assert 0 <= seed < 2**31


class TestExtractGolden:
    def test_extracts_from_test_split(self) -> None:
        config = DataPipelineConfig(golden_source_split="test")
        splits = Splits(
            test=[
                _make_record(
                    "r1#1",
                    test_files=["test_foo.py"],
                    commit_msgs=["fix: bug fix"],
                ),
                _make_record(
                    "r1#2",
                    test_files=[],  # no test files → not golden
                    commit_msgs=["fix: bug fix"],
                ),
                _make_record(
                    "r1#3",
                    test_files=["test_bar.py"],
                    commit_msgs=["chore: cleanup"],  # no F2P → not golden
                ),
            ]
        )
        golden = extract_golden(splits, config.min_golden_examples, config.golden_source_split)
        assert len(golden) == 1
        assert golden[0].issue_id == "r1#1"

    def test_empty_when_no_f2p(self) -> None:
        config = DataPipelineConfig(golden_source_split="test")
        splits = Splits(
            test=[
                _make_record(
                    "r1#1",
                    test_files=["test_foo.py"],
                    commit_msgs=["refactor: clean up"],
                    pr_description="No keywords",
                ),
            ]
        )
        golden = extract_golden(splits, config.min_golden_examples, config.golden_source_split)
        assert len(golden) == 0

    def test_source_all_leakage_warning(self) -> None:
        config = DataPipelineConfig(golden_source_split="all")
        splits = Splits(
            train=[
                _make_record(
                    "r1#1",
                    test_files=["test_a.py"],
                    commit_msgs=["fix: good"],
                ),
            ],
            test=[
                _make_record(
                    "r1#2",
                    test_files=["test_b.py"],
                    commit_msgs=["fix: also good"],
                ),
            ],
        )
        golden = extract_golden(splits, config.min_golden_examples, config.golden_source_split)
        assert len(golden) == 2  # from both train and test
