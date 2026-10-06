"""Table-driven tests for the six quality gates in ``data_engineering.clean``.

Each gate decides whether a SWE-bench sample is admitted to (or dropped from)
training data. The gates were previously only exercised indirectly through
``clean_records``; these tests pin each gate's accept path, reject path, and
boundary conditions directly, using realistic unified diffs.
"""

from __future__ import annotations

import pytest

from data_engineering.clean import (
    PYTHON_RATIO_THRESHOLD,
    _check_binary,
    _check_empty_body,
    _check_non_python,
    _check_patch_size,
    _has_f2p_keywords,
    _is_binary_diff,
    clean_records,
)
from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord

pytestmark = pytest.mark.unit

# ── Realistic fixtures ─────────────────────────────────────────────────────

TEXT_DIFF = (
    "diff --git a/src/parser.py b/src/parser.py\n"
    "index 3f9a1c2..8d4e5b6 100644\n"
    "--- a/src/parser.py\n"
    "+++ b/src/parser.py\n"
    "@@ -10,7 +10,9 @@ def parse(raw: str) -> dict:\n"
    "     if not raw:\n"
    "-        return {}\n"
    "+        raise ValueError('empty input')\n"
    "+    return json.loads(raw)\n"
)

BINARY_MARKER_DIFF = (
    "diff --git a/assets/logo.png b/assets/logo.png\n"
    "index e69de29..ab2ed5d 100644\n"
    "Binary files a/assets/logo.png and b/assets/logo.png differ\n"
)

SINGULAR_BINARY_MARKER = "Binary file img.png differs\n"

# A purely textual diff whose *added line* mentions the phrase "Binary file".
# It is not a git binary marker; before the fix this was misclassified.
PHRASE_DIFF = (
    "--- a/src/detect.py\n"
    "+++ b/src/detect.py\n"
    "@@ -3,4 +3,5 @@ def detect(path):\n"
    "     data = path.read_bytes()\n"
    '+    # git emits a "Binary file" marker for non-text files\n'
    "     return None\n"
)


def _record(**overrides: object) -> IssueRecord:
    """Build a valid IssueRecord with sensible defaults; override per test."""
    fields: dict = {
        "issue_id": "owner/repo#123",
        "repo": "owner/repo",
        "issue_body": "Parser raises on empty input instead of returning {}.",
        "patch_diff": TEXT_DIFF,
        "files_changed": ["src/parser.py"],
        "test_files_changed": ["tests/test_parser.py"],
        "commit_messages": ["fix: handle empty input in parser"],
        "pr_description": "",
    }
    fields.update(overrides)
    return IssueRecord(**fields)


def _record_bypassing_validators(issue_body: str) -> IssueRecord:
    """Record built via ``model_construct``.

    The schema validator rejects empty/whitespace-only issue bodies at
    construction time, but the gate must still catch them for records that
    reach ``clean_records`` through paths that skip validation.
    """
    return IssueRecord.model_construct(
        issue_id="owner/repo#7",
        repo="owner/repo",
        issue_body=issue_body,
        patch_diff=TEXT_DIFF,
        files_changed=["src/parser.py"],
        test_files_changed=["tests/test_parser.py"],
        commit_messages=["fix: handle empty input"],
        pr_description="",
        metadata={},
    )


def _diff_with_lines(total: int) -> str:
    """A valid unified diff whose ``splitlines()`` length is exactly *total*."""
    assert total >= 5, "need room for headers plus one added line"
    removed = ["-    return None"]
    added = [f"+    value{i} = compute({i})" for i in range(total - 4)]
    header = [
        "--- a/src/parser.py",
        "+++ b/src/parser.py",
        f"@@ -10,{len(removed)} +10,{len(added)} @@ def parse(raw):",
    ]
    lines = header + removed + added
    assert len(lines) == total
    return "\n".join(lines) + "\n"


# ── _has_f2p_keywords ──────────────────────────────────────────────────────


class TestHasF2pKeywords:
    @pytest.mark.parametrize(
        ("commit_messages", "pr_description", "metadata", "expected"),
        [
            # keyword in commit message (each accepted stem form)
            (["fix: null pointer in lexer"], "", {}, True),
            (["Fixes #4821"], "", {}, True),
            (["Fixed: race in worker pool"], "", {}, True),
            (["fixing the flaky ordering test"], "", {}, True),
            (["Closes #991"], "", {}, True),
            (["closed as duplicate"], "", {}, True),
            (["resolve the deadlock in the queue"], "", {}, True),
            (["resolved merge fallout"], "", {}, True),
            # case-insensitive
            (["FIX THE BUG IN THE PARSER"], "", {}, True),
            # keyword in PR description instead of commits
            ([], "This PR fixes the off-by-one in pagination", {}, True),
            ([], "See linked issue", {"has_test_patch": True}, True),
            # metadata fallback: FAIL_TO_PASS test patch present
            (["chore: bump dependencies"], "", {"has_test_patch": True}, True),
            # reject: no keyword anywhere, no metadata signal
            (["chore: bump dependencies"], "", {}, False),
            (["refactor: rename variables"], "Routine cleanup", {}, False),
            (["chore: bump deps"], "", {"has_test_patch": False}, False),
            # reject: stem only matches on word boundaries
            (["prefix the config keys"], "", {}, False),
            (["the affix table is wrong"], "", {}, False),
            (["fixup! correct the typo"], "", {}, False),
            # Documenting current pattern scope (not asserting this is ideal):
            # only the "fix" stem accepts the -ing form; "closing"/"resolving"
            # gerunds do not match, mirroring GitHub's keyword list which also
            # excludes gerunds.
            (["closing #123"], "", {}, False),
            (["resolving #456"], "", {}, False),
        ],
        ids=[
            "fix",
            "fixes",
            "fixed",
            "fixing",
            "closes",
            "closed",
            "resolve",
            "resolved",
            "case-insensitive",
            "keyword-in-pr-description",
            "metadata-fallback",
            "metadata-without-keyword",
            "no-keyword",
            "no-keyword-with-description",
            "metadata-false",
            "prefix-not-match",
            "affix-not-match",
            "fixup-not-match",
            "closing-gerund-current",
            "resolving-gerund-current",
        ],
    )
    def test_keyword_matrix(
        self,
        commit_messages: list[str],
        pr_description: str,
        metadata: dict,
        expected: bool,
    ) -> None:
        record = _record(
            commit_messages=commit_messages,
            pr_description=pr_description,
            metadata=metadata,
        )
        assert _has_f2p_keywords(record) is expected

    def test_empty_signals_rejected(self) -> None:
        record = _record(commit_messages=[], pr_description="", metadata={})
        assert _has_f2p_keywords(record) is False


# ── _is_binary_diff ────────────────────────────────────────────────────────


class TestIsBinaryDiff:
    @pytest.mark.parametrize(
        ("diff", "expected"),
        [
            # git's plural marker, as emitted by GitHub .diff endpoints
            (BINARY_MARKER_DIFF, True),
            # older git emitted the singular form
            (SINGULAR_BINARY_MARKER, True),
            # marker appearing mid-diff after index header
            (
                "diff --git a/x.bin b/x.bin\nBinary files /dev/null and b/x.bin differ\n",
                True,
            ),
            # ordinary text diff
            (TEXT_DIFF, False),
            # BUG FIX regression: a text diff whose added line merely
            # *mentions* the phrase must not be treated as a binary diff.
            (PHRASE_DIFF, False),
            # an added line that adds the literal marker text is still content
            (
                "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n+Binary files a/x and b/x differ\n",
                False,
            ),
        ],
        ids=[
            "plural-marker",
            "singular-marker",
            "marker-after-index",
            "text-diff",
            "phrase-in-added-line",
            "marker-as-added-line",
        ],
    )
    def test_binary_matrix(self, diff: str, expected: bool) -> None:
        assert _is_binary_diff(diff) is expected

    def test_empty_diff(self) -> None:
        assert _is_binary_diff("") is False


# ── _check_binary ──────────────────────────────────────────────────────────


class TestCheckBinary:
    def test_binary_marker_returns_reason(self) -> None:
        record = _record(patch_diff=BINARY_MARKER_DIFF)
        assert _check_binary(record) == "binary"

    def test_text_diff_passes(self) -> None:
        assert _check_binary(_record()) is None

    def test_text_diff_mentioning_binary_phrase_passes(self) -> None:
        """Regression for the substring-match bug: content is not a marker."""
        assert _check_binary(_record(patch_diff=PHRASE_DIFF)) is None


# ── _check_patch_size ──────────────────────────────────────────────────────


class TestCheckPatchSize:
    @pytest.mark.parametrize("max_lines", [10, 500])
    def test_patch_exactly_at_threshold_accepted(self, max_lines: int) -> None:
        record = _record(patch_diff=_diff_with_lines(max_lines))
        assert len(record.patch_diff.splitlines()) == max_lines
        assert _check_patch_size(record, max_lines) is None

    @pytest.mark.parametrize("max_lines", [10, 500])
    def test_patch_one_line_over_threshold_rejected(self, max_lines: int) -> None:
        record = _record(patch_diff=_diff_with_lines(max_lines + 1))
        assert _check_patch_size(record, max_lines) == "patch_too_large"

    def test_small_realistic_patch_accepted(self) -> None:
        assert _check_patch_size(_record(), 500) is None

    def test_trailing_newline_not_counted(self) -> None:
        """``splitlines`` ignores the trailing newline: 500 lines + \\n stays 500."""
        record = _record(patch_diff=_diff_with_lines(500))
        assert _check_patch_size(record, 500) is None
        assert record.patch_diff.endswith("\n")


# ── _check_non_python ──────────────────────────────────────────────────────


class TestCheckNonPython:
    @pytest.mark.parametrize(
        ("files_changed", "expected_reason", "expect_warning"),
        [
            # no file info at all: kept, silent
            ([], None, False),
            # pure-python changes: kept, silent
            (["src/parser.py"], None, False),
            (["src/parser.py", "tests/test_parser.py"], None, False),
            # ratio exactly at threshold: kept, but warned
            (["src/parser.py", "web/app.js"], None, True),
            (["src/a.py", "src/b.py", "web/c.js", "web/d.js"], None, True),
            # ratio below threshold: removed
            (["src/parser.py", "web/app.js", "web/style.css"], "non_python", None),
            (["web/app.js"], "non_python", None),
            (["a.py", "b.js", "c.js", "d.js"], "non_python", None),
            (["a.js", "b.js"], "non_python", None),
        ],
        ids=[
            "empty-list",
            "single-py",
            "two-py",
            "exactly-half",
            "exactly-half-four-files",
            "one-of-three",
            "no-py",
            "one-of-four",
            "no-py-two-files",
        ],
    )
    def test_ratio_matrix(
        self,
        files_changed: list[str],
        expected_reason: str | None,
        expect_warning: bool,
    ) -> None:
        reason, warning = _check_non_python(_record(files_changed=files_changed))
        assert reason == expected_reason
        if expect_warning:
            assert warning is not None
            assert warning.startswith("non_python_files:")
        else:
            assert warning is None

    def test_threshold_constant(self) -> None:
        """Pin the documented threshold: half the changed files must be Python."""
        assert PYTHON_RATIO_THRESHOLD == 0.5


# ── _check_empty_body ──────────────────────────────────────────────────────


class TestCheckEmptyBody:
    @pytest.mark.parametrize(
        "issue_body",
        ["", "   ", "\n\t  \n"],
        ids=["empty", "spaces", "whitespace-only"],
    )
    def test_empty_bodies_rejected(self, issue_body: str) -> None:
        assert _check_empty_body(_record_bypassing_validators(issue_body)) == "empty_body"

    @pytest.mark.parametrize(
        "issue_body",
        [
            "Parser raises on empty input instead of returning {}.",
            "x",
            "First line\nsecond line with more detail",
        ],
        ids=["normal", "single-char", "multiline"],
    )
    def test_non_empty_bodies_accepted(self, issue_body: str) -> None:
        assert _check_empty_body(_record(issue_body=issue_body)) is None


# ── End-to-end regression through clean_records ────────────────────────────


class TestCleanRecordsRegression:
    def test_text_diff_mentioning_binary_phrase_is_kept(
        self,
    ) -> None:
        """A text patch that mentions "Binary file" must survive cleaning.

        Before the ``_is_binary_diff`` fix the substring match dropped this
        record via the ``binary`` filter.
        """
        config = DataPipelineConfig(max_patch_lines=500)
        record = _record(patch_diff=PHRASE_DIFF)
        cleaned, stats = clean_records([record], config)
        assert len(cleaned) == 1
        assert stats.removed_binary == 0

    def test_real_binary_marker_still_removed(self) -> None:
        config = DataPipelineConfig(max_patch_lines=500)
        record = _record(patch_diff=BINARY_MARKER_DIFF)
        cleaned, stats = clean_records([record], config)
        assert cleaned == []
        assert stats.removed_binary == 1
