"""Unit tests for data_engineering.diff_utils (shared diff parser)."""

from __future__ import annotations

from data_engineering.diff_utils import diff_git_paths, parse_files


class TestParseFilesGitStyle:
    def test_single_file(self) -> None:
        patch = "diff --git a/src/app.py b/src/app.py\n@@ -1,2 +1,2 @@\n x\n+x\n"
        assert parse_files(patch) == ["src/app.py"]

    def test_multi_file_order_preserved(self) -> None:
        patch = (
            "diff --git a/one.py b/one.py\n@@ -1 +1 @@\n"
            "diff --git a/two/two.py b/two/two.py\n@@ -1 +1 @@\n"
        )
        assert parse_files(patch) == ["one.py", "two/two.py"]

    def test_dedup(self) -> None:
        patch = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        assert parse_files(patch) == ["x.py"]


class TestParseFilesHeaderStyle:
    def test_bare_headers_no_diff_git(self) -> None:
        patch = "--- a/auth.py\n+++ b/auth.py\n@@ -1,2 +1,2 @@\n x\n"
        assert parse_files(patch) == ["auth.py"]

    def test_multi_file_bare_headers(self) -> None:
        patch = (
            "--- a/auth.py\n+++ b/auth.py\n@@ -1,2 +1,2 @@\n x\n"
            "--- a/register.py\n+++ b/register.py\n@@ -1,2 +1,2 @@\n y\n"
        )
        assert parse_files(patch) == ["auth.py", "register.py"]

    def test_bare_repo_relative_paths(self) -> None:
        patch = "--- src/foo.py\n+++ src/foo.py\n@@ -1 +1 @@\n"
        assert parse_files(patch) == ["src/foo.py"]


class TestParseFilesDevNull:
    def test_new_file(self) -> None:
        patch = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1 @@\n"
            "+x\n"
        )
        assert parse_files(patch) == ["new.py"]

    def test_deleted_file(self) -> None:
        patch = (
            "diff --git a/gone.py /dev/null\n"
            "deleted file mode 100644\n"
            "--- a/gone.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-x\n"
        )
        assert parse_files(patch) == ["gone.py"]

    def test_new_file_bare_headers_only(self) -> None:
        patch = "--- /dev/null\n+++ b/fresh.py\n@@ -0,0 +1 @@\n+x\n"
        assert parse_files(patch) == ["fresh.py"]


class TestParseFilesRename:
    def test_rename_b_side_wins(self) -> None:
        patch = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 90%\n"
            "rename from old.py\n"
            "rename to new.py\n"
            "index 1234567..89abcde 100644\n"
            "--- a/old.py\n"
            "+++ b/new.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        # Post-image of a rename is the new name; the a-side (old.py) must
        # not appear, otherwise _validate_applied chokes on the missing path.
        assert parse_files(patch) == ["new.py"]

    def test_rename_bare_headers_only(self) -> None:
        patch = "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-o\n+n\n"
        assert parse_files(patch) == ["new.py"]


class TestParseFilesQuotedAndEdge:
    def test_quoted_paths_with_spaces(self) -> None:
        patch = 'diff --git "a/dir with space/f.py" "b/dir with space/f.py"\n@@ -1 +1 @@\n'
        assert parse_files(patch) == ["dir with space/f.py"]

    def test_empty(self) -> None:
        assert parse_files("") == []
        assert parse_files("   \n") == []

    def test_non_diff_text(self) -> None:
        assert parse_files("no diff markers here") == []

    def test_mixed_styles(self) -> None:
        patch = (
            "diff --git a/git.py b/git.py\n"
            "--- a/git.py\n"
            "+++ b/git.py\n"
            "@@ -1 +1 @@\n"
            "--- a/plain.py\n"
            "+++ b/plain.py\n"
            "@@ -1 +1 @@\n"
        )
        assert parse_files(patch) == ["git.py", "plain.py"]


class TestDiffGitPaths:
    def test_plain(self) -> None:
        assert diff_git_paths("diff --git a/old.py b/new.py") == ("old.py", "new.py")

    def test_quoted(self) -> None:
        assert diff_git_paths('diff --git "a/dir/f.py" "b/dir/f.py"') == ("dir/f.py", "dir/f.py")

    def test_deleted_file_returns_none(self) -> None:
        assert diff_git_paths("diff --git a/gone.py /dev/null") is None

    def test_not_a_diff_git_line(self) -> None:
        assert diff_git_paths("--- a/x.py") is None
        assert diff_git_paths("@@ -1 +1 @@") is None
