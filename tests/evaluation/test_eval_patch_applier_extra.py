"""Edge-branch coverage for ``evaluation.patch_applier``.

Complements ``tests/test_eval_unit.py`` (real git/unidiff fixtures) with the
branches that need mocking: GNU-patch subprocess paths (skipped on macOS
without GNU patch), threading timeouts, ``_find_target`` fallbacks, the
``--directory=`` prefix retry, ``_validate_applied`` revert, and the
repaired-hunk-header retry chain. No real W&B/Modal/network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import evaluation.patch_applier as pa

GIT_DIFF = """\
diff --git a/greeting.py b/greeting.py
--- a/greeting.py
+++ b/greeting.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"Hello {name}"
+    return f"Hello {name}!"
"""


def _error(res: pa.PatchApplicationResult) -> str:
    assert res.error is not None
    return res.error


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo


def _rev_parse(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


# ── _error_message / _files_from_patch / _find_target ──────────────────────


class TestHelpers:
    def test_error_message_cap_detail(self) -> None:
        exc = subprocess.CalledProcessError(1, ["git"], output="o", stderr="boom " * 500)
        msg = pa._error_message(exc)
        assert "(boom boom" in msg

    def test_error_message_calledprocess_no_detail(self) -> None:
        exc = subprocess.CalledProcessError(1, ["git"], output=None, stderr=None)
        assert pa._error_message(exc) == str(exc)

    def test_files_from_patch_dedups_and_skips_dev_null(self) -> None:
        patch = "+++ b/x.py\ncontext\n+++ b/x.py\n+++ /dev/null\n+++ a/y.txt\n"
        assert pa._files_from_patch(patch) == ["x.py", "y.txt"]

    def test_find_target_exact_hit(self, tmp_path: Path) -> None:
        (tmp_path / "exact.py").write_text("x")
        assert pa._find_target(tmp_path, "exact.py") == tmp_path / "exact.py"

    def test_find_target_prefix_hit(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "pkg.py").write_text("x")
        assert pa._find_target(tmp_path, "pkg.py") == src / "pkg.py"

    def test_find_target_rglob_hit(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "deep"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("x")
        assert pa._find_target(tmp_path, "deep.py") == nested / "deep.py"

    def test_find_target_rglob_dir_only(self, tmp_path: Path) -> None:
        # rglob finds a *directory* named like the target -> not a file -> skip
        (tmp_path / "src" / "target.py").mkdir(parents=True)
        assert pa._find_target(tmp_path, "target.py") is None

    def test_find_target_no_match(self, tmp_path: Path) -> None:
        assert pa._find_target(tmp_path, "ghost.py") is None

    def test_find_target_oserror_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _BoomThread:
            daemon = False

            def __init__(self, *a: object, **k: object) -> None:
                raise OSError("boom")

        monkeypatch.setattr(pa.threading, "Thread", _BoomThread)
        assert pa._find_target(tmp_path, "ghost.py") is None


# ── apply_patch_git branches ────────────────────────────────────────────────


class TestApplyPatchGitExtra:
    def test_skip_checkout_success(self, git_repo: Path) -> None:
        patch = GIT_DIFF + "\n"
        (git_repo / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
        result = pa.apply_patch_git(git_repo, patch, "deadbeef" * 5, skip_checkout=True)
        assert result.success
        assert result.method_used == "git_apply"

    def test_prefix_directory_retry_success(self, git_repo: Path) -> None:
        # File lives under src/; the model patch omits the prefix.
        src = git_repo / "src"
        src.mkdir()
        (src / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
        (git_repo / "greeting.py").unlink()
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "move to src")
        patch = GIT_DIFF + "\n"
        result = pa.apply_patch_git(git_repo, patch, _rev_parse(git_repo))
        assert result.success
        assert result.method_used == "git_apply"
        assert "greeting.py" in result.files_modified
        assert "!" in (src / "greeting.py").read_text()

    def test_prefix_retry_skip_failing_prefix(self, git_repo: Path) -> None:
        # src/ has the file with wrong content (apply fails there); packages/
        # has the right content -> loop continues to the working prefix.
        src = git_repo / "src"
        src.mkdir()
        (src / "greeting.py").write_text("completely different content\n")
        pkgs = git_repo / "packages"
        pkgs.mkdir()
        (pkgs / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
        (git_repo / "greeting.py").unlink()
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "both prefixes")
        patch = GIT_DIFF + "\n"
        result = pa.apply_patch_git(git_repo, patch, _rev_parse(git_repo))
        assert result.success
        assert result.method_used == "git_apply"
        assert "greeting.py" in result.files_modified

    def test_apply_check_ok_apply_fails(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = pa._run_git

        def fake_run(repo: Path, args: list[str], stdin: str | None = None) -> str:
            if args[:2] == ["apply", "--check"]:
                return "ok"
            if args[0] == "apply":
                raise subprocess.CalledProcessError(1, ["git", "apply"], stderr="boom")
            return real(repo, args, stdin)

        monkeypatch.setattr(pa, "_run_git", fake_run)
        result = pa.apply_patch_git(git_repo, GIT_DIFF, _rev_parse(git_repo))
        assert not result.success
        assert "git apply:" in _error(result)

    def test_checkout_timeout_error(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*_a: object, **_k: object) -> str:
            raise subprocess.TimeoutExpired("git", 600)

        monkeypatch.setattr(pa, "_run_git", fake_run)
        result = pa.apply_patch_git(git_repo, GIT_DIFF, "c0ffee" * 5)
        assert not result.success
        assert "checkout" in _error(result)

    def test_prefix_loop_skipped_when_no_files_in_patch(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # git apply fails AND the patch has no "+++ " headers -> files == [].
        real = pa._run_git

        def fake_run(repo: Path, args: list[str], stdin: str | None = None) -> str:
            if args[0] == "checkout":
                return real(repo, args, stdin)
            raise subprocess.CalledProcessError(1, ["git", "apply"], stderr="no")

        monkeypatch.setattr(pa, "_run_git", fake_run)
        result = pa.apply_patch_git(
            git_repo, "not a patch at all\nplain text\n", _rev_parse(git_repo)
        )
        assert not result.success
        assert "git apply --check:" in _error(result)


# ── GNU patch path (normally skipif'd on macOS) ─────────────────────────────


class TestGnuPatch:
    def test_gnu_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        assert pa._gnu_patch_supported() is False
        result = pa.apply_patch_gnu(Path(), GIT_DIFF)
        assert not result.success
        assert "gnu patch not installed" in _error(result)

    def test_gnu_version_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired("patch", 10)

        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/patch")
        monkeypatch.setattr(pa.subprocess, "run", boom)
        assert pa._gnu_patch_supported() is False

    @pytest.fixture
    def gnu_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pa, "_gnu_patch_supported", lambda: True)

    def test_empty_patch(self, gnu_true) -> None:
        result = pa.apply_patch_gnu(Path(), "")
        assert not result.success
        assert result.error == "patch is empty"

    def test_dry_run_fails(self, gnu_true, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(args: list[str], *_a: object, **_k: object) -> subprocess.CompletedProcess:
            dry = "--dry-run" in args
            return subprocess.CompletedProcess(
                args, 1 if dry else 0, stdout="", stderr="stuck hunk"
            )

        monkeypatch.setattr(pa.subprocess, "run", fake_run)
        result = pa.apply_patch_gnu(Path(), GIT_DIFF)
        assert not result.success
        assert "dry-run" in _error(result)

    def test_apply_fails(self, gnu_true, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(args: list[str], *_a: object, **_k: object) -> subprocess.CompletedProcess:
            dry = "--dry-run" in args
            return subprocess.CompletedProcess(
                args, 0 if dry else 2, stdout="", stderr="reversed patch"
            )

        monkeypatch.setattr(pa.subprocess, "run", fake_run)
        result = pa.apply_patch_gnu(Path(), GIT_DIFF)
        assert not result.success
        assert "gnu patch (2)" in _error(result)

    def test_run_timeout(self, gnu_true, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired("patch", 600)

        monkeypatch.setattr(pa.subprocess, "run", boom)
        result = pa.apply_patch_gnu(Path(), GIT_DIFF)
        assert not result.success
        assert "gnu patch:" in _error(result)

    def test_success(self, gnu_true, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(args: list[str], *_a: object, **_k: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(pa.subprocess, "run", fake_run)
        result = pa.apply_patch_gnu(tmp_path, GIT_DIFF)
        assert result.success
        assert result.method_used == "gnu_patch_fuzz"
        assert result.files_modified == ["greeting.py"]

    def test_trailing_newline_added(
        self, gnu_true, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(args: list[str], *_a: object, **_k: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(pa.subprocess, "run", fake_run)
        result = pa.apply_patch_gnu(tmp_path, "+++ b/x.py\n@@ -1 +1 @@\n+a")
        assert result.success


# ── unidiff edge branches ───────────────────────────────────────────────────


class TestUnidiffExtra:
    def test_parse_exception(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(_patch: str) -> object:
            raise ValueError("bad hunk")

        monkeypatch.setattr(pa, "PatchSet", boom)
        result = pa.apply_patch_unidiff(tmp_path, GIT_DIFF)
        assert not result.success
        assert "unidiff parse:" in _error(result)

    def test_parse_times_out(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class _LazyThread:
            daemon = False

            def __init__(self, *a: object, **k: object) -> None:
                return None

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return True

        monkeypatch.setattr(pa.threading, "Thread", _LazyThread)
        result = pa.apply_patch_unidiff(tmp_path, GIT_DIFF)
        assert not result.success
        assert "timed out" in _error(result)

    def test_find_target_resolves_missing_file(self, tmp_path: Path) -> None:
        # Patch names a path that doesn't exist, but the basename is findable.
        repo = tmp_path / "repo"
        nested = repo / "pkg" / "deep"
        nested.mkdir(parents=True)
        (nested / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
        patch = GIT_DIFF.replace("b/greeting.py", "b/misplaced/greeting.py").replace(
            "--- a/greeting.py", "--- a/misplaced/greeting.py"
        )
        result = pa.apply_patch_unidiff(repo, patch)
        assert result.success
        assert "pkg/deep/greeting.py" in result.files_modified

    def test_unidiff_apply_missing_no_resolution(self, tmp_path: Path) -> None:
        result = pa.apply_patch_unidiff(tmp_path, GIT_DIFF)
        assert not result.success
        assert "unidiff apply:" in _error(result)

    def test_file_change_times_out(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # First thread (parse) runs to completion; second (file change) hangs.
        class _PhaseThread:
            created = 0

            daemon = False

            def __init__(self, *a: object, **k: object) -> None:
                _PhaseThread.created += 1
                target = k.get("target")
                assert callable(target)
                self._target: object = target

            def start(self) -> None:
                if _PhaseThread.created == 1 and callable(self._target):
                    self._target()

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return _PhaseThread.created > 1

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "greeting.py").write_text('def greet(name):\n    return f"Hello {name}"\n')
        monkeypatch.setattr(pa.threading, "Thread", _PhaseThread)
        result = pa.apply_patch_unidiff(repo, GIT_DIFF)
        assert not result.success
        assert "timed out" in _error(result)

    def test_repair_hunk_headers_malformed_header(self) -> None:
        patch = "@@ broken header @@\n context line\n"
        assert pa._repair_hunk_headers(patch) == patch


# ── _validate_applied ──────────────────────────────────────────────────────


class TestValidateApplied:
    def _ok(self, files: list[str]) -> pa.PatchApplicationResult:
        return pa.PatchApplicationResult(
            success=True, method_used="git_apply", files_modified=files
        )

    def test_non_python_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello\ndef x():\n  return\n")  # invalid py but not .py
        (tmp_path / "b.py").write_text("x = 1\n")
        result = pa._validate_applied(tmp_path, self._ok(["a.txt", "b.py"]))
        assert result.success

    def test_invalid_syntax_reverts(self, git_repo: Path) -> None:
        (git_repo / "bad.py").write_text("if True:\npass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "bad state")
        result = pa._validate_applied(git_repo, self._ok(["bad.py"]))
        assert not result.success
        assert "applied but invalid syntax" in _error(result)
        assert "bad.py" not in (git_repo / "bad.py").read_text()

    def test_revert_failure_swallowed(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (git_repo / "bad.py").write_text("if True:\npass\n")
        _git(git_repo, "add", ".")
        _git(git_repo, "commit", "-m", "bad state")

        def boom(*_a: object, **_k: object) -> str:
            raise subprocess.CalledProcessError(1, ["git"], stderr="nope")

        monkeypatch.setattr(pa, "_run_git", boom)
        result = pa._validate_applied(git_repo, self._ok(["bad.py"]))
        assert not result.success
        assert "applied but invalid syntax" in _error(result)


# ── apply_patch orchestration retry chain ──────────────────────────────────


class TestApplyPatchChain:
    def test_gnu_fallback_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        failed = pa.PatchApplicationResult(success=False, method_used="failed", error="no")
        ok = pa.PatchApplicationResult(success=True, method_used="gnu_patch_fuzz")
        monkeypatch.setattr(pa, "apply_patch_git", lambda *a, **k: failed)
        monkeypatch.setattr(pa, "apply_patch_gnu", lambda *a, **k: ok)
        monkeypatch.setattr(pa, "_validate_applied", lambda _r, res: res)
        result = pa.apply_patch(Path(), GIT_DIFF, "x" * 40)
        assert result.success

    def test_repaired_headers_retry_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        failed = pa.PatchApplicationResult(
            success=False, method_used="failed", error="git apply --check: boom"
        )
        failed_gnu = pa.PatchApplicationResult(
            success=False, method_used="gnu_patch_fuzz", error="gnu patch dry-run (1): nope"
        )
        failed_uni = pa.PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error="unidiff apply: nope"
        )

        def _fake_git(repo: Path, patch: str, base: str, **k: object) -> pa.PatchApplicationResult:
            return failed

        def _fake_gnu(repo: Path, patch: str) -> pa.PatchApplicationResult:
            return failed_gnu

        def _fake_uni(repo: Path, patch: str) -> pa.PatchApplicationResult:
            return failed_uni

        monkeypatch.setattr(pa, "apply_patch_git", _fake_git)
        monkeypatch.setattr(pa, "apply_patch_gnu", _fake_gnu)
        monkeypatch.setattr(pa, "apply_patch_unidiff", _fake_uni)
        monkeypatch.setattr(pa, "_validate_applied", lambda _r, res: res)
        monkeypatch.setattr(pa, "_repair_hunk_headers", lambda _p: GIT_DIFF + "\nchanged")
        result = pa.apply_patch(Path(), GIT_DIFF, "x" * 40)
        assert not result.success
        assert "repaired: git:" in _error(result)

    @pytest.mark.parametrize(
        ("gain", "method"),
        [
            ("git", "git_apply"),
            ("gnu", "gnu_patch_fuzz"),
            ("unidiff", "unidiff_fallback"),
        ],
    )
    def test_repaired_header_retry_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, gain: str, method: str
    ) -> None:
        REPAIRED = "REPAIRED"  # noqa: N806
        failed = pa.PatchApplicationResult(success=False, method_used="failed", error="boom")
        failed_gnu = pa.PatchApplicationResult(
            success=False, method_used="gnu_patch_fuzz", error="nope"
        )
        failed_uni = pa.PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error="nope"
        )
        ok = pa.PatchApplicationResult(success=True, method_used=method)  # type: ignore[arg-type]

        def _fake_git(repo: Path, patch: str, base: str, **k: object) -> pa.PatchApplicationResult:
            return ok if gain == "git" and patch == REPAIRED else failed

        def _fake_gnu(repo: Path, patch: str) -> pa.PatchApplicationResult:
            return ok if gain == "gnu" and patch == REPAIRED else failed_gnu

        def _fake_uni(repo: Path, patch: str) -> pa.PatchApplicationResult:
            return ok if gain == "unidiff" and patch == REPAIRED else failed_uni

        monkeypatch.setattr(pa, "apply_patch_git", _fake_git)
        monkeypatch.setattr(pa, "apply_patch_gnu", _fake_gnu)
        monkeypatch.setattr(pa, "apply_patch_unidiff", _fake_uni)
        monkeypatch.setattr(pa, "_validate_applied", lambda _r, res: res)
        monkeypatch.setattr(pa, "_repair_hunk_headers", lambda _p: REPAIRED)
        result = pa.apply_patch(Path(), GIT_DIFF, "x" * 40)
        assert result.success
        assert result.method_used == method
