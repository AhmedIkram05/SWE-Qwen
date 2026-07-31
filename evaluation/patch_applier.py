"""Patch application: git apply with a manual unidiff fallback.

Primary path uses the ``git`` binary (checkout ``base_sha`` then ``git
apply``), which is the most faithful to the real workflow. When that
fails (non-git working dirs, index-less patches, malformed input), the
patch is parsed with :mod:`unidiff` and hunks are applied manually.
Never raises — always returns a :class:`PatchApplicationResult`.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from unidiff.patch import PatchedFile, PatchSet

from evaluation.schema import PatchApplicationResult

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECONDS = 600  # Match test_runner.py timeout for consistency
_DEV_NULL = "/dev/null"


def _run_git(repo_path: Path, args: list[str], stdin: str | None = None) -> str:
    """Run a git command in *repo_path*, returning stdout.

    Raises:
        subprocess.CalledProcessError: on non-zero exit.
        subprocess.TimeoutExpired: if the command exceeds the timeout.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            ["git", *args],
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def _error_message(exc: Exception) -> str:
    """Human-readable error from a subprocess failure."""
    if isinstance(exc, subprocess.CalledProcessError):
        detail = (exc.stderr or exc.output or "").strip()
        return f"{exc} ({detail})" if detail else str(exc)
    return str(exc)


def _files_from_patch(patch: str) -> list[str]:
    """Extract touched file paths from unified diff ``+++`` headers."""
    files: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        path = line[4:].strip()
        if path.startswith(("a/", "b/")):
            path = path[2:]
        if path and path != _DEV_NULL and path not in files:
            files.append(path)
    return files


def apply_patch_git(repo_path: Path, patch: str, base_sha: str) -> PatchApplicationResult:
    """Check out *base_sha*, then ``git apply`` the patch.

    Steps: ``git checkout --quiet <base_sha>`` → ``git apply --check
    --quiet -`` → ``git apply -`` (patch via stdin).

    Args:
        repo_path: Git repository root.
        patch: Unified diff (git format preferred).
        base_sha: Commit to reset the working tree to before applying.

    Returns:
        ``method_used="git_apply"`` on success, ``"failed"`` otherwise.
    """
    if not patch.strip():
        return PatchApplicationResult(success=False, method_used="failed", error="patch is empty")
    try:
        _run_git(repo_path, ["checkout", "--quiet", base_sha])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("git checkout %s failed in %s: %s", base_sha, repo_path, exc)
        return PatchApplicationResult(
            success=False, method_used="failed", error=f"checkout {base_sha}: {_error_message(exc)}"
        )
    try:
        _run_git(repo_path, ["apply", "--check", "--quiet", "-"], patch)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("git apply --check failed in %s: %s", repo_path, exc)
        return PatchApplicationResult(
            success=False, method_used="failed", error=f"git apply --check: {_error_message(exc)}"
        )
    try:
        _run_git(repo_path, ["apply", "-"], patch)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("git apply failed in %s: %s", repo_path, exc)
        return PatchApplicationResult(
            success=False, method_used="failed", error=f"git apply: {_error_message(exc)}"
        )
    return PatchApplicationResult(
        success=True, method_used="git_apply", files_modified=_files_from_patch(patch)
    )


def apply_patch_unidiff(repo_path: Path, patch: str) -> PatchApplicationResult:
    """Apply *patch* by parsing it with unidiff and rewriting files manually.

    Handles added, modified and removed files. Hunks are applied
    bottom-up so earlier hunk positions stay valid after later ones
    shift the file.

    Args:
        repo_path: Working directory (does not need to be a git repo).
        patch: Unified diff.

    Returns:
        ``method_used="unidiff_fallback"`` on success, ``"failed"`` otherwise.
    """
    if not patch.strip():
        return PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error="patch is empty"
        )

    # Add timeout protection for unidiff parsing
    def parse_patch():
        return PatchSet(patch)

    try:
        # Use threading timer to enforce timeout
        result_container = []  # type: list
        exception_container = []  # type: list

        def target():
            try:
                parsed_result = parse_patch()
                result_container.append(parsed_result)
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30 second timeout for parsing

        if thread.is_alive():
            return PatchApplicationResult(
                success=False, method_used="unidiff_fallback", error="unidiff parsing timed out"
            )

        if exception_container:
            raise exception_container[0]  # noqa: TRY301

        patchset = result_container[0]
    except Exception as exc:
        logger.warning("unidiff parse failed: %s", exc)
        return PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error=f"unidiff parse: {exc}"
        )
    if len(patchset) == 0:
        logger.warning("unidiff parsed no files from patch")
        return PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error="no files parsed"
        )
    files_modified: list[str] = []
    try:
        for pfile in patchset:
            _apply_file_change(repo_path / pfile.path, pfile)
            files_modified.append(pfile.path)
    except Exception as exc:
        logger.warning("manual unidiff apply failed in %s: %s", repo_path, exc)
        return PatchApplicationResult(
            success=False, method_used="unidiff_fallback", error=f"unidiff apply: {exc}"
        )
    return PatchApplicationResult(
        success=True, method_used="unidiff_fallback", files_modified=files_modified
    )


def _apply_file_change(target: Path, pfile: PatchedFile) -> None:
    """Rewrite *target* on disk according to one unidiff patched file.

    Hunks are applied bottom-up so earlier hunk positions stay valid
    after later ones shift the file.

    Args:
        target: Absolute path of the file to rewrite.
        pfile: A :class:`unidiff.PatchedFile`.

    Raises:
        FileNotFoundError: if a modified file is missing on disk.
        TimeoutError: if file operations take too long.
    """

    # Add timeout protection for file operations
    def do_file_change():
        if pfile.is_removed_file:
            target.unlink(missing_ok=True)
            return
        if not target.exists() and not pfile.is_added_file:
            raise FileNotFoundError(f"target file missing: {pfile.path}")
        lines = target.read_text().splitlines(keepends=True) if target.exists() else []
        for hunk in reversed(list(pfile)):
            start = hunk.source_start - 1
            end = start + hunk.source_length
            new_section = [line.value for line in hunk if line.is_added or line.is_context]
            lines[start:end] = new_section
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines))

    # Use threading timer to enforce timeout
    result_container = []  # type: list
    exception_container = []  # type: list

    def target_func():
        try:
            do_file_change()
            result_container.append(True)
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=target_func)
    thread.daemon = True
    thread.start()
    thread.join(timeout=30)  # 30 second timeout for file operations

    if thread.is_alive():
        raise TimeoutError(f"file operation timed out for {pfile.path}")

    if exception_container:
        raise exception_container[0]


def apply_patch(repo_path: Path, patch: str, base_sha: str) -> PatchApplicationResult:
    """Main entry: ``git apply`` first, unidiff manual apply as fallback.

    Args:
        repo_path: Working directory (git repo preferred).
        patch: Unified diff.
        base_sha: Commit to reset the working tree to before applying.

    Returns:
        The successful result (git or unidiff), or
        ``method_used="failed"`` when both attempts fail.
    """
    result = apply_patch_git(repo_path, patch, base_sha)
    if result.success:
        return result
    logger.warning("git apply failed (%s), trying unidiff fallback", result.error)
    fallback = apply_patch_unidiff(repo_path, patch)
    if fallback.success:
        return fallback
    return PatchApplicationResult(
        success=False,
        method_used="failed",
        error=f"git apply: {result.error}; unidiff fallback: {fallback.error}",
    )
