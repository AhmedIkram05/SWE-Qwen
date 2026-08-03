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


def _find_target(repo_path: Path, patch_path: str) -> Path | None:
    """Resolve a patch file path to an actual file in *repo_path*.

    Tries, in order:
    1. Exact path (``{repo_path}/{patch_path}``)
    2. Common source prefixes (``src/``, ``packages/``, ``lib/``)
    3. ``rglob`` by basename (last resort, timeout-guarded)

    Returns the first match or *None*.
    """
    target = repo_path / patch_path
    if target.is_file():
        return target
    for prefix in ("src", "packages", "lib"):
        target = repo_path / prefix / patch_path
        if target.is_file():
            return target
    # Last resort: search by basename (fast on SSD, rare edge case)
    try:
        import threading as _thr

        result_holder: list[Path | None] = [None]

        def _search() -> None:
            basename = Path(patch_path).name
            for f in repo_path.rglob(basename):
                if f.is_file():
                    result_holder[0] = f
                    return

        t = _thr.Thread(target=_search, daemon=True)
        t.start()
        t.join(timeout=5)
        return result_holder[0]
    except (PermissionError, OSError):
        return None


def apply_patch_git(
    repo_path: Path, patch: str, base_sha: str, *, skip_checkout: bool = False
) -> PatchApplicationResult:
    """Check out *base_sha*, then ``git apply`` the patch.

    Steps: ``git checkout --quiet <base_sha>`` → ``git apply --check
    --quiet -`` → ``git apply -`` (patch via stdin).

    Args:
        repo_path: Git repository root.
        patch: Unified diff (git format preferred).
        base_sha: Commit to reset the working tree to before applying.
        skip_checkout: If True, skip the ``git checkout`` step (caller already
            reset to ``base_sha``). Avoids redundant checkout on FUSE volumes.

    Returns:
        ``method_used="git_apply"`` on success, ``"failed"`` otherwise.
    """
    if not patch.strip():
        return PatchApplicationResult(success=False, method_used="failed", error="patch is empty")
    # ponytail: git apply via stdin needs a trailing newline; extract_patch
    # strips it, causing rc=128 "corrupt patch" on every valid patch.
    if not patch.endswith("\n"):
        patch = patch + "\n"
    if not skip_checkout:
        try:
            _run_git(repo_path, ["checkout", "--quiet", base_sha])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("git checkout %s failed in %s: %s", base_sha, repo_path, exc)
            return PatchApplicationResult(
                success=False,
                method_used="failed",
                error=f"checkout {base_sha}: {_error_message(exc)}",
            )
    try:
        _run_git(repo_path, ["apply", "--check", "--quiet", "-"], patch)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # The model may generate patch paths that don't match the repo's
        # layout (missing ``src/`` prefix, wrong nesting).  Try common
        # ``--directory`` prefixes when the target file doesn't exist at
        # the bare path but exists under a known prefix.
        files = _files_from_patch(patch)
        if files:
            for prefix in ("src", "packages", "lib"):
                prefix_dir = repo_path / prefix
                if prefix_dir.is_dir() and any(
                    not (repo_path / f).exists() and (prefix_dir / f).exists() for f in files
                ):
                    try:
                        _run_git(
                            repo_path,
                            ["apply", "--check", f"--directory={prefix}/", "--quiet", "-"],
                            patch,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        continue
                    _run_git(repo_path, ["apply", f"--directory={prefix}/", "-"], patch)
                    return PatchApplicationResult(
                        success=True,
                        method_used="git_apply",
                        files_modified=_files_from_patch(patch),
                    )
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
    # ponytail: normalize trailing newline (same reason as git apply)
    if not patch.endswith("\n"):
        patch = patch + "\n"

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
            rel = pfile.path
            target = repo_path / rel
            try:
                _apply_file_change(target, pfile)
            except FileNotFoundError:
                # Some repos (sqlfluff, astropy) use a ``src/`` layout.
                # Retry with the ``src/`` prefix when available.
                if (repo_path / "src").is_dir():
                    rel = "src/" + pfile.path
                    target = repo_path / rel
                    _apply_file_change(target, pfile)
                else:
                    raise
            files_modified.append(rel)
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


def _repair_hunk_headers(patch: str) -> str:
    """Recompute ``@@`` hunk line counts from the hunk bodies.

    Small models frequently emit ``@@ -l,o +l,n @@`` headers whose counts
    don't match the actual hunk content (e.g. ``+102,14`` for an 11-line
    hunk). Both ``git apply`` ("corrupt patch") and unidiff ("Hunk is
    shorter than expected") reject these outright even when the hunk
    content is otherwise correct. Counting the real lines and rewriting
    the header fixes the most common patch-apply failure.
    """
    import re

    header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    out: list[str] = []
    hunk_ctx: list[str] = []  # body lines collected since the last @@ header
    in_hunk = False

    def flush() -> None:
        nonlocal hunk_ctx, in_hunk
        if not in_hunk:
            return
        old_count = sum(1 for l in hunk_ctx if not l.startswith(("+", "\\")))  # noqa: E741
        new_count = sum(1 for l in hunk_ctx if not l.startswith(("-", "\\")))  # noqa: E741
        header = out.pop()  # the @@ line we appended while collecting
        m = header_re.match(header)
        if m:
            old_start, old_n, new_start, new_n = m.groups()
            old_spec = f"{old_start},{old_count}" if old_n is not None else old_start
            new_spec = f"{new_start},{new_count}" if new_n is not None else new_start
            section = header[m.end() :]  # keep trailing section name (e.g. " def foo():")
            out.append(f"@@ -{old_spec} +{new_spec} @@{section}")
        else:  # malformed header; leave as-is
            out.append(header)
        out.extend(hunk_ctx)
        hunk_ctx = []
        in_hunk = False

    for line in patch.splitlines(keepends=True):
        if line.startswith("@@"):
            flush()
            out.append(line)  # placeholder; body comes next
            in_hunk = True
        elif in_hunk and line.startswith(("diff ", "--- ", "+++ ")):
            flush()  # next file's header lines end the current hunk
            out.append(line)
        elif in_hunk:
            hunk_ctx.append(line)
        else:
            out.append(line)
    flush()
    return "".join(out)


def _validate_applied(repo_path: Path, result: PatchApplicationResult) -> PatchApplicationResult:
    """Verify a successful apply didn't corrupt Python sources.

    LLM patches that ``git apply``/unidiff accept can still be garbage that
    produces syntactically invalid files (seen: response.py ``SyntaxError``,
    pycode.py ``IndentationError``). Compile every modified ``.py`` file; on
    any failure, revert the touched files and return a failed result.
    """
    if not result.success or not result.files_modified:
        return result
    bad: list[str] = []
    for rel in result.files_modified:
        if not rel.endswith(".py"):
            continue
        path = repo_path / rel
        try:
            compile(path.read_text(encoding="utf-8"), str(rel), "exec")
        except (OSError, SyntaxError, UnicodeDecodeError):
            bad.append(rel)
    if not bad:
        return result
    try:
        _run_git(repo_path, ["checkout", "--", *result.files_modified])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("failed to revert invalid patch files in %s", repo_path)
    return PatchApplicationResult(
        success=False,
        method_used="failed",
        error=f"applied but invalid syntax in: {', '.join(bad[:3])}",
    )


def apply_patch(
    repo_path: Path, patch: str, base_sha: str, *, skip_checkout: bool = False
) -> PatchApplicationResult:
    """Main entry: ``git apply`` first, unidiff manual apply as fallback.

    If both fail and the patch looks malformed (hunk headers with
    miscounted line numbers — a common LLM output error), the headers are
    recomputed from the hunk bodies and both paths are retried once.

    Args:
        repo_path: Working directory (git repo preferred).
        patch: Unified diff.
        base_sha: Commit to reset the working tree to before applying.
        skip_checkout: If True, skip the ``git checkout`` step (caller already
            reset to ``base_sha``).

    Returns:
        The successful result (git or unidiff), or
        ``method_used="failed"`` when both attempts fail.
    """
    result = apply_patch_git(repo_path, patch, base_sha, skip_checkout=skip_checkout)
    if result.success:
        return _validate_applied(repo_path, result)
    logger.warning("git apply failed (%s), trying unidiff fallback", result.error)
    fallback = _validate_applied(repo_path, apply_patch_unidiff(repo_path, patch))
    if fallback.success:
        return fallback

    # LLM patches frequently have miscounted hunk headers; recompute and retry.
    repaired = _repair_hunk_headers(patch)
    if repaired != patch:
        logger.warning(
            "patch apply failed on both paths; repaired hunk headers and retrying "
            "(git err: %s; unidiff err: %s)",
            result.error,
            fallback.error,
        )
        result2 = _validate_applied(
            repo_path, apply_patch_git(repo_path, repaired, base_sha, skip_checkout=skip_checkout)
        )
        if result2.success:
            return result2
        fallback2 = _validate_applied(repo_path, apply_patch_unidiff(repo_path, repaired))
        if fallback2.success:
            return fallback2
        return PatchApplicationResult(
            success=False,
            method_used="failed",
            error=(
                f"git apply: {result.error}; unidiff: {fallback.error}; "
                f"repaired: git: {result2.error}; unidiff: {fallback2.error}"
            ),
        )
    return PatchApplicationResult(
        success=False,
        method_used="failed",
        error=f"git apply: {result.error}; unidiff fallback: {fallback.error}",
    )
