"""Shared diff parsing for ingest, prompts, and patch application.

Single source of truth for extracting file paths from unified diffs.
Handles ``diff --git`` headers, ``---``/``+++`` header pairs,
``/dev/null`` (new/deleted files), renames (new/b-side name wins),
quoted paths, and ``a/``/``b/`` prefixes. Pure stdlib so
data_engineering, inference, and evaluation can all import it.
"""

from __future__ import annotations

import re

_DEV_NULLS = {"/dev/null", "dev/null"}

# One ``diff --git a/old b/new`` line; each side may be quoted (paths with
# spaces) or bare. Group pairs: (1, 2) = a-side, (3, 4) = b-side.
_DIFF_GIT_RE = re.compile(r'^diff --git\s+(?:"a/(.+?)"|a/(\S+))\s+(?:"b/(.+?)"|b/(\S+))')


def _strip_quotes(path: str) -> str:
    path = path.strip()
    if not path or not path.startswith('"') or not path.endswith('"'):
        return path
    return path[1:-1].strip()


def _strip_ab_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    if path.startswith(('"a/', '"b/')) and path.endswith('"'):
        return path[1:-1][2:]
    return path


def _clean_header_path(raw: str) -> str | None:
    """Normalize one ``---``/``+++`` header path. None = skip (/dev/null, empty)."""
    path = _strip_quotes(raw.split("\t", maxsplit=1)[0].strip())
    path = _strip_ab_prefix(path)
    return path if path and path not in _DEV_NULLS else None


def diff_git_paths(line: str) -> tuple[str, str] | None:
    """Parse a ``diff --git a/old b/new`` line into ``(old, new)`` paths.

    Handles quoted paths (paths with spaces). Returns None when *line* is
    not a git diff header (including deletion headers whose b-side is
    ``/dev/null``, which carry no b/ path).
    """
    m = _DIFF_GIT_RE.match(line.strip())
    if not m:
        return None
    old = (m.group(1) or m.group(2) or "").strip()
    new = (m.group(3) or m.group(4) or "").strip()
    return old, new


def parse_files(patch: str) -> list[str]:
    """Extract changed file paths from a unified diff (b-side wins).

    For each file section the ``+++``/b-side path is the changed file;
    deletions (``+++ /dev/null``) fall back to the ``---``/a-side path.
    Handles ``diff --git a/old b/new`` (rename: new name wins), bare
    ``---``/``+++`` header pairs without ``diff --git``, ``/dev/null``,
    and quoted paths. Order-preserving, deduplicated.
    """
    if not patch or not patch.strip():
        return []
    files: list[str] = []
    seen: set[str] = set()

    def _add(path: str | None) -> None:
        if path and path not in seen:
            seen.add(path)
            files.append(path)

    def _file_from_pair(old: str | None, new: str | None) -> None:
        _add(new or old)

    pending_old: str | None = None
    for line in patch.splitlines():
        stripped = line.strip()
        if stripped.startswith("diff --git"):
            git = diff_git_paths(line)
            if git is not None:
                _file_from_pair(git[0], git[1])
            pending_old = None
            continue
        if line.startswith("@@"):
            pending_old = None
            continue
        if line.startswith("--- "):
            pending_old = _clean_header_path(line[4:])
        elif line.startswith("+++ "):
            _file_from_pair(pending_old, _clean_header_path(line[4:]))
            pending_old = None
    return files
