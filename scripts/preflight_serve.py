#!/usr/bin/env python3
"""Single-boot validation of a RUNNING Phase 6 serving endpoint.

Used against ``modal serve`` output or a deployed endpoint.  Exits 0 when all
steps pass, 1 on the first failing step, 2 when the env vars are unset.

Usage::

    MODAL_WEB_URL=https://... MODAL_WEB_TOKEN=... \\
        python scripts/preflight_serve.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Repo-root bootstrap so repo packages import when the script is executed
# directly from any cwd (mirrors how scripts/* are run from the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODEL_BASE = "qwen3-14b"
# Rank-32 LoRA adapter — proves max_lora_rank=64 serving config.
_MODEL_LORA = "qwen3-14b:higher_rank_14b"
_PROMPT = "Write a one-line Python function."


def _env() -> tuple[str, str]:
    url = os.environ.get("MODAL_WEB_URL", "")
    token = os.environ.get("MODAL_WEB_TOKEN", "")
    if not url or not token:
        print("ERROR: MODAL_WEB_URL and MODAL_WEB_TOKEN must both be set.", file=sys.stderr)
        print("  e.g. MODAL_WEB_URL=https://xxx.modal.run MODAL_WEB_TOKEN=xxx \\", file=sys.stderr)
        print("       python scripts/preflight_serve.py", file=sys.stderr)
        sys.exit(2)
    return url, token


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
    return ok


def _non_stream(client: Any, model: str) -> bool:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=64,
            stream=False,
        )
        content = resp.choices[0].message.content
        ok = (
            isinstance(content, str)
            and bool(content)
            and resp.object == "chat.completion"
            and resp.usage is not None
            and resp.usage.completion_tokens > 0
        )
    except Exception as exc:
        print(f"    error: {type(exc).__name__}: {exc}")
        return False
    else:
        if not ok:
            print(f"    detail: object={resp.object!r} content={content!r} usage={resp.usage}")
        return ok


def _stream(client: Any) -> bool:
    try:
        stream = client.chat.completions.create(
            model=_MODEL_BASE,
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=64,
            stream=True,
        )
        # Iterating the SDK stream consumes chunks up to and including the
        # terminating "[DONE]" frame; exhaustion below proves termination.
        chunks = list(stream)
        content_deltas = [c for c in chunks if c.choices and c.choices[0].delta.content]
        final = chunks[-1].choices[0].finish_reason if chunks and chunks[-1].choices else None
        ok = bool(content_deltas) and bool(final)
    except Exception as exc:
        print(f"    error: {type(exc).__name__}: {exc}")
        return False
    else:
        if not ok:
            print(
                "    detail: "
                f"chunks={len(chunks)} content_deltas={len(content_deltas)} final={final!r}"
            )
        return ok


def main() -> int:
    url, token = _env()
    import httpx
    import openai

    # OpenAI SDK does NOT append /v1 — MODAL_WEB_URL is the bare root, and the
    # API routes live under /v1 (health lives at the root, fetched via httpx).
    client = openai.OpenAI(base_url=url.rstrip("/") + "/v1", api_key=token)
    t0 = time.perf_counter()

    # 1. Health endpoint (OpenAI SDK has no raw GET; httpx is a project dep).
    try:
        with httpx.Client(base_url=url, timeout=30) as http:
            resp = http.get("/health")
        body = resp.json()
        ok = resp.status_code == httpx.codes.OK and body.get("status") == "ok"
    except Exception as exc:
        ok, body = False, f"{type(exc).__name__}: {exc}"
    if not _check('1. GET /health -> {"status": "ok", ...}', ok, str(body)[:120]):
        return 1

    # 2. Non-stream chat, base model (no LoRA).
    if not _check(f"2. non-stream chat, model={_MODEL_BASE}", _non_stream(client, _MODEL_BASE)):
        return 1

    # 3. Non-stream chat, rank-32 LoRA adapter (adapter resolves server-side).
    if not _check(f"3. non-stream chat, model={_MODEL_LORA}", _non_stream(client, _MODEL_LORA)):
        return 1

    # 4. Streaming chat: >=1 content delta, final finish_reason, [DONE] termination.
    if not _check(f"4. stream chat, model={_MODEL_BASE}", _stream(client)):
        return 1

    print(f"preflight passed in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
