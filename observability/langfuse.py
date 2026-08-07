"""Langfuse trace emission: per-example eval generations + sampled serving traces.

Dual-write partner to ``metrics.py``: W&B holds aggregates, Langfuse holds the
per-call traces, cross-linked by ``run_id``/``instance_id`` metadata (plan §5.5).
The SDK is imported lazily and every SDK call is wrapped so Langfuse can never
break the eval harness or the serving path (non-goal). Without
``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY`` everything is a silent no-op
(decision 2) — local dev and CI run without credentials.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-check only, never imported at runtime
    from langfuse import Langfuse

_LOGGER = logging.getLogger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"

_client: Langfuse | None = None  # built lazily on first use; None also means "not yet tried"


def _get_client() -> Langfuse | None:
    """Build and cache the Langfuse client from env vars; ``None`` when keys are absent."""
    global _client  # noqa: PLW0603 — module-level lazy singleton, the required cache
    if _client is not None:
        return _client
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    from langfuse import Langfuse

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", _DEFAULT_HOST),
        )
    except Exception as exc:  # noqa: BLE001 — Langfuse must never break the caller
        _LOGGER.warning("langfuse client init failed, traces disabled: %s", exc)
        return None
    return _client


def _enabled() -> bool:
    """True when Langfuse credentials are configured."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def trace_generation(  # noqa: PLR0913 — plan §5.5 mandated kw-only signature
    *,
    name: str,
    model: str,
    prompt: str,
    completion: str,
    metadata: dict,
    scores: dict[str, float] | None = None,
) -> None:
    """One Langfuse generation trace (prompt → completion) plus per-key scores.

    *metadata* is passed through unchanged — callers include ``run_id`` and
    ``instance_id`` for the W&B ↔ Langfuse dual-write link (plan §5.5). Each
    entry in *scores* (e.g. ``{"f2p": 1.0, "p2p": 0.0}``) is attached to the
    trace via the SDK's score API. Any SDK failure is logged and swallowed.
    """
    client = _get_client()
    if client is None:
        return
    try:
        generation = client.start_observation(
            name=name,
            as_type="generation",
            input=prompt,
            output=completion,
            model=model,
            metadata=metadata,
        )
        if generation is None:
            return
        generation.end()
        for score_name, value in (scores or {}).items():
            client.create_score(
                name=score_name,
                value=value,
                trace_id=generation.trace_id,
                observation_id=generation.id,
            )
    except Exception as exc:  # noqa: BLE001 — Langfuse must never break the caller
        _LOGGER.warning("langfuse trace_generation(%r) failed: %s", name, exc)


def trace_request(
    *,
    model: str,
    template_name: str | None,
    ttfbs_ms: float | None,
    latency_ms: float,
    output_tokens: int,
) -> None:
    """One sampled serving trace: generation with model + template + timing metadata."""
    client = _get_client()
    if client is None:
        return
    try:
        generation = client.start_observation(
            name="serve/request",
            as_type="generation",
            input=template_name or "unknown-template",
            output=f"{output_tokens} tokens",
            model=model,
            metadata={
                "template_name": template_name,
                "ttfbs_ms": ttfbs_ms,
                "latency_ms": latency_ms,
                "output_tokens": output_tokens,
            },
        )
        if generation is None:
            return
        generation.end()
    except Exception as exc:  # noqa: BLE001 — Langfuse must never break the caller
        _LOGGER.warning("langfuse trace_request(%r) failed: %s", model, exc)
