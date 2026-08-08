"""Variant-pinned Modal deploy, health probe, alias sync, rollback (Phase 9, task 9.4, spec §4.5).

The champion variant is pinned **before** deploy: ``SERVING_DEFAULT_VARIANT``
is the env var ``inference.config.ServeConfig`` resolves ``default_variant``
from at app start (pydantic-settings ``env_prefix="SERVING_"`` auto-override,
fallback hardcoded ``higher_lr_14b``).  A variant absent from
``ServeConfig.variants`` aborts with a ``config-gap`` reason before anything
executes (spec decision 6: v1 promotes only among trained variants).  The
probe is ``POST /v1/chat/completions`` with a bearer token —
``GET /health`` is a cheap liveness pre-check only (spec decision 6).
Alias-sync failure aborts the deploy, and ``rollback`` re-promotes the
previous champion through the same pipeline (spec decision 7).

Exec/network lines are ``# pragma: no cover`` (sanctioned by plan §4.11):
the unit/E2E path passes ``dry_run=True`` or module-level fakes.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from typing import cast

import httpx

from evaluation.config import EvalConfig
from inference.config import ServeConfig
from promotion.registry import ChampionRecord, sync_alias

__all__ = [
    "ConfigGapError",
    "ProbeError",
    "assert_variant_known",
    "deploy",
    "health_check",
    "rollback",
    "sync_alias_or_abort",
]

# Loose TTFB enforcement (spec §4.5 documents the 500 ms target): only a 10x
# overshoot — a stuck container — fails the probe; normal jitter does not.
_PROBE_SLOW_FACTOR = 10


class ConfigGapError(Exception):
    """Candidate variant is not among the trained ``ServeConfig.variants``."""


class ProbeError(Exception):
    """Health probe failed (non-200 response or transport exception)."""


def assert_variant_known(variant: str, config: ServeConfig) -> None:
    """Raise :class:`ConfigGapError` unless *variant* is in *config.variants*."""
    if variant not in config.variants:
        raise ConfigGapError(f"variant {variant!r} not in ServeConfig.variants {config.variants}")


def _deploy_env(variant: str, env: dict[str, str] | None) -> dict[str, str]:
    """Return the deploy subprocess env: ``os.environ`` + *env* overrides + the pin.

    Credentials (``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``/``HF_TOKEN``) are
    never hardcoded here — they must arrive through *env*.  The
    ``SERVING_DEFAULT_VARIANT`` pin is applied last so it always wins: that
    env var is the mechanism that makes the deployed app serve the champion
    variant (``ServeConfig`` resolves ``default_variant`` from it at start).
    """
    merged = dict(os.environ)
    if env:
        merged.update(env)
    merged["SERVING_DEFAULT_VARIANT"] = variant
    return merged


def deploy(
    champion_key: str,
    config: ServeConfig,
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Pin and deploy the champion variant to Modal (spec §4.5, decision 6).

    ``champion_key`` is ``"model:variant"``; the variant must be among
    ``config.variants`` or :class:`ConfigGapError` aborts before anything
    executes.  With ``dry_run=True`` nothing is executed — a fake completed
    process is returned (unit/E2E path).
    """
    variant = champion_key.partition(":")[2]
    assert_variant_known(variant, config)
    cmd = ["uv", "run", "modal", "deploy", "-m", "inference.modal_serve"]
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="(dry-run)", stderr="")
    return subprocess.run(  # pragma: no cover — real deploy only
        cmd,
        capture_output=True,
        text=True,
        env=_deploy_env(variant, env),
        check=False,
    )


def health_check(base_url: str, token: str, *, ttfpb_target_ms: int = 500) -> float:
    """Probe a deployed app and return time-to-first-byte in seconds.

    ``GET {base_url}/health`` is a cheap liveness pre-check (static, no auth —
    spec decision 6); the real probe is a short generation on
    ``POST {base_url}/v1/chat/completions`` with ``Bearer <token>``.  Elapsed
    time is read from the response's ``elapsed`` attribute (httpx timedelta;
    test stubs may use a plain float).

    Args:
        base_url: App root (bare — no ``/v1`` suffix), e.g. ``https://x.modal.run``.
        token: Bearer credential for the chat endpoint (``MODAL_SERVE_TOKEN``).
        ttfpb_target_ms: documented TTFB ceiling; enforced loosely at 10x — a
            grossly stuck container fails fast, normal jitter does not.

    Returns:
        Time to first byte in seconds.

    Raises:
        ProbeError: liveness non-200, chat non-200, transport exception, or
            a >10x TTFB overshoot.
    """
    root = base_url.rstrip("/")
    try:
        response = httpx.get(f"{root}/health", timeout=10.0)  # pragma: no cover
    except Exception as exc:  # noqa: BLE001 — transport failure is a probe failure
        raise ProbeError(f"liveness pre-check failed: {exc}") from exc
    if response.status_code != httpx.codes.OK:
        raise ProbeError(f"liveness pre-check returned HTTP {response.status_code}")

    payload = {
        "model": "qwen3-14b",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "stream": False,
    }
    try:
        response = httpx.post(  # pragma: no cover
            f"{root}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001 — transport failure is a probe failure
        raise ProbeError(f"chat probe failed: {exc}") from exc
    if response.status_code != httpx.codes.OK:
        raise ProbeError(
            f"chat probe returned HTTP {response.status_code}: {response.text[:200]!r}"
        )
    elapsed_attr = getattr(response, "elapsed", 0.0)
    if isinstance(elapsed_attr, timedelta):
        elapsed: float = elapsed_attr.total_seconds()
    else:
        elapsed = float(elapsed_attr or 0.0)
    ceiling_ms = ttfpb_target_ms * _PROBE_SLOW_FACTOR
    if elapsed > ceiling_ms / 1000:
        raise ProbeError(f"probe too slow: {elapsed:.2f}s (ceiling {ceiling_ms}ms)")
    return elapsed


def sync_alias_or_abort(champion_key: str, config: EvalConfig) -> None:
    """Sync the W&B ``champion`` alias; abort the deploy on failure.

    ``promotion.registry.sync_alias`` returns ``None`` on any failure (no
    wandb, no network, API error); a champion whose alias cannot be synced
    must not be deployed (spec decision 6: alias failure aborts).  Runs AFTER
    the probe passes (spec §4.5).
    """
    if sync_alias(champion_key, config) is None:
        raise RuntimeError("alias sync failed — aborting deploy")


def rollback(
    previous: ChampionRecord,
    config: ServeConfig,
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, str | bool]:
    """Re-promote the previous champion through the same pipeline (spec decision 7).

    ``previous.model_ref`` is already ``"model:variant"`` (champion.json
    schema), so it is the champion key; the variant is pinned and deployed
    exactly like a normal promote, then probed and alias-synced.  In
    ``dry_run`` mode nothing executes and probe/alias are skipped.  Failures
    propagate (e.g. :class:`ProbeError`) — the workflow records the rollback
    status.  Probe credentials come from *env* (``MODAL_WEB_URL`` /
    ``MODAL_SERVE_TOKEN``, the preflight conventions).
    """
    champion_key = previous.model_ref
    assert_variant_known(previous.variant, config)
    if dry_run:
        deploy(champion_key, config, dry_run=True, env=env)
        outcome = "rollback-dry-run"
    else:
        deploy(champion_key, config, env=env)  # pragma: no cover — real deploy
        merged = _deploy_env(previous.variant, env)
        health_check(  # pragma: no cover — real probe
            merged.get("MODAL_WEB_URL", ""),
            merged.get("MODAL_SERVE_TOKEN", ""),
        )
        sync_alias_or_abort(  # pragma: no cover — real alias sync
            champion_key,
            # ServeConfig carries the same wandb entity/project + lora pattern
            # fields that sync_alias reads off EvalConfig (duck-typed).
            cast(EvalConfig, config),
        )
        outcome = "rollback"
    return {
        "variant": previous.variant,
        "champion_key": champion_key,
        "outcome": outcome,
        "deployed": not dry_run,
        "probe_ok": not dry_run,
        "alias_synced": not dry_run,
        "triggered_at": datetime.now(UTC).isoformat(),
    }
